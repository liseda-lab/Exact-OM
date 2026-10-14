"""Bounded, credential-free HTTPS delivery checks for a managed review environment.

Uses its trusted certificate and existing export only. Client close attempts are not
proof that the server was interrupted; matched terminal observations decide that.
Run under the shared storage guard. Never overwrite an earlier receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import socket
import ssl
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from exact_inspect.public_assets import IMMUTABLE_CACHE, build_public_asset_registry

HEADERS = (
    "cache-control",
    "content-type",
    "content-length",
    "etag",
    "x-content-type-options",
    "x-exact-read-id",
    "x-exact-asset-sha256",
)
OBSERVATIONS = (
    "status",
    "declared_length",
    "sent_bytes",
    "complete",
    "disconnect_seen",
    "error_type",
    "length_matches",
    "head_body_empty",
    "offloaded",
    "elapsed_ms",
    "asset_sha256",
)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            result.update(block)
    return result.hexdigest()


def exchange(port, context, path, method="GET", interrupted=False):
    connection = http.client.HTTPSConnection("127.0.0.1", port, context=context, timeout=15)
    try:
        connection.connect()
        if interrupted:
            connection.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2048)
        receive_buffer = connection.sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        connection.request(method, path, headers={"Accept-Encoding": "identity"})
        response = connection.getresponse()
        body = response.read(1024 if interrupted else 2_000_001)
        if len(body) > 2_000_000:
            raise ValueError("Bounded response size exceeded")
        headers = {key: response.getheader(key, "")[:512] for key in HEADERS}
        csp = response.getheader("content-security-policy", "")
        directives = dict(
            part.strip().split(" ", 1) for part in csp.split(";") if " " in part.strip()
        )
        return {
            "method": method,
            "status": response.status,
            "headers": headers,
            "received_bytes": len(body),
            "body_sha256": hashlib.sha256(body).hexdigest(),
            "client_close_attempt": interrupted,
            "receive_buffer_bytes": receive_buffer,
            "csp_checks": {
                "script_hashes": "'sha256-" in directives.get("script-src", ""),
                "script_no_unsafe_inline": "'unsafe-inline'"
                not in directives.get("script-src", ""),
                "connect_self": directives.get("connect-src") == "'self'",
                "object_none": directives.get("object-src") == "'none'",
                "strict_styles": directives.get("style-src") == "'self'",
            },
        }
    finally:
        connection.close()


def observations(root, role, identities):
    found = {}
    for suffix in (".2", ".1", ""):
        path = root / "logs" / (role + "-reads.jsonl" + suffix)
        if not path.is_file():
            continue
        if path.stat().st_size > 1_100_000:
            raise ValueError("Diagnostic log exceeded its bound")
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue  # A concurrently appended final line can be incomplete.
            if (
                row.get("read_id") in identities
                and row.get("event") == "finished"
                and row.get("schema") == "exact-static-transfer/1"
            ):
                found[row["read_id"]] = {key: row.get(key) for key in OBSERVATIONS}
    return found


def verify(config, root, report):
    frontend = Path(config["frontend"])
    binding = config["hashes"]
    report["binding"] = {
        key: binding[key]
        for key in ("frontend", "package", "publication", "source_publication", "certificate")
    }
    report["backend"] = {
        key: config["backend"][key] for key in ("git_commit", "source_sha256", "file_count")
    }

    def check(label, condition):
        report["checks"] += 1
        if not condition:
            report["failures"].append(label)

    tree = hashlib.sha256()
    for path in sorted(frontend.rglob("*")):
        if path.is_file():
            tree.update(
                str(path.relative_to(frontend)).encode() + b"\0" + digest(path).encode() + b"\n"
            )
    check("frontend_binding", tree.hexdigest() == binding["frontend"])
    for key in ("package", "publication", "source_publication", "certificate"):
        check(key + "_binding", digest(config[key]) == binding[key])
    assets = build_public_asset_registry(frontend.resolve())
    css = [(name, asset) for name, asset in assets.items() if asset.path.suffix == ".css"]
    scripts = [(name, asset) for name, asset in assets.items() if asset.path.suffix == ".js"]
    if not 0 < len(css) <= 32 or not scripts:
        raise ValueError("Expected bounded CSS and admitted JavaScript assets")
    largest, script = max(scripts, key=lambda item: item[1].stat.st_size)
    report["interruption_asset"] = {"sha256": script.sha256, "bytes": script.stat.st_size}
    resource = json.loads(Path(config["publication"]).read_text())["definition"]["assets"][0][
        "asset_id"
    ]
    context = ssl.create_default_context(cafile=config["certificate"])
    for role in ("exploration", "study"):
        port = config["ports"][role]
        rows = report["profiles"][role] = []

        def request(label, path, method="GET", interrupted=False):
            row = exchange(port, context, path, method, interrupted)
            row["check"] = label
            rows.append(row)
            return row

        def css_read(name, asset, method="GET", label="css"):
            row = request(label, "/" + name, method)
            headers = row["headers"]
            expected = asset.stat.st_size
            check(
                role + ":" + label + ":" + method,
                row["status"] == 200
                and headers["content-type"].startswith("text/css")
                and headers["content-length"] == str(expected)
                and headers["cache-control"] == IMMUTABLE_CACHE
                and headers["x-content-type-options"] == "nosniff"
                and headers["x-exact-asset-sha256"] == asset.sha256
                and headers["etag"] == '"' + asset.sha256 + '"'
                and bool(re.fullmatch("[0-9a-f]{32}", headers["x-exact-read-id"]))
                and row["received_bytes"] == (0 if method == "HEAD" else expected)
                and (method == "HEAD" or row["body_sha256"] == asset.sha256),
            )

        for name, asset in css:
            css_read(name, asset)
            css_read(name, asset, "HEAD")
        paths = [
            ("html", "/participate/" if role == "study" else "/", 200),
            ("health", "/api/health" if role == "study" else "/api/v1/health", 200),
            ("profile_denial", "/browse/" if role == "study" else "/participate/", 404),
            ("missing_asset", "/_next/static/css/ffffffffffffffff.css", 404),
            ("missing_api", "/api/v1/no-such-delivery-check", 404),
            (
                "private_resource",
                "/api/v1/study/resources/" + quote(resource, safe=""),
                401 if role == "study" else 404,
            ),
        ]
        for label, path, status in paths:
            row = request(label, path)
            check(
                role + ":" + label,
                row["status"] == status and "no-store" in row["headers"]["cache-control"],
            )
            if label == "html":
                checks = row["csp_checks"]
                check(
                    role + ":document_csp",
                    all(
                        checks[key]
                        for key in (
                            "script_hashes",
                            "script_no_unsafe_inline",
                            "connect_self",
                            "object_none",
                        )
                    )
                    and (role != "study" or checks["strict_styles"]),
                )
            elif label != "health":
                check(
                    role + ":" + label + ":not_html",
                    "text/html" not in row["headers"]["content-type"],
                )
        for index in range(3):
            row = request("client_close_" + str(index + 1), "/" + largest, interrupted=True)
            check(
                role + ":close_attempt",
                row["status"] == 200
                and row["received_bytes"] < script.stat.st_size
                and row["headers"]["x-exact-asset-sha256"] == script.sha256,
            )
            css_read(*css[0], label="after_close_" + str(index + 1))
        static = [row for row in rows if row["headers"]["x-exact-asset-sha256"]]
        identities = {row["headers"]["x-exact-read-id"] for row in static}
        deadline = time.monotonic() + 2
        while True:
            observed = observations(root, role, identities)
            if identities <= observed.keys() or time.monotonic() >= deadline:
                break
            time.sleep(0.1)  # Bounded local log completion wait; never retry HTTP reads.
        for row in static:
            terminal = observed.get(row["headers"]["x-exact-read-id"])
            row["server_observation"] = terminal
            check(role + ":diagnostic_correlation", terminal is not None)
            if terminal and not row["client_close_attempt"]:
                check(
                    role + ":completed_static",
                    terminal["complete"] is True
                    and terminal["error_type"] is None
                    and terminal["asset_sha256"] == row["headers"]["x-exact-asset-sha256"]
                    and (
                        terminal["head_body_empty"] is True
                        if row["method"] == "HEAD"
                        else terminal["length_matches"] is True
                    ),
                )
            if row["client_close_attempt"]:
                row["interpretation"] = (
                    "server_completed"
                    if terminal and terminal["complete"] is True
                    else (
                        "server_incomplete_observed"
                        if terminal and terminal["complete"] is False
                        else "server_completion_unknown"
                    )
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite a retained receipt")
    report = {
        "schema": "exact-static-delivery-verification/1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "checks": 0,
        "failures": [],
        "profiles": {},
        "limitations": [
            "Client closure does not prove a server interruption; terminal observations are reported separately.",
            "Successful delivery checks do not establish or repair the original HR1 cause.",
            "No authenticated participant or researcher requests are made.",
        ],
    }
    try:
        verify(json.loads((args.root / "environment.json").read_text()), args.root, report)
    except Exception as exc:
        report["failures"].append(type(exc).__name__)
    report.update(
        status="failed" if report["failures"] else "passed",
        finished_at=datetime.now(timezone.utc).isoformat(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(
        os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "w"
    ) as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {"status": report["status"], "checks": report["checks"], "failures": report["failures"]}
        )
    )
    raise SystemExit(bool(report["failures"]))


if __name__ == "__main__":
    main()
