"""Authenticated researcher CLI; no invitation distribution or participant contact."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def main():
    """Write reviewable researcher results to private local files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=["publish", "invitations", "progress", "export", "reissue", "revoke", "close"],
    )
    parser.add_argument("--study")
    parser.add_argument("--session")
    parser.add_argument("--publication", type=Path)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Issue non-test invitations only for an approved live publication",
    )
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--include-keys", action="store_true")
    parser.add_argument("--format", choices=["json", "csv"], default="json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    origin = os.environ["EXACT_STUDY_ORIGIN"].rstrip("/")
    if not origin.startswith("https://"):
        parser.error("Researcher API requires HTTPS")
    body, method = {}, "POST"
    if args.action == "publish":
        if not args.publication:
            parser.error("publish requires --publication")
        body = json.loads(args.publication.read_text())
        route = "/admin/studies"
    elif args.action in {"reissue", "revoke"}:
        if not args.session:
            parser.error("reissue/revoke require --session")
        route = f"/admin/invitations/{urllib.parse.quote(args.session, safe='')}/{args.action}"
    else:
        if not args.study:
            parser.error("This action requires --study")
        suffix = {
            "invitations": "invitations",
            "progress": "progress",
            "export": "exports",
            "close": "close",
        }[args.action]
        route = f"/admin/studies/{urllib.parse.quote(args.study, safe='')}/{suffix}"
        if args.action == "invitations":
            body = {"count": args.count, "test": not args.live}
        elif args.action == "progress":
            method = "GET"
        elif args.action == "export":
            route += "?" + urllib.parse.urlencode(
                {
                    "include_test": str(args.include_test).lower(),
                    "include_keys": str(args.include_keys).lower(),
                    "format": args.format,
                }
            )
    request = urllib.request.Request(
        origin + "/api/v1" + route,
        data=json.dumps(body).encode() if method == "POST" else None,
        method=method,
        headers={
            "Authorization": f"Bearer {os.environ['EXACT_STUDY_RESEARCHER_TOKEN']}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = response.read()
    except urllib.error.HTTPError as exc:
        parser.error(f"Researcher action failed (HTTP {exc.code}); no result was written")
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(result if args.format == "csv" else result + b"\n")


if __name__ == "__main__":
    main()
