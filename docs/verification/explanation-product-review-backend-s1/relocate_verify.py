"""Private one-shot APFS relocation and actual HTTPS acceptance orchestration.

No provider calls. Inputs are public frozen package/study artifacts. Existing
services and source bytes remain untouched; source rename is restored in finally.
"""

import argparse
import ctypes
import hashlib
import json
import os
import ssl
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from uuid import uuid4


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(data)
    return result.hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def safe_relative(root, relative):
    candidate = root / relative
    if candidate.is_symlink() or not candidate.resolve().is_relative_to(root.resolve()):
        raise RuntimeError("clone_input_containment")
    if not candidate.is_file():
        raise RuntimeError("clone_input_missing")
    return candidate


def selected_files(source, frontend):
    publication = read(source / "publication.json")
    package = read(source / "package/package.json")
    selected = {
        "input/publication.json": (source / "publication.json", None),
        "input/package/package.json": (source / "package/package.json", None),
    }

    def add(relative, expected=None):
        expected = expected.removeprefix("sha256:") if expected else None
        path = safe_relative(source, relative)
        previous = selected.get("input/" + relative)
        if previous and previous[1] and expected and previous[1] != expected:
            raise RuntimeError("clone_conflicting_hash")
        selected["input/" + relative] = (path, expected or (previous and previous[1]))

    for artifact in package["artifacts"]:
        add("package/" + artifact["path"], artifact["sha256"])
    for asset in publication["definition"]["assets"]:
        add(asset["path"], asset["sha256"])
        if asset.get("admission_receipt_path"):
            add(asset["admission_receipt_path"], asset["admission_receipt_sha256"])
    for scope in publication["definition"]["workspace_scopes"]:
        add(scope["index_path"], scope["index_sha256"])
        for context in scope["contexts"]:
            add(context["path"] + "/manifest.json", context["manifest_sha256"])
            add(context["path"] + "/context.sqlite", context["database_sha256"])
    for path in sorted(frontend.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("frontend_symlink")
        if path.is_file():
            selected["frontend/" + str(path.relative_to(frontend))] = (path, None)
    return selected


def clone_file(source, destination):
    """Invoke Darwin clonefile directly: unsupported clone operations raise, never copy."""
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    function = library.clonefile
    function.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int)
    function.restype = ctypes.c_int
    if function(os.fsencode(source), os.fsencode(destination), 0) != 0:
        raise OSError(ctypes.get_errno(), "clonefile failed; no physical-copy fallback")


class Statfs64(ctypes.Structure):
    # Darwin SDK sys/mount.h __DARWIN_STRUCT_STATFS64, including reserved storage.
    _fields_ = [
        ("f_bsize", ctypes.c_uint32),
        ("f_iosize", ctypes.c_int32),
        *[
            (name, ctypes.c_uint64)
            for name in ("f_blocks", "f_bfree", "f_bavail", "f_files", "f_ffree")
        ],
        ("f_fsid", ctypes.c_int32 * 2),
        *[(name, ctypes.c_uint32) for name in ("f_owner", "f_type", "f_flags", "f_fssubtype")],
        ("f_fstypename", ctypes.c_char * 16),
        ("f_mntonname", ctypes.c_char * 1024),
        ("f_mntfromname", ctypes.c_char * 1024),
        ("f_flags_ext", ctypes.c_uint32),
        ("f_reserved", ctypes.c_uint32 * 7),
    ]


def filesystem(path):
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    function = library.statfs64
    function.argtypes = (ctypes.c_char_p, ctypes.POINTER(Statfs64))
    function.restype = ctypes.c_int
    result = Statfs64()
    if function(os.fsencode(path), ctypes.byref(result)) != 0:
        raise OSError(ctypes.get_errno(), "filesystem identification failed")
    return result.f_fstypename.decode("ascii")


def clone(selected, destination):
    inventory = []
    for relative, (source, expected) in sorted(selected.items()):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise RuntimeError("clone_target_exists")
        # APFS clones must succeed; never silently create a second 10GB physical copy.
        clone_file(source, target)
        if source.stat().st_ino == target.stat().st_ino:
            raise RuntimeError("clone_must_have_distinct_inode")
        before, after = digest(source), digest(target)
        if before != after or (expected and before != expected.removeprefix("sha256:")):
            raise RuntimeError("clone_hash_mismatch")
        inventory.append(dict(path=relative, size=target.stat().st_size, sha256=after))
    encoded = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    representatives = {}
    for path, _ in selected.values():
        representatives.setdefault(path.stat().st_dev, path.parent)
    return dict(
        source_filesystems=sorted({filesystem(path) for path in representatives.values()}),
        destination_filesystem=filesystem(destination),
        source_devices=sorted({path.stat().st_dev for path, _ in selected.values()}),
        destination_device=destination.stat().st_dev,
        files=len(inventory),
        logical_bytes=sum(row["size"] for row in inventory),
        inventory_sha256=hashlib.sha256(encoded).hexdigest(),
        entries=inventory,
        copy_mode="Darwin clonefile(2), flags=0; no copy fallback; different inode and every source/destination hash verified",
    )


def cpu_seconds(value):
    pieces = value.split(":")
    total = 0.0
    for part in pieces:
        total = total * 60 + float(part)
    return total


class Samples:
    def __init__(self, env, config):
        self.env, self.config, self.stop_event = env, config, threading.Event()
        self.values = {}
        self.thread = threading.Thread(target=self.collect, daemon=True)

    def collect(self):
        while not self.stop_event.is_set():
            for role, record in self.env.processes(self.config).items():
                if not self.env.owned(self.config, role, record):
                    continue
                raw = subprocess.run(
                    ["/bin/ps", "-p", str(record["pid"]), "-o", "rss=,time=,%cpu="],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                parts = raw.stdout.split()
                if len(parts) != 3:
                    continue
                row = self.values.setdefault(
                    role,
                    dict(
                        samples=0,
                        peak_rss_kib=0,
                        first_cpu_seconds=cpu_seconds(parts[1]),
                        observed_cpu_delta_seconds=0.0,
                        last_cpu_seconds=0.0,
                        peak_cpu_percent=0.0,
                    ),
                )
                row["samples"] += 1
                row["peak_rss_kib"] = max(row["peak_rss_kib"], int(parts[0]))
                row["last_cpu_seconds"] = max(row["last_cpu_seconds"], cpu_seconds(parts[1]))
                row["observed_cpu_delta_seconds"] = round(
                    row["last_cpu_seconds"] - row["first_cpu_seconds"], 3
                )
                row["peak_cpu_percent"] = max(row["peak_cpu_percent"], float(parts[2]))
            self.stop_event.wait(0.5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join(timeout=5)


def probe_frontend(config):
    import httpx
    from exact_inspect.frontend import content_security_policy
    from tools.verify_review_environment import validate_response_csp

    study_origin = f"https://127.0.0.1:{config['ports']['study']}"
    context = ssl.create_default_context(cafile=config["certificate"])
    results = {}
    with httpx.Client(
        verify=context, trust_env=False, timeout=30, headers={"Accept-Encoding": "identity"}
    ) as client:
        for role, route, relative in (
            ("exploration", "/", "index.html"),
            ("study", "/participate/", "participate/index.html"),
        ):
            html = Path(config["frontend"]) / relative
            expected_hash = digest(html)
            expected_csp = content_security_policy(html, strict_styles=role == "study")
            samples = []
            for _ in range(3):
                started = time.monotonic()
                response = client.get(f"https://127.0.0.1:{config['ports'][role]}" + route)
                elapsed = time.monotonic() - started
                if (
                    response.status_code != 200
                    or hashlib.sha256(response.content).hexdigest() != expected_hash
                ):
                    raise RuntimeError("frontend_http_body_identity")
                if int(response.headers.get("content-length", "-1")) != len(response.content):
                    raise RuntimeError("frontend_http_body_length")
                validate_response_csp(response, study_origin=study_origin, label=role + "_homepage")
                if response.headers.get("content-security-policy") != expected_csp:
                    raise RuntimeError("frontend_exact_configured_csp")
                samples.append(round(elapsed, 4))
            results[role] = dict(
                body_sha256=expected_hash,
                bytes=len(response.content),
                csp_sha256=hashlib.sha256(expected_csp.encode()).hexdigest(),
                exact_configured_csp=True,
                seconds=samples,
            )
    return results


def main():
    parser = argparse.ArgumentParser()
    for field in ("repo", "source", "frontend", "root", "python", "pgbin"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--source-hide-window-authorized", action="store_true", required=True)
    args = parser.parse_args()
    args.repo, args.source, args.frontend, args.root = [
        x.resolve() for x in (args.repo, args.source, args.frontend, args.root)
    ]
    if args.root.exists():
        raise RuntimeError("delivered_root_must_be_new")
    current = subprocess.check_output(
        ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if current != args.expected_commit:
        raise RuntimeError("source_commit_changed")
    sys.path.insert(0, str(args.repo))
    from tools import review_environment as env
    from tools.verify_review_environment import Journey, fingerprint

    class ObservedJourney(Journey):
        """Retain only public form dictionary fields from an existing successful reply."""

        def request(self, method, path, **kwargs):
            response = super().request(method, path, **kwargs)
            target = Path("/tmp/exact-s1-delivered-forms-excerpt.json")
            if (
                not target.exists()
                and response.status_code == 200
                and response.headers.get("content-type", "").startswith("application/json")
            ):
                data = response.json()
                if (
                    isinstance(data, dict)
                    and isinstance(data.get("forms"), dict)
                    and isinstance(data.get("protocol_versions"), dict)
                ):
                    env.private_json(target, dict(
                        schema="exact-public-forms-excerpt/1",
                        scope="Actual delivered HTTPS response; public definitions only",
                        method=method,
                        route=path.split("?")[0],
                        status=200,
                        response_sha256=fingerprint(response.content),
                        forms=data["forms"],
                        protocol_versions=data["protocol_versions"],
                    ))
            return response

    ports = dict(postgres=19183, study=19184, exploration=19185)
    if not all(env.port_available(port) for port in ports.values()):
        raise RuntimeError("delivered_ports_in_use")
    selected = selected_files(args.source, args.frontend)
    args.root.mkdir(mode=0o700)
    report = dict(
        schema="exact-review-portable-verification/1",
        status="running",
        boundary="Actual HTTPS API and process checks; no browser acceptance claim",
        process_sampling_scope="Owned main PIDs only; descendants not summed. Last CPU is process cumulative, delta covers observed samples. Cold means first launch against newly initialized PG; OS file cache was not cleared. During the API journey, independent read-only main-app browser and example capture may overlap; journey latency is not an isolated benchmark.",
        git_commit=current,
        orchestration_sha256=digest(Path(__file__)),
        phases={},
        source_hidden=False,
        source_restored=False,
    )
    receipt = args.root / "verification/relocation.json"
    receipt.parent.mkdir(mode=0o700)
    (receipt.parent / "relocate_verify.py").write_bytes(Path(__file__).read_bytes())

    def record():
        env.private_json(receipt, report)

    phase = "clone"
    hidden = args.source.with_name(args.source.name + ".verification-hidden-" + uuid4().hex[:8])
    config = None
    with env.environment_lock(args.root):
        try:
            print(json.dumps(dict(phase=phase)), flush=True)
            started = time.monotonic()
            report["clone"] = clone(selected, args.root)
            report["phases"]["clone"] = dict(seconds=round(time.monotonic() - started, 3))
            env.private_json(args.root / "verification/clone-inventory.json", report["clone"])
            # No input source is now readable at its former path. Caller coordinates readers.
            args.source.rename(hidden)
            report["source_hidden"] = not args.source.exists() and hidden.exists()
            record()
            phase = "prepare"
            print(json.dumps(dict(phase=phase)), flush=True)
            started = time.monotonic()
            env.prepare(
                SimpleNamespace(
                    root=args.root,
                    publication=args.root / "input/publication.json",
                    assets=args.root / "input",
                    package=args.root / "input/package/package.json",
                    frontend=args.root / "frontend",
                    python=args.python,
                    pgbin=args.pgbin,
                    postgres_port=19183,
                    study_port=19184,
                    exploration_port=19185,
                    days=30,
                )
            )
            config = env.load(args.root)
            report["backend"] = config["backend"]
            report["phases"]["prepare"] = dict(seconds=round(time.monotonic() - started, 3))
            for phase in ("cold_start", "warm_start"):
                print(json.dumps(dict(phase=phase)), flush=True)
                if phase == "warm_start":
                    for role in reversed(env.ROLES):
                        env.stop_role(config, role)
                started = time.monotonic()
                with Samples(env, config) as sample:
                    result = env.start(config)
                    start_seconds = round(time.monotonic() - started, 3)
                    frontends = probe_frontend(config)
                report["phases"][phase] = dict(
                    start_seconds=start_seconds,
                    frontend_probes=frontends,
                    seconds=round(time.monotonic() - started, 3),
                    processes=sample.values,
                    services=result["services"],
                )
                record()
            phase = "api_journey"
            print(json.dumps(dict(phase=phase)), flush=True)
            started = time.monotonic()
            with Samples(env, config) as sample:
                ObservedJourney(config, receipt=args.root / "verification/api-journey.json").run()
            report["phases"][phase] = dict(
                seconds=round(time.monotonic() - started, 3), processes=sample.values
            )
            report["api_receipt_sha256"] = digest(args.root / "verification/api-journey.json")
            phase = "frozen_copy_unchanged"
            started = time.monotonic()
            for entry in report["clone"]["entries"]:
                path = args.root / entry["path"]
                if path.stat().st_size != entry["size"] or digest(path) != entry["sha256"]:
                    raise RuntimeError("frozen_copy_changed_during_verification")
            report["phases"][phase] = dict(
                seconds=round(time.monotonic() - started, 3),
                files=len(report["clone"]["entries"]),
                unchanged=True,
            )
            report["status"] = "passed"
        except BaseException as error:
            report.update(status="failed", failed_phase=phase, exception_type=type(error).__name__)
            if config:
                for role in reversed(env.ROLES):
                    try:
                        env.stop_role(config, role)
                    except Exception:
                        report.setdefault("stop_failures", []).append(role)
            raise
        finally:
            if hidden.exists():
                if args.source.exists():
                    report["source_restore_conflict"] = True
                else:
                    hidden.rename(args.source)
            report["source_restored"] = args.source.exists() and not hidden.exists()
            if not report["source_restored"]:
                report.update(status="failed", failed_phase="source_restoration")
            record()
            print(
                json.dumps(
                    dict(
                        status=report["status"],
                        phase=phase,
                        source_restored=report["source_restored"],
                    )
                ),
                flush=True,
            )
            if not report["source_restored"]:
                raise RuntimeError("original_source_restore_conflict")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        # Preserve only type, never service responses, credentials or invitation links.
        print(json.dumps(dict(status="failed", exception_type=type(error).__name__)), flush=True)
        raise SystemExit(1)
