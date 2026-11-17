#!/usr/bin/python3
"""Owner-run setup: install reviewed demo files and an expiring, scoped sudo grant.

Adapted from liseda_cluster's temporary-admin-access.py: NOTAFTER, visudo,
root-owned grant, explicit revoke. This grant cannot run arbitrary root commands.
"""

import argparse
import datetime
import json
import os
import pwd
import socket
import subprocess
import tempfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hours", type=int, default=4)
    p.add_argument("--evidence-dir", type=Path, default=Path("/home/pcotovio/exact-demo/private"))
    a = p.parse_args()
    if os.geteuid() != 0 or socket.gethostname().split(".")[0] != "exact-explain":
        raise SystemExit("Run with sudo on AppVM only")
    if not 1 <= a.hours <= 24:
        raise SystemExit("Choose 1–24 hours")
    if pwd.getpwnam("pcotovio").pw_uid != 1000:
        raise SystemExit("Unexpected account UID")
    evidence = a.evidence_dir
    api = json.loads((evidence / "api-verification.json").read_text())
    routing = json.loads((evidence / "routing-verification.json").read_text())
    browser = json.loads((evidence / "browser-verification.json").read_text())
    if (
        api["pairs_checked"] != 50
        or api.get("restored_trace_pairs") != 50
        or not routing["study_pages_and_assets_identical"]
        or browser["failed"]
    ):
        raise SystemExit("Demo acceptance has not passed")
    src = Path(__file__).resolve().parent
    root = Path("/var/lib/exact-demo-deploy")
    root.mkdir(mode=0o700, exist_ok=True)
    # Refuse a later replacement of another administrator's files.
    helper = Path("/usr/local/sbin/exact-demo-maintenance")
    if helper.exists() and "Root-owned, fixed-command deployment helper" not in helper.read_text():
        raise SystemExit("Unrecognised helper")
    for source, target, mode in [
        (src / "maintenance.py", helper, 0o755),
        (src / "exact-demo.service", Path("/etc/systemd/system/exact-demo.service"), 0o644),
        (src / "exact-study-with-demo.nginx", root / "demo-nginx", 0o600),
    ]:
        data = source.read_bytes()
        target.write_bytes(data)
        os.chown(target, 0, 0)
        target.chmod(mode)
    Path("/etc/nginx/conf.d/exact-demo-limits.conf").write_text(
        "limit_req_zone $binary_remote_addr zone=exact_demo:1m rate=30r/s;\n"
    )
    expiry = (
        datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=a.hours)
    ).strftime("%Y%m%d%H%M%SZ")
    policy = (
        "# Exact demo temporary maintenance\n"
        + f"pcotovio ALL=(root) NOTAFTER={expiry} NOPASSWD: /usr/local/sbin/exact-demo-maintenance activate, /usr/local/sbin/exact-demo-maintenance rollback, /usr/local/sbin/exact-demo-maintenance revoke\n"
    )
    target = Path("/etc/sudoers.d/99-exact-demo-maintenance")
    if target.exists() and not target.read_text().startswith(
        "# Exact demo temporary maintenance\n"
    ):
        raise SystemExit("Unrecognised grant")
    fd, tmp = tempfile.mkstemp(prefix=".exact-demo-", dir="/etc/sudoers.d")
    try:
        os.fchmod(fd, 0o440)
        with os.fdopen(fd, "w") as f:
            f.write(policy)
        subprocess.run(["/usr/sbin/visudo", "-cf", tmp], check=True)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    subprocess.run(["/usr/sbin/visudo", "-c"], check=True)
    print(f"Demo-only deployment grant expires {expiry}. No study service was restarted.")


if __name__ == "__main__":
    main()
