#!/usr/bin/python3
"""Root-owned, fixed-command deployment helper; never restarts the study service."""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path("/var/lib/exact-demo-deploy")
GRANT = Path("/etc/sudoers.d/99-exact-demo-maintenance")
TARGET = Path("/etc/nginx/sites-available/exact-study")


def run(*args):
    subprocess.run(args, check=True)


def ready():
    for _ in range(180):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8010/api/v1/health", timeout=2) as r:
                if json.load(r)["profile"] == "public_demo":
                    return
        except (OSError, ValueError):
            pass
        time.sleep(1)
    raise RuntimeError("Demo not ready; study routing was not changed")


def main():
    if os.geteuid() != 0:
        raise SystemExit("Run through sudo")
    action = sys.argv[1] if len(sys.argv) == 2 else ""
    if action == "activate":
        # These files were installed root-owned by the reviewed bootstrap.
        run("/usr/bin/systemctl", "daemon-reload")
        run("/usr/bin/systemctl", "enable", "exact-demo.service")
        run("/usr/bin/systemctl", "restart", "exact-demo.service")
        ready()
        before = TARGET.read_bytes()
        if not (ROOT / "original-nginx").exists():
            (ROOT / "original-nginx").write_bytes(before)
        try:
            TARGET.write_bytes((ROOT / "demo-nginx").read_bytes())
            run("/usr/sbin/nginx", "-t")
            run("/usr/bin/systemctl", "reload", "nginx.service")
        except BaseException:
            TARGET.write_bytes(before)
            run("/usr/sbin/nginx", "-t")
            run("/usr/bin/systemctl", "reload", "nginx.service")
            raise
        print("Demo activated. Study process and database unchanged.")
    elif action == "rollback":
        if not (ROOT / "original-nginx").exists():
            raise SystemExit("No saved routing to restore")
        TARGET.write_bytes((ROOT / "original-nginx").read_bytes())
        run("/usr/sbin/nginx", "-t")
        run("/usr/bin/systemctl", "reload", "nginx.service")
        run("/usr/bin/systemctl", "disable", "--now", "exact-demo.service")
    elif action == "revoke":
        if GRANT.exists():
            if not GRANT.read_text().startswith("# Exact demo temporary maintenance\n"):
                raise SystemExit("Unrecognised grant")
            GRANT.unlink()
            run("/usr/sbin/visudo", "-c")
    else:
        raise SystemExit("Use activate, rollback or revoke")


if __name__ == "__main__":
    main()
