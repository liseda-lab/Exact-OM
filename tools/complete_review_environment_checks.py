"""Run owned local startup, fresh HTTPS journey and restart/resume without polling a chat.

All progress is written to a private receipt. Failure stays failure; no readiness
timeout, assertion, CSP or certificate check is relaxed. Successful checks leave
the review services running for frontend integration.
"""

import argparse
import json
import ssl
import time
from pathlib import Path

import httpx

from tools import review_environment as environment
from tools.verify_review_environment import Journey, fingerprint, validate_response_csp


def run(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError("Use a new verification output directory")
    output.mkdir(parents=True, mode=0o700)
    config = environment.load(root)
    record = {
        "schema": "exact-review-environment-followup/1",
        "status": "running",
        "verifier_sha256": environment.file_hash(__file__),
        "startup_kind": "fresh owned workers; an existing cluster/publication may be reused",
        "backend": config["backend"],
        "publication_sha256": config["hashes"]["publication"],
    }

    def save(phase, **values):
        record.update(phase=phase, **values)
        environment.private_json(output / "status.json", record)

    try:
        save("startup")
        started = time.monotonic()
        with environment.environment_lock(root):
            environment.start(config)
        save("fresh_https_journey", startup_seconds=time.monotonic() - started)
        for role in ("study", "exploration"):
            path = root / "logs" / (role + "-readiness.json")
            if path.exists():
                environment.private_json(
                    output / (role + "-readiness.json"), environment.read_json(path)
                )
        journey = Journey(config, receipt=output / "journey.json")
        journey.run()
        state, cookies = journey.state, journey.client.cookies
        save("restart_resume")
        with environment.environment_lock(root):
            for role in reversed(environment.ROLES):
                environment.stop_role(config, role)
            started = time.monotonic()
            environment.start(config)
        record["warm_restart_seconds"] = time.monotonic() - started
        with httpx.Client(
            verify=ssl.create_default_context(cafile=config["certificate"]),
            trust_env=False,
            cookies=cookies,
            timeout=90,
        ) as client:
            response = client.get(
                f"https://127.0.0.1:{config['ports']['study']}/api/v1/study/state"
            )
            assert response.status_code == 200
            resumed = response.json()
            assert (
                resumed["session_id"] == state["session_id"]
                and resumed["revision"] == state["revision"]
                and resumed["stage"] == state["stage"] == "completed"
            )
            assert int(response.headers["content-length"]) == len(response.content)
            validate_response_csp(
                response,
                study_origin=f"https://127.0.0.1:{config['ports']['study']}",
                label="post_restart_resume",
            )
        save(
            "complete",
            status="passed",
            session_sha256=fingerprint(state["session_id"]),
            resumed_stage=resumed["stage"],
        )
    except Exception as error:
        save(
            record.get("phase", "initialization"),
            status="failed",
            exception_type=type(error).__name__,
        )
        raise
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.output), indent=2))
