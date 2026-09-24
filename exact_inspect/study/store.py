"""Transactional durable study store, immutable publications and analysis exports.

PostgreSQL is required in deployment. The explicit SQLite adapter exists only for
small isolated tests; both adapters exercise the same transactional state machine.
"""

from __future__ import annotations

import hashlib
import json
import random
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .forms import definitions, validate_answers
from .models import Publish
from .scoring import score_response, summarize


def utcnow():
    """UTC receipt time, independent of untrusted client wall clocks."""
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    """Stable bytes shared by idempotency, manifests and resource checks."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value):
    """Hash a string without persisting bearer capabilities."""
    return hashlib.sha256(value.encode()).hexdigest()


class StudyError(Exception):
    """A public, non-sensitive API error and optional current revision."""

    def __init__(self, status, message, revision=None):
        self.status = status
        self.message = message
        self.revision = revision
        super().__init__(message)


class Connection:
    """Minimal DB-API placeholder adaptation, keeping SQL portable in tests."""

    def __init__(self, connection, postgres):
        self.connection = connection
        self.postgres = postgres

    def execute(self, sql, args=()):
        return self.connection.execute(sql.replace("?", "%s") if self.postgres else sql, args)

    def lock(self, sql, args=()):
        return self.execute(sql + (" FOR UPDATE" if self.postgres else ""), args)


class StudyStore:
    """Own all state transitions and enforce scope without client-selected owners."""

    def __init__(self, database_url, assets_dir=None, *, allow_test_sqlite=False):
        self.database_url = database_url
        self.assets_dir = Path(assets_dir).resolve() if assets_dir else None
        self.postgres = database_url.startswith(("postgresql://", "postgres://"))
        if not self.postgres and not (allow_test_sqlite and database_url.startswith("sqlite:///")):
            raise ValueError("Study persistence requires PostgreSQL; SQLite is test-only")
        self.migrate()

    @contextmanager
    def transaction(self):
        """Commit before acknowledgement; row locks serialize mutations per session."""
        if self.postgres:
            import psycopg
            from psycopg.rows import dict_row

            connection = psycopg.connect(self.database_url, row_factory=dict_row)
        else:
            connection = sqlite3.connect(self.database_url.removeprefix("sqlite:///"), timeout=30)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("BEGIN IMMEDIATE")
        try:
            yield Connection(connection, self.postgres)
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def migrate(self):
        """Apply versioned idempotent schema migration under a PostgreSQL lock."""
        with self.transaction() as db:
            if self.postgres:
                db.execute("SELECT pg_advisory_xact_lock(173912467)")
            sql = (Path(__file__).parent / "migrations" / "001_initial.sql").read_text()
            for statement in sql.split(";"):
                if statement.strip():
                    db.execute(statement)
            db.execute(
                "INSERT INTO study_schema(version, applied_at) VALUES (1, ?) ON CONFLICT(version) DO NOTHING",
                (utcnow(),),
            )

    def ready(self):
        """Check the live database and expected migration version."""
        with self.transaction() as db:
            return (
                db.execute("SELECT MAX(version) AS version FROM study_schema").fetchone()["version"]
                == 1
            )

    def _study(self, db, revision, *, lock=False):
        row = (db.lock if lock else db.execute)(
            "SELECT * FROM studies WHERE revision = ?", (revision,)
        ).fetchone()
        if row is None:
            raise StudyError(404, "Study not found")
        return row, json.loads(row["payload"])

    def _session(self, db, session_id, generation=None):
        row = db.lock("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise StudyError(401, "Session unavailable")
        study_row, study = self._study(db, row["study_revision"])
        if row["revoked"] or (generation is not None and generation != row["generation"]):
            raise StudyError(401, "Invitation unavailable")
        if study_row["closed"] or datetime.fromisoformat(study["closes_at"]) <= datetime.now(
            timezone.utc
        ):
            raise StudyError(410, "Study closed")
        return row, json.loads(row["state"]), study

    def publish(self, publication: Publish):
        """Verify and freeze participant packages; store keys in a separate table."""
        study = publication.definition.model_dump(mode="json")
        keys = {k.case_id: k.model_dump(mode="json") for k in publication.case_keys}
        if len(keys) != len(publication.case_keys) or set(keys) != {
            c["case_id"] for c in study["cases"]
        }:
            raise StudyError(422, "Each case requires an independently adjudicated key")
        for case in study["cases"]:
            key = keys[case["case_id"]]
            candidate_ids = {c["candidate_id"] for c in case["candidates"]}
            accepted = set(key["acceptable_candidate_ids"])
            if not accepted <= candidate_ids or len(accepted) != len(
                key["acceptable_candidate_ids"]
            ):
                raise StudyError(422, "Invalid adjudicated candidate identity")
            if (key["case_kind"] == "answer_present" and not accepted) or (
                key["case_kind"] == "answer_absent" and accepted
            ):
                raise StudyError(422, "Adjudication does not match case kind")
            if set(key["original_production_ranks"]) != candidate_ids:
                raise StudyError(
                    422, "Production rank provenance is required for all five candidates"
                )
            ranks = list(key["original_production_ranks"].values())
            if len(set(ranks)) != 5:
                raise StudyError(422, "Production ranks must be unique")
            if key["origin"] == "constructed" and not key["construction_provenance"]:
                raise StudyError(
                    422, "Constructed candidates require private original-set provenance"
                )
            if key["origin"] == "constructed":
                original = key["construction_provenance"]
                for candidate in case["candidates"]:
                    cid = candidate["candidate_id"]
                    if (
                        cid in original["original_scores"]
                        and candidate["score"] != original["original_scores"][cid]
                    ):
                        raise StudyError(
                            422, "Construction may not alter original candidate scores"
                        )
            if key["origin"] == "natural" and [
                key["original_production_ranks"][c["candidate_id"]] for c in case["candidates"]
            ] != list(range(1, 6)):
                raise StudyError(422, "Natural candidates must retain the untouched top five")
        admitted_ontologies = {}
        admitted_explanations = {}
        for asset in study["assets"]:
            self._verified_asset_path(asset)
            try:
                if asset["kind"] == "explanation":
                    from .resources import validate_explanation_resource

                    admitted_explanations[asset["asset_id"]] = validate_explanation_resource(
                        self._asset_bytes(asset), study
                    )
                else:
                    from ..context_resources import validate_ontology_resource

                    receipt_path = self._resource_path(asset["admission_receipt_path"])
                    if (
                        not receipt_path.is_file()
                        or hashlib.sha256(receipt_path.read_bytes()).hexdigest()
                        != asset["admission_receipt_sha256"]
                    ):
                        raise ValueError("Missing or changed ontology filtering receipt")
                    receipt = validate_ontology_resource(
                        self._resource_path(asset["path"]),
                        receipt_path,
                        policy_hash="sha256:" + study["policy_hash"],
                        ontology_ids=study["visibility_policy"]["ontology_ids"],
                    )
                    if asset["media_type"] != receipt["format"]:
                        raise ValueError(
                            "Ontology serialization differs from its download metadata"
                        )
                    admitted_ontologies[asset["asset_id"]] = receipt["ontology_version_id"]
            except ValueError as exc:
                raise StudyError(
                    422, "Prepared resource failed policy or provenance admission"
                ) from exc
        for case in study["cases"]:
            available = {admitted_ontologies[aid] for aid in case["ontology_resource_ids"]}
            required = {
                entity["ontology_version_id"]
                for entity in [case["source"], *(c["entity"] for c in case["candidates"])]
            }
            if not required <= available:
                raise StudyError(
                    422, "Case ontology identities do not match their admitted downloads"
                )
            allowed_entities = {
                (entity["ontology_version_id"], entity["iri"], entity["kind"])
                for entity in [case["source"], *(c["entity"] for c in case["candidates"])]
            }
            allowed_candidates = {candidate["candidate_id"] for candidate in case["candidates"]}
            for asset_id in case["explanation_refs"]:
                resource = admitted_explanations[asset_id]
                focal_entities = resource.entities + [
                    entity
                    for claim in resource.entity_profiles + resource.pair_comparison
                    for entity in claim.scoped_entities
                ]
                if any(
                    (entity.ontology_version_id, entity.iri, entity.kind) not in allowed_entities
                    for entity in focal_entities
                ) or any(
                    evidence.candidate_id not in allowed_candidates
                    for evidence in resource.evidence
                ):
                    raise StudyError(422, "Explanation resource is outside its case scope")
        study["questionnaires"] = definitions(study["components"], study["form_version"])
        payload = canonical(study)
        with self.transaction() as db:
            if self.postgres:
                db.execute("SELECT pg_advisory_xact_lock(173912468)")
            for published in db.execute("SELECT payload FROM studies").fetchall():
                previous = json.loads(published["payload"])
                if (
                    previous["form_version"] == study["form_version"]
                    and previous["questionnaires"] != study["questionnaires"]
                ):
                    raise StudyError(
                        409, "Form version is already frozen with different definitions"
                    )
            existing = db.execute(
                "SELECT frozen_hash FROM studies WHERE revision = ?",
                (study["study_revision"],),
            ).fetchone()
            if existing:
                if existing["frozen_hash"] != digest(payload):
                    raise StudyError(
                        409, "Published revisions are immutable; publish a new revision"
                    )
                stored = json.loads(
                    db.execute(
                        "SELECT payload FROM researcher_case_keys WHERE study_revision = ?",
                        (study["study_revision"],),
                    ).fetchone()["payload"]
                )
                if stored != keys:
                    raise StudyError(409, "Published adjudication is immutable")
            else:
                db.execute(
                    "INSERT INTO studies(revision, payload, frozen_hash) VALUES (?, ?, ?)",
                    (study["study_revision"], payload, digest(payload)),
                )
                db.execute(
                    "INSERT INTO researcher_case_keys(study_revision, payload) VALUES (?, ?)",
                    (study["study_revision"], canonical(keys)),
                )
        return {
            "study_revision": study["study_revision"],
            "frozen_hash": digest(payload),
            "synthetic": study["synthetic"],
        }

    def _resource_path(self, locator):
        """Resolve a publication-owned path without escaping verified asset storage."""
        if self.assets_dir is None:
            raise StudyError(422, "Frozen asset storage is unavailable")
        relative = Path(locator)
        path = self.assets_dir / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or path.resolve().is_relative_to(self.assets_dir) is False
            or any(part.is_symlink() for part in [path, *path.parents] if part != self.assets_dir)
        ):
            raise StudyError(422, "Invalid frozen resource path")
        return path

    def _verified_asset_path(self, asset):
        """Hash large immutable downloads in bounded memory before streaming them."""
        path = self._resource_path(asset["path"])
        if not path.is_file() or path.stat().st_size != asset["size_bytes"]:
            raise StudyError(422, "Frozen resource is missing or changed")
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(block)
        if checksum.hexdigest() != asset["sha256"]:
            raise StudyError(422, "Frozen resource checksum mismatch")
        return path

    def _asset_bytes(self, asset):
        content = self._verified_asset_path(asset).read_bytes()
        if asset["media_type"] == "application/json":
            try:
                data = json.loads(content)
            except ValueError as exc:
                raise StudyError(422, "Frozen JSON resource is invalid") from exc
            self._check_safe_json(data)
        return content

    @staticmethod
    def _check_safe_json(data):
        forbidden = {
            "ground_truth",
            "case_kind",
            "acceptable_candidate_ids",
            "gold_target",
            "reference_membership",
            "answer_key",
            "original_production_ranks",
            "construction_provenance",
        }
        if isinstance(data, dict):
            if any(k.lower() in forbidden for k in data):
                raise StudyError(422, "Answer-bearing field in participant resource")
            for value in data.values():
                StudyStore._check_safe_json(value)
        elif isinstance(data, list):
            for value in data:
                StudyStore._check_safe_json(value)

    def issue(self, revision, count=1, *, test=True):
        """Issue reusable fragment links without creating assignments or access logs."""
        issued = []
        with self.transaction() as db:
            row, study = self._study(db, revision)
            if row["closed"] or datetime.fromisoformat(study["closes_at"]) <= datetime.now(
                timezone.utc
            ):
                raise StudyError(410, "Study closed")
            if study["synthetic"] and not test:
                raise StudyError(422, "Synthetic studies only permit test sessions")
            for _ in range(count):
                secret, sid = secrets.token_urlsafe(32), str(uuid4())
                state = {
                    "stage": "welcome",
                    "revision": 0,
                    "assignment": None,
                    "case_index": 0,
                    "consent": None,
                    "setup": None,
                    "questionnaires": {},
                    "rankings": {},
                    "consultations": {},
                    "pauses": [],
                    "gaps": [],
                    "missingness_reason": None,
                }
                db.execute(
                    "INSERT INTO sessions(id, study_revision, generation, invite_digest, test, state) VALUES (?, ?, 1, ?, ?, ?)",
                    (sid, revision, digest(secret), int(test), canonical(state)),
                )
                issued.append({"session_id": sid, "invitation": f"/participate#invite={secret}"})
        return {"invitations": issued}

    def exchange(self, secret):
        """Reuse the same invitation after cookie loss without starting participation."""
        with self.transaction() as db:
            found = db.execute(
                "SELECT id FROM sessions WHERE invite_digest = ?", (digest(secret),)
            ).fetchone()
            if not found:
                raise StudyError(401, "Invitation unavailable")
            try:
                row, state, study = self._session(db, found["id"])
            except StudyError as exc:
                raise StudyError(401, "Invitation unavailable") from exc
            if state["stage"] == "closed":
                raise StudyError(401, "Invitation unavailable")
            return row["id"], row["generation"], self._public_state(row, state, study)

    def reissue(self, sid, *, revoke=False):
        """Invalidate old links AND cookies while preserving the same session."""
        with self.transaction() as db:
            row = db.lock("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
            if row is None:
                raise StudyError(404, "Session not found")
            secret = secrets.token_urlsafe(32)
            db.execute(
                "UPDATE sessions SET generation = generation + 1, invite_digest = ?, revoked = ? WHERE id = ?",
                (digest(secret), int(revoke), sid),
            )
        return {
            "session_id": sid,
            "revoked": revoke,
            **({} if revoke else {"invitation": f"/participate#invite={secret}"}),
        }

    def close(self, revision):
        """Close a frozen study without replacing any participant records."""
        with self.transaction() as db:
            self._study(db, revision)
            db.execute("UPDATE studies SET closed = 1 WHERE revision = ?", (revision,))
        return {"study_revision": revision, "closed": True}

    def state(self, sid, generation):
        """Restore acknowledged progress and drafts using server-owned identity."""
        with self.transaction() as db:
            row, state, study = self._session(db, sid, generation)
            return self._public_state(row, state, study)

    @staticmethod
    def _current(state):
        assignment = state["assignment"]
        if assignment and state["case_index"] < len(assignment["presentations"]):
            return assignment["presentations"][state["case_index"]]
        return None

    def _public_state(self, row, state, study):
        current = self._current(state)
        case_id = current["case_id"] if current else None
        return {
            "artifact_type": "study_state",
            "contract_version": "exact-study/1.0",
            "study_revision": row["study_revision"],
            "session_id": row["id"],
            "revision": state["revision"],
            "stage": state["stage"],
            "assignment_id": (
                state["assignment"]["assignment_id"] if state["assignment"] else None
            ),
            "current_case_id": case_id,
            "current_presentation_id": current["presentation_id"] if current else None,
            "completed_cases": len(state["consultations"]),
            "assigned_case_count": (
                len(state["assignment"]["presentations"]) if state["assignment"] else 0
            ),
            "ranking": state["rankings"].get(case_id),
            "consultation": state["consultations"].get(case_id),
            "consent": state["consent"],
            "setup": state["setup"],
            "questionnaires": state["questionnaires"],
            "information_version": study["information_version"],
            "information_text": study["information_text"],
            "consent_text": study["consent_text"],
            "instructions": study["instructions"],
            "setup_instructions": study["setup_instructions"],
            "tutorial_steps": study["tutorial_steps"],
            "forms": study["questionnaires"],
            "ontology_resources": [
                {
                    k: v
                    for k, v in a.items()
                    if k not in {"path", "admission_receipt_path", "admission_receipt_sha256"}
                }
                for a in study["assets"]
                if a["kind"] == "ontology"
            ],
            "synthetic": bool(row["test"]),
            "gap_recovery": "Report external_work, break or unknown through resume after an unexpected disconnect; wall time is never assumed active.",
        }

    def _allocate(self, db, row, state, study):
        # The study row is locked only at allocation; invitations alone consume no slot.
        self._study(db, row["study_revision"], lock=True)
        # Preview sessions never consume production allocation slots.
        sequence = sum(
            bool(json.loads(record["state"])["assignment"])
            for record in db.execute(
                "SELECT state FROM sessions WHERE study_revision = ? AND test = ?",
                (row["study_revision"], row["test"]),
            ).fetchall()
        )
        schedule = study["schedules"][sequence % len(study["schedules"])]
        seed = secrets.randbits(63)
        rng = random.Random(seed)
        presentations = []
        for block_index, block in enumerate(schedule["blocks"]):
            case_ids = list(block["case_ids"])
            rng.shuffle(case_ids)
            for cid in case_ids:
                presentations.append(
                    {
                        "case_id": cid,
                        "condition": block["condition"],
                        "block_index": block_index,
                        "presentation_id": str(uuid4()),
                    }
                )
        state["assignment"] = {
            "assignment_id": str(uuid4()),
            "schedule_id": schedule["schedule_id"],
            "seed": seed,
            "presentations": presentations,
            "assigned_at": utcnow(),
        }
        db.execute(
            "UPDATE studies SET allocation_count = allocation_count + 1 WHERE revision = ?",
            (row["study_revision"],),
        )
        state["stage"] = "case"

    def mutate(self, sid, generation, operation, payload, case_id=None):
        """Replay an identical acknowledged mutation before checking stale revisions."""
        data = payload.model_dump(mode="json")
        request_hash = digest(
            canonical({"operation": operation, "case_id": case_id, "payload": data})
        )
        with self.transaction() as db:
            row, state, study = self._session(db, sid, generation)
            previous = db.execute(
                "SELECT * FROM mutations WHERE session_id = ? AND key = ?",
                (sid, data["idempotency_key"]),
            ).fetchone()
            if previous:
                if previous["payload_hash"] != request_hash:
                    raise StudyError(
                        409,
                        "Idempotency key was used for different content",
                        state["revision"],
                    )
                return json.loads(previous["result"])
            if data["expected_revision"] != state["revision"]:
                raise StudyError(
                    409, "Stale revision; restore the acknowledged state", state["revision"]
                )
            if state["stage"] in {"closed", "completed"}:
                raise StudyError(409, "Participation is closed", state["revision"])
            content = {
                k: v for k, v in data.items() if k not in {"idempotency_key", "expected_revision"}
            }
            self._transition(db, row, state, study, operation, content, case_id)
            state["revision"] += 1
            db.execute(
                "UPDATE sessions SET revision = ?, state = ? WHERE id = ?",
                (state["revision"], canonical(state), sid),
            )
            db.execute(
                "INSERT INTO history(session_id, revision, operation, payload, saved_at) VALUES (?, ?, ?, ?, ?)",
                (
                    sid,
                    state["revision"],
                    operation,
                    canonical({"case_id": case_id, **content}),
                    utcnow(),
                ),
            )
            result = self._public_state(row, state, study)
            db.execute(
                "INSERT INTO mutations(session_id, key, payload_hash, result) VALUES (?, ?, ?, ?)",
                (sid, data["idempotency_key"], request_hash, canonical(result)),
            )
            return result

    def _transition(self, db, row, state, study, operation, data, case_id):
        stage = state["stage"]
        if operation == "consent":
            if stage != "welcome" or data["information_version"] != study["information_version"]:
                raise StudyError(409, "Consent version or step does not match")
            state["consent"] = {**data, "acknowledged_at": utcnow()}
            state["stage"] = "setup" if data["accepted"] else "closed"
            return
        if not state["consent"] or not state["consent"]["accepted"]:
            raise StudyError(403, "Consent is required")
        if operation == "setup":
            if stage not in {"setup", "practice"}:
                raise StudyError(409, "Setup is unavailable at this step")
            state["setup"] = {
                **data,
                "verification": "self_reported_task_confirmed",
                "saved_at": utcnow(),
            }
            ready = all(
                data[key]
                for key in (
                    "protege_installed",
                    "source_opened",
                    "target_opened",
                    "practice_source_located",
                    "practice_definition_parents_inspected",
                )
            )
            if stage == "setup" and ready:
                state["stage"] = "background"
            elif (
                stage == "practice"
                and ready
                and set(data["completed_tutorial_steps"])
                == set(range(len(study["tutorial_steps"])))
            ):
                self._allocate(db, row, state, study)
            return
        if operation.startswith("questionnaire:"):
            form_id = operation.split(":", 1)[1]
            if form_id not in {"background", "final"} or stage != form_id:
                raise StudyError(409, "Questionnaire is unavailable at this step")
            if data["form_version"] != study["form_version"]:
                raise StudyError(409, "Questionnaire version changed")
            answers, answer_states = validate_answers(
                study["questionnaires"][form_id], data["answers"], submitted=data["submitted"]
            )
            state["questionnaires"][form_id] = {
                **data,
                "answers": answers,
                "answer_states": answer_states,
                "saved_at": utcnow(),
            }
            if data["submitted"] and form_id == "background":
                state["stage"] = "practice"
            return
        if operation == "pause":
            if stage == "paused":
                raise StudyError(409, "Already paused")
            state["paused_stage"] = stage
            state["stage"] = "paused"
            state["pauses"].append(
                {
                    "started_at": utcnow(),
                    "ended_at": None,
                    "case_id": (self._current(state) or {}).get("case_id"),
                }
            )
            return
        if operation == "resume":
            if stage == "paused":
                state["stage"] = state.pop("paused_stage")
                state["pauses"][-1]["ended_at"] = utcnow()
            last_receipts = [
                db.execute(
                    "SELECT MAX(saved_at) AS received FROM history WHERE session_id = ?",
                    (row["id"],),
                ).fetchone()["received"]
            ]
            last_receipts.extend(
                db.execute(
                    f"SELECT MAX(received_at) AS received FROM {table} WHERE session_id = ?",
                    (row["id"],),
                ).fetchone()["received"]
                for table in ("events", "timing_segments")
            )
            previous = max((value for value in last_receipts if value), default=None)
            reported = utcnow()
            state["gaps"].append(
                {
                    "reported_at": reported,
                    "last_server_receipt_at": previous,
                    "raw_gap_seconds": (
                        (
                            datetime.fromisoformat(reported) - datetime.fromisoformat(previous)
                        ).total_seconds()
                        if previous
                        else None
                    ),
                    "activity": data["gap_activity"],
                    "case_id": (self._current(state) or {}).get("case_id"),
                    "exact_duration_known": False,
                }
            )
            return
        if operation == "complete":
            if stage != "final" or not state["questionnaires"].get("final", {}).get("submitted"):
                raise StudyError(409, "Required final questionnaire is incomplete")
            state["stage"] = "completed"
            state["completed_at"] = utcnow()
            return
        current = self._current(state)
        if not current or current["case_id"] != case_id:
            raise StudyError(403, "Case is not the current assignment")
        if operation in {"draft", "submit"}:
            if stage != "case":
                raise StudyError(409, "The ranking is read-only")
            if data["presentation_id"] != current["presentation_id"]:
                raise StudyError(403, "Presentation is not authorized")
            case = next(c for c in study["cases"] if c["case_id"] == case_id)
            if not set(data["ranked_candidate_ids"]) <= {
                c["candidate_id"] for c in case["candidates"]
            }:
                raise StudyError(422, "Only presented candidates may be ranked")
            if operation == "submit" and data["response_type"] is None:
                raise StudyError(422, "Select an explicit response before submitting")
            if operation == "submit":
                # Saving an answer must survive telemetry loss. Missing case-ready
                # evidence stays explicit in the timing export instead of blocking it.
                state["stage"] = "consultation"
            now = utcnow()
            state["rankings"][case_id] = {
                "artifact_type": "ranking_response",
                "contract_version": "exact-study/1.0",
                "study_revision": row["study_revision"],
                "session_id": row["id"],
                "case_id": case_id,
                **data,
                "revision": state["revision"] + 1,
                "workflow_state": "submitted" if operation == "submit" else "draft",
                "saved_at": now,
                "submitted_at": now if operation == "submit" else None,
            }
            return
        if operation == "consultation":
            if stage != "consultation":
                raise StudyError(409, "Commit the ranking before consultation")
            state["consultations"][case_id] = {**data, "saved_at": utcnow()}
            state["case_index"] += 1
            state["stage"] = "case" if self._current(state) else "final"
            return
        raise StudyError(404, "Unknown study operation")

    def current_case(self, sid, generation):
        """Project a frozen case through its server-owned condition allowlist."""
        with self.transaction() as db:
            row, state, study = self._session(db, sid, generation)
            current = self._current(state)
            if not current or state["stage"] not in {"case", "consultation", "paused"}:
                raise StudyError(409, "No case available at this step")
            case = next(c for c in study["cases"] if c["case_id"] == current["case_id"])
            allowed = {
                k: case[k]
                for k in (
                    "case_id",
                    "source",
                    "source_label",
                    "candidates",
                    "ontology_resource_ids",
                    "package_version",
                )
            }
            return {
                "artifact_type": "study_case",
                "contract_version": "exact-study/1.0",
                "study_revision": row["study_revision"],
                **allowed,
                "presentation_id": current["presentation_id"],
                "condition": current["condition"],
                "explanation_refs": (
                    case["explanation_refs"] if current["condition"] == "explanation" else []
                ),
            }

    def resource(self, sid, generation, asset_id):
        """Verify bytes on every read; baseline never receives explanation resources."""
        with self.transaction() as db:
            _, state, study = self._session(db, sid, generation)
            if (
                not state["consent"]
                or not state["consent"]["accepted"]
                or state["stage"] == "closed"
            ):
                raise StudyError(403, "Consent is required")
            assets = {a["asset_id"]: a for a in study["assets"]}
            asset = assets.get(asset_id)
            if not asset:
                raise StudyError(404, "Resource unavailable")
            if asset["kind"] == "explanation":
                current = self._current(state)
                if (
                    not current
                    or current["condition"] != "explanation"
                    or state["stage"] not in {"case", "consultation", "paused"}
                ):
                    raise StudyError(403, "Resource unavailable in this condition")
                case = next(c for c in study["cases"] if c["case_id"] == current["case_id"])
                if asset_id not in case["explanation_refs"]:
                    raise StudyError(403, "Resource is outside the current case")
            elif self._current(state):
                case = next(
                    c for c in study["cases"] if c["case_id"] == self._current(state)["case_id"]
                )
                if asset_id not in case["ontology_resource_ids"]:
                    raise StudyError(403, "Resource is outside the current case")
            content = (
                self._verified_asset_path(asset)
                if asset["kind"] == "ontology"
                else self._asset_bytes(asset)
            )
            return content, asset["media_type"]

    @staticmethod
    def _records(db, table, sid):
        # Table names are internal constants, never participant input.
        return [
            json.loads(r["payload"])
            for r in db.execute(
                f"SELECT payload FROM {table} WHERE session_id = ? ORDER BY received_at",
                (sid,),
            ).fetchall()
        ]

    def events(self, sid, generation, batch):
        """Deduplicate telemetry independently of answer revisions and report gaps."""
        with self.transaction() as db:
            _, state, _ = self._session(db, sid, generation)
            if not state["consent"] or not state["consent"]["accepted"]:
                raise StudyError(403, "Consent is required")
            presentations = {
                p["case_id"]: p
                for p in (state["assignment"] or {}).get("presentations", [])[
                    : state["case_index"] + 1
                ]
            }
            acknowledged, gaps = [], []
            for model in batch.events:
                event = model.model_dump(mode="json")
                payload_hash = digest(canonical(event))
                existing = db.execute(
                    "SELECT payload_hash FROM events WHERE session_id = ? AND event_id = ?",
                    (sid, event["event_id"]),
                ).fetchone()
                if existing:
                    if existing["payload_hash"] != payload_hash:
                        raise StudyError(409, "Event ID reused for different content")
                    acknowledged.append(event["event_id"])
                    continue
                presentation = presentations.get(event["case_id"])
                if not presentation or presentation["presentation_id"] != event["presentation_id"]:
                    raise StudyError(403, "Event presentation is unauthorized")
                if presentation["condition"] == "ontology_baseline" and event["type"] in {
                    "hierarchy_expand",
                    "hierarchy_collapse",
                    "definition_open",
                    "axiom_open",
                    "evidence_open",
                    "table_open",
                    "graph_open",
                    "comparison_open",
                    "graph_zoom",
                    "graph_fit",
                }:
                    raise StudyError(403, "Component unavailable in baseline")
                if event["type"] == "case_ready" and (
                    state["stage"] != "case"
                    or (self._current(state) or {}).get("case_id") != event["case_id"]
                ):
                    raise StudyError(409, "Case timer cannot start outside its ranking step")
                same_sequence = db.execute(
                    "SELECT event_id FROM events WHERE session_id = ? AND page_id = ? AND sequence = ?",
                    (sid, event["page_instance_id"], event["sequence"]),
                ).fetchone()
                if same_sequence:
                    raise StudyError(409, "Event sequence already acknowledged")
                last = db.execute(
                    "SELECT MAX(sequence) AS sequence FROM events WHERE session_id = ? AND page_id = ?",
                    (sid, event["page_instance_id"]),
                ).fetchone()["sequence"]
                if event["sequence"] > (last + 1 if last is not None else 0):
                    gaps.append(
                        {
                            "page_instance_id": event["page_instance_id"],
                            "from": last + 1 if last is not None else 0,
                            "to": event["sequence"] - 1,
                        }
                    )
                event["server_received_at"] = utcnow()
                event["sequence_gap_before"] = (
                    gaps[-1]
                    if gaps
                    and gaps[-1]["page_instance_id"] == event["page_instance_id"]
                    and gaps[-1]["to"] == event["sequence"] - 1
                    else None
                )
                db.execute(
                    "INSERT INTO events(session_id, event_id, page_id, sequence, payload_hash, payload, received_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        sid,
                        event["event_id"],
                        event["page_instance_id"],
                        event["sequence"],
                        payload_hash,
                        canonical(event),
                        event["server_received_at"],
                    ),
                )
                acknowledged.append(event["event_id"])
            return {"acknowledged_event_ids": acknowledged, "sequence_gaps": gaps}

    def timing(self, sid, generation, model):
        """Save bounded observed intervals; gaps and hidden tabs are never auto-pauses."""
        segment = model.model_dump(mode="json")
        payload_hash = digest(canonical(segment))
        with self.transaction() as db:
            _, state, _ = self._session(db, sid, generation)
            if not state["consent"] or not state["consent"]["accepted"]:
                raise StudyError(403, "Consent is required")
            previous = db.execute(
                "SELECT payload_hash FROM timing_segments WHERE session_id = ? AND segment_id = ?",
                (sid, segment["segment_id"]),
            ).fetchone()
            if previous:
                if previous["payload_hash"] != payload_hash:
                    raise StudyError(409, "Timing segment ID reused for different content")
                return {"acknowledged_segment_id": segment["segment_id"]}
            if state["stage"] != segment["stage"]:
                raise StudyError(409, "Timing stage is no longer current")
            if segment["stage"] in {"case", "consultation"}:
                current = self._current(state)
                if (
                    not current
                    or current["case_id"] != segment["case_id"]
                    or current["presentation_id"] != segment["presentation_id"]
                ):
                    raise StudyError(403, "Timing presentation is unauthorized")
                if segment["stage"] == "case" and not any(
                    e["type"] == "case_ready"
                    and e["case_id"] == segment["case_id"]
                    and e["page_instance_id"] == segment["page_instance_id"]
                    and e["client_monotonic_ms"] <= segment["monotonic_start_ms"]
                    for e in self._records(db, "events", sid)
                ):
                    raise StudyError(409, "Case content has not been acknowledged usable")
            elif segment["case_id"] is not None or segment["presentation_id"] is not None:
                raise StudyError(422, "Non-case timing must not name a case")
            for earlier in self._records(db, "timing_segments", sid):
                if earlier["page_instance_id"] == segment["page_instance_id"] and max(
                    earlier["monotonic_start_ms"], segment["monotonic_start_ms"]
                ) < min(earlier["monotonic_end_ms"], segment["monotonic_end_ms"]):
                    raise StudyError(409, "Observed timing segments must not overlap")
            segment["server_received_at"] = utcnow()
            db.execute(
                "INSERT INTO timing_segments(session_id, segment_id, payload_hash, payload, received_at) VALUES (?, ?, ?, ?, ?)",
                (
                    sid,
                    segment["segment_id"],
                    payload_hash,
                    canonical(segment),
                    segment["server_received_at"],
                ),
            )
        return {"acknowledged_segment_id": segment["segment_id"]}

    def progress(self, revision):
        """Aggregate workflow counts without bearer secrets or questionnaire contents."""
        with self.transaction() as db:
            self._study(db, revision)
            counts = {}
            for row in db.execute(
                "SELECT state, test FROM sessions WHERE study_revision = ?", (revision,)
            ).fetchall():
                key = ("test:" if row["test"] else "participant:") + json.loads(row["state"])[
                    "stage"
                ]
                counts[key] = counts.get(key, 0) + 1
            return {"study_revision": revision, "counts": counts}

    def export(self, revision, *, include_test=False, include_keys=False):
        """Freeze a reproducible anonymous analysis export; credentials never enter it."""
        with self.transaction() as db:
            study_row, study = self._study(db, revision)
            keys = json.loads(
                db.execute(
                    "SELECT payload FROM researcher_case_keys WHERE study_revision = ?",
                    (revision,),
                ).fetchone()["payload"]
            )
            sessions, scored = [], []
            for row in db.execute(
                "SELECT id, state, test, revoked FROM sessions WHERE study_revision = ? ORDER BY id",
                (revision,),
            ).fetchall():
                if row["test"] and not include_test:
                    continue
                state = json.loads(row["state"])
                events = self._records(db, "events", row["id"])
                segments = self._records(db, "timing_segments", row["id"])
                forms = json.loads(canonical(state["questionnaires"]))
                for form_id, form in forms.items():
                    text_ids = {
                        q["id"] for q in study["questionnaires"][form_id] if q["options"] is None
                    }
                    form["answers"] = {
                        k: v for k, v in form["answers"].items() if k not in text_ids
                    }
                consultations = {
                    cid: {
                        k: v
                        for k, v in value.items()
                        if k not in {"other_editor", "other_resource"}
                    }
                    for cid, value in state["consultations"].items()
                }
                rows = []
                for presentation in (state["assignment"] or {}).get("presentations", []):
                    cid = presentation["case_id"]
                    case = next(c for c in study["cases"] if c["case_id"] == cid)
                    key = keys[cid]
                    response = state["rankings"].get(cid)
                    scores = score_response(
                        key["case_kind"],
                        key["acceptable_candidate_ids"],
                        response,
                        [c["candidate_id"] for c in case["candidates"]],
                        origin=key["origin"],
                    )
                    ready = [
                        e["server_received_at"]
                        for e in events
                        if e["case_id"] == cid and e["type"] == "case_ready"
                    ]
                    end = response.get("submitted_at") if response else None
                    elapsed = (
                        (
                            datetime.fromisoformat(end) - datetime.fromisoformat(min(ready))
                        ).total_seconds()
                        if end and ready
                        else None
                    )
                    observed = sum(
                        (s["monotonic_end_ms"] - s["monotonic_start_ms"]) / 1000
                        for s in segments
                        if s["case_id"] == cid and s["stage"] == "case"
                    )
                    timing = {
                        "raw_elapsed_seconds": elapsed,
                        "observed_segment_seconds": observed,
                        "unobserved_elapsed_seconds": (
                            max(0, elapsed - observed) if elapsed is not None else None
                        ),
                        "active_duration_known": False,
                        "gap_reports": [g for g in state["gaps"] if g["case_id"] == cid],
                        "explicit_breaks": [p for p in state["pauses"] if p["case_id"] == cid],
                    }
                    rows.append(
                        {
                            **presentation,
                            "candidates": case["candidates"],
                            "package_version": case["package_version"],
                            "case_kind": key["case_kind"],
                            "case_origin": key["origin"],
                            "first_response": (
                                response
                                if response and response["workflow_state"] == "submitted"
                                else None
                            ),
                            "final_response": (
                                response
                                if response and response["workflow_state"] == "submitted"
                                else None
                            ),
                            "draft_response": (
                                response
                                if response and response["workflow_state"] == "draft"
                                else None
                            ),
                            "consultation": consultations.get(cid),
                            "timing": timing,
                            "missingness_reason": (
                                None
                                if response and response["workflow_state"] == "submitted"
                                else state["missingness_reason"]
                                or (
                                    "revoked"
                                    if row["revoked"]
                                    else ("study_closed" if study_row["closed"] else "unsubmitted")
                                )
                            ),
                            "score": scores,
                        }
                    )
                    scored.append(scores)
                sessions.append(
                    {
                        "session_id": row["id"],
                        "test": bool(row["test"]),
                        "stage": state["stage"],
                        "revoked": bool(row["revoked"]),
                        "assignment": state["assignment"],
                        "questionnaires": forms,
                        "consent": state["consent"],
                        "setup": {
                            k: v
                            for k, v in (state["setup"] or {}).items()
                            if k != "protege_version"
                        },
                        "cases": rows,
                        "events": events,
                        "timing_segments": segments,
                    }
                )
            content = {
                "study_revision": revision,
                "software_version": study["software_version"],
                "policy_hash": study["policy_hash"],
                "questionnaire_definitions": study["questionnaires"],
                "sessions": sessions,
                "summary": summarize(scored),
                "condition_summaries": {
                    condition: summarize(
                        [
                            case["score"]
                            for session in sessions
                            for case in session["cases"]
                            if case["condition"] == condition
                        ]
                    )
                    for condition in ("explanation", "ontology_baseline")
                },
                "data_dictionary": {
                    "rr": "First acceptable candidate reciprocal rank; submitted positive cases only",
                    "system_mrr": "Mean original-system RR on the same submitted positive cases with original order available",
                    "mean_delta_rr": "Mean participant-minus-original-system RR on submitted positive cases",
                    "condition_summaries": "The same quality and missingness metrics restricted to each server-assigned condition",
                    "correct_to_wrong": "Count of natural submitted positives changing original correct top-1 to participant wrong/empty top-1",
                    "wrong_to_correct": "Count of natural submitted positives changing original wrong top-1 to participant correct top-1",
                    "draft_response": "Current recoverable unsubmitted answer; never included in submitted quality denominators",
                    "raw_gap_seconds": "Server receipt interval preceding a reconnect report; activity remains self-reported and uncertain",
                    "missing": "Unsubmitted cases are excluded from quality denominators",
                    "raw_elapsed_seconds": "Server usable-content receipt to ranking submit, including breaks and unknown gaps",
                    "observed_segment_seconds": "Bounded client monotonic observations; not proof of task activity",
                    "first_response": "First intentional submitted ranking, otherwise null; recoverable draft is separate",
                    "test": "Synthetic/preview records excluded unless include_test is explicitly requested",
                    "questionnaire_answer_states": "answered, skipped, not_answered; free text omitted from anonymous export",
                },
            }
            if include_keys:
                content["researcher_case_keys"] = keys
            export_id, now = str(uuid4()), utcnow()
            manifest = {
                "artifact_type": "export_manifest",
                "contract_version": "exact-study/1.0",
                "export_id": export_id,
                "created_at": now,
                "study_revision": revision,
                "content_sha256": digest(canonical(content)),
                "include_test": include_test,
                "includes_private_key_join": include_keys,
                "schema": "exact-study-analysis/1",
                "session_count": len(sessions),
            }
            result = {"manifest": manifest, "data": content}
            db.execute(
                "INSERT INTO study_exports(export_id, study_revision, payload, created_at) VALUES (?, ?, ?, ?)",
                (export_id, revision, canonical(result), now),
            )
            return result

    def saved_export(self, export_id):
        """Retrieve the immutable bytes of a previous researcher export."""
        with self.transaction() as db:
            row = db.execute(
                "SELECT payload FROM study_exports WHERE export_id = ?", (export_id,)
            ).fetchone()
            if not row:
                raise StudyError(404, "Export not found")
            return json.loads(row["payload"])
