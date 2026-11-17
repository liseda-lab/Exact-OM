# Reviewer demo — 17 November 2026

## Scope and current state

Prepared and acceptance-tested on AppVM; public activation is pending the owner's
one-time sudo bootstrap. The active study has not been restarted, republished or
modified. Its process remains PID 230372, started 14 November at 16:50:50 UTC.

The demo uses only the latest deployed study revision,
`exact2-ncit-doid-production-20261114-r2`, including its description corrections.
Its ten sources are drawn from that revision's 40, with two sources per private
reference-rank position 1–5 and five unchanged candidates per source. Selection
favours distinguishing definitions, useful evidence and varied subject matter.
The two available rank-4 cases necessarily retain some sparse ontology descriptions.
No extra clinical facts, generated descriptions, matcher results or reasoning runs
were created for this demo.

Selected concepts (not grouped by reference rank):

- Chronic Infantile Neurological Cutaneous and Articular Syndrome.
- Spinal Muscular Atrophy Type 3 (the initial landing example).
- Mitochondrial Neurogastrointestinal Encephalopathy.
- Peripheral Neuropathy.
- Encephalitis Lethargica.
- Chronic Myeloid Leukemia, BCR-ABL1 Positive.
- Uterine Corpus Leiomyoma.
- Complete Atrioventricular Block.
- Colorectal Adenocarcinoma.
- Meningothelial Cell Neoplasm.

The private selection report retains reference-rank evidence; the public package
and served resources do not contain reference keys or answers. This is a purposive
showcase, not an accuracy sample. The owner explicitly requested reuse of study
cases: process isolation does not prevent a participant from independently visiting
the public demo, so these should not be described as inaccessible external stimuli.

## Data and serving

`tools/prepare_reviewer_demo.py` reads the latest publication's verified workspace
indexes. Profiles, comparisons and selected evidence come from those indexes,
not an older generation cache. It verifies context/index hashes and preserves
original candidate scores, ordering and entity identities. The full existing,
policy-filtered ontology databases are referenced in place; they are neither
copied nor hardlinked. The new demo payload is about 7.5 MB.

Package: `~/exact-study/assets/reviewer-demo-20261117-package.json`.
Package ID: `sha256:00b0fbaf9c1eebeca44c6bb40a53522fda85952676ae55fdb5e9c521f7b1ff86`.
Private preparation, selection, validation and browser logs: `~/exact-demo/private`.
Versioned code/export: `~/exact-demo/releases/reviewer-20261117`, exposed through
`~/exact-demo/release`.

The complete Compare and Browse interfaces include independent source/target
context, original/generated text and citations, evidence list/graph, scores,
recorded ranks, hierarchy and full-ontology search. Asserted, structural and partial
ELK views retain their original semantics. The source study export did not retain
full pipeline event history or final alignment membership; trace views report those
as unrecorded/unknown rather than inventing decisions.

`public_demo` disables imports and bundle management at the API boundary. No
library is configured. The server has no study database connection or credentials.
The separate `exact-demo.service` listens on loopback port 8010, capped at one CPU,
2 GB RAM and 24 concurrent requests. Its filesystem is read-only, with study
credentials and private preparation directories additionally inaccessible.

## Routing and study preservation

The reviewed Nginx configuration directs only `/`, `/browse`, `/browse/` and
non-study `/api/v1/` requests to the demo. `/participate/`, `/admin/`, study APIs,
admin APIs and `/api/ready` retain the existing port-8000 upstream. Study build
assets are served from that upstream first; only unknown asset hashes fall back
to the demo. No file in the live study frontend is replaced.

An unprivileged staging Nginx on port 18081 verified that study pages and all their
referenced scripts/styles are byte-identical to the existing upstream, and that
study/admin API authentication responses are unchanged. The deployment helper
reloads Nginx gracefully and never restarts `exact-study`. It saves the old vhost
and restores it if configuration validation/reload fails. `rollback` explicitly
restores that vhost and stops only the demo.

Normal rich-context rendering exceeded the initial 10 requests/second, burst-40
limit. The corrected demo-only allowance is 30 requests/second with burst 200;
unchanged browser acceptance tests then passed. CPU, memory, concurrency and upload
limits remain separate. IT's proxy may group clients under one address, so this
limit can be shared; monitor actual reviewer load rather than calling it a capacity
benchmark.

## Verification

- All 50 pairs have unchanged identities, raw scores and recorded ordering.
- 110 profile/comparison reads equal their latest study workspace claims and packets.
- All pairs have evidence; both ontology contexts and all three hierarchy bases read
  correctly, including explicit `partial` status for ELK.
- Import and bundle-management endpoints reject requests; demo study/admin endpoints
  are absent, while the public router preserves their separate authenticated service.
- Four browser tests passed through the exact proposed router: comparison/tab and
  bookmark navigation, every visible generated-text citation and focus return,
  graph controls/style persistence, ten-source pool, themes, responsive layout and
  browse. No browser errors. A dark-mode screenshot was visually inspected.
- Frontend production build, TypeScript checks and all 144 unit tests passed.
- Native `local_app` HTTP smoke check passed Library rendering and import-job
  creation using a temporary library. No live library or study data was changed.
- `visudo -cf` accepted the scoped expiry policy; `systemd-analyze verify` accepted
  the service unit. Docker was unavailable locally, so Compose image build/run was
  not executed. The native serving path was tested; do not call this a Docker test.

A test harness initially needed unavailable `httpx2`; verification used a real
short-lived HTTP server instead, without installing anything into the study venv.

## Temporary deployment access

Inspected the local `liseda_cluster/scripts/temporary-admin-access.py`. Its NOTAFTER,
validation and revocation approach is adapted in `deploy/appvm/install.py` and
`maintenance.py`, with a narrower grant: only fixed `activate`, `rollback` and
`revoke` commands of a root-owned helper, not unrestricted root access. Default
expiry is four hours, bounded to 1–24 hours. The helper cannot restart the study.

Once acceptance receipts are present, the owner runs from their own terminal:

```sh
ssh -t AppVM 'sudo /usr/bin/python3 ~/exact-demo/release/deploy/appvm/install.py --hours 4'
```

This uses normal interactive sudo on AppVM; no password enters chat or a file.
Afterward the agent can activate with:

```sh
ssh AppVM 'sudo -n /usr/local/sbin/exact-demo-maintenance activate'
```

Verify public demo, participant/admin paths, study PID and asset identities, then
revoke with `sudo -n /usr/local/sbin/exact-demo-maintenance revoke`. Expiry remains a
backstop if the session is interrupted. To renew for later demo maintenance, rerun
the owner bootstrap. The grant intentionally does not cover study administration.

## Local installation

The root README links `docs/guides/explanation-local-install.md`. The package now
includes `deploy/local/compose.yaml` for a persistent local import library and
`deploy/local/start.sh` for a Python/Node installation without Docker. Both bind
loopback by default. Offline preparation remains documented separately; prepared
viewing requires neither a GPU nor a provider key. The backend guide's outdated
claim that the Docker image lacked a frontend was corrected.
