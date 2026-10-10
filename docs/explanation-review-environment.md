# Durable local development review

This entrypoint is for a development review of real alignment data. It accepts only an
explicit `development_review` publication with `synthetic=false`, disabled correctness
metrics, and the `study-review/1` form/method contract. It issues test invitations only.
It does not establish research approval or make research correctness claims.

Use a durable local root such as
`~/Library/Caches/exact-om-review/s1-20261010`, or set `EXACT_REVIEW_ROOT` to another persistent,
ignored directory. Keep the prepared package, admitted assets, source publication and compiled
frontend in durable directories too. Temporary directories are rejected as their sole location.
Nothing below reinstalls dependencies, rebuilds ontology assets, contacts an LLM, or deletes a
previous PostgreSQL cluster.

Prepare once from the checkout containing the corrected backend and the compiled frontend:

```sh
REVIEW_PYTHON="$HOME/Library/Caches/exact-om-dev/pyenv/bin/python"
REVIEW_PGBIN="$HOME/Library/Caches/exact-om-dev/pyenv/lib/python3.12/site-packages/pgserver/pginstall/bin"
REVIEW_ROOT="$HOME/Library/Caches/exact-om-review/s1-20261010"

"$REVIEW_PYTHON" -m tools.review_environment prepare \
  --root "$REVIEW_ROOT" \
  --python "$REVIEW_PYTHON" --pgbin "$REVIEW_PGBIN" \
  --publication /absolute/durable/path/publication.json \
  --assets /absolute/durable/path/admitted-assets \
  --package /absolute/durable/path/package.json \
  --frontend /absolute/durable/path/compiled-frontend \
  --postgres-port 19183 --study-port 19184 --exploration-port 19185
```

Replace the four input paths with the reviewed bundle's actual paths. Preparation verifies the
input contracts and immutable bytes, records dependency versions and frontend/package hashes,
creates a private local TLS certificate, and initializes a new dedicated PostgreSQL cluster only
if no database directory already exists. Its new publication revision has a new expiry; the source
publication is not edited. Repeating `prepare` validates and reuses the existing configuration.
Changing paths or ports requires a separate root.

The certificate is at `REVIEW_ROOT/tls/localhost.pem`. Inspect its SHA-256 fingerprint with
`openssl x509 -in "$REVIEW_ROOT/tls/localhost.pem" -noout -fingerprint -sha256`. To trust it in a
local browser on macOS, import that certificate into the login Keychain, inspect its name and
fingerprint, and explicitly trust it for SSL. This is a local choice; the tool does not change
system or browser trust settings. Its own HTTPS requests trust only this configured certificate
and verify the hostname. Do not disable TLS verification globally. The certificate includes both
`localhost` and `127.0.0.1`; preflight rejects expired certificates or a mismatched private key.
If the certificate has expired, rotate it explicitly while keeping the same database:

```sh
"$REVIEW_PYTHON" -m tools.review_environment renew-certificate --root "$REVIEW_ROOT"
```

This verifies the unchanged inputs, stops only the owned HTTPS workers and writes a new
certificate/key pair without overwriting the previous files. PostgreSQL and participation data
remain intact. Inspect and trust the newly reported certificate, then run `start` or, if the
publication also expired, `fresh-session --new-revision`.

```sh
"$REVIEW_PYTHON" -m tools.review_environment start --root "$REVIEW_ROOT"
"$REVIEW_PYTHON" -m tools.review_environment status --root "$REVIEW_ROOT"
"$REVIEW_PYTHON" -m tools.review_environment fresh-session --root "$REVIEW_ROOT"
```

The study is at `https://127.0.0.1:19184/participate/`; exploration is at
`https://127.0.0.1:19185/`. All services bind to loopback. PostgreSQL accepts authenticated TCP
connections on its own port and does not create Unix sockets. Keep the two browsing profiles
separate: exploration does not become participant consultation evidence.

`fresh-session` writes the test invitation to a private file under `REVIEW_ROOT/invitations/`.
Its output identifies that file without printing the invitation secret. Open that local receipt
and use its invitation URL. The participant client must send `client_contract="study-review/1"`
at session exchange. Do not share private receipts or `environment.json` as public review evidence.
No invite secret is put in a process command line or service access log.

If a publication expires, preserve it and request a new revision and expiry:

```sh
"$REVIEW_PYTHON" -m tools.review_environment fresh-session \
  --root "$REVIEW_ROOT" --new-revision --days 30
```

This creates a different immutable publication file, publishes that new revision and issues a new
test invitation. It can restart stopped owned services against the preserved database. Existing
revisions, sessions, answers and exports remain in the database. An expired TLS certificate still
fails preflight; changing the publication cannot bypass certificate checks.

```sh
"$REVIEW_PYTHON" -m tools.review_environment stop --root "$REVIEW_ROOT"
```

Stop checks the recorded PID, process start identity and random ownership nonce before signaling
anything. It stops only this environment's processes and leaves the database, publications,
certificates, invitations and logs in place. Repeating start/stop is safe. A foreign process or
occupied port causes an explicit failure without terminating that process. A stale PID record is
not authority to stop a different process. Do not remove a database directory to fix a startup
problem; inspect the private role logs under `REVIEW_ROOT/logs/`.

Development tests cover stale/foreign PID protection, owned child shutdown, exact TLS SAN and key
checks, immutable revisions, asset integrity, private test invitation receipts and real PostgreSQL
stop/start data preservation on a separate port. These tests do not replace the final real-bundle
study and exploration browser journeys after backend/frontend integration.
