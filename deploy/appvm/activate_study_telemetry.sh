#!/usr/bin/env bash
# Owner-authenticated activation after real participants finish. No invitation/revision changes.
set -euo pipefail
umask 077
commit=${1:?Pass the tested release commit}
repo="$HOME/Exact-OM"
release="$HOME/exact-study/releases/telemetry-resilient-20261117"
private="$HOME/exact-study/telemetry-resilient-20261117/private"
revision=exact2-ncit-doid-production-20261117-traces
export PYTHONPATH="$release"
cd "$release"
test -f "$release/explanations_visualizer/out/participate/index.html"
"$repo/.venv/bin/python" - "$private" "$commit" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]); receipt=json.loads((p/'release-acceptance.json').read_text())
assert receipt['commit']==sys.argv[2]
assert receipt['frontend_tests']==162 and receipt['backend_tests']==50
assert receipt['typecheck'] and receipt['production_build']
assert json.loads((p/'persisted-verification.json').read_text())['passed']
b=json.loads((p/'browser-report.json').read_text()); assert len(b['journeys'])==2 and not b['browser_errors']
for filename,digest in receipt['files'].items():
 assert hashlib.sha256((Path.cwd()/filename).read_bytes()).hexdigest()==digest, filename
PY
if [ "$(git -C "$repo" rev-parse HEAD)" = "$commit" ] && [ "$(readlink -f "$repo/explanations_visualizer/out")" = "$release/explanations_visualizer/out" ] && [ -f "$private/activation.json" ]; then
 curl -fsS https://exact-explain.lasige.di.fc.ul.pt/api/ready
 printf '\nThe complete logging fix is already active; no restart performed.\n'
 exit 0
fi
test "$(git -C "$repo" branch --show-current)" = study-deploy
git -C "$repo" diff --quiet
git -C "$repo" diff --cached --quiet
git -C "$repo" fetch origin study-deploy
git -C "$repo" merge-base --is-ancestor "$commit" origin/study-deploy
# Authenticate before the last idle check so a password delay cannot bypass it.
sudo -v
"$repo/.venv/bin/python" -m tools.audit_study_telemetry --environment "$HOME/exact-study/study.env" --revision "$revision" --require-idle > "$private/activation-before-audit.json"
"$repo/.venv/bin/python" - "$private" "$revision" <<'PY'
import hashlib,json,os,subprocess,sys
from pathlib import Path
import psycopg
from psycopg.conninfo import conninfo_to_dict
from tools.audit_study_telemetry import read_environment
p=Path(sys.argv[1]); url=read_environment(Path.home()/'exact-study/study.env')['EXACT_STUDY_DATABASE_URL']
with psycopg.connect(url) as db:
 db.execute('SET TRANSACTION READ ONLY')
 rows=db.execute('SELECT id,study_revision,generation,invite_digest,revoked,test FROM sessions WHERE test=0 ORDER BY id').fetchall()
 frozen=db.execute('SELECT frozen_hash FROM studies WHERE revision=%s',(sys.argv[2],)).fetchone()[0]
 receipt={'invitation_fingerprint':hashlib.sha256(json.dumps(rows).encode()).hexdigest(),'frozen_hash':frozen,'invitations':len(rows)}
(p/'activation-identities.json').write_text(json.dumps(receipt))
env=os.environ.copy()
for key,value in conninfo_to_dict(url).items():
 name={'host':'PGHOST','port':'PGPORT','dbname':'PGDATABASE','user':'PGUSER','password':'PGPASSWORD'}.get(key)
 if name:env[name]=value
backup=p/'before-complete-fix.pgdump'
if not backup.exists():
 temporary=backup.with_suffix('.pgdump.partial')
 try:
  with temporary.open('wb') as f:subprocess.run(['pg_dump','--format=custom'],env=env,stdout=f,check=True)
  temporary.replace(backup)
 finally:
  temporary.unlink(missing_ok=True)
PY
previous_commit=$(git -C "$repo" rev-parse HEAD)
previous_frontend=$(readlink -f "$repo/explanations_visualizer/out")
printf '%s\n%s\n' "$previous_commit" "$previous_frontend" > "$private/previous-release.txt"
cp -an "$previous_frontend/_next/." "$release/explanations_visualizer/out/_next/"
# A final check immediately before activation catches someone starting during preparation.
"$repo/.venv/bin/python" -m tools.audit_study_telemetry --environment "$HOME/exact-study/study.env" --revision "$revision" --require-idle > "$private/activation-before-audit.json"
git -C "$repo" merge --ff-only "$commit"
ln -s "$release/explanations_visualizer/out" "$repo/explanations_visualizer/out.telemetry-new"
mv -Tf "$repo/explanations_visualizer/out.telemetry-new" "$repo/explanations_visualizer/out"
rollback() {
 git -C "$repo" reset --hard "$previous_commit"
 ln -s "$previous_frontend" "$repo/explanations_visualizer/out.telemetry-rollback"
 mv -Tf "$repo/explanations_visualizer/out.telemetry-rollback" "$repo/explanations_visualizer/out"
 sudo systemctl restart exact-study
 echo 'Activation failed; previous clean release restored. Participant data was not altered.' >&2
}
if ! sudo systemctl restart exact-study; then rollback; exit 1; fi
ready=0
for attempt in $(seq 1 30); do
 if curl -fsS http://127.0.0.1:8000/api/ready >/dev/null; then ready=1; break; fi
 sleep 1
done
if [ "$ready" != 1 ]; then rollback; exit 1; fi
"$repo/.venv/bin/python" - "$private" "$revision" <<'PY'
import hashlib,json,sys
from pathlib import Path
from urllib.request import urlopen
import psycopg
from tools.audit_study_telemetry import read_environment
p=Path(sys.argv[1]); expected=json.loads((p/'activation-identities.json').read_text())
with psycopg.connect(read_environment(Path.home()/'exact-study/study.env')['EXACT_STUDY_DATABASE_URL']) as db:
 db.execute('SET TRANSACTION READ ONLY')
 rows=db.execute('SELECT id,study_revision,generation,invite_digest,revoked,test FROM sessions WHERE test=0 ORDER BY id').fetchall()
 assert hashlib.sha256(json.dumps(rows).encode()).hexdigest()==expected['invitation_fingerprint']
 assert db.execute('SELECT frozen_hash FROM studies WHERE revision=%s',(sys.argv[2],)).fetchone()[0]==expected['frozen_hash']
origin='https://exact-explain.lasige.di.fc.ul.pt'
with urlopen(origin+'/api/ready',timeout=30) as response:assert json.load(response)['ready']
with urlopen(origin+'/participate/',timeout=30) as response:
 assert response.read()==(Path.cwd()/'explanations_visualizer/out/participate/index.html').read_bytes()
for route in ('/','/browse/','/admin/'):
 with urlopen(origin+route,timeout=30) as response:assert response.status==200
(p/'activation.json').write_text(json.dumps({'ready':True,'invitation_count':len(rows),'invitation_identities_preserved':True,'frozen_study_unchanged':True,'public_frontend_verified':True}))
PY
printf '\nComplete logging fix active. Existing study and invitation URLs are unchanged.\nAsk the next participants to refresh any study page already open before starting.\n'
