"""Cold-import the Dockerfile's copied source subset without building a Docker image."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

REPOSITORY = Path(__file__).resolve().parents[1]
DOCKERFILE = REPOSITORY / "deploy/render/exact_study.Dockerfile"


def copy_study_runtime(app_root):
    """Reproduce final Python-stage local COPY layout, including package data files."""
    python_stage = False
    copied = []
    # Shell-form local COPY is what this Dockerfile uses; fail visibly if its
    # layout changes to an unsupported syntax instead of silently missing files.
    lines = DOCKERFILE.read_text().replace("\\\n", " ").splitlines()
    for line in lines:
        tokens = shlex.split(line, comments=True)
        if not tokens:
            continue
        if tokens[0].upper() == "FROM":
            python_stage = tokens[1].startswith("python:")
        if not python_stage or tokens[0].upper() != "COPY":
            continue
        if any(token.startswith("--from=") for token in tokens[1:]):
            continue
        assert not tokens[1].startswith(("--", "[")), "Update source-layout test for COPY syntax"
        sources, destination = tokens[1:-1], PurePosixPath(tokens[-1])
        assert destination.is_relative_to("/app"), "Study COPY must stay inside /app"
        destination_root = app_root / destination.relative_to("/app")
        for source in sources:
            source_path = REPOSITORY / source
            assert source_path.exists(), f"Missing Docker source: {source}"
            if not source_path.is_dir() and source_path.suffix != ".py":
                continue
            copied.append(source)
            if source_path.is_dir():
                shutil.copytree(
                    source_path,
                    destination_root,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
            else:
                target = (
                    destination_root / source_path.name
                    if tokens[-1].endswith("/") or len(sources) > 1
                    else destination_root
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target)
    assert "exact_inspect/study" in copied
    assert not (app_root / "exact").exists()
    assert not (app_root / "exact_inspect/__init__.py").exists()
    return copied


def test_study_docker_source_subset_cold_imports_without_matcher_or_native_parser(tmp_path):
    app_root = tmp_path / "app"
    copied = copy_study_runtime(app_root)
    script = r"""
import importlib.abc, json, sys
from pathlib import Path
app_root, repository = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
# -I excludes cwd and PYTHONPATH; also remove repository paths from any editable
# installation supplied by the invoking environment. Dependencies remain usable.
sys.path[:] = [entry for entry in sys.path if entry and not Path(entry).resolve().is_relative_to(repository)]
sys.path.insert(0, str(app_root))
assert all(not Path(entry).resolve().is_relative_to(repository) for entry in sys.path)
for name in list(sys.modules):
    if name == 'exact_inspect' or name.startswith('exact_inspect.'):
        del sys.modules[name]
forbidden = {'exact', 'torch', 'pyowl_core', 'transformers', 'sentence_transformers'}
assert not forbidden.intersection(sys.modules)
class ForbidMatcher(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in forbidden:
            raise AssertionError('Minimal study runtime attempted forbidden import: ' + fullname)
sys.meta_path.insert(0, ForbidMatcher())
from exact_inspect.study.api import create_study_app
assets = app_root / 'study-assets'
assets.mkdir()
app = create_study_app('sqlite:///' + str(app_root / 'study.sqlite'), 's' * 40, 'r' * 40,
                       'https://study.example.test', assets, allow_test_sqlite=True)
assert app.state.study_store.ready()
paths = app.openapi()['paths']
for route in ('/api/v1/study/state', '/api/v1/study/setup', '/api/v1/study/tutorial/progress',
              '/api/v1/study/tutorial/assessment', '/api/v1/study/tutorial/complete',
              '/api/v1/study/workspace/{scope_id}/capabilities', '/api/v1/admin/studies'):
    assert route in paths, route
assert not any(route.startswith(('/api/v1/bundles', '/api/v1/runs', '/api/v1/entity-context')) for route in paths)
health = next(route.endpoint for route in app.routes if getattr(route, 'path', '') == '/api/health')()
assert health['status'] == 'ok'
loaded = []
for name, module in sys.modules.items():
    if name == 'exact_inspect' or name.startswith('exact_inspect.'):
        path = getattr(module, '__file__', None)
        if path:
            assert Path(path).resolve().is_relative_to(app_root), (name, path)
            loaded.append(name)
assert not forbidden.intersection(sys.modules)
print(json.dumps({'copied_runtime_import': 'passed', 'route_count': len(paths), 'loaded_local_modules': sorted(loaded)}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(app_root), str(REPOSITORY)],
        cwd=app_root,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    receipt = json.loads(result.stdout)
    assert receipt["copied_runtime_import"] == "passed"
    assert "exact_inspect.study.workspace" in receipt["loaded_local_modules"]
    assert receipt["route_count"] > 20
    assert len(copied) > 3
