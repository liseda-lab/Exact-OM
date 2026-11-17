#!/usr/bin/env python3
"""Exercise installed CLI entrypoints offline, without model or provider requests.

Run with the distribution environment's Python from outside the source checkout.
Native encoder/GPU execution is deliberately a separate qualification step.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
import zipfile
from importlib.metadata import version
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="Optional JSON receipt outside temporary fixtures"
    )
    args = parser.parse_args()
    import exact

    package = Path(exact.__file__).resolve()
    if not package.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(
            f"Smoke must import the installed artifact inside this environment: {package}"
        )
    commands = []
    environment = {
        k: v
        for k, v in os.environ.items()
        if k not in {"PYTHONPATH", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"}
    }
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    with tempfile.TemporaryDirectory(prefix="exact-installed-") as temporary:
        root = Path(temporary)

        def cli(*argv, executable="exact"):
            command = [str(Path(sys.executable).parent / executable), *map(str, argv)]
            result = subprocess.run(
                command, cwd=root, env=environment, text=True, capture_output=True, timeout=120
            )
            commands.append(
                {"command": [executable, *map(str, argv)], "returncode": result.returncode}
            )
            if result.returncode:
                raise RuntimeError(f"{command} failed:\n{result.stdout}\n{result.stderr}")
            return result.stdout

        cli("--version")
        for command in (
            "align",
            "score",
            "decisions",
            "acceptance",
            "rank",
            "eval",
            "config",
            "data",
            "run",
        ):
            cli(command, "--help")
        for executable in ("exact-eval", "bioml-eval"):
            cli("--help", executable=executable)
        config = root / "config.yaml"
        cli("config", "default", "-o", config)
        migrated = root / "migrated.yaml"
        cli("config", "migrate", config, "-o", migrated)
        from exact.core.entities.configs.config import ConfigModel

        assert (
            ConfigModel.load_config(config).model_dump()
            == ConfigModel.load_config(migrated).model_dump()
        )
        tracks = cli("data", "list")
        assert "conference" in tracks and "anatomy" in tracks
        cli("data", "status", "--root", root / "empty-tracks")
        cli("data", "verify", "--root", root / "empty-tracks")
        refs = root / "reference.tsv"
        refs.write_text(
            "SrcEntity\tTgtEntity\tScore\nurn:s:a\turn:t:a\t1.0\nurn:s:b\turn:t:b\t1.0\n"
        )
        for executable in ("exact-eval", "bioml-eval"):
            cli("-a", refs, "-f", refs, "-o", root / executable, "-e", executable=executable)
        cli("eval", "-a", refs, "-f", refs, "-o", root / "unified-eval", "-e")
        from exact.runs import ExplanationStore, RunLayout, RunManifest, refresh_manifest

        layout = RunLayout.create(root / "run")
        ExplanationStore(layout.explanations_dir, run_id="smoke").append(
            [{"src_iri": "urn:s:a", "tgt_iri": "urn:t:a", "score": 0.8}]
        )
        checkpoint = layout.checkpoints_dir / "inference_smoke.json"
        checkpoint.write_text("{}")
        manifest = RunManifest.create(layout, run_id="smoke")
        manifest.register(checkpoint, kind="checkpoint", checksum=False)
        manifest.write()
        refresh_manifest(layout, run_id="smoke", manifest=manifest)
        cli("run", "info", layout.root)
        for format in ("json", "jsonl", "csv"):
            cli(
                "run",
                "export",
                layout.root,
                "--format",
                format,
                "--output",
                root / ("export." + format),
            )
        cli("run", "clean", layout.root, "--dry-run")
        assert checkpoint.exists()
        cli("run", "clean", layout.root)
        assert not checkpoint.exists()
        # A local HTTP server exercises the real HTTP track adapter without an external request.
        archive = root / "fixture.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            stream.writestr(
                "source.owl", '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>'
            )
            stream.writestr(
                "target.owl", '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/>'
            )
            stream.writestr("reference.tsv", refs.read_text())

        class Handler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *_):
                pass

        server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), functools.partial(Handler, directory=root)
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            descriptor = root / "track.yaml"
            descriptor.write_text(
                json.dumps(
                    {
                        "descriptor_version": 1,
                        "name": "fixture",
                        "provider": "http",
                        "upstream": {
                            "revision": "1",
                            "urls": [
                                {
                                    "url": f"http://127.0.0.1:{server.server_port}/fixture.zip",
                                    "filename": "fixture.zip",
                                    "extract": True,
                                    "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                                }
                            ],
                        },
                        "tasks": {
                            "demo": {
                                "source": "fixture/source.owl",
                                "target": "fixture/target.owl",
                                "refs": {"test": "fixture/reference.tsv"},
                            }
                        },
                    }
                )
            )
            for command in ("pull", "status", "verify"):
                cli(
                    "data",
                    command,
                    "fixture/demo",
                    "--root",
                    root / "tracks",
                    "--descriptor",
                    descriptor,
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        # Cached-only decision execution checks real prepare/run/apply console wrappers.
        rows = [
            {
                "Src": "urn:s:a",
                "Tgt": "urn:t:a",
                "S_base": 0.6,
                "U": 0.5,
                "selected_labels": {"source": "A", "target": "B"},
                "llm_structured_evidence_packet": "fixture",
            }
        ]
        pairs, sources, cache = root / "pairs.jsonl", root / "sources.txt", root / "answers.sqlite"
        pairs.write_text("".join(json.dumps(r) + "\n" for r in rows))
        sources.write_text("urn:s:a\n")
        cli(
            "decisions",
            "prepare",
            "--candidates",
            pairs,
            "--sources",
            sources,
            "--output",
            root / "decisions",
        )
        from exact.decisions import run

        class FakeEngine:
            def predict(self, payload):
                return {"probability": 0.9, "fixture": True}

        run(root / "decisions/plan.json", cache, engine=FakeEngine())
        cli(
            "decisions",
            "run",
            "--plan",
            root / "decisions/plan.json",
            "--cache",
            cache,
            "--cache-only",
        )
        cli(
            "decisions",
            "apply",
            "--plan",
            root / "decisions/plan.json",
            "--cache",
            cache,
            "--candidates",
            pairs,
            "--output",
            root / "scored.jsonl",
        )
        scored = json.loads((root / "scored.jsonl").read_text())
        assert abs(scored["S_pair_final"] - 0.72) < 1e-12
        # The installed fitting commands operate on tiny explicit-label fixtures.
        import numpy as np
        import pandas as pd
        from exact.acceptance.features import RAW

        rng = np.random.default_rng(9)
        rows = []
        for source in range(8):
            for target in range(4):
                row = {name: float(rng.uniform(0.1, 0.9)) for name in RAW}
                row.update(
                    Src=f"source{source}", Tgt=f"target{target}", original_exact_anchor=False
                )
                row["S_pair_final"] = 0.9 if source == target else 0.1
                rows.append(row)
        table, train = root / "acceptance.jsonl", root / "train.tsv"
        table.write_text("".join(json.dumps(r) + "\n" for r in rows))
        pd.DataFrame(
            [(f"source{i}", f"target{i}") for i in range(4)], columns=["Src", "Tgt"]
        ).to_csv(train, sep="\t", index=False)
        cli(
            "acceptance",
            "fit",
            "--candidates",
            table,
            "--train",
            train,
            "--output",
            root / "acceptance",
            "--family",
            "logistic",
            "--folds",
            "2",
        )
        for policy in ("partial", "full"):
            cli(
                "acceptance",
                "apply",
                "--candidates",
                table,
                "--model",
                root / "acceptance",
                "--policy",
                policy,
                "--output",
                root / ("acceptance-" + policy),
            )
        cli(
            "acceptance",
            "import-model",
            "--model",
            root / "acceptance/model.joblib",
            "--trusted-model",
            "--threshold",
            ".5",
            "--mode",
            "assignment",
            "--policy",
            "partial",
            "--output",
            root / "imported",
        )
        rank_config = ConfigModel()
        rank_config.selector.enabled = True
        rank_config.selector.rerank.mode = "pairwise"
        rank_config.pipeline[1].params["calibration"] = {"max_epochs": 3, "validation_folds": 2}
        rank_config_path = root / "rank-config.yaml"
        rank_config_path.write_text(json.dumps(rank_config.model_dump(mode="json")))
        for split in ("train", "test"):
            rank_rows, queries = [], []
            for i in range(4):
                source = f"{split}:{i}"
                for target, score in (("gold", 0.8), ("low", 0.3), ("high", 0.95)):
                    rank_rows.append(
                        dict(
                            Src=source,
                            Tgt=target,
                            S_final=score,
                            s_label=score,
                            S_struct=0.5,
                            s_hier=0.5,
                            s_sim=0.5,
                            s_attr=0.5,
                            s_diff=0.5,
                            cand_sim=score,
                            selector_evidence_items=[dict(key=target, strength=1.0)],
                        )
                    )
                query = dict(qid=i, source=source, candidates=["gold", "low", "high"])
                if split == "train":
                    query["gold_target"] = "gold"
                queries.append(query)
            (root / f"rank-{split}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rank_rows)
            )
            (root / f"queries-{split}.json").write_text(json.dumps(queries))
        cli(
            "rank",
            "fit",
            "--candidates",
            root / "rank-train.jsonl",
            "--queries",
            root / "queries-train.json",
            "--model",
            root / "rank.json",
            "--config",
            rank_config_path,
            "--confirmed-negatives",
        )
        cli(
            "rank",
            "apply",
            "--candidates",
            root / "rank-test.jsonl",
            "--queries",
            root / "queries-test.json",
            "--model",
            root / "rank.json",
            "--output",
            root / "ranking.tsv",
        )
        receipt = {
            "status": "passed",
            "version": version("Exact-OM"),
            "installed_package": str(package),
            "commands": commands,
            "provider_calls": 0,
            "new_model_calls": 0,
            "decision_backend": "offline fake probability; installed cached-only CLI",
            "native_encoder_execution": "separately qualified; not run by this script",
            "dependency_installation": "not performed by this script",
        }
    text = json.dumps(receipt, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
