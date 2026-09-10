#!/usr/bin/env python3
"""Bounded, real NCIT–DOID model/recovery proof; never a benchmark campaign.

Uses four canonically selected development sources and four unlabeled candidates
per source, with one-hop ontology declarations retained in a streamed OWL slice.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def owl_slice(source, destination, seeds):
    from lxml import etree

    rdf = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    owl = "http://www.w3.org/2002/07/owl#"
    selected = {}
    declarations = {}
    wanted = set(seeds)
    for _ in range(2):
        related = set()
        for _, node in etree.iterparse(str(source), events=("end",), huge_tree=True):
            if node.getparent() is None or node.getparent().tag != f"{{{rdf}}}RDF":
                continue
            iri = node.get(f"{{{rdf}}}about")
            if (
                node.tag
                in {
                    f"{{{owl}}}AnnotationProperty",
                    f"{{{owl}}}ObjectProperty",
                    f"{{{owl}}}DatatypeProperty",
                }
                and iri not in declarations
            ):
                declarations[iri] = copy.deepcopy(node)
            if node.tag == f"{{{owl}}}Class" and iri in wanted and iri not in selected:
                selected[iri] = copy.deepcopy(node)
                related.update(
                    item.get(f"{{{rdf}}}resource")
                    for item in node.iter()
                    if item.get(f"{{{rdf}}}resource")
                )
            node.clear()
            while node.getprevious() is not None:
                del node.getparent()[0]
        wanted.update(related)
    if set(seeds) - selected.keys():
        raise ValueError(
            f"Missing requested ontology entities: {sorted(set(seeds) - selected.keys())}"
        )
    root = etree.Element(f"{{{rdf}}}RDF", nsmap={"rdf": rdf, "owl": owl})
    etree.SubElement(root, f"{{{owl}}}Ontology", {f"{{{rdf}}}about": "urn:exact:operational-slice"})
    for iri in sorted(declarations):
        root.append(declarations[iri])
    for iri in sorted(selected):
        root.append(selected[iri])
    destination.write_bytes(etree.tostring(root, xml_declaration=True, encoding="utf-8"))
    return {
        "seed_entities": len(seeds),
        "retained_declarations": len(selected),
        "retained_property_declarations": len(declarations),
        "parent_sha256": digest(source),
        "slice_sha256": digest(destination),
        "slice_bytes": destination.stat().st_size,
        "policy": "seed declarations plus one-hop referenced declarations",
    }


def prepare(data, output):
    import csv

    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    with (data / "prepared/valid.candidates.tsv").open() as stream:
        rows = sorted(csv.DictReader(stream, delimiter="\t"), key=lambda row: row["SrcEntity"])[:4]
    sources = [row["SrcEntity"] for row in rows]
    targets = set()
    pair_count = 0
    with (inputs / "candidates.tsv").open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["SrcEntity", "TgtEntity", "TgtCandidates"], delimiter="\t"
        )
        writer.writeheader()
        for row in rows:
            candidates = sorted(ast.literal_eval(row["TgtCandidates"]))[:4]
            targets.update(candidates)
            pair_count += len(candidates)
            writer.writerow(
                {"SrcEntity": row["SrcEntity"], "TgtEntity": "", "TgtCandidates": repr(candidates)}
            )
    (inputs / "sources.txt").write_text("\n".join(sources) + "\n")
    with (data / "prepared/valid.reference.tsv").open() as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        refs = [row for row in reader if row["SrcEntity"] in sources]
        with (inputs / "reference.tsv").open("w") as destination:
            writer = csv.DictWriter(destination, fieldnames=reader.fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(refs)
    source = next((data / "ontologies").glob("NCIT-*.owl"))
    target = next((data / "ontologies").glob("DOID-*.owl"))
    return {
        "source": owl_slice(source, inputs / "source.owl", sources),
        "target": owl_slice(target, inputs / "target.owl", targets),
        "sources": len(sources),
        "pairs": pair_count,
        "source_universe_sha256": digest(inputs / "sources.txt"),
        "candidate_sha256": digest(inputs / "candidates.tsv"),
        "reference_sha256": digest(inputs / "reference.tsv"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reuse-inputs", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    evidence_path = output / "proof.json"
    evidence = {
        "scope": "operational real ontology slice; no full-graph throughput or quality claim",
        "requested_roles": ["lexical_encoder", "context_encoder", "deterministic_verbalisation"],
        "execution": "actual run_alignment in-process through instrumented harness subprocess seam",
        "started_at": time.time(),
        "status": "running",
    }
    if args.reuse_inputs and evidence_path.is_file():
        evidence["inputs"] = json.loads(evidence_path.read_text())["inputs"]
    elif args.reuse_inputs and (output / "inputs/target.owl").is_file():
        evidence["inputs"] = {
            "policy": "reuse previously streamed one-hop slices from this proof preparation",
            "sources": 4,
            "pairs": 16,
            "materialized": {
                path.name: digest(path) for path in (output / "inputs").iterdir() if path.is_file()
            },
            "parents": json.loads((args.data_root / "inputs.lock.json").read_text()),
        }
    else:
        evidence["inputs"] = prepare(args.data_root.resolve(), output)
    evidence_path.write_text(json.dumps(evidence, indent=2))
    snapshot = output / "source-snapshot"
    if not snapshot.exists():
        shutil.copytree(
            repo / "exact",
            snapshot / "exact",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        (snapshot / "tools").mkdir()
        shutil.copyfile(repo / "tools/run_exact_job.py", snapshot / "tools/run_exact_job.py")
    evidence["source_files"] = {
        str(path.relative_to(snapshot)): digest(path) for path in snapshot.rglob("*.py")
    }
    sys.path.insert(0, str(snapshot))
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="4",
    )
    from dataclasses import replace

    import torch

    from exact.core.actions.alignment import run_alignment
    from exact.core.entities.configs.config import ConfigModel
    from exact.experiments import harness
    from exact.experiments.harness import LoadedSuite, RunCell
    from exact.experiments.schema import ResourceConfig
    from exact.impl.models.scorer_common import ScorerCommonMixin
    from exact.llm.routing import OpenRouterClient

    inputs = output / "inputs"
    revision = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    encoder = "sentence-transformers/all-MiniLM-L6-v2"
    model_root = (
        Path.home()
        / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots"
        / revision
    )
    evidence["model_lock"] = {
        "requested_id": encoder,
        "resolved_revision": revision,
        "files": {path.name: digest(path) for path in model_root.iterdir() if path.is_file()},
    }
    evidence["device"] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "name": torch.cuda.get_device_name(0),
    }
    config = ConfigModel().model_dump(mode="json", by_alias=True)
    config["data"].update(
        root=str(inputs),
        source=str(inputs / "source.owl"),
        target=str(inputs / "target.owl"),
        candidates=str(inputs / "candidates.tsv"),
        source_universe=str(inputs / "sources.txt"),
        refs={"valid": str(inputs / "reference.tsv")},
        reference_role="valid",
        execution_mode="global_alignment",
        candidate_provenance="benchmark_supplied",
    )
    config["run"].update(seed=7, logging_level="WARNING", source_cap=None, experiment_audit=True)
    config["dataset"].update(
        filter_exact_matches=False, num_workers=0, verbalization_mode="deterministic"
    )
    primary = config["pipeline"][0]
    primary["params"].update(
        lexical_model_name=encoder,
        lexical_model_revision=revision,
        context_model_name=encoder,
        context_model_revision=revision,
        use_llm=False,
        generate_llm_rationales=False,
        fp16_inference=False,
    )
    config["pipeline"] = [primary]
    config["llm"]["verbaliser"]["model"] = None
    config["inference"].update(
        batch_size=4, num_workers=0, checkpoint_every=1, log_every=1, mixed_precision=False
    )
    config = ConfigModel.from_mapping(config, warn_v1=False).model_dump(mode="json", by_alias=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    suite = LoadedSuite(
        "operational-proof", "baseline", (), None, "operational-proof", None, None, {}
    )
    counter = {"encoder_batches": 0, "encoded_texts": 0, "llm_calls": 0}
    encode = ScorerCommonMixin._encode_texts

    def measured_encode(self, tokenizer, model, texts, max_len):
        counter["encoder_batches"] += 1
        counter["encoded_texts"] += len(texts)
        return encode(self, tokenizer, model, texts, max_len)

    def forbidden_llm(*_args, **_kwargs):
        counter["llm_calls"] += 1
        raise AssertionError("Generative call attempted in the fully non-generative proof")

    ScorerCommonMixin._encode_texts = measured_encode
    OpenRouterClient.chat_completion = forbidden_llm
    OpenRouterClient.completion = forbidden_llm

    def execute_alignment(command, *, cwd, stdout_path, stderr_path, env=None):
        import yaml

        wrapper = yaml.safe_load(Path(command[-1]).read_text())
        started = time.monotonic()
        previous = {key: os.environ.get(key) for key in (env or {})}
        os.environ.update(env or {})
        try:
            run_alignment(
                output_dir_path=Path(wrapper["job"]["output_dir"]),
                configs_file_path=Path(wrapper["job"]["config_file"]),
                run_eval=True,
                device=0,
            )
            return 0, time.monotonic() - started, None
        except KeyboardInterrupt:
            return 130, time.monotonic() - started, None
        except Exception:
            import traceback

            traceback.print_exc()
            raise
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    harness._run_subprocess = execute_alignment
    base = RunCell(
        "operational-proof",
        "E00",
        "screen",
        "production",
        "baseline",
        "NCIT-DOID-slice",
        "development",
        "valid",
        "complete",
        7,
        None,
        ResourceConfig(kind="gpu", device="cuda:0"),
        output / "uninterrupted/run",
        config,
        harness.hash_payload(config),
        "proof",
        "proof",
        None,
        "target_label_free",
        {},
        "not_applicable",
        recovery={"root": str(output / "uninterrupted")},
    )
    cases = [
        base,
        replace(
            base,
            output_dir=output / "interrupted/run",
            recovery={"root": str(output / "interrupted"), "stop_after_checkpoint": True},
        ),
        replace(
            base,
            output_dir=output / "relocated/run",
            recovery={
                "root": str(output / "relocated"),
                "resume_from": str(output / "interrupted"),
            },
        ),
        replace(
            base,
            output_dir=output / "replayed/run",
            recovery={"root": str(output / "replayed"), "resume_from": str(output / "relocated")},
        ),
    ]
    evidence["runs"] = []
    evidence_path.write_text(json.dumps(evidence, indent=2))
    for number, cell in enumerate(cases):
        torch.cuda.reset_peak_memory_stats(0)
        before = dict(counter)
        start = time.monotonic()
        result = harness.execute_cell(cell, suite, workdir=snapshot, resume=False)
        row = {
            "case": number,
            "root": str(cell.output_dir),
            "status": result["status"],
            "wall_seconds": time.monotonic() - start,
            "new_calls": {key: counter[key] - before[key] for key in counter},
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "manifest": result,
        }
        evidence["runs"].append(row)
        evidence_path.write_text(json.dumps(evidence, indent=2, default=str))
        expected = "interrupted" if number == 1 else "complete"
        if result["status"] != expected:
            raise RuntimeError(
                f"Operational case {number}: {result.get('failure', result['status'])}"
            )
    expected_maps = (cases[0].output_dir / "alignment/maps_global.tsv").read_bytes()
    assert (cases[2].output_dir / "alignment/maps_global.tsv").read_bytes() == expected_maps
    assert (cases[3].output_dir / "alignment/maps_global.tsv").read_bytes() == expected_maps
    from exact.runs.store import ExplanationStore

    def numerical_rows(cell):
        rows = list(ExplanationStore(cell.output_dir / "explanations").iter_all())
        result = {(row["src_iri"], row["tgt_iri"]): row["confidences"] for row in rows}
        assert len(rows) == len(result) == 16
        return result

    expected_rows = numerical_rows(cases[0])
    assert numerical_rows(cases[2]) == expected_rows
    assert numerical_rows(cases[3]) == expected_rows
    assert evidence["runs"][3]["new_calls"]["encoder_batches"] == 0
    assert evidence["runs"][0]["new_calls"]["encoded_texts"] == sum(
        evidence["runs"][i]["new_calls"]["encoded_texts"] for i in (1, 2)
    )
    assert counter["llm_calls"] == 0
    evidence["verified_pair_scores"] = len(expected_rows)
    evidence["status"] = "passed"
    evidence["completed_at"] = time.time()
    evidence["alignment_sha256"] = hashlib.sha256(expected_maps).hexdigest()
    evidence_path.write_text(json.dumps(evidence, indent=2, default=str))
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "proof": str(evidence_path),
                "runs": [
                    {key: row[key] for key in ("case", "status", "wall_seconds", "new_calls")}
                    for row in evidence["runs"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
