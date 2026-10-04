"""Strict no-call proof for legacy sampled-pool metadata in an E21 comparison."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

from exact.utils.fitted_artifacts import fingerprint
from exact.utils.provenance import sha256_file


def observed_judge_population(judge_item, baseline_item, judge_config, baseline_config):
    """Admit omitted sampled encoder metadata only with identical observed inputs."""
    from .staged_selection import PrerequisiteUnavailable

    proofs, inventories, policies, manifests, samples = [], [], [], [], []

    def require(condition, reason):
        if not condition:
            raise PrerequisiteUnavailable("Judge benefit same population proof failed: " + reason)

    for item, config in ((judge_item, judge_config), (baseline_item, baseline_config)):
        root = Path(item["fingerprint_payload"]["output_dir"])
        bindings = {}

        def read(relative):
            path = root / relative
            require(path.is_file(), f"missing {relative}")
            bindings[relative] = {"path": str(path), "sha256": sha256_file(path)}
            return json.loads(path.read_text())

        full = read("dataset/candidate_pool_manifest.json")
        sample = read("dataset/candidate_pool_sample_manifest.json")
        require(sample.get("fingerprint") == item["candidate_pool_fingerprint"], "sample identity")
        normalized = deepcopy(sample)
        normalized.pop("fingerprint")
        # Older recovered sampled manifests omitted encoder/retrieval provenance.
        # Only absent metadata may be restored from the byte-identical parent pool.
        encoder = normalized["models"]["encoder"]
        expected_encoder = full["models"]["encoder"]
        require(set(encoder) == set(expected_encoder), "encoder metadata schema")
        require(
            all(value is None or value == expected_encoder[key] for key, value in encoder.items()),
            "encoder metadata conflict",
        )
        normalized["models"]["encoder"] = expected_encoder
        retrieval = normalized["retrieval_config"]
        expected_retrieval = full["retrieval_config"]
        require("source_sample" in retrieval, "missing sampled source binding")
        require(
            all(
                key in expected_retrieval and value == expected_retrieval[key]
                for key, value in retrieval.items()
                if key != "source_sample"
            ),
            "retrieval metadata conflict",
        )
        normalized["retrieval_config"] = {
            **expected_retrieval,
            "source_sample": retrieval["source_sample"],
        }
        samples.append(normalized)
        manifests.append(bindings["dataset/candidate_pool_manifest.json"]["sha256"])

        decisions = read("source_decisions.json")
        require(
            decisions.get("schema_version") == 2
            and decisions.get("reference_labels_used") is False,
            "unsafe source decisions",
        )
        records = decisions["records"]
        sources, pairs = set(), []
        for record in records:
            source = record["Src"]
            require(source not in sources, "duplicate source record")
            sources.add(source)
            for row in record["candidates"]:
                require(isinstance(row["protected_exact"], bool), "unknown exact-anchor status")
                anchor = None
                if row["S_base"] is None or row["U"] is None:
                    # Fixed exact anchors bypass the scorer; null features are
                    # observed absence, never invented numeric training evidence.
                    require(
                        row["protected_exact"]
                        and row["S_base"] is None
                        and row["U"] is None
                        and row.get("S_final") == 1.0
                        and isinstance(row.get("emitted"), bool)
                        and isinstance(row.get("pre_typing_selected"), bool)
                        and (row["emitted"], row["pre_typing_selected"], row.get("reason"))
                        in {
                            (True, True, "emitted"),
                            (False, False, "cardinality_or_extraction"),
                        },
                        "inconsistent protected exact anchor",
                    )
                    anchor = {
                        key: row[key]
                        for key in ("S_final", "emitted", "pre_typing_selected", "reason")
                    }
                else:
                    require(
                        all(math.isfinite(float(row[key])) for key in ("S_base", "U")),
                        "nonfinite scores",
                    )
                pairs.append(
                    [source, row["target"], row["S_base"], row["U"], row["protected_exact"], anchor]
                )
        require(len({(row[0], row[1]) for row in pairs}) == len(pairs), "duplicate pairs")
        require(sources == set(decisions["source_universe"]), "incomplete source inventory")
        inventories.append(sorted(pairs))
        policy = {
            key: value
            for key, value in decisions["policy"].items()
            if key not in {"llm", "extraction_diagnostics"}
        }
        require(policy["threshold"] == config.matching.threshold, "effective acceptance threshold")
        policies.append(policy)
        stats = read("stats/run_stats.json")["source_sampling"]
        require(
            stats["candidate_pool_fingerprint"] == item["candidate_pool_fingerprint"],
            "sampling identity",
        )
        require(
            stats["cap"] == item["source_cap"] and stats["seed"] == item["seed"], "sampling scope"
        )
        require(stats["selected_sources"] == len(sources), "sampling source count")
        evaluation = read("evaluation/evaluation_results.json")
        references = {}
        for role in ("full_reference", "train_reference"):
            ref = evaluation["meta"]["refs"][role]
            ref_path = Path(ref["path"])
            require(
                ref_path.is_file() and sha256_file(ref_path) == ref["sha256"], f"changed {role}"
            )
            references[role] = {key: ref[key] for key in ("sha256", "bytes", "rows")}
            bindings[role] = {"path": str(ref_path), "sha256": ref["sha256"]}
        proofs.append(
            {
                "inputs": bindings,
                "source_universe": decisions["source_universe"],
                "dataset_signature": decisions["dataset_signature"],
                "sample_sha256": stats["sample_sha256"],
                "references": references,
            }
        )
    require(manifests[0] == manifests[1], "parent pool manifest bytes differ")
    require(samples[0] == samples[1], "sampled inputs differ beyond omitted provenance")
    require(
        inventories[0] == inventories[1] and inventories[0],
        "candidate scores or protected anchors differ",
    )
    require(policies[0] == policies[1], "effective acceptance differs")
    require(
        {key: value for key, value in proofs[0].items() if key != "inputs"}
        == {key: value for key, value in proofs[1].items() if key != "inputs"},
        "source sample or evaluation references differ",
    )
    return {
        "kind": "identical_observed_population_with_legacy_sample_metadata",
        "inputs": [proof["inputs"] for proof in proofs],
        "parent_pool_sha256": manifests[0],
        "sample_sha256": proofs[0]["sample_sha256"],
        "source_count": len(proofs[0]["source_universe"]),
        "pair_count": len(inventories[0]),
        "source_scores_sha256": fingerprint(inventories[0]),
        "effective_acceptance": policies[0],
        "references": proofs[0]["references"],
    }
