"""One bounded, auditable anchor second pass over a frozen candidate pool."""

import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

from exact.impl.models.selector.fitting import fingerprint, freeze_json
from exact.utils.data import read_table

from .audit_io import _semantic_collate_fn
from .fitting import source_batches


def predicted_anchors(rows, *, threshold, margin):
    frame = pd.DataFrame(rows, columns=["Src", "Tgt", "score"]).sort_values(
        ["score", "Src", "Tgt"], ascending=[False, True, True]
    )
    if frame.empty:
        return []
    source_top, target_top = {}, {}
    for column, dest in (("Src", source_top), ("Tgt", target_top)):
        for iri, group in frame.groupby(column, sort=True):
            best = group.iloc[0]
            gap = float(best.score) - (float(group.iloc[1].score) if len(group) > 1 else 0.0)
            dest[str(iri)] = (str(best.Src), str(best.Tgt), gap)
    return [
        {"Src": str(row.Src), "Tgt": str(row.Tgt), "score": float(row.score), "origin": "predicted"}
        for row in frame.itertuples()
        if row.score >= threshold
        and source_top[str(row.Src)][:2]
        == target_top[str(row.Tgt)][:2]
        == (str(row.Src), str(row.Tgt))
        and min(source_top[str(row.Src)][2], target_top[str(row.Tgt)][2]) >= margin
    ]


def corrupt_anchors(rows, *, fraction, seed, target_kinds):
    """Nested deterministic endpoint replacements; never consult an alignment reference."""
    ordered = sorted(
        rows, key=lambda row: (fingerprint([seed, row["Src"], row["Tgt"]]), row["Src"], row["Tgt"])
    )
    targets = defaultdict(set)
    for row in ordered:
        targets[target_kinds[row["Tgt"]]].add(row["Tgt"])
    count = math.ceil(float(fraction) * len(ordered))
    changed, output = [], []
    for index, row in enumerate(ordered):
        updated = dict(row)
        if index < count:
            options = sorted(targets[target_kinds[row["Tgt"]]] - {row["Tgt"]})
            if not options:
                raise ValueError(
                    "Anchor noise needs at least two distinct targets in each selected kind"
                )
            replacement = options[
                int(fingerprint([seed, row["Src"], row["Tgt"], "replacement"]), 16) % len(options)
            ]
            updated.update(Tgt=replacement, origin="corrupted_" + row.get("origin", "anchor"))
            changed.append(
                {
                    "source": row["Src"],
                    "original_target": row["Tgt"],
                    "replacement_target": replacement,
                }
            )
        output.append(updated)
    return output, changed


class AnchorPreparationMixin:
    def _anchor_base_rows(self, batch_size, directory):
        rows = []
        use_llm = self.model.use_llm
        self.model.use_llm = False
        try:
            for indices in source_batches(self.dataset.dataframe, batch_size):
                identifiers = (
                    self.dataset.dataframe.iloc[indices][["Src", "Tgt"]].astype(str).values.tolist()
                )
                shard = directory / (fingerprint(identifiers) + ".json")
                if shard.exists():
                    rows.extend(json.loads(shard.read_text())["rows"])
                    continue
                batch = _semantic_collate_fn([self.dataset[index] for index in indices])
                with torch.no_grad():
                    out = self.model.forward(
                        src_iris=batch["src_iri"],
                        tgt_iris=batch["tgt_iri"],
                        src_label_lists=batch["src_labels"],
                        tgt_label_lists=batch["tgt_labels"],
                        src_contexts=batch.get("src_contexts"),
                        tgt_contexts=batch.get("tgt_contexts"),
                    )
                scored = [
                    {"Src": str(src), "Tgt": str(tgt), "score": float(out["S_base"][offset])}
                    for offset, (src, tgt) in enumerate(identifiers)
                ]
                freeze_json(shard, {"rows": scored})
                rows.extend(scored)
        finally:
            self.model.use_llm = use_llm
        return rows

    def prepare_anchors(self, *, batch_size):
        config = getattr(self, "anchor_config", {})
        if config.get("mode", "off") == "off" or getattr(self, "_anchors_prepared", False):
            return
        if config.get("mode") != "one_pass":
            raise ValueError("The bounded v2 anchor runner supports one_pass only")
        if not hasattr(self.model, "hier_config"):
            raise ValueError("Anchor rescoring requires the pair-adaptive structural scorer")
        seed = getattr(self.model, "request_seed", None)
        if seed is None:
            raise ValueError("Anchor rescoring requires an explicit seed")
        origin = "exact" if config.get("exact_only") else config.get("source", "predicted")
        rule = {"threshold": config.get("threshold", 0.95), "margin": config.get("margin", 0.1)}
        if origin == "predicted":
            if not config.get("rule_artifact"):
                raise ValueError(
                    "Predicted anchors require an immutable development-selected rule artifact"
                )
            rule = json.loads(Path(config["rule_artifact"]).read_text())
            if (
                rule.get("schema_version") != 1
                or rule.get("mode") != "predicted_anchor_rule"
                or not rule.get("development_provenance")
            ):
                raise ValueError(
                    "Predicted anchor rule needs schema, mode and development provenance"
                )
            if any(not 0 <= float(rule[name]) <= 1 for name in ("threshold", "margin")):
                raise ValueError("Predicted anchor threshold/margin must be finite in [0,1]")
        inventory_identity = {
            "dataset_signature": self.dataset.dataset_signature,
            "config": config,
            "rule": rule,
            "seed": seed,
            "base_scorer": self.model.runtime_fingerprint_payload(),
            "candidate_pairs": self.dataset.dataframe[["Src", "Tgt"]].astype(str).values.tolist(),
        }
        if origin == "trusted":
            if not config.get("trusted_file"):
                raise ValueError("Trusted anchors need an explicit training/trusted file")
            inventory_identity["trusted_sha256"] = fingerprint(
                Path(config["trusted_file"]).read_text()
            )
        directory = (
            self.output_dir / "anchors" / fingerprint(self._json_safe_value(inventory_identity))
        )
        artifact_path = directory / "inventory.json"
        if artifact_path.exists():
            payload = json.loads(artifact_path.read_text())
        else:
            if origin == "predicted":
                rows = predicted_anchors(
                    self._anchor_base_rows(batch_size, directory / "base"),
                    threshold=float(rule["threshold"]),
                    margin=float(rule["margin"]),
                )
            else:
                if origin == "exact":
                    self.dataset.get_exact_matches()
                    frame = self.dataset.exact_matches
                else:
                    frame = read_table(Path(config["trusted_file"])).iloc[:, :2].copy()
                    frame.columns = ["Src", "Tgt"]
                    if set(frame.Src.astype(str)) & set(self.dataset.dataframe.Src.astype(str)):
                        raise ValueError("Training/trusted anchors overlap reporting source groups")
                rows = [
                    {"Src": str(src), "Tgt": str(tgt), "score": 1.0, "origin": origin}
                    for src, tgt in frame[["Src", "Tgt"]]
                    .drop_duplicates()
                    .itertuples(index=False, name=None)
                ]
            valid = []
            for row in rows:
                source_kind = self.dataset.entity_kind_for(row["Src"], "src", warn_unknown=False)
                target_kind = self.dataset.entity_kind_for(row["Tgt"], "tgt", warn_unknown=False)
                if source_kind != target_kind:
                    raise ValueError("Anchor inventory contains a cross-kind pair")
                if row["Src"] not in self.dataset.source.entities(source_kind) or row[
                    "Tgt"
                ] not in self.dataset.target.entities(target_kind):
                    raise ValueError(
                        "Anchor inventory references an entity outside the source signatures"
                    )
                valid.append(row)
            rows, changed = corrupt_anchors(
                valid,
                fraction=config.get("corruption_fraction", 0.0),
                seed=seed,
                target_kinds={
                    row["Tgt"]: self.dataset.entity_kind_for(
                        row["Tgt"], "tgt", warn_unknown=False
                    ).value
                    for row in valid
                },
            )
            payload = {
                "schema_version": 1,
                "mode": "anchor_inventory",
                "origin": origin,
                "rows": rows,
                "seed": seed,
                "requested_corruption_fraction": config.get("corruption_fraction", 0.0),
                "diagnostic": bool(config.get("diagnostic")),
                "corruptions": changed,
                "realized_corruption_fraction": len(changed) / max(1, len(rows)),
                "inventory_sha256": fingerprint(rows),
                "source_ids": sorted({row["Src"] for row in rows}),
                "query_self_support": "excluded",
                "rule": rule,
                "input_identity": fingerprint(self._json_safe_value(inventory_identity)),
            }
            freeze_json(artifact_path, payload)
        source_map, target_map = defaultdict(set), defaultdict(set)
        for row in payload["rows"]:
            source_map[row["Src"]].add(row["Tgt"])
            target_map[row["Tgt"]].add(row["Src"])
        self.model._exact_anchor_src_to_tgt = dict(source_map)
        self.model._exact_anchor_tgt_to_src = dict(target_map)
        self.model.hier_config.update(enabled=True, mode="labels_overlap")
        self.model.hier_enabled = True
        self.model._anchor_manifest = payload
        self.anchor_manifest = {
            "artifact": str(artifact_path),
            **{key: value for key, value in payload.items() if key != "rows"},
        }
        self._anchors_prepared = True
        self._checkpoint_fingerprint_payload = self._build_checkpoint_fingerprint_payload()
        self._checkpoint_fingerprint = self._hash_checkpoint_fingerprint_payload(
            self._checkpoint_fingerprint_payload
        )
