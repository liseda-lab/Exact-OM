"""One shared encoder pass over a separate training pool, then small CPU head fits."""

from __future__ import annotations

import copy
import json
from ast import literal_eval
from pathlib import Path

import pandas as pd
import torch

from exact.impl.models.selector.fitting import (
    fingerprint,
    freeze_json,
    safe_training_labels,
)
from exact.utils.data import read_table

from .audit_io import _semantic_collate_fn


def source_batches(frame, batch_size):
    """Pack complete contiguous source groups without changing source semantics."""
    batch = []
    columns = [column for column in ("Src", "SrcKind") if column in frame]
    for _, group in frame.groupby(columns, sort=False):
        indices = list(group.index)
        if batch and len(batch) + len(indices) > batch_size:
            yield batch
            batch = []
        batch.extend(indices)
    if batch:
        yield batch


class TrainingPoolMixin:
    def fit_relation_head(self):
        config = getattr(self, "relation_config", {})
        if config.get("relation_prediction") not in {"learned_three_way", "semantic_then_learned"}:
            return
        if getattr(self, "_relation_head_fitted", False):
            return
        from exact.io.relation_head import fit_relation_artifact, typed_reference_frame

        artifact = config.get("relation_artifact")
        training_file = config.get("relation_training_file")
        if training_file:
            typed = typed_reference_frame(training_file)
            artifact = artifact or self.output_dir / "fitting" / "relation_head.json"
            state = fit_relation_artifact(
                typed[["Src", "Tgt"]],
                typed,
                self.dataset.source,
                self.dataset.target,
                artifact,
                application={
                    "dataset_signature": self.dataset.dataset_signature,
                    "source_ids": sorted(
                        set(
                            getattr(self.dataset, "eligible_source_iris", None)
                            or self.dataset.dataframe.Src.astype(str)
                        )
                    ),
                },
                seed=getattr(self.model, "request_seed", None) or 17,
            )
            self.relation_fit_report = {
                "artifact": str(artifact),
                "fit_identity": state["fit_identity"],
                "relation_counts": state["relation_counts"],
                "folds": state["folds"],
            }
        elif not artifact or not Path(artifact).is_file():
            raise ValueError(
                "Learned relation typing requires a fitted artifact or disjoint typed training file"
            )
        self.relation_artifact = artifact
        self._relation_head_fitted = True

    def fit_training_pool(self, *, batch_size=8):
        if getattr(self.dataset, "transfer_artifact", None) is not None:
            from exact.utils.artifact_transfer import validate_transfer

            manifest = validate_transfer(self.dataset)
            if any(
                getattr(model, "nil_config", {}).get("training_source_labels")
                for model in getattr(self, "models", [])[1:]
            ):
                raise ValueError("Transferred NIL cannot consume recipient training source labels")
            if any(
                getattr(self, key, None)
                for key in ("fitting_fusion_config", "fitting_graph_config", "fitting_llm_config")
            ):
                raise ValueError("Donor transfer cannot fit missing recipient artifacts")
            if getattr(self, "relation_config", {}).get("relation_training_file"):
                raise ValueError("Donor transfer cannot fit recipient relation labels")
            self.transfer_report = {"recipient_refit": False, "artifacts": manifest["artifacts"]}
            self._training_pool_fitted = True
            return
        self.fit_relation_head()
        if getattr(self, "_training_pool_fitted", False):
            return
        path = getattr(self, "training_candidates_file_path", None)
        if not path:
            return
        consumers = [
            model
            for model in self.models[1:]
            if hasattr(model, "fit_training_artifact") and model.training_reference_file_path
        ]
        nil_consumers = [
            model
            for model in self.models[1:]
            if getattr(model, "nil_config", {}).get("mode") == "fitted"
            and model.nil_config.get("training_source_labels")
        ]
        pending = any(
            getattr(self, key, None)
            for key in ("fitting_fusion_config", "fitting_graph_config", "fitting_llm_config")
        )
        if not consumers and not pending and not nil_consumers:
            return
        seed = getattr(self.model, "request_seed", consumers[0].request_seed if consumers else 17)
        supervision = getattr(self, "supervision_config", {})
        policy = supervision.get("negative_label_policy", "unknown")
        if (consumers or pending) and policy not in {"complete_reference", "confirmed_negatives"}:
            raise ValueError(
                "Supervised fitting lacks a safe negative-label policy: a known positive does not establish other candidate negatives"
            )
        raw = read_table(Path(path))
        if len(raw.columns) == 3 and any("cand" in str(column).lower() for column in raw.columns):
            # Bio-ML's middle column is reporting gold, never a candidate feature.
            rows = [
                (str(source), str(target))
                for source, _, candidates in raw.itertuples(index=False, name=None)
                for target in literal_eval(str(candidates))
            ]
            raw = pd.DataFrame(rows, columns=["Src", "Tgt"])
        else:
            raw = raw[
                [
                    column
                    for column in (
                        "Src",
                        "Tgt",
                        "SrcKind",
                        "TgtKind",
                        "cand_sim",
                        "confirmed_label",
                    )
                    if column in raw
                ]
            ].copy()
        if not {"Src", "Tgt"}.issubset(raw):
            raise ValueError(
                "Training candidate pool must provide Src/Tgt or a Bio-ML candidate list"
            )
        raw = raw.drop_duplicates(["Src", "Tgt"]).sort_values(["Src", "Tgt"]).reset_index(drop=True)
        reporting_sources = set(
            getattr(self.dataset, "eligible_source_iris", None)
            or self.dataset.dataframe.Src.astype(str)
        )
        if set(raw.Src.astype(str)) & reporting_sources:
            raise ValueError("Training candidate pool overlaps reporting source groups")
        application = {
            "dataset_signature": getattr(self.dataset, "dataset_signature", None),
            "source_ids": sorted(reporting_sources),
            "negative_label_policy": policy,
            "entity_kinds": [
                getattr(kind, "value", kind)
                for kind in getattr(self.dataset, "_entity_kinds", ("class",))
            ],
        }
        reference_path = getattr(self, "training_reference_file_path", None)
        if reference_path:
            table = read_table(Path(reference_path))
            reference = {
                (str(source), str(target))
                for source, target in table.iloc[:, :2].itertuples(index=False, name=None)
            }
        elif consumers:
            reference = consumers[0]._load_training_reference_pairs(getattr(self, "logger", None))
        else:
            raise ValueError("Train-only fitting requires an explicit training reference")
        router_requested = (getattr(self, "fitting_llm_config", None) or {}).get("gate", {}).get(
            "mode"
        ) == "learned"
        original_counts = raw.groupby("Src").size() if router_requested else None
        if not nil_consumers:
            raw, reference = safe_training_labels(raw, reference, application)
        if router_requested:
            labeled = safe_training_labels(raw, reference, application)[0] if nil_consumers else raw
            labeled_counts = labeled.groupby("Src").size()
            application["fully_labeled_training_sources"] = sorted(
                str(source)
                for source, count in original_counts.items()
                if int(labeled_counts.get(source, 0)) == int(count)
            )
        identity = fingerprint(
            self._json_safe_value(
                {
                    "pairs": raw.to_dict("records"),
                    "model": (
                        self.model.runtime_fingerprint_payload()
                        if hasattr(self.model, "runtime_fingerprint_payload")
                        else type(self.model).__name__
                    ),
                }
            )
        )
        cache_dir = self.output_dir / "fitting" / identity
        frame_path = cache_dir / "training_scores.json"
        if frame_path.exists():
            training = pd.DataFrame(json.loads(frame_path.read_text())["rows"])
        else:
            view = copy.copy(self.dataset)
            view._candidates = raw.copy()
            if hasattr(view, "_annotate_candidate_similarity_stats"):
                view._annotate_candidate_similarity_stats()
            view._df = view.get_features(view._candidates)
            view._df[view.default_kind] = True
            if hasattr(view, "_invalidate_active_dataframe_cache"):
                view._invalidate_active_dataframe_cache()
            records = []
            # Train-only head fitting never requests decision/brief/rationale calls.
            original_llm = getattr(self.model, "use_llm", False)
            self.model.use_llm = False
            try:
                for indices in source_batches(view._df, int(batch_size)):
                    source_ids = list(view._df.iloc[indices].Src.astype(str).unique())
                    shard = cache_dir / (fingerprint(source_ids) + ".json")
                    if shard.exists():
                        records.extend(json.loads(shard.read_text())["rows"])
                        continue
                    batch = _semantic_collate_fn([view[index] for index in indices])
                    with torch.no_grad():
                        output = self.model.forward(
                            src_iris=batch["src_iri"],
                            tgt_iris=batch["tgt_iri"],
                            src_label_lists=batch["src_labels"],
                            tgt_label_lists=batch["tgt_labels"],
                            src_contexts=batch.get("src_contexts"),
                            tgt_contexts=batch.get("tgt_contexts"),
                        )
                    rows = []
                    for offset, index in enumerate(indices):
                        row = {
                            "Src": str(batch["src_iri"][offset]),
                            "Tgt": str(batch["tgt_iri"][offset]),
                            "SrcKind": str(batch.get("src_kind", ["class"] * len(indices))[offset]),
                            "TgtKind": str(batch.get("tgt_kind", ["class"] * len(indices))[offset]),
                        }
                        for key, values in output.items():
                            if (
                                isinstance(values, torch.Tensor)
                                and values.ndim == 1
                                and len(values) == len(indices)
                            ):
                                row[key] = float(values[offset].detach().cpu())
                        row["src_label_text"] = " | ".join(batch["src_labels"][offset])
                        row["tgt_label_text"] = " | ".join(batch["tgt_labels"][offset])
                        row["llm_evidence_packet"] = output.get(
                            "llm_evidence_packets", [""] * len(indices)
                        )[offset]
                        if output.get("graph_features"):
                            row["graph_features"] = output["graph_features"][offset]
                        if "confirmed_label" in raw:
                            label = raw.iloc[index].confirmed_label
                            row["confirmed_label"] = None if pd.isna(label) else float(label)
                        row["fusion_channels"] = {
                            name: {
                                field: float(values[offset].detach().cpu())
                                for field, values in channel.items()
                            }
                            for name, channel in output.get("fusion_channels", {}).items()
                        }
                        explanations = output.get("explanations", [])
                        if offset < len(explanations):
                            row["selector_evidence_items"] = (
                                self._selector_evidence_items_for_record(explanations[offset])
                                if hasattr(self, "_selector_evidence_items_for_record")
                                else []
                            )
                        row.update(
                            {
                                column: value
                                for column, value in view._df.iloc[index].items()
                                if str(column).startswith("cand_")
                            }
                        )
                        rows.append(row)
                    freeze_json(shard, {"source_ids": source_ids, "rows": rows})
                    records.extend(rows)
            finally:
                self.model.use_llm = original_llm
            freeze_json(frame_path, {"identity": identity, "rows": records})
            training = pd.DataFrame(records)
        from exact.impl.models.selector.label_budget import select_label_budget

        budget_reference, membership = select_label_budget(
            training,
            reference,
            budget=supervision.get("label_budget"),
            seed=seed,
            selection=supervision.get("label_selection", "passive"),
        )
        if supervision.get("label_budget") is not None:
            cache_dir = cache_dir / ("labels-" + fingerprint(membership))
        freeze_json(cache_dir / "label_budget.json", membership)
        self._training_effective_units = {**membership, "binding": application}
        freeze_json(cache_dir / "training_units.json", self._training_effective_units)
        graph_config = getattr(self, "fitting_graph_config", None)
        if graph_config:
            from exact.impl.models.graph_head import fit_graph_artifact
            from exact.impl.models.pair_adaptive_experiments import (
                JsonExperimentArtifact,
            )

            graph_application = {
                **application,
                "negative_label_policy": graph_config.get("negative_label_policy") or policy,
            }
            artifact = graph_config.get("artifact") or cache_dir / "graph.json"
            payload = fit_graph_artifact(
                training, budget_reference, artifact, application=graph_application, seed=seed
            )
            self.model.graph_config = {**graph_config, "artifact": str(artifact)}
            self.model._graph_artifact = JsonExperimentArtifact.load(
                artifact, expected_mode="inductive", kind="graph head"
            )
            oof = {(row["Src"], row["Tgt"]): row["score"] for row in payload["oof_predictions"]}
            training = training[
                training.Src.astype(str).isin({source for source, _ in oof})
            ].reset_index(drop=True)
            for index, row in training.iterrows():
                channels = dict(row.fusion_channels)
                active = float(row.graph_features["active"])
                channels["graph"] = {
                    "score": oof[(row.Src, row.Tgt)],
                    "quality": active,
                    "active": active,
                }
                training.at[index, "fusion_channels"] = channels
            self._replay_training_fusion(training)
        fusion_config = getattr(self, "fitting_fusion_config", None)
        if fusion_config:
            from exact.impl.models.pair_adaptive_experiments import (
                JsonExperimentArtifact,
            )
            from exact.impl.models.selector.fusion_fitting import fit_fusion_artifact

            artifact = fusion_config.get("artifact") or cache_dir / "fusion.json"
            placement = (
                self.model.strsim_config["placement"] if self.model.strsim_enabled else "off"
            )
            payload = fit_fusion_artifact(
                training,
                budget_reference,
                artifact,
                mode=fusion_config["mode"],
                application=application,
                seed=seed,
                strsim_placement=placement,
            )
            self.model.fusion_config = {**fusion_config, "artifact": str(artifact)}
            self.model._fusion_artifact = JsonExperimentArtifact.load(
                artifact, expected_mode=fusion_config["mode"], kind="fusion"
            )
            self.model._validate_fusion_artifact_contract(fusion_config["mode"])
            if fusion_config["mode"] == "analytic_fitted":
                self.model.tau = payload["parameters"]["tau"]
                self.model.gamma = payload["parameters"]["gamma"]
            oof = {(row["Src"], row["Tgt"]): row for row in payload["oof_predictions"]}
            for index, row in training.iterrows():
                if (row.Src, row.Tgt) in oof:
                    for name, value in oof[(row.Src, row.Tgt)].items():
                        if name not in {"Src", "Tgt"}:
                            training.at[index, name] = value
            self._refresh_training_uncertainty(training)
        llm_config = getattr(self, "fitting_llm_config", None)
        if llm_config:
            from exact.impl.models.pair_adaptive_experiments import (
                JsonExperimentArtifact,
            )
            from exact.impl.models.selector.llm_learning import fit_llm_artifacts

            if llm_config["gate"]["mode"] == "learned" and any(
                selector.enabled for selector in consumers
            ):
                raise ValueError(
                    "Benefit-router fitting currently requires fixed pair-threshold acceptance without an additional selector; full-selector counterfactual outcomes must be supplied separately"
                )
            fitting_frame = (
                training[training.Src.astype(str).isin({source for source, _ in budget_reference})]
                .copy()
                .reset_index(drop=True)
            )
            learned = fit_llm_artifacts(
                self.model,
                fitting_frame,
                budget_reference,
                cache_dir,
                config=llm_config,
                application=application,
            )
            for key in ("exemplars", "exemplar_count", "distill", "student_training"):
                self.model.llm_experiment_config[key] = llm_config[key]
            for key in ("exemplar_artifact", "distill_artifact"):
                if key in learned:
                    self.model.llm_experiment_config[key] = learned[key]
            self.model._exemplar_artifact = learned.get(
                "exemplars", getattr(self.model, "_exemplar_artifact", None)
            )
            self.model._student_artifact = learned.get(
                "student", getattr(self.model, "_student_artifact", None)
            )
            if "router" in learned:
                self.model.llm_experiment_config["gate"] = {
                    **llm_config["gate"],
                    "artifact": learned["gate_artifact"],
                }
                self.model._gate_artifact = JsonExperimentArtifact.load(
                    learned["gate_artifact"], expected_mode="learned", kind="LLM gate"
                )
            if "student" in learned:
                oof = {
                    (row["source"], str(fitting_frame.iloc[row["index"]].Tgt)): row["score"]
                    for row in learned["student"]["oof_predictions"]
                }
                for index, row in training.iterrows():
                    if (row.Src, row.Tgt) in oof:
                        weight = (
                            llm_config["constant_weight"]
                            if llm_config["fusion_weight"] == "constant"
                            else min(1.0, self.model.beta * row.U)
                        )
                        training.at[index, "S_final"] = (1 - weight) * row.S_base + weight * oof[
                            (row.Src, row.Tgt)
                        ]
        for selector in nil_consumers:
            from exact.impl.models.selector.nil_head import fit_nil_artifact

            labels = read_table(Path(selector.nil_config["training_source_labels"]))
            artifact = selector.nil_config.get("artifact") or cache_dir / "natural_nil.json"
            fit_nil_artifact(
                training,
                labels,
                reference,
                artifact,
                application={
                    **application,
                    "nil_label_semantics": selector.nil_config.get("label_semantics"),
                },
                seed=seed,
            )
            selector.nil_config["artifact"] = str(artifact)
        for selector in consumers:
            reference = selector._load_training_reference_pairs(getattr(self, "logger", None))
            if not reference:
                raise ValueError("Separate training candidates require a training reference")
            reference = reference & budget_reference
            selector_training = training.copy()
            if selector.matching_calibration["mode"] != "none":
                fitted_calibrator = selector.fit_score_calibration_artifact(
                    training,
                    reference,
                    selector.matching_calibration["artifact"],
                    application=application,
                )
                oof = {
                    (row["source"], row["target"]): row["probability"]
                    for row in fitted_calibrator["oof_predictions"]
                }
                selector_training = selector_training[
                    selector_training.apply(lambda row: (row.Src, row.Tgt) in oof, axis=1)
                ].copy()
                selector_training["S_final"] = [
                    oof[(row.Src, row.Tgt)] for row in selector_training.itertuples()
                ]
                selector_training["S_pair_final"] = selector_training.S_final
            if selector.strategy == "calibrated_rank_accept" or selector.rerank_config.get(
                "artifact"
            ):
                artifact = selector.rerank_config.get("artifact") or cache_dir / "selector.json"
                selector.fit_training_artifact(
                    selector_training,
                    reference,
                    artifact,
                    application=application,
                    logger=getattr(self, "logger", None),
                )
        self._checkpoint_fingerprint_payload = self._build_checkpoint_fingerprint_payload()
        self._checkpoint_fingerprint = self._hash_checkpoint_fingerprint_payload(
            self._checkpoint_fingerprint_payload
        )
        self._training_pool_fitted = True

    def _replay_training_fusion(self, training):
        from exact.impl.models.selector.fusion_fitting import fusion_scores

        names = sorted(training.iloc[0].fusion_channels) + ["lex", "struct"]
        channels = tuple(
            torch.tensor(
                [
                    [row.get(name, {}).get(field, 0.0) for name in names]
                    for row in training.fusion_channels
                ],
                dtype=torch.float64,
            )
            for field in ("score", "quality", "active")
        )
        placement = self.model.strsim_config["placement"] if self.model.strsim_enabled else "off"
        values = fusion_scores(
            channels,
            names,
            (torch.tensor(self.model.tau), torch.tensor(self.model.gamma), torch.ones(len(names))),
            mode="analytic_fitted",
            strsim_placement=placement,
            return_components=True,
        )
        for name, column in values.items():
            training[name] = column.tolist()
        self._refresh_training_uncertainty(training)

    def _refresh_training_uncertainty(self, training):
        if hasattr(self.model, "_uncertainty_components"):
            columns = [
                torch.tensor(training[name].tolist(), dtype=torch.float64)
                for name in ("S_base", "s_label_star", "S_struct", "q_lex", "Q_struct")
            ]
            u_ind, u_dis = self.model._uncertainty_components(*columns)
            training["U"] = torch.maximum(u_ind, u_dis).tolist()

    def prepare_population_gate(self, *, batch_size=8):
        """Freeze exact-budget selection from an unlabeled, decision-off first pass."""
        config = getattr(self, "fitting_gate_config", None)
        if not config:
            return
        import time

        from exact.impl.models.pair_adaptive_experiments import JsonExperimentArtifact
        from exact.impl.models.selector.llm_gate import select_inference_gate_artifact

        started = time.perf_counter()
        frame = self.dataset._active_dataframe().reset_index(drop=True)
        signature = getattr(self.dataset, "dataset_signature", None)
        identity = fingerprint(
            self._json_safe_value(
                {
                    "dataset": signature,
                    "pairs": frame[["Src", "Tgt"]].values.tolist(),
                    "model": self.model.runtime_fingerprint_payload(),
                    "policy": config,
                }
            )
        )
        directory = self.output_dir / "routing" / identity
        records = []
        original_llm = self.model.use_llm
        self.model.use_llm = False
        try:
            for indices in source_batches(frame, int(batch_size)):
                source_ids = sorted(set(frame.iloc[indices].Src.astype(str)))
                shard = directory / (fingerprint(source_ids) + ".json")
                if shard.exists():
                    records.extend(json.loads(shard.read_text())["rows"])
                    continue
                batch = _semantic_collate_fn([self.dataset[index] for index in indices])
                with torch.no_grad():
                    output = self.model.forward(
                        src_iris=batch["src_iri"],
                        tgt_iris=batch["tgt_iri"],
                        src_label_lists=batch["src_labels"],
                        tgt_label_lists=batch["tgt_labels"],
                        src_contexts=batch.get("src_contexts"),
                        tgt_contexts=batch.get("tgt_contexts"),
                        src_ctx_raw=batch.get("src_ctx_raw_triples"),
                        tgt_ctx_raw=batch.get("tgt_ctx_raw_triples"),
                    )
                rows = [
                    {
                        "source_iri": str(source),
                        "target_iri": str(target),
                        "score": float(output["S_base"][offset]),
                        "U": float(output["U"][offset]),
                    }
                    for offset, (source, target) in enumerate(
                        zip(batch["src_iri"], batch["tgt_iri"])
                    )
                ]
                freeze_json(shard, {"source_ids": source_ids, "rows": rows})
                records.extend(rows)
        finally:
            self.model.use_llm = original_llm
        universe = (
            getattr(self.dataset, "eligible_source_iris", None)
            or getattr(self, "source_universe", None)
            or sorted(set(self.dataset.dataframe.Src.astype(str)))
        )
        artifact = config.get("artifact") or directory / "selection.json"
        if config["mode"] == "forced_sample":
            from exact.impl.models.selector.llm_gate import fit_forced_sample_artifact

            strata = (
                json.loads(Path(config["strata_artifact"]).read_text())
                if config.get("strata_artifact")
                else None
            )
            if strata is not None and strata.get("dataset_signature") != signature:
                raise ValueError("Forced-source strata differ from the bound development dataset")
            payload = fit_forced_sample_artifact(
                records,
                source_strata=strata["source_strata"] if strata else None,
                strata_provenance=strata["provenance"] if strata else None,
                sample_size=config["forced_sample_size"],
                seed=self.model.request_seed,
                task_id=str(signature),
                entity_kind="|".join(
                    sorted({str(kind) for kind in getattr(self.dataset, "entity_kinds", ["class"])})
                ),
                dataset_signature=signature,
            )
            payload["selected_count"] = payload["sample_size"]
            payload["no_candidate_sources"] = sorted(
                set(universe) - {row["source_iri"] for row in records}
            )
        else:
            payload = select_inference_gate_artifact(
                records,
                mode=config["mode"],
                fraction=config["quantile_fraction"],
                dataset_signature=signature,
                source_universe=universe,
            )
        freeze_json(artifact, payload)
        self.model.llm_experiment_config["gate"] = {**config, "artifact": str(artifact)}
        self.model._gate_artifact = JsonExperimentArtifact.load(
            artifact, expected_mode=config["mode"], kind="LLM gate"
        )
        self.model._validate_gate_artifact_contract(config["mode"])
        self._gate_prepass = {
            "seconds": time.perf_counter() - started,
            "rows": len(records),
            "selection_artifact": str(artifact),
            "eligible_sources": len(set(row["source_iri"] for row in records)),
            "selected_count": payload["selected_count"],
            "no_candidate_sources": payload["no_candidate_sources"],
            "evidence_reuse": "shared_persistent_encoder_cache",
        }
        freeze_json(
            directory / "prepass.json",
            {key: value for key, value in self._gate_prepass.items() if key != "seconds"},
        )
        self.fitting_gate_config = None
        self._checkpoint_fingerprint_payload = self._build_checkpoint_fingerprint_payload()
        self._checkpoint_fingerprint = self._hash_checkpoint_fingerprint_payload(
            self._checkpoint_fingerprint_payload
        )
