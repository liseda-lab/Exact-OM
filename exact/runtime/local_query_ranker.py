"""Pairwise ranking on original local query contexts.

The production selector supplies feature extraction, RankNet fitting and scoring.
Each query has its own competition; folds still group by the original source.
No acceptance model, NIL class, calibration threshold or global extraction is used.
"""

from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path

import pandas as pd

from exact.core.entities.configs.config import ConfigModel
from exact.impl.models.selector import CandidateSetSelector
from exact.impl.models.selector.feature_contract import apply_feature_contract, feature_contract
from exact.impl.models.selector.fitting import safe_training_labels
from exact.utils.fitted_artifacts import fingerprint, freeze_json
from exact.utils.provenance import sha256_file

RECIPE_VERSION = 1


def _settings(config):
    config = ConfigModel.model_validate(config)
    specs = config.get_model_sequence()[1:]
    if (
        len(specs) != 1
        or (specs[0].name if isinstance(specs[0].name, str) else specs[0].name.__name__)
        != "CandidateSetSelector"
    ):
        raise ValueError("Local query fitting requires the declared CandidateSetSelector")
    if (
        config.selector.rerank.mode != "pairwise"
        or config.selector.rerank.model != "current_linear"
        or config.matching.calibration.mode != "none"
        or config.matching.nil.mode != "off"
    ):
        raise ValueError(
            "Local query adapter supports pairwise/current_linear ranking without calibration or NIL"
        )
    params = dict(specs[0].params)
    if params.get("llm", {}).get("enabled"):
        raise ValueError("Local query ranking never enables secondary hosted decisions")
    params.pop("training_reference_file_path", None)
    params.update(
        enabled=True,
        global_only=False,
        request_seed=config.seed,
        experiment_config=config.selector.model_dump(mode="json"),
    )
    params["experiment_config"]["enabled"] = True
    params["experiment_config"]["rerank"]["artifact"] = None
    return params


def _queries(queries, *, labels):
    rows, qids = [], set()
    for query in queries:
        needed = {"qid", "source", "candidates"} | ({"gold_target"} if labels else set())
        if not needed <= set(query):
            raise ValueError("Local queries require original qid/source/candidates and TRAIN gold")
        qid, source, targets = (query[k] for k in ("qid", "source", "candidates"))
        if (
            type(qid) is not int
            or qid < 0
            or qid in qids
            or not isinstance(source, str)
            or not source
        ):
            raise ValueError("Invalid or duplicate original query identity")
        if (
            not isinstance(targets, list)
            or any(not isinstance(t, str) or not t for t in targets)
            or len(targets) != len(set(targets))
        ):
            raise ValueError("Query candidates must preserve unique target identities")
        if labels and (not isinstance(query["gold_target"], str) or not query["gold_target"]):
            raise ValueError("Only explicit public TRAIN query targets may label local fitting")
        qids.add(qid)
        rows.append({key: query[key] for key in needed})
    if not rows:
        raise ValueError("Original local queries cannot be omitted")
    return rows


def _frame(rows):
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if not {"Src", "Tgt", "S_final"} <= set(frame) or frame.duplicated(["Src", "Tgt"]).any():
        raise ValueError("Local primary scores require unique source-target rows")
    if "S_pair_final" not in frame:
        frame["S_pair_final"] = frame["S_final"]
    for field in ("S_final", "S_pair_final"):
        if not all(
            math.isfinite(float(value)) and 0 <= float(value) <= 1 for value in frame[field]
        ):
            raise ValueError("Local primary scores must be finite probabilities")
    return frame.reset_index(drop=True)


def _features(selector, frame, queries, *, records=(), labels=None, logger=None):
    """Build each query independently; no qid, source name or label is a feature."""
    lookup = {(str(row.Src), str(row.Tgt)): index for index, row in frame.iterrows()}
    evidence = {(str(r["src_iri"]), str(r["tgt_iri"])): r for r in records}
    for row in frame.itertuples():
        key = str(row.Src), str(row.Tgt)
        if key not in evidence:
            if not hasattr(row, "selector_evidence_items"):
                raise ValueError(
                    "Native compact evidence must be available consistently for TRAIN and application"
                )
            evidence[key] = dict(
                resident_evidence_schema=1, selector_evidence_items=row.selector_evidence_items
            )
    features, groups, identities, offset = {}, [], {}, 0
    log = logger or logging.getLogger(__name__)
    for query in queries:
        pairs = [(query["source"], target) for target in query["candidates"]]
        if any(pair not in lookup for pair in pairs):
            raise ValueError("A local query candidate has no materialized native primary features")
        group = frame.loc[[lookup[pair] for pair in pairs]].copy()
        group.index = range(offset, offset + len(group))
        offset += len(group)
        distinctive = selector._distinctive_scores(group, evidence, logger=log)
        features.update(
            selector._rank_feature_rows(group, distinctive, selector._reciprocity_scores(group))
        )
        indices = list(group.index)
        for index, pair in zip(indices, pairs, strict=True):
            identities[index] = dict(qid=query["qid"], source=pair[0], target=pair[1])
        if labels is not None:
            known = [index for index, pair in zip(indices, pairs, strict=True) if pair in labels]
            positive = [
                index for index, pair in zip(indices, pairs, strict=True) if labels.get(pair) == 1
            ]
            if positive and len(known) > len(positive):
                groups.append(
                    dict(
                        qid=query["qid"],
                        source=query["source"],
                        indices=known,
                        positive_indices=positive,
                    )
                )
    return features, groups, identities


def _rank(selector, model, features, identities, queries):
    by_qid = {query["qid"]: [] for query in queries}
    for index, row in features.items():
        item = identities[index]
        by_qid[item["qid"]].append(
            dict(target=item["target"], utility=selector._score_rank_model(row, model))
        )
    output = []
    for query in queries:
        items = sorted(by_qid[query["qid"]], key=lambda item: (-item["utility"], item["target"]))
        probabilities = (
            selector._softmax([item["utility"] for item in items], temperature=selector.temperature)
            if items
            else []
        )
        output.append(
            dict(
                qid=query["qid"],
                source=query["source"],
                candidates=[
                    {**item, "score": float(score)}
                    for item, score in zip(items, probabilities, strict=True)
                ],
            )
        )
    return output


def fit(
    training_frame,
    original_train_queries,
    path,
    *,
    selector_config,
    application,
    fold_membership=None,
    input_views=None,
    logger=None,
):
    """Fit explicit-label query groups, with OOF membership keyed by original source."""
    if (
        application.get("split_role") != "train"
        or application.get("negative_label_policy") != "confirmed_negatives"
    ):
        raise ValueError(
            "Local rank fitting requires designated TRAIN and explicit benchmark-scoped candidate labels"
        )
    if input_views is not None:
        training_frame = input_views.full()
        application = {**application, "consumer_inputs": input_views.manifest}
    queries = _queries(original_train_queries, labels=True)
    sources = {query["source"] for query in queries}
    if sources & set(application.get("source_ids", [])) or len(sources) < 2:
        raise ValueError(
            "OOF TRAIN source groups must be disjoint from reporting and number at least two"
        )
    settings, frame = _settings(selector_config), _frame(training_frame)
    if set(frame.Src.astype(str)) != sources:
        raise ValueError("Native TRAIN source membership differs from the complete query support")
    safe, _ = safe_training_labels(
        frame, {(q["source"], q["gold_target"]) for q in queries}, application
    )
    labels = {(str(row.Src), str(row.Tgt)): int(row.confirmed_label) for row in safe.itertuples()}
    selector = CandidateSetSelector(**settings)
    contract = feature_contract(frame)
    frame = apply_feature_contract(frame, contract)
    features, groups, identities = _features(selector, frame, queries, labels=labels, logger=logger)
    if not groups:
        raise ValueError("No query has both an explicitly positive and negative TRAIN candidate")
    if fold_membership is None:
        shuffled = sorted(sources)
        random.Random(selector.request_seed).shuffle(shuffled)
        count = min(len(sources), max(2, int(selector.calibration["validation_folds"])))
        folds = {source: i % count for i, source in enumerate(shuffled)}
    else:
        folds = dict(fold_membership)
        if (
            set(folds) != sources
            or any(type(v) is not int or v < 0 for v in folds.values())
            or len(set(folds.values())) < 2
        ):
            raise ValueError(
                "Fold membership must bind every original TRAIN source to at least two folds"
            )
    if input_views is not None and input_views.membership != folds:
        raise ValueError("Original-query fold input membership changed")
    target_sources = {}
    for (source, target), label in labels.items():
        if label == 1:
            target_sources.setdefault(target, set()).add(source)
    crossing = {
        target: sorted(members)
        for target, members in target_sources.items()
        if len({folds[source] for source in members}) > 1
    }
    provenance = dict(
        recipe_version=RECIPE_VERSION,
        feature_contract=contract,
        selector_settings=settings,
        implementation=selector._implementation_fingerprint(),
        adapter_sha256=sha256_file(Path(__file__)),
        rank_features=selector.RANK_FEATURE_NAMES,
        training_sources=sorted(sources),
        original_queries=queries,
        folds=folds,
        reference_overlap_audit=dict(
            positive_target_count=len(target_sources),
            cross_fold_positive_targets=len(crossing),
            affected_source_count=len(
                {source for members in crossing.values() for source in members}
            ),
            crossing_membership_sha256=fingerprint(crossing),
            independent_reference_components=not crossing,
            fold_unit="source_group",
        ),
        features_sha256=fingerprint([[identities[i], row] for i, row in features.items()]),
        labels_sha256=fingerprint(sorted((s, t, label) for (s, t), label in labels.items())),
        application=application,
        context="original_query_only",
        target_reciprocity="within_original_query",
        acceptance=False,
        nil=False,
        anchor_policy="native_hard_anchor_bypass_no_feature_fabrication",
    )
    identity, path = fingerprint(provenance), Path(path)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("fit_identity") != identity or saved.get("sha256") != fingerprint(
            {k: v for k, v in saved.items() if k != "sha256"}
        ):
            raise ValueError("Local query head belongs to different inputs, folds or recipe")
        return saved
    checkpoints = []
    full_features, full_groups, full_identities = features, groups, identities
    for fold in sorted(set(folds.values())):
        heldout = {source for source, value in folds.items() if value == fold}
        if input_views is not None:
            scoped = _frame(
                input_views.fold(fold, train_sources=sources - heldout, heldout_sources=heldout)
            )
            if set(zip(scoped.Src.astype(str), scoped.Tgt.astype(str))) != set(
                zip(frame.Src.astype(str), frame.Tgt.astype(str))
            ):
                raise ValueError("Scoped local inputs changed complete original candidate pairs")
            scoped = apply_feature_contract(scoped, contract)
            features, groups, identities = _features(
                selector, scoped, queries, labels=labels, logger=logger
            )
        checkpoint = path.parent / (path.name + ".folds") / f"{identity}-{fold}.json"
        if checkpoint.exists():
            record = json.loads(checkpoint.read_text())
            if (
                record.get("fit_identity") != identity
                or record.get("heldout_sources") != sorted(heldout)
                or record.get("sha256")
                != fingerprint({k: v for k, v in record.items() if k != "sha256"})
            ):
                raise ValueError("Local OOF checkpoint identity changed")
        else:
            train_groups = [group for group in groups if group["source"] not in heldout]
            model = selector._fit_rank_model(
                features, train_groups, logger=logger or logging.getLogger(__name__)
            )
            if model is None:
                raise ValueError("Local OOF fold lacks explicit positive/negative training queries")
            held_features = {
                i: row for i, row in features.items() if identities[i]["source"] in heldout
            }
            ranked = _rank(
                selector,
                model,
                held_features,
                identities,
                [q for q in queries if q["source"] in heldout],
            )
            record = dict(
                fit_identity=identity,
                fold=fold,
                train_sources=sorted(sources - heldout),
                heldout_sources=sorted(heldout),
                query_predictions=ranked,
            )
            record["sha256"] = fingerprint(record)
            freeze_json(checkpoint, record)
        checkpoints.append(record)
    features, groups, identities = full_features, full_groups, full_identities
    model = selector._fit_rank_model(features, groups, logger=logger or logging.getLogger(__name__))
    if model is None:
        raise ValueError("Local rank fit produced no model")
    artifact = dict(
        schema_version=1,
        kind="local_query_pairwise_ranker",
        fit_identity=identity,
        fit_provenance=provenance,
        rank_model=model,
        folds=checkpoints,
        acceptance=False,
        local_complete_ranking=True,
    )
    artifact["sha256"] = fingerprint(artifact)
    freeze_json(path, artifact)
    return artifact


def apply(
    artifact_or_path,
    primary_rows,
    original_valid_queries,
    *,
    records=(),
    anchors=(),
    application=None,
):
    """Return every candidate per original query; reporting labels are never read."""
    artifact = (
        json.loads(Path(artifact_or_path).read_text())
        if isinstance(artifact_or_path, (str, Path))
        else artifact_or_path
    )
    if (
        artifact.get("kind") != "local_query_pairwise_ranker"
        or artifact.get("fit_identity") != fingerprint(artifact.get("fit_provenance"))
        or artifact.get("sha256")
        != fingerprint({k: v for k, v in artifact.items() if k != "sha256"})
    ):
        raise ValueError("Invalid local query rank artifact provenance")
    provenance = artifact["fit_provenance"]
    if (
        provenance.get("recipe_version") != RECIPE_VERSION
        or artifact.get("acceptance") is not False
    ):
        raise ValueError("Unsupported local query ranking recipe")
    queries = _queries(original_valid_queries, labels=False)
    if {q["source"] for q in queries} & set(provenance["training_sources"]):
        raise ValueError("Reporting source groups overlap local ranker TRAIN sources")
    if application is not None and application != provenance["application"]:
        raise ValueError("Local ranker application binding changed")
    selector = CandidateSetSelector(**provenance["selector_settings"])
    if selector._implementation_fingerprint() != provenance["implementation"] or provenance.get(
        "adapter_sha256"
    ) != sha256_file(Path(__file__)):
        raise ValueError("Local ranker implementation changed; refit required")
    protected = {}
    for anchor in anchors:
        pair = str(anchor["Src"]), str(anchor["Tgt"])
        if pair in protected or float(anchor["S_final"]) != 1.0:
            raise ValueError("Hard anchors require unique explicit native score-one pairs")
        protected[pair] = 1.0
    population = {(q["source"], t) for q in queries for t in q["candidates"]}
    if set(protected) - population:
        raise ValueError("Hard anchor lies outside the original query population")
    scoring_queries = [
        {**q, "candidates": [t for t in q["candidates"] if (q["source"], t) not in protected]}
        for q in queries
    ]
    if any(q["candidates"] for q in scoring_queries):
        frame = _frame(primary_rows)
        frame = frame.loc[
            [(str(row.Src), str(row.Tgt)) not in protected for row in frame.itertuples()]
        ].copy()
        frame = apply_feature_contract(frame, provenance["feature_contract"])
        features, _, identities = _features(selector, frame, scoring_queries, records=records)
    else:
        features, identities = {}, {}
    predictions = _rank(selector, artifact["rank_model"], features, identities, scoring_queries)
    for query, prediction in zip(queries, predictions, strict=True):
        prediction["candidates"].extend(
            dict(target=t, score=1.0, utility=None, protected_exact=True)
            for t in query["candidates"]
            if (query["source"], t) in protected
        )
        prediction["fixed_anchor_count"] = sum(
            bool(row.get("protected_exact")) for row in prediction["candidates"]
        )
        prediction["candidates"].sort(key=lambda row: (-row["score"], row["target"]))
    return dict(
        schema_version=1,
        kind="local_query_rankings",
        fit_identity=artifact["fit_identity"],
        queries=predictions,
        unchanged_anchor_pairs=len(protected),
        candidate_occurrences=sum(len(q["candidates"]) for q in queries),
        original_query_count=len(queries),
        source_count=len({q["source"] for q in queries}),
        complete=True,
        acceptance=False,
    )
