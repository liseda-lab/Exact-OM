"""Composed source-fold dependencies and authenticated native consumer inputs.

A node's influence includes model selection, not only optimizer examples. This
module prepares dependencies; it does not admit a request bank or certify a
native worker merely because its dependency graph was compiled.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

from exact.utils.fitted_artifacts import fingerprint
from exact.utils.provenance import sha256_file


def membership(sources, *, seed, folds):
    sources = sorted(set(sources))
    count = min(len(sources), max(2, int(folds)))
    if count < 2:
        raise ValueError("At least two source groups required")
    random.Random(seed).shuffle(sources)
    return {source: index % count for index, source in enumerate(sources)}


def compile_chain(sources, valid_sources, components):
    """Derive every subset membership under an explicit native input contract.

    Components follow native execution order and carry the actual seed/fold
    recipe. Retrieval is a leaf with checkpoint selection inside its scope.
    Numerical nodes expose final-fit inputs and each internal selection fold.
    An explicit frozen membership is restricted only when the component declares
    that policy; native consumers read these exact memberships from FoldInputs.
    An upstream fusion fit on T includes regularization selection on T, so its
    output for a held-out row must come from a complete separate fit on T minus H.
    """
    sources, valid_sources = sorted(set(sources)), sorted(set(valid_sources))
    if not sources or not valid_sources or set(sources) & set(valid_sources):
        raise ValueError("Disjoint complete TRAIN/public VALID populations required")
    allowed = {"fusion", "calibration", "selector", "original_query_ranker"}
    if any(c["kind"] not in allowed for c in components):
        raise ValueError("Unsupported numerical consumer")
    nodes, leaves = {}, {}

    def folds_for(index, population):
        c = components[index]
        if "membership" in c:
            if set(c["membership"]) != set(population):
                if c.get("subset_policy") != "restrict_authenticated_membership" or not set(
                    population
                ) <= set(c["membership"]):
                    raise ValueError("Explicit consumer membership must match this exact subset")
            result = {s: f for s, f in c["membership"].items() if s in population}
            if len(set(result.values())) < 2:
                raise ValueError("Nested component has fewer than two remaining folds")
            return result
        return membership(population, seed=c["seed"], folds=c["folds"])

    def fit(index, population):
        population = sorted(population)
        if index < 0:
            key = fingerprint({"label_influence_sources": population})
            leaves.setdefault(
                key,
                {
                    "label_influence_sources": population,
                    "excluded_sources": sorted(set(sources) - set(population)),
                },
            )
            return {"kind": "retrieval", "id": key}
        recipe = {"components": components[: index + 1], "label_influence_sources": population}
        key = fingerprint(recipe)
        if key in nodes:
            return {"kind": "consumer", "id": key}
        node = {
            "recipe": recipe,
            "component": components[index],
            "label_influence_sources": population,
            "fold_membership": folds_for(index, population),
            "full": oof(index - 1, population, components[index]),
            "application_dependency": fit(index - 1, population),
            "folds": {},
        }
        # Selection labels in every internal fold are conservatively part of
        # this complete fit's influence, even if optimizer rows are a subset.
        for fold in sorted(set(node["fold_membership"].values())):
            held = sorted(s for s, f in node["fold_membership"].items() if f == fold)
            train = sorted(set(population) - set(held))
            node["folds"][str(fold)] = {
                "train_sources": train,
                "heldout_sources": held,
                "training": oof(index - 1, train, components[index]),
                "application": [{"sources": held, "dependency": fit(index - 1, train)}],
            }
        nodes[key] = node
        return {"kind": "consumer", "id": key}

    def oof(index, population, consumer):
        # Upstream learned components use their own native fold recipe. The
        # leaf retrieval uses the immediate consumer's fold recipe.
        c = components[index] if index >= 0 else consumer
        if index >= 0:
            groups = folds_for(index, population)
        elif "membership" in c and c.get("subset_policy") == "restrict_authenticated_membership":
            groups = {s: f for s, f in c["membership"].items() if s in population}
            if len(set(groups.values())) < 2:
                raise ValueError("Nested retrieval has fewer than two remaining folds")
        else:
            groups = membership(population, seed=c["seed"], folds=c["folds"])
        entries = []
        for fold in sorted(set(groups.values())):
            held = sorted(s for s, f in groups.items() if f == fold)
            entries.append({"sources": held, "dependency": fit(index, set(population) - set(held))})
        return entries

    root = fit(len(components) - 1, sources)
    plan = {
        "schema": "luna-composed-consumer-folds-v1",
        "sources": sources,
        "valid_sources": valid_sources,
        "components": copy.deepcopy(components),
        "nodes": nodes,
        "retrieval_fits": leaves,
        "public_valid": [{"sources": valid_sources, "dependency": root}],
    }
    plan["sha256"] = fingerprint(plan)
    return plan


def validate_chain(plan):
    if plan != compile_chain(plan["sources"], plan["valid_sources"], plan["components"]):
        raise ValueError("Composed fold graph, memberships or dependencies changed")
    return plan


def _verified(bound):
    path = Path(bound["path"])
    if sha256_file(path) != bound["sha256"]:
        raise ValueError("Consumer input binding changed")
    return path


def _read_bound(bound):
    return json.loads(_verified(bound).read_text())


class FoldInputs:
    """Load byte-bound, per-consumer native frames with complete influence proof.

    The producer must supply each requested view, with separate dependency
    receipts per shard. Labels selecting upstream checkpoints count as influence.
    No pooled row-wise OOF declaration can replace these receipts. Construction
    validates the entire bundle before any native fitting/checkpoint reuse.
    """

    def __init__(self, manifest, *, consumer, training_sources, application_sources):
        self.manifest = dict(manifest)
        self.payload = _read_bound(manifest)
        p = self.payload
        if p.get("schema") != "luna-native-consumer-inputs-v1" or p.get("consumer") != consumer:
            raise ValueError("Wrong native consumer input contract")
        self.training_sources = set(training_sources)
        self.application_sources = set(application_sources)
        if self.training_sources & self.application_sources:
            raise ValueError("Consumer TRAIN/application overlap")
        if p["training_sources"] != sorted(self.training_sources) or p[
            "application_sources"
        ] != sorted(self.application_sources):
            raise ValueError("Consumer population changed")
        graph = validate_chain(_read_bound(p["graph"]))
        node = graph["nodes"][p["node_id"]]
        if (
            node["component"]["kind"] != consumer
            or set(node["label_influence_sources"]) != self.training_sources
        ):
            raise ValueError("Consumer node scope changed")
        self.membership = dict(node["fold_membership"])
        self.graph = graph
        expected = {
            "full": {"heldout_sources": sorted(self.application_sources), "entries": node["full"]}
        }
        expected.update(
            {
                f"fold-{fold}": {
                    "heldout_sources": view["heldout_sources"],
                    "entries": view["training"] + view["application"],
                }
                for fold, view in node["folds"].items()
            }
        )
        if set(p["views"]) != set(expected):
            raise ValueError("Missing native consumer fold input")
        self.frames = {}
        for name, view in p["views"].items():
            if view["heldout_sources"] != expected[name]["heldout_sources"]:
                raise ValueError("Consumer held-out membership changed")
            if [
                {"sources": s["sources"], "dependency": s["dependency"]} for s in view["shards"]
            ] != expected[name]["entries"]:
                raise ValueError("Consumer dependency graph projection changed")
            self.frames[name] = self._load(view)
        if "full" not in self.frames:
            raise ValueError("Missing final-fit input view")
        self._check_sources(self.frames["full"], self.training_sources)

    @staticmethod
    def _check_sources(frame, expected):
        if set(frame.Src.astype(str)) != set(expected):
            raise ValueError("Consumer view omitted or added source rows")
        if frame[["Src", "Tgt"]].astype(str).duplicated().any():
            raise ValueError("Duplicate candidate pairs in consumer view")

    def _load(self, view):
        import pandas as pd

        frames, seen = [], set()
        heldout = set(view["heldout_sources"])
        for shard in view["shards"]:
            frame = pd.DataFrame(_read_bound(shard["rows"]))
            rows = set(shard["sources"])
            if not rows or seen & rows:
                raise ValueError("Repeated/empty consumer shard")
            self._check_sources(frame, rows)
            pairs = _read_bound(shard["candidate_pairs"])
            actual_pairs = sorted([str(r.Src), str(r.Tgt)] for r in frame.itertuples())
            if actual_pairs != pairs:
                raise ValueError("Consumer candidate inventory changed or rows missing")
            seen |= rows
            if not shard["dependencies"]:
                raise ValueError("Supervised input requires authenticated dependencies")
            for bound in shard["dependencies"]:
                receipt = _read_bound(bound)
                if receipt["dependency"] != shard["dependency"]:
                    raise ValueError("Wrong upstream dependency receipt")
                dep = shard["dependency"]
                planned = self.graph["retrieval_fits" if dep["kind"] == "retrieval" else "nodes"][
                    dep["id"]
                ]
                influence = set(receipt["label_influence_sources"])
                if influence != set(planned["label_influence_sources"]):
                    raise ValueError("Upstream complete label influence changed")
                optimizer = set(receipt["optimizer_sources"])
                selected = set(receipt["checkpoint_selection_sources"])
                if not (optimizer | selected) <= influence:
                    raise ValueError("Checkpoint-selection label influence omitted")
                if not influence <= self.training_sources or influence & (
                    heldout | rows | self.application_sources
                ):
                    raise ValueError("Upstream heldout/self label influence")
                if receipt.get("status") != "complete" or not receipt.get("artifacts"):
                    raise ValueError("Uncompleted upstream dependency")
                for artifact in receipt["artifacts"]:
                    _verified(artifact)
            frames.append(frame)
        if not frames:
            raise ValueError("Missing consumer rows")
        return pd.concat(frames, ignore_index=True)

    def full(self):
        return self.frames["full"].copy(deep=True)

    def fold(self, fold, *, train_sources, heldout_sources):
        name = f"fold-{fold}"
        if name not in self.frames:
            raise ValueError("Missing native consumer fold input")
        view = self.payload["views"][name]
        if set(view["heldout_sources"]) != set(heldout_sources):
            raise ValueError("Native consumer fold membership changed")
        frame = self.frames[name].copy(deep=True)
        self._check_sources(frame, set(train_sources) | set(heldout_sources))
        return frame
