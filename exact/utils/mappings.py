# Adapted from https://github.com/KRR-Oxford/DeepOnto

from ast import literal_eval

import pandas as pd

from exact.core.entities.mappings import EntityMapping


def fill_anchored_scores(ref_anchored_maps, pred_maps):
    """Fill scores of the anchored reference mappings with the scores of the predicted mappings."""

    pred_maps_tuples = EntityMapping.as_tuples(pred_maps, with_score=True)

    pred_maps_dict = {}
    for source, tgt, score in pred_maps_tuples:
        if source not in pred_maps_dict:
            pred_maps_dict[source] = {}
        pred_maps_dict[source][tgt] = score

    results = []
    for src_ref_class, tgt_ref_class, tgt_cands in ref_anchored_maps:
        tgt_cands = literal_eval(tgt_cands)
        scored_cands = []
        for tgt_cand in tgt_cands:
            try:
                scored_cands.append((tgt_cand, pred_maps_dict[src_ref_class][tgt_cand]))

            except KeyError:
                scored_cands.append((tgt_cand, 0.0))

        results.append((src_ref_class, tgt_ref_class, scored_cands))
    return results


def candidate_table_views(frame):
    """Return scorer pairs and a local writer view without leaking reference targets.

    Benchmark rows carry source, gold target and a candidate-list column. Frozen
    generated pools carry ordinary source/target rows and may include retrieval
    scores and kinds. Their local writer view has no invented gold target.
    """
    if len(frame.columns) < 2:
        raise ValueError("candidate tables need source and target columns")
    source, target = frame.columns[:2]
    list_column = next(
        (name for name in ("TgtCandidates", "Candidates", "candidates") if name in frame.columns),
        None,
    )
    if list_column is not None:
        anchored = frame[[source, target, list_column]].copy()
        anchored.columns = ["Src", "Tgt", "Candidates"]
        rows = []
        for src, _gold, encoded in anchored.itertuples(index=False, name=None):
            candidates = literal_eval(encoded) if isinstance(encoded, str) else encoded
            if not isinstance(candidates, (list, tuple)) or any(
                not isinstance(iri, str) for iri in candidates
            ):
                raise ValueError("candidate lists must contain target IRI strings")
            rows.extend((str(src), iri) for iri in candidates)
        pairs = pd.DataFrame(rows, columns=["Src", "Tgt"])
    else:
        retained = [source, target] + [
            name for name in ("SrcKind", "TgtKind", "cand_sim") if name in frame.columns
        ]
        pairs = frame[retained].rename(columns={source: "Src", target: "Tgt"}).copy()
        anchored = pd.DataFrame(
            [
                (str(src), "", repr(sorted(set(group["Tgt"].astype(str)))))
                for src, group in pairs.groupby("Src", sort=True)
            ],
            columns=["Src", "Tgt", "Candidates"],
        )
    pairs = pairs.dropna(subset=["Src", "Tgt"]).drop_duplicates(["Src", "Tgt"])
    pairs["Label"] = 0
    return pairs.reset_index(drop=True), anchored
