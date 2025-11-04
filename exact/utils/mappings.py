# Adapted from https://github.com/KRR-Oxford/DeepOnto

from ast import literal_eval

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