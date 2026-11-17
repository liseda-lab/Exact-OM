"""Native numerical/evidence adapters used by the frozen matching architecture."""
from __future__ import annotations
import copy
import collections
import math

def primary_parameters(config, *, enhanced_channels=None, device="cuda"):
    """Match native alignment's primary constructor and pre-fit fusion stage."""
    from exact.core.entities.configs.config import ConfigModel

    cfg = ConfigModel.from_mapping(config, warn_v1=False)
    params = copy.deepcopy(cfg.pipeline[0].params)
    params.update(cfg.matching.channels.model_dump(mode="python"))
    fusion = cfg.matching.fusion.model_dump(mode="python")
    if enhanced_channels is not None:
        params.update(copy.deepcopy(enhanced_channels))
        # Native alignment uses analytic_shipped until the separately fitted
        # analytic_fitted artifact exists. This output is explicitly pre-fit.
        fusion.update(enabled=True, mode="analytic_shipped", artifact=None)
    params.update(
        fusion_config=fusion,
        llm_experiment_config=cfg.llm.experiment.model_dump(mode="python"),
        llm_profiles={k: v.model_dump() for k, v in cfg.llm_profiles.items()},
        llm_routing=cfg.llm_routing.model_dump(),
        request_seed=cfg.seed,
        **cfg.alignment_params.model_dump(exclude_none=True),
    )
    params.update(
        use_llm=False,
        force_llm_summaries=False,
        generate_llm_rationales=False,
        use_llm_calibration=False,
        return_explanations=True,
        persist_cache_to_disk=False,
        device=device,
    )
    if params["graph"]["mode"] != "off" or params["fusion_config"]["mode"] != "analytic_shipped":
        raise ValueError("Primary production cannot load a supervised numerical artifact")
    return params


def decorate_rows(frame, rows):
    """Attach current retriever metadata after projecting/reusing native scores."""
    from exact.impl.models.selector.feature_contract import SPECIFICITY_FEATURES

    return [
        {
            **row,
            **{
                k: v
                for k, v in source.items()
                if str(k).startswith("cand_") or k in SPECIFICITY_FEATURES
            },
        }
        for row, (_, source) in zip(rows, frame.iterrows())
    ]


def rows_from_output(frame, output, *, facts, decorate=True):
    import torch
    from exact.impl.trainer.study_training import structured_training_packet

    count = len(frame)
    columns = {
        k: v.detach().cpu().tolist()
        for k, v in output.items()
        if isinstance(v, torch.Tensor) and v.ndim == 1 and len(v) == count
    }
    if not {"S_base", "S_final", "U"} <= set(columns):
        raise ValueError("Missing native primary score columns")
    channels = {
        k: {f: v.detach().cpu().tolist() for f, v in ch.items()}
        for k, ch in output["fusion_channels"].items()
    }
    explanations = output.get("explanations", [])
    if facts and len(explanations) != count:
        raise ValueError("Native selected evidence missing")
    result = []
    for i, (_, source) in enumerate(frame.iterrows()):
        row = {k: str(source.get(k, "class")) for k in ("Src", "Tgt", "SrcKind", "TgtKind")}
        row.update({k: float(v[i]) for k, v in columns.items()})
        row["fusion_channels"] = {
            k: {f: float(v[i]) for f, v in ch.items()} for k, ch in channels.items()
        }
        row["src_label_text"] = " | ".join(source.SrcLabels)
        row["tgt_label_text"] = " | ".join(source.TgtLabels)
        if not all(math.isfinite(row[k]) for k in columns):
            raise ValueError("Nonfinite native primary scores")
        if facts:
            row["selected_labels"] = explanations[i]["selected_labels"]
            row["llm_evidence_packet"] = output["llm_evidence_packets"][i]
            row["llm_structured_evidence_packet"] = structured_training_packet(explanations[i])
            # Retain the actual selected attribution proof, never unselected bundles.
            row["selected_attributions"] = {
                k: explanations[i][k] for k in ("triple_attributions", "attributes")
            }
        result.append(row)
    return decorate_rows(frame, result) if decorate else result


def resident_items(row, selector):
    """Recover exactly the fields consumed by _record_evidence_items.

    Native hierarchy/similarity source membership is iff reduced row-max support
    is positive. Difference links cover every selected item; selected attributes
    exist only with a nonempty, valid anchor bank. Context triples are the selected
    triples in their original order. Pair/link scores are not used by the selector.
    This does not claim to restore the omitted cross-side link pairings.
    """
    attrs=row['selected_attributions']
    if set(attrs)!={'triple_attributions','attributes'}:
        raise ValueError('Incomplete fixed selected attributions')
    triples=attrs['triple_attributions']
    if set(triples)!={'hierarchy','similarity','difference'}:
        raise ValueError('Changed selected attribution channels')
    linked=[]
    for channel,payload in [('hierarchy',p) for p in triples['hierarchy'].values()]+[('similarity',triples['similarity']),('difference',triples['difference']),('attribute',attrs['attributes'])]:
        if set(payload)!={'source','target'}:raise ValueError('Incomplete selected attribution sides')
        for side in ['source','target']:
            for item in payload[side]:
                if not item.get('item_id'):raise ValueError('Selected native item has no identity')
                if channel in ['difference','attribute'] or float(item['support'])>0:
                    linked.append(dict(item_id=item['item_id']))
    record={**attrs,'context_triples':{
        'hierarchy_target':{family:[item['triple'] for item in p['target']] for family,p in triples['hierarchy'].items()},
        'similarity_target':[item['triple'] for item in triples['similarity']['target']],
        'difference_target':[item['triple'] for item in triples['difference']['target']]},
        'cross_side_provenance':{'selector_linked_membership_only':linked}}
    return [dict(key=str(k),strength=float(v)) for k,v in selector._record_evidence_items(record)]


def recompute_retrieval(rows):
    """Match BaseDataset arithmetic/order, after removing ineligible endpoints."""
    eps = 2**-23
    total = sum(float(r['cand_sim']) for r in rows)
    mean = total / len(rows) if rows else 0.
    probs = [float(r['cand_sim']) / (total or eps) for r in rows]
    top = sum(probs[:2])
    rest = max(0., 1. - top)
    ratio = math.log((top + eps) / (rest + eps))
    return [dict(r, cand_sim_src_mean=mean, cand_sim_prob=p, cand_share_top=top,
                 cand_share_rest=rest, cand_share_log_ratio=ratio) for r, p in zip(rows, probs)]


def distinctive(items):
    """Native per-source IDF/maximum normalization, independently testable."""
    df = collections.Counter(k for group in items for k in {x['key'] for x in group})
    n = len(items)
    raw = [sum(max(0., x['strength']) * max(0., math.log((n + 1.) / (df[x['key']] + 1.)))
               for x in group) for group in items]
    maximum = max(raw, default=0.)
    return [x / maximum if maximum > 1e-8 else 0. for x in raw]
