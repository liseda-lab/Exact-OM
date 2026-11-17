"""Label-free, per-edge features and deterministic global assignment."""

import numpy as np
from scipy.special import expit, logit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

RAW = [
    "S_base",
    "S_pair_final",
    "U",
    "s_label",
    "S_struct",
    "s_hier",
    "s_sim",
    "s_diff",
    "s_attr",
    "cand_sim",
    "cand_sim_prob",
    "cand_share_log_ratio",
    "repair_distinctive",
    "original_exact_anchor",
    "src_obj_ic_mean",
    "tgt_obj_ic_mean",
    "src_hier_total_count",
    "tgt_hier_total_count",
    "src_obj_count",
    "tgt_obj_count",
    "src_attr_count",
    "tgt_attr_count",
]
FEATURES = RAW + [
    "source_final_margin",
    "target_final_margin",
    "source_label_margin",
    "target_label_margin",
    "source_struct_margin",
    "target_struct_margin",
    "source_rank",
    "target_rank",
    "source_count_log",
    "target_count_log",
    "specificity_difference",
]


def group_stats(values, codes, n):
    """Stable rank and margin against the strongest *other* edge; singleton gap=0."""
    if not len(values):
        return np.empty(0), np.empty(0, np.int32), np.zeros(n, np.int64)
    order = np.lexsort((np.arange(len(values)), -values, codes))
    cc = codes[order]
    starts = np.r_[0, np.flatnonzero(cc[1:] != cc[:-1]) + 1]
    lengths = np.diff(np.r_[starts, len(order)])
    rank = np.empty(len(order), np.int32)
    rank[order] = np.arange(len(order)) - np.repeat(starts, lengths)
    best = np.zeros(n)
    second = np.zeros(n)
    count = np.bincount(codes, minlength=n)
    best[cc[starts]] = values[order[starts]]
    second[cc[starts]] = values[order[starts + (lengths > 1)]]
    other = best[codes]
    first = order[starts]
    other[first] = second[codes[first]]
    gap = values - other
    gap[count[codes] == 1] = 0.0
    return gap, rank, count


def features(raw, src, tgt, ns, nt, out):
    out[:, : len(RAW)] = raw
    for j, name in enumerate(["S_pair_final", "s_label", "S_struct"]):
        v = np.asarray(raw[:, RAW.index(name)], dtype=np.float64)
        sg, sr, sc = group_stats(v, src, ns)
        tg, tr, tc = group_stats(v, tgt, nt)
        out[:, len(RAW) + 2 * j] = sg
        out[:, len(RAW) + 2 * j + 1] = tg
        if j == 0:
            out[:, len(RAW) + 6] = 1.0 / (sr + 1)
            out[:, len(RAW) + 7] = 1.0 / (tr + 1)
            out[:, len(RAW) + 8] = np.log1p(sc[src])
            out[:, len(RAW) + 9] = np.log1p(tc[tgt])

    def specificity(side):
        z = 0.6 * np.clip(raw[:, RAW.index(side + "_obj_ic_mean")], 0.0, 1.0)
        for c, cap, w in [
            ("hier_total_count", 20.0, 0.2),
            ("obj_count", 50.0, 0.1),
            ("attr_count", 20.0, 0.1),
        ]:
            z += w * np.minimum(
                1.0, np.log1p(np.maximum(0.0, raw[:, RAW.index(side + "_" + c)])) / np.log1p(cap)
            )
        return z

    out[:, -1] = abs(specificity("src") - specificity("tgt"))
    for start in range(0, len(out), 100000):
        if not np.isfinite(out[start : start + 100000]).all():
            raise ValueError("Nonfinite edge features")


def ordered(prob, src, tgt, threshold=0.0):
    idx = np.flatnonzero(prob > threshold)
    return idx[np.lexsort((tgt[idx], src[idx], -prob[idx]))]


def greedy(prob, src, tgt, threshold=0.0, winner_only=False):
    ns = int(src.max()) + 1 if len(src) else 0
    nt = int(tgt.max()) + 1 if len(tgt) else 0
    used_s = np.zeros(ns, bool)
    used_t = np.zeros(nt, bool)
    seen_s = np.zeros(ns, bool)
    result = []
    for i in ordered(prob, src, tgt, threshold):
        s, t = src[i], tgt[i]
        if winner_only:
            if seen_s[s]:
                continue
            seen_s[s] = True
        if not used_s[s] and not used_t[t]:
            result.append(i)
            used_s[s] = True
            used_t[t] = True
    return np.asarray(result, dtype=np.int64)


def assignment(prob, src, tgt, threshold, max_alternatives=5):
    """Maximize summed log-odds above threshold; private dummy per source permits NIL.

    No reference membership enters eligibility. At most five highest-scoring
    retained candidate edges/source bound the sparse graph. Positive costs avoid
    scipy treating a stored zero as an absent edge. Input order is canonical.
    """
    if not 0 < threshold < 1:
        raise ValueError("Assignment threshold must be strictly inside (0,1)")
    if max_alternatives < 1:
        raise ValueError("max_alternatives must be positive")
    if not len(src):
        return np.empty(0, dtype=np.int64)
    eligible = ordered(prob, src, tgt, threshold)
    counts = np.zeros(int(src.max()) + 1, np.int32)
    kept = []
    for i in eligible:
        if counts[src[i]] < max_alternatives:
            kept.append(i)
            counts[src[i]] += 1
    idx = np.asarray(kept, np.int64)
    if not len(idx):
        return idx
    ss, si = np.unique(src[idx], return_inverse=True)
    tt, ti = np.unique(tgt[idx], return_inverse=True)
    gain = logit(np.clip(prob[idx].astype(float), 1e-8, 1 - 1e-8)) - logit(threshold)
    shift = float(gain.max()) + 1.0
    rr = np.r_[si, np.arange(len(ss))]
    cc = np.r_[ti, len(tt) + np.arange(len(ss))]
    costs = np.r_[shift - gain, np.full(len(ss), shift)]
    graph = coo_matrix((costs, (rr, cc)), shape=(len(ss), len(tt) + len(ss))).tocsr()
    row, col = min_weight_full_bipartite_matching(graph)
    key = si.astype(np.int64) * len(tt) + ti
    order = np.argsort(key)
    real = col < len(tt)
    wanted = row[real] * len(tt) + col[real]
    pos = np.searchsorted(key[order], wanted)
    if not np.array_equal(key[order[pos]], wanted):
        raise ValueError("Assignment selected absent edge")
    chosen = idx[order[pos]]
    return chosen[np.lexsort((tgt[chosen], src[chosen]))]


def select(prob, src, tgt, *, threshold, mode):
    """Select one-to-one mappings with an explicitly frozen operating policy."""
    prob = np.asarray(prob)
    if not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
        raise ValueError("Acceptance probabilities must be finite and in [0, 1]")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1]")
    if mode == "assignment":
        return assignment(prob, src, tgt, threshold)
    if mode not in {"greedy", "winner"}:
        raise ValueError("Selection mode must be greedy, winner, or assignment")
    return greedy(prob, src, tgt, threshold, winner_only=mode == "winner")
