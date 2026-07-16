from collections import defaultdict, deque, namedtuple
from typing import Callable, Dict, List, Optional, Set, Tuple

from mowl.projection import Edge


def best_path_dp(
    out_edges: Dict[str, List[Edge]],
    edge_ic: Dict[Tuple[str, str, str], float],
    start: str,
    N: int,
    E_used: Set[Tuple[str, str, str]],
    C_rem: int,
    alpha: float,
    cost_fn: Callable[[Tuple[str, str, str]], int],
) -> Tuple[Optional[List[Tuple[str, str, str]]], int]:
    """
    Full label-setting DP over (node, cost, hops) states:
      maximize reward = sum(edge_ic) - alpha*hops
    subject to cost <= C_rem, hops <= N.
    Prunes dominated labels (cost↑, reward↓, hops↑).
    Returns the best path and its marginal cost.
    """
    Label = namedtuple("Label", ["node", "cost", "reward", "hops", "pred", "triple"])
    # labels[v] = list of non-dominated labels ending at v
    labels: Dict[str, List[Label]] = {start: [Label(start, 0, 0.0, 0, None, None)]}
    Q = deque(labels[start])

    best_label: Optional[Label] = None

    while Q:
        L = Q.popleft()
        if L.hops >= N:
            continue

        for e in out_edges.get(L.node, []):
            t = e.astuple()
            # incremental cost only if not already used
            if t in E_used:
                c_e, r_inc = 0, -alpha
            else:
                c_e = cost_fn(t)
                r_inc = edge_ic.get(t, 0.0) - alpha

            c2 = L.cost + c_e
            if c2 > C_rem:
                continue

            # accumulate reward and subtract per-hop penalty
            r2 = L.reward + r_inc
            h2 = L.hops + 1
            L2 = Label(e.dst, c2, r2, h2, L, t)

            # dominance
            doms = labels.get(e.dst, [])
            if any(d.cost <= L2.cost and d.reward >= L2.reward and d.hops <= L2.hops for d in doms):
                continue
            new = [
                d
                for d in doms
                if not (L2.cost <= d.cost and L2.reward >= d.reward and L2.hops <= d.hops)
            ]
            new.append(L2)
            labels[e.dst] = new
            Q.append(L2)
            if best_label is None or L2.reward > best_label.reward:
                best_label = L2

    if best_label is None:
        return None, 0

    # backtrack from best_label to build the path
    path: List[Tuple[str, str, str]] = []
    cur = best_label
    while cur.triple is not None:
        path.append(cur.triple)
        cur = cur.pred  # type: ignore
    path.reverse()

    # compute marginal cost of *new* edges
    marginal_cost = sum(cost_fn(t) for t in path if t not in E_used)
    return path, marginal_cost


def best_path_lagrangian_relaxation(
    out_edges: Dict[str, List[Edge]],
    edge_ic: Dict[Tuple[str, str, str], float],
    start: str,
    N: int,
    E_used: Set[Tuple[str, str, str]],
    C_rem: int,
    alpha: float,
    cost_fn: Callable[[Tuple[str, str, str]], int],
    iters: int = 8,
) -> Tuple[Optional[List[Tuple[str, str, str]]], int]:

    # 1) Prepare bounds for λ
    all_edges = [e.astuple() for edges in out_edges.values() for e in edges]
    c_min = min(cost_fn(t) for t in all_edges) if all_edges else 1
    r_max = max(edge_ic[t] for t in all_edges) if all_edges else 0.0

    def dp_for_lambda(lam: float):
        # best[h][v] = best λ‐weighted score to reach v in h hops
        best = [defaultdict(lambda: -1e9) for _ in range(N + 1)]
        prev = [{} for _ in range(N + 1)]
        best[0][start] = 0.0

        for h in range(1, N + 1):
            bh = best[h]
            ph = prev[h]
            for u, bu in best[h - 1].items():
                for e in out_edges.get(u, ()):
                    t = e.astuple()
                    # λ‐weighted edge score
                    if t in E_used:
                        w = -alpha
                    else:
                        w = edge_ic.get(t, 0.0) - alpha - lam * cost_fn(t)
                    val = bu + w
                    if val > bh.get(e.dst, -1e9):
                        bh[e.dst] = val
                        ph[e.dst] = u

        # pick best endpoint & hops
        end, end_h, best_val = None, 0, -1e9
        for h in range(1, N + 1):
            for v, val in best[h].items():
                if val > best_val:
                    best_val, end, end_h = val, v, h
        if end is None:
            return None, -1e9

        # backtrack triple‐path
        path = []
        v, h = end, end_h
        while h > 0:
            u = prev[h][v]
            # find the corresponding edge e
            for e in out_edges[u]:
                if e.dst == v:
                    path.append(e.astuple())
                    break
            v, h = u, h - 1
        return list(reversed(path)), best_val

    # 2) bisection on λ
    lo, hi = 0.0, (r_max / c_min if c_min > 0 else r_max)
    best_path, best_cost = None, 0

    for _ in range(iters):
        lam = 0.5 * (lo + hi)
        p, _ = dp_for_lambda(lam)
        if not p:
            hi = lam
            continue
        marg_cost = sum(cost_fn(t) for t in p if t not in E_used)
        if marg_cost > C_rem:
            lo = lam
        else:
            hi = lam
            best_path, best_cost = p, marg_cost

    return best_path or None, best_cost


def best_path_local(
    out_edges: Dict[str, List[Edge]],
    edge_ic: Dict[Tuple[str, str, str], float],
    start: str,
    N: int,
    E_used: Set[Tuple[str, str, str]],
    C_rem: int,
    alpha: float,
    cost_fn: Callable[[Tuple[str, str, str]], int],
) -> Tuple[Optional[List[Tuple[str, str, str]]], int]:
    """
    At each of up to N hops, pick the single outgoing edge e from the current node
    that maximizes (IC_edge(e)-alpha)/cost(e).  Runs in O(N * deg).
    """
    path = []
    total_cost = 0
    node = start

    for _ in range(N):
        best_score = float("-inf")
        best_edge = None
        best_c = 0

        nbrs = out_edges.get(node)
        if not nbrs:
            break
        for e in nbrs:
            triple = e.astuple()
            if triple in E_used:
                c = 0
                r = -alpha
            else:
                c = cost_fn(triple)
                r = edge_ic.get(triple, 0.0) - alpha
            if total_cost + c > C_rem:
                continue
            score = (r / c) if c else (r * 1e6)
            if score > best_score:
                best_score = score
                best_edge = e
                best_c = c

        if best_edge is None:
            break

        # commit that edge
        triple = best_edge.astuple()
        path.append(triple)
        total_cost += best_c
        E_used.add(triple)
        node = best_edge.dst

    return (path, total_cost) if path else (None, 0)


def extract_entity_context(args):
    """
    Unpack the tuple and call get_context_subgraph.
    Must live at top level so it can be pickled by ProcessPoolExecutor.
    """
    ent, n_hops, method, best_path_method, budget, hop_penalty = args
    return ent.get_context_subgraph(
        n_hops,
        human_readable=True,
        method=method,
        best_path_method=best_path_method,
        budget=budget,
        hop_penalty=hop_penalty,
    )
