import logging
import math
import re
import sys
import time
from collections import Counter, defaultdict, deque
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

from exact.core.contracts.knowledge import KnowledgeSource
from exact.core.entities.configs.dataset import BestPathMethod, ContextMethod
from exact.core.entities.graph import Edge
from exact.core.entities.kinds import EntityKind
from exact.utils.graph_search import (
    best_path_dp,
    best_path_lagrangian_relaxation,
    best_path_local,
)


class OntologyGraph:
    """
    Builds a cached graph structure for an ontology by projecting it using OWL2VecStarProjector.
    The edges are stored with IRIs. Label extractions are cached for efficiency.
    """

    def __init__(
        self,
        ontology: KnowledgeSource,
        reasoner: object | None = None,
        only_taxonomy: bool = False,
        include_literals: bool = False,
    ) -> None:
        self.source = ontology
        # ``ontology`` remains as a compatibility alias for callers that only
        # passed it through.  It is now a KnowledgeSource, never an OWLAPI object.
        self.ontology = ontology
        self.reasoner = reasoner
        self.only_taxonomy = only_taxonomy
        self.include_literals = include_literals
        self._logger = logging.getLogger("exact")
        self.edges: Optional[List[Edge]] = None
        self.out_edges: Dict[str, List[Edge]] = None
        self.graph: Optional[Dict[str, Set[str]]] = None
        self.label_cache: Dict[str, List[str]] = {}  # Cache: IRI string -> list of labels (str)

        # caches (IRI-based)
        self._relations_cache: Optional[Set[str]] = None
        self._dr_cache: Optional[Dict[str, Dict[str, List[str]]]] = None
        self._missing_dom_cache: Optional[List[str]] = None
        self._missing_rng_cache: Optional[List[str]] = None
        self._example_triples_cache: Optional[Dict[str, List[Tuple[str]]]] = None
        self._context_subgraph_cache: Dict[Tuple[str, int, bool], List[Tuple[str]]] = {}

        # IC caches

        self._node_ic_cache: Optional[Dict[Tuple[str, str], float]] = None
        self._edge_ic_cache: Optional[Dict[Tuple[str, str, str], float]] = None
        self._edge_ic_max_cache: Optional[float] = None

        # edge cost cache
        self._edge_cost_cache: Dict[Tuple[str, str, str], int] = None

        # adjacency cache including relation labels (for shortest paths)
        self._rel_adj_cache: Optional[Dict[str, List[Tuple[str, str]]]] = None
        self._incident_edge_cache: Optional[Dict[str, List[Tuple[str, str, str]]]] = None
        self._raw_neighborhood_cache: Dict[Tuple[str, int, bool], List[Tuple[str, str, str]]] = {}

        self._build_projection()
        self.precompute_all_labels()

    def __repr__(self) -> str:
        return (
            f"OntologyGraph(origin={self.source.origin!s}, "
            f"NumberOfEdges={len(self.edges) if self.edges else 0}, "
            f"NumberOfNodes={len(self.graph) if self.graph else 0})"
        )

    def __len__(self) -> int:
        return len(self.edges) if self.edges is not None else 0

    def __getitem__(self, item: str) -> Set[str]:
        """
        Allows access to the graph structure using the IRI as a key.
        Returns a set of connected nodes (IRIs) for the given IRI.
        """
        if self.graph is None:
            raise ValueError("Graph has not been built yet.")
        return self.graph.get(item, set())

    @property
    def node_ic(self) -> Dict[Tuple[str, str], float]:
        """Relation-local node surprisal: ICr(x) = −log(deg_r(x)/|E_r|)."""
        if self._node_ic_cache is None:
            rel2counts = defaultdict(Counter)
            rel2size = defaultdict(int)
            for e in self.edges:
                rel2counts[e.rel][e.src] += 1
                rel2counts[e.rel][e.dst] += 1
                rel2size[e.rel] += 1
            self._node_ic_cache = {}
            for r, counts in rel2counts.items():
                m = rel2size[r]
                for x, deg in counts.items():
                    self._node_ic_cache[(r, x)] = -math.log(deg / m)
        return self._node_ic_cache

    @property
    def edge_ic(self) -> Dict[Tuple[str, str, str], float]:
        """Per-edge IC = avg of endpoints’ relation-local surprisal."""
        if self._edge_ic_cache is None:
            # ensure node_ic is populated
            _ = self.node_ic
            self._edge_ic_cache = {}
            for e in self.edges:
                u, r, v = e.astuple()
                ic_u = self._node_ic_cache[(r, u)]
                ic_v = self._node_ic_cache[(r, v)]
                self._edge_ic_cache[(u, r, v)] = 0.5 * (ic_u + ic_v)
            self._edge_ic_max_cache = max(self._edge_ic_cache.values(), default=1.0) or 1.0
        return self._edge_ic_cache

    @property
    def avg_edge_ic(self) -> float:
        """Average edge IC across all edges in the graph."""
        return sum(self.edge_ic.values()) / len(self.edge_ic)

    @property
    def cost_fn(self) -> Callable[[Tuple[str, str, str]], int]:
        """
        Returns a fast lookup into the precomputed edge‐cost cache.
        """
        if self._edge_cost_cache is None:
            raise ValueError("Cost function has not been set or edges have not been built yet.")

        return lambda triple: self._edge_cost_cache[triple]

    @cost_fn.setter
    def cost_fn(self, fn: Callable[[Tuple[str, str, str]], int]):
        """
        When the user assigns a new cost function, precompute its value
        for every edge in the graph and cache it for O(1) lookup.
        """
        self._cost_fn = fn
        # Build the cache once
        self._edge_cost_cache = {}
        for e in self.edges:
            t = e.astuple()
            human_readable = (self.get_labels(i)[0] for i in t)
            self._edge_cost_cache[t] = fn(human_readable)

    def _build_projection(self) -> None:

        proj_name = "TaxonomyProjector" if self.only_taxonomy else "OWL2VecStarProjector"
        start = time.time()
        self._log(
            f"OntologyGraph: projecting ontology ({proj_name}), only_taxonomy={self.only_taxonomy}",
            level="info",
        )

        method = "taxonomy" if self.only_taxonomy else "owl2vecstar"
        self.edges = self.source.projection_edges(
            method=method, include_literals=self.include_literals
        )
        self.out_edges = self._build_out_edges()
        self.graph = self._build_graph(self.edges)
        self._log(
            f"OntologyGraph: projection finished ({proj_name}) in {time.time() - start:.2f}s with {len(self.edges)} edges",
            level="info",
        )

    # ------------------------------------------------------------------
    # Logging helper that falls back to stderr if logger is misconfigured.
    # ------------------------------------------------------------------
    def _log(self, msg: str, level: str = "info") -> None:
        try:
            log_fn = getattr(self._logger, level, self._logger.info)
            log_fn(msg)
        except Exception:
            print(msg, file=sys.stderr, flush=True)

    def _build_out_edges(self) -> Dict[str, List[Edge]]:
        """
        Builds a mapping of source IRIs to their outgoing edges.
        This is useful for quickly accessing all edges originating from a specific node.
        """
        if self.edges is None:
            raise ValueError("Edges have not been built yet.")

        out_edges = defaultdict(list)
        for edge in self.edges:
            out_edges[edge.src].append(edge)

        return out_edges

    def _build_graph(self, edges: List[Edge]) -> Dict[str, Set[str]]:
        graph = defaultdict(set)
        for edge in edges:
            # Each edge connects its source and destination (both are IRIs).
            graph[edge.src].add(edge.dst)
            graph[edge.dst].add(edge.src)
        return dict(graph)

    def get_relations(self, human_readable: bool = True) -> Set[str]:
        if self._relations_cache is None:
            self._relations_cache = {e.rel for e in self.edges}
        if not human_readable:
            return set(self._relations_cache)
        # human_readable: return each relation’s full-label tuple
        return {self.get_labels(rel)[0] for rel in self._relations_cache}

    def get_property_domains_and_ranges(
        self, human_readable: bool = True
    ) -> Dict[Union[str, Tuple[str, ...]], Dict[str, List[Union[str, List[str]]]]]:
        """
        Returns a map from relation → {'domain': [...], 'range': [...]}
        If human_readable=True, keys and all class IRIs become label-lists.
        """
        if self._dr_cache is None:
            dr: Dict[str, Dict[str, List[str]]] = {}
            for rel in self.get_relations(human_readable=False):
                dr[rel] = {
                    "domain": self.source.property_domains(rel),
                    "range": self.source.property_ranges(rel),
                }
            self._dr_cache = dr

        if not human_readable:
            return self._dr_cache

        # convert IRIs → lists of labels
        hr: Dict[Tuple[str, ...], Dict[str, List[List[str]]]] = {}
        for rel_iri, drmap in self._dr_cache.items():
            rel_lbl = tuple(self.get_labels(rel_iri))
            hr[rel_lbl] = {
                "domain": [self.get_labels(iri) for iri in drmap["domain"]],
                "range": [self.get_labels(iri) for iri in drmap["range"]],
            }
        return hr

    def get_relations_missing_domain(self, human_readable: bool = True) -> List[str]:
        if self._missing_dom_cache is None:
            self._missing_dom_cache = [
                rel
                for rel, dr in self.get_property_domains_and_ranges(human_readable=False).items()
                if not dr["domain"]
            ]
        if not human_readable:
            return list(self._missing_dom_cache)
        return [self.get_labels(rel)[0] for rel in self._missing_dom_cache]

    def get_relations_missing_range(self, human_readable: bool = True) -> List[str]:
        if self._missing_rng_cache is None:
            self._missing_rng_cache = [
                rel
                for rel, dr in self.get_property_domains_and_ranges(human_readable=False).items()
                if not dr["range"]
            ]
        if not human_readable:
            return list(self._missing_rng_cache)
        return [self.get_labels(rel)[0] for rel in self._missing_rng_cache]

    def get_example_triples(
        self, n: int, exclude_missing_dr: bool = False, human_readable: bool = True
    ) -> Dict[str, List[Tuple[str]]]:
        """
        For each unique relation, return up to `n` example triples (src, rel, dst).

        :param n: number of examples per relation
        :param exclude_missing_dr: if True, skip relations that have no declared domain or no declared range
        :param human_readable: if True, convert IRIs to their full list of labels
        :return: dict mapping each relation (IRI or label‐tuple) to a list of example triples
        """
        if self._example_triples_cache is None:
            rels = set(self.get_relations(human_readable=False))
            if exclude_missing_dr:
                missing = set(self.get_relations_missing_domain(human_readable=False)) | set(
                    self.get_relations_missing_range(human_readable=False)
                )
                rels = rels - missing

            examples: Dict = {}
            for rel in rels:

                # gather up to n triples for this relation
                exs = []
                for edge in self.edges:
                    if edge.rel != rel:
                        continue
                    else:
                        src, dst, rel_lbl = str(edge.src), str(edge.dst), str(edge.rel)

                    exs.append((src, rel_lbl, dst))
                    if len(exs) >= n:
                        break

                examples[rel] = exs

            self._example_triples_cache = examples

        if human_readable:
            # convert IRIs to their full list of labels
            examples = {}
            for rel, exs in self._example_triples_cache.items():
                examples[self.get_labels(rel)[0]] = [
                    (self.get_labels(src)[0], self.get_labels(rel_lbl)[0], self.get_labels(dst)[0])
                    for src, rel_lbl, dst in exs
                ]
            return examples

        else:
            # return IRIs
            return self._example_triples_cache

    def get_labels(self, iri: str) -> List[str]:
        """
        Extracts (and caches) the list of labels for a given IRI from the ontology.
        If no label is found, returns a list containing the short form of the IRI.
        """
        if iri not in self.label_cache:
            if iri is None:
                self.label_cache[iri] = [""]
                return self.label_cache[iri]
            iri_text = str(iri)
            if self._looks_like_literal_or_blank(iri_text):
                self.label_cache[iri] = [self._normalize_literal_text(iri_text)]
                return self.label_cache[iri]
            labels = self.source.labels(iri_text)
            self.label_cache[iri] = labels if labels else [self.source.short_form(iri_text)]
        return self.label_cache[iri]

    @staticmethod
    def _looks_like_literal_or_blank(text: str) -> bool:
        if text is None:
            return True
        txt = str(text).strip()
        if not txt:
            return True
        if txt.startswith("_:"):
            return True
        if txt.startswith('"') and txt.endswith('"'):
            return True
        if txt.startswith("'") and txt.endswith("'"):
            return True
        if "^^" in txt or txt.startswith("literal:"):
            return True
        return not bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", txt))

    @staticmethod
    def _normalize_literal_text(text: str) -> str:
        txt = str(text).strip()
        txt = re.sub(r"^literal:", "", txt, flags=re.IGNORECASE).strip()
        txt = re.sub(r"\^\^.*$", "", txt).strip()
        txt = txt.strip("'\"")
        return txt or str(text)

    def edge_ic_norm(self, triple: Tuple[str, str, str]) -> float:
        edge_ic = self.edge_ic
        val = float(edge_ic.get(triple, 0.0))
        if not edge_ic:
            return 0.0
        mx = self._edge_ic_max_cache or 1.0
        return max(0.0, min(1.0, val / mx))

    def _incident_edges(self) -> Dict[str, List[Tuple[str, str, str]]]:
        if self._incident_edge_cache is None:
            incident = defaultdict(list)
            for edge in self.edges:
                triple = (str(edge.src), str(edge.rel), str(edge.dst))
                incident[triple[0]].append(triple)
                incident[triple[2]].append(triple)
            self._incident_edge_cache = dict(incident)
        return self._incident_edge_cache

    def get_raw_neighborhood(
        self,
        start: str,
        n: int,
        include_reverse: bool = True,
    ) -> List[Tuple[str, str, str]]:
        key = (start, int(n), bool(include_reverse))
        if key in self._raw_neighborhood_cache:
            return list(self._raw_neighborhood_cache[key])

        visited = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            dist = visited[node]
            if dist < n:
                for nbr in self.graph.get(node, []):
                    if nbr not in visited:
                        visited[nbr] = dist + 1
                        queue.append(nbr)
        reachable = set(visited.keys())
        triples: List[Tuple[str, str, str]] = []
        seen = set()
        incident = self._incident_edges()
        candidate_nodes = reachable if include_reverse else {start}
        for node in candidate_nodes:
            for src, rel, dst in incident.get(node, []):
                if src in reachable and dst in reachable:
                    triple = (src, rel, dst)
                    if triple in seen:
                        continue
                    seen.add(triple)
                    triples.append(triple)
        self._raw_neighborhood_cache[key] = list(triples)
        return triples

    def precompute_all_labels(self) -> None:
        """
        Precomputes and caches labels for all IRIs that appear in the projection.
        This includes source, destination, and relation IRIs. The extraction is parallelized.
        """
        iris = set()
        # Collect IRIs from each edge in the projection.
        for edge in self.edges:
            iris.add(edge.src)
            iris.add(edge.dst)
            iris.add(edge.rel)

        for iri in iris:
            self.get_labels(iri)

    def get_context_subgraph(
        self,
        start: str,
        n: int,
        human_readable: bool = True,
        method: ContextMethod = ContextMethod.bfs,  # "bfs" or "greedy"
        best_path_method: Optional[BestPathMethod] = None,  # "dp", "lagrangian", "greedy"
        budget: Optional[int] = None,  # token budget
        hop_penalty: Optional[float] = 0.0,  # α
        all_labels: bool = False,  # If True, return all labels for each node and relation
    ) -> List[Tuple[str, str, str]]:
        """
        If method="bfs": original n-hop BFS (cached).
        If method="greedy": runs budget-aware greedy extraction.
        """

        key = (start, n, method, human_readable)
        if key in self._context_subgraph_cache:
            return self._context_subgraph_cache[key]

        if method is ContextMethod.bfs:

            visited = {start: 0}
            queue = deque([start])
            while queue:
                node = queue.popleft()
                dist = visited[node]
                if dist < n:
                    for nbr in self.graph.get(node, []):
                        if nbr not in visited:
                            visited[nbr] = dist + 1
                            queue.append(nbr)
            reachable = set(visited.keys())

            subgraph_edges: List[Tuple[str, str, str]] = []

            for edge in self.edges:
                if edge.src in reachable and edge.dst in reachable:
                    # get either IRIs or labels
                    if human_readable:
                        src = self.get_labels(edge.src)
                        dst = self.get_labels(edge.dst)
                        rel = self.get_labels(edge.rel)
                    else:
                        src, dst, rel = [str(edge.src)], [str(edge.dst)], [str(edge.rel)]

                    if all_labels:
                        # return all combinations of labels
                        for s in src:
                            for d in dst:
                                for r in rel:
                                    subgraph_edges.append((s, r, d))
                    else:
                        # return single labels
                        subgraph_edges.append((src[0], rel[0], dst[0]))

            self._context_subgraph_cache[key] = subgraph_edges

            return subgraph_edges

        else:

            # Adjusting hop penalty based on average edge IC
            hop_penalty = hop_penalty * self.avg_edge_ic

            triples = self._greedy_context_subgraph(
                start, n, best_path_method, human_readable, budget, hop_penalty, self.cost_fn
            )

            self._context_subgraph_cache[key] = triples

            return triples

    def _greedy_context_subgraph(
        self,
        start: str,
        n: int,
        method: BestPathMethod,
        human_readable: bool,
        budget: Optional[int],
        hop_penalty: float,
        token_cost_fn: Optional[callable],
    ) -> List[Tuple[str, str, str]]:

        E_used = set()
        C_rem = budget if budget is not None else float("inf")

        if method is BestPathMethod.dp:
            best_path_fn = best_path_dp
        elif method is BestPathMethod.lagrangian:
            best_path_fn = best_path_lagrangian_relaxation
        else:
            best_path_fn = best_path_local

        selected = []
        while C_rem > 0:
            path, delta_c = best_path_fn(
                self.out_edges, self.edge_ic, start, n, E_used, C_rem, hop_penalty, token_cost_fn
            )
            if path is None or delta_c == 0 or delta_c > C_rem:
                break
            selected.extend(path)
            E_used.update(path)
            C_rem -= delta_c
        return self._format_subgraph(E_used, human_readable)

    def _format_subgraph(
        self, selected: List[Tuple[str, str, str]], human_readable: bool
    ) -> List[Tuple[str, str, str]]:
        """
        Formats the selected subgraph edges based on whether human-readable labels are requested.
        """
        if human_readable:
            return [
                (self.get_labels(src)[0], self.get_labels(rel)[0], self.get_labels(dst)[0])
                for src, rel, dst in selected
            ]
        return selected

    def _is_subclass_rel(self, rel_iri: str) -> bool:
        """
        Lightweight check to treat variations of the subclass relation as taxonomy edges.
        """
        if rel_iri is None:
            return False
        r = rel_iri.lower()
        return "subclassof" in r or "subclass_of" in r

    @property
    def rel_adj(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Cached adjacency list that keeps relation labels for each neighbor traversal.
        Used for shortest-path bridging in explanations.
        """
        if self._rel_adj_cache is None:
            adj = defaultdict(list)
            for e in self.edges:
                adj[e.src].append((e.dst, e.rel))
                adj[e.dst].append((e.src, e.rel))
            self._rel_adj_cache = dict(adj)
        return self._rel_adj_cache

    def _shortest_path_to_set(
        self,
        start: str,
        targets: Set[str],
        max_depth: Optional[int] = None,
    ) -> List[Tuple[str, str, str]]:
        """
        BFS to the nearest node in `targets`, returning the edge sequence as triples.
        If no path is found (or depth exceeded), returns [].
        """
        if not targets:
            return []
        q = deque([(start, [])])
        seen = {start}
        adj = self.rel_adj
        while q:
            node, path = q.popleft()
            if node in targets and path:
                return path
            if max_depth is not None and len(path) >= max_depth:
                continue
            for nbr, rel in adj.get(node, []):
                if nbr in seen:
                    continue
                seen.add(nbr)
                q.append((nbr, path + [(node, rel, nbr)]))
        return []

    def _connectivity_bridges(
        self,
        start: str,
        selected: List[Tuple[str, str, str]],
        max_depth: Optional[int] = None,
    ) -> List[Tuple[str, str, str]]:
        """
        Ensure every selected triple can reach `start` by adding bridging triples.
        Taxonomy-only components get a direct subclass bridge; other components
        receive the shortest available path in the projected graph (bounded by `max_depth`).
        """
        if not selected:
            return []

        # Build adjacency for the selected subgraph (undirected for connectivity)
        adj = defaultdict(set)
        for u, _, v in selected:
            adj[u].add(v)
            adj[v].add(u)

        # Reachable nodes within the selected triples from the start
        reachable = set()
        dq = deque([start])
        while dq:
            node = dq.popleft()
            if node in reachable:
                continue
            reachable.add(node)
            for nbr in adj.get(node, []):
                if nbr not in reachable:
                    dq.append(nbr)

        bridges: List[Tuple[str, str, str]] = []
        visited = set()

        # Identify disconnected components and bridge them
        for node in adj.keys():
            if node in reachable or node in visited:
                continue
            comp_nodes: Set[str] = set()
            dq = deque([node])
            while dq:
                cur = dq.popleft()
                if cur in comp_nodes:
                    continue
                comp_nodes.add(cur)
                for nbr in adj.get(cur, []):
                    if nbr not in comp_nodes:
                        dq.append(nbr)
            visited.update(comp_nodes)

            comp_edges = [(u, r, v) for (u, r, v) in selected if u in comp_nodes or v in comp_nodes]
            # Taxonomy components: add a direct subclass bridge to the start
            if comp_edges and all(self._is_subclass_rel(r) for _, r, _ in comp_edges):
                bridge_rel = comp_edges[0][1]
                bridges.append((node, bridge_rel, start))
                reachable.update(comp_nodes)
                reachable.add(start)
                continue

            # General case: shortest path from component to any reachable node
            path = self._shortest_path_to_set(node, reachable, max_depth=max_depth)
            if path:
                bridges.extend(path)
                for u, _, v in path:
                    reachable.add(u)
                    reachable.add(v)

        return bridges

    def get_context_subgraph_with_bridges(
        self,
        start: str,
        n: int,
        human_readable: bool = True,
        method: ContextMethod = ContextMethod.bfs,
        best_path_method: Optional[BestPathMethod] = None,
        budget: Optional[int] = None,
        hop_penalty: Optional[float] = 0.0,
        max_bridge_hops: Optional[int] = None,
        all_labels: bool = False,
    ) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        """
        Returns (selected_context_triples, bridge_triples) where bridge_triples are
        added only for explainability to ensure connectivity to `start`.
        """
        raw_selected = self.get_context_subgraph(
            start,
            n,
            human_readable=False,
            method=method,
            best_path_method=best_path_method,
            budget=budget,
            hop_penalty=hop_penalty,
            all_labels=all_labels,
        )
        bridges_raw = self._connectivity_bridges(start, raw_selected, max_depth=max_bridge_hops)
        selected_fmt = self._format_subgraph(raw_selected, human_readable)
        bridges_fmt = self._format_subgraph(bridges_raw, human_readable)
        return selected_fmt, bridges_fmt

    @staticmethod
    def normalize_label(text: str) -> str:
        """Lowercase and remove spaces/punctuation for exact-matching."""
        import re

        if text is None:
            return ""
        t = text.lower()
        t = re.sub(r"[\W_]+", "", t)  # drop non-alnum incl. underscores
        return t

    def get_all_classes(self) -> List[str]:
        """
        Return all named class IRIs present in the ontology.
        Uses the OWLAPI signature for robustness.
        """
        return list(self.source.entities(EntityKind.CLASS))

    def get_labels_map(self) -> Dict[str, List[str]]:
        """
        Map: class IRI -> list of labels (cached).
        Ensures at least one fallback short form if none found.
        """
        return {iri: self.get_labels(iri) for iri in self.get_all_classes()}

    def get_primary_label(self, iri: str) -> str:
        """Return the first (primary) label for a class IRI (always exists)."""
        return self.get_labels(iri)[0]


class Entity:
    """
    Represents an ontology class. It can use a shared OntologyGraph for efficient,
    cached lookups. If an OntologyGraph is provided, properties like labels,
    subclasses, and superclasses leverage the cached graph; otherwise, they
    use the reasoner and ontology directly.
    """

    def __init__(
        self,
        class_iri: str,
        ontology: KnowledgeSource,
        reasoner: object | None = None,
        ontology_graph: Optional[OntologyGraph] = None,
    ) -> None:
        self._class_iri = str(class_iri)
        self.source = ontology
        self.ontology = ontology  # compatibility alias
        self.reasoner = reasoner
        self.ontology_graph = ontology_graph
        self._labels: Optional[List[str]] = None
        self._subclasses: Optional[List[str]] = None
        self._superclasses: Optional[List[str]] = None
        self._top_superclass: Optional[str] = None
        self._reasoner_warned = False

    def __repr__(self) -> str:
        return (
            f"Entity(name={self.name}, labels={self.labels}, subclasses={self.subclasses}, "
            f"top_superclass={self.top_superclass}, superclasses={self.superclasses})"
        )

    @property
    def owl_class(self) -> str:
        """Compatibility alias; OWLAPI class objects are no longer exposed."""

        return self._class_iri

    @property
    def name(self) -> str:
        return self.source.short_form(self._class_iri)

    @property
    def class_iri(self) -> str:
        return self._class_iri

    @property
    def labels(self) -> List[str]:
        if self._labels is None:
            if self.ontology_graph is not None:
                # Use the cached labels from the OntologyGraph.
                self._labels = self.ontology_graph.get_labels(self.class_iri)
            else:
                self._labels = self.source.labels(self.class_iri) or [self.name]
        return self._labels

    @property
    def subclasses(self) -> List[str]:
        if self._subclasses is None:
            self._subclasses = [
                (self.source.labels(iri) or [self.source.short_form(iri)])[0]
                for iri in self.source.direct_children(self.class_iri)
            ]
        return self._subclasses

    @property
    def superclasses(self) -> List[str]:
        if self._superclasses is None:
            self._superclasses = [
                (self.source.labels(iri) or [self.source.short_form(iri)])[0]
                for iri in self.source.direct_parents(self.class_iri)
            ]
        return self._superclasses

    @property
    def top_superclass(self) -> List[str]:
        if self._top_superclass is None:
            top_iri = self._find_top_superclass()
            self._top_superclass = (
                self.source.labels(top_iri) or [self.source.short_form(top_iri)]
            )[0]
        return [self._top_superclass]

    def _find_top_superclass(self) -> str:
        current = self.class_iri
        seen = {current}
        while True:
            parents = self.source.direct_parents(current)
            if not parents:
                return current
            current = parents[0]
            if current in seen:
                return current
            seen.add(current)

    @staticmethod
    def _get_owl_class(class_iri: str, ontology: KnowledgeSource) -> str:
        """Deprecated compatibility helper returning the normalized IRI string."""

        return str(class_iri)

    @classmethod
    def _extract_labels(cls, owl_class: str, ontology: KnowledgeSource) -> List[str]:
        iri = str(owl_class)
        return ontology.labels(iri) or [ontology.short_form(iri)]

    @classmethod
    def load_from_list(
        cls,
        class_iris: List[str],
        ontology: KnowledgeSource,
        reasoner: object | None = None,
        ontology_graph: Optional[OntologyGraph] = None,
        load_onto_graph: bool = False,
    ) -> List["Entity"]:
        if reasoner is None and ontology_graph is not None:
            reasoner = ontology_graph.reasoner
        if load_onto_graph and ontology_graph is None:
            ontology_graph = OntologyGraph(ontology, reasoner)
        return [
            cls(class_iri=ci, ontology=ontology, reasoner=reasoner, ontology_graph=ontology_graph)
            for ci in class_iris
        ]

    def get_context_subgraph(
        self, n: int, human_readable: bool = True, **kwargs
    ) -> List[Tuple[Union[str, List[str]], Union[str, List[str]], Union[str, List[str]]]]:
        """
        Returns the subgraph (as a list of triples) for the current entity up to n hops.
        If an OntologyGraph is provided, uses the cached projection (converting IRIs to labels
        if human_readable is True); otherwise, builds a subgraph from immediate reasoner queries
        only considering subclass and superclass relations.

        :param n: Maximum hop distance.
        :param human_readable: If True, output human readable labels for nodes and relations.
        :return: List of triples (src, rel, dst). When human_readable is True, each element is a list of labels.
        """
        if self.ontology_graph is not None:
            return self.ontology_graph.get_context_subgraph(
                self.class_iri, n, human_readable, **kwargs
            )
        local_graph = OntologyGraph(self.source, only_taxonomy=True)
        return local_graph.get_context_subgraph(
            self.class_iri, n, human_readable=human_readable, **kwargs
        )
