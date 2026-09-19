"""Standard PyG graph encoders and shared repair readouts (optional neural import)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, fields
from typing import Any, Iterable, Mapping, cast

import pyowl_core as owl
import torch
from torch import Tensor, nn
from torch_geometric.nn import HGTConv, RGCNConv

from .graph import (
    GraphNode,
    ObservableGraph,
    feature_vector,
    structural_arguments,
    structural_id,
)


def _bucket(value: str, size: int = 256) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), "big") % size


def _mlp(inputs: int, outputs: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, outputs))


@dataclass
class GraphMemory:
    """Per-node contextual rows retained for shared object/candidate attention."""

    graph: ObservableGraph
    rows: Mapping[str, Tensor]
    structures: dict[str, Tensor] = field(default_factory=dict)
    contexts: dict[str, Tensor] = field(default_factory=dict)
    scopes: dict[str, tuple[str, ...]] = field(default_factory=dict)


class RepairModel(nn.Module):
    """HGT, matched R-GCN, or no-graph encoder with identical prediction heads.

    The vocabulary is the finite observed graph memory. No output parameter is
    associated with a particular ontology IRI. Benefits exclude explicit costs.
    """

    def __init__(
        self,
        metadata: tuple[Iterable[str], Iterable[tuple[str, str, str]]],
        *,
        feature_dim: int = 128,
        hidden_dim: int = 128,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        encoder: str = "hgt",
        pairwise: bool = False,
    ) -> None:
        super().__init__()
        if encoder not in {"hgt", "rgcn", "none"}:
            raise ValueError("encoder must be hgt, rgcn, or none")
        if hidden_dim < 1 or heads < 1 or hidden_dim % heads or layers < 0:
            raise ValueError("hidden_dim must be positive and divisible by heads; layers >= 0")
        self.metadata = (tuple(sorted(metadata[0])), tuple(sorted(metadata[1])))
        if not self.metadata[0]:
            raise ValueError("The encoder requires at least one node type")
        self.feature_dim, self.hidden_dim, self.encoder = feature_dim, hidden_dim, encoder
        self.config = {
            "feature_dim": feature_dim,
            "hidden_dim": hidden_dim,
            "layers": layers,
            "heads": heads,
            "dropout": dropout,
            "encoder": encoder,
            "pairwise": pairwise,
        }
        self.input_projection = nn.ModuleDict(
            {kind: nn.Linear(feature_dim, hidden_dim) for kind in self.metadata[0]}
        )
        self.graph_layers = nn.ModuleList()
        for _ in range(layers if encoder != "none" else 0):
            if encoder == "hgt":
                self.graph_layers.append(
                    HGTConv(hidden_dim, hidden_dim, self.metadata, heads=heads)
                )
            else:
                self.graph_layers.append(
                    RGCNConv(hidden_dim, hidden_dim, num_relations=len(self.metadata[1]))
                )
        self.norms = nn.ModuleList(
            nn.ModuleDict({kind: nn.LayerNorm(hidden_dim) for kind in self.metadata[0]})
            for _ in self.graph_layers
        )
        self.feedforwards = nn.ModuleList(
            _mlp(hidden_dim, hidden_dim, hidden_dim * 2) for _ in self.graph_layers
        )
        self.dropout = nn.Dropout(dropout)
        self.role_embeddings = nn.Embedding(256, hidden_dim)
        self.syntax_embeddings = nn.Embedding(256, hidden_dim)
        self.argument_head = _mlp(2 * hidden_dim, hidden_dim, hidden_dim)
        self.composition = _mlp(2 * hidden_dim, hidden_dim, hidden_dim)
        self.target_head = _mlp(2 * hidden_dim, hidden_dim, hidden_dim)
        self.object_attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.context_head = _mlp(2 * hidden_dim, hidden_dim, hidden_dim)
        self.candidate_query = _mlp(2 * hidden_dim, hidden_dim, hidden_dim)
        self.candidate_attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.value_head = _mlp(3 * hidden_dim, 1, hidden_dim)
        self.pair_head = _mlp(3 * hidden_dim, 1, hidden_dim) if pairwise else None
        self.empty_bundle = nn.Parameter(torch.zeros(hidden_dim))
        self.slot_projection = nn.Linear(hidden_dim, hidden_dim)
        self.entity_projection = nn.Linear(hidden_dim, hidden_dim)
        self.proposal_context = _mlp(2 * hidden_dim, hidden_dim, hidden_dim)
        self.profile_projection = nn.Linear(feature_dim, hidden_dim)

    def encode(self, graph: ObservableGraph) -> GraphMemory:
        """Contextualise the observed graph using the chosen standard encoder."""
        device = self.empty_bundle.device
        grouped: dict[str, list[GraphNode]] = {kind: [] for kind in self.metadata[0]}
        for node in graph.nodes:
            if node.kind not in grouped:
                raise ValueError(f"Unregistered node type: {node.kind}")
            grouped[node.kind].append(node)
        x = {
            kind: self.input_projection[kind](
                torch.tensor(
                    [feature_vector(node, self.feature_dim) for node in nodes],
                    dtype=self.empty_bundle.dtype,
                    device=device,
                ).reshape(-1, self.feature_dim)
            )
            for kind, nodes in grouped.items()
        }
        locations = {
            node.node_id: (kind, index)
            for kind, nodes in grouped.items()
            for index, node in enumerate(nodes)
        }
        edge_lists: dict[tuple[str, str, str], list[tuple[int, int]]] = {
            edge: [] for edge in self.metadata[1]
        }
        for src, role, dst in graph.edges:
            src_kind, src_index = locations[src]
            dst_kind, dst_index = locations[dst]
            key = (src_kind, role, dst_kind)
            if key not in edge_lists:
                raise ValueError(f"Unregistered edge type: {key}")
            edge_lists[key].append((src_index, dst_index))
        edge_indices = {
            key: torch.tensor(rows, dtype=torch.long, device=device).reshape(-1, 2).T.contiguous()
            for key, rows in edge_lists.items()
        }
        for layer, norms, feedforward in zip(self.graph_layers, self.norms, self.feedforwards):
            if self.encoder == "hgt":
                changed = layer(x, edge_indices)
            else:
                offsets: dict[str, int] = {}
                offset = 0
                for kind, rows in x.items():
                    offsets[kind], offset = offset, offset + len(rows)
                flat_edges, flat_types = [], []
                for edge_type, (key, index) in enumerate(edge_indices.items()):
                    flat_edges.append(
                        index + torch.tensor([[offsets[key[0]]], [offsets[key[2]]]], device=device)
                    )
                    flat_types.append(
                        torch.full((index.shape[1],), edge_type, dtype=torch.long, device=device)
                    )
                flat = layer(
                    torch.cat(list(x.values())),
                    torch.cat(flat_edges, dim=1),
                    torch.cat(flat_types),
                )
                changed = {
                    kind: flat[offsets[kind] : offsets[kind] + len(rows)]
                    for kind, rows in x.items()
                }
            updated = {}
            for kind, rows in x.items():
                change = changed.get(kind)
                h = rows if change is None else norms[kind](rows + self.dropout(change))
                updated[kind] = norms[kind](h + self.dropout(feedforward(h)))
            x = updated
        return GraphMemory(graph, {key: x[kind][index] for key, (kind, index) in locations.items()})

    def _role(self, name: str) -> Tensor:
        return cast(Tensor, self.role_embeddings.weight[_bucket(name)])

    def encode_structure(self, value: Any, memory: GraphMemory) -> Tensor:
        """Compose complete emitted syntax, preserving roles and symmetric operand order."""
        node_id = structural_id(value)
        if node_id in memory.structures:
            return memory.structures[node_id]
        if isinstance(value, owl.Entity):
            if node_id not in memory.rows:
                raise ValueError("Candidate symbol was not retrieved into observable graph memory")
            return memory.rows[node_id]
        arguments = [
            self.argument_head(torch.cat((self._role(role), self.encode_structure(child, memory))))
            for role, child in structural_arguments(value)
        ]
        aggregate = torch.stack(arguments).sum(0) if arguments else self.empty_bundle
        syntax = self.syntax_embeddings.weight[_bucket(type(value).__name__)]
        for attribute in fields(value):
            scalar = getattr(value, attribute.name)
            if isinstance(scalar, (str, int, float, bool)):
                syntax = syntax + self._role(f"scalar:{attribute.name}:{scalar}")
        result = cast(Tensor, self.composition(torch.cat((syntax, aggregate))))
        memory.structures[node_id] = result
        return result

    @staticmethod
    def _attend(query: Tensor, memory: GraphMemory, ids: Iterable[str], layer: Any) -> Tensor:
        rows = torch.stack([memory.rows[key] for key in sorted(set(ids))]).unsqueeze(0)
        return cast(Tensor, layer(query.reshape(1, 1, -1), rows, rows, need_weights=False)[0][0, 0])

    def object_context(self, object_id: str, memory: GraphMemory) -> Tensor:
        """Read statement syntax and all contextual support using a shared target query."""
        if object_id in memory.contexts:
            return memory.contexts[object_id]
        target = memory.graph.object_lookup[object_id]
        arguments = [
            self.argument_head(torch.cat((self._role(role), memory.rows[dst])))
            for role, dst in memory.graph.outgoing[target]
            if not role.startswith("reverse_") and role != "self"
        ]
        aggregate = torch.stack(arguments).sum(0) if arguments else self.empty_bundle
        query = self.target_head(torch.cat((memory.rows[target], aggregate)))
        memory.scopes[object_id] = memory.graph.context_ids(object_id)
        attended = self._attend(query, memory, memory.scopes[object_id], self.object_attention)
        result = self.object_context_head(query, attended)
        memory.contexts[object_id] = result
        return result

    def object_context_head(self, query: Tensor, attended: Tensor) -> Tensor:
        """Apply the common object readout used for every encoder and object."""
        return cast(Tensor, self.context_head(torch.cat((query, attended))))

    def candidate_embedding(self, candidate: Any, memory: GraphMemory) -> Tensor:
        """Encode every emitted axiom and every active non-vacuity obligation."""
        parts = [
            self.argument_head(torch.cat((self._role("emits"), self.encode_structure(a, memory))))
            for a in sorted(candidate.axioms, key=structural_id)
        ]
        parts.extend(
            self.argument_head(
                torch.cat((self._role("activates"), self.encode_structure(e, memory)))
            )
            for e in sorted(candidate.active_expressions, key=structural_id)
        )
        return torch.stack(parts).sum(0) if parts else self.empty_bundle

    def candidate_value(
        self,
        candidate: Any,
        memory: GraphMemory,
        context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return semantic benefit and complete candidate encoding, without edit costs."""
        context = self.object_context(candidate.object_id, memory) if context is None else context
        emitted = self.candidate_embedding(candidate, memory)
        query = self.candidate_query(torch.cat((context, emitted)))
        ids = set(
            memory.scopes.get(candidate.object_id) or memory.graph.context_ids(candidate.object_id)
        )
        for axiom in (*candidate.axioms, *candidate.active_expressions):
            ids.update(
                structural_id(node) for node in owl.walk(axiom) if isinstance(node, owl.Entity)
            )
        attended = self._attend(query, memory, ids, self.candidate_attention)
        return self.value_head(torch.cat((context, emitted, attended))).squeeze(-1), emitted

    def interaction(self, first: Tensor, second: Tensor, joint_context: Tensor) -> Tensor:
        """Predict one symmetric, frozen pair coefficient for an observable pair set."""
        if self.pair_head is None:
            raise ValueError("Pair interactions were not enabled")
        return cast(
            Tensor,
            self.pair_head(torch.cat((first + second, first * second, joint_context))).squeeze(-1),
        )

    def proposal_logits(
        self,
        context: Tensor,
        choices: Tensor,
        *,
        mixtures: int = 1,
        profile_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute all mixture and Boolean-choice logits before circuit conditioning.

        ``choices`` contains shared syntax/slot/menu embeddings, shape (variables,d).
        Mixture component queries use shared role embeddings, with no ontology output layer.
        """
        if mixtures < 1 or mixtures > 256:
            raise ValueError("mixtures must be in [1,256]")
        if profile_features is not None:
            context = context + self.profile_projection(profile_features)
        queries = torch.stack(
            [
                self.proposal_context(torch.cat((context, self._role(f"mixture:{index}"))))
                for index in range(mixtures)
            ]
        )
        logits = self.slot_projection(queries) @ self.entity_projection(choices).T
        weights = (queries * context).sum(-1) / self.hidden_dim**0.5
        return weights, logits / self.hidden_dim**0.5

    def score_inventory(
        self,
        objects: Iterable[Any],
        memory: GraphMemory,
        *,
        interaction_pairs: Iterable[tuple[int, int]] = (),
    ) -> tuple[tuple[Tensor, ...], dict[tuple[int, int, int, int], Tensor]]:
        """Score one frozen inventory, computing each context/candidate encoding once.

        Pair indices refer to observable, predeclared object pairs. These tensors
        remain differentiable for full-repair losses; detach them before a solve.
        """
        objects = tuple(objects)
        contexts = [self.object_context(obj.object_id, memory) for obj in objects]
        unary, embeddings = [], []
        for obj, context in zip(objects, contexts):
            values = [
                self.candidate_value(candidate, memory, context) for candidate in obj.candidates
            ]
            unary.append(torch.stack([value for value, _ in values]))
            embeddings.append([embedding for _, embedding in values])
        pairs = {}
        seen = set()
        for i, j in interaction_pairs:
            if not 0 <= i < j < len(objects) or (i, j) in seen:
                raise ValueError("Interaction pairs must be unique ordered object indices")
            seen.add((i, j))
            context = (contexts[i] + contexts[j]) / 2
            for a, first in enumerate(embeddings[i]):
                for b, second in enumerate(embeddings[j]):
                    pairs[i, a, j, b] = self.interaction(first, second, context)
        return tuple(unary), pairs


def repair_benefits(
    assignments: Iterable[tuple[int, ...]],
    unary: tuple[Tensor, ...],
    pairs: Mapping[tuple[int, int, int, int], Tensor] | None = None,
) -> Tensor:
    """Sum frozen unary/pair semantic factors for complete assignments, without costs."""
    pairs = pairs or {}
    totals = []
    for assignment in assignments:
        if len(assignment) != len(unary):
            raise ValueError("Assignment must choose exactly one candidate per object")
        terms = [row[choice] for row, choice in zip(unary, assignment)]
        terms.extend(
            value
            for (i, a, j, b), value in pairs.items()
            if assignment[i] == a and assignment[j] == b
        )
        if not terms:
            raise ValueError("Benefit prediction requires at least one object")
        totals.append(torch.stack(terms).sum())
    if not totals:
        raise ValueError("At least one labelled assignment is required")
    return cast(Tensor, torch.stack(totals))
