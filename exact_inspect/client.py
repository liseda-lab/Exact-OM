"""Small typed fixture client for backend consumers; independent from the deferred frontend."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx

from .contracts import EntityRef, Page
from .models import (
    AxiomResponse,
    Candidate,
    EntityContextResponse,
    Fact,
    GeneratedExplanationResponse,
    HierarchyPage,
    PairResponse,
    SelectedEvidence,
)
from .service import Health


class InspectClient:
    """Explicit synchronous prepared-resource requests with typed response validation."""

    def __init__(self, base_url: str, *, transport: httpx.BaseTransport | None = None):
        self.http = httpx.Client(base_url=base_url, transport=transport, timeout=10)

    def close(self) -> None:
        """Close the client's pooled connections."""
        self.http.close()

    def _get(self, route: str, **params: Any) -> Any:
        response = self.http.get(
            "/api/v1/" + route, params={k: v for k, v in params.items() if v is not None}
        )
        response.raise_for_status()
        return response.json()

    def health(self) -> Health:
        """Read readiness; this never triggers preparation."""
        return Health.model_validate(self._get("health"))

    def entities(
        self,
        ontology_version_id: str,
        *,
        term: str = "",
        limit: int = 20,
        cursor: str | None = None,
    ) -> Page[dict[str, Any]]:
        """Read one bounded entity-search page."""
        return Page[dict[str, Any]].model_validate(
            self._get(
                "entities",
                ontology_version_id=ontology_version_id,
                term=term,
                limit=limit,
                cursor=cursor,
            )
        )

    def context(self, entity: EntityRef) -> EntityContextResponse:
        """Inspect ontology context without a run or candidate binding."""
        return EntityContextResponse.model_validate(
            self._get("entity-context", **entity.model_dump())
        )

    def pair(self, run_id: str, pair_id: str) -> PairResponse:
        """Read only one pair's immutable decision history."""
        return PairResponse.model_validate(
            self._get("runs/" + quote(run_id, safe="") + "/pair", pair_id=pair_id)
        )

    def facts(
        self,
        entity: EntityRef,
        *,
        category: str | None = None,
        basis: str = "asserted",
        limit: int = 20,
        cursor: str | None = None,
    ) -> Page[Fact]:
        """Read original facts with explicit category, interpretation and completeness."""
        return Page[Fact].model_validate(
            self._get(
                "entity-facts",
                **entity.model_dump(),
                category=category,
                basis=basis,
                limit=limit,
                cursor=cursor,
            )
        )

    def hierarchy(
        self,
        entity: EntityRef,
        *,
        direction: str = "parents",
        basis: str = "literal_asserted",
        limit: int = 50,
        cursor: str | None = None,
    ) -> HierarchyPage:
        """Navigate a bounded hierarchy page without changing ontology scope."""
        return HierarchyPage.model_validate(
            self._get(
                "hierarchy",
                **entity.model_dump(),
                direction=direction,
                basis=basis,
                limit=limit,
                cursor=cursor,
            )
        )

    def axiom(self, ontology_version_id: str, axiom_id: str) -> AxiomResponse:
        """Read a typed original axiom; oversized originals expose separate streaming access."""
        return AxiomResponse.model_validate(
            self._get("axioms/" + quote(axiom_id, safe=""), ontology_version_id=ontology_version_id)
        )

    def candidates(
        self,
        run_id: str,
        source: str,
        *,
        source_kind: str = "class",
        limit: int = 20,
        cursor: str | None = None,
    ) -> Page[Candidate]:
        """List typed scores, ordinal ranks and saved membership without loading evidence."""
        return Page[Candidate].model_validate(
            self._get(
                "runs/" + quote(run_id, safe="") + "/candidates",
                source=source,
                source_kind=source_kind,
                limit=limit,
                cursor=cursor,
            )
        )

    def evidence(
        self, run_id: str, pair_id: str, *, limit: int = 20, cursor: str | None = None
    ) -> Page[SelectedEvidence]:
        """Read the selected feature identities and visible original-fact links."""
        return Page[SelectedEvidence].model_validate(
            self._get(
                "runs/" + quote(run_id, safe="") + "/pair-evidence",
                pair_id=pair_id,
                limit=limit,
                cursor=cursor,
            )
        )

    def explanation(self, explanation_id: str) -> GeneratedExplanationResponse:
        """Read prepared text with its generation and grounding status."""
        return GeneratedExplanationResponse.model_validate(
            self._get("explanations/" + quote(explanation_id, safe=""))
        )
