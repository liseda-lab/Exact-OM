"""Public-safe metadata shared by exploration and the provisional study."""

from typing import Literal
from pydantic import Field
from .contracts import WireModel, EntityRef


class ReviewOntology(WireModel):
    ontology_version_id: str
    name: str
    context_sha256: str
    root_sha256: str
    scope: str
    imports_complete: bool
    entity_count: int = Field(ge=0)
    axiom_count: int = Field(ge=0)
    inference_status: str


class ReviewCaseSummary(WireModel):
    case_id: str
    source: EntityRef
    targets: list[EntityRef] = Field(min_length=1, max_length=100)


class ReviewBundleSummary(WireModel):
    schema_version: Literal["exact-review-about/1"] = "exact-review-about/1"
    purpose: Literal["development_review"] = "development_review"
    title: str = Field(max_length=160)
    prepared_at: str
    source_run_id: str
    run_manifest_sha256: str
    selection_sha256: str
    policy_sha256: str
    source_count: int = Field(ge=1)
    presented_pairs: int = Field(ge=1)
    correctness_metrics: Literal["disabled"] = "disabled"
    coverage: dict[str, int]
    cases: list[ReviewCaseSummary] = Field(default_factory=list, max_length=200)
    limitations: list[str] = Field(max_length=20)
    ontologies: list[ReviewOntology] = Field(max_length=20)
