"""Real-data development review is a versioned purpose, never a launch approval."""

from typing import Literal

from pydantic import Field, model_validator

from ..review_metadata import ReviewBundleSummary
from .models import Identifier, StrictModel
from .v2_models import StudyDefinitionV2


class ReviewProvenance(StrictModel):
    bundle_version: Literal["exact-development-review/1"] = "exact-development-review/1"
    run_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    selection_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_run_id: Identifier
    limitation: str = Field(min_length=1, max_length=4000)


class DevelopmentReviewDefinition(StudyDefinitionV2):
    purpose: Literal["development_review"]
    synthetic: Literal[False] = False
    correctness_metrics: Literal["disabled"] = "disabled"
    review_provenance: ReviewProvenance
    review_bundle: ReviewBundleSummary | None = None

    @model_validator(mode="after")
    def review_contract(self):
        if self.form_version != "exact-study-forms/3":
            raise ValueError("Development review requires explicit forms/3 negotiation")
        if self.launch_approvals:
            raise ValueError("Development review is not research launch approval")
        if self.review_bundle:
            about = self.review_bundle
            if about.cases and [c.model_dump() for c in about.cases] != [
                {
                    "case_id": c.case_id,
                    "source": c.source.model_dump(),
                    "targets": [t.entity.model_dump() for t in c.candidates],
                }
                for c in self.cases
            ]:
                raise ValueError(
                    "Review catalogue case identities differ from the frozen publication"
                )
            if (
                about.source_count != len(self.cases)
                or about.presented_pairs != sum(len(case.candidates) for case in self.cases)
                or about.policy_sha256.removeprefix("sha256:") != self.policy_hash
                or about.run_manifest_sha256.removeprefix("sha256:")
                != self.review_provenance.run_manifest_sha256
                or about.selection_sha256.removeprefix("sha256:")
                != self.review_provenance.selection_manifest_sha256
                or about.source_run_id != self.review_provenance.source_run_id
            ):
                raise ValueError("Review catalogue differs from the frozen publication")
        return self
