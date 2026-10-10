"""Copy only exact typed pair evidence from a physically policy-filtered decision package."""

from exact_inspect.contracts import canonical_hash
from exact_inspect.evidence_metadata import PairEvidenceMetadata


def frozen_case_pair_inputs(run, source, candidates):
    evidence, metadata = {}, {}
    for candidate in candidates:
        target = candidate["entity"]
        pair_id = canonical_hash(
            {"run_id": run.manifest()["run_id"], "source": source, "target": target}
        )
        trace = run.pair(pair_id)
        if trace["source"] != source or trace["target"] != target:
            raise ValueError("Prepared evidence belongs to another typed pair")
        value = PairEvidenceMetadata.model_validate(trace["evidence_metadata"])
        if value.scope != "prepared_visibility_policy":
            raise ValueError("Study pair evidence requires the policy-exported decision package")
        identity = candidate["candidate_id"]
        evidence[identity] = run.evidence(pair_id)
        metadata[identity] = {
            "source": trace["source"],
            "target": trace["target"],
            "evidence_metadata": value.model_dump(mode="json"),
        }
    return evidence, metadata
