from dataclasses import replace

import pytest
import pyowl_core as owl

from exact.repair import kernel, detection
from exact.repair.records import PolicyV2, RepairInputV2


def test_omitting_rejection_acceleration_still_requires_full_native_policy(monkeypatch):
    pytest.importorskip("pyhermit")
    a = owl.Class(owl.IRI("urn:packet-precheck:A"))
    b = owl.Class(owl.IRI("urn:packet-precheck:B"))
    ab = owl.SubClassOf(a, b)
    problem = RepairInputV2((ab,), (), PolicyV2((a, b), required=(ab,)))
    # The rejection accelerator must not run, but required/native obligations
    # must still be complete and actually reject a prohibited consequence.
    monkeypatch.setattr(detection, "detect_violations", lambda *a: pytest.fail("precheck"))
    good = kernel.verify_assignment(problem, (), rejection_precheck=False)
    assert good.authorizes and good.obligations
    assert all(o.complete and o.verdict == "pass" for o in good.obligations)
    bad = replace(problem, policy=PolicyV2((a, b), prohibited=(ab,)))
    assert not kernel.verify_assignment(bad, (), rejection_precheck=False).authorizes
    inconsistent = replace(problem, fixed_axioms=(ab, owl.SubClassOf(a, owl.OWL_NOTHING)))
    assert not kernel.verify_assignment(inconsistent, (), rejection_precheck=False).authorizes


def test_invalid_precheck_option_cannot_disable_checks():
    with pytest.raises(ValueError, match="boolean"):
        kernel.verify_assignment(RepairInputV2((), (), PolicyV2()), (), rejection_precheck="false")


def test_exhausted_optional_detection_still_runs_native_acceptance_and_rejection(monkeypatch):
    a, b = (owl.Class(owl.IRI("urn:bounded-precheck:" + name)) for name in ("A", "B"))
    ab = owl.SubClassOf(a, b)
    real = detection.detect_violations
    monkeypatch.setattr(detection, "detect_violations", lambda *args: real(*args, max_work=1))
    good = RepairInputV2((ab,), (), PolicyV2((a, b), required=(ab,)))
    assert kernel.verify_assignment(good, ()).authorizes
    bad = replace(good, policy=PolicyV2((a, b), prohibited=(ab,)))
    result = kernel.verify_assignment(bad, ())
    assert any(o.verdict == "fail" for o in result.obligations)
    assert not result.authorizes


def test_large_acyclic_closure_is_bounded_within_one_context():
    classes = [owl.Class(owl.IRI(f"urn:bounded-precheck:{i}")) for i in range(100)]
    axioms = tuple(owl.SubClassOf(a, b) for a, b in zip(classes, classes[1:]))
    completion = detection._Completion(axioms, max_contexts=512, max_depth=16, max_work=200)
    completion.complete({classes[0]: frozenset()})
    assert completion.truncated and completion.work <= 200
    assert not completion.cache  # incomplete closure is never a reusable complete result
