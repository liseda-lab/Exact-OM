"""Positive and negative conformance checks for the XR-2 protocol."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from validate_protocol import resolve, validate

ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.pilot = resolve(ROOT / "protocol/pilot.json")

    def test_current_protocol_counts(self):
        self.assertEqual(validate(self.pilot)["requested_generated_total"], 576)
        smoke = validate(resolve(ROOT / "protocol/smoke.json"))
        self.assertEqual(smoke["requested_generated_total"], 96)

    def test_unknown_and_obsolete_fields(self):
        for path, value in [
            (("schema_version",), "xr-p1-v1"),
            (("graph", "new_unrecognised_flag"), True),
            (("policy", "unknown_authorizes_acceptance"), True),
            (("policy", "all_off_required"), True),
            (("solver", "pending_in_upper_bound"), False),
            (("solver", "pending_are_logical_cuts"), True),
            (("training", "held_out_supervision"), True),
            (("data", "evidence", "latent_corruption_features"), True),
            (("circuit", "posterior_weights"), "pi_only"),
            (("circuit", "normaliser_in_training"), False),
            (("teacher", "disjointness_nonvacuity"), "intersection"),
            (("teacher", "exact_distribution_requires_complete_cache"), False),
        ]:
            c = deepcopy(self.pilot)
            dst = c
            for k in path[:-1]:
                dst = dst[k]
            dst[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate(c)

    def test_bad_numbers_and_cross_field_constraints(self):
        for path, value in [
            (("graph", "hidden_width"), 127), (("graph", "attention_heads"), 0),
            (("graph", "dropout"), 1), (("circuit", "components"), True),
            (("circuit", "components"), 0), (("circuit", "compile_seconds"), 1000),
            (("teacher", "temperature"), 0),
            (("preferences", "cost_weights", "human_ontology_edit"), 0),
            (("preferences", "cost_weights", "ontology_edit"), -1),
            (("resources", "campaign_deadline_seconds"), 10),
            (("resources", "run_deadline_seconds"), float("inf")),
            (("training", "seeds"), [13, 13]), (("training", "seeds"), [True]),
            (("training", "patience"), 101),
        ]:
            c = deepcopy(self.pilot)
            dst = c
            for k in path[:-1]:
                dst = dst[k]
            dst[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate(c)

    def test_missing_action_and_cohort(self):
        for path in (("actions", "enabled"), ("data", "cohorts"), ("data", "families")):
            c = deepcopy(self.pilot)
            c[path[0]][path[1]].pop()
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate(c)

    def test_inheritance_cycles_and_duplicate_fields(self):
        with TemporaryDirectory() as directory:
            p = Path(directory) / "a.json"
            p.write_text('{"extends":"a.json"}')
            with self.assertRaises(ValueError):
                resolve(p)
            p.write_text('{"same":1,"same":2}')
            with self.assertRaises(ValueError):
                resolve(p)
            p.write_text('{"number":NaN}')
            with self.assertRaises(ValueError):
                resolve(p)

    def test_arrays_replace_in_inheritance(self):
        smoke = resolve(ROOT / "protocol/smoke.json")
        self.assertEqual(smoke["training"]["seeds"], [13])
        self.assertEqual(smoke["graph"], self.pilot["graph"])


if __name__ == "__main__":
    unittest.main()
