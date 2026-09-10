# E16 donor binding

The campaign's `donor_transfer: {producer: E18}` frozen constant activates
`exact.experiments.transfer.materialize_transfer`. After the E18 decision completes,
it takes the selected arm (or the completed fitted baseline if screened out),
verifies that run's resolved configuration, and discovers the actual fitted
selector plus any required fitted fusion or score calibration. Missing or
ambiguous donor artifacts stop preparation; the adapter never starts a donor fit.

One or two development recipient tasks at seed 17 share the three declared
comparators. The transfer arm uses the donor's scoring recipe, fitted heads and
fixed thresholds. The in-pair arm refits the corresponding heads using only its
recipient training data. The label-free arm clears fitted heads and uses the
selected label-free selector control, shipped analytic fusion and uncalibrated
pair scores where those fitted heads were present.

Each recipient gets a separate immutable application manifest. A small bundle
selects the application by runtime dataset identity. This preserves the existing
single `supervision.transfer_artifact` configuration path across a two-task matrix.
The manifests bind donor head bytes, feature schema, model/evidence settings,
ontology content hashes, recipient source population and donor-training-source
non-overlap. They record shared ontology bytes explicitly; two disjoint pairs do
not establish a shared-ontology transfer result.

Relocating byte-identical recipient inputs creates a fresh application binding
for the legacy runtime path identity. The donor artifact is reused unchanged.
Changing ontology bytes or the frozen recipient population invalidates the
application. The consumer checks configuration compatibility before constructing
the dataset and refuses any recipient fitting.

The bounded implementation supports selector, analytic-fitted/global-learned
fusion and scalar score-calibration transfer. Graph/NIL/relation heads, adaptive
thresholds, adaptive fusion and fitted LLM state are explicitly unsupported donor
recipes. They are not silently stripped from a claimed transferred stack.

`tests/artifact_transfer_test.py` fits a small donor, consumes its completed output
through this adapter, applies it to two recipients, forbids recipient fitting,
checks immutable replay, and rebinds a relocated recipient without changing donor
weights. No actual experiment or hosted request is run by these tests.
