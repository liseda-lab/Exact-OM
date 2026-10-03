# Approved public-input amendment — 2026-10-03

The user approved this amendment after reviewing E14 availability. Preparation and
implementation are agent-owned; no manual annotation, new collected dataset or private
test reference is permitted. Existing completed results and frozen running attempts remain
unchanged. This record supersedes the optional-input blockers in the September 29 planning
snapshot; it does not revise its historical completion counts.

| Branch | Approved implementation and claim | Execution gate |
| --- | --- | --- |
| E08 signed identifiers | Defer only the signed/exclusive identifier follow-up. Public xrefs do not establish exclusive identifiers, and forbidden external reference-building resources cannot supply independent evidence. The three completed main E08 arms remain valid. | No further run; explicit deferred disposition, not a negative result. |
| E24 asymmetric difference | Both fixed `<` and `>` hypotheses on existing public T0 train/validation known pairs; shared production evidence/support matrix, reversed-input check, ties/missing evidence reported, labels used only for reporting. | Checksum-bound recipe; 300 source groups per split, seed 17; diagnostic only, no setting selection or end-to-end F1 claim. |
| E23 natural KG | OpenEA v2.0 `D_W_15K_V1`, official `721_5fold/1` train/valid. A separate `K0_OpenEA` case preserves the eight-feature supervised graph learner and its four controls. Keep StarWars K0 unchanged for other experiments. | Its own frozen label-free retrieval pools; at most 2,000 training sources and 300 validation sources, seed 17. No private/test/combined alignment file enters preparation. |
| E14 native bridge | Exact recovered NCIT 26.04d and DOID commit `a3447b9f2464872d7f584baca1e7d55663528718`, with verified offline imports. Compare the existing named hierarchy view and full native OWL on the same public validation pairs and training-equivalence anchors. | Full original-ontology admission with repaired native packages before dispatch; fail closed on missing imports, unsupported mapping or resource failure. No Python ontology/reasoner fallback. |

E23 negatives are **benchmark-bijection-derived nonmatches**, only between endpoints in
different official training pairs. Other candidate pairs remain unknown and are excluded
from supervised graph fitting. This does not establish natural NIL, verified real-world
negatives, or a general one-to-one requirement for Exact-OM. Candidate retrieval receives
no alignment labels; labels are added to its immutable training output afterwards. The
OpenEA adapter records masked-ID/name sparsity and any omitted empty attribute values.
See [E23 OpenEA preparation](E23-OpenEA-preparation.md) for the pinned archive and commands.

E14 is a **known-pair relation diagnostic** alongside the existing main E14 end-to-end
CSV experiments. Its graph and native paths receive one identical frozen anchor inventory:
all public training equivalences, never validation relation labels or inferred extra anchors.
Both exclude the queried bridge. Full OWL contains information absent from the CSV hierarchy,
so disagreements are not automatically a parity failure. Native endpoint satisfiability is
checked before reporting positive entailment; this is not full ontology coherence certification.
Confidence magnitudes are not compared across the two methods. No diagnostic winner changes
production settings or the final study automatically.

The native bridge compiles each distinct leave-query-out anchor world once, retains only one
reasoner at a time, and checkpoints each query with document/import, algorithm and installed
native-wheel identities. Operational failures (memory exhaustion or backend faults) fail the
worker rather than becoming scientific abstentions. The approved E14 recipe has no wall-clock
timeout; its explicit native memory allowance remains subordinate to available node resources.
An explicit native compilation work allowance covers every compiler phase; structural and
overflow checks remain enforced. Defaults for all existing experiments are unchanged.
Native HermiT profile validation retains its logical checks, threads the configured memory
allowance through compilation, and releases temporary accounting when temporary buffers are
freed. Native imported-declaration support must preserve strict RDF mapping and cache identity.

New diagnostics use the existing supervised numeric Slurm-step launcher, storage guard and
cumulative accounting lineage. They emit `selection_eligible: false`, incur no hosted calls,
and keep rationales off. The GPU lane remains serialized. The E14 candidate environment is
isolated; never replace packages inside the environment of an active experiment.

Full validation/queue receipts belong under the ignored `data/experiments-v2` tree. A blocked
native admission is not readiness: retain its detailed failure and repair it. The comparison
descriptor depends on a completed admission whose original inputs, imports and installed native
packages match the comparison. Admission checkpoints each successfully validated ontology,
so a target failure does not repeat a compatible successful source check. The supervisor may prepare and run E23's own pool and comparison under this
approval; it does not need another decision about the public dataset substitution.
