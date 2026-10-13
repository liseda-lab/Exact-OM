# Prospective bounded recovery for failed judge requests

The user requested one or two retries when future LLM calls fail on 13 October
2026. This authorizes technical request recovery, not repeated sampling until a
judge gives a preferred answer. Completed DEV observations, masks, training
labels and the six fixed model endpoints remain unchanged.

New annotation plans opt into `tools.repair.judge_retry.policy()` through their
frozen `retry_policy`. `tools.repair.probabilistic_judge.one` is the shared entry
point. Plans without this field retain their original one-shot behavior. This
implementation supports ungated Chat requests; other interfaces are rejected
when the policy is requested.

- At most two additional physical attempts per original presentation, with
  deterministic 5/15-second backoff. Replacement jobs inherit the saved chain.
- Retry HTTP 408/429/5xx, including provider error envelopes inside HTTP 200,
  and confirmed unknown delivery after a timeout or a proven dead local sender.
  Never duplicate an unresolved live or remote sender. An uncertain old charge
  remains fully reserved even if a new attempt succeeds.
- Malformed or truncated output may receive one neutral JSON-format reminder
  within the same two-retry total. Keep all evidence, rubric, model, provider,
  token limits, case identity, orientation and repetition unchanged. The reminder
  supplies neither the prior answer nor a target answer.
- Valid abstentions, ties, disagreements and schema-valid semantic contradictions
  are observations, not reasons to retry. Authentication, identity mismatches,
  permanent request errors and refusals require diagnosis, not blind repetition.
- Preserve every request, raw HTTP response, failure receipt, parent identity
  and cost. A chain supplies at most one logical observation, with its effective
  physical request identified explicitly. It never creates extra independent votes.

Reserve the complete mandatory baseline schedule before any retry. Admit each
additional attempt atomically against the live dollar, total-request, phase and
deadline limits. Never finance retries by dropping baseline comparisons or swaps.
Report costs across all attempts and preserve unknown billing. `cost_summary`
provides aggregation that includes failed attempts rather than only the final
successful response. Exhausted retries produce missingness and do not halt other
eligible cases. They do not consume the supervisor's separate technical-repair
attempt counter.

For the upcoming independent TEST schedule, the original **128 comparisons plus
26 order swaps remain 154 logical presentations**, including the same 43 Conference
slots. A contingency of at most 308 additional physical requests raises TEST's
attempt ceiling from 154 to 462 and the cumulative attempt ceiling from 1076 to
1384. Calibration91/TRAIN251/DEV580, the **$35 cumulative ceiling**, every prior
reservation, models, splits, metrics and production-matcher deferral are unchanged.
The extra attempt capacity is permission, not a spending reservation: use it only
for qualified failures that fit after all baseline calls have been reserved.

Before paid TEST annotation, the supervisor must bind this amendment, the updated
phase-state snapshot and policy into the new annotation plan; preflight every
mandatory packet; use the shared retry entry point; and verify its independent
Gemini parser with offline controls. The old independent-assessor protocol and
55 native TEST jobs remain immutable historical bindings. Apply this prospective
transport overlay when freezing the annotation batch. Preserve the original
TEST opening time unless a separately recorded user amendment changes it.

Current DEV evidence supports proceeding to the independent held-out comparison,
without more training or a perfect-judge gate: 192/192 logical outputs verified,
110/111 valid semantic responses, 28/32 numerically usable comparisons. The paired
LLM-supervised minus symbolic-supervised HGT effect is -0.0208 on a 0–1 scale;
the descriptive parent bootstrap interval [-0.1667, 0.1042] includes zero.
The small authored DEV sample does not establish a benefit from LLM supervision.

Offline validation: 82 focused tests passed, including the observed embedded-502
case, timeout/replacement recovery, preserved unknown charges, format reminders,
non-retryable opinions, exhausted attempt limits, whole-panel admission and TEST
phase/dollar/deadline limits. Targeted mypy passed for all four changed modules.
No hosted calls were made for this correction. The campaign records retain the
initial failed check launcher and both successful validation steps and their
Slurm resource usage.
