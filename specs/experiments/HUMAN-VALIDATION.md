# Human auditability: separate validation protocol

The unattended node campaign tests decision/evidence integrity and may prepare study artifacts.
It cannot establish human benefit by replacing participants with an LLM evaluator. Recruitment,
consent/ethics where applicable, and participant availability have their own schedule; they are
not implied to fit the 2–3 week compute budget.

Prepare a frozen, counterbalanced case pack with source and target context and known correct,
incorrect and unresolved cases. Include confidently incorrect system choices, close alternatives,
missing-evidence cases and qualifier differences. Keep the final-study cases separate from cases
used to design the interface or prompts. Do not contact people without user authorization.

Compare the same interface in three conditions: scores/ranking; factual evidence and recorded
contributions; the same evidence plus an OpenRouter-generated comparative rationale. Counterbalance
cases and candidate order across participants, and prevent the same participant's repeated-case
memory from determining the result. Score-blind/order-blind controls should separate readable
evidence from following the model's rank.

Primary outcome: correct validation/error detection on the frozen cases. Secondary: ranking,
time, confidence, automation-induced errors, evidence citation correctness and unsupported prose.
Choose participant/case counts and power assumptions before recruitment/results; report a small
pilot as descriptive. Include an evidence-ID verifier and a factuality audit, but neither is a
substitute for expert judgment.

The resulting claim concerns the aid actually tested. A helpful rationale can be valuable even
if it never changes automatic matching scores. Conversely, fluent explanations or a reproduced
machine ranking do not establish improved independent validation.
