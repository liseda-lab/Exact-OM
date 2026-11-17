# Gated Elephant-1 decisions

`exact decisions` applies the completed Elephant-1 decision stage to existing
candidate evidence. It makes **local model calls only**. Preparing requests and
applying saved answers need no GPU or model installation. The acceptance stage
runs after this stage and must recompute features derived from pair scores.

## Input and routing

Supply UTF-8 JSONL with candidate rows contiguous by `Src`, unique `(Src,Tgt)`
pairs, and finite `S_base` and `U`. Supply a separate file with **every eligible
source IRI**, one per line, including sources without candidates. Do not derive
that universe from a reference alignment: inference does not read labels.

Rows retain arbitrary existing evidence fields. Requested rows additionally need:

```json
{"Src":"urn:source:A","Tgt":"urn:target:B","S_base":0.62,"U":0.45,"selected_labels":{"source":"heart","target":"cardiac organ"},"llm_structured_evidence_packet":"Existing structured pair evidence","llm_evidence_packet":"Existing local pair brief"}
```

The prompt is the unchanged binary equivalence predicate over source and target
labels and the recorded pair brief. Global uses `llm_structured_evidence_packet`;
local uses `llm_evidence_packet`. An embedded `decision_request` from an earlier
export is also accepted, but its payload hash and endpoint identities are checked.
The stage neither invents missing evidence nor substitutes a chat prompt.

Four explicit profiles preserve the experiment contracts:

| `--policy` | Source gate | Exact anchors | Evidence |
|---|---|---|---|
| `global` (default) | Highest-ambiguity 20% of the scoped source universe, shared by TRAIN and inference | Primitive score; ordinary candidate, no bypass | Structured |
| `global-original` | Highest-ambiguity 20% of the original unfiltered universe | Score one in gap; no decision request | Structured |
| `local-application` | Highest-ambiguity 20% of original query-source unions | Score one in gap; no decision request | Local brief |
| `local-train` | All TRAIN sources | Score one in gap; no decision request | Local brief |

Corrected SNOMED–FMA and SNOMED–NCIT used `global`. Historical NCIT–DOID retained
its original global bank, then excluded contributions outside the corrected
scope. Reproducing that result requires `global-original` before scope filtering;
switching it to the corrected gate would be a new experiment. Original local
TRAIN was ungated; local application used 20%.

The exact source ambiguity is `1 - clip(top1 - top2, 0, 1)`, with a missing second
candidate scored zero and an empty source assigned ambiguity zero. Select
`ceil(0.20 * N)` sources, breaking equal ambiguity by source IRI. Within admitted
sources select at most five candidates by descending `S_base`, then target IRI.
For original profiles either mark scored anchors with `exact_anchor: true`
in the input, or supply the native `anchors.json` sidecar with `--anchors`.
The sidecar requires unique `(Src,Tgt)` pairs with an explicit `S_final` or
`Score` of exactly one. It contributes to ambiguity without adding candidate rows.
`original_exact_anchor` is only a historical feature in corrected global and
does **not** bypass normal scoring. Global scope filtering and
recomputation of scope-relative features must happen upstream, while preserving
the primitive evidence scored on the original candidate pool. This stage never fills
missing generated candidates with local benchmark pools.

## Prepare, run, and apply

```console
exact decisions prepare --candidates candidates.jsonl --sources sources.txt --output decisions --policy global
exact decisions run --plan decisions/plan.json --cache elephant.sqlite --model-dir /models/elephant-1
exact decisions apply --candidates candidates.jsonl --plan decisions/plan.json --cache elephant.sqlite --output scored.jsonl --beta 0.8
```

For an entirely cached run, replace `--model-dir` with `--cache-only`. Missing
responses produce an error; there is no provider fallback. The SQLite cache can
be reused across tasks when the exact prompt, model revision and runtime match.
Each answer is committed before continuing. Restart the same `run` command to
resume; completed responses are not recomputed. The cache serializes writers and
checks response checksums. An interruption between model completion and the
transaction commit may repeat that one unfinished local decision.

Plan, implementation, candidate and request hashes prevent accidental resume
against changed inputs or gate code. Changed source evidence, policies or
implementation require a new plan directory; compatible cached probabilities
remain reusable because their identity binds the prompt, model and runtime. Input
and output paths are portable; output files publish atomically. Preparation
streams one source group at a time, retaining source-level summaries. Application
streams candidates and retains only the admitted probability map.

The deployed mixture is **not constant 50%**:

```text
weight = clip(0.8 * U, 0, 1)
S_pair_final = (1 - weight) * S_base + weight * P(equivalent)
```

Pairs without a routed answer retain `S_base`. A response with zero uncertainty
can be consumed while making no effective score contribution. Application
exports `cached_response_reused`, `local_model_call_attempted`,
`live_provider_call_attempted`, `effective_pair_score_contribution` and
`selector_stage_consultation` separately. Application itself performs no model
calls; the cache preserves local inference metadata and `status.json` records
new local calls during inference. No selector-stage consultation occurs here.

## Install the optional decision runtime

The base Exact installation does not install or download Elephant. Use a separate
GPU environment to avoid forcing Elephant's serving dependencies into the CPU
matching environment. The measured runtime is:

- Model: `baselabs/elephant-1`, revision
  `b3f8a93462fb1051a9de031e94a7ff380e254072`.
- OneJev: `https://github.com/OmniJev/OneJev`, commit
  `81ce62f1597c91e46767d4d02d6ac2e18534fe94`, with the model snapshot's `serve.patch`.
- Qualified numerical stack: PyTorch `2.8.0+cu128`, Transformers `5.19.0`,
  and pinned tokenizer, safetensors and linear-attention packages, checked against
  the packaged `runtime.json`.
- `bfloat16`, float32 readout head, sequential branching, noncausal attention,
  `slot_lm` readout, 16,384-token request and branch limits.

Obtain the **complete pinned model snapshot**, including its tokenizer,
`readout.json`, `constraints.txt`, and `serve.patch`. The upstream snapshot's
README and setup instructions specify the serving environment. For a dedicated
installation, the relevant runtime source steps are:

```console
git clone https://github.com/OmniJev/OneJev.git
cd OneJev
git checkout --detach 81ce62f1597c91e46767d4d02d6ac2e18534fe94
git apply /models/elephant-1/serve.patch
```

Install this patched runtime and its pinned dependencies in that GPU environment,
following the model snapshot's `constraints.txt` and CUDA installation guidance.
The upstream setup also installs a benchmark kit; that kit is not required by
Exact. Install the same Exact wheel there with `python -m pip install --no-deps /path/to/exact_om.whl` to expose this stdlib-only decision CLI without changing the
serving environment. Run preparation, fitting and evaluation in the main Exact
environment. Sharing the JSONL plan and SQLite cache requires no Python-object
serialization between environments.

The adapter verifies the pinned model and patched runtime file checksums before
loading weights. This includes a sequential read of the model weights once per
process. Missing or modified content fails explicitly. Models are loaded only
from the given local directory with offline Hugging Face settings. A generic
causal-language-model adapter is incompatible with the qualified decision
readout and is never substituted. Check GPU memory against the roughly 18.8 GB
weight file plus runtime/attention memory; the qualified primary hardware used
32 GB GPUs. Hardware changes need separate numerical qualification.
