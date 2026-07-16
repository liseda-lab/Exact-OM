# LLM configuration

LLM use is optional. Exact's lexical and structural channels work without a hosted key or a
local generative model.

`llm.profiles` defines backends; `llm.routing` assigns a profile to verbalisation, summary,
decision, and rationale tasks. Keep task routing explicit so audit records identify the model
that produced each value.

```yaml
llm:
  profiles:
    local:
      backend: local_hf
      model: your-local-model
    hosted:
      backend: openrouter
      model: provider/model-name
  routing:
    summary_profile: local
    decision_profile: hosted
    rationale_profile: local
```

Hosted profiles read `OPENROUTER_API_KEY`, then an explicitly configured key file, then the
user configuration path. Never store a key in YAML, logs, a run manifest, or version control.

Decision routing is capability-gated: a hosted model must expose usable binary chat log
probabilities. Exact uses the configured local fallback when that signal is unavailable.
Summary and rationale calls record cache/fallback/backend metadata independently.

Disable all generative work for hermetic or CPU smoke tests in the primary pipeline model:

```yaml
pipeline:
  - name: PairAdaptiveSemanticScorer
    params:
      use_llm: false
      generate_llm_rationales: false
```

Use `exact-llm-debug` to inspect resolved profiles and task routes without running a full
alignment. Redact keys and prompts when sharing debug output.
