# Exact-OM 2.0

Exact combines lexical and structural evidence, optional gated Elephant-1
decisions, and fitted ranking or acceptance. It produces complete local rankings
or global alignments with inspectable evidence.

Start with [installation](installation.md) and the [staged workflow](pipeline.md).
The [configuration guide](configuration.md) explains bundled profiles and inputs.

The final workflow separates [decision inference](decisions.md),
[global acceptance](acceptance.md), and [local ranking](ranking.md).
Full and partial policies are distinct; imported-context entities remain context
rather than mapping endpoints in the scoped global adapter.

Use [evaluation](evaluation.md) to score submissions, [artifacts and resume](runs.md)
to manage storage, and the [Python API](api.md) for application integration.

[Exact Explain](local-tool.md) is the local app for comparing candidates, browsing
ontologies, checking evidence and decision traces, and managing prepared bundles.
The participant study, administrative services, ontology repair and historical
experiments are maintained separately from this public package.
