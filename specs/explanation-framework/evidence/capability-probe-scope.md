# Scope of the retained capability probe

The [archived JSON](../../../docs/archive/native-stack/owl-context-capability-probe.json)
and [Functional Syntax fixture](../../../docs/archive/native-stack/owl-context-capability-probe.ofn)
record a tiny synthetic Python-backend probe; all 12 checks passed. It demonstrates API
feasibility, not installed-release/native/RDFXML parity, full Bio-ML coverage, performance
or reasoner completeness. Load options included offline local-import handling,
`preserve_source_map=True` and `collect_provenance=True`. Names/citation URLs are synthetic.
The archived source identity and local paths are historical provenance only.

B0/B2 must reproduce required capabilities with the [published native stack](../../native-stack.md)
and actual locked inputs. Exact currently locks `pyowl-core==0.2.1` and supports
`>=0.2.1,<0.3`. Record the installed package and artifact identities for new checks;
the archived probe does not establish acceptance for the published release.
