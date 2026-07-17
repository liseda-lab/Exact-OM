# Software bill of materials

Release CI emits `exact-om.spdx.json`, an SPDX 2.3 software bill of materials generated from
the installed base wheel and its reachable dependency metadata. The base graph must contain
`pyowl-core` and `pyowl2vec-star-projector` and must not contain pyELK, pyHermiT,
`py-horned-owl`, mOWL, JPype, OWLAPI, or Java components.

Generate the same artifact for an installed environment with:

```console
python tools/generate_sbom.py --output exact-om.spdx.json
```

Generate a second SBOM after installing `exact-om[reasoning]` when distributing a reasoner
environment. Optional native wheels may add platform-specific components; preserve their own
license and SBOM material alongside Exact's artifact. The run-level `ontology_stack` block is
runtime semantic provenance and complements, but does not replace, the distribution SBOM.
