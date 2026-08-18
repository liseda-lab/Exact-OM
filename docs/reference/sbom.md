# Software bill of materials

Release CI generates `exact-om.spdx.json`, an SPDX 2.3 software bill of materials from a
clean installation of the exact tested base wheel. The artifact is uploaded beside the same
wheel and sdist that passed the Python 3.10–3.12 smoke matrix.

The base graph must contain the published `pyowl-core` and
`pyowl2vec-star-projector` 0.2 distributions. It must not contain pyELK, pyHermiT,
`py-horned-owl`, mOWL, DeepOnto, JPype, OWLAPI, or Java components. The wheel metadata must
keep visualization and reasoner packages behind their independent extras.

Generate the same base artifact from an installed release wheel with:

```console
python tools/generate_sbom.py --output exact-om.spdx.json
```

Generate a separate SBOM after installing `exact-om[reasoning]` when distributing a reasoner
environment. Optional native wheels may add platform-specific components; preserve their own
license and SBOM material alongside Exact's artifact.

`release/core-compatibility.json` records the exact published ontology package set and public
schema-2 contract tested for the release. The run-level `ontology_stack` block records
runtime semantic provenance. These records complement the dependency SBOM; none replaces the
others.
