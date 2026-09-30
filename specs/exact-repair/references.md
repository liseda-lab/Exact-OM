# Primary references used by XR-2.1

These sources motivate components; the revised combination and its empirical advantages remain a research proposal. References are not proof that the repository implements the cited method. The source-bound pilot and external report evidence are described in [11](11-benchmark-evidence.md).

- W3C, [OWL 2 Primer](https://www.w3.org/TR/owl2-primer/), [Direct Semantics](https://www.w3.org/TR/owl2-direct-semantics/) and [Profiles](https://www.w3.org/TR/owl2-profiles/): axiom and expression semantics, consistency, profiles and query scope.
- Baader et al. (2018), [Gentle Repair of Description Logic Ontologies through Axiom Weakening](https://arxiv.org/abs/1808.00248): principled weakening. Not every endpoint substitution or added necessary condition is a weakening.
- Troquard et al. (2018), [Repairing Ontologies via Axiom Weakening](https://arxiv.org/abs/1711.03430): fine-grained ontology revision.
- Jiménez-Ruiz and Cuenca Grau (2011), [LogMap: Logic-based and Scalable Ontology Matching](https://www.cs.ox.ac.uk/isg/projects/LogMap/papers/paper_ISWC2011.pdf): scalable incomplete conflict detection. A LogMap-style detector is not a complete OWL verifier.
- Kazakov et al. (2014), [The Incredible ELK](https://doi.org/10.1007/s10817-013-9296-3): scalable reasoning for the supported EL fragment.
- Glimm et al. (2014), [HermiT](https://doi.org/10.1007/s10817-014-9305-1): expressive OWL reasoning; no guarantee of meeting our deadlines.
- Hu et al. (2020), [Heterogeneous Graph Transformer](https://arxiv.org/abs/2003.01332): typed attention backbone.
- Schlichtkrull et al. (2018), [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103): relational convolution control.
- Ahmed, Teso, Chang, Van den Broeck and Vergari (2022), [Semantic Probabilistic Layers for Neuro-Symbolic Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c182ec594f38926b7fcb827635b9a8f4-Abstract-Conference.html): neural distributions constrained by logical formulae; applied here to bounded candidate encodings.
- Darwiche and Marquis (2002), [A Knowledge Compilation Map](https://www.cs.cmu.edu/afs/cs.cmu.edu/project/jair/pub/volume17/darwiche02a.pdf): representation properties and tractable circuit operations.
- Kisa et al. (2014), [Probabilistic Sentential Decision Diagrams](https://web.cs.ucla.edu/~guyvdb/papers/KisaKR14.pdf): an alternative structured distribution over a logical support. The specified initial generator uses conditioned mixtures of product distributions, not a claim to implement every PSDD learning method.
- [PySAT RC2 documentation](https://pysathq.github.io/docs/html/api/examples/rc2.html): weighted MaxSAT backend candidate, with explicit optimality status and finite-pool scope.
- He et al. (2022), [Bio-ML](https://arxiv.org/abs/2205.03447): benchmark motivation; annual release counts are kept separately in [11](11-benchmark-evidence.md).
- [OAEI Conference 2025](https://oaei.ontologymatching.org/2025/conference/index.html) and [evaluation](https://oaei.ontologymatching.org/2025/results/conference/index.html): real alignment and logical evaluation evidence.

The shared-core Java-free architecture remains a repository constraint. Citations to Java reasoners describe optional qualified comparison adapters, not a requirement to add a Java dependency to production Exact-OM.


## Sources added for the implementation revision

- PySDD, [SddManager](https://pysdd.readthedocs.io/en/latest/classes/SddManager.html), [Vtree](https://pysdd.readthedocs.io/en/latest/classes/Vtree.html) and [formula construction example](https://pysdd.readthedocs.io/en/latest/examples/build_formula.html): public native ownership, collection, allocation and variable-tree interfaces. Pin actual installed library/API versions in compiler tests; the documentation version label is not a runtime version guarantee.
- Goodman (1999), [Semiring Parsing](https://aclanthology.org/J99-4004/): background for a possible bounded-grammar sum-product alternative. This is a fallback research comparison, not the first implementation or a proof that arbitrary semantic constraints admit a compact chart.
- Zheng et al. (2023), [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685): motivates controls for position, presentation and judge biases. Its chat-evaluation findings do not validate OWL repair judgements; XR-E09 must test those directly.
- Lee et al. (2023), [RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267): precedent for treating AI feedback as supervision. XR-2.1 uses supervised semantic-fidelity learning and keeps MaxSAT; it does not adopt an RL training objective from this citation.

The existing OpenRouter implementation (`exact/llm/routing.py`, `exact/llm/ledger.py`) is the integration source of truth. Reuse its public client, profile, request identity and accounting contracts; provider/model availability is resolved for each future run rather than inferred from a paper or a historical model name.
