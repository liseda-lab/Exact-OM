# Evaluation

Evaluate saved global or local output independently of inference:

```console
exact eval -a alignment.tsv -f test-reference.tsv -o evaluation -e
exact-eval -a rankings.tsv -c test.cands.tsv -o local-evaluation -e
```

`bioml-eval` is an alias for the Exact evaluation CLI. Install the
`bioml-eval` extra to use the official optional backend:

```console
exact eval --help
exact eval -a alignment.rdf -f test-reference.tsv -o evaluation \
  --eval-backends builtin bioml -e
```

Supply the reference set and protocol intended by the benchmark. A repaired
reference is a different evaluation target from an unrepaired reference;
evaluating against it does not invoke ontology repair.

The study's full global metric counts all eligible submitted mappings after
the official exclusion rules. The DISO-style partial metric counts a submitted
mapping only when its source **or** target appears in the evaluated positive
reference. These metrics have different denominators. Removing exact TRAIN
and VALID reference pairs is not the same as removing all mappings involving
their endpoints.

The acceptance command's validation partial metric is an explicit development
objective. It does not replace official hidden-test evaluation. The final
study archives include independently reconciled full and partial test
accounting, separate policies and unknown-reference coverage. Do not combine
the best task scores selected after looking at TEST results into a new
unreported submission.

For new tasks, fit on TRAIN, qualify on VALID, freeze policy/model artifacts,
then evaluate TEST. Local ranking reports every original candidate occurrence;
global matching reports the selected mapping set.
