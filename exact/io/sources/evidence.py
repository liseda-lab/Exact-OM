"""Backend-neutral asserted property evidence shared by OWL and normalized CSV."""

import hashlib


def property_schema_evidence(source):
    existing = getattr(source, "normalized_property_axioms", None)
    if existing is not None:
        return list(existing)
    snapshot = getattr(source, "owl_snapshot", None)
    if not callable(snapshot):
        return []
    rows = []
    for axiom in snapshot().iter_axioms():
        name = type(axiom).__name__
        axiom_id = hashlib.sha256(repr(axiom).encode()).hexdigest()
        if name == "InverseObjectProperties":
            for prop, inverse in ((axiom.first, axiom.second), (axiom.second, axiom.first)):
                if hasattr(prop, "iri") and hasattr(inverse, "iri"):
                    rows.append(
                        {
                            "subject_iri": prop.iri.value,
                            "object_iri": inverse.iri.value,
                            "rel_iri": "http://www.w3.org/2002/07/owl#inverseOf",
                            "relation_label": "inverseOf",
                            "object_label": None,
                            "evidence_group": "signature",
                            "axiom_id": axiom_id,
                        }
                    )
        elif name in {
            "FunctionalObjectProperty",
            "InverseFunctionalObjectProperty",
            "FunctionalDataProperty",
            "SymmetricObjectProperty",
            "AsymmetricObjectProperty",
            "TransitiveObjectProperty",
            "ReflexiveObjectProperty",
            "IrreflexiveObjectProperty",
        }:
            rows.append(
                {
                    "subject_iri": axiom.property.iri.value,
                    "object_iri": "http://www.w3.org/2002/07/owl#"
                    + name.replace("ObjectProperty", "Property").replace(
                        "DataProperty", "Property"
                    ),
                    "rel_iri": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                    "relation_label": "characteristic",
                    "object_label": name,
                    "evidence_group": "characteristics",
                    "axiom_id": axiom_id,
                }
            )
    return sorted(rows, key=lambda row: (row["subject_iri"], row["rel_iri"], row["object_iri"]))
