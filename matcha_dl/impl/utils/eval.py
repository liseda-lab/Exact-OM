
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, DefaultDict

from matcha_dl.core.entities.mappings import EntityMapping, ReferenceMapping
from matcha_dl.core.values import ANNOTATION_IRI

from mowl.owlapi import OWLOntology
from org.semanticweb.owlapi.model import OWLAnnotationProperty, OWLClass, OWLLiteral, IRI

class MetricUtils:
    @staticmethod
    def compute_intersection_and_union(
        preds: List[EntityMapping],
        refs: List[ReferenceMapping],
        null_refs: Optional[List[ReferenceMapping]] = None
    ) -> Tuple[set, set]:
        preds_set = {p.to_tuple() for p in preds}
        refs_set = {r.to_tuple() for r in refs}
        null_refs_set = {n.to_tuple() for n in null_refs or []}
        
        # Remove null references
        preds_set -= null_refs_set
        refs_set -= null_refs_set
        
        return preds_set, refs_set
    
    def get_ignored_class_index(
        ontology: OWLOntology,
        annotation_iri: str = ANNOTATION_IRI
    ) -> DefaultDict[str, bool]:
        """
        Get an index for filtering classes that are marked as not used in alignment.

        This is indicated by the special class annotation `use_in_alignment` with the provided IRI.

        Parameters:
            ontology (OWLOntology): The ontology object to process.
            annotation_iri (str): The IRI of the annotation indicating usage in alignment. Defaults to
                                "http://oaei.ontologymatching.org/bio-ml/ann/use_in_alignment".

        Returns:
            DefaultDict[str, bool]: A dictionary mapping class IRIs to a boolean indicating if they are ignored.
        """
        ignored_class_index: DefaultDict[str, bool] = defaultdict(lambda: False)

        # Create an OWLAnnotationProperty for the given IRI
        factory = ontology.getOWLOntologyManager().getOWLDataFactory()
        annotation_property = factory.getOWLAnnotationProperty(IRI.create(annotation_iri))

        # Iterate through all classes in the ontology
        for owl_class in ontology.getClassesInSignature():
            # Get annotations for the class
            for axiom in ontology.getAnnotationAssertionAxioms(owl_class.getIRI()):
                if axiom.getProperty() == annotation_property:
                    value = axiom.getValue()
                    if isinstance(value, OWLLiteral) and value.getLiteral().lower() == "false":
                        ignored_class_index[owl_class.getIRI().toString()] = True

        return ignored_class_index

    @staticmethod
    def remove_ignored_mappings(
        mappings: List[EntityMapping],
        ignored_class_index: Dict[str, bool]
    ) -> List[EntityMapping]:
        """
        Filter prediction mappings to remove ignored classes.

        Parameters:
            mappings (List[EntityMapping]): List of predicted mappings.
            ignored_class_index (Dict[str, bool]): Dictionary indicating ignored classes.

        Returns:
            List[EntityMapping]: Filtered list of mappings excluding ignored classes.
        """
        return [
            mapping for mapping in mappings
            if not (ignored_class_index[mapping.head] or ignored_class_index[mapping.tail])
        ]