
from typing import List, Optional, Union

from mowl.owlapi import OWLOntology

from org.semanticweb.owlapi.search import EntitySearcher

from org.semanticweb.HermiT import Reasoner
from org.semanticweb.owlapi.model import (
    IRI,
    OWLAnnotationProperty,
    OWLClass,
)



class Entity:
    def __init__(self, class_iri: str, ontology: OWLOntology, reasoner: Reasoner) -> None:

        self._owl_class = self._get_owl_class(class_iri, ontology)
        self.ontology = ontology
        self.reasoner = reasoner
        self._labels: Optional[List[str]] = None
        self._subclasses: Optional[List[str]] = None
        self._superclasses: Optional[List[str]] = None
        self._top_superclass: Optional[str] = None

    def __repr__(self) -> str:
        return f"Entity(name={self.name}, labels={self.labels}, subclasses={self.subclasses}, " \
               f"top_superclass={self.top_superclass}, superclasses={self.superclasses})"
    
    @property
    def owl_class(self) -> OWLClass:
        return self._owl_class

    @property
    def name(self) -> str:
        return self._owl_class.getIRI().getShortForm()
    
    @property
    def class_iri(self) -> str:
        return self._owl_class.getIRI().toString()

    @property
    def labels(self) -> List[str]:
        if self._labels is None:
            self._labels = self._extract_labels(self.owl_class, self.ontology)
        return self._labels

    @property
    def subclasses(self) -> List[str]:
        if self._subclasses is None:
            self._subclasses = [self._extract_labels(subcls.getIRI(), self.ontology)[0] for subcls in self.reasoner.getSubClasses(self.owl_class, True).getFlattened() if not subcls.isOWLNothing()]
        return self._subclasses

    @property
    def superclasses(self) -> List[str]:
        if self._superclasses is None:
            self._superclasses = [self._extract_labels(supercls.getIRI(), self.ontology)[0] for supercls in self.reasoner.getSuperClasses(self.owl_class, True).getFlattened() if not supercls.isOWLThing()]
        return self._superclasses

    @property
    def top_superclass(self) -> List[str]:
        if self._top_superclass is None:
            self._top_superclass = self._extract_labels(self._find_top_superclass(), self.ontology)[0]
        return [self._top_superclass]

    def _find_top_superclass(self) -> str:
        owl_class = self.owl_class
        while True:
            superclasses = self.reasoner.getSuperClasses(owl_class, True).getFlattened()
            valid_superclasses = [supercls for supercls in superclasses if not supercls.isOWLThing()]
            if not valid_superclasses:
                break
            owl_class = valid_superclasses[0]
        return owl_class.getIRI()

    @staticmethod
    def _get_owl_class(class_iri: Union[IRI, str], ontology: OWLOntology) -> OWLClass:
        if isinstance(class_iri, str):
            class_iri = IRI.create(class_iri)
        factory = ontology.getOWLOntologyManager().getOWLDataFactory()
        return factory.getOWLClass(class_iri)
    
    @classmethod
    def _extract_labels(cls, owl_class: Union[OWLClass, str], ontology: OWLOntology) -> List[str]:
        if isinstance(owl_class, str):
            owl_class = cls._get_owl_class(owl_class, ontology)
        label_property = ontology.getOWLOntologyManager().getOWLDataFactory().getRDFSLabel()
        return [str(annotation.getValue().asLiteral().get().getLiteral()) for annotation in EntitySearcher.getAnnotations(owl_class, ontology, label_property) if annotation.getValue().isLiteral()]

    @classmethod
    def load_from_list(cls, class_iris: List[str], ontology: OWLOntology, reasoner: Optional[Reasoner] = None) -> List['Entity']:

        if reasoner is None:
            reasoner = Reasoner.ReasonerFactory().createReasoner(ontology)
        return [cls(class_iri=class_iri, ontology=ontology, reasoner=reasoner) for class_iri in class_iris]
