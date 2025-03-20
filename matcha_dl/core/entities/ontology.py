
from typing import List, Tuple, Optional

from mowl.owlapi import OWLOntology, OWLClass, OWLAnnotationProperty, IRI, OWLReasoner
from mowl.owlapi import OWLReasonerFactory, OWLManager

class Entity:
    def __init__(self, class_iri: str, ontology: OWLOntology, reasoner: OWLReasoner) -> None:

        self._owl_class = self._get_owl_class(class_iri)
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
        return self._owlclass.getIRI().getShortForm()
    
    @property
    def class_iri(self) -> str:
        return self._owlclass.getIRI().toString()

    @property
    def labels(self) -> List[str]:
        if self._labels is None:
            self._labels = [annotation.getValue().getLiteral() for annotation in self.owl_class.getAnnotations(self.ontology)
                            if isinstance(annotation.getProperty(), OWLAnnotationProperty) and annotation.getProperty().isLabel()]
        return self._labels

    @property
    def subclasses(self) -> List[str]:
        if self._subclasses is None:
            self._subclasses = [subcls.getIRI().getShortForm() for subcls in self.reasoner.getSubClasses(self.owl_class, True).getFlattened() if not subcls.isOWLNothing()]
        return self._subclasses

    @property
    def superclasses(self) -> List[str]:
        if self._superclasses is None:
            self._superclasses = [supercls.getIRI().getShortForm() for supercls in self.reasoner.getSuperClasses(self.owl_class, True).getFlattened() if not supercls.isOWLThing()]
        return self._superclasses

    @property
    def top_superclass(self) -> List[str]:
        if self._top_superclass is None:
            self._top_superclass = self._find_top_superclass()
        return [self._top_superclass]

    def _find_top_superclass(self) -> str:
        owl_class = self.owl_class
        while True:
            superclasses = self.reasoner.getSuperClasses(owl_class, True).getFlattened()
            valid_superclasses = [supercls for supercls in superclasses if not supercls.isOWLThing()]
            if not valid_superclasses:
                break
            owl_class = valid_superclasses[0]
        return owl_class.getIRI().getShortForm()

    @staticmethod
    def _get_owl_class(class_iri) -> OWLClass:
        data_factory = OWLManager.getOWLDataFactory()
        return data_factory.getOWLClass(IRI.create(class_iri))

    @classmethod
    def load_from_list(cls, class_iris: List[str], ontology: OWLOntology) -> List['Entity']:
        reasoner = OWLReasonerFactory().createReasoner(ontology)
        return [cls(class_iri=class_iri, ontology=ontology, reasoner=reasoner) for class_iri in class_iris]
