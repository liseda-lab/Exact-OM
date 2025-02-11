# Adapted from https://github.com/KRR-Oxford/DeepOnto

from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple

from mowl.owlapi import OWLOntology
from org.semanticweb.owlapi.model import (
    IRI,
    OWLAnnotationProperty,
    OWLClass,
    OWLLiteral,
)

from matcha_dl.core.entities.mappings import EntityMapping, ReferenceMapping
from matcha_dl.core.values import ANNOTATION_IRI
from matcha_dl.impl.utils.data import read_table


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
    
    @staticmethod
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
    
    @staticmethod
    def read_candidate_mappings(cand_maps_file: str) -> List[Tuple[ReferenceMapping, List[EntityMapping]]]:
        r"""Load scored or already ranked candidate mappings.

        The predicted candidate mappings are formatted the same as `test.cands.tsv`, with three columns:
        `"SrcEntity"`, `"TgtEntity"`, and `"TgtCandidates"`, indicating the source reference class IRI, the
        target reference class IRI, and a list of **tuples** in the form of `(target_candidate_class_IRI, score)` where
        `score` is optional if the candidate mappings have been ranked. For the Bio-LLM special sub-track, `"TgtCandidates"`
        refers to a list of **triples** in the form of `(target_candidate_class_IRI, score, answer)` where the `answer` is
        required for computing matching scores.

        This method loads the candidate mappings in this format and parse them into the inputs of [`mean_reciprocal_rank`][deeponto.align.evaluation.AlignmentEvaluator.mean_reciprocal_rank]
        and [`hits_at_K`][[`mean_reciprocal_rank`][deeponto.align.evaluation.AlignmentEvaluator.hits_at_K].
        
        """

        all_cand_maps = read_table(cand_maps_file).values.tolist()
        cands = []

        for src_ref_class, tgt_ref_class, tgt_cands in all_cand_maps:
            ref_map = ReferenceMapping(src_ref_class, tgt_ref_class, "=")
            tgt_cands = eval(tgt_cands)
            has_score = True if all([not isinstance(x, str) for x in tgt_cands]) else False
            cand_maps = []
            if has_score:
                cand_maps = [EntityMapping(src_ref_class, t, "=", s) for t, s in tgt_cands]
            else:
                cand_maps = [
                    EntityMapping(src_ref_class, t, "=", (len(tgt_cands) - i) / len(tgt_cands))
                    for i, t in enumerate(tgt_cands)
                ]
            cand_maps = EntityMapping.sort_entity_mappings_by_score(cand_maps)
            cands.append((ref_map, cand_maps))

        return cands
    
    @classmethod
    def ranking_result_file_check(cls, cand_maps_file: str, ref_cand_maps_file: str) -> None:
        r"""Check if the ranking result file is formatted correctly as the original
        `test.cands.tsv` file provided in the dataset.
        """
        formatted_cand_maps = cls.read_candidate_mappings(cand_maps_file)
        formatted_ref_cand_maps = cls.read_candidate_mappings(ref_cand_maps_file)
        assert len(formatted_cand_maps) == len(
            formatted_ref_cand_maps
        ), f"Mismatched number of reference mappings: {len(formatted_cand_maps)}; should be {len(formatted_ref_cand_maps)}."
        for i in range(len(formatted_cand_maps)):
            anchor, cands = formatted_cand_maps[i]
            ref_anchor, ref_cands = formatted_ref_cand_maps[i]
            assert (
                anchor.to_tuple() == ref_anchor.to_tuple()
            ), f"Mismatched reference mapping: {anchor}; should be {ref_anchor}."
            cands = [c.to_tuple() for c in cands]
            ref_cands = [rc.to_tuple() for rc in ref_cands]
            assert not (
                set(cands) - set(ref_cands)
            ), f"Mismatch set of candidate mappings for the reference mapping: {anchor}."
    
