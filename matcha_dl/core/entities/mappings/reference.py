# Adapted from https://github.com/KRR-Oxford/DeepOnto

import pprint
from typing import List, Optional, Union
from pathlib import Path

from matcha_dl.core.contracts.dataset import DataFrame
from matcha_dl.core.entities.mappings import EntityMapping
from matcha_dl.core.values import DEFAULT_REL


class ReferenceMapping(EntityMapping):
    r"""A datastructure for entity mapping that acts as a reference mapping.

    A reference mapppings is a ground truth entity mapping (with $score = 1.0$) and can
    have several entity mappings as candidates. These candidate mappings should have the
    same `head` (i.e., source entity) as the reference mapping.

    Attributes:
        src_entity_iri (str): The IRI of the source entity, usually its IRI if available.
        tgt_entity_iri (str): The IRI of the target entity, usually its IRI if available.
        relation (str, optional): A symbol that represents what semantic relation this mapping stands for. Defaults to `<?rel>` which means unspecified.
            Suggested inputs are `"<EquivalentTo>"` and `"<SubsumedBy>"`.
    """

    def __init__(
        self,
        src_entity_iri: str,
        tgt_entity_iri: str,
        relation: str = DEFAULT_REL,
        candidate_mappings: Optional[List[EntityMapping]] = [],
    ):
        r"""Intialise a reference mapping.

        Args:
            src_entity_iri (str): The IRI of the source entity, usually its IRI if available.
            tgt_entity_iri (str): The IRI of the target entity, usually its IRI if available.
            relation (str, optional): A symbol that represents what semantic relation this mapping stands for. Defaults to `<?rel>` which means unspecified.
                Suggested inputs are `"<EquivalentTo>"` and `"<SubsumedBy>"`.
            candidate_mappings (List[EntityMapping], optional): A list of entity mappings that are candidates for this reference mapping. Defaults to `[]`.
        """
        super().__init__(src_entity_iri, tgt_entity_iri, relation, 1.0)
        self.candidates = []
        for candidate in candidate_mappings:
            self.add_candidate(candidate)

    def __repr__(self):
        reference_mapping_str = f"ReferenceMapping({self.head} {self.relation} {self.tail}, 1.0)"
        if self.candidates:
            candidate_mapping_str = pprint.pformat(self.candidates)
            reference_mapping_str += f" with candidates:\n{candidate_mapping_str}"
        return reference_mapping_str

    def add_candidate(self, candidate_mapping: EntityMapping):
        """Add a candidate mapping whose relation and head entity are the
        same as the reference mapping's.
        """
        if self.relation != candidate_mapping.relation:
            raise ValueError(
                f"Expect relation of candidate mapping to be {self.relation} but got {candidate_mapping.relation}"
            )
        if self.head != candidate_mapping.head:
            raise ValueError("Candidate mapping does not have the same head entity as the anchor mapping.")
        self.candidates.append(candidate_mapping)

    @staticmethod
    def read_table_mappings(table_of_mappings_file: Union[Path, DataFrame], relation: str = DEFAULT_REL):
        r"""Read reference mappings from `.csv` or `.tsv` files.
        
        !!! note "Mapping Table Format"
        
            The columns of the mapping table must have the headings: `"SrcEntity"`, `"TgtEntity"`, and `"Score"`.

        Args:
            table_of_mappings_file (str): The path to the table (`.csv` or `.tsv`) of mappings.
            relation (str, optional): A symbol that represents what semantic relation this mapping stands for. Defaults to `<?rel>` which means unspecified.
                Suggested inputs are `"<EquivalentTo>"` and `"<SubsumedBy>"`.

        Returns:
            (List[ReferenceMapping]): A list of reference mappings loaded from the table file.
        """
        return EntityMapping.read_table_mappings(table_of_mappings_file, relation=relation, is_reference=True)