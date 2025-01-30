# Adapted from https://github.com/KRR-Oxford/DeepOnto

from typing import List, Optional, Union
from pathlib import Path

from matcha_dl.core.contracts.dataset import DataFrame
from matcha_dl.core.entities.mappings import ReferenceMapping
from matcha_dl.core.values import DEFAULT_REL
from matcha_dl.impl.utils import read_table


class EntityMapping:
    r"""A datastructure for entity mapping.

    Such entities should be named and have an IRI.

    Attributes:
        src_entity_iri (str): The IRI of the source entity, usually its IRI if available.
        tgt_entity_iri (str): The IRI of the target entity, usually its IRI if available.
        relation (str, optional): A symbol that represents what semantic relation this mapping stands for. Defaults to `<?rel>` which means unspecified.
            Suggested inputs are `"<EquivalentTo>"` and `"<SubsumedBy>"`.
        score (float, optional): The score that indicates the confidence of this mapping. Defaults to `0.0`.
    """

    def __init__(self, src_entity_iri: str, tgt_entity_iri: str, relation: str = DEFAULT_REL, score: float = 0.0):
        """Intialise an entity mapping.

        Args:
            src_entity_iri (str): The IRI of the source entity, usually its IRI if available.
            tgt_entity_iri (str): The IRI of the target entity, usually its IRI if available.
            relation (str, optional): A symbol that represents what semantic relation this mapping stands for. Defaults to `<?rel>` which means unspecified.
                Suggested inputs are `"<EquivalentTo>"` and `"<SubsumedBy>"`.
            score (float, optional): The score that indicates the confidence of this mapping. Defaults to `0.0`.
        """
        self.head = src_entity_iri
        self.tail = tgt_entity_iri
        self.relation = relation
        self.score = score

    def to_tuple(self, with_score: bool = False):
        """Transform an entity mapping (`self`) to a tuple representation

        Note that `relation` is discarded and `score` is optionally preserved).
        """
        if with_score:
            return (self.head, self.tail, self.score)
        else:
            return (self.head, self.tail)

    @staticmethod
    def as_tuples(entity_mappings: List['EntityMapping'], with_score: bool = False):
        """Transform a list of entity mappings to their tuple representations.

        Note that `relation` is discarded and `score` is optionally preserved).
        """
        return [m.to_tuple(with_score=with_score) for m in entity_mappings]

    @staticmethod
    def sort_entity_mappings_by_score(entity_mappings: List['EntityMapping'], k: Optional[int] = None):
        r"""Sort the entity mappings in a list by their scores in descending order.

        Args:
            entity_mappings (List[EntityMapping]): A list entity mappings to sort.
            k (int, optional): The number of top $k$ scored entities preserved if specified. Defaults to `None` which
                means to return **all** entity mappings.

        Returns:
            (List[EntityMapping]): A list of sorted entity mappings.
        """
        return list(sorted(entity_mappings, key=lambda x: x.score, reverse=True))[:k]

    def __repr__(self):
        return f"EntityMapping({self.head} {self.relation} {self.tail}, {round(self.score, 6)})"
    
    @staticmethod
    def read_table_mappings(
        table_of_mappings_file: str,
        threshold: Optional[float] = None,
        relation: str = DEFAULT_REL,
        is_reference: bool = False,
    ) -> List['EntityMapping']:
        r"""Read entity mappings from `.csv` or `.tsv` files.
        
        !!! note "Mapping Table Format"
        
            The columns of the mapping table must have the headings: `"SrcEntity"`, `"TgtEntity"`, and `"Score"`.

        Args:
            table_of_mappings_file (str): The path to the table (`.csv` or `.tsv`) of mappings.
            threshold (Optional[float], optional): Mappings with scores less than `threshold` will not be loaded. Defaults to 0.0.
            relation (str, optional): A symbol that represents what semantic relation this mapping stands for. Defaults to `<?rel>` which means unspecified.
                Suggested inputs are `"<EquivalentTo>"` and `"<SubsumedBy>"`.
            is_reference (bool): Whether the loaded mappings are reference mappigns; if so, `threshold` is disabled and mapping scores
                are all set to $1.0$. Defaults to `False`.

        Returns:
            (List[EntityMapping]): A list of entity mappings loaded from the table file.
        """
        df = read_table(table_of_mappings_file)
        entity_mappings = []
        for dp in df.itertuples():
            if is_reference:
                entity_mappings.append(ReferenceMapping(dp.SrcEntity, dp.TgtEntity, relation))
            else:
                # allow `None` for threshold
                if not threshold or dp["Score"] >= threshold:
                    entity_mappings.append(EntityMapping(dp.SrcEntity, dp.TgtEntity, relation, dp.Score))
        return entity_mappings

    def __repr__(self):
        return f"EntityMapping({self.head} {self.relation} {self.tail}, {round(self.score, 6)})"