# Adapted from https://github.com/KRR-Oxford/DeepOnto

from typing import List, Optional, Union, TYPE_CHECKING
from pathlib import Path

from collections import defaultdict
from heapq import nlargest

from matcha_dl.core.values import DEFAULT_REL
from matcha_dl.utils.data import read_table

if TYPE_CHECKING:
    from matcha_dl.core.contracts.dataset import DataFrame

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
    
    @staticmethod
    def filter_entity_mappings_by_score(entity_mappings: List['EntityMapping'], threshold: float) -> List['EntityMapping']:
        r"""Filter the entity mappings in a list by their scores.

        Args:
            entity_mappings (List[EntityMapping]): A list entity mappings to filter.
            threshold (float): The threshold value to filter the entity mappings.

        Returns:
            (List[EntityMapping]): A list of filtered entity mappings.
        """
        return [m for m in entity_mappings if m.score >= threshold]
    
    @staticmethod
    def filter_top_n_entity_mappings(preds: List['EntityMapping'], n: int) -> List['EntityMapping']:
        r"""Filter the entity mappings in a list by their scores.
        Args:
            preds (List[EntityMapping]): A list entity mappings to filter.
            n (int): The number of top $n$ scored entities preserved.
        Returns:
            (List[EntityMapping]): A list of filtered entity mappings.
        """
        all_sources = defaultdict(list)
        
        for ent_map in preds:
            all_sources[ent_map.head].append(ent_map)
        
        filtered_mappings = []
        for head, mappings in all_sources.items():
            top_n_mappings = nlargest(n, mappings, key=lambda x: x.score)
            filtered_mappings.extend(top_n_mappings)
        
        return filtered_mappings

    def __repr__(self):
        return f"EntityMapping({self.head} {self.relation} {self.tail}, {round(self.score, 6)})"
    
    @classmethod
    def read_table_mappings(
        cls,
        table_of_mappings_file: Union[Path, 'DataFrame'],
        threshold: Optional[float] = None,
        cardinality: Optional[int] = None,
        relation: str = DEFAULT_REL,
    ) -> List['EntityMapping']:
        
        r"""Read entity mappings from `.csv` or `.tsv` files."""

        if isinstance(table_of_mappings_file, Path):
            table_of_mappings_file = read_table(table_of_mappings_file)

        mappings = [cls(dp.SrcEntity, dp.TgtEntity, relation, dp.Score) for dp in table_of_mappings_file.itertuples()]

        if threshold is not None:
            mappings = cls.filter_entity_mappings_by_score(mappings, threshold)

        if cardinality is not None:
            mappings = cls.filter_top_n_entity_mappings(mappings, cardinality)
        return mappings