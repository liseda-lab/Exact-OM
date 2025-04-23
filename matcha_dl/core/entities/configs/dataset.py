
from enum import Enum

class Separator(str, Enum):
    comma = 'comma'
    paranthesis = 'paranthesis'

class ComparisonType(str, Enum):
    synonyms = 'synonyms'
    equivalent = 'equivalent'
    a_match = 'a_match'

class ContextType(str, Enum):
    superclasses = 'superclasses'
    subclasses = 'subclasses'
    top_superclass = 'top_superclass'

class ContextSemantics(str, Enum):
    a_part_of = 'a_part_of'
    a_kind_of = 'a_kind_of'
    a_type_of = 'a_type_of'
    a_subclass_of = 'a_subclass_of'
    with_subclass = 'with_subclass'
    with_part = 'with_part'
    with_type = 'with_type'
    with_kind = 'with_kind'

class Likelihood(str, Enum):
    float = 'float'
    cat = 'cat'

class AggregationStrategy(str, Enum):
    JOIN = "join"
    SUMMARISE = "summarise"