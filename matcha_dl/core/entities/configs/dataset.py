
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
    part_of = 'part_of'
    kind_of = 'kind_of'
    type_of = 'type_of'
    subclass_of = 'subclass_of'
    with_subclass = 'with_subclass'
    with_part = 'with_part'
    with_type = 'with_type'
    with_kind = 'with_kind'

class Likelihood(str, Enum):
    float = 'float'
    cat = 'cat'