
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

class BatchLengthSortMode(str, Enum):
    max = "max"
    sum = "sum"

class ContextMethod(str, Enum):
    bfs = "bfs"
    greedy = "greedy"

class BestPathMethod(str, Enum):
    dp = "dp"
    lagrangian = "lagrangian"
    greedy = "greedy"

class PLotAgregationMethod(str, Enum):
    mean = "mean"
    max = "max"
    min = "min"
    sum = "sum"
    median = "median"
    mode = "mode"
    std = "std"
    var = "var"
    count = "count"
