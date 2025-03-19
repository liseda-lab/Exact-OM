
from enum import Enum

class Matchers(str, Enum):
    LM = 'LM'
    BKM = 'BKM'
    SM = 'SM'
    WM = 'WM'
    LLM = 'LLM'

class Sampler(str, Enum):
    all = 'all'
    rand = 'rand'
    diff = 'diff'