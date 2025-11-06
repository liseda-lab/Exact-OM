
from enum import Enum

class ContextMethod(str, Enum):
    bfs = "bfs"
    greedy = "greedy"

class BestPathMethod(str, Enum):
    dp = "dp"
    lagrangian = "lagrangian"
    greedy = "greedy"

class DatasetMask(str, Enum):
    inference = "inference"
    prefiltered = "prefiltered"