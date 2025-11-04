
from enum import Enum
from exact.core.entities.configs.dataset import Likelihood

class StaticPrompts:
    POSITIVE_SOLUTION = "yes"
    NEGATIVE_SOLUTION = "no"
    INSTRUCTION = f"I am going to ask you a question and you should answer '{POSITIVE_SOLUTION}' or '{NEGATIVE_SOLUTION}'. "
    VERBALIZATION_SKELETON = (
        "You are a natural language generator that converts structured data into a coherent sentence. "
        "Given the triple below, generate one, single complete, grammatically correct sentence that clearly expresses "
        "the relationship between the head and the tail as indicated by the relation. "
        "You must use the exact wording of `$HEAD` and `$TAIL`—do not paraphrase, modify, or omit them. "
        "Do not include any additional commentary or information. "
        "The triple is: head: $HEAD, relation: $REL, tail: $TAIL. The sentence should be: <think>"
    )
    CORRECTIVE_VERBALIZATION_SKELETON = (
        "You are a natural language generator that converts structured data into a coherent sentence. "
        "Given the triple head: $HEAD, relation: $REL, tail: $TAIL, "
        "You generated the following incorrect sentence: '$SENTENCE'. "
        "Please regenerate this sentence, ensuring you use exactly the words '$HEAD' and '$TAIL'." \
        "Do not paraphrase, modify, or omit them. "
        "Do not include any additional commentary or information. "
        "Regenerated sentence: <think>"
    )
    SUMMARIZATION_SKELETON = (
        "You are an expert summariser. Given the following context subgraph, presented as a list of triples "
        "in the format 'head, relation, tail', that describe various aspects and relationships of the entity "
        "'$ENTITY', please provide a concise and informative summary that captures the key characteristics of "
        "$ENTITY. The context subgraph is as follows:\n$CONTEXT\nSummary: <think>"
    )
    
    @classmethod
    def get_verbalization(cls, head: str, relation: str, tail: str) -> str:
        return cls.VERBALIZATION_SKELETON.replace("$HEAD", head).replace("$REL", relation).replace("$TAIL", tail)
    @classmethod
    def get_corrective_verbalization(cls, sentence: str, head: str, tail: str) -> str:
        return cls.CORRECTIVE_VERBALIZATION_SKELETON.replace("$SENTENCE", sentence).replace("$HEAD", head).replace("$TAIL", tail)
    @classmethod
    def get_summarization(cls, entity: str, context: str) -> str:
        return cls.SUMMARIZATION_SKELETON.replace("$ENTITY", entity).replace("$CONTEXT", context)
    

class DatasetMask(str, Enum):
    inference = "inference"
    prefiltered = "prefiltered"