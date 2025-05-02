
from enum import Enum
from matcha_dl.core.entities.configs.dataset import Likelihood

class StaticPrompts:
    POSITIVE_SOLUTION = "yes"
    NEGATIVE_SOLUTION = "no"
    UNCERTAIN_SOLUTION = "uncertain"
    POSITIVE_CONFIDENCE = "confident"
    NEGATIVE_CONFIDENCE = "not confident"
    VERY_POSITIVE_CONFIDENCE = "very confident"
    TASK_CONTEXT = "You are doing an ontology alignment task. "
    INSTRUCTION = f"I am going to ask you a question and you should answer '{POSITIVE_SOLUTION}' or '{NEGATIVE_SOLUTION}'. "
    FIRST_EXAMPLE = "For example"
    FOLLOWING_EXAMPLE = "Another example"
    EXAMPLE_BASE = "$START given the question '$EXAMPLE' you should respond '$SOLUTION'. "
    SKELETON = "$TC$I$E$CONF. Question: Are '$S' $CTX_S and '$T' $CTX_T $TYPE?"
    EXAMPLE = "Are '$S' and '$T' equivalent?"
    CONFIDENCE = {
        Likelihood.float: f"Followed by your confidence in your answer as a score from 0 to 1, like this: '{POSITIVE_SOLUTION}:0.8'",
        Likelihood.cat: f"Followed by your confidence in your answer '{NEGATIVE_CONFIDENCE}', '{POSITIVE_CONFIDENCE}', '{VERY_POSITIVE_CONFIDENCE}', like this: '{POSITIVE_SOLUTION}:confident'"
    }
    CRITIC_SKELETON = f"You are verifying an ontology aligmnent task, given the question $P the response was $R is this correct? You should answer '{POSITIVE_SOLUTION}' or '{NEGATIVE_SOLUTION}'$U."
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
    def get_task_context(cls, task_context:bool) -> str:
        if task_context:
            return cls.TASK_CONTEXT
        else:
            return ""

    @classmethod
    def get_example(cls, source: str, target: str, solution: bool, first: bool) -> str:
        example = cls.EXAMPLE.replace("$S", source).replace("$T", target)
        return cls.EXAMPLE_BASE.replace("$START",cls.FIRST_EXAMPLE if first else cls.FOLLOWING_EXAMPLE).replace("$EXAMPLE", example).replace("$SOLUTION", cls.POSITIVE_SOLUTION if solution else cls.NEGATIVE_SOLUTION)

    @classmethod
    def get_confidence(cls, likelihood: Likelihood) -> str:
        return cls.CONFIDENCE.get(likelihood, "")
    
    @classmethod
    def get_critic(cls, prompt: str, response: str, uncertainty: bool = False) -> str:
        critic = cls.CRITIC_SKELETON.replace("$P", prompt).replace("$R", response)
        if uncertainty:
            return critic.replace("$U", f" or {cls.UNCERTAIN_SOLUTION}")
        return cls.CRITIC_SKELETON.replace("$U", "")
    
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
    train = "train"
    validation = "validation"
    inference = "inference"
    prefiltered = "prefiltered"