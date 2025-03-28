
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
    

class DatasetMask(str, Enum):
    train = "train"
    validation = "validation"
    inference = "inference"