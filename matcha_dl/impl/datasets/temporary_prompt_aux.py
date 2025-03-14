from itertools import product
from typing import List
from abc import ABC, abstractmethod
PROMPT_EXAMPLE = "Are 'dog' and 'canine' equivalent?"
EXAMPLE_SOLUTION = "yes"
BASE_SKELETON = "$TC$I$CONF: 'Are '$CO_A'$SUPSUB_A and '$CO_B'$SUPSUB_B $E?'"

# Define a simple base class to handle dynamic attributes
class BaseInformation:
    def __init__(self, value: str):
        self.value = value

class TaskContext(BaseInformation):

    @abstractmethod
    def process(skeleton, value):
        if value:
            return skeleton.replace('$TC', "You are doing an ontology aligmnent task, ")
        return skeleton.replace('$TC', '')

class ComparisonType(BaseInformation):
    S = "Simple"
    E = "Equivalent"
    M = "Match"

    @abstractmethod
    def process(skeleton, enum_value):
        
        s_ending = "synonyms"
        e_ending = "equivalent"
        m_ending = "a match"

        if enum_value == ComparisonType.S:
            return skeleton.replace('$E', s_ending)
        elif enum_value == ComparisonType.E:
            return skeleton.replace('$E', e_ending)
        elif enum_value == ComparisonType.M:
            return skeleton.replace('$E', m_ending)

        return skeleton.replace('$E', ' ')

class InstructionInformation(BaseInformation):

    @abstractmethod
    def process(skeleton, value):
        
        iquery = f"I am going to ask you a question and you should answer 'yes' or 'no'."
        iwquery = iquery+" "+f"For example given the question '{PROMPT_EXAMPLE}' you should respond '{EXAMPLE_SOLUTION}'"

        if value:
            return skeleton.replace('$I', iwquery)
        return skeleton.replace('$I', iquery)

class LabelInformation(BaseInformation):
    @abstractmethod
    def process(class_1, class_2, skeleton, value):

        number, join_strategy = value.split("___")
        number = int(number)

        class_1_labels = class_1.labels[:number]
        class_2_labels = class_2.labels[:number]

        # LOOK AT JOIN STRATEGY
        if join_strategy == "comma":
            class_1_labels = ", ".join(class_1_labels)
            class_2_labels = ", ".join(class_2_labels)

        elif join_strategy == "parenthesis":

            class_1_labels = "("+") (".join(class_1_labels)+")"
            class_2_labels = "("+") (".join(class_2_labels)+")"

        # APPLY JOIN STRATEGY
        return skeleton.replace('$CO_A', class_1_labels).replace('$CO_B', class_2_labels)

class ClassInformation(BaseInformation):
    # Defining the literal values (constants)
    DS_SON_OF = "DS_SON_OF"
    DS_PART_OF = "DS_PART_OF"
    DS_TYPE_OF = "DS_TYPE_OF"
    DS_KIND_OF = "DS_KIND_OF"
    DS_SUBCLASS_OF = "DS_SUBCLASS_OF"
    
    S_WITH_SON = "S_WITH_SON"
    S_WITH_SUBCLASS = "S_WITH_SUBCLASS"
    
    TSC_PART_OF = "TSC_PART_OF"
    TSC_SUBCLASS_OF = "TSC_SUBCLASS_OF"

    @staticmethod
    def process(class_1, class_2, skeleton, value):
        # Split the value into classtype, cardinality, semantics, and separator
        classtype, cardinality, semantics, separator = value.split("___")

        cardinality = int(cardinality)

        # Get the relevant labels based on cardinality
        class_1_labels = class_1.labels[:cardinality]
        class_2_labels = class_2.labels[:cardinality]

        # Mapping classtype to "Parent", "Child", "Top"
        if classtype == "DS":
            classtype = "Parent"
        elif classtype == "S":
            classtype = "Child"
        elif classtype == "TSC":
            classtype = "Top"

        # Function to apply the join strategy (comma, parenthesis, etc.)
        def apply_join_strategy(values, strategy):
            if strategy == "comma":
                return ", ".join(values)
            elif strategy == "parenthesis":
                return "(" + ") (".join(values) + ")"
            else:
                # Default behavior: just join by space
                return " ".join(values)

        # Handling Parent (DS), Child (S), and Top (TSC)
        if classtype == "Parent":
            # Compare semantics for Parent: "part_of", "subclass_of", "kind_of", "type_of"
            if semantics == "part_of":
                parts_of_a = class_1.parts_of
                parts_of_b = class_2.parts_of

                # Apply join strategy to the parts_of attributes
                parts_of_a = apply_join_strategy(parts_of_a, separator)
                parts_of_b = apply_join_strategy(parts_of_b, separator)

                skeleton = skeleton.replace('$SUPSUB_A', f" with part '{parts_of_a}'")
                skeleton = skeleton.replace('$SUPSUB_B', f" with part '{parts_of_b}'")

            elif semantics == "subclass_of":
                subclasses_of_a = class_1.subclasses
                subclasses_of_b = class_2.subclasses

                # Apply join strategy to the subclasses attributes
                subclasses_of_a = apply_join_strategy(subclasses_of_a, separator)
                subclasses_of_b = apply_join_strategy(subclasses_of_b, separator)

                skeleton = skeleton.replace('$SUPSUB_A', f" subclass of '{subclasses_of_a}'")
                skeleton = skeleton.replace('$SUPSUB_B', f" subclass of '{subclasses_of_b}'")
            
            elif semantics == "kind_of":
                kinds_of_a = class_1.kinds_of
                kinds_of_b = class_2.kinds_of

                # Apply join strategy to the kinds_of attributes
                kinds_of_a = apply_join_strategy(kinds_of_a, separator)
                kinds_of_b = apply_join_strategy(kinds_of_b, separator)

                skeleton = skeleton.replace('$SUPSUB_A', f" kind of '{kinds_of_a}'")
                skeleton = skeleton.replace('$SUPSUB_B', f" kind of '{kinds_of_b}'")
            
            elif semantics == "type_of":
                types_of_a = class_1.types_of
                types_of_b = class_2.types_of

                # Apply join strategy to the types_of attributes
                types_of_a = apply_join_strategy(types_of_a, separator)
                types_of_b = apply_join_strategy(types_of_b, separator)

                skeleton = skeleton.replace('$SUPSUB_A', f" with type '{types_of_a}'")
                skeleton = skeleton.replace('$SUPSUB_B', f" with type '{types_of_b}'")

        elif classtype == "Child":
            # Compare semantics for Child: "with_subclass", "with_part", "with_type", "with_kind"
            if semantics == "with_subclass":
                subclasses_of_a = class_1.subclasses
                subclasses_of_b = class_2.subclasses

                # Apply join strategy to the subclasses attributes
                subclasses_of_a = apply_join_strategy(subclasses_of_a, separator)
                subclasses_of_b = apply_join_strategy(subclasses_of_b, separator)

                skeleton = skeleton.replace('$SUPSUB_A', f" with subclass '{subclasses_of_a}'")
                skeleton = skeleton.replace('$SUPSUB_B', f" with subclass '{subclasses_of_b}'")

            elif semantics == "with_part":
                parts_of_a = class_1.parts_of
                parts_of_b = class_2.parts_of

                # Apply join strategy to the parts_of attributes
                parts_of_a = apply_join_strategy(parts_of_a, separator)
                parts_of_b = apply_join_strategy(parts_of_b, separator)

                skeleton = skeleton.replace('$SUPSUB_A', f" with part '{parts_of_a}'")
                skeleton = skeleton.replace('$SUPSUB_B', f" with part '{parts_of_b}'")
            
            elif semantics == "with_type":
                types_of_a = class_1.types_of
                types_of_b = class_2.types_of

                # Apply join strategy to the types_of attributes
                types_of_a = apply_join_strategy(types_of_a, separator)
                types_of_b = apply_join_strategy(types_of_b, separator)

                skeleton = skeleton.replace('$SUPSUB_A', f" with type '{types_of_a}'")
                skeleton = skeleton.replace('$SUPSUB_B', f" with type '{types_of_b}'")

            elif semantics == "with_kind":
                kinds_of_a = class_1.kinds_of
                kinds_of_b = class_2.kinds_of

                # Apply join strategy to the kinds_of attributes
                kinds_of_a = apply_join_strategy(kinds_of_a, separator)
                kinds_of_b = apply_join_strategy(kinds_of_b, separator)

                skeleton = skeleton.replace('$SUPSUB_A', f" kind of '{kinds_of_a}'")
                skeleton = skeleton.replace('$SUPSUB_B', f" kind of '{kinds_of_b}'")
        
        elif classtype == "Top":
            # For Top (TSC), compare semantics for "subclass_of" and "part_of"
            if semantics == "subclass_of":
                subclasses_of_a = class_1.subclasses
                subclasses_of_b = class_2.subclasses

                # Apply join strategy to the subclasses attributes
                subclasses_of_a = apply_join_strategy(subclasses_of_a, separator)
                subclasses_of_b = apply_join_strategy(subclasses_of_b, separator)

                skeleton = skeleton.replace('$SUPSUB_A', f" subclass of '{subclasses_of_a}'")
                skeleton = skeleton.replace('$SUPSUB_B', f" subclass of '{subclasses_of_b}'")

            elif semantics == "part_of":
                parts_of_a = class_1.parts_of
                parts_of_b = class_2.parts_of

                # Apply join strategy to the parts_of attributes
                parts_of_a = apply_join_strategy(parts_of_a, separator)
                parts_of_b = apply_join_strategy(parts_of_b, separator)

                skeleton = skeleton.replace('$SUPSUB_A', f" with part '{parts_of_a}'")
                skeleton = skeleton.replace('$SUPSUB_B', f" with part '{parts_of_b}'")

        # Return the modified skeleton
        return skeleton

class Confidence(BaseInformation):
    LFLOAT = "float"
    LINT = "int"
    LCAT = "cat"


    @abstractmethod
    def process(skeleton, value):
        if value == Confidence.LFLOAT:
            return skeleton.replace('$CONF', ", followed by your confidence in your answer as a score from 0 to 1, like this: 'yes:0.8'")
        elif value == Confidence.LINT:
            return skeleton.replace('$CONF', ", followed by your confidence in your answer as a score from 0 to 10, like this: 'yes:8'")
        elif value == Confidence.LCAT:
            return skeleton.replace('$CONF', ", followed by your confidence in your answer 'Not Confident', 'Confident', 'Very Confidant', like this: 'yes:Confidant'")

        return skeleton.replace("$CONF", "")

class Critic(BaseInformation):
    # TODO: THIS SHOULD NOT BE PROCESS
    @abstractmethod
    def process(self, skeleton, value):
        pass

def get_instances(value_lists: List[List], cls):
    # Zip the lists together so the values at the same index are paired
    zipped_values = zip(*value_lists)
    
    instances = []
    
    for combo in zipped_values:
        # Create a unique string key from the combination of values
        value_str = "___".join(str(val) for val in combo)
        # Create an instance of the respective class with the generated value
        instances.append(cls(value_str))
    
    return instances

class ConfigMock:
    def __init__(self):
        # Define the values for each category
        self.task_context = [True, True]  # TaskContext
        self.example = [True, False]  # InstructionInformation
        self.comparison = ["Simple", "Match"] # ComparisonType
        self.task_context = [True, False]  # TaskContext
        self.separator = ['comma', 'parenthesis']  # LabelInformation
        self.label_cardinality = [1, 5]  # LabelInformation
        self.context_type = ['Parent', "Child"]  # ClassInformation
        self.context_cardinality = [1, 5]  # ClassInformation
        self.context_semantics = ["part_of", "subclass_of"]  # ClassInformation
        self.likelihood = ["float", "cat"]  # Confidence
        self.critic = [True, False]  # Critic

        # Create instances grouped by their respective categories
        self.grouped_info = {
            "TaskContext": get_instances([self.task_context], TaskContext),
            "ComparisonType": get_instances([self.comparison], ComparisonType),
            "InstructionInformation": get_instances([self.example], InstructionInformation),
            "LabelInformation": get_instances([self.label_cardinality, self.separator], LabelInformation),
            "ClassInformation": get_instances([self.context_type, self.context_cardinality, self.context_semantics, self.separator], ClassInformation),
            "Confidence": get_instances([self.likelihood], Confidence),
            "Critic": get_instances([self.critic], Critic)
        }

        static_enums = ["TaskContext", "ComparisonType", "InstructionInformation", "Confidence"]
        dynamic_enums = ["LabelInformation", "ClassInformation", "Critic"] # Coloquei o critic aqui porque é suposto fazer no fim, não tem que estar aqui especificamente
        self.static_info = {x: y for x, y in self.grouped_info.items() if x in static_enums}
        self.dynamic_info = {x: y for x, y in self.grouped_info.items() if x in dynamic_enums}

# TODO: Modify so config_info is content from a file
# Outputs: List of lists (each inner list is a query configuration)
def get_experiment_from_config(config_info : ConfigMock) -> List[List[BaseInformation]]:    
    
    instance_lists = list(config_info.grouped_info.values())

    return list(zip(*instance_lists))

class MockTerm:
        def __init__(self, name, labels):
            self.name = name
            self.labels = labels
            self.parts_of = [f"PO {self.name} ID {i}" for i in range(1,3)]
            self.types_of = [f"TO {self.name} ID {i}" for i in range(1,3)]
            self.kinds_of = [f"KO {self.name} ID {i}" for i in range(1,3)]

            self.subclasses = "subclass_of"

# Gets list with mockterms (fake terms with labels, parts_of, types_of, kinds_of, subclasses)
def get_mock_terms() -> List[MockTerm]:
    

    terms = []
    for i in range(1,3):
        for j in range(1,3):
            terms.append(MockTerm(f"Term {i} {j}", [f"({i} {j}) Label 1", f"({i} {j}) Label 2", f"({i} {j}) Label 3"]))
    return terms

# Generates lists of skeletons by completing a base skeleton with non-term-sensitive information
def generate_static_skeletons(static_query_configurations : List[List[BaseInformation]]) -> List[str]:
    skeletons = []
    #print(static_query_configurations)
    for run in static_query_configurations:
        
        #print("run", [(x.__class__.__name__, x) for x in run])
        
        skeleton = BASE_SKELETON

        try:
            for instance in run:
                category = instance.__class__
                print(f"{category.__name__}:")
                # Process the skeleton with each instance's value
                skeleton = instance.__class__.process(skeleton, instance.value)
                print(f"  {skeleton}")
        except TypeError:
            print("---- Assuming", category.__name__, "is not static ----")
        skeletons.append(skeleton)
    return skeletons

# Generates a list of queries by completing static skeletons with term-sensitive information
def generate_queries(t1 : MockTerm, t2 : MockTerm, dynamic_query_configurations : List[List[BaseInformation]], static_skeletons : List[str]) -> List[List[str]]:
    results = []

    for i, run in enumerate(dynamic_query_configurations):
        #print("run", [(x.__class__.__name__, x.value) for x in run])
        
        # Continue to complete the skeleton
        skeleton = static_skeletons[i]

        # Go through all the dynamic instances
        for instance in run:
            category = instance.__class__
            try:
                print(f"{category.__name__}:")
                # Process the skeleton with each instance's value
                skeleton = instance.__class__.process(t1, t2, skeleton, instance.value)
                print(f"  {skeleton}")

            except TypeError:
                print("---- Assuming", category.__name__, "is not dynamic ----")
        
        # TODO: IF CRITIC
        if False:
            critic_query = get_critic_query(skeleton)
            results.append([skeleton, critic_query])
        else:
            results.append([skeleton])

    print(results)
    return results

def split_instances(grouped_info):
    # Determine the maximum number of elements in any list
    max_size = max(len(values) for values in grouped_info.values())

    # Create a list of empty lists, one for each possible index (0 to max_size - 1)
    result = [[] for _ in range(max_size)]

    # Iterate over each key-value pair in the dictionary
    for values in grouped_info.values():
        for i in range(len(values)):  # Loop through available elements
            result[i].append(values[i])

    return result