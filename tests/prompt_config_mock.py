from matcha_dl.impl.datasets.temporary_prompt_aux import split_instances, generate_static_skeletons, generate_queries, BaseInformation, ConfigMock, MockTerm

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

config_mock = ConfigMock()

static_info = split_instances(config_mock.static_info)

dynamic_info = split_instances(config_mock.dynamic_info)

static_skeletons = generate_static_skeletons(static_info)

queries = []

terms = get_mock_terms()

#for _, row in dataset.iterrows():

for source in terms:
    for target in terms:


        try:
            #source = row["Src"]
            #target = row["Tgt"]
            query = generate_queries(source, target, dynamic_info, static_skeletons)
            #vector = self.matcha_features.get(row["Src"]).get(row["Tgt"])
        except AttributeError:
            self.log("Attribute error in query generation", level="error", exc_info=True)
            raise ValueError("Scores for source {} and target {} not found.".format(row["Src"], row["Tgt"]))
        
        queries.append(query)
    
#dataset["Features"] = queries

print(queries)