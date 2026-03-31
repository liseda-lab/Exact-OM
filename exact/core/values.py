# MAPPINGS

DEFAULT_REL = "<?rel>"

# EVALUATION

ANNOTATION_IRI = "http://oaei.ontologymatching.org/bio-ml/ann/use_in_alignment"
TIMING_STEP_ORDER = (
    "Total",
    "Dataset",
    "Dataset.LoadOntologies",
    "Dataset.LoadCandidates",
    "Dataset.Process",
    "Dataset.Save",
    "Dataset.Plotting",
    "Dataset.CacheLoad",
    "Alignment",
    "Alignment.Inference",
    "Alignment.PostInference",
    "Alignment.Prefilter",
    "Postprocess",
    "Postprocess.Rationales",
    "Postprocess.Outputs",
    "Postprocess.Plotting",
    "Postprocess.Evaluation",
)

# DATASET

DATASET_URL = "https://zenodo.org/api/records/13119437/files-archive"
CONFERENCE_URL = "https://oaei.ontologymatching.org/2025/conference/data/conference.zip"
REFERENCE_ALIGNMENT_URL = (
    "https://oaei.ontologymatching.org/2025/conference/data/reference-alignment.zip"
)
