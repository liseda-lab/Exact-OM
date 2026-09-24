"""Frozen questionnaire wording, codes, branching and draft validation."""

from __future__ import annotations

from copy import deepcopy

YEARS = {
    "none": "None",
    "less_than_1": "Less than 1 year",
    "1_2": "1–2 years",
    "3_4": "3–4 years",
    "5_9": "5–9 years",
    "10_plus": "10 or more years",
    "prefer_not_to_say": "Prefer not to say",
}
YES_NO = {"yes": "Yes", "no": "No", "prefer_not_to_say": "Prefer not to say"}
FAMILIARITY = {
    "unheard": "Had not heard of it",
    "heard": "Had heard of it but not used it",
    "used": "Had used it",
    "prefer_not_to_say": "Prefer not to say",
}
CONTEXTS = {
    "research": "Research",
    "clinical": "Clinical work",
    "teaching": "Teaching",
    "curation": "Curation",
    "other": "Other",
    "prefer_not_to_say": "Prefer not to say",
}
COMPONENTS = {
    "original_context": "Original entity definitions and context",
    "entity_description": "Generated entity descriptions",
    "hierarchy": "Hierarchy browser",
    "evidence_table": "Comparison and evidence table",
    "evidence_graph": "Evidence graph",
    "pair_comparison": "Generated pair comparison",
}
HELPFUL = {
    "not_helpful": "Not helpful",
    "slightly": "Slightly helpful",
    "moderately": "Moderately helpful",
    "very": "Very helpful",
    "cannot_judge": "Did not use / cannot judge",
}
EFFORT = {
    "very_low": "Very low",
    "low": "Low",
    "moderate": "Moderate",
    "high": "High",
    "very_high": "Very high",
    "cannot_judge": "Cannot judge",
}
SPECIAL = {"prefer_not_to_say", "never_used", "all_equally", "none_helpful", "cannot_judge"}


def question(
    identifier, label, options=None, *, multiple=False, required=True, when=None, matrix=None
):
    """Build the public form descriptor from its frozen semantic pieces."""
    return {
        "id": identifier,
        "label": label,
        "options": options,
        "multiple": multiple,
        "required": required,
        "show_if": when,
        "matrix": matrix,
    }


BACKGROUND = [
    question(
        "cs_experience_years",
        "How much study or professional experience do you have in computer science or computing?",
        YEARS,
    ),
    question(
        "health_bio_experience_years",
        "How much study or professional experience do you have in health, medicine or biosciences?",
        YEARS,
    ),
    question(
        "domain_specialty",
        "What is your main health or bioscience field, if any? Please do not identify an institution or employer.",
        required=False,
    ),
    question(
        "semantic_web_experience",
        "Have you used semantic-web technologies or worked with ontologies in study or professional work?",
        YES_NO,
    ),
    question(
        "clinical_experience", "Do you have practical experience in clinical medicine?", YES_NO
    ),
    question(
        "bioinformatics_experience",
        "Do you have practical experience in bioinformatics?",
        YES_NO,
    ),
    question(
        "doid_familiarity",
        "Before this study, how familiar were you with the Disease Ontology (DOID)?",
        FAMILIARITY,
    ),
    question(
        "doid_contexts",
        "In which contexts have you used DOID?",
        CONTEXTS,
        multiple=True,
        when={"doid_familiarity": "used"},
    ),
    question(
        "ncit_familiarity",
        "Before this study, how familiar were you with the NCI Thesaurus (NCIT)?",
        FAMILIARITY,
    ),
    question(
        "ncit_contexts",
        "In which contexts have you used NCIT?",
        CONTEXTS,
        multiple=True,
        when={"ncit_familiarity": "used"},
    ),
    question(
        "roles",
        "Which roles describe your current work or study?",
        {
            "undergraduate": "Undergraduate student",
            "postgraduate": "Postgraduate student",
            "researcher": "Researcher",
            "senior_researcher": "Senior researcher",
            "clinician": "Medical doctor / clinician",
            "other": "Other",
            "prefer_not_to_say": "Prefer not to say",
        },
        multiple=True,
    ),
    question(
        "correspondence_confidence",
        "How confident are you in identifying when two medical or disease terms refer to the same concept?",
        {
            "not_at_all": "Not at all",
            "slightly": "Slightly",
            "moderately": "Moderately",
            "highly": "Highly",
            "prefer_not_to_say": "Prefer not to say",
        },
    ),
    question(
        "ontology_activities",
        "Which of the following have you done with ontologies?",
        {
            "annotated": "Described / annotated data",
            "software": "Used in automated analysis or software",
            "curation": "Used for manual curation",
            "developed": "Developed / maintained ontologies",
            "other": "Other",
            "never_used": "Never used",
            "prefer_not_to_say": "Prefer not to say",
        },
        multiple=True,
    ),
    question(
        "protege_experience",
        "Before preparing for this study, how often had you used Protégé?",
        {
            "never": "Never",
            "occasionally": "Tried occasionally",
            "regularly": "Used regularly",
            "prefer_not_to_say": "Prefer not to say",
        },
    ),
]


def definitions(components=None, version="exact-study-forms/1"):
    """Return a detached, ordered definition; historical versions are frozen in the DB."""
    selected = {k: COMPONENTS[k] for k in (components if components is not None else COMPONENTS)}
    final = [
        question(
            "component_usefulness",
            "How helpful was each component for deciding your ranking?",
            HELPFUL,
            matrix=selected,
        ),
        question(
            "most_helpful_components",
            "Which explanation component or components were most helpful?",
            {
                **selected,
                "all_equally": "All equally helpful",
                "none_helpful": "None helpful",
                "cannot_judge": "Cannot judge",
            },
            multiple=True,
        ),
        question(
            "workflow_preference",
            "Which approach would you prefer for similar tasks?",
            {
                "explanation": "Explanation interface",
                "ontology_baseline": "Candidates/scores with separate ontology inspection",
                "no_preference": "No preference",
                "cannot_judge": "Cannot judge",
            },
        ),
        question(
            "mental_effort",
            "How much mental effort did each approach require?",
            EFFORT,
            matrix={
                "explanation": "Explanation interface",
                "ontology_baseline": "Candidates/scores with separate ontology inspection",
            },
        ),
        question(
            "preference_reason",
            "Why did you prefer those components or that approach?",
            required=False,
        ),
        question(
            "comments",
            "What was confusing, missing or difficult, and what would you change? Please avoid identifying details.",
            required=False,
        ),
    ]
    return deepcopy(
        {
            "version": version,
            "background": BACKGROUND,
            "consultation": [
                question(
                    "consulted_external_ontologies",
                    "For this case, did you consult the ontology resources outside the study's explanation panels?",
                    {"true": "Yes", "false": "No"},
                ),
                question(
                    "methods",
                    "How did you consult them? Select all that apply.",
                    {
                        "protege": "Protégé",
                        "other_editor": "Another ontology editor",
                        "plain_files": "The ontology files directly (for example, in a text editor)",
                        "other_resource": "Another ontology resource or viewer",
                    },
                    multiple=True,
                    when={"consulted_external_ontologies": True},
                ),
                question(
                    "other_editor",
                    "Which other ontology editor did you use? Please do not include a personal name, institution or URL.",
                    required=False,
                    when={"methods": {"contains": "other_editor"}},
                ),
                question(
                    "other_resource",
                    "Which other ontology resource or viewer did you use? Please do not include a personal name, institution or URL.",
                    required=False,
                    when={"methods": {"contains": "other_resource"}},
                ),
            ],
            "final": final,
            "experience_note": "Years mean approximate combined study/work experience; count overlapping years once.",
        }
    )


def validate_answers(form, answers, *, submitted):
    """Validate visible answers and clear hidden current values without erasing history."""
    if set(answers) - {q["id"] for q in form}:
        raise ValueError("Unknown questionnaire field")
    result = {}
    states = {}
    for q in form:
        key = q["id"]
        if q["show_if"] and any(
            (
                (v["contains"] not in (answers.get(k) or []))
                if isinstance(v, dict) and "contains" in v
                else answers.get(k) != v
            )
            for k, v in q["show_if"].items()
        ):
            states[key] = "skipped"
            continue
        value = answers.get(key)
        if value is None or value == "" or value == []:
            if submitted and q["required"]:
                raise ValueError(f"Required answer: {key}")
            states[key] = "not_answered"
            continue
        options = q["options"]
        if options is None:
            if not isinstance(value, str) or len(value) > 2000:
                raise ValueError(f"Invalid optional text: {key}")
        elif q["matrix"]:
            if not isinstance(value, dict) or set(value) - set(q["matrix"]):
                raise ValueError(f"Invalid matrix: {key}")
            if submitted and set(value) != set(q["matrix"]):
                raise ValueError(f"Complete all ratings: {key}")
            if any(not isinstance(v, str) or v not in options for v in value.values()):
                raise ValueError(f"Invalid rating: {key}")
        elif q["multiple"]:
            if (
                not isinstance(value, list)
                or any(not isinstance(v, str) for v in value)
                or len(value) != len(set(value))
                or not set(value) <= set(options)
            ):
                raise ValueError(f"Invalid choices: {key}")
            if len(value) > 1 and set(value) & SPECIAL:
                raise ValueError(f"Special choice must be exclusive: {key}")
        elif not isinstance(value, str) or value not in options:
            raise ValueError(f"Invalid choice: {key}")
        result[key] = value
        states[key] = "answered"
    return result, states
