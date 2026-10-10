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


def definitions(components=None, version="exact-study-forms/1", *, ordered=False):
    """Return a detached, ordered definition; historical versions are frozen in the DB."""
    if version == "exact-study-forms/3":
        return definitions3(components)
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
    result = deepcopy(
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

    if ordered or version == "exact-study-forms/2":
        next(q for q in result["background"] if q["id"] == "protege_experience")["required"] = False
        methods = result["consultation"][1]
        result["consultation"][0][
            "label"
        ] = "For the source and candidates you just ranked, did you inspect ontology information outside this study interface?"
        methods["label"] = (
            "Select all methods you actually used for this case. You may combine methods and change them between cases."
        )
        methods["options"].update(
            {
                "queries_scripts": "Queries or scripts",
                "reasoner": "A reasoner",
                "other_method": "Another inspection method",
            }
        )
        result["consultation"].extend(
            [
                question(
                    "other_method",
                    "Optional method name; do not include identities, paths or URLs.",
                    required=False,
                    when={"methods": {"contains": "other_method"}},
                ),
                question(
                    "resource_scope",
                    "Which ontology resources did you inspect? (Optional)",
                    {
                        "supplied_only": "Only the supplied ontology versions",
                        "different_or_additional": "Different or additional resources",
                        "unsure": "Unsure",
                    },
                    required=False,
                    when={"consulted_external_ontologies": True},
                ),
            ]
        )
        for form_id in ("background", "consultation", "final"):
            for q in result[form_id]:
                q["option_order"] = list(q["options"] or {})
                q["row_order"] = list(q["matrix"] or {})
        validate_ordered_forms(result)
    return result


def validate_ordered_forms(forms):
    """Require explicit exact permutations; canonical object-key sorting is irrelevant."""
    for name in ("background", "consultation", "final", *(("setup",) if "setup" in forms else ())):
        for question in forms[name]:
            for values, order in (("options", "option_order"), ("matrix", "row_order")):
                codes = set(question[values] or {})
                arranged = question.get(order)
                if (
                    not isinstance(arranged, list)
                    or len(arranged) != len(set(arranged))
                    or set(arranged) != codes
                ):
                    raise ValueError(f"Invalid explicit {order} for {question['id']}")


def validate_answers(form, answers, *, submitted, version=None):
    """Validate visible answers and clear hidden current values without erasing history."""
    if set(answers) - {q["id"] for q in form}:
        raise ValueError("Unknown questionnaire field")
    result = {}
    states = {}
    corrected = version == "exact-study-forms/3"
    for q in form:
        visible_answers = result if corrected else answers
        key = q["id"]
        if q["show_if"] and any(
            (
                (v["contains"] not in (visible_answers.get(k) or []))
                if isinstance(v, dict) and "contains" in v
                else visible_answers.get(k) != v
            )
            for k, v in q["show_if"].items()
        ):
            states[key] = "skipped"
            continue
        value = answers.get(key)
        if corrected and q["options"] is None and isinstance(value, str):
            value = value.strip()
        if value is None or value == "" or value == []:
            if submitted and q["required"]:
                raise ValueError(f"Required answer: {key}")
            states[key] = "not_answered"
            continue
        options = q["options"]
        if options is None:
            maximum = q.get("text_max_length", 2000) if corrected else 2000
            minimum = q.get("text_min_length", 0) if corrected else 0
            if (
                not isinstance(value, str)
                or len(value) > maximum
                or (submitted and len(value) < minimum)
            ):
                raise ValueError(
                    f"Invalid text: {key}" if corrected else f"Invalid optional text: {key}"
                )
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


METHODS3 = {"protege": "Protégé", "plain_files": "Text editor", "other_method": "Other method"}
YEARS3 = {
    "none": "None",
    "less_than_1": "Less than 1 year",
    "1_3": "1–3 years",
    "4_5": "4–5 years",
    "6_9": "6–9 years",
    "10_plus": "10+ years",
}
METHOD_COLLECTION_NOTICE = (
    "We record your selected methods and any short method names you enter. "
    "Default de-identified analysis exports omit method names. An authorized researcher can explicitly "
    "include those names in a restricted analysis export. Do not enter personal names, institutions, "
    "file paths or URLs. Familiarity is optional and does not require you to use or install a tool."
)


def definitions3(components=None):
    """Author a new codebook without modifying either historical definition."""
    result = definitions(components, version="exact-study-forms/2")
    result["version"] = "exact-study-forms/3"
    result["method_version"] = "study-methods/3"
    result["method_dictionary"] = dict(METHODS3)
    result["collection_notice"] = METHOD_COLLECTION_NOTICE
    result["experience_note"] = (
        "Count completed years, counting overlapping study/work years once. Zero experience is None; a positive duration under one year is Less than 1 year."
    )
    background = []
    other_fields = {
        "roles": "What other role describes your work or study?",
        "ontology_activities": "What other ontology activity have you done?",
        "doid_contexts": "In what other context have you used DOID?",
        "ncit_contexts": "In what other context have you used NCIT?",
    }
    for q in result["background"]:
        if q["options"] is not None:
            q["options"].pop("prefer_not_to_say", None)
        if q["id"] in {"cs_experience_years", "health_bio_experience_years"}:
            q["options"] = dict(YEARS3)
        if q["id"] == "domain_specialty":
            q["label"] = "What is your main health or bioscience field, if any?"
        background.append(q)
        if q["id"] in other_fields:
            background.append(
                question(
                    q["id"] + "_other", other_fields[q["id"]], when={q["id"]: {"contains": "other"}}
                )
            )
    result["background"] = background
    result["setup"] = [
        question(
            "familiar_methods",
            "Which of these methods are familiar to you? (Optional)",
            {**METHODS3, "none": "None of these"},
            multiple=True,
            required=False,
        ),
        question(
            "familiar_other_method",
            "Which other methods are familiar to you?",
            when={"familiar_methods": {"contains": "other_method"}},
        ),
    ]
    result["consultation"] = [
        question(
            "consulted_external_ontologies",
            "For the source and candidates you just ranked, did you inspect ontology information outside this study interface?",
            {"true": "Yes", "false": "No"},
        ),
        question(
            "methods",
            "Select all methods you actually used for this case. You may combine methods and change them between cases.",
            dict(METHODS3),
            multiple=True,
            when={"consulted_external_ontologies": True},
        ),
        question(
            "other_method",
            "Which other methods did you use?",
            when={"methods": {"contains": "other_method"}},
        ),
        question(
            "resource_scope",
            "Which ontology resources did you inspect? (Optional)",
            {
                "supplied_only": "Only the supplied ontology versions",
                "different_or_additional": "Different or additional resources",
                "unsure": "Unsure",
            },
            required=False,
            when={"consulted_external_ontologies": True},
        ),
    ]
    for name in ("background", "setup", "consultation", "final"):
        for q in result[name]:
            q["option_order"] = list(q["options"] or {})
            q["row_order"] = list(q["matrix"] or {})
            if q["options"] is None:
                q["text_min_length"] = 1 if q["required"] else 0
                q["text_max_length"] = 160 if q["required"] else 2000
            else:
                q["text_min_length"] = 0
                q["text_max_length"] = 2000
    validate_ordered_forms(result)
    return result
