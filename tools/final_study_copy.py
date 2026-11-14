"""Owner-approved participant copy for the final comparative study."""

TASK = (
    "You will see a concept from the National Cancer Institute Thesaurus (NCIT) and five possible matches from the Disease Ontology (DOID). "
    "Your task is to decide which candidates describe the same concept as the source. "
    "For example, two names may refer to the same disease, while a similar name may refer to a "
    "broader group of diseases or a more specific subtype.\n\n"
    "Add the candidates you consider plausible matches to your answer and order them from best "
    "to least likely, with your best match first. Leave out candidates you do not consider matches. "
    "You can keep the system's order or change it. If none fits, choose ‘None of these’. "
    "If you cannot decide from the available information, choose ‘Insufficient information’.\n\n"
    "There are ten cases in two parts. In five cases, this tool provides definitions, ontology "
    "context and explanations to help you compare the concepts. In the other five, the page "
    "shows only names, identifiers, the system's proposed order and scores. For those cases, "
    "you can inspect the supplied ontology files using your own methods."
)

SETUP = (
    "Download the two ontology files below. Use these versions when inspecting the concepts. "
    "You can use Protégé, a text editor or another method, and combine or change methods between cases."
)

REQUIRED_DETAILS = (
    "researcher_and_contact",
    "data_access_storage_retention",
    "withdrawal_and_deletion",
)

# Owner-supplied facts, 11 November 2026. One month describes recruitment/data
# collection, not the time required to complete a participant's ten cases.
OWNER_DETAILS = {
    "researcher_and_contact": "Pedro Giesteira Cotovio — pgcotovio@ciencias.ulisboa.pt.",
    "collection_period": "The study will run for one month.",
    "data_access_storage_retention": (
        "Study responses will be stored on the study server and retained for one year after "
        "data collection ends. Access is restricted to the research team. Findings will be "
        "reported as group results, without identifying individual participants."
    ),
    "withdrawal_and_deletion": (
        "You may stop taking part at any time. To request removal of your session data "
        "while it is retained, email pgcotovio@ciencias.ulisboa.pt with the session code "
        "shown at the bottom of the study page. Do not send your private link. Your "
        "session will then be excluded from further analysis and its stored responses "
        "removed. Results already published in aggregate form cannot be withdrawn individually."
    ),
    "ethics_status": "The researcher is checking whether institutional review is required.",
    "ethics_confirmed": False,
}


def participant_information(details=None):
    """Render the requested information; do not invent missing contact or data policies.

    Readiness here covers this copy only, not institutional approval or data admission.
    The owner removed collection/duration estimates and the ethics-status paragraph.
    No claim of approval or exemption is made.
    """
    details = details or {}
    missing = [key for key in REQUIRED_DETAILS if not str(details.get(key, "")).strip()]

    def value(key, label):
        return str(details.get(key) or f"[{label} — to be confirmed]")

    information = "\n\n".join(
        [
            "We are studying whether explanations help people assess matches between biomedical "
            "ontologies. An ontology organises concepts, their descriptions and their relationships. "
            "In this study, you will compare concepts from the National Cancer Institute Thesaurus "
            "(NCIT) with concepts from the Disease Ontology (DOID).",
            "You will answer a few background questions, practise with the tool, and then assess "
            "ten concepts. For each one, choose which of five proposed matches have the same "
            "meaning and rank the plausible matches, with the best match first. Five cases include "
            "explanations from the tool. The other five show names, identifiers, scores and the "
            "system's proposed order; you can inspect the supplied ontology files using your own "
            "methods. At the end, you will answer a few questions about your experience.",
            "Your answers, the actions you take in the study pages and how long each step takes "
            "will be recorded. "
            + value("data_access_storage_retention", "Data access, storage and retention policy"),
            "Taking part is voluntary. "
            + value("withdrawal_and_deletion", "How to request withdrawal or removal of your data"),
            "If you have any questions about the study, contact "
            + value("researcher_and_contact", "Researcher and contact details"),
        ]
    )
    consent = (
        "By selecting ‘I agree to take part’, I confirm that I have read the information above, "
        "understand what the study involves and how my data will be used, and agree to participate."
    )
    pending = list(missing)
    return {
        "information_text": information,
        "consent_text": consent,
        "instructions": TASK,
        "setup_instructions": SETUP,
        "missing_owner_details": missing,
        "pending_review_items": pending,
        "ready_for_participant_publication": not pending,
    }
