"""Prepare revised participant copy in a NEW publication, reusing unchanged assets.

This does not publish, migrate sessions or change launch eligibility. Pass the resulting
JSON to the authenticated publication endpoint. Old invitations retain their frozen text.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

INFORMATION = (
    "This study examines how explanations help people assess ontology matches. "
    "You will compare concepts, rank proposed matches and answer questions about your experience."
)
CONSENT = "By selecting ‘I agree to take part’, you confirm that you have read the information above and agree to participate."
INSTRUCTIONS = (
    "For each source concept, choose the candidates you think have the same meaning and rank them, "
    "with the best match first. You can include one or more candidates. "
    "Choose ‘None of these’ if none has the same meaning, or ‘Insufficient information’ if you cannot decide. "
    "After each case, tell us which inspection methods you used, if any."
)
SETUP = (
    "Use only the information in the two supplied ontology files and the study pages. "
    "You can inspect the files with Protégé, a text editor or another method, and combine or change methods between cases."
)
INTRO = (
    "Practise using the tool with an invented example, then answer five short questions. "
    "The example is separate from the study cases. You can retry any exercise or question."
)


def revise(publication, revision, *, now=None):
    """Change copy and its version identity without altering case, asset or eligibility data."""
    result = deepcopy(publication)
    definition = result["definition"]
    if revision == definition["study_revision"]:
        raise ValueError("Choose a new study revision; published studies are immutable")
    if (
        definition["contract_version"] != "exact-study/2.0"
        or definition["form_version"] != "exact-study-forms/3"
    ):
        raise ValueError("This copy revision requires study/2.0 with forms/3")
    definition.update(
        study_revision=revision,
        form_version="exact-study-forms/3-copy-20261110",
        information_version="participant-information-20261110",
        information_text=INFORMATION,
        consent_text=CONSENT,
        instructions=INSTRUCTIONS,
        setup_instructions=SETUP,
        closes_at=((now or datetime.now(timezone.utc)) + timedelta(days=30)).isoformat(),
    )
    tutorial = definition["tutorial"]
    tutorial.update(intro=INTRO, version="tutorial/2.0-native-2-copy-20261110", hash="")
    definition["protocol_versions"].update(
        information=definition["information_version"],
        tutorial=tutorial["version"],
        forms=definition["form_version"],
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    result = revise(json.loads(args.source.read_text()), args.revision)
    # Exclusive creation protects existing publications, including hardlinked sources.
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
