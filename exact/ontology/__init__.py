"""Java-free OWL ontology backend."""

from pathlib import Path
from typing import Sequence

from exact.ontology.store import OwlOntologySource


def load_ontology(
    path: Path, *, label_properties: Sequence[str] | None = None
) -> OwlOntologySource:
    """Load and index an OWL ontology document."""

    return OwlOntologySource.from_path(Path(path), label_properties=label_properties)


__all__ = ["OwlOntologySource", "load_ontology"]
