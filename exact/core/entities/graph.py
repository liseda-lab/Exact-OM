from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Edge:
    """A backend-neutral projected graph edge."""

    src: str
    rel: str
    dst: str

    def astuple(self) -> tuple[str, str, str]:
        return self.src, self.rel, self.dst


@dataclass(frozen=True, slots=True)
class AnnotationValue:
    """A normalized annotation assertion value."""

    property_iri: str
    value: str
    is_literal: bool
    lang: str | None = None
    datatype: str | None = None
