"""Format-dispatching knowledge sources and alignment writers."""

from .sources import resolve as resolve_source
from .writers import write as write_alignment

__all__ = ["resolve_source", "write_alignment"]
