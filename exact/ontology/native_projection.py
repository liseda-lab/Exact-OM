"""Exact's strict defaults over the public native projector API."""

from __future__ import annotations

from dataclasses import replace
from os import PathLike

import pyowl2vec_star_projector as upstream
import pyowl_core
from pyowl2vec_star_projector import Edge, StreamingLimits
from pyowl2vec_star_projector.options import Backend, DuplicatePolicy, EdgeOrder
from pyowl2vec_star_projector.streaming import CancellationTokenLike


def require_native_support() -> None:
    """Reject an incompatible installed stack before expensive ontology loading."""
    check = getattr(upstream, "require_native_pipeline_support", None)
    if not callable(check):
        raise upstream.NativeBackendUnavailableError(
            "Exact requires the projector's public native pipeline capability"
        )
    check()
    for name in (
        "AnnotationAssertionIndex",
        "AxiomTypeIndex",
        "ClassFeatureView",
        "AssertedPropertyHierarchyView",
        "PropertyDomainRangeView",
    ):
        view = getattr(pyowl_core, name, None)
        available = getattr(view, "supports_native", None)
        if not callable(available) or not available():
            raise upstream.NativeBackendUnavailableError(
                f"Exact requires native pyowl-core {name} support"
            )


class NativeProjector(upstream.Projector):
    """Require native validation, compilation and canonical output on Exact calls."""

    def project(
        self, view: object, *, options: upstream.ProjectionOptions | None = None
    ) -> list[Edge]:
        selected = options or upstream.ProjectionOptions(backend="native")
        if selected.backend != "native" or selected.compatibility_state != "isolated":
            raise ValueError("Exact requires isolated native projection without fallback")
        return super().project(view, options=replace(selected, require_native_pipeline=True))

    def project_taxonomy(
        self,
        view: object,
        *,
        bidirectional: bool = False,
        duplicates: DuplicatePolicy = "preserve",
        order: EdgeOrder = "canonical",
        backend: Backend = "native",
        require_native_pipeline: bool = True,
        buffer_edges: int = 250_000,
        temp_directory: PathLike[str] | None = None,
        streaming_limits: StreamingLimits | None = None,
        cancellation_token: CancellationTokenLike | None = None,
    ) -> list[Edge]:
        if backend != "native" or require_native_pipeline is not True:
            raise ValueError("Exact requires native projection without fallback")
        return super().project_taxonomy(
            view,
            bidirectional=bidirectional,
            duplicates=duplicates,
            order=order,
            backend=backend,
            require_native_pipeline=True,
            buffer_edges=buffer_edges,
            temp_directory=temp_directory,
            streaming_limits=streaming_limits,
            cancellation_token=cancellation_token,
        )
