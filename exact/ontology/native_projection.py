"""Strict selection of the installed 0.2 projector's encoded native compiler.

Upstream's public ``backend='native'`` permits scalar compilation followed by a
native edge pass. This narrow bridge calls the same compiler, canonical edge
policy, and report builder, but raises before that fallback. It owns no OWL
projection semantics. The private report helper is guarded by an exact API and
package-version contract; upgrading the dependency requires the parity tests.
"""

from __future__ import annotations

from os import PathLike
from time import perf_counter

import pyowl2vec_star_projector as upstream
from pyowl2vec_star_projector.encoded import select_ingestion
from pyowl2vec_star_projector.model import Edge
from pyowl2vec_star_projector.native import (
    NativeEncodedDirectUnsupported,
    native_runtime_metadata,
    prepare_native_encoded_compilation,
)
from pyowl2vec_star_projector.options import Backend, DuplicatePolicy, EdgeOrder
from pyowl2vec_star_projector.streaming import (
    CancellationTokenLike,
    StreamingLimits,
    iter_edge_policy,
)


class NativeProjector(upstream.Projector):
    """Use upstream native compilation exclusively, retaining the input owner.

    Exact exposes only ``project`` and ``project_taxonomy`` through its adapter.
    Inherited upstream streaming methods are not supported Exact entry points.
    """

    def project(
        self, view: object, *, options: upstream.ProjectionOptions | None = None
    ) -> list[Edge]:
        return self._project(view, options or upstream.ProjectionOptions(backend="native"))

    def project_taxonomy(
        self,
        view: object,
        *,
        bidirectional: bool = False,
        duplicates: DuplicatePolicy = "preserve",
        order: EdgeOrder = "canonical",
        backend: Backend = "native",
        buffer_edges: int = 250_000,
        temp_directory: PathLike[str] | None = None,
        streaming_limits: StreamingLimits | None = None,
        cancellation_token: CancellationTokenLike | None = None,
    ) -> list[Edge]:
        return self._project(
            view,
            upstream.ProjectionOptions(
                backend=backend,
                bidirectional_taxonomy=bidirectional,
                only_taxonomy=True,
                duplicates=duplicates,
                order=order,
            ),
            asserted_taxonomy_only=True,
            buffer_edges=buffer_edges,
            temp_directory=temp_directory,
            streaming_limits=streaming_limits,
            cancellation_token=cancellation_token,
        )

    def _project(
        self,
        view: object,
        options: upstream.ProjectionOptions,
        *,
        asserted_taxonomy_only: bool = False,
        buffer_edges: int = 250_000,
        temp_directory: PathLike[str] | None = None,
        streaming_limits: StreamingLimits | None = None,
        cancellation_token: CancellationTokenLike | None = None,
    ) -> list[Edge]:
        if (
            upstream.__version__ != "0.2.0"
            or upstream.PROJECTOR_API_VERSION != 1
            or upstream.COMPILER_CACHE_SCHEMA != "pyowl-projector.compiler-cache/1"
        ):
            raise RuntimeError("native projector bridge requires the tested 0.2.0/API-1 contract")
        if options.backend != "native" or options.compatibility_state != "isolated":
            raise ValueError("Exact requires isolated native projection without fallback")
        if type(buffer_edges) is not int or buffer_edges < 1:
            raise ValueError("buffer_edges must be a positive integer")
        with self._metadata_lock:
            self._last_report = None
            self._last_view = view
        upstream.select_backend("native")
        native_version, features = native_runtime_metadata()
        started = perf_counter()
        ingestion = select_ingestion(view, selected_backend="native", native_features=features)
        publication_seconds = perf_counter() - started
        if ingestion.path != "encoded-native" or ingestion.lease is None:
            raise upstream.NativeBackendUnavailableError(
                f"Exact forbids scalar projection: {ingestion.reason}"
            )
        limits = streaming_limits or StreamingLimits()
        started = perf_counter()
        try:
            compilation, reason = prepare_native_encoded_compilation(
                view,
                ingestion.lease,
                options,
                batch_edges=buffer_edges,
                max_total_edges=limits.max_total_edges,
                cancellation_token=cancellation_token,
                asserted_taxonomy_only=asserted_taxonomy_only,
            )
        except NativeEncodedDirectUnsupported as error:
            raise upstream.NativeBackendUnavailableError(
                f"Exact forbids scalar projection: {error}"
            ) from error
        if compilation is None:
            raise upstream.NativeBackendUnavailableError(
                f"Exact forbids scalar projection: {reason or 'native compiler declined'}"
            )
        compile_seconds = perf_counter() - started
        try:
            edges = list(
                iter_edge_policy(
                    compilation.iter_raw_edges(cancellation_token),
                    duplicates=options.duplicates,
                    order=options.order,
                    buffer_edges=buffer_edges,
                    temp_directory=temp_directory,
                    limits=limits,
                    statistics=compilation.statistics,
                    cancellation_token=cancellation_token,
                    metrics_sink=self._remember_spill_metrics,
                )
            )
            report = self._report(
                view,
                options,
                "direct",
                len(edges),
                compilation.statistics.duplicate_edges,
                compilation.statistics.skipped_axioms,
                compilation.statistics.ignored_shapes,
                compilation.diagnostics,
                None,
                1,
                "native",
                native_version,
                ingestion,
                publication_seconds,
                compile_seconds,
                None,
                compilation,
            )
            with self._metadata_lock:
                self._last_report = report
            return edges
        finally:
            compilation.batches.close()
