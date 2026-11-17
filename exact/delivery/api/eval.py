import warnings
from typing import Any, Mapping, Optional

from exact.delivery.common import (
    EvaluationInvocation,
    execute_evaluation,
    prepare_evaluation,
    warn_ignored_jvm,
)


class EvaluationRunner:
    """Validate and execute one or more registered evaluation backends."""

    def __init__(
        self,
        alignment_file: str,
        output_dir: str,
        error_on_fail: bool = False,
        K: Optional[list] = None,
        source_ontology_file: Optional[str] = None,
        target_ontology_file: Optional[str] = None,
        train_reference_file: Optional[str] = None,
        full_reference_file: Optional[str] = None,
        reference_candidates: Optional[str] = None,
        log_level: Optional[str] = None,
        save_logs: bool = False,
        jvm_heap_size: Optional[str] = None,
        backends: Optional[list[str]] = None,
        backend_options: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ):
        """Create a reusable evaluation runner.

        Args:
            alignment_file: Mapping or ranking file to evaluate.
            output_dir: Existing directory for evaluation reports.
            error_on_fail: Raise backend failures instead of reporting them.
            K: Ranking cutoffs; defaults to 1, 5, and 10.
            source_ontology_file: Optional source ontology used by a backend.
            target_ontology_file: Optional target ontology used by a backend.
            train_reference_file: Optional training mappings excluded from scoring.
            full_reference_file: Optional complete reference alignment.
            reference_candidates: Optional candidate-ranking reference.
            log_level: Optional logging-level override.
            save_logs: Write the evaluation log under ``output_dir``.
            jvm_heap_size: Deprecated compatibility argument; ignored.
            backends: Ordered evaluator registry names.
            backend_options: Per-backend option mappings.
        """
        self.alignment_file = alignment_file
        self.output_dir = output_dir
        self.error_on_fail = error_on_fail
        self.K = K
        self.source_ontology_file = source_ontology_file
        self.target_ontology_file = target_ontology_file
        self.train_reference_file = train_reference_file
        self.full_reference_file = full_reference_file
        self.reference_candidates = reference_candidates
        self.log_level = log_level
        self.save_logs = save_logs
        self.jvm_heap_size = jvm_heap_size
        self.backends = list(backends or ["builtin"])
        self.backend_options = dict(backend_options or {})

    def run_evaluation(self):
        """Execute evaluation using the prepared request."""

        return execute_evaluation(self._prepare(create_output=True))

    def _prepare(self, *, create_output: bool) -> EvaluationInvocation:
        """Resolve paths and options into the shared delivery request."""

        return prepare_evaluation(
            alignment_file=self.alignment_file,
            output_dir=self.output_dir,
            error_on_fail=self.error_on_fail,
            k=self.K if self.K else None,
            source_ontology_file=self.source_ontology_file,
            target_ontology_file=self.target_ontology_file,
            train_reference_file=self.train_reference_file,
            full_reference_file=self.full_reference_file,
            reference_candidates=self.reference_candidates,
            log_level=self.log_level,
            save_logs=self.save_logs,
            backends=self.backends,
            backend_options=self.backend_options,
            create_output=create_output,
            log_filename="oaei_bio_ml_eval.log",
            train_reference_label="Train reference file",
        )

    def validate_files(self) -> None:
        """Raise when an input or output path is invalid."""

        self._prepare(create_output=False)
        warn_ignored_jvm(self.jvm_heap_size, "jvm_heap_size")

    def run(self):
        """Validate inputs and return the combined evaluation result."""
        self.validate_files()

        return self.run_evaluation()


def __getattr__(name: str):
    """Resolve the historical misspelled runner alias with a warning."""

    if name == "EvalutionRunner":
        warnings.warn(
            "EvalutionRunner is deprecated; use EvaluationRunner instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return EvaluationRunner
    raise AttributeError(name)
