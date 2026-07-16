import warnings
from typing import Any, Mapping, Optional

from exact.delivery.common import (
    EvaluationInvocation,
    execute_evaluation,
    prepare_evaluation,
    warn_ignored_jvm,
)


class EvaluationRunner:
    """
    Main class to run the evaluation.
    """

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
        """

        Args:
            alignment_file (str): Path to the alignment file.
            output_dir (str): Path to the output directory.
            error_on_fail (bool, optional): Raise an error if evaluation fails. Defaults to False.
            K (Optional[list], optional): The number of top-K elements to consider in the evaluation. Defaults to [1, 5, 10].
            source_ontology_file (Optional[str], optional): Path to the source ontology file. Defaults to None.
            target_ontology_file (Optional[str], optional): Path to the target ontology file. Defaults to None.
            train_reference_file (Optional[str], optional): Path to the train reference file. Defaults to None.
            full_reference_file (Optional[str], optional): Path to the full reference file. Defaults to None.
            reference_candidates (Optional[str], optional): Path to the reference candidates file. Defaults to None.
            log_file_path (Optional[str], optional): Path to the log file. Defaults to None.
            log_level (Optional[str], optional): Log level. Defaults to None.
            save_logs (Optional[bool], optional): Whether to save logs. Defaults to False.
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
        return execute_evaluation(self._prepare(create_output=True))

    def _prepare(self, *, create_output: bool) -> EvaluationInvocation:
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
        self._prepare(create_output=False)
        warn_ignored_jvm(self.jvm_heap_size, "jvm_heap_size")

    def run(self):
        """
        Run the evaluation.
        """
        self.validate_files()

        return self.run_evaluation()


def __getattr__(name: str):
    if name == "EvalutionRunner":
        warnings.warn(
            "EvalutionRunner is deprecated; use EvaluationRunner instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return EvaluationRunner
    raise AttributeError(name)
