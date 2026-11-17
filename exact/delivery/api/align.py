from typing import Any, Mapping, Optional, Sequence

from exact.delivery.common import (
    AlignmentInvocation,
    execute_alignment,
    prepare_alignment,
    warn_ignored_jvm,
)


class AlignmentRunner:
    """Validate and execute one ontology or knowledge-graph alignment."""

    def __init__(
        self,
        source_ontology_file: str,
        target_ontology_file: str,
        output_dir: str,
        training_reference_file: Optional[str] = None,
        full_reference_file: Optional[str] = None,
        candidates_file: Optional[str] = None,
        config_file: Optional[str] = None,
        save_logs: bool = False,
        jvm_heap_size: Optional[str] = None,
        run_eval: bool = False,
        device: Optional[
            int
        ] = None,  # Device for running the alignment, e.g., 0 for GPU 0, None for CPU
        input_format: Optional[str] = None,
        source_options: Optional[Mapping[str, Any]] = None,
        target_options: Optional[Mapping[str, Any]] = None,
        output_formats: Optional[Sequence[str]] = None,
        relation_prediction: Optional[str] = None,
    ):
        """Create a reusable alignment runner.

        Args:
            source_ontology_file: Source knowledge-source path.
            target_ontology_file: Target knowledge-source path.
            output_dir: Directory in which to create a versioned run.
            training_reference_file: Optional training or calibration mapping.
            full_reference_file: Optional complete evaluation reference.
            candidates_file: Optional one-to-many candidate file. Supplying it selects
                local-ranking mode.
            config_file: Optional version-1 or version-2 YAML configuration.
            save_logs: Write ``exact.log`` in the run directory.
            jvm_heap_size: Deprecated compatibility argument; ignored.
            run_eval: Evaluate the saved alignment after inference.
            device: CUDA device index, or ``None`` for CPU.
            input_format: Input adapter name, or ``None`` to use the config.
            source_options: Source-adapter-specific option overrides.
            target_options: Target-adapter-specific option overrides.
            output_formats: Ordered writer-name overrides.
            relation_prediction: Optional relation-typing mode override.
        """
        self.source_ontology_file = source_ontology_file
        self.target_ontology_file = target_ontology_file
        self.output_dir = output_dir
        self.training_reference_file = training_reference_file
        self.full_reference_file = full_reference_file
        self.candidates_file = candidates_file
        self.config_file = config_file
        self.save_logs = save_logs
        self.jvm_heap_size = jvm_heap_size
        self.run_eval = run_eval
        self.device = device
        self.input_format = input_format
        self.source_options = dict(source_options) if source_options is not None else None
        self.target_options = dict(target_options) if target_options is not None else None
        self.output_formats = list(output_formats) if output_formats is not None else None
        self.relation_prediction = relation_prediction

    def run_alignment(self) -> None:
        """Execute alignment using the already validated invocation."""

        execute_alignment(self._prepare())

    def _prepare(self) -> AlignmentInvocation:
        """Resolve paths and configuration into the shared delivery request."""

        return prepare_alignment(
            source_ontology_file=self.source_ontology_file,
            target_ontology_file=self.target_ontology_file,
            output_dir=self.output_dir,
            training_reference_file=self.training_reference_file,
            full_reference_file=self.full_reference_file,
            candidates_file=self.candidates_file,
            config_file=self.config_file,
            save_logs=self.save_logs,
            run_eval=self.run_eval,
            device=self.device,
            full_reference_label="Test reference file",
            input_format=self.input_format,
            source_options=self.source_options,
            target_options=self.target_options,
            output_formats=self.output_formats,
            relation_prediction=self.relation_prediction,
        )

    def validate_files(self) -> None:
        """Raise when an input path or output policy is invalid."""

        self._prepare()

    def run(self) -> None:
        """Validate inputs and execute the alignment."""

        warn_ignored_jvm(self.jvm_heap_size, "jvm_heap_size")
        self.validate_files()
        self.run_alignment()
