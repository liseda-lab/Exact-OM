from typing import Optional

from exact.delivery.common import (
    AlignmentInvocation,
    execute_alignment,
    prepare_alignment,
    warn_ignored_jvm,
)


class AlignmentRunner:
    """
    Main class to run the alignment.
    """

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
    ):
        """

        Args:
            source_ontology_file (str): Path to the source ontology file.
            target_ontology_file (str): Path to the target ontology file.
            output_dir (str): Path to the output directory.
            training_reference_file (str, optional): Path to the training reference file. Defaults to None.
            full_reference_file (str, optional): Path to the full reference file. Defaults to None.
            If full reference is provided, it will be used for evaluation. If not provided, no evaluation will be performed.
            candidates_file (str, optional): Path to the candidates file. Defaults to None.
            config_file (str, optional): Path to the configuration file. Defaults to None.
            save_logs (bool, optional): Whether to save logs. Defaults to None.
            run_eval (bool, optional): Whether to run evaluation. Defaults to False.
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

    def run_alignment(self) -> None:
        execute_alignment(self._prepare())

    def _prepare(self) -> AlignmentInvocation:
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
        )

    def validate_files(self) -> None:
        self._prepare()

    def run(self) -> None:
        warn_ignored_jvm(self.jvm_heap_size, "jvm_heap_size")
        self.validate_files()
        self.run_alignment()
