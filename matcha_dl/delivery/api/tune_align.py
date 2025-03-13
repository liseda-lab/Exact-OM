from pathlib import Path
from typing import Optional

from matcha_dl import init_jvm

class TuningAlignmentRunner:
    """
    Main class to run the tuning alignment.
    """

    def __init__(
        self,
        source_ontology_file: str,
        target_ontology_file: str,
        output_dir: str,
        reference_file: str,
        full_reference_file: str,
        candidates_file: str,
        config_file: str,
        save_logs: bool = False,
        devices: Optional[list[int]] = None,
        max_workers: Optional[int] = None,
        max_combinations: Optional[int] = None,
        jvm_heap_size: str = "32G",
    ):
        """
        Args:
            source_ontology_file (str): Path to the source ontology file.
            target_ontology_file (str): Path to the target ontology file.
            output_dir (str): Path to the output directory.
            reference_file (str): Path to the reference file.
            full_reference_file (str): Path to the full reference file.
            candidates_file (str): Path to the candidates file.
            config_file (str): Path to the configuration file.
            save_logs (bool, optional): Whether to save logs. Defaults to False.
            run_eval (bool, optional): Whether to run evaluation. Defaults to False.
            devices (list, optional): List of GPU device IDs to use (if None CPU is user)".
            max_workers (int, optional): Maximum number of workers to use (if None all available are used).
            max_combinations (int, optional): Maximum number of combinations to evaluate (if None all combinations are used).
            jvm_heap_size (str, optional): JVM heap size. Defaults to "32G".
        """
        self.source_ontology_file = source_ontology_file
        self.target_ontology_file = target_ontology_file
        self.output_dir = output_dir
        self.reference_file = reference_file
        self.full_reference_file = full_reference_file
        self.candidates_file = candidates_file
        self.config_file = config_file
        self.save_logs = save_logs
        self.devices = devices
        self.max_workers = max_workers
        self.max_combinations = max_combinations
        self.jvm_heap_size = jvm_heap_size

    def run_tuning_alignment(self) -> None:
        from matcha_dl.core.actions.alignment import TuningAlignmentAction

        TuningAlignmentAction.run(
            source_file_path=Path(self.source_ontology_file).resolve(),
            target_file_path=Path(self.target_ontology_file).resolve(),
            output_dir_path=Path(self.output_dir).resolve(),
            configs_file_path=Path(self.config_file).resolve(),
            reference_file_path=Path(self.reference_file).resolve(),
            full_reference_file_path=Path(self.full_reference_file).resolve(),
            candidates_file_path=Path(self.candidates_file).resolve(),
            save_logs=self.save_logs,
            devices=self.devices,
            max_workers=self.max_workers,
            max_combinations=self.max_combinations,
        )

    def validate_files(self) -> None:
        if not Path(self.source_ontology_file).exists():
            raise Exception(f"Source ontology file {self.source_ontology_file} does not exist")
        if not Path(self.target_ontology_file).exists():
            raise Exception(f"Target ontology file {self.target_ontology_file} does not exist")
        if not Path(self.reference_file).exists():
            raise Exception(f"Reference file {self.reference_file} does not exist")
        if not Path(self.full_reference_file).exists():
            raise Exception(f"Full reference file {self.full_reference_file} does not exist")
        if not Path(self.candidates_file).exists():
            raise Exception(f"Candidates file {self.candidates_file} does not exist")
        if not Path(self.config_file).exists():
            raise Exception(f"Configuration file {self.config_file} does not exist")
        if not Path(self.output_dir).exists():
            Path(self.output_dir).mkdir(parents=True)

        if self.jvm_heap_size.isdigit():
            self.jvm_heap_size += 'G'
        elif not (self.jvm_heap_size[:-1].isdigit() and self.jvm_heap_size[-1].lower() == 'g'):
            raise Exception(f"JVM heap size {self.jvm_heap_size} is not valid, please provide a valid format")

    def run(self) -> None:
        self.validate_files()
        init_jvm(self.jvm_heap_size)
        self.run_tuning_alignment()
