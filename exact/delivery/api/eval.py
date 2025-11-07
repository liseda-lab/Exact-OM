from pathlib import Path
from typing import Optional

from exact import init_jvm

class EvalutionRunner:
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
        jvm_heap_size: str = "32g",
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

    def run_evaluation(self) -> None:

        from exact.core.actions.evaluation import EvaluationAction

        EvaluationAction.run(
            alignment=Path(self.alignment_file).resolve(),
            output_dir_path=Path(self.output_dir).resolve(),
            error_on_fail=self.error_on_fail,
            K=self.K if self.K else None,
            source_file_path=Path(self.source_ontology_file).resolve() if self.source_ontology_file else None,
            target_file_path=Path(self.target_ontology_file).resolve() if self.target_ontology_file else None,
            train_reference_file_path=Path(self.train_reference_file).resolve() if self.train_reference_file else None,
            full_reference_file_path=Path(self.full_reference_file).resolve() if self.full_reference_file else None,
            reference_candidates=Path(self.reference_candidates).resolve() if self.reference_candidates else None,
            log_file_path=Path(self.output_dir).resolve() / "oaei_bio_ml_eval.log" if self.save_logs else None,
            log_level=self.log_level,
        )

    def validate_files(self) -> None:
        
        if not Path(self.alignment_file).exists():
            raise FileNotFoundError(f"Alignment file {self.alignment_file} does not exist")
        if not Path(self.output_dir).exists():
            raise FileNotFoundError(f"Output directory {self.output_dir} does not exist")
        if self.source_ontology_file and not Path(self.source_ontology_file).exists():
            raise FileNotFoundError(f"Source ontology file {self.source_ontology_file} does not exist")
        if self.target_ontology_file and not Path(self.target_ontology_file).exists():
            raise FileNotFoundError(f"Target ontology file {self.target_ontology_file} does not exist")
        if self.train_reference_file and not Path(self.train_reference_file).exists():
            raise FileNotFoundError(f"Train reference file {self.train_reference_file} does not exist")
        if self.full_reference_file and not Path(self.full_reference_file).exists():
            raise FileNotFoundError(f"Full reference file {self.full_reference_file} does not exist")
        if self.reference_candidates and not Path(self.reference_candidates).exists():
            raise FileNotFoundError(f"Reference candidates file {self.reference_candidates} does not exist")
        
        if self.jvm_heap_size.isdigit():
            self.jvm_heap_size += 'G'
        elif not (self.jvm_heap_size[:-1].isdigit() and self.jvm_heap_size[-1].lower() == 'g'):
            raise Exception(f"JVM heap size {self.jvm_heap_size} is not valid, please provide a valid format")
        
    def run(self) -> None:
        """
        Run the evaluation.
        """
        self.validate_files()
        
        init_jvm(self.jvm_heap_size)

        self.run_evaluation()