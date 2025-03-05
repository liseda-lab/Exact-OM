from pathlib import Path
from typing import Optional

from matcha_dl import init_jvm

class AlignmentRunner:
    """
    Main class to run the alignment.
    """

    def __init__(
        self,
        source_ontology_file: str,
        target_ontology_file: str,
        output_dir: str,
        reference_file: Optional[str] = None,
        full_reference_file: Optional[str] = None,
        candidates_file: Optional[str] = None,
        config_file: Optional[str] = None,
        save_logs: Optional[bool] = False,
        jvm_heap_size: Optional[str] = "32g",
        run_eval: bool = False,
    ):
        """

        Args:
            source_ontology_file (str): Path to the source ontology file.
            target_ontology_file (str): Path to the target ontology file.
            output_dir (str): Path to the output directory.
            reference_file (str, optional): Path to the reference file. Defaults to None.
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
        self.reference_file = reference_file
        self.full_reference_file = full_reference_file
        self.candidates_file = candidates_file
        self.config_file = config_file
        self.save_logs = save_logs
        self.jvm_heap_size = jvm_heap_size
        self.run_eval = run_eval

    def run_alignment(self) -> None:

        from matcha_dl.core.actions.alignment import AlignmentAction

        AlignmentAction.run(
            source_file_path=Path(self.source_ontology_file).resolve(),
            target_file_path=Path(self.target_ontology_file).resolve(),
            output_dir_path=Path(self.output_dir).resolve(),
            configs_file_path=Path(self.config_file).resolve() if self.config_file else None,
            reference_file_path=Path(self.reference_file).resolve() if self.reference_file else None,
            full_reference_file_path=Path(self.full_reference_file).resolve() if self.full_reference_file else None,
            candidates_file_path=Path(self.candidates_file).resolve() if self.candidates_file else None,
            log_file_path=Path(self.output_dir).resolve() / "matcha_dl.log" if self.save_logs else None,
            run_eval=self.run_eval,
        )

    def validate_files(self) -> None:

        if not Path(self.source_ontology_file).exists():
            raise Exception(f"Source ontology file {self.source_ontology_file} does not exist")
        if not Path(self.target_ontology_file).exists():
            raise Exception(f"Target ontology file {self.target_ontology_file} does not exist")
        if self.reference_file and not Path(self.reference_file).exists():
            raise Exception(f"Reference file {self.reference_file} does not exist")
        if self.full_reference_file and not Path(self.full_reference_file).exists():
            raise Exception(f"Test reference file {self.full_reference_file} does not exist")
        if self.candidates_file and not Path(self.candidates_file).exists():
            raise Exception(f"Candidates file {self.candidates_file} does not exist")
        if not Path(self.output_dir).exists():
            Path(self.output_dir).mkdir(parents=True)

        if self.config_file:
            config_file = Path(self.config_file)
            if not config_file.exists():
                raise Exception(f"Configuration file {self.config_file} does not exist")
            
        if self.jvm_heap_size:
            if self.jvm_heap_size.isdigit():
                self.jvm_heap_size += 'G'
            elif not (self.jvm_heap_size[:-1].isdigit() and self.jvm_heap_size[-1].lower() == 'g'):
                raise Exception(f"JVM heap size {self.jvm_heap_size} is not valid, please provide a valid format")
        else:
            self.jvm_heap_size = '32G'

    def run(self) -> None:

        self.validate_files()

        init_jvm(self.jvm_heap_size)

        self.run_alignment()
