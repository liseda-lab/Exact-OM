from pathlib import Path
from typing import Optional, List

from matcha_dl import init_jvm

class DirectoryAlignmentRunner:
    """
    Main class to run the directory alignment.
    """

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        config_file: Optional[str] = None,
        save_logs: bool = False,
        jvm_heap_size: str = "32g",
        run_eval: bool = False,
        devices: Optional[List[int]] = None,
    ):
        """

        Args:
            data_dir (str): Path to the directory containing OAEI data.
            output_dir (str): Path to the output directory.
            config_file (str, optional): Path to the configuration file. Defaults to None.
            save_logs (bool, optional): Whether to save logs. Defaults to None.
            jvm_heap_size (str, optional): JVM heap size. Defaults to "32g".
            run_eval (bool, optional): Whether to run evaluation. Defaults to False.
            devices (List[int], optional): List of GPU device IDs to use. Defaults to None.
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.config_file = config_file
        self.save_logs = save_logs
        self.jvm_heap_size = jvm_heap_size
        self.run_eval = run_eval
        self.devices = devices

    def run_alignment(self) -> None:
        from matcha_dl.core.actions.alignment import DirectoryAlignmentAction

        DirectoryAlignmentAction.run(
            data_dir=Path(self.data_dir).resolve(),
            output_dir_path=Path(self.output_dir).resolve(),
            configs_file_path=Path(self.config_file).resolve() if self.config_file else None,
            run_eval=self.run_eval,
            save_logs=self.save_logs,
            devices=self.devices
        )

    def validate_files(self) -> None:
        if not Path(self.data_dir).is_dir():
            raise Exception(f"Data directory {self.data_dir} does not exist or is not a directory.")
        if not Path(self.output_dir).exists():
            Path(self.output_dir).mkdir(parents=True)

        if self.config_file:
            config_file = Path(self.config_file)
            if not config_file.exists():
                raise Exception(f"Configuration file {self.config_file} does not exist")
            
        if self.jvm_heap_size.isdigit():
            self.jvm_heap_size += 'G'
        elif not (self.jvm_heap_size[:-1].isdigit() and self.jvm_heap_size[-1].lower() == 'g'):
            raise Exception(f"JVM heap size {self.jvm_heap_size} is not valid, please provide a valid format")

    def run(self) -> None:
        self.validate_files()
        init_jvm(self.jvm_heap_size)
        self.run_alignment()
