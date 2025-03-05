
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class OEAIDir:
    """
    Dataclass representing a directory containing necessary files for OEAI processing.
    
    Attributes:
        source_file_path (Path): Path to the source ontology file.
        target_file_path (Path): Path to the target ontology file.
        output_dir_path (Path): Path to the directory containing these files.
        configs_file_path (Optional[Path]): Path to an optional configuration YAML file.
        reference_file_path (Path): Path to the train.tsv file.
        full_reference_file_path (Path): Path to the full.tsv file.
        candidates_file_path (Path): Path to the test.cands.tsv file.
    """
    source_file_path: Path
    target_file_path: Path
    output_dir_path: Path
    configs_file_path: Optional[Path]
    reference_file_path: Path
    full_reference_file_path: Path
    candidates_file_path: Path