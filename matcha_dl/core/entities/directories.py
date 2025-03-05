
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class OEAIDir:
    """
    Dataclass representing a directory containing necessary files for OEAI processing.
    
    Attributes:
        data_name (str): Name of the directory.
        source_file_path (Path): Path to the source ontology file.
        target_file_path (Path): Path to the target ontology file.
        reference_file_path (Path): Path to the train.tsv file.
        full_reference_file_path (Path): Path to the full.tsv file.
        candidates_file_path (Path): Path to the test.cands.tsv file.
    """
    data_name: str
    source_file_path: Path
    target_file_path: Path
    reference_file_path: Path
    full_reference_file_path: Path
    candidates_file_path: Path