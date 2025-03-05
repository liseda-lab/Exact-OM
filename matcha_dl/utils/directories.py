from pathlib import Path
import re
from typing import List

from matcha_dl.core.entities.directories import OEAIDir

class OEAIDirSearcher:
    """
    Searches a given directory for subdirectories following the pattern 'some_source-some_target'.
    
    A valid subdirectory must contain the following files:
      - some_source.owl
      - some_target.owl
      - train.tsv
      - full.tsv
      - test.cands.tsv
    
    It may optionally contain a configuration file with a .yaml extension.
    
    Args:
        base_dir (str): The base directory to search in.
    
    Returns:
        List[OEAIDir]: A list of found directories matching the criteria.
    """
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
    
    def find_oeai_dirs(self) -> List[OEAIDir]:
        """
        Finds directories that match the naming convention and contain the required files.
        
        Returns:
            List[OEAIDir]: A list of valid OEAIDir objects.
        """
        oeai_dirs = []
        pattern = re.compile(r'^(?P<source>[^-]+)-(?P<target>[^-]+)$')
        
        for dir_path in self.base_dir.rglob('*'):
            if dir_path.is_dir() and pattern.match(dir_path.name):
                match = pattern.match(dir_path.name)
                if match:
                    source_name, target_name = match.group("source"), match.group("target")
                    
                    # Required files
                    source_file = dir_path / f"{source_name}.owl"
                    target_file = dir_path / f"{target_name}.owl"
                    train_file = dir_path / "train.tsv"
                    full_file = dir_path / "full.tsv"
                    candidates_file = dir_path / "test.cands.tsv"
                    
                    if all(f.exists() for f in [source_file, target_file, train_file, full_file, candidates_file]):
                        oeai_dirs.append(OEAIDir(
                            data_name=dir_path.name,
                            source_file_path=source_file,
                            target_file_path=target_file,
                            reference_file_path=train_file,
                            full_reference_file_path=full_file,
                            candidates_file_path=candidates_file
                        ))
        return oeai_dirs