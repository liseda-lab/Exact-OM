
import pandas as pd
from pathlib import Path

na_vals = pd.io.parsers.readers.STR_NA_VALUES.difference({"NULL", "null", "n/a"})

def read_table(file_path: Path) -> pd.DataFrame:
    """Read tsv file as pandas dataframe without treating "null" as empty string."""
    sep = "\t" if file_path.endswith(".tsv") else ","
    return pd.read_csv(str(file_path), sep=sep, na_values=na_vals, keep_default_na=False)