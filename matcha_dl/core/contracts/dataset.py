from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matcha_dl.core.contracts.base import SelfRegisteringComponent, LoggingClass
from matcha_dl.core.entities.registry import ComponentType
from matcha_dl.utils.data import read_table

from mowl.datasets import PathDataset as OWLDataset


# from jpype import java


DataFrame = pd.DataFrame

# TODO look into x an y datatypes might not be compatible with graphdataset
# TODO inherit from logging class


class IDataset(SelfRegisteringComponent, LoggingClass):

    component_type = ComponentType.DATASET

    def __init__(self, 
                 output_path: Path, 
                 matchers: List[str], 
                 **kwargs
        ) -> None:


        self._output_path = output_path / "dataset"
        self._output_path.mkdir(parents=True, exist_ok=True)
        self._matchers = matchers

        self._source = None
        self._target = None
        self._candidates = None
        self._reference = None
        self._negatives = None
        self._matcha_features = None

        self._logger = kwargs.get("logger")
        self._cache_ok = kwargs.get("cache_ok", True)

        LoggingClass.__init__(logger=self._logger)

    @property
    def matchers(self) -> List[str]:
        return self._matchers

    @property
    def source(self) -> OWLDataset:
        return self._source
    
    @property
    def target(self) -> OWLDataset:
        return self._target
    
    @property
    def candidates(self) -> DataFrame:
        return self._candidates
    
    @property
    def reference(self) -> DataFrame:
        return self._reference
    
    @property
    def negatives(self) -> DataFrame:
        return self._negatives
    
    @property
    def matcha_features(self) -> Dict[str, Dict[str, List[float]]]:
        return self._matcha_features
    
    @property
    def output_path(self) -> Path:
        return self._output_path
    
    def load_ontologies(self, source_path: Path, target_path: Path) -> None:
        
        self._source = OWLDataset(source_path)
        
        self.log("#Loaded Source...", level="debug")
        
        self._target = OWLDataset(target_path)
        
        self.log("#Loaded Target...", level="debug")

    def load_candidates(self, file_path: Path) -> None:

        if not file_path.exists():
            self.log(f"Candidates file not found at {file_path}", level="error")
            raise FileNotFoundError(f"Candidates file not found at {file_path}")
                                    
        self._candidates = read_table(file_path)
        self._candidates.columns = ["Src", "Tgt", "Label"]

        self.log("#Loaded Candidates...", level="debug")
              
    def load_reference(self, file_path: Path) -> None:
        
        self._reference = read_table(file_path)
        self._reference.columns = ["Src", "Tgt", "Label"]

        self.log("#Loaded Reference...", level="debug")

    def load_negatives(self, file_path: str) -> None:

        if not file_path.exists():
            self.log(f"Negatives file not found at {file_path}", level="error")
            raise FileNotFoundError(f"Negatives file not found at {file_path}")
        
        self._negatives = read_table(file_path)
        self._negatives.columns = ["Src", "Tgt", "Label"]

        self.log("#Loaded Negatives...", level="debug")

    
    def load_data(self, matcha_features_file: Path) -> None:

        if not matcha_features_file.exists():
            self.log(f"Matcha Features file not found at {matcha_features_file}", level="error")
            raise FileNotFoundError(f"Matcha Features file not found at {matcha_features_file}")

        df = read_table(matcha_features_file)
        df.columns = ["Src", "Tgt"] + self.matchers

        self._matcha_features = {
            src_ent: {
                row["Tgt"]: row[self.matchers].values.tolist()
                for _, row in df[df["Src"] == src_ent].iterrows()
            }
            for src_ent in df["Src"].unique()
        }

        self.log("#Loaded Matcha Features...", level="debug")
    
    def plot_negative_distributions(self,
                                    figsize: Tuple[int, int] = (8, 5),
                                    kde: bool = True,
                                    bins: int = 20,
                                    color: str = "blue",
                                    alpha: float = 0.6,
                                    dpi: int = 300,
                                    grid: bool = True,
                                    **kwargs
        ) -> Path:
        """
        Generates and saves distribution plots for each negative similarity and an average distribution plot.
        """

        # copy negatives
        negatives_df = self.negatives.copy()

        negatives_df = self._get_matcha_features(negatives_df)

        
        # Create a DataFrame of features
        feature_df = pd.DataFrame(negatives_df['Features'].tolist(), columns=self.matchers)
        
        # Plot distribution of each feature
        for feature in self.matchers:
            plt.figure(figsize=figsize)
            sns.histplot(feature_df[feature], kde=kde, bins=bins, color=color, alpha=alpha)
            plt.xlabel("Similarity Score")
            plt.ylabel("Density")
            plt.title(f"Distribution scores according to {feature}")
            plt.grid(grid)
            
            # Save plot
            plot_path = self.output_path / f"{feature}_distribution.png"
            plt.savefig(str(plot_path.resolve()), dpi=dpi)
            plt.close()
        
        # Compute and plot average distribution
        avg_feature_scores = feature_df.mean(axis=1)
        plt.figure(figsize=figsize)
        sns.histplot(avg_feature_scores, kde=kde, bins=bins, color=color, alpha=alpha)
        plt.xlabel("Average Feature Score")
        plt.ylabel("Density")
        plt.title("Average Distribution of All Features")
        plt.grid(grid)
        
        # Save average distribution plot
        avg_plot_path = self.output_path / "average_feature_distribution.png"
        plt.savefig(str(avg_plot_path.resolve()), dpi=dpi)
        plt.close()
        
        self.log(f"Plots saved in: {self.output_path}", level="info")
        return self.output_path

    def _get_matcha_features(self, dataset: pd.DataFrame) -> pd.DataFrame:

        feats = []
        
        for _, row in dataset.iterrows():

            try:
                vector = self.matcha_features.get(row["Src"]).get(row["Tgt"])
            except AttributeError:
                self.log("Matcha features not loaded.", level="error", exc_info=True)
                raise ValueError("Scores for source {} and target {} not found.".format(row["Src"], row["Tgt"]))
            
            feats.append(vector)
            

        dataset["Features"] = feats

        return dataset

    @abstractmethod
    def __getitem__(self, idx: int, kind: str = "train") -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def save(self) -> Path:
        pass

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def has_cache(self) -> None:
        pass

    @abstractmethod
    def process(self) -> None:
        pass

    @abstractmethod
    def x(self, kind: Optional[str] = "train") -> np.ndarray:
        pass

    @abstractmethod
    def y(self, kind="train") -> np.ndarray:
        pass

