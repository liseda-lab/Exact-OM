import logging
import os
import queue
import select
import subprocess
import sys
import threading
import time
from abc import abstractmethod
from ast import literal_eval
from pathlib import Path
from typing import List, Optional
import random

import pandas as pd

from matcha_dl.core.contracts import LoggingClass
from matcha_dl.utils.data import read_table
from matcha_dl.core.entities.configs.matcha import Matchers, Sampler


class IMatcha(LoggingClass):
    def __init__(
        self,
        threshold: float,
        max_heap: str,
        output_path: Path,
        matchers: List[Matchers],
        negthreshold: float,
        samplers: Sampler,
        negcardinality: Optional[int] = None,
        generate_reference: bool = False,
        min_threshold_hard_positives: float = 0,
        min_threshold_soft_positives: float = 0,
        hard_positives: float = 0,
        soft_positives: float = 0,
        calculate_scores: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize Matcher.

        Args:
            threshold (float): The threshold to use for matching.
            output_path (Path): The path to the output directory.
            matchers (List[str]): The list of matchers to use.
            **kwargs: Additional keyword arguments.
        """

        self._threshold = threshold
        self._max_heap = max_heap
        self._output_path = output_path / "matcha"
        self._output_path.mkdir(parents=True, exist_ok=True)
        self._matchers = matchers
        self._negcardinality = negcardinality
        self._negthreshold = negthreshold
        self._samplers = samplers
        self._generate_reference = generate_reference
        self.min_threshold_hard_positives: float = min_threshold_hard_positives
        self.min_threshold_soft_positives: float = min_threshold_soft_positives
        self.hard_positives: float = hard_positives
        self.soft_positives: float = soft_positives
        self._calculate_scores = calculate_scores

        self._source = None
        self._target = None
        self._reference = None
        self._candidates = self.output_path / "candidates.tsv"
        self._negatives = self.output_path / "negatives.tsv"
        self._matcha_features = self.output_path / "matcha_features.tsv"
        self._log_file = self.output_path / "matcha_error.log"
        self._generated_reference = self.output_path / "reference.tsv"

        # If reference was generated and cached before, use it
        self._reference = self._generated_reference if self._generated_reference.exists() else None

        self._cache_ok = kwargs.get("cache_ok", True)

        LoggingClass.__init__(self, logger=kwargs.get("logger"))

    @property
    @abstractmethod
    def matcha_path(self) -> Path:
        """
        Get the path to the matcha directory.

        Returns:
            Path: The path to the matcha directory.
        """
        pass

    @property
    @abstractmethod
    def jar_path(self) -> Path:
        """
        Get the path to the matcha.jar file.

        Returns:
            Path: The path to the matcha.jar file.
        """
        pass

    @property
    def log_file(self) -> Path:
        return self._log_file

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def matchers(self) -> List[str]:
        return self._matchers

    @property
    def source(self) -> Path:
        return self._source
    
    @property
    def target(self) -> Path:
        return self._target
    
    @property
    def reference(self) -> Path:
        return self._reference
    
    @property
    def candidates(self) -> Path:
        return self._candidates
    
    @property
    def negatives(self) -> Path:
        return self._negatives
    
    @property
    def matcha_features(self) -> Path:
        return self._matcha_features
    
    @property
    def output_path(self) -> Path:
        return self._output_path
    
    @property
    def generated_reference(self) -> Path:
        return self._generated_reference

    @property
    def has_scores(self) -> bool:
        if self.calculate_scores:
            if self._cache_ok:
                return self.matcha_features.exists()
            return False
        return True
    
    @property
    def has_negatives(self) -> bool:
        if self.reference is not None and self.reference.exists():
            if self._cache_ok:
                return self.negatives.exists()
            return False
        return True
    
    @property
    def has_cache(self) -> bool:
        return self.candidates.exists() and self.has_scores and self.has_negatives and not self.generate_reference
        
    @property
    def max_heap(self) -> str:
        return self._max_heap

    @property
    def negcardinality(self) -> int:
        if self._negcardinality is None:
            # Calculate reference cardinality if not provided
            if self.reference is not None and self.reference.exists():
                df = read_table(self.reference)
                df.columns = ["Src", "Tgt", "Label"]
                
                # Get average number of target entities per unique source
                self._negcardinality = int(df.groupby("Src")["Tgt"].nunique().mean())

                self.log(f"No negcardinality provided. Calculated average cardinality in reference set: {self._negcardinality}", level="debug")

        return self._negcardinality
    
    @property
    def negthreshold(self) -> float:
        return self._negthreshold
    
    @property
    def seed(self) -> int:
        return random.randint(0, 2**32 - 1)
    
    @property
    def samplers(self) -> int:
        return self._samplers.value
    
    @property
    def generate_reference(self) -> bool:
        return self._generate_reference if self.reference is None and not self.generated_reference.exists() else False
    
    @property
    def calculate_scores(self) -> bool:
        return self._calculate_scores
    
    def load_ontologies(self, source_path: Path, target_path: Path) -> None:

        self._source = source_path
        self._target = target_path

        self.log("#Loaded Ontologies Path...", level="debug")

    def load_reference(self, file_path: Path) -> None:
        self._reference = file_path

        self.log("#Loaded Reference Path...", level="debug")

    def load_candidates(self, file_path: Path) -> None:

        def get_cands(df: pd.DataFrame) -> pd.DataFrame:

            return pd.DataFrame([
                    [source, cand, 0]
                    for source, _, target_cands in df.values
                    for cand in literal_eval(target_cands)
                ], columns=["Src", "Tgt", "Score"])

        # Load One2Many candidates file
        candidates = read_table(str(file_path))
        candidates.columns = ["Src", "Tgt", "Candidates"]

        # Get One2One candidates df
        candidates = get_cands(candidates)

        # Save One2One candidates
        candidates.to_csv(self.candidates, sep="\t", index=False)

        self.log("#Loaded Candidates Path...", level="debug")

    def match(self) -> None:

        def read_output(process, output_queue, stop_event):
            while not stop_event.is_set():
                # Use select to wait for the process's stdout to be ready for reading
                ready, _, _ = select.select([process.stdout], [], [], 1.0)
                if ready:
                    line = process.stdout.readline().strip()
                    if line:
                        output_queue.put(line)
                    else:
                        break

        def wait_for_reply(matcha_process, termination):

            # Create a queue to hold the output
            output_queue = queue.Queue()

            # Create an Event object to signal the thread to stop
            stop_event = threading.Event()

            # Start a thread to read the output
            output_thread = threading.Thread(target=read_output, args=(matcha_process, output_queue, stop_event))
            output_thread.start()

            # Wait for a response from the process
            while matcha_process.poll() is None:
                try:
                    # Get a line of output from the queue
                    line = output_queue.get(timeout=1)
                    self.log(f"[Matcha] {line}", level="debug")
                    # Check for a specific response indicating the command is done
                    if termination.lower() in line.lower():
                        break
                except queue.Empty:
                    # No output received within the timeout period
                    continue

            # Signal the output thread to stop
            stop_event.set()

            # Ensure the output thread is finished
            output_thread.join()

            if matcha_process.poll() == 0:
                self.log(f"Matcha process finished without error code unespectedly", level="error")
            elif matcha_process.poll() is not None:
                self.log(f"Matcha process finished with error code {matcha_process.returncode}", level="error")
                raise RuntimeError(f"Matcha subprocess returned with error code {matcha_process.returncode} check error log at {self.log_file}")

        def comunicate_matcha_process(matcha_process, input, termination=None):
            self.log(f"Running command in Matcha: {input}", level="debug")
            matcha_process.stdin.write(input + "\n")
            matcha_process.stdin.flush()

            if termination is None:
                time.sleep(1)
                return

            wait_for_reply(matcha_process, termination)

        
        def add_matchers(matcha_process):
            command = f"Matchers {{{', '.join(self.matchers)}}}"
            comunicate_matcha_process(matcha_process, command, 'matchers set')

        def generate_negatives(matcha_process):

            command = f"Negatives {self.reference} {self.negatives} {self.negcardinality} {self.negthreshold} {self.samplers} {self.seed}"
            comunicate_matcha_process(matcha_process, command, 'finished generating negatives')

        def generate_candidates(matcha_process):
            command = f"Match {self.threshold} {self.candidates}"
            comunicate_matcha_process(matcha_process, command, 'finished matching')

        def generate_scores(matcha_process, pairs_file, save_path=None):
            save_path = self.matcha_features if save_path is None else save_path

            command = f"Score {pairs_file} {save_path}"
            comunicate_matcha_process(matcha_process, command, 'finished calculating scores')

        if self.has_cache:

            self.log(
                f"All required files cached. Skipping computation.",
                level="info",
            )

            return

        if self.source is None or self.target is None:
            self.log("Ontologies not loaded", level="error")
            raise FileNotFoundError("Ontologies not loaded")
        
        # Main Matcha Execution

        current_cwd = os.getcwd()
        os.chdir(self.matcha_path)

        jar_command = [
            "java",
            "-jar",
            f"-Xmx{self.max_heap}",
            str(self.jar_path),
            '-s', str(self.source),
            '-t', str(self.target),
            '-p', sys.executable,
        ]

        try:

            # Load Matcha jar file with ontologies

            self.log(f"Running Matcha with command: {jar_command}", level="debug")

            with open(self.log_file, "w") as f:

                matcha_process = subprocess.Popen(jar_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=f, text=True)

                wait_for_reply(matcha_process, 'Matcha CLI activated')

                # If candidates exist, skip, otherwise get candidates with matcha Match command

                if not self.candidates.exists():
                    generate_candidates(matcha_process)

                    if not self.candidates.exists():
                        self.log(f"Matcha failed to generate candidates at {self.candidates}", level="error")
                        raise FileNotFoundError(f"Matcha failed to generate candidates at {self.candidates}")

                else:
                    self.log(f"Matcha candidates already exist at {self.candidates}. Skipping computation.", level="info")

                # If reference exists, get negatives from reference, otherwirse skip

                if not self.has_negatives:
                    generate_negatives(matcha_process)
                    if not self.negatives.exists():
                        self.log(f"Matcha failed to generate negatives at {self.negatives}", level="error")
                        raise FileNotFoundError(f"Matcha failed to generate negatives at {self.negatives}")
                    else:
                        self.log(f"Matcha negatives generated at {self.negatives}", level="debug")
                else:
                    self.log(f"Skipping generating negatives", level="info")

                # Compile all files to generate scores into a single file (reference/negatives/candidates)

                if not self.has_scores:

                    pairs_file = self.output_path / "pairs.tsv"

                    with open(pairs_file, "w") as pairs:
                        with open(self.candidates, "r") as candidates:
                            pairs.write(candidates.read())
                        if self.reference is not None and self.reference.exists():
                            with open(self.reference, "r") as reference:
                                next(reference)  # Skip header
                                pairs.write(reference.read())
                            with open(self.negatives, "r") as negatives:
                                next(negatives)  # Skip header
                                pairs.write(negatives.read())

                    # Load matchers to use

                    add_matchers(matcha_process)

                    # Use Scores command to get scores from matcha

                    generate_scores(matcha_process, pairs_file)

                    if not self.matcha_features.exists():
                        self.log(f"Matcha failed to generate matcha features at {self.matcha_features}", level="error")
                        raise FileNotFoundError(f"Matcha failed to generate matcha features at {self.matcha_features}")
                    else:
                        self.log(f"Matcha features generated at {self.matcha_features}", level="debug")
                    
                else:
                    self.log(f"Skipping generating matcha features...", level="info")

                # If generate reference is True and reference does not exist, generate it

                if self.generate_reference:
                    self.get_generated_reference()
                    if not self.generated_reference.exists():
                        self.log(f"Matcha failed to generate reference at {self.generated_reference}", level="error")
                        raise FileNotFoundError(f"Matcha failed to generate reference at {self.generated_reference}")
                    else:
                        # Generate negatives from generated reference
                        generate_negatives(matcha_process)
                        if not self.negatives.exists():
                            self.log(f"Matcha failed to generate negatives at {self.negatives}", level="error")
                            raise FileNotFoundError(f"Matcha failed to generate negatives at {self.negatives}")
                        else:
                            self.log(f"Matcha negatives generated at {self.negatives}", level="debug")

                            neg_scores_path = self.output_path / "neg_scores_for_gen_ref.tsv"

                            generate_scores(matcha_process, self.negatives, neg_scores_path)

                            if not neg_scores_path.exists():
                                self.log(f"Matcha failed to generate negative scores at {neg_scores_path}", level="error")
                                raise FileNotFoundError(f"Matcha failed to generate negative scores at {neg_scores_path}")
                            else:
                                self.log(f"Matcha negative scores generated at {neg_scores_path}", level="debug")

                            # Add negative scores to matcha features file skipping first row (header)
                            with open(self.matcha_features, "a") as matcha_features_file:
                                with open(neg_scores_path, "r") as neg_scores_file:
                                    next(neg_scores_file)
                                    matcha_features_file.write(neg_scores_file.read())

                            self.log(f"Matcha features updated with negative scores at {self.matcha_features}", level="debug")

        except subprocess.CalledProcessError as e:
            self.log(f"Matcha subprocess returned with error code {e.returncode}", level="error")
            raise RuntimeError(f"Matcha subprocess returned with error code {e.returncode}")

        finally:

            # stop matcha process
            matcha_process.terminate()

            # change back to original directory
            os.chdir(current_cwd)

        return
    
    def get_generated_reference(self) -> None:
        """
        Generate a reference file from the given path.

        Args:
            reference_path (Path): The path to the reference file.
        """

        # Load Matcha Features file

        self.log("Generating reference for self-supervised setting...", level="debug")

        if not self.matcha_features.exists():
            self.log(f"Matcha Features not generated.. cannot be found at {self.matcha_features}", level="error")
            raise FileNotFoundError(f"Matcha Features file not found at {self.matcha_features}")

        features_df = read_table(self.matcha_features)
        features_df.columns = ["Src", "Tgt"] + self.matchers

        candidates_df = read_table(self.candidates)
        candidates_df.columns = ["Src", "Tgt", "Label"]

        # Merge Matcha Features with Candidates
        merged_df = pd.merge(features_df, candidates_df, on=["Src", "Tgt"], how="inner")

        # Keep only the max value has score for each Src-Tgt pair from the matchers columns
        merged_df["maxscore"] = merged_df[self.matchers].max(axis=1)
        merged_df = merged_df[["Src", "Tgt", "maxscore"]]

        # Filter by threshold label has hard positives soft positives or drop values below both thresholds
        if self.min_threshold_hard_positives > 0:
            hard_positives_df = merged_df[merged_df["maxscore"] >= self.min_threshold_hard_positives].copy()
            hard_positives_df["Score"] = 1
            hard_positives_df = hard_positives_df[["Src", "Tgt", "Score"]]
        else:
            hard_positives_df = pd.DataFrame(columns=["Src", "Tgt", "Label"])

        if self.min_threshold_soft_positives > 0:
            soft_positives_df = merged_df[merged_df["maxscore"] >= self.min_threshold_soft_positives].copy()
            soft_positives_df["Score"] = 1
            soft_positives_df = soft_positives_df[["Src", "Tgt", "Score"]]
        else:
            soft_positives_df = pd.DataFrame(columns=["Src", "Tgt", "Label"])
        if hard_positives_df.empty and soft_positives_df.empty:
            self.log("No hard or soft positives found. No reference generated.", level="warning")
            return
        
        self.log(f"Hard positives found: {len(hard_positives_df)}", level="debug")
        self.log(f"Soft positives found: {len(soft_positives_df)}", level="debug")

        # Sample hard and soft positives if specified
        if self.hard_positives > 0:
            hard_positives_df = hard_positives_df.sample(frac=self.hard_positives, random_state=self.seed)
        if self.soft_positives > 0:
            soft_positives_df = soft_positives_df.sample(frac=self.soft_positives, random_state=self.seed)
        # If no hard or soft positives, create empty DataFrames
        if hard_positives_df.empty and soft_positives_df.empty:
            self.log("No hard or soft positives found. No reference generated.", level="warning")
            return
        
        self.log(f"Hard positives sampled: {len(hard_positives_df)}", level="debug")
        self.log(f"Soft positives sampled: {len(soft_positives_df)}", level="debug")
        
        # Concatenate hard and soft positives
        reference_df = pd.concat([hard_positives_df, soft_positives_df], ignore_index=True)

        # Save the reference file
        reference_df.to_csv(self.generated_reference, sep="\t", index=False)
        self.log(f"Reference generated at {self.generated_reference}", level="debug")
        self._reference = self.generated_reference
        self.log("#Generated Reference Path...", level="debug")
        return



    




        

