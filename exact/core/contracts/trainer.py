import random
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Tuple, TYPE_CHECKING, Union

import logging

import numpy as np
import pandas as pd
import torch as th
from torch import device as tdevice
import json

import seaborn as sns
import matplotlib.pyplot as plt

from exact.core.contracts import SelfRegisteringComponent, LoggingClass
from exact.core.entities.registry import ComponentType
from exact.core.entities.mappings import EntityMapping
from exact.utils.mappings import fill_anchored_scores
from exact.utils.data import read_table
from exact.core.entities.configs.dataset import DatasetMask

if TYPE_CHECKING:
    from exact.core.contracts.dataset import IDataset
    from exact.core.contracts.model import IModel


class ITrainer(SelfRegisteringComponent, LoggingClass):

    component_type = ComponentType.TRAINER

    def __init__(
        self,
        dataset: 'IDataset',
        model: Optional[Type['IModel']] = None,
        model_params: Optional[Dict[str, Any]] = None,
        models: Optional[List[Tuple[Type['IModel'], Dict[str, Any]]]] = None,
        device: tdevice = tdevice("cuda" if th.cuda.is_available() else "cpu"),
        output_dir: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        
        LoggingClass.__init__(self, logger=logger)

        # Load Args

        self._dataset = dataset
        self._device = device
        model_specs: List[Tuple[Type['IModel'], Dict[str, Any]]] = []
        if models:
            model_specs = [(m, params or {}) for m, params in models]
        elif model is not None:
            model_specs = [(model, model_params or {})]
        else:
            raise ValueError("At least one model definition must be provided to the trainer.")

        def _bind_logger(obj: Any) -> None:
            if logger is None:
                return
            obj.logger = logger

            def _log(msg: str, level: str = "info", traceback: bool = False):
                log_method = getattr(logger, level, None) or logger.info
                log_method(msg, exc_info=traceback)

            obj.log = _log

        self._models: List['IModel'] = []
        for model_cls, params in model_specs:
            instance = model_cls(device=device, **params)
            _bind_logger(instance)
            if hasattr(instance, "attach_dataset"):
                try:
                    instance.attach_dataset(dataset)
                except Exception as exc:
                    self.log(f"Failed to attach dataset to model {model_cls.__name__}: {exc}", level="warning")
            self._models.append(instance.to(self.device))
        self._model = self._models[0]

        self._results_json: List[Dict[str, Any]] = []
        self._results_df: Optional[pd.DataFrame] = None
        
        self._output_dir = output_dir

        # Create output directories
        self.alignment_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    @property
    def device(self) -> th.device:
        return th.device(self._device if th.cuda.is_available() else "cpu")

    @property
    def dataset(self) -> 'IDataset':
        return self._dataset

    @property
    def model(self) -> 'IModel':
        return self._model

    @property
    def models(self) -> List['IModel']:
        return self._models

    @property
    def output_dir(self) -> Path:
        return self._output_dir
    
    @property
    def plot_dir(self) -> Path:
        return (self._output_dir / "plots").resolve()

    @property
    def alignment_dir(self) -> Path:
        return (self._output_dir / "alignment").resolve()

    @property
    def checkpoint_dir(self) -> Path:
        path = (self._output_dir / "checkpoints").resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def results_json(self) -> List[Dict[str, Any]]:
        return self._results_json

    @property
    def results_df(self) -> Optional[pd.DataFrame]:
        return self._results_df

    @results_df.setter
    def results_df(self, value: Optional[pd.DataFrame]) -> None:
        self._results_df = value

    @abstractmethod
    def predict(self, kind: DatasetMask = DatasetMask.inference, 
                threshold: Optional[float] = 0.7,
                **kwargs
    ) -> Tuple[List[EntityMapping], float]:
        
        pass

    def apply_prefilter(self,
                        alignment: List[EntityMapping],
                        threshold: Optional[float] = None,
                        cardinality: Optional[int] = None,
                        target_cardinality: Optional[int] = None,
                        **kwargs
    ) -> List[EntityMapping]:
        """
        Apply prefiltering to the dataset based on the features.
        """
        selector_target_conflict_enabled = getattr(self, "_selector_target_conflict_enabled", None)
        if selector_target_conflict_enabled is False:
            target_cardinality = None
        
        df = self.dataset.dataframe.copy()
        df = df[df[DatasetMask.prefiltered] == True]

        if df.empty:
            self.log("No data to prefilter", level="warning")
            if target_cardinality is not None:
                filtered = EntityMapping.filter_top_n_target_entity_mappings(alignment, target_cardinality)
                self.log(
                    (
                        "Target conflict resolver: "
                        f"target_cardinality={target_cardinality}, protected_exact_mappings=0, "
                        f"removed_mappings={len(alignment) - len(filtered)}"
                    ),
                    level="info",
                )
                return filtered
            return alignment

        score_column = None
        for candidate in ("Scores", "Score"):
            if candidate in df.columns:
                score_column = candidate
                break

        if score_column is None:
            self.log("Prefiltered dataframe missing score column; skipping prefilter step.", level="warning")
            return alignment

        prefilter_df = df[["Src", "Tgt", score_column]].copy()
        prefilter_df.columns = ["Src", "Tgt", "Score"]

        prefiltered_mappings = EntityMapping.read_table_mappings(prefilter_df, threshold=threshold)
        protected_pairs = {
            (mapping.head, mapping.tail)
            for mapping in prefiltered_mappings
        }
        final_alignment = prefiltered_mappings + alignment
        
        if cardinality is not None:
            final_alignment = EntityMapping.filter_top_n_entity_mappings(
                final_alignment,
                cardinality,
                protected_pairs=protected_pairs,
            )
        if target_cardinality is not None:
            before_target_conflicts = len(final_alignment)
            final_alignment = EntityMapping.filter_top_n_target_entity_mappings(
                final_alignment,
                target_cardinality,
                protected_pairs=protected_pairs,
            )
            self.log(
                (
                    "Target conflict resolver: "
                    f"target_cardinality={target_cardinality}, "
                    f"protected_exact_mappings={len(protected_pairs)}, "
                    f"removed_mappings={before_target_conflicts - len(final_alignment)}"
                ),
                level="info",
            )
        
        return final_alignment

    def save_results(
        self,
        preds: List[EntityMapping],
        sub_dir: Optional[str] = None,
        candidates_one2many_path: Optional[Path] = None,
        save_json: bool = True,
        save_csv: bool = True,
        save_stats_csv: bool = True,
        append_stats_to_summary_csv: bool = False,
        review_low: Optional[float] = None,
        review_high: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Path]:
        """
        Saves TSVs (alignment), JSON (explanations), CSV (summary), and run-level stats.
        Returns dict of output file paths.
        """
        output_paths = {}

        # ---- Alignments (existing behavior)
        align_path = self.save_alignment(
            preds=preds,
            candidates_one2many_path=candidates_one2many_path,
            sub_dir=sub_dir,
        )
        output_paths["alignment_tsv"] = Path(align_path)

        out_dir = (self.alignment_dir / (sub_dir or "default")).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # ---- Explanations JSON
        writer = getattr(self, "write_full_explanations_json", None)
        has_streamed_explanations = getattr(self, "has_streamed_explanations", None)
        can_stream_explanations = callable(writer) and (
            bool(self.results_json)
            or (callable(has_streamed_explanations) and bool(has_streamed_explanations()))
        )
        if save_json and (self.results_json or can_stream_explanations):
            json_path = out_dir / "full_explanations.json"
            if callable(writer):
                writer(json_path)
            else:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(self.results_json, f, ensure_ascii=False, separators=(",", ":"), default=str)
            self.log(f"Saved full explanations JSON → {json_path}", level="info")
            output_paths["explanations_json"] = json_path

        # ---- Summary CSV
        results_df = self.results_df
        if save_csv and results_df is not None and not results_df.empty:
            csv_path = out_dir / "summary_metrics.csv"
            results_df.to_csv(csv_path, sep="\t", index=False)
            self.log(f"Saved numeric summary CSV → {csv_path}", level="info")
            output_paths["summary_csv"] = csv_path

            # ---- Run-level stats
            stats = self._compute_run_stats(results_df, review_low=review_low, review_high=review_high)
            summary_stats = getattr(self, "_llm_summary_stats", None)
            if summary_stats is not None:
                stats["llm_summary_stats"] = summary_stats
            stats_json_path = out_dir / "run_stats.json"
            with open(stats_json_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            self.log(f"Saved run-level stats JSON → {stats_json_path}", level="info")
            output_paths["run_stats_json"] = stats_json_path

            if save_stats_csv:
                # One-row CSV for spreadsheet usage
                flat = {
                    "n_mappings": stats.get("n_mappings"),
                    "n_llm": stats.get("n_llm"),
                    "frac_llm": stats.get("frac_llm"),
                    "frac_review_band": stats.get("frac_review_band"),
                }
                # flatten metric aggregates: e.g., S_final_mean, S_final_std, ...
                for col, agg in (stats.get("metrics") or {}).items():
                    for k, v in agg.items():
                        flat[f"{col}_{k}"] = v
                # add triple_importance aggregates if available
                if stats.get("triple_importance"):
                    for k, v in stats["triple_importance"].items():
                        flat[f"triple_importance_{k}"] = v
                if summary_stats is not None:
                    flat["llm_summary_requested"] = summary_stats.get("requested")
                    flat["llm_summary_usable"] = summary_stats.get("usable")
                    flat["llm_summary_empty"] = summary_stats.get("empty")
                    flat["llm_summary_empty_fraction"] = summary_stats.get("empty_fraction")

                stats_csv_path = out_dir / "run_stats.csv"
                pd.DataFrame([flat]).to_csv(stats_csv_path, index=False)
                self.log(f"Saved run-level stats CSV → {stats_csv_path}", level="info")
                output_paths["run_stats_csv"] = stats_csv_path

            if append_stats_to_summary_csv:
                # Append a footer block to the summary CSV (human-readable)
                with open(csv_path, "a", encoding="utf-8") as f:
                    f.write("\n# ---- RUN STATS (human-readable footer) ----\n")
                    f.write(json.dumps(stats, indent=2))
                self.log("Appended run stats as footer to summary CSV.", level="info")

        calib_report = getattr(self, "_llm_calibration_report", None)
        if calib_report and (calib_report.get("messages") or calib_report.get("learned")):
            calib_path = out_dir / "llm_calibration.json"
            with open(calib_path, "w", encoding="utf-8") as f:
                json.dump(calib_report, f, indent=2, ensure_ascii=False)
            self.log(f"Saved LLM calibration metadata → {calib_path}", level="info")
            output_paths["llm_calibration_json"] = calib_path

        return output_paths

    def save_alignment(self, 
                       preds: List[EntityMapping], 
                       candidates_one2many_path: Optional[Path] = None,
                       sub_dir: Optional[str] = None
                       ) -> None:
        
        if sub_dir is not None:
            alignment_dir = self.alignment_dir / sub_dir
            alignment_dir.mkdir(parents=True, exist_ok=True)

        else:
            alignment_dir = self.alignment_dir

        if candidates_one2many_path is not None:
            candidates_one2many = read_table(candidates_one2many_path)
            candidates_one2many.columns = ["Src", "Tgt", "Candidates"]
            return self._save_local_alignment(preds, candidates_one2many, alignment_dir)

        else:
            return self._save_global_alignment(preds, alignment_dir)

    def _save_global_alignment(self, preds: List[EntityMapping], save_dir: Optional[Path] = None):

        # Extract the mappings as tuples

        global_alignment = EntityMapping.as_tuples(preds, with_score=True)

        # Save the global alignment

        global_dir = str(save_dir) + f"/{'src2tgt.maps'}_global.tsv"

        pd.DataFrame(global_alignment, columns=["SrcEntity", "TgtEntity", "Score"]).to_csv(
            global_dir, sep="\t", index=False
        )

        return global_dir

    def _save_local_alignment(self, preds: List[EntityMapping], candidates_one2many: pd.DataFrame, save_dir: Optional[Path] = None):

        # candidates is now a 1-1 format for this the original candidates are required

        ranking_results = fill_anchored_scores(candidates_one2many.values, preds)

        local_dir = str(save_dir) + f"/{'src2tgt.maps'}_local.tsv"

        pd.DataFrame(ranking_results, columns=["SrcEntity", "TgtEntity", "TgtCandidates"]).to_csv(
            local_dir, sep="\t", index=False
        )

        return local_dir
    
    def _make_summary_dataframe(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extracts numeric metrics and weights into a flat DataFrame for plotting.
        """
        rows = []
        for rec in records:
            base = {
                "src_iri": rec.get("src_iri"),
                "tgt_iri": rec.get("tgt_iri"),
            }
            pred = rec.get("prediction") or {}
            if "ground_truth" in pred:
                base["ground_truth"] = pred.get("ground_truth")
            for key in ("threshold_positive", "saved_alignment_member", "rationale_positive"):
                if key in pred:
                    base[key] = pred.get(key)
            for key, value in pred.items():
                if key.startswith("selector_"):
                    base[key] = value
            conf = rec.get("confidences", {})
            wts = rec.get("weights", {})
            imps = rec.get("importances", {})
            # If the LLM did not fire, force its importance to 0 to avoid misleading plots
            if imps and "I_llm" in imps:
                p_llm = conf.get("p_llm")
                llm_decision = pred.get("llm_decision")
                if (p_llm is None or float(p_llm) == 0.0) or (llm_decision in ("", None)):
                    imps = dict(imps)
                    imps["I_llm"] = 0.0
            base.update({**conf, **wts, **imps})
            rows.append(base)
        return pd.DataFrame(rows)

    
    def plot_distributions(
        self,
        which: List[str] = ["S_final", "I_label", "I_ctx", "I_llm", "w_c", "w_i"],
        kind: DatasetMask = DatasetMask.inference,
        bins: int = 30,
        kde: bool = True,
        figsize: Tuple[int, int] = (7, 5),
        alpha: float = 0.6,
        dpi: int = 300,
        **kwargs,
    ):
        """
        Plots histogram/KDE for any numeric metrics in the summary CSV or DataFrame.
        Example columns: ["S_final", "I_label", "I_ctx", "I_llm", "w_c", "w_i"].
        """
        if self.results_df is None or self.results_df.empty:
            self.log("No numeric results to plot.", level="warning")
            return

        metrics = which or []
        if not metrics:
            self.log("No metrics specified for distribution plots.", level="warning")
            return

        for col in metrics:
            if col not in self.results_df.columns:
                self.log(f"Column {col} missing in summary DF; skipping.", level="warning")
                continue

            plt.figure(figsize=figsize)
            sns.histplot(
                self.results_df[col].dropna(),
                kde=kde,
                bins=bins,
                color="royalblue",
                alpha=alpha,
                stat="probability",
            )
            plt.title(f"{kind.name}: {col} distribution")
            plt.xlabel(col)
            plt.ylabel("Probability")
            plt.grid(True)
            plt.tight_layout()

            out_path = self.plot_dir / f"{kind.name.lower()}_{col}_dist.png"
            plt.savefig(out_path, dpi=dpi)
            plt.close()
            self.log(f"Saved {col} distribution plot to {out_path}", level="debug")

    def plot_scores_vs_labels(
        self,
        which: Optional[List[str]],
        kind: DatasetMask = DatasetMask.inference,
        figsize: Tuple[int, int] = (7, 5),
        alpha: float = 0.6,
        dpi: int = 300,
        jitter: float = 0.2,
    ) -> None:
        if self.results_df is None or self.results_df.empty:
            self.log("No numeric results to plot.", level="warning")
            return
        if "ground_truth" not in self.results_df.columns:
            self.log("Summary DF lacks ground_truth column; skipping score-vs-label plots.", level="warning")
            return
        metrics = which or []
        if not metrics:
            self.log("No metrics specified for score-vs-label plots.", level="warning")
            return

        for col in metrics:
            if col not in self.results_df.columns:
                self.log(f"Column {col} missing in summary DF; skipping label plot.", level="warning")
                continue

            plot_df = self.results_df[["ground_truth", col]].dropna()
            if plot_df.empty:
                self.log(f"No data to plot for column {col} vs ground_truth.", level="warning")
                continue

            try:
                plot_df = plot_df.assign(ground_truth=plot_df["ground_truth"].astype(int))
            except (ValueError, TypeError):
                plot_df = plot_df.assign(ground_truth=plot_df["ground_truth"])

            plt.figure(figsize=figsize)
            sns.stripplot(
                data=plot_df,
                x="ground_truth",
                y=col,
                jitter=jitter,
                alpha=alpha,
                dodge=False,
            )
            plt.title(f"{kind.name}: {col} vs ground_truth")
            plt.xlabel("ground_truth")
            plt.ylabel(col)
            plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
            plt.tight_layout()
            out_path = self.plot_dir / f"{kind.name.lower()}_{col}_vs_label.png"
            plt.savefig(out_path, dpi=dpi)
            plt.close()
            self.log(f"Saved {col} vs ground_truth plot to {out_path}", level="debug")

    def _compute_run_stats(
        self,
        df: pd.DataFrame,
        review_low: Optional[float] = None,
        review_high: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Produces a dict with:
          - dataset-level counters (n_mappings, n_llm, frac_llm, frac_review_band)
          - numeric column aggregates (mean, std, min, max, p25, p50, p75)
          - triple-importance aggregates (if 'triple_importances' present in JSON)
        """
        stats: Dict[str, Any] = {}

        # --- basics
        n = len(df)
        stats["n_mappings"] = int(n)

        # LLM usage if available
        if "w_i" in df.columns:
            n_llm = int((df["w_i"] > 0).sum())
        elif "need_llm" in df.columns:
            n_llm = int(df["need_llm"].astype(bool).sum())
        else:
            n_llm = None
        stats["n_llm"] = n_llm
        stats["frac_llm"] = (n_llm / n) if (n_llm is not None and n > 0) else None

        # review band if thresholds are recorded in rows (optional)
        frac_review = None
        if review_low is not None and review_high is not None and "S_final" in df.columns and n > 0:
            in_band = ((df["S_final"] >= review_low) & (df["S_final"] <= review_high)).sum()
            frac_review = in_band / n
        stats["frac_review_band"] = frac_review

        # --- numeric aggregates
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        per_col = {}
        for col in numeric_cols:
            series = df[col].dropna().astype(float)
            if series.empty:
                continue
            per_col[col] = {
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "min": float(series.min()),
                "p25": float(series.quantile(0.25)),
                "p50": float(series.quantile(0.50)),
                "p75": float(series.quantile(0.75)),
                "max": float(series.max()),
            }
        stats["metrics"] = per_col

        # --- triple-level importance aggregates (if present in self.results_json)
        # Flatten all I_i across mappings, if present
        all_I = []
        for rec in getattr(self, "results_json", []) or []:
            ctx = rec.get("context", {})
            # support either list of importances or list of triples-with-importance
            if "triple_importances" in ctx and isinstance(ctx["triple_importances"], list):
                # format: [{"triple": "...", "importance": float, ...}, ...] OR [float, ...]
                vals = []
                for entry in ctx["triple_importances"]:
                    if isinstance(entry, dict) and "importance" in entry:
                        vals.append(entry["importance"])
                    elif isinstance(entry, (int, float)):
                        vals.append(float(entry))
                all_I.extend(vals)

        if all_I:
            series = pd.Series(all_I, dtype=float)
            stats["triple_importance"] = {
                "n_edges": int(series.size),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "min": float(series.min()),
                "p25": float(series.quantile(0.25)),
                "p50": float(series.quantile(0.50)),
                "p75": float(series.quantile(0.75)),
                "max": float(series.max()),
            }
        else:
            stats["triple_importance"] = None

        return stats
