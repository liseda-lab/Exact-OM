
import pandas as pd
import time
from typing import List, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from exact.core.contracts.trainer import ITrainer
from exact.core.entities.mappings import EntityMapping
from exact.core.entities.configs.dataset import DatasetMask


def _semantic_collate_fn(batch):
    """
    Keep variable-length fields (labels/contexts) as plain Python lists while default-collating the rest.
    This avoids torch's default_collate trying to stack ragged sequences of strings.
    """
    if not batch:
        return {}

    ragged_keys = {"src_labels", "tgt_labels", "src_ctx_triples", "tgt_ctx_triples"}
    collated = {}
    for key in batch[0].keys():
        values = [sample[key] for sample in batch]
        if key in ragged_keys or values[0] is None:
            collated[key] = values
        else:
            collated[key] = default_collate(values)

    if "src_ctx_triples" in collated:
        collated["src_contexts"] = collated["src_ctx_triples"]
    if "tgt_ctx_triples" in collated:
        collated["tgt_contexts"] = collated["tgt_ctx_triples"]
    return collated


class SemanticAlignmentRunner(ITrainer):
    """
    External loop orchestrator for SemanticScorer inference.
    Collects all outputs (scores, weights, explanations) for saving & plotting.
    """

    @torch.no_grad()
    def predict(
        self,
        kind: DatasetMask = DatasetMask.inference,
        threshold: Optional[float] = 0.7,
        cardinality: Optional[int] = None,
        batch_size: int = 8,
        num_workers: int = 0,
        log_every: int = 1,
        mixed_precision: bool = False,
        **kwargs,
    ) -> Tuple[List[EntityMapping], float]:
        self.dataset.default_kind = kind
        self.model.eval()
        self.results_json.clear()
        self.results_df = None

        dl = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else None,
            collate_fn=_semantic_collate_fn,
        )

        all_mappings = []
        total_batches = len(dl)
        total_examples = len(self.dataset)
        start_time = time.perf_counter()

        self.log(f"Running Semantic Alignment on {total_examples} pairs", "info")

        for step, batch in enumerate(dl, start=1):
            src_iri = batch["src_iri"]
            tgt_iri = batch["tgt_iri"]
            src_labels = batch["src_labels"]
            tgt_labels = batch["tgt_labels"]
            src_ctxs = batch.get("src_contexts", None)
            tgt_ctxs = batch.get("tgt_contexts", None)

            with torch.amp.autocast("cuda", enabled=mixed_precision):
                out = self.model.forward(
                    src_iris=src_iri,
                    tgt_iris=tgt_iri,
                    src_label_lists=src_labels,
                    tgt_label_lists=tgt_labels,
                    src_contexts=src_ctxs,
                    tgt_contexts=tgt_ctxs,
                )

            # Accumulate mappings
            s_final = out["S_final"].detach().cpu().numpy().tolist()
            for s, t, score in zip(src_iri, tgt_iri, s_final):
                all_mappings.append((s, t, float(score)))

            # Accumulate full JSONs 
            if "explanations" in out:
                self.results_json.extend(out["explanations"])

            if step % log_every == 0 or step == total_batches:
                elapsed = time.perf_counter() - start_time
                avg_batch_time = elapsed / step
                remaining_batches = max(0, total_batches - step)
                remaining_time = remaining_batches * avg_batch_time
                self.log(
                    (
                        f"Batch {step}/{total_batches} done "
                        f"(avg {avg_batch_time:.2f}s/batch, ETA {remaining_time:.1f}s)"
                    ),
                    "debug",
                )

        total_time = time.perf_counter() - start_time
        avg_t = total_time / max(1, total_examples)
        avg_bt = total_time / max(1, total_batches)
        self.log(
            f"Completed {kind.name} in {total_time:.2f}s (~{avg_t:.3f}s/example, ~{avg_bt:.2f}s/batch)",
            "info",
        )

        df = pd.DataFrame(all_mappings, columns=["Src", "Tgt", "Scores"])
        preds = EntityMapping.read_table_mappings(df, threshold=threshold, cardinality=cardinality)
        self.results_df = self._make_summary_dataframe(self.results_json)
        return preds, avg_t
