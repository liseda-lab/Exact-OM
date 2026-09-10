"""One bounded E20 recipe; native Trainer checkpoints preserve optimizer/RNG state."""

from __future__ import annotations

import hashlib
import json
import os
import random
import signal
import time
from pathlib import Path

import pandas as pd
import torch

from exact.impl.models.selector.fitting import (
    fingerprint,
    freeze_json,
    safe_training_labels,
)

from .artifacts import resolve_local_retrieval_artifact


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mine_training_examples(
    frame, reference_pairs, *, application, seed=17, max_negatives=4, source_cap=400
):
    """Mine only verified negatives; never use other sources as implicit negatives."""
    frame, reference = safe_training_labels(frame, reference_pairs, application)
    if set(frame.Src.astype(str)) & set(application.get("source_ids", [])):
        raise ValueError("Retrieval training overlaps reporting source groups")
    if not {"src_text", "tgt_text"}.issubset(frame):
        raise ValueError("Retrieval training needs ontology-only src_text/tgt_text")
    sources = sorted(frame.Src.astype(str).unique())
    random.Random(seed).shuffle(sources)
    sources = sources[: int(source_cap)]
    examples, discarded = [], []
    for source in sources:
        group = frame[frame.Src.astype(str) == source].copy()
        group["_score"] = pd.to_numeric(
            group.get("cand_sim", pd.Series(0.0, index=group.index)), errors="raise"
        )
        positives = group[
            [pair in reference for pair in zip(group.Src.astype(str), group.Tgt.astype(str))]
        ]
        negatives = (
            group[
                [
                    pair not in reference
                    for pair in zip(group.Src.astype(str), group.Tgt.astype(str))
                ]
            ]
            .sort_values(["_score", "Tgt"], ascending=[False, True])
            .head(max_negatives)
        )
        if positives.empty or negatives.empty:
            discarded.append(
                {
                    "source": source,
                    "reason": "no_positive_in_pool" if positives.empty else "no_verified_negative",
                }
            )
            continue
        for positive in positives.itertuples():
            for negative in negatives.itertuples():
                examples.append(
                    {
                        "source": source,
                        "positive": str(positive.Tgt),
                        "negative": str(negative.Tgt),
                        "texts": [
                            str(positive.src_text),
                            str(positive.tgt_text),
                            str(negative.tgt_text),
                        ],
                    }
                )
    if len({row["source"] for row in examples}) < 3:
        raise ValueError("Retrieval fitting needs at least three safely labeled source groups")
    heldout = set(sources[::5])
    if not any(row["source"] in heldout for row in examples):
        heldout = {examples[0]["source"]}
    train = [row for row in examples if row["source"] not in heldout]
    validation = [row for row in examples if row["source"] in heldout]
    if not train:
        raise ValueError("Retrieval fitting has no source-disjoint training fold")
    return {
        "schema_version": 1,
        "train": train,
        "validation": validation,
        "discarded_sources": discarded,
        "source_order": sources,
        "negative_label_policy": application["negative_label_policy"],
        "in_batch_negatives": False,
        "exclude_known_positives": True,
    }


def _local_model(base_model, revision, kind, device):
    from huggingface_hub import snapshot_download
    from sentence_transformers import CrossEncoder, SentenceTransformer

    if not revision:
        raise ValueError("E20 requires an explicit frozen base-model revision")
    path = Path(base_model)
    if not path.is_dir():
        path = Path(snapshot_download(base_model, revision=revision, local_files_only=True))
    if kind == "contrastive_encoder":
        return SentenceTransformer(str(path), device=device, trust_remote_code=False)
    return CrossEncoder(
        str(path), num_labels=1, max_length=256, device=device, trust_remote_code=False
    )


def fit_retrieval_artifact(
    frame,
    reference_pairs,
    path,
    *,
    kind,
    base_model,
    revision,
    application,
    seed=17,
    epochs=3,
    max_steps=1000,
    batch_size=4,
    accumulation=8,
    checkpoint_steps=25,
    patience=3,
    device="cuda",
    model_bundle=None,
    stop_after_steps=None,
):
    """Fit a local encoder/head before any reporting-pool consumer runs.

    ``model_bundle`` is an injectable local model for bounded CPU recovery tests.
    Normal execution loads only the pre-resolved local model snapshot.
    """
    from transformers import (
        EarlyStoppingCallback,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    if kind not in {"contrastive_encoder", "cross_encoder"}:
        raise ValueError("Unsupported E20 fitted artifact kind")
    if (
        not 1 <= epochs <= 3
        or min(max_steps, batch_size, accumulation, checkpoint_steps, patience) < 1
    ):
        raise ValueError("E20 requires 1–3 epochs and positive bounded training controls")
    mining = mine_training_examples(frame, reference_pairs, application=application, seed=seed)
    recipe = {
        "kind": kind,
        "base_model": {"identifier": str(base_model), "revision": revision},
        "application": application,
        "seed": seed,
        "epochs": epochs,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "accumulation": accumulation,
        "checkpoint_steps": checkpoint_steps,
        "patience": patience,
        "mining_sha256": fingerprint(mining),
        "loss": (
            "hard_negative_softplus_cosine_temperature_0.05"
            if kind == "contrastive_encoder"
            else "binary_cross_entropy"
        ),
        "max_length": 256,
    }
    identity = fingerprint(recipe)
    path = Path(path)
    if (path / "retrieval_artifact.json").exists():
        artifact = resolve_local_retrieval_artifact(path, expected_kind=kind)
        if artifact.metadata.get("fit_identity") != identity:
            raise ValueError("Retrieval artifact training inputs or recipe changed")
        return artifact
    freeze_json(path / "recipe.json", recipe)
    freeze_json(path / "mining.json", mining)
    # Seed newly initialized heads before loading, not only the Trainer sampler.
    from transformers import set_seed

    set_seed(seed)
    bundle = model_bundle or _local_model(base_model, revision, kind, device)
    model = bundle if kind == "contrastive_encoder" else bundle.model
    if kind == "contrastive_encoder":
        bundle.max_seq_length = 256
    work = path / "checkpoints"
    work.mkdir(parents=True, exist_ok=True)
    stop_file = (
        Path(os.environ["EXACT_EXPERIMENT_STOP_FILE"])
        if os.getenv("EXACT_EXPERIMENT_STOP_FILE")
        else path / "STOP"
    )

    class Boundaries(TrainerCallback):
        last_save = time.monotonic()
        stopped = False

        def on_step_end(self, args, state, control, **kwargs):
            if stop_file.exists() or (
                stop_after_steps is not None and state.global_step >= stop_after_steps
            ):
                self.stopped = True
                control.should_save = control.should_evaluate = control.should_training_stop = True
            elif time.monotonic() - self.last_save >= 240:
                control.should_save = control.should_evaluate = True
            return control

        def on_save(self, args, state, control, **kwargs):
            self.last_save = time.monotonic()
            checkpoint = work / f"checkpoint-{state.global_step}"
            files = {
                item.name: _file_sha256(item)
                for item in checkpoint.iterdir()
                if item.is_file() and item.name != "committed.json"
            }
            for item in checkpoint.iterdir():
                if item.is_file():
                    with item.open("rb") as stream:
                        os.fsync(stream.fileno())
            freeze_json(
                checkpoint / "committed.json",
                {
                    "fit_identity": identity,
                    "step": state.global_step,
                    "epoch": state.epoch,
                    "files": files,
                },
            )

    boundaries = Boundaries()

    def collate(rows):
        if kind == "contrastive_encoder":
            return {
                "views": [
                    bundle.tokenize([row["texts"][column] for row in rows]) for column in range(3)
                ],
                "labels": torch.ones(len(rows)),
            }
        left = [row["texts"][0] for row in rows for _ in range(2)]
        right = [row["texts"][column] for row in rows for column in (1, 2)]
        batch = bundle.tokenizer(
            left, right, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        return {"features": dict(batch), "labels": torch.tensor([1.0, 0.0] * len(rows))}

    class PairTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if kind == "contrastive_encoder":
                vectors = [
                    torch.nn.functional.normalize(model(view)["sentence_embedding"].float(), dim=-1)
                    for view in inputs["views"]
                ]
                positive = (vectors[0] * vectors[1]).sum(-1)
                negative = (vectors[0] * vectors[2]).sum(-1)
                loss = torch.nn.functional.softplus((negative - positive) / 0.05).mean()
                logits = positive - negative
            else:
                logits = model(**inputs["features"]).logits.squeeze(-1).float()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, inputs["labels"].float()
                )
            return (loss, {"logits": logits}) if return_outputs else loss

    # max_steps is a cap, never a request to repeat beyond three epochs.
    steps_per_epoch = max(
        1, (len(mining["train"]) + batch_size * accumulation - 1) // (batch_size * accumulation)
    )
    step_limit = min(max_steps, steps_per_epoch * epochs)
    interval = min(checkpoint_steps, step_limit)
    arguments = TrainingArguments(
        output_dir=str(work),
        num_train_epochs=epochs,
        max_steps=step_limit,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=accumulation,
        max_grad_norm=1.0,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="linear",
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=interval,
        eval_steps=interval,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=interval,
        report_to=[],
        dataloader_num_workers=0,
        remove_unused_columns=False,
        label_names=["labels"],
        seed=seed,
        data_seed=seed,
        use_cpu=str(device) == "cpu",
        bf16=str(device).startswith("cuda"),
        fp16=False,
        save_safetensors=False,
        disable_tqdm=True,
    )
    trainer = PairTrainer(
        model=model,
        args=arguments,
        data_collator=collate,
        train_dataset=mining["train"],
        eval_dataset=mining["validation"],
        callbacks=[boundaries, EarlyStoppingCallback(early_stopping_patience=patience)],
    )
    resume = None
    for checkpoint in sorted(
        work.glob("checkpoint-*"), key=lambda item: int(item.name.split("-")[-1]), reverse=True
    ):
        try:
            record = json.loads((checkpoint / "committed.json").read_text())
            valid = record["fit_identity"] == identity and all(
                _file_sha256(checkpoint / name) == digest
                for name, digest in record["files"].items()
            )
        except (OSError, ValueError, KeyError):
            valid = False
        if valid:
            resume = str(checkpoint)
            break
        checkpoint.rename(work / ("invalid-" + checkpoint.name + "-" + str(time.time_ns())))
    attempt_started = time.perf_counter()
    attempt_id = str(time.time_ns())
    previous_signal = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: stop_file.touch())
    try:
        result = trainer.train(resume_from_checkpoint=resume)
    finally:
        signal.signal(signal.SIGTERM, previous_signal)
    freeze_json(
        path / "attempts" / f"{attempt_id}.json",
        {
            "fit_identity": identity,
            "resume_checkpoint": resume,
            "completed_step": trainer.state.global_step,
            "stopped": boundaries.stopped,
            "wall_seconds": time.perf_counter() - attempt_started,
            "metrics": result.metrics,
        },
    )
    if boundaries.stopped:
        raise InterruptedError(
            "Retrieval training stopped after a committed optimizer-step checkpoint"
        )
    model_path = path / "model"
    if kind == "contrastive_encoder":
        bundle.save(str(model_path))
    else:
        bundle.save_pretrained(str(model_path))
    manifest = {
        "schema_version": 1,
        "artifact_type": kind,
        "model_path": "model",
        "fit_identity": identity,
        "base_model": recipe["base_model"],
        "training_pairs": [
            {"source": source, "split_role": "train"}
            for source in sorted({row["source"] for row in mining["train"]})
        ],
        "reference_completeness": (
            "complete"
            if application["negative_label_policy"] == "complete_reference"
            else "known_incomplete"
        ),
        "negative_policy": (
            "complete_reference"
            if application["negative_label_policy"] == "complete_reference"
            else "confirmed_negative"
        ),
        "dataset_lock": application.get("dataset_lock_sha256") or fingerprint(application),
        "seed": seed,
        "epochs": epochs,
        "completed_steps": trainer.state.global_step,
        "effective_batch_size": batch_size * accumulation,
        "gradient_accumulation": accumulation,
        "precision": "bf16" if arguments.bf16 else "fp32",
        "training_metrics_optimistic": result.metrics,
        "validation_sources": sorted({row["source"] for row in mining["validation"]}),
        "mining": {
            "candidate_pool_fingerprint": fingerprint(frame[["Src", "Tgt"]].values.tolist()),
            "top_k": 20,
            "max_negatives_per_source": 4,
            "max_training_pairs": len(mining["train"]),
            "exclude_known_positives": True,
            "in_batch_negatives": False,
        },
    }
    manifest["training_attempts"] = [
        json.loads(item.read_text()) for item in sorted((path / "attempts").glob("*.json"))
    ]
    manifest["total_training_wall_seconds"] = sum(
        attempt["wall_seconds"] for attempt in manifest["training_attempts"]
    )
    freeze_json(path / "retrieval_artifact.json", manifest)
    return resolve_local_retrieval_artifact(path, expected_kind=kind)


def prepare_retrieval_training(
    dataset, configs, *, training_reference_path, reporting_candidates_path, output_dir, device
):
    """Action seam: fit missing artifacts before dataset caches or pool generation."""
    pending = []
    for name, kind in (
        ("encoder_finetune", "contrastive_encoder"),
        ("cross_encoder", "cross_encoder"),
    ):
        config = getattr(configs.candidates, name)
        if config.mode != "off" and config.training is not None:
            pending.append((name, kind, config))
    if not pending:
        return
    if configs.data.train_candidates is None or training_reference_path is None:
        raise ValueError(
            "E20 training requires separate data.train_candidates and training reference"
        )
    from exact.utils.data import read_table
    from exact.utils.mappings import candidate_table_views

    raw = read_table(Path(configs.data.train_candidates))
    frame, _ = candidate_table_views(raw)
    if "confirmed_label" in raw:
        confirmed = raw.iloc[:, :2].copy()
        confirmed.columns = ["Src", "Tgt"]
        confirmed["confirmed_label"] = raw.confirmed_label
        frame = frame.merge(confirmed, on=["Src", "Tgt"], validate="one_to_one")
    reference_frame = read_table(Path(training_reference_path))
    reference = {
        (str(source), str(target))
        for source, target in reference_frame.iloc[:, :2].itertuples(index=False, name=None)
    }
    if configs.data.source_universe is not None:
        reporting_sources = Path(configs.data.source_universe).read_text().splitlines()
    elif reporting_candidates_path is not None:
        reporting, _ = candidate_table_views(read_table(Path(reporting_candidates_path)))
        reporting_sources = sorted(reporting.Src.astype(str).unique())
    else:
        raise ValueError(
            "Supervised generated retrieval requires an explicit disjoint data.source_universe"
        )
    application = {
        "dataset_signature": dataset.dataset_signature,
        "source_ids": reporting_sources,
        "negative_label_policy": configs.supervision.negative_label_policy,
    }
    # This validates unknown/incomplete labels before tokenization or GPU loading.
    frame, reference = safe_training_labels(frame, reference, application)
    frame["src_text"] = [dataset.source_graph.get_primary_label(str(iri)) for iri in frame.Src]
    frame["tgt_text"] = [dataset.target_graph.get_primary_label(str(iri)) for iri in frame.Tgt]
    for name, kind, config in pending:
        path = config.artifact or Path(output_dir) / "fitting" / name
        artifact = fit_retrieval_artifact(
            frame,
            reference,
            path,
            kind=kind,
            application=application,
            seed=configs.seed,
            device=str(device),
            **config.training.model_dump(),
        )
        config.artifact = artifact.manifest_path.parent
        if name == "encoder_finetune":
            config.negative_policy = artifact.metadata["negative_policy"]
        dataset._candidate_generation_params[name] = config.model_dump(mode="python")
    dataset._retrieval_artifacts.clear()
