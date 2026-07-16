import argparse
import json
import logging
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List

import torch


def _build_model(configs, device: torch.device):
    model_spec = configs.get_model_sequence()[0]
    model_cls = model_spec.name
    params = {
        **dict(model_spec.params or {}),
        "llm_profiles": {k: v.model_dump() for k, v in configs.llm_profiles.items()},
        "llm_routing": configs.llm_routing.model_dump(),
        **configs.alignment_params.model_dump(exclude_none=True),
        "use_lexical": False,
        "use_context": False,
        "use_llm": True,
        "return_explanations": False,
        "generate_llm_rationales": True,
    }
    model = model_cls(device=device, **params)
    return model


def _load_examples(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of examples.")
    return [dict(item or {}) for item in data]


def _normalize_decision(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"match", "yes", "positive"}:
        return "Match"
    if normalized in {"no match", "no", "negative"}:
        return "No match"
    raise ValueError(f"Unsupported rationale decision label: {value!r}")


def run_llm_debug(args) -> None:
    from exact.core.entities.configs.config import ConfigModel

    configs = (
        ConfigModel.load_config(Path(args.config_file).resolve())
        if args.config_file
        else ConfigModel()
    )
    logger = logging.getLogger("exact")
    logger.setLevel(getattr(logging, str(args.logging_level).upper()))
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.handlers = [handler]

    device = (
        torch.device(args.device)
        if args.device is not None and torch.cuda.is_available()
        else torch.device("cpu")
    )
    model = _build_model(configs, device)

    def _model_log(message, level="info", *extra, **kwargs):
        log_fn = getattr(logger, str(level).lower(), logger.info)
        log_fn(str(message))

    model.log = _model_log

    examples = _load_examples(Path(args.input_file).resolve())
    outputs: List[Dict[str, Any]] = []

    for idx, example in enumerate(examples, start=1):
        task = str(example.get("task", "")).strip().lower()
        result: Dict[str, Any] = {"index": idx, "task": task, "input": example}
        if task == "summary":
            src_label = str(
                example.get("src_label", example.get("source_label", example.get("label", "")))
            )
            tgt_label = str(example.get("tgt_label", example.get("target_label", "")))
            context = str(example.get("context", ""))
            pair_packet = str(example.get("pair_packet", example.get("brief_input", context)))
            if hasattr(model, "generate_pair_briefs_batched") and tgt_label:
                brief = model.generate_pair_briefs_batched([src_label], [tgt_label], [pair_packet])[
                    0
                ]
                result["output"] = {"pair_brief": brief}
            else:
                summary = model.generate_summaries_batched([src_label], [context])[0]
                result["output"] = {"summary": summary}
            result["backend_usage"] = {
                "summary": dict(getattr(model, "_last_summary_backend_meta", {}))
            }
        elif task == "rationale":
            src_label = str(example.get("src_label", example.get("source_label", "")))
            tgt_label = str(example.get("tgt_label", example.get("target_label", "")))
            decision = _normalize_decision(
                example.get("decision", example.get("rationale_decision_label", ""))
            )
            src_summary = str(example.get("src_summary", example.get("source_summary", "")))
            tgt_summary = str(example.get("tgt_summary", example.get("target_summary", "")))
            pair_brief = str(example.get("pair_brief", example.get("brief", "")))
            if hasattr(model, "generate_pair_briefs_batched") and tgt_label:
                if not pair_brief:
                    pair_packet = str(example.get("pair_packet", example.get("brief_input", "")))
                    if pair_packet:
                        pair_brief = model.generate_pair_briefs_batched(
                            [src_label], [tgt_label], [pair_packet]
                        )[0]
                    elif src_summary:
                        pair_brief = src_summary
                rationale = model.generate_rationales_batched(
                    [src_label],
                    [tgt_label],
                    [pair_brief],
                    [""],
                    [decision],
                )[0]
                result["output"] = {
                    "pair_brief": pair_brief,
                    "rationale": rationale,
                    "decision": decision,
                }
            else:
                if not src_summary:
                    src_context = str(example.get("src_context", example.get("source_context", "")))
                    src_summary = model.generate_summaries_batched([src_label], [src_context])[0]
                if not tgt_summary:
                    tgt_context = str(example.get("tgt_context", example.get("target_context", "")))
                    tgt_summary = model.generate_summaries_batched([tgt_label], [tgt_context])[0]
                rationale = model.generate_rationales_batched(
                    [src_label],
                    [tgt_label],
                    [src_summary],
                    [tgt_summary],
                    [decision],
                )[0]
                result["output"] = {
                    "src_summary": src_summary,
                    "tgt_summary": tgt_summary,
                    "rationale": rationale,
                    "decision": decision,
                }
            result["backend_usage"] = {
                "summary": dict(getattr(model, "_last_summary_backend_meta", {})),
                "rationale": dict(getattr(model, "_last_rationale_backend_meta", {})),
            }
        else:
            raise ValueError(f"Unsupported task {task!r}. Use 'summary' or 'rationale'.")
        outputs.append(result)

    output_path = Path(args.output_file).resolve() if args.output_file else None
    rendered = json.dumps(outputs, ensure_ascii=False, indent=2)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        print(f"Wrote {len(outputs)} debug outputs to {output_path}")
    else:
        print(rendered)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run a few summary/rationale LLM calls without the full alignment pipeline."
    )
    parser.add_argument(
        "--input_file", "-i", type=str, required=True, help="Path to a JSON list of debug examples."
    )
    parser.add_argument(
        "--output_file", "-o", type=str, required=False, help="Where to write the JSON outputs."
    )
    parser.add_argument(
        "--config_file", "-y", type=str, required=False, help="Path to the YAML config file."
    )
    parser.add_argument(
        "--device", "-d", type=int, required=False, help="GPU device ID to use for local fallback."
    )
    parser.add_argument("--jvm_heap_size", "-m", type=str, required=False, help=argparse.SUPPRESS)
    parser.add_argument(
        "--logging_level",
        "-l",
        type=str,
        default="INFO",
        help="Logger level: DEBUG, INFO, WARNING, ERROR.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.jvm_heap_size is not None:
        warnings.warn(
            "--jvm_heap_size/-m is deprecated and ignored; Exact-OM no longer needs Java.",
            DeprecationWarning,
            stacklevel=2,
        )
    try:
        run_llm_debug(args)
    except Exception as exc:  # noqa: BLE001
        print(str(exc), flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
