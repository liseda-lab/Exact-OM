#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import yaml
from jpype import JVMNotFoundException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exact.analysis import user_study
from exact.analysis.study_visualizer import StudyOntologyLookup
from exact.utils.data import read_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a minimal, commit-friendly study visualizer bundle from an Exact run."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to the existing Exact run directory.",
    )
    parser.add_argument(
        "--bundle-dir",
        type=str,
        required=True,
        help="Output directory for the deployable study bundle.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=str,
        required=False,
        help="Optional analysis directory. Defaults to <run-dir>/analysis/user_study.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=False,
        help="Optional config override. Defaults to <run-dir>/config.yaml.",
    )
    parser.add_argument(
        "--bundle-name",
        type=str,
        required=False,
        help="Optional display name used in bundle metadata. Defaults to the run directory name.",
    )
    parser.add_argument(
        "--jvm-heap-size",
        type=str,
        default=os.getenv("EXACT_STUDY_JVM_HEAP_SIZE", "8G"),
        help="JVM heap size used for the one-time ontology cache export.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing bundle directory.",
    )
    parser.add_argument(
        "--logging-level",
        type=str,
        default="INFO",
        help="Logger level: DEBUG, INFO, WARNING, ERROR.",
    )
    return parser.parse_args()


def _setup_logger(level_name: str) -> logging.Logger:
    level = getattr(logging, str(level_name).upper())
    logger = logging.getLogger("exact.prepare_study_bundle")
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        for handler in list(logger.handlers):
            try:
                handler.close()
            except Exception:
                pass
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.handlers = [handler]
    return logger


def _resolve_selected_records_path(analysis_dir: Path) -> Path:
    preferred = analysis_dir / "study_selected_records_with_rationales.json"
    fallback = analysis_dir / "study_selected_records.json"
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Study bundle export expected {preferred.name} or {fallback.name} under {analysis_dir}."
    )


def _resolve_config_path(run_dir: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        path = explicit.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        return path
    for candidate in [run_dir / "config.yaml", run_dir / "config.yml"]:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"No config.yaml/config.yml found under {run_dir}")


def _resolve_dataset_spec(run_dir: Path, logger: logging.Logger) -> Tuple[Path, Dict[str, Any]]:
    candidate_specs = sorted(run_dir.glob("*.yaml")) + sorted(run_dir.glob("*.yml"))
    for spec_path in candidate_specs:
        try:
            payload = read_yaml(spec_path)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skipping unreadable run spec %s: %s", spec_path, exc)
            continue
        dataset_cfg = dict(payload.get("dataset") or {})
        if dataset_cfg.get("data_dir") and dataset_cfg.get("source") and dataset_cfg.get("target"):
            return spec_path, dict(payload)
    raise FileNotFoundError(f"Could not resolve a dataset spec with dataset.data_dir/source/target under {run_dir}")


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _iter_record_entity_iris(record: Dict[str, Any], side: str) -> Iterable[str]:
    endpoint_iri = user_study._safe_text(record.get("src_iri" if side == "source" else "tgt_iri"))
    if endpoint_iri:
        yield endpoint_iri

    for item in user_study._hierarchy_items(record, side):
        for key in ["subject_iri", "object_iri"]:
            iri = user_study._safe_text(item.get(key))
            if iri:
                yield iri

    for channel in ["similarity", "difference"]:
        for item in user_study._channel_items(record, channel, side):
            for key in ["subject_iri", "object_iri"]:
                iri = user_study._safe_text(item.get(key))
                if iri:
                    yield iri

    for item in user_study._attribute_items(record, side):
        iri = user_study._safe_text(item.get("entity_iri"))
        if iri:
            yield iri


def _collect_entity_iris(selected_records: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    collected: Dict[str, Set[str]] = {"source": set(), "target": set()}
    for record in selected_records:
        for side in ["source", "target"]:
            for iri in _iter_record_entity_iris(record, side):
                collected[side].add(iri)
    return {
        "source": sorted(collected["source"]),
        "target": sorted(collected["target"]),
    }


def _build_precomputed_ontology_cache(
    *,
    run_dir: Path,
    config_path: Path,
    selected_records: List[Dict[str, Any]],
    logger: logging.Logger,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    side_to_iris = _collect_entity_iris(selected_records)
    lookup = StudyOntologyLookup(run_dir=run_dir, config_path=config_path, logger=logger, enabled=True)
    cache: Dict[str, Dict[str, Dict[str, Any]]] = {"source": {}, "target": {}}

    total = len(side_to_iris["source"]) + len(side_to_iris["target"])
    logger.info(
        "Precomputing ontology cache for study bundle: source_entities=%d target_entities=%d total=%d",
        len(side_to_iris["source"]),
        len(side_to_iris["target"]),
        total,
    )

    processed = 0
    for side in ["source", "target"]:
        for iri in side_to_iris[side]:
            cache[side][iri] = {
                **lookup.annotation_payload(iri, side),
                "parents": lookup.direct_parents(iri, side, limit=3),
                "children": lookup.direct_children(iri, side, limit=3),
                "neighbors": lookup.direct_neighbors(iri, side, limit=4),
            }
            processed += 1
            if processed % 50 == 0 or processed == total:
                logger.info("Ontology cache progress: %d/%d entities processed", processed, total)
    return cache


def _bundle_readme(
    *,
    bundle_name: str,
    source_run_dir: Path,
    selected_records_name: str,
) -> str:
    return (
        f"# Study Bundle: {bundle_name}\n\n"
        "This directory is a minimal deployable bundle for the Exact study visualizer.\n"
        "It contains only the files needed to serve the selected user-study cases and the\n"
        "precomputed one-hop ontology extension layer.\n\n"
        "## Included assets\n\n"
        f"- `config.yaml`: copied from `{source_run_dir}` and used to resolve the dataset class\n"
        f"- `study-bundle.yaml`: lightweight dataset spec for the visualizer bundle\n"
        f"- `analysis/user_study/study_mapping.json`: final user-study graph payload\n"
        f"- `analysis/user_study/{selected_records_name}`: selected pair records used by the UI\n"
        "- `analysis/user_study/ontology_cache.json`: precomputed labels, annotations, and one-hop expansion neighborhoods for the entities used in the study\n"
        "- `study_bundle.json`: metadata manifest describing the bundle contents\n\n"
        "The visualizer can be pointed at this directory directly through `EXACT_STUDY_RUN_DIR`.\n"
    )


def main() -> None:
    args = parse_args()
    logger = _setup_logger(args.logging_level)
    os.environ["EXACT_STUDY_JVM_HEAP_SIZE"] = str(args.jvm_heap_size)

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    analysis_dir = Path(args.analysis_dir).resolve() if args.analysis_dir else (run_dir / "analysis" / "user_study")
    if not analysis_dir.exists():
        raise FileNotFoundError(f"Analysis directory not found: {analysis_dir}")
    bundle_dir = Path(args.bundle_dir).resolve()
    if bundle_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Bundle directory already exists: {bundle_dir}. Use --overwrite to replace it.")
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    selected_records_path = _resolve_selected_records_path(analysis_dir)
    study_mapping_path = analysis_dir / "study_mapping.json"
    if not study_mapping_path.exists():
        raise FileNotFoundError(f"Study mapping not found: {study_mapping_path}")
    selected_records = json.loads(selected_records_path.read_text(encoding="utf-8"))

    config_path = _resolve_config_path(run_dir, Path(args.config_path) if args.config_path else None)
    dataset_spec_path, dataset_spec = _resolve_dataset_spec(run_dir, logger)
    bundle_name = args.bundle_name or run_dir.name
    bundle_analysis_dir = bundle_dir / "analysis" / "user_study"

    logger.info("Exporting study bundle '%s' from %s to %s", bundle_name, run_dir, bundle_dir)

    _copy_file(config_path, bundle_dir / "config.yaml")
    _copy_file(study_mapping_path, bundle_analysis_dir / "study_mapping.json")
    _copy_file(selected_records_path, bundle_analysis_dir / selected_records_path.name)

    try:
        ontology_cache = _build_precomputed_ontology_cache(
            run_dir=run_dir,
            config_path=config_path,
            selected_records=selected_records,
            logger=logger,
        )
    except JVMNotFoundException as exc:
        raise RuntimeError(
            "Study bundle export needs Java once to precompute ontology_cache.json. "
            "Run this exporter in a Java-enabled environment, then commit the resulting bundle for Render."
        ) from exc
    (bundle_analysis_dir / "ontology_cache.json").write_text(
        json.dumps(ontology_cache, indent=2),
        encoding="utf-8",
    )

    original_dataset = dict(dataset_spec.get("dataset") or {})
    bundle_dataset_spec = {
        "dataset": {
            "data_dir": original_dataset.get("data_dir"),
            "source": original_dataset.get("source"),
            "target": original_dataset.get("target"),
        },
        "bundle": {
            "name": bundle_name,
            "source_run_dir": str(run_dir),
            "source_dataset_spec": str(dataset_spec_path),
        },
    }
    for key in ["full_reference", "candidates", "training_reference"]:
        value = original_dataset.get(key)
        if value:
            bundle_dataset_spec["dataset"][key] = value
    (bundle_dir / "study-bundle.yaml").write_text(
        yaml.safe_dump(bundle_dataset_spec, sort_keys=False),
        encoding="utf-8",
    )

    manifest = {
        "bundle_name": bundle_name,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(run_dir),
        "source_analysis_dir": str(analysis_dir),
        "config_path": "config.yaml",
        "dataset_spec_path": "study-bundle.yaml",
        "analysis_dir": "analysis/user_study",
        "study_mapping_path": "analysis/user_study/study_mapping.json",
        "selected_records_path": f"analysis/user_study/{selected_records_path.name}",
        "ontology_cache_path": "analysis/user_study/ontology_cache.json",
    }
    (bundle_dir / "study_bundle.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (bundle_dir / "README.md").write_text(
        _bundle_readme(
            bundle_name=bundle_name,
            source_run_dir=run_dir,
            selected_records_name=selected_records_path.name,
        ),
        encoding="utf-8",
    )

    logger.info("Study bundle written to %s", bundle_dir)
    logger.info("Included files:")
    logger.info("- %s", bundle_dir / "config.yaml")
    logger.info("- %s", bundle_dir / "study-bundle.yaml")
    logger.info("- %s", bundle_analysis_dir / "study_mapping.json")
    logger.info("- %s", bundle_analysis_dir / selected_records_path.name)
    logger.info("- %s", bundle_analysis_dir / "ontology_cache.json")


if __name__ == "__main__":
    main()
