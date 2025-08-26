
from pathlib import Path
from typing import List

from multiprocessing.synchronize import Lock, Condition

from matcha_dl.core.entities.configs.config import ConfigModel

def flatten_config(config: ConfigModel) -> dict:
    """Flatten the ConfigModel into a dictionary."""
    config_dict = config.model_dump()
    flattened_config = {}
    for key, value in config_dict.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flattened_config[f"{key}.{sub_key}"] = sub_value
        else:
            flattened_config[key] = value
    return flattened_config

def process_config(
        tag: str, 
        config: ConfigModel, 
        available_devices: List[int],
        condition: Condition,
        source_file_path: Path, 
        target_file_path: Path, 
        reference_file_path: Path, 
        candidates_file_path: Path, 
        output_dir_path: Path, 
        full_reference_file_path: Path, 
        save_logs: bool) -> dict:

    from matcha_dl.core.actions.alignment import AlignmentAction

    temp_output_dir = output_dir_path / tag
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("global_supervised", None, reference_file_path),
        ("local_supervised", candidates_file_path, reference_file_path),
        ("global_unsupervised", None, None),
        ("local_unsupervised", candidates_file_path, None),
    ]

    task_results = {}

    with condition:
        if available_devices is not None:
            while len(available_devices) == 0:
                condition.wait()
            device = available_devices.pop(0)
        else:
            device = None

    try:
        for task_name, candidates_file_path, reference_file_path in tasks:
            results = AlignmentAction.run(
                source_file_path=source_file_path,
                target_file_path=target_file_path,
                output_dir_path=temp_output_dir,
                configs_file_path=config,
                reference_file_path=reference_file_path,
                full_reference_file_path=full_reference_file_path,
                candidates_file_path=candidates_file_path,
                log_file_path=temp_output_dir / f"{task_name}.log" if save_logs else None,
                run_eval=True,
                task_name=task_name,
                device=device,
            )
            task_results[task_name] = results
    finally:
        with condition:
            if device is not None:
                available_devices.append(device)
                condition.notify_all()

    return {"config_tag": tag, "config": flatten_config(config), "results": task_results}