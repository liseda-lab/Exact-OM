from pathlib import Path

from exact.core.values import TIMING_STEP_ORDER


def load_recorded_timings(times_file_path: Path) -> dict[str, float]:
    if not times_file_path.exists():
        return {}

    timings: dict[str, float] = {}
    for line in times_file_path.read_text(encoding="utf-8").splitlines():
        step, separator, value_text = line.partition(":")
        if not separator:
            continue

        cleaned_value = value_text.strip().removesuffix(" minutes").strip()
        try:
            timings[step.strip()] = float(cleaned_value)
        except ValueError:
            continue

    return timings


def write_recorded_timings(times_file_path: Path, timings: dict[str, float]) -> None:
    times_file_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_steps = [step for step in TIMING_STEP_ORDER if step in timings]
    remaining_steps = sorted(step for step in timings if step not in TIMING_STEP_ORDER)
    lines = [f"{step}: {timings[step]:.1f} minutes" for step in [*ordered_steps, *remaining_steps]]
    times_file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_recorded_timings(
    times_file_path: Path,
    updates: dict[str, float],
) -> dict[str, float]:
    timings = load_recorded_timings(times_file_path)
    timings.update({step: value for step, value in updates.items() if value is not None})
    write_recorded_timings(times_file_path, timings)
    return timings
