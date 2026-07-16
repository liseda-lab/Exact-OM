import io
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

from exact.utils.formatting import format_duration

if TYPE_CHECKING:
    from exact.utils.timing import TimingLedger


def categorize_log_message(message: str) -> str:
    text = (message or "").strip().lower()
    if text.startswith("[") and "]" in text[:32]:
        return text[1 : text.index("]")].strip() or "runtime"
    category_patterns = (
        ("progress", ("progress:", " eta ", "current=", "missing=")),
        ("setup", ("configuration", "seed", "logging", "device")),
        ("dataset", ("dataset", "ontology", "candidate generation", "sanity")),
        ("alignment", ("alignment", "semantic alignment", "inference", "prefilter")),
        ("selector", ("candidate-set selector", "target conflict resolver")),
        ("calibration", ("calibrated selector", "calibration")),
        ("llm", ("llm", "rationale", "summary/brief", "openrouter")),
        ("checkpoint", ("checkpoint", "cache")),
        ("output", ("saved", "writing", "written", "plot", "export")),
        ("evaluation", ("evaluation", "evaluating")),
    )
    for category, patterns in category_patterns:
        if any(pattern in text for pattern in patterns):
            return category
    return "runtime"


class ExactLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "category"):
            record.category = categorize_log_message(record.getMessage())
        return super().format(record)


def configure_exact_logger(
    logger: logging.Logger,
    level: int,
    log_file_path: Optional[Path] = None,
    stream: bool = True,
) -> logging.Logger:
    logger.setLevel(level)
    logger.propagate = False
    requested_log_path = Path(log_file_path).resolve() if log_file_path is not None else None
    for handler in list(logger.handlers):
        handler_path = getattr(handler, "_exact_log_file_path", None)
        if handler_path is not None and handler_path != requested_log_path:
            logger.removeHandler(handler)
            handler.close()
    formatter = ExactLogFormatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(category)s | %(message)s"
    )

    for handler in logger.handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)

    if stream and not any(
        getattr(handler, "_exact_stream_handler", False) for handler in logger.handlers
    ):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        stream_handler._exact_stream_handler = True
        logger.addHandler(stream_handler)

    if requested_log_path is not None:
        resolved_path = requested_log_path
        has_file_handler = any(
            getattr(handler, "_exact_log_file_path", None) == resolved_path
            for handler in logger.handlers
        )
        if not has_file_handler:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(resolved_path)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            file_handler._exact_log_file_path = resolved_path
            logger.addHandler(file_handler)
            logger.info("Logging to file %s", resolved_path)

    return logger


@dataclass(frozen=True)
class ProgressTask:
    key: str
    label: str
    estimate_seconds: float = 60.0


class RunProgressLogger:
    def __init__(
        self,
        logger: logging.Logger,
        tasks: Sequence[ProgressTask | tuple[str, str]],
        estimates_minutes: Optional[Mapping[str, float]] = None,
        min_log_interval_seconds: float = 30.0,
        bar_width: int = 24,
    ) -> None:
        self.logger = logger
        self.tasks = [
            task if isinstance(task, ProgressTask) else ProgressTask(key=task[0], label=task[1])
            for task in tasks
        ]
        self.estimates_seconds = {
            task.key: self._estimate_seconds(task, estimates_minutes or {}) for task in self.tasks
        }
        self.min_log_interval_seconds = max(0.0, float(min_log_interval_seconds))
        self.bar_width = max(8, int(bar_width))
        self.started_at = time.perf_counter()
        self.current_key: Optional[str] = None
        self.current_started_at: Optional[float] = None
        self.completed_seconds: dict[str, float] = {}
        self.fractions: dict[str, float] = {}
        self.last_log_at = 0.0

    @staticmethod
    def _estimate_seconds(task: ProgressTask, estimates_minutes: Mapping[str, float]) -> float:
        value = estimates_minutes.get(task.key)
        if value is None:
            value = estimates_minutes.get(task.label)
        try:
            seconds = float(value) * 60.0 if value is not None else float(task.estimate_seconds)
        except (TypeError, ValueError):
            seconds = float(task.estimate_seconds)
        return max(1.0, seconds)

    def start(self, key: str, detail: Optional[str] = None, force: bool = True) -> None:
        now = time.perf_counter()
        self.current_key = key
        self.current_started_at = now
        self.fractions.setdefault(key, 0.0)
        self._emit("started", detail=detail, force=force)

    def update(
        self,
        key: str,
        completed: Optional[float] = None,
        total: Optional[float] = None,
        fraction: Optional[float] = None,
        detail: Optional[str] = None,
        force: bool = False,
    ) -> None:
        if self.current_key != key:
            self.current_key = key
            self.current_started_at = time.perf_counter()
        if fraction is None and completed is not None and total:
            fraction = float(completed) / max(1.0, float(total))
        if fraction is not None:
            self.fractions[key] = min(1.0, max(0.0, float(fraction)))
        self._emit("progress", detail=detail, force=force)

    def finish(self, key: str, detail: Optional[str] = None, force: bool = True) -> None:
        now = time.perf_counter()
        started = self.current_started_at if self.current_key == key else None
        elapsed = (
            max(0.0, now - started) if started is not None else self.estimates_seconds.get(key, 1.0)
        )
        self.completed_seconds[key] = elapsed
        self.fractions[key] = 1.0
        previous_current = self.current_key
        previous_started = self.current_started_at
        if self.current_key != key:
            self.current_key = key
            self.current_started_at = previous_started
        self._emit("completed", detail=detail, force=force)
        self.current_key = previous_current
        self.current_started_at = previous_started
        if self.current_key == key:
            self.current_key = None
            self.current_started_at = None

    def complete(self, detail: Optional[str] = None) -> None:
        for task in self.tasks:
            self.fractions[task.key] = 1.0
            self.completed_seconds.setdefault(task.key, self.estimates_seconds.get(task.key, 1.0))
        self.current_key = None
        self.current_started_at = None
        self._emit("completed", detail=detail, force=True)

    def _emit(self, event: str, detail: Optional[str], force: bool) -> None:
        now = time.perf_counter()
        if not force and (now - self.last_log_at) < self.min_log_interval_seconds:
            return
        self.last_log_at = now
        percent = self._overall_fraction() * 100.0
        current = self._current_label()
        missing = self._missing_labels()
        elapsed = format_duration(now - self.started_at)
        eta = format_duration(self._eta_seconds(now))
        bar = self._progress_bar(percent)
        detail_suffix = f" | {detail}" if detail else ""
        self.logger.info(
            (
                f"[progress] {bar} {percent:5.1f}% | event={event} | "
                f"current={current} | missing={missing} | elapsed={elapsed} | eta={eta}"
                f"{detail_suffix}"
            )
        )

    def _overall_fraction(self) -> float:
        total = sum(self.estimates_seconds.get(task.key, 1.0) for task in self.tasks)
        if total <= 0:
            return 1.0
        done = 0.0
        for task in self.tasks:
            estimate = self.estimates_seconds.get(task.key, 1.0)
            fraction = self.fractions.get(task.key, 0.0)
            done += estimate * min(1.0, max(0.0, fraction))
        return min(1.0, max(0.0, done / total))

    def _eta_seconds(self, now: float) -> float:
        remaining = 0.0
        for task in self.tasks:
            fraction = min(1.0, max(0.0, self.fractions.get(task.key, 0.0)))
            if fraction >= 1.0:
                continue
            estimate = self.estimates_seconds.get(task.key, 1.0)
            if task.key == self.current_key and self.current_started_at is not None:
                current_elapsed = max(0.0, now - self.current_started_at)
                if fraction > 0.01:
                    remaining += current_elapsed * (1.0 - fraction) / fraction
                else:
                    remaining += max(0.0, estimate - current_elapsed)
            else:
                remaining += estimate * (1.0 - fraction)
        return remaining

    def _current_label(self) -> str:
        if self.current_key is None:
            return "none"
        for idx, task in enumerate(self.tasks, start=1):
            if task.key == self.current_key:
                return f"{idx}/{len(self.tasks)} {task.label}"
        return self.current_key

    def _missing_labels(self, max_labels: int = 4) -> str:
        missing = [
            task.label
            for task in self.tasks
            if self.fractions.get(task.key, 0.0) < 1.0 and task.key != self.current_key
        ]
        if not missing:
            return "none"
        shown = missing[:max_labels]
        if len(missing) > max_labels:
            shown.append(f"+{len(missing) - max_labels} more")
        return ", ".join(shown)

    def _progress_bar(self, percent: float) -> str:
        filled = int(round(self.bar_width * min(100.0, max(0.0, percent)) / 100.0))
        return "[" + ("#" * filled) + ("-" * (self.bar_width - filled)) + "]"


def summarize_progress_estimates(
    recorded_timings: Optional[Mapping[str, float]] = None,
    *,
    ledger: Optional["TimingLedger"] = None,
    config_fingerprint: Optional[str] = None,
) -> dict[str, float]:
    """Build minute estimates for :class:`RunProgressLogger`.

    Ledger estimates are stored in seconds and converted here.  The optional
    mapping remains for callers reading a pre-ledger ``times.txt`` file.
    """

    if ledger is not None:
        recorded_timings = {
            stage: seconds / 60.0
            for stage, seconds in ledger.estimates(config_fingerprint=config_fingerprint).items()
        }
    recorded_timings = recorded_timings or {}
    estimates = {
        "Setup": 0.2,
        "Dataset": 5.0,
        "Trainer": 0.5,
        "Inference": 10.0,
        "PostInference": 1.0,
        "Prefilter": 0.5,
        "Outputs": 1.0,
        "Plots": 1.0,
        "Evaluation": 1.0,
    }
    if recorded_timings.get("Dataset.CacheLoad") is not None:
        estimates["Dataset"] = recorded_timings["Dataset.CacheLoad"]
    elif recorded_timings.get("Dataset") is not None:
        estimates["Dataset"] = recorded_timings["Dataset"]
    if recorded_timings.get("Alignment.Inference") is not None:
        estimates["Inference"] = recorded_timings["Alignment.Inference"]
    elif recorded_timings.get("Alignment") is not None:
        estimates["Inference"] = recorded_timings["Alignment"]
    mapping = {
        "Alignment.PostInference": "PostInference",
        "Alignment.Prefilter": "Prefilter",
        "Postprocess.Outputs": "Outputs",
        "Postprocess.Plotting": "Plots",
        "Postprocess.Evaluation": "Evaluation",
    }
    for timing_key, task_key in mapping.items():
        if recorded_timings.get(timing_key) is not None:
            estimates[task_key] = recorded_timings[timing_key]
    return estimates


class TqdmLogger(io.TextIOBase):
    """
    A file‐like object for tqdm that redirects all writes into self.log.
    """

    def __init__(self, logger_fn, level="debug"):
        super().__init__()
        self.logger_fn = logger_fn
        self.level = level

    def write(self, buf):
        # tqdm sometimes writes empty lines or carriage returns—ignore those
        for line in buf.rstrip().splitlines():
            self.logger_fn(line, level=self.level)

    def flush(self):
        # tqdm may call flush(); we don't need to do anything special
        pass


@contextmanager
def capture_stdout(logger_fn, level="info"):
    """
    Captures:
      1) Python-level writes (print(), tqdm without file=)
      2) C-level writes to file descriptors 1 and 2
    and redirects them into logger_fn(message, level).
    """
    # --- 1) Set up a pipe for C-level FD redirection
    r_fd, w_fd = os.pipe()
    orig_stdout_fd = sys.stdout.fileno()
    orig_stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(orig_stdout_fd)
    saved_stderr_fd = os.dup(orig_stderr_fd)

    # Overwrite the real FDs so anything C-level goes to our w_fd
    os.dup2(w_fd, orig_stdout_fd)
    os.dup2(w_fd, orig_stderr_fd)
    os.close(w_fd)

    # --- 2) Replace Python sys.stdout/sys.stderr
    logger_io = TqdmLogger(logger_fn, level=level)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = logger_io, logger_io

    # --- 3) Spawn a thread to read from r_fd and feed lines to logger
    def _reader():
        with os.fdopen(r_fd, encoding="utf-8", errors="ignore") as pipe:
            for raw_line in pipe:
                line = raw_line.rstrip("\n")
                if line:
                    logger_fn(line, level=level)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    try:
        yield
    finally:
        # --- restore Python‐level streams
        sys.stdout, sys.stderr = old_stdout, old_stderr
        # --- restore original FDs
        os.dup2(saved_stdout_fd, orig_stdout_fd)
        os.dup2(saved_stderr_fd, orig_stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        # wait for the reader thread to drain any remaining output
        t.join()
