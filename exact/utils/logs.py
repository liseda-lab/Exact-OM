
import os
import sys
import threading
import io
from contextlib import contextmanager, redirect_stdout, redirect_stderr

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
      2) C-level writes to fd 1 & 2 (so Java System.out/err via JPype/JYPe)
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