"""Fixed reviewer demo: a separate process without study credentials or database access."""

import os
from pathlib import Path
from exact_inspect.service import create_prepared_app


def app():
    return create_prepared_app(
        Path(os.environ["EXACT_DEMO_PACKAGE"]),
        profile="public_demo",
        frontend_dir=Path(os.environ["EXACT_DEMO_FRONTEND"]),
    )
