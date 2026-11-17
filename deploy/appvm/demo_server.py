"""Fixed reviewer demo: a separate process without study credentials or database access."""

import os
from pathlib import Path
from urllib.parse import urlencode

from starlette.responses import RedirectResponse

from exact_inspect.service import create_prepared_app


def app():
    instance = create_prepared_app(
        Path(os.environ["EXACT_DEMO_PACKAGE"]),
        profile="public_demo",
        frontend_dir=Path(os.environ["EXACT_DEMO_FRONTEND"]),
    )

    @instance.middleware("http")
    async def initial_example(request, call_next):
        # Start with a richly documented example, rather than the first entity hash.
        # Explicit bookmarked selections remain untouched.
        if request.method in {"GET", "HEAD"} and request.url.path == "/" and not request.url.query:
            return RedirectResponse(
                "/?"
                + urlencode(
                    {"source": "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C118847"}
                ),
                status_code=302,
            )
        return await call_next(request)

    return instance
