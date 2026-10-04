"""Interrupted study responses must not become completed, truncated HTTP messages."""

import asyncio

import h11
import pytest
from fastapi.testclient import TestClient
from starlette.responses import Response

from exact_inspect.study.api import create_study_app
from tests.explanation_study_test import ADMIN, ORIGIN, SECRET


def scope(path="/interrupted", method="GET"):
    return dict(
        type="http",
        asgi={"version": "3.0", "spec_version": "2.3"},
        http_version="1.1",
        method=method,
        scheme="https",
        path=path,
        raw_path=path.encode(),
        query_string=b"",
        headers=[(b"host", b"study.example.test")],
        server=("study.example.test", 443),
        client=("127.0.0.1", 1234),
    )


def application(tmp_path, frontend=None):
    return create_study_app(
        f"sqlite:///{tmp_path / 'study.sqlite'}",
        SECRET,
        ADMIN,
        ORIGIN,
        allow_test_sqlite=True,
        frontend_dir=frontend,
    )


@pytest.mark.parametrize("failure", [asyncio.CancelledError, RuntimeError])
@pytest.mark.parametrize("after_bytes", [0, 4])
def test_interrupted_producer_preserves_failure_and_never_fabricates_completion(
    tmp_path, failure, after_bytes
):
    app = application(tmp_path)
    original = failure("controlled interrupted producer")

    class InterruptedResponse:
        async def __call__(self, request_scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-length", b"8")],
                }
            )
            if after_bytes:
                await send({"type": "http.response.body", "body": b"part", "more_body": True})
            raise original

    app.router.add_route("/interrupted", InterruptedResponse())

    async def run():
        transport = h11.Connection(h11.SERVER)
        transport.receive_data(b"GET /interrupted HTTP/1.1\r\nHost: study.example.test\r\n\r\n")
        transport.next_event()
        transport.next_event()
        messages = []
        request_received = False

        async def receive():
            nonlocal request_received
            if not request_received:
                request_received = True
                return {"type": "http.request", "body": b"", "more_body": False}
            await asyncio.Event().wait()

        async def send(message):
            messages.append(message)
            if message["type"] == "http.response.start":
                transport.send(
                    h11.Response(status_code=message["status"], headers=message["headers"])
                )
            elif message["type"] == "http.response.body":
                transport.send(h11.Data(data=message.get("body", b"")))
                if not message.get("more_body", False):
                    transport.send(h11.EndOfMessage())

        with pytest.raises(failure) as raised:
            await asyncio.wait_for(app(scope(), receive, send), timeout=2)
        assert raised.value is original
        bodies = [m for m in messages if m["type"] == "http.response.body"]
        assert sum(len(m.get("body", b"")) for m in bodies) == after_bytes
        assert all(m.get("more_body") for m in bodies)
        assert dict(messages[0]["headers"])[b"content-length"] == b"8"

    asyncio.run(run())
    # Cancellation/failure belongs to this request, not future requests.
    assert TestClient(app).get("/api/health").json()["status"] == "ok"


def test_send_disconnect_propagates_and_closes_producer_without_extra_messages(tmp_path):
    app = application(tmp_path)
    closed = []

    class ResponseProducer:
        async def __call__(self, request_scope, receive, send):
            try:
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [(b"content-length", b"8")],
                    }
                )
                await send({"type": "http.response.body", "body": b"part", "more_body": True})
                await send({"type": "http.response.body", "body": b"rest", "more_body": False})
            finally:
                closed.append(True)

    app.router.add_route("/interrupted", ResponseProducer())

    async def run():
        messages = []
        original = OSError("controlled disconnected transport")

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            messages.append(message)
            if message["type"] == "http.response.body":
                raise original

        with pytest.raises(OSError) as raised:
            await app(scope(), receive, send)
        assert raised.value is original
        assert len(messages) == 2
        assert closed == [True]

    asyncio.run(run())


def test_origin_auth_body_limits_headers_and_head_remain_intact(tmp_path):
    frontend = tmp_path / "out"
    (frontend / "participate").mkdir(parents=True)
    (frontend / "index.html").write_text("index")
    html = b"<html><script>self.__boot=1</script><body>Study</body></html>"
    (frontend / "participate" / "index.html").write_bytes(html)
    app = application(tmp_path, frontend)

    @app.get("/cookies")
    def cookies():
        response = Response(b"unaltered", media_type="text/plain")
        response.set_cookie("first", "one", secure=True, httponly=True)
        response.set_cookie("second", "two", secure=True, httponly=True)
        return response

    # Put the diagnostic route before the intentional frontend catch-all.
    app.router.routes.insert(0, app.router.routes.pop())
    client = TestClient(app, base_url=ORIGIN)
    full = client.get("/participate/")
    assert full.content == html
    assert int(full.headers["content-length"]) == len(html)
    assert "sha256-" in full.headers["content-security-policy"]
    assert "unsafe-inline" not in full.headers["content-security-policy"]
    head = client.head("/participate/")
    assert head.content == b"" and head.headers["content-length"] == full.headers["content-length"]
    assert head.headers["content-security-policy"] == full.headers["content-security-policy"]
    response = client.get("/cookies")
    assert response.content == b"unaltered"
    assert len(response.headers.get_list("set-cookie")) == 2
    for result in [full, head, response, client.get("/api/health")]:
        assert result.headers["cache-control"] == "private, no-store"
        assert result.headers["referrer-policy"] == "no-referrer"
        assert result.headers["x-content-type-options"] == "nosniff"
    assert client.post("/api/v1/study/session", json={"secret": "bad" * 20}).status_code == 403
    assert (
        client.post(
            "/api/v1/study/session", json={"secret": "bad" * 20}, headers={"Origin": ORIGIN}
        ).status_code
        == 401
    )
    assert client.get("/api/v1/admin/studies").status_code == 401
    assert (
        client.post(
            "/api/v1/study/session",
            content=b" " * 1_000_001,
            headers={"Origin": ORIGIN, "Content-Type": "application/json"},
        ).status_code
        == 413
    )
