"""Exception handlers and request-size middleware. Clients never see tracebacks."""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from llmops_cache.embeddings.base import EmbeddingError
from llmops_cache.providers.base import ProviderError

logger = logging.getLogger(__name__)


def _error(status: int, code: str, message: str, request: Request) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=status,
        content={"error": code, "message": message, "request_id": request_id},
    )


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err.get('loc', ()))}: {err.get('msg')}"
            for err in exc.errors()
        )
        return _error(422, "validation_error", details or "invalid request", request)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        codes = {
            400: "bad_request",
            401: "unauthorized",
            404: "not_found",
            405: "method_not_allowed",
            413: "payload_too_large",
        }
        response = _error(
            exc.status_code, codes.get(exc.status_code, "http_error"), str(exc.detail), request
        )
        if exc.headers:
            response.headers.update(exc.headers)
        return response

    @app.exception_handler(ProviderError)
    async def _provider(request: Request, exc: ProviderError) -> JSONResponse:
        status = 503 if exc.retryable else 502
        return _error(
            status, "provider_unavailable", "The LLM provider failed. Try again.", request
        )

    @app.exception_handler(EmbeddingError)
    async def _embedding(request: Request, exc: EmbeddingError) -> JSONResponse:
        return _error(503, "embedding_unavailable", "The embedding provider failed.", request)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("request.unhandled_error", extra={"path": request.url.path})
        return _error(500, "internal_error", "Internal server error.", request)


class RequestContextMiddleware:
    """Assigns a request id and rejects bodies larger than ``max_bytes`` (pure ASGI)."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid.uuid4().hex[:16]
        scope.setdefault("state", {})["request_id"] = request_id

        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await _send_too_large(send, request_id)
            return

        # Buffer the (bounded) body up front so oversized chunked uploads are rejected
        # before any application code runs.
        chunks: list[bytes] = []
        size = 0
        more = True
        while more:
            message = await receive()
            if message["type"] != "http.request":
                break
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self.max_bytes:
                await _send_too_large(send, request_id)
                return
            chunks.append(chunk)
            more = message.get("more_body", False)
        body = b"".join(chunks)
        replayed = False

        async def replay_receive() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).append((b"x-request-id", request_id.encode()))
            await send(message)

        await self.app(scope, replay_receive, send_with_id)


async def _send_too_large(send: Send, request_id: str) -> None:
    body = (
        b'{"error":"payload_too_large","message":"Request body too large.","request_id":"'
        + request_id.encode()
        + b'"}'
    )
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"x-request-id", request_id.encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
