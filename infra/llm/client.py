"""Asynchronous client for the local OpenAI-compatible LM Studio API."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Final, Protocol

import httpx
from loguru import logger
from pydantic import ValidationError

from domain.exceptions import LLMConnectionError, LLMError
from infra.llm.schemas import AnalysisResponse, validate_response


class ReasoningSink(Protocol):
    """Synchronous consumer of streamed reasoning text chunks."""

    def __call__(self, chunk: str) -> None:
        """Consume one reasoning chunk."""
        ...


class LLMClient(ABC):
    """Abstract interface for language-model services."""

    @abstractmethod
    async def healthcheck(self) -> None:
        """Raise a domain error if the server or configured model is unavailable."""

    @abstractmethod
    async def analyze(
        self,
        *,
        messages: list[dict[str, str]],
        known_player_ids: frozenset[int],
        role_names_ru: frozenset[str],
        role_alias_resolver: Callable[[str], str | None] | None = None,
        silent_player_ids: frozenset[int] = frozenset(),
    ) -> AnalysisResponse:
        """Return a validated non-streaming analysis response."""

    @abstractmethod
    async def analyze_stream(
        self,
        *,
        messages: list[dict[str, str]],
        known_player_ids: frozenset[int],
        role_names_ru: frozenset[str],
        role_alias_resolver: Callable[[str], str | None] | None = None,
        on_reasoning_chunk: ReasoningSink | None = None,
        log_reasoning_chunks: bool = True,
        silent_player_ids: frozenset[int] = frozenset(),
    ) -> AnalysisResponse:
        """Return validated analysis while forwarding reasoning chunks.

        Args:
            log_reasoning_chunks: Log each reasoning chunk at DEBUG when true.
        """

    @abstractmethod
    async def aclose(self) -> None:
        """Close any transport resources owned by the implementation."""


#: How many stream attempts may end in truncated JSON before the client gives
#: up. Each attempt costs the full token budget, so two is the latency cap.
_EOF_MAX_ATTEMPTS: Final[int] = 2
#: Token limit used by the single EOF retry (4096 keeps the retry bounded).
_EOF_RETRY_MAX_TOKENS: Final[int] = 4096


def empty_analysis_response() -> AnalysisResponse:
    """Return a valid response with no predictions (truncated-JSON fallback).

    An empty prediction list is a legitimate outcome: the analyzer persists
    nothing and the stats window simply keeps its dashes, which is far better
    than blocking the scanner for another full token budget.
    """
    return AnalysisResponse(
        analysis=[],
        game_read="Модель вернула обрезанный JSON; предсказаний в этом цикле нет.",
        reasoning="Ответ модели не был завершён (EOF), повторные попытки отключены.",
    )


class _TransientEOF(Exception):
    """Internal retryable truncated or invalid model JSON."""


class _TransientHTTP(Exception):
    """Internal retryable HTTP status failure."""

    def __init__(self, status: int, body: str = "") -> None:
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


class _SchemaRejectedHTTP(Exception):
    """Internal marker: the server refused the requested response format."""

    def __init__(self, status: int, body: str = "") -> None:
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


class LMStudioClient(LLMClient):
    """Async client for LM Studio's OpenAI-compatible REST API."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        request_timeout_sec: float = 120.0,
        connect_timeout_sec: float = 5.0,
        max_retries: int = 3,
        retry_backoff_sec: float = 2.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a client and retain model, sampling, and retry settings."""
        if max_retries > 3:
            raise ValueError("max_retries must not exceed 3")
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")

        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.request_timeout_sec = request_timeout_sec
        self.connect_timeout_sec = connect_timeout_sec
        self._max_retries = max_retries
        # Retain the public setting for callers that inspect client configuration.
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self._closed = False

        timeout = httpx.Timeout(
            request_timeout_sec,
            connect=connect_timeout_sec,
            read=request_timeout_sec,
            write=request_timeout_sec,
        )
        self._client = client or httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )
        self._owns_client = client is None

    async def aclose(self) -> None:
        """Close the owned transport once and prevent future requests."""
        if self._closed:
            return
        self._closed = True
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()

    async def healthcheck(self) -> None:
        """Check that LM Studio responds and has the configured model loaded."""
        self._ensure_open()
        try:
            response = await self._client.get(self._url("/models"))
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as error:
            raise LLMConnectionError(f"LM Studio недоступен: {error}") from error

        if response.status_code == 404:
            raise self._model_not_found_error()
        if response.status_code != 200:
            raise LLMError(f"healthcheck status={response.status_code}")

        try:
            body: Any = response.json()
        except (json.JSONDecodeError, ValueError) as error:
            raise LLMError("healthcheck returned invalid JSON") from error
        if not isinstance(body, dict) or not isinstance(body.get("data"), list):
            raise LLMError("healthcheck returned an invalid model list")

        model_ids = {
            item.get("id")
            for item in body["data"]
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        if self.model not in model_ids:
            raise LLMError(f"model '{self.model}' not loaded")

        logger.bind(channel="llm").info("LM Studio healthcheck passed")

    async def analyze(
        self,
        *,
        messages: list[dict[str, str]],
        known_player_ids: frozenset[int],
        role_names_ru: frozenset[str],
        role_alias_resolver: Callable[[str], str | None] | None = None,
        silent_player_ids: frozenset[int] = frozenset(),
    ) -> AnalysisResponse:
        """Analyze once, sharing a bounded HTTP-request budget across fallbacks."""
        self._ensure_open()
        budget = self._max_retries
        messages_current = list(messages)
        use_schema = True
        last_exc: Exception | None = None

        while budget > 0:
            budget -= 1
            payload = self._build_payload(
                messages_current,
                response_format="json_schema" if use_schema else "json_object",
            )
            try:
                response = await self._client.post(self._url("/chat/completions"), json=payload)
            except (httpx.ConnectError, httpx.ConnectTimeout) as error:
                raise LLMConnectionError(f"LM Studio недоступен: {error}") from error
            except (
                httpx.ReadTimeout,
                httpx.WriteTimeout,
                httpx.ReadError,
                httpx.RemoteProtocolError,
            ) as error:
                last_exc = error
                if budget == 0:
                    break
                await self._sleep_before_retry(budget)
                continue

            if response.status_code == 404:
                raise self._model_not_found_error()

            if (
                use_schema
                and response.status_code in (400, 422)
                and "schema" in response.text.lower()
            ):
                use_schema = False
                if budget == 0:
                    raise LLMError(f"HTTP {response.status_code}: {_snip(response.text)}")
                continue

            if response.status_code == 429 or 500 <= response.status_code < 600:
                last_exc = httpx.HTTPStatusError(
                    f"HTTP {response.status_code}",
                    request=response.request,
                    response=response,
                )
                if budget == 0:
                    break
                await self._sleep_before_retry(budget)
                continue

            if response.status_code >= 300:
                raise LLMError(f"HTTP {response.status_code}: {_snip(response.text)}")

            try:
                body: Any = response.json()
                raw_content = body["choices"][0]["message"]["content"]
                if not isinstance(raw_content, str):
                    raise TypeError("message content is not a string")
            except (ValueError, KeyError, IndexError, TypeError) as error:
                raise LLMError(f"malformed chat response: {type(error).__name__}") from error

            try:
                validated, warnings = validate_response(
                    raw_content,
                    known_player_ids=known_player_ids,
                    role_names_ru=role_names_ru,
                    role_alias_resolver=role_alias_resolver,
                    silent_player_ids=silent_player_ids,
                )
            except ValidationError:
                messages_current = list(messages) + [
                    {
                        "role": "user",
                        "content": "Верни ТОЛЬКО валидный JSON по схеме, "
                        "без markdown и комментариев.",
                    }
                ]
                use_schema = False
                if budget == 0:
                    break
                continue

            self._log_warnings(warnings)
            return validated

        reason = type(last_exc).__name__ if last_exc is not None else "validation"
        raise LLMError(f"max retries ({self._max_retries}) reached: {reason}")

    async def analyze_stream(
        self,
        *,
        messages: list[dict[str, str]],
        known_player_ids: frozenset[int],
        role_names_ru: frozenset[str],
        role_alias_resolver: Callable[[str], str | None] | None = None,
        on_reasoning_chunk: ReasoningSink | None = None,
        log_reasoning_chunks: bool = True,
        silent_player_ids: frozenset[int] = frozenset(),
    ) -> AnalysisResponse:
        """Stream a final JSON response under one shared HTTP-request budget.

        The stream starts with ``response_format="json_schema"``. When the server
        rejects that format (LM Studio accepts only ``json_schema`` or ``text``),
        the request is retried once with ``response_format="text"`` inside the
        same HTTP-request budget, and markdown fences are stripped before the
        completed payload is validated. Reasoning chunks are always forwarded to
        ``on_reasoning_chunk``; set ``log_reasoning_chunks=False`` to suppress
        per-chunk DEBUG log entries.
        """
        self._ensure_open()
        budget = self._max_retries
        token_limit = self.max_tokens
        # Truncated JSON (EOF) is the expensive failure: each attempt spends the
        # whole token budget (4096 × 3 ≈ 100 s). One retry is enough — a third
        # identical run almost never recovers, it only stalls the table.
        eof_attempts = 0
        use_schema = True
        last_exc: Exception | None = None

        while budget > 0:
            budget -= 1
            payload = self._build_payload(
                messages,
                response_format="json_schema" if use_schema else "text",
            )
            payload["stream"] = True
            payload["max_tokens"] = token_limit
            try:
                return await self._stream_once(
                    payload=payload,
                    known_player_ids=known_player_ids,
                    role_names_ru=role_names_ru,
                    role_alias_resolver=role_alias_resolver,
                    on_reasoning_chunk=on_reasoning_chunk,
                    log_reasoning_chunks=log_reasoning_chunks,
                    allow_markdown_fence=not use_schema,
                    silent_player_ids=silent_player_ids,
                )
            except (httpx.ConnectError, httpx.ConnectTimeout) as error:
                raise LLMConnectionError(f"LM Studio недоступен: {error}") from error
            except _TransientEOF as error:
                last_exc = error
                eof_attempts += 1
                # Cap the token limit once; never escalate twice.
                token_limit = min(_EOF_RETRY_MAX_TOKENS, token_limit * 2)
                if eof_attempts >= _EOF_MAX_ATTEMPTS:
                    logger.bind(channel="llm").warning(
                        "stream: {} truncated-JSON attempt(s) — giving up "
                        "instead of retrying (latency cap)",
                        eof_attempts,
                    )
                    return empty_analysis_response()
                if budget == 0:
                    break
                await self._sleep_before_retry(budget)
            except _SchemaRejectedHTTP as error:
                if budget == 0:
                    raise LLMError(f"HTTP {error.status}: {error.body}") from error
                if use_schema:
                    use_schema = False
                    logger.bind(channel="llm").warning(
                        "stream: falling back to response_format=text (server rejected json_schema)"
                    )
                    continue
                # The plain-text fallback is already active, so spend one more
                # unit of the shared budget instead of failing outright.
                last_exc = error
                continue
            except (
                _TransientHTTP,
                httpx.ReadTimeout,
                httpx.WriteTimeout,
                httpx.ReadError,
                httpx.RemoteProtocolError,
            ) as error:
                last_exc = error
                if budget == 0:
                    break
                await self._sleep_before_retry(budget)

        reason = type(last_exc).__name__ if last_exc is not None else "stream"
        raise LLMError(f"max retries ({self._max_retries}) reached: {reason}")

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: str | None,
    ) -> dict[str, Any]:
        """Build a chat-completions payload for the selected response format."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if response_format == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "AnalysisResponse",
                    "schema": AnalysisResponse.model_json_schema(),
                    "strict": True,
                },
            }
        elif response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        elif response_format == "text":
            payload["response_format"] = {"type": "text"}
        elif response_format is not None:
            raise ValueError(f"unsupported response format: {response_format}")
        return payload

    async def _stream_once(
        self,
        *,
        payload: dict[str, Any],
        known_player_ids: frozenset[int],
        role_names_ru: frozenset[str],
        role_alias_resolver: Callable[[str], str | None] | None,
        on_reasoning_chunk: ReasoningSink | None,
        log_reasoning_chunks: bool,
        allow_markdown_fence: bool = False,
        silent_player_ids: frozenset[int] = frozenset(),
    ) -> AnalysisResponse:
        """Perform one HTTP stream request and validate its completed JSON."""
        content: list[str] = []
        done = False
        async with self._client.stream(
            "POST", self._url("/chat/completions"), json=payload
        ) as response:
            if response.status_code == 404:
                raise self._model_not_found_error()
            if response.status_code == 429 or 500 <= response.status_code < 600:
                body = (await response.aread()).decode("utf-8", errors="replace")
                raise _TransientHTTP(response.status_code, _snip(body))
            if response.status_code >= 300:
                body = (await response.aread()).decode("utf-8", errors="replace")
                if _is_response_format_rejection(response.status_code, body):
                    raise _SchemaRejectedHTTP(response.status_code, _snip(body))
                raise LLMError(f"HTTP {response.status_code}: {_snip(body)}")

            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    done = True
                    break
                if not data:
                    continue
                try:
                    chunk: Any = json.loads(data)
                except json.JSONDecodeError as error:
                    raise LLMError(f"invalid streaming JSON chunk: {error}") from error

                choices = chunk.get("choices") if isinstance(chunk, dict) else None
                if not isinstance(choices, list) or not choices:
                    continue
                first_choice = choices[0]
                delta = first_choice.get("delta") if isinstance(first_choice, dict) else None
                if not isinstance(delta, dict):
                    continue

                reasoning_piece = delta.get("reasoning_content") or delta.get("reasoning")
                if isinstance(reasoning_piece, str) and reasoning_piece:
                    if on_reasoning_chunk is not None:
                        on_reasoning_chunk(reasoning_piece)
                    if log_reasoning_chunks:
                        logger.bind(channel="reasoning").debug(reasoning_piece)

                content_piece = delta.get("content")
                if isinstance(content_piece, str) and content_piece:
                    content.append(content_piece)

        if not done:
            raise LLMError("stream ended before [DONE]")
        raw_text = "".join(content)
        if allow_markdown_fence:
            raw_text = _strip_markdown_fence(raw_text)
        try:
            result, warnings = validate_response(
                raw_text,
                known_player_ids=known_player_ids,
                role_names_ru=role_names_ru,
                role_alias_resolver=role_alias_resolver,
                silent_player_ids=silent_player_ids,
            )
        except ValidationError as error:
            msg = str(error).lower()
            if "eof" in msg or "invalid json" in msg:
                raise _TransientEOF(f"JSON EOF: {error}") from error
            raise LLMError(f"invalid streaming response: {error}") from error
        self._log_warnings(warnings)
        return result

    async def _sleep_before_retry(self, budget: int) -> None:
        """Wait exponentially according to the already-consumed request budget."""
        delay = self.retry_backoff_sec * 2 ** (self._max_retries - budget - 1)
        await asyncio.sleep(delay)

    def _ensure_open(self) -> None:
        """Reject operations after this wrapper has been closed."""
        if self._closed:
            raise LLMError("client is closed")

    def _model_not_found_error(self) -> LLMError:
        """Create the documented non-retryable model configuration error."""
        return LLMError(f"model '{self.model}' not found. Сверьте llm.model с GET /v1/models.")

    def _url(self, path: str) -> str:
        """Join a relative API path to the configured base URL."""
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    @staticmethod
    def _log_warnings(warnings: list[str]) -> None:
        """Write contextual validation warnings to the LLM log channel."""
        llm_logger = logger.bind(channel="llm")
        for warning in warnings:
            llm_logger.warning(warning)


def _snip(text: str, n: int = 200) -> str:
    """Return at most n characters and an ellipsis when truncated."""
    return text[:n] + ("…" if len(text) > n else "")


_RESPONSE_FORMAT_REJECTION_MARKERS = ("response_format", "json_schema", "must be")


def _is_response_format_rejection(status: int, body: str) -> bool:
    """Return whether a status/body pair reports an unsupported response format."""
    if status not in (400, 422):
        return False
    lowered = body.lower()
    return any(marker in lowered for marker in _RESPONSE_FORMAT_REJECTION_MARKERS)


def _strip_markdown_fence(text: str) -> str:
    """Remove one surrounding ```json fence from a plain-text model answer.

    This is only applied to the ``response_format="text"`` fallback, where the
    server is free to wrap the JSON payload in a markdown code block.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped[3:]
    newline_index = body.find("\n")
    if newline_index == -1:
        return stripped
    info_string = body[:newline_index].strip().lower()
    if info_string not in ("", "json"):
        return stripped
    body = body[newline_index + 1 :]
    body = body.rstrip()
    body = body.removesuffix("```")
    return body.strip()
