"""Asynchronous client for the local OpenAI-compatible LM Studio API."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypeVar

import httpx
from loguru import logger
from pydantic import ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from domain.exceptions import LLMConnectionError, LLMError
from infra.llm.schemas import AnalysisResponse, validate_response

_ResponseT = TypeVar("_ResponseT")


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
    ) -> AnalysisResponse:
        """Return validated analysis while forwarding reasoning chunks."""

    @abstractmethod
    async def aclose(self) -> None:
        """Close any transport resources owned by the implementation."""


class _TransientHTTP(Exception):
    """Internal retryable HTTP status or timeout failure."""

    def __init__(self, status: int | None = None, body: str = "") -> None:
        message = f"HTTP {status}: {body}" if status is not None else body
        super().__init__(message)
        self.status = status
        self.body = body


class _SchemaFormatRejected(LLMError):
    """Internal signal that the server rejects JSON Schema response format."""


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
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec

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
        """Close the underlying HTTP client only when this instance owns it."""
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()

    async def healthcheck(self) -> None:
        """Check that LM Studio responds and has the configured model loaded."""
        try:
            response = await self._client.get(self._url("/models"))
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as error:
            raise LLMConnectionError(f"LM Studio недоступен: {error}") from error

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
    ) -> AnalysisResponse:
        """Request, validate, and return a non-streaming analysis response."""
        current_messages = messages
        for validation_attempt in range(2):
            payload = self._build_payload(
                current_messages,
                response_format="json_schema",
            )
            try:
                response_body = await self._post_with_schema_fallback(payload, current_messages)
                raw_content = self._extract_content(response_body)
                response, warnings = validate_response(
                    raw_content,
                    known_player_ids=known_player_ids,
                    role_names_ru=role_names_ru,
                    role_alias_resolver=role_alias_resolver,
                )
            except ValidationError as error:
                if validation_attempt == 0:
                    current_messages = [
                        *messages,
                        {
                            "role": "user",
                            "content": "Предыдущий ответ не прошёл проверку. "
                            "Верни только JSON по заданной схеме.",
                        },
                    ]
                    logger.bind(channel="llm").warning(
                        "LLM response validation failed; requesting JSON retry"
                    )
                    continue
                raise LLMError(f"invalid LLM response after retry: {error}") from error

            self._log_warnings(warnings)
            return response

        raise LLMError("invalid LLM response after retry")

    async def analyze_stream(
        self,
        *,
        messages: list[dict[str, str]],
        known_player_ids: frozenset[int],
        role_names_ru: frozenset[str],
        role_alias_resolver: Callable[[str], str | None] | None = None,
        on_reasoning_chunk: ReasoningSink | None = None,
    ) -> AnalysisResponse:
        """Stream reasoning deltas and return the validated final JSON response."""
        payload = self._build_payload(messages, response_format="json_object")
        payload["stream"] = True

        async def stream_request() -> AnalysisResponse:
            return await self._stream_once(
                payload=payload,
                known_player_ids=known_player_ids,
                role_names_ru=role_names_ru,
                role_alias_resolver=role_alias_resolver,
                on_reasoning_chunk=on_reasoning_chunk,
            )

        return await self._with_retries(stream_request)

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
        elif response_format is not None:
            raise ValueError(f"unsupported response format: {response_format}")
        return payload

    async def _post_chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Post one non-streaming chat-completions request."""
        try:
            response = await self._client.post(self._url("/chat/completions"), json=payload)
        except (httpx.ConnectError, httpx.ConnectTimeout) as error:
            raise LLMConnectionError(f"LM Studio недоступен: {error}") from error
        except (httpx.ReadTimeout, httpx.WriteTimeout) as error:
            raise _TransientHTTP(body=f"request timeout: {error}") from error

        if response.status_code != 200:
            self._raise_for_status(response.status_code, response.text)

        try:
            body: Any = response.json()
        except (json.JSONDecodeError, ValueError) as error:
            raise LLMError("LLM returned invalid response JSON") from error
        if not isinstance(body, dict):
            raise LLMError("LLM returned an invalid response object")
        return body

    async def _post_with_schema_fallback(
        self,
        payload: dict[str, Any],
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Post with JSON Schema, falling back once for unsupported schemas."""
        try:
            return await self._with_retries(lambda: self._post_chat_completions(payload))
        except _SchemaFormatRejected:
            fallback_payload = self._build_payload(
                messages,
                response_format="json_object",
            )
            return await self._with_retries(lambda: self._post_chat_completions(fallback_payload))

    async def _with_retries(self, operation: Callable[[], Awaitable[_ResponseT]]) -> _ResponseT:
        """Retry transient transport failures, then convert exhaustion to LLMError."""
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(
                    multiplier=self.retry_backoff_sec,
                    min=self.retry_backoff_sec,
                    max=self.retry_backoff_sec * 4,
                ),
                retry=retry_if_exception_type((httpx.HTTPError, _TransientHTTP)),
                reraise=True,
            ):
                with attempt:
                    return await operation()
        except (httpx.HTTPError, _TransientHTTP) as error:
            raise LLMError(f"max retries reached: {error}") from error
        raise LLMError("retry operation completed without a result")

    async def _stream_once(
        self,
        *,
        payload: dict[str, Any],
        known_player_ids: frozenset[int],
        role_names_ru: frozenset[str],
        role_alias_resolver: Callable[[str], str | None] | None,
        on_reasoning_chunk: ReasoningSink | None,
    ) -> AnalysisResponse:
        """Perform one streaming request and validate its accumulated content."""
        content: list[str] = []
        done = False
        try:
            async with self._client.stream(
                "POST", self._url("/chat/completions"), json=payload
            ) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    self._raise_for_status(response.status_code, body)

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
                    delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
                    text = delta.get("content") if isinstance(delta, dict) else None
                    if isinstance(text, str) and text:
                        content.append(text)
                        logger.bind(channel="reasoning").debug(text)
                        if on_reasoning_chunk is not None:
                            on_reasoning_chunk(text)
        except (httpx.ConnectError, httpx.ConnectTimeout) as error:
            raise LLMConnectionError(f"LM Studio недоступен: {error}") from error
        except (httpx.ReadTimeout, httpx.WriteTimeout) as error:
            raise _TransientHTTP(body=f"stream timeout: {error}") from error

        if not done:
            raise LLMError("stream ended before [DONE]")
        try:
            result, warnings = validate_response(
                "".join(content),
                known_player_ids=known_player_ids,
                role_names_ru=role_names_ru,
                role_alias_resolver=role_alias_resolver,
            )
        except ValidationError as error:
            raise LLMError(f"invalid streaming response: {error}") from error
        self._log_warnings(warnings)
        return result

    def _url(self, path: str) -> str:
        """Join a relative API path to the configured base URL."""
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    def _raise_for_status(self, status_code: int, body: str) -> None:
        """Raise the appropriate domain or retryable error for an HTTP status."""
        snippet = _snip(body)
        if status_code == 404:
            raise LLMError(f"model not found: {self.model}")
        if status_code == 429 or status_code >= 500:
            raise _TransientHTTP(status=status_code, body=snippet)
        if status_code in (400, 422) and "schema" in body.casefold():
            raise _SchemaFormatRejected(f"HTTP {status_code}: {snippet}")
        if 400 <= status_code < 500:
            raise LLMError(f"HTTP {status_code}: {snippet}")
        raise LLMError(f"unexpected HTTP status {status_code}: {snippet}")

    @staticmethod
    def _extract_content(response: dict[str, Any]) -> str:
        """Extract the assistant message content from a chat completion."""
        try:
            choices = response["choices"]
            message = choices[0]["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise LLMError("LLM response is missing choices[0].message.content") from error
        if not isinstance(content, str):
            raise LLMError("LLM response content must be a string")
        return content

    @staticmethod
    def _log_warnings(warnings: list[str]) -> None:
        """Write contextual validation warnings to the LLM log channel."""
        llm_logger = logger.bind(channel="llm")
        for warning in warnings:
            llm_logger.warning(warning)


def _snip(text: str, n: int = 200) -> str:
    """Return at most n characters and an ellipsis when truncated."""
    return text[:n] + ("…" if len(text) > n else "")
