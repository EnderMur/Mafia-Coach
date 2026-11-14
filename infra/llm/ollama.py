"""Клиент нативного Ollama API с обязательным отключением thinking."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any, Final

import httpx
from loguru import logger
from pydantic import ValidationError

from domain.exceptions import LLMConnectionError, LLMError
from infra.llm.client import LLMClient, ReasoningSink
from infra.llm.schemas import AnalysisResponse, validate_response

OLLAMA_THINK_PARAM: Final[str] = "think"
OLLAMA_THINK_VALUE: Final[bool] = False


class _RetryableStatus(Exception):
    """Временный ответ сервера, который допускает повторный запрос."""


class OllamaClient(LLMClient):
    """Нативный /api/chat; не изменяет OpenAI-совместимый клиент LM Studio."""

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
        if not 1 <= max_retries <= 3:
            raise ValueError("max_retries должен быть от 1 до 3")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self._closed = False
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(request_timeout_sec, connect=connect_timeout_sec),
            headers={"Content-Type": "application/json"},
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise LLMError("Клиент Ollama закрыт")

    def _build_payload(self, messages: list[dict[str, str]], *, stream: bool) -> dict[str, Any]:
        """Единственное место формирования тела каждого запроса к /api/chat."""
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "format": AnalysisResponse.model_json_schema(),
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.max_tokens,
            },
            OLLAMA_THINK_PARAM: OLLAMA_THINK_VALUE,
        }

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            await self._client.aclose()

    async def healthcheck(self) -> None:
        self._ensure_open()
        try:
            response = await self._client.get(f"{self.base_url}/api/tags")
        except httpx.RequestError as error:
            raise LLMConnectionError(f"Ollama недоступен: {error}") from error
        if response.status_code != 200:
            raise LLMError(f"Ошибка проверки Ollama: HTTP {response.status_code}")
        try:
            models = response.json()["models"]
            if not isinstance(models, list):
                raise TypeError("models не является списком")
        except (ValueError, KeyError, TypeError) as error:
            raise LLMError("Некорректный список моделей Ollama") from error
        if not any(
            isinstance(item, dict) and self.model in (item.get("name"), item.get("model"))
            for item in models
        ):
            raise LLMError(f"Модель Ollama {self.model!r} не найдена")
        logger.bind(channel="llm").info("Проверка Ollama выполнена")

    @staticmethod
    def _content(body: Any) -> tuple[str, bool]:
        if not isinstance(body, dict):
            raise LLMError("Некорректный ответ Ollama")
        if body.get("error"):
            raise LLMError(f"Ошибка Ollama: {body['error']}")
        message = body.get("message")
        if not isinstance(message, dict):
            raise LLMError("В ответе Ollama отсутствует message")
        thinking = message.get("thinking")
        if thinking is not None and not isinstance(thinking, str):
            raise LLMError("Некорректное поле thinking в ответе Ollama")
        if thinking:
            raise LLMError(
                "Ollama вернул thinking при think=false; проверьте совместимость версии и модели"
            )
        content = message.get("content")
        if not isinstance(content, str):
            raise LLMError("Некорректное поле content в ответе Ollama")
        done = body.get("done")
        if not isinstance(done, bool):
            raise LLMError("В ответе Ollama отсутствует признак завершения")
        if done and body.get("done_reason") == "length":
            raise LLMError("Ответ Ollama усечён по лимиту токенов")
        return content, done

    async def _request(self, messages: list[dict[str, str]], *, stream: bool) -> str:
        payload = self._build_payload(messages, stream=stream)
        async with self._client.stream(
            "POST", f"{self.base_url}/api/chat", json=payload
        ) as response:
            if response.status_code == 429 or 500 <= response.status_code < 600:
                raise _RetryableStatus(f"HTTP {response.status_code}")
            if response.status_code >= 300:
                body = (await response.aread()).decode("utf-8", errors="replace")[:200]
                raise LLMError(f"Ошибка Ollama HTTP {response.status_code}: {body}")
            try:
                if not stream:
                    body = json.loads(await response.aread())
                    content, done = self._content(body)
                    if not done:
                        raise LLMError("Незавершённый ответ Ollama")
                    return content
                parts: list[str] = []
                done = False
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    if done:
                        raise LLMError("Получены данные после завершения потока Ollama")
                    fragment, done = self._content(json.loads(line))
                    parts.append(fragment)
                if not done:
                    raise LLMError("Поток Ollama оборван до завершения")
                return "".join(parts)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise LLMError("Некорректный JSON в ответе Ollama") from error

    async def _analyze(
        self,
        *,
        messages: list[dict[str, str]],
        known_player_ids: frozenset[int],
        role_names_ru: frozenset[str],
        role_alias_resolver: Callable[[str], str | None] | None,
        silent_player_ids: frozenset[int],
        stream: bool,
    ) -> AnalysisResponse:
        self._ensure_open()
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                raw = await self._request(messages, stream=stream)
                result, warnings = validate_response(
                    raw,
                    known_player_ids=known_player_ids,
                    role_names_ru=role_names_ru,
                    role_alias_resolver=role_alias_resolver,
                    silent_player_ids=silent_player_ids,
                )
                for warning in warnings:
                    logger.bind(channel="llm").warning("{}", warning)
                return result
            except (httpx.ConnectError, httpx.ConnectTimeout) as error:
                raise LLMConnectionError(f"Ollama недоступен: {error}") from error
            except (httpx.RequestError, _RetryableStatus, ValidationError) as error:
                last_error = error
                if attempt + 1 < self.max_retries:
                    await asyncio.sleep(self.retry_backoff_sec * (2**attempt))
        raise LLMError(f"Ошибка Ollama после {self.max_retries} попыток: {last_error}")

    async def analyze(
        self,
        *,
        messages: list[dict[str, str]],
        known_player_ids: frozenset[int],
        role_names_ru: frozenset[str],
        role_alias_resolver: Callable[[str], str | None] | None = None,
        silent_player_ids: frozenset[int] = frozenset(),
    ) -> AnalysisResponse:
        return await self._analyze(
            messages=messages,
            known_player_ids=known_player_ids,
            role_names_ru=role_names_ru,
            role_alias_resolver=role_alias_resolver,
            silent_player_ids=silent_player_ids,
            stream=False,
        )

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
        # thinking отключён; отдельный поток reasoning не генерируется.
        return await self._analyze(
            messages=messages,
            known_player_ids=known_player_ids,
            role_names_ru=role_names_ru,
            role_alias_resolver=role_alias_resolver,
            silent_player_ids=silent_player_ids,
            stream=True,
        )
