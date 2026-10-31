"""Mocked HTTP tests for the LM Studio client."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
import respx
from loguru import logger

from domain.exceptions import LLMConnectionError, LLMError
from infra.llm.client import LMStudioClient, _snip
from infra.llm.schemas import AnalysisResponse
from infra.logging.setup import setup_logging

BASE_URL = "http://localhost:1234/v1"
MODEL = "test-model"
ROLE_NAME = "test-role"
VALID_CONTENT = json.dumps(
    {
        "analysis": [
            {
                "player_id": 100,
                "player_name": "Player A",
                "suspected_role": ROLE_NAME,
                "confidence": 0.8,
                "reasons": ["observed behavior"],
            }
        ],
        "game_read": "A concise read.",
        "reasoning": "A short explanation.",
    }
)
MESSAGES = [{"role": "user", "content": "Analyze this game."}]


@pytest.fixture
def respx_mock() -> Iterator[respx.MockRouter]:
    """Intercept every HTTP request and prohibit unmocked network traffic."""
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest_asyncio.fixture
async def client() -> AsyncIterator[LMStudioClient]:
    """Create a client with short retry waits for deterministic tests."""
    instance = LMStudioClient(
        base_url=BASE_URL,
        model=MODEL,
        retry_backoff_sec=0.001,
    )
    try:
        yield instance
    finally:
        await instance.aclose()


def valid_chat_response(content: str = VALID_CONTENT) -> httpx.Response:
    """Create a successful completion response with assistant content."""
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def stream_response(content: str = VALID_CONTENT) -> httpx.Response:
    """Create a finite SSE response containing JSON text and a DONE marker."""
    fragments = [content[:35], content[35:105], content[105:]]
    lines = [
        "data: " + json.dumps({"choices": [{"delta": {"content": part}}]}) for part in fragments
    ]
    lines.append("data: [DONE]")
    return httpx.Response(
        200,
        text="\n\n".join(lines) + "\n\n",
        headers={"content-type": "text/event-stream"},
    )


async def run_analysis(client: LMStudioClient) -> AnalysisResponse:
    """Call analyze with the shared valid test context."""
    return await client.analyze(
        messages=MESSAGES,
        known_player_ids=frozenset({100}),
        role_names_ru=frozenset({ROLE_NAME}),
    )


async def test_healthcheck_ok(client: LMStudioClient, respx_mock: respx.MockRouter) -> None:
    """A loaded configured model passes the healthcheck."""
    respx_mock.get(f"{BASE_URL}/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": MODEL}]}),
    )
    await client.healthcheck()


async def test_healthcheck_model_missing(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """A healthy server without the configured model raises LLMError."""
    respx_mock.get(f"{BASE_URL}/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "another-model"}]}),
    )
    with pytest.raises(LLMError, match="not loaded"):
        await client.healthcheck()


async def test_healthcheck_connection_error(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """Connection failures become LLMConnectionError."""
    respx_mock.get(f"{BASE_URL}/models").mock(
        side_effect=httpx.ConnectError("refused"),
    )
    with pytest.raises(LLMConnectionError, match="LM Studio недоступен"):
        await client.healthcheck()


async def test_healthcheck_status_500(client: LMStudioClient, respx_mock: respx.MockRouter) -> None:
    """Non-success healthcheck statuses become LLMError."""
    respx_mock.get(f"{BASE_URL}/models").mock(return_value=httpx.Response(500))
    with pytest.raises(LLMError, match="healthcheck status=500"):
        await client.healthcheck()


def test_snip_truncates_long_text() -> None:
    """HTTP error snippets are capped with a trailing ellipsis."""
    text = "x" * 201
    assert _snip(text) == "x" * 200 + "…"
    assert _snip("short") == "short"


def test_max_retries_too_high() -> None:
    """Retry counts are capped at three attempts."""
    with pytest.raises(ValueError, match="must not exceed 3"):
        LMStudioClient(base_url=BASE_URL, model=MODEL, max_retries=4)


async def test_analyze_ok(client: LMStudioClient, respx_mock: respx.MockRouter) -> None:
    """A valid chat completion returns an AnalysisResponse."""
    respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        return_value=valid_chat_response(),
    )
    response = await run_analysis(client)
    assert isinstance(response, AnalysisResponse)
    assert len(response.analysis) == 1


async def test_analyze_validation_retry(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """One schema validation failure causes one corrective retry."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=[valid_chat_response("not JSON"), valid_chat_response()],
    )
    response = await run_analysis(client)
    assert len(response.analysis) == 1
    assert route.call_count == 2
    retry_body: Any = json.loads(route.calls[1].request.content)
    assert "Верни только JSON" in retry_body["messages"][-1]["content"]


async def test_analyze_json_object_fallback(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """Schema rejection triggers exactly one JSON-object fallback request."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=[
            httpx.Response(400, text="unsupported response schema"),
            valid_chat_response(),
        ],
    )
    response = await run_analysis(client)
    assert len(response.analysis) == 1
    assert route.call_count == 2
    fallback_body: Any = json.loads(route.calls[1].request.content)
    assert fallback_body["response_format"]["type"] == "json_object"


async def test_analyze_connection_error_is_not_retried(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """Connection refusal is surfaced immediately for a later event retry."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=httpx.ConnectError("refused"),
    )
    with pytest.raises(LLMConnectionError):
        await run_analysis(client)
    assert route.call_count == 1


async def test_analyze_404_model_not_found(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """Model-not-found responses fail without retries."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(404, text="model absent"),
    )
    with pytest.raises(LLMError, match="model not found"):
        await run_analysis(client)
    assert route.call_count == 1


async def test_analyze_other_4xx_fails_without_retry(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """Non-schema client errors fail immediately without a retry."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(400, text="invalid request"),
    )
    with pytest.raises(LLMError, match="HTTP 400"):
        await run_analysis(client)
    assert route.call_count == 1


async def test_analyze_read_timeout_retries(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """Read timeouts are retried and exhaustion becomes LLMError."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=[httpx.ReadTimeout("timed out") for _ in range(3)],
    )
    with pytest.raises(LLMError, match="max retries reached"):
        await run_analysis(client)
    assert route.call_count == 3


async def test_analyze_429_retries(client: LMStudioClient, respx_mock: respx.MockRouter) -> None:
    """Rate limiting retries up to the configured attempt limit."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=[httpx.Response(429, text="busy") for _ in range(3)],
    )
    with pytest.raises(LLMError, match="max retries reached"):
        await run_analysis(client)
    assert route.call_count == 3


async def test_analyze_5xx_then_success(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """Transient server failures retry until a successful response arrives."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=[
            httpx.Response(500, text="temporary one"),
            httpx.Response(503, text="temporary two"),
            valid_chat_response(),
        ],
    )
    response = await run_analysis(client)
    assert len(response.analysis) == 1
    assert route.call_count == 3


async def test_analyze_stream_retries_server_error(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """Streaming server errors retry and return after recovery."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=[
            httpx.Response(500, text="temporary"),
            stream_response(),
        ],
    )
    response = await client.analyze_stream(
        messages=MESSAGES,
        known_player_ids=frozenset({100}),
        role_names_ru=frozenset({ROLE_NAME}),
    )
    assert len(response.analysis) == 1
    assert route.call_count == 2


async def test_analyze_stream_calls_callback(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """Each SSE content delta reaches the callback and forms the final JSON."""
    respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        return_value=stream_response(),
    )
    chunks: list[str] = []

    def collect_chunk(chunk: str) -> None:
        chunks.append(chunk)

    response = await client.analyze_stream(
        messages=MESSAGES,
        known_player_ids=frozenset({100}),
        role_names_ru=frozenset({ROLE_NAME}),
        on_reasoning_chunk=collect_chunk,
    )
    assert isinstance(response, AnalysisResponse)
    assert chunks
    assert "A short explanation." in "".join(chunks)


async def test_analyze_stream_reasoning_logged(
    client: LMStudioClient,
    respx_mock: respx.MockRouter,
    tmp_path: Path,
) -> None:
    """Streamed reasoning is written to the dedicated reasoning log."""
    respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        return_value=stream_response(),
    )
    logger.remove()
    try:
        setup_logging(log_dir=tmp_path)
        await client.analyze_stream(
            messages=MESSAGES,
            known_player_ids=frozenset({100}),
            role_names_ru=frozenset({ROLE_NAME}),
        )
        logger.complete()
        reasoning_log = (tmp_path / "reasoning.log").read_text(encoding="utf-8")
        assert "A short explanation." in reasoning_log
    finally:
        logger.complete()
        logger.remove()


async def test_build_payload_contains_schema(client: LMStudioClient) -> None:
    """The JSON Schema request format embeds the public response schema."""
    payload = client._build_payload(MESSAGES, response_format="json_schema")
    assert "response_format" in payload
    assert payload["response_format"]["type"] == "json_schema"
    schema = payload["response_format"]["json_schema"]
    assert schema["name"] == "AnalysisResponse"
    assert "properties" in schema["schema"]
    without_format = client._build_payload(MESSAGES, response_format=None)
    assert "response_format" not in without_format


async def test_aclose_closes_owned_client(client: LMStudioClient) -> None:
    """Closing a client-owned transport closes the HTTP client."""
    await client.aclose()
    assert client._client.is_closed is True


async def test_aclose_does_not_close_external_client() -> None:
    """Closing the wrapper leaves an externally supplied client open."""
    external = httpx.AsyncClient(base_url=BASE_URL)
    instance = LMStudioClient(base_url=BASE_URL, model=MODEL, client=external)
    await instance.aclose()
    assert external.is_closed is False
    await external.aclose()
