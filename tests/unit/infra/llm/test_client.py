"""Mocked HTTP tests for the LM Studio client."""

from __future__ import annotations

import io
import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import pytest_asyncio
import respx
from loguru import logger

if TYPE_CHECKING:
    from loguru import Record

from domain.exceptions import LLMConnectionError, LLMError
from infra.llm.client import LMStudioClient, _snip, empty_analysis_response
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
                "reason": "Observed behavior.",
                "reasons": ["observed behavior"],
            }
        ],
        "game_read": "A concise read.",
        "reasoning": "A short explanation.",
    }
)
MESSAGES = [{"role": "user", "content": "Analyze this game."}]


class BrokenSSEStream(httpx.AsyncByteStream):
    """Yield one partial event and then simulate a transport disconnect."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
        raise httpx.ReadError("stream disconnected")

    async def aclose(self) -> None:
        """Close the synthetic stream."""


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


def sse_response(*events: dict[str, Any] | str) -> httpx.Response:
    """Create an SSE response from event objects or raw data payloads."""
    lines = [event if isinstance(event, str) else json.dumps(event) for event in events]
    return httpx.Response(
        200,
        text="".join(f"data: {line}\n\n" for line in lines),
        headers={"content-type": "text/event-stream"},
    )


def stream_response(content: str = VALID_CONTENT) -> httpx.Response:
    """Create an SSE response with separate reasoning and final-content deltas."""
    fragments = [content[:35], content[35:105], content[105:]]
    events: list[dict[str, Any] | str] = [
        {"choices": [{"delta": {"reasoning_content": "A short "}}]},
        {"choices": [{"delta": {"reasoning": "explanation."}}]},
        *({"choices": [{"delta": {"content": part}}]} for part in fragments),
        "[DONE]",
    ]
    return sse_response(*events)


async def run_analysis(client: LMStudioClient) -> AnalysisResponse:
    """Call analyze with the shared valid test context."""
    return await client.analyze(
        messages=MESSAGES,
        known_player_ids=frozenset({100}),
        role_names_ru=frozenset({ROLE_NAME}),
    )


async def run_stream(client: LMStudioClient) -> AnalysisResponse:
    """Call analyze_stream with the shared valid test context."""
    return await client.analyze_stream(
        messages=MESSAGES,
        known_player_ids=frozenset({100}),
        role_names_ru=frozenset({ROLE_NAME}),
    )


def is_reasoning_log(record: Record) -> bool:
    """Select log records bound to the reasoning channel."""
    return bool(record["extra"].get("channel") == "reasoning")


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


async def test_healthcheck_invalid_json(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """Invalid healthcheck JSON is reported as an LLMError."""
    respx_mock.get(f"{BASE_URL}/models").mock(
        return_value=httpx.Response(200, content=b"not-json"),
    )
    with pytest.raises(LLMError, match="invalid JSON"):
        await client.healthcheck()


async def test_healthcheck_data_must_be_a_list(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """A non-list model collection is rejected."""
    respx_mock.get(f"{BASE_URL}/models").mock(
        return_value=httpx.Response(200, json={"data": "not a list"}),
    )
    with pytest.raises(LLMError, match="invalid model list"):
        await client.healthcheck()


async def test_healthcheck_model_item_without_id_is_not_loaded(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """Model entries without string identifiers do not match the configured model."""
    respx_mock.get(f"{BASE_URL}/models").mock(
        return_value=httpx.Response(200, json={"data": [{}]}),
    )
    with pytest.raises(LLMError, match="not loaded"):
        await client.healthcheck()


async def test_healthcheck_404_includes_models_hint(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """A missing health endpoint includes the documented model-list hint."""
    respx_mock.get(f"{BASE_URL}/models").mock(return_value=httpx.Response(404))
    with pytest.raises(LLMError, match="/v1/models"):
        await client.healthcheck()


def test_snip_truncates_long_text() -> None:
    """HTTP error snippets are capped with a trailing ellipsis."""
    text = "x" * 201
    assert _snip(text) == "x" * 200 + "…"
    assert _snip("short") == "short"


def test_max_retries_too_high() -> None:
    """Retry counts are capped at three HTTP requests."""
    with pytest.raises(ValueError, match="must not exceed 3"):
        LMStudioClient(base_url=BASE_URL, model=MODEL, max_retries=4)


def test_max_retries_too_low() -> None:
    """The shared HTTP-request budget must contain at least one request."""
    with pytest.raises(ValueError, match="must be at least 1"):
        LMStudioClient(base_url=BASE_URL, model=MODEL, max_retries=0)


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
    """One schema validation failure consumes one HTTP request from the budget."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=[valid_chat_response("not JSON"), valid_chat_response()],
    )
    response = await run_analysis(client)
    assert len(response.analysis) == 1
    assert route.call_count == 2
    retry_body: Any = json.loads(route.calls[1].request.content)
    assert "Верни ТОЛЬКО валидный JSON" in retry_body["messages"][-1]["content"]
    assert retry_body["response_format"]["type"] == "json_object"


async def test_analyze_schema_fallback_success(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """Schema rejection and its JSON-object fallback use two budget units."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=[
            httpx.Response(400, text="unsupported response schema"),
            valid_chat_response(),
        ],
    )
    response = await run_analysis(client)
    assert isinstance(response, AnalysisResponse)
    assert route.call_count == 2
    fallback_body: Any = json.loads(route.calls[1].request.content)
    assert fallback_body["response_format"]["type"] == "json_object"


async def test_analyze_schema_fallback_exhausts_budget(
    respx_mock: respx.MockRouter,
) -> None:
    """A schema probe cannot exceed a one-request budget to make its fallback."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(400, text="schema unsupported"),
    )
    client = LMStudioClient(base_url=BASE_URL, model=MODEL, max_retries=1, retry_backoff_sec=0)
    with pytest.raises(LLMError, match="HTTP 400"):
        await run_analysis(client)
    assert route.call_count == 1
    await client.aclose()


async def test_analyze_schema_fallback_non_success(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """A failed JSON-object fallback is not retried outside the shared budget."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=[
            httpx.Response(400, text="schema unsupported"),
            httpx.Response(400, text="invalid request"),
        ],
    )
    with pytest.raises(LLMError, match="HTTP 400"):
        await run_analysis(client)
    assert route.call_count == 2


async def test_analyze_404_single_request_and_hint(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """Model-not-found fails once and directs the operator to the model list."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(404, text="model absent"),
    )
    with pytest.raises(LLMError, match="/v1/models"):
        await run_analysis(client)
    assert route.call_count == 1


async def test_analyze_404_after_transient_uses_two_requests(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """A non-retryable 404 ends the call after a preceding transient request."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=[
            httpx.Response(500, text="temporary"),
            httpx.Response(404, text="model absent"),
        ],
    )
    with pytest.raises(LLMError, match="/v1/models"):
        await run_analysis(client)
    assert route.call_count == 2


async def test_analyze_connection_error_is_not_retried(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """Connection refusal is surfaced immediately for a later event retry."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=httpx.ConnectError("refused"),
    )
    with pytest.raises(LLMConnectionError, match="LM Studio недоступен"):
        await run_analysis(client)
    assert route.call_count == 1


async def test_analyze_other_4xx_fails_without_retry(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """A client error without a schema marker fails immediately."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(400, text="invalid request"),
    )
    with pytest.raises(LLMError, match="HTTP 400"):
        await run_analysis(client)
    assert route.call_count == 1


async def test_analyze_three_transient_responses_use_three_requests(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """Three 5xx responses exhaust exactly three HTTP-request budget units."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=[httpx.Response(500, text="busy") for _ in range(3)],
    )
    with pytest.raises(LLMError, match="max retries"):
        await run_analysis(client)
    assert route.call_count == 3


async def test_analyze_two_transient_then_success_uses_three_requests(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """Two transient failures followed by success fit the full budget."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=[
            httpx.Response(500, text="temporary one"),
            httpx.Response(503, text="temporary two"),
            valid_chat_response(),
        ],
    )
    response = await run_analysis(client)
    assert isinstance(response, AnalysisResponse)
    assert route.call_count == 3


async def test_analyze_three_invalid_json_responses_use_three_requests(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """Validation retries and schema-format changes share the same budget."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=[valid_chat_response("not JSON") for _ in range(3)],
    )
    with pytest.raises(LLMError, match="max retries.*validation"):
        await run_analysis(client)
    assert route.call_count == 3


async def test_analyze_malformed_response_missing_choices(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """Malformed successful bodies fail without spending another request."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json={"unexpected": []}),
    )
    with pytest.raises(LLMError, match="malformed chat response"):
        await run_analysis(client)
    assert route.call_count == 1


async def test_analyze_non_string_content_is_malformed(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """A non-string assistant content value is rejected as malformed."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": {"text": "x"}}}]}
        ),
    )
    with pytest.raises(LLMError, match="malformed chat response: TypeError"):
        await run_analysis(client)
    assert route.call_count == 1


async def test_analyze_redirect_status_fails_without_json_parse(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """A redirect status is rejected before parsing a response body."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(302, text="not a completion"),
    )
    with pytest.raises(LLMError, match="HTTP 302"):
        await run_analysis(client)
    assert route.call_count == 1


async def test_analyze_read_timeout_retries_within_budget(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """Read timeouts consume one HTTP-request budget unit each."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=[httpx.ReadTimeout("timed out") for _ in range(3)],
    )
    with pytest.raises(LLMError, match="max retries"):
        await run_analysis(client)
    assert route.call_count == 3


async def test_analyze_429_uses_three_requests(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """Rate limiting exhausts exactly three request budget units."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=[httpx.Response(429, text="busy") for _ in range(3)],
    )
    with pytest.raises(LLMError, match="max retries"):
        await run_analysis(client)
    assert route.call_count == 3


async def test_analyze_logs_contextual_warnings(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """Context validation warnings pass through the configured logger."""
    respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        return_value=valid_chat_response(),
    )
    response = await client.analyze(
        messages=MESSAGES,
        known_player_ids=frozenset(),
        role_names_ru=frozenset({ROLE_NAME}),
    )
    assert response.analysis == []


async def test_analyze_stream_retries_server_error_within_budget(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """A transient stream response consumes one request before recovery."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=[httpx.Response(500, text="temporary"), stream_response()],
    )
    response = await run_stream(client)
    assert isinstance(response, AnalysisResponse)
    assert route.call_count == 2


async def test_analyze_stream_connect_error_is_not_retried(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """A stream connection refusal becomes LLMConnectionError immediately."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=httpx.ConnectError("refused"),
    )
    with pytest.raises(LLMConnectionError, match="LM Studio недоступен"):
        await run_stream(client)
    assert route.call_count == 1


async def test_analyze_stream_404_fails_without_retry(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """A stream 404 is non-retryable and includes the configured-model hint."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(404, text="missing"),
    )
    with pytest.raises(LLMError, match="/v1/models"):
        await run_stream(client)
    assert route.call_count == 1


async def test_analyze_stream_other_4xx_fails_without_retry(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """Other stream client errors are surfaced immediately."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(400, text="invalid stream request"),
    )
    with pytest.raises(LLMError, match="HTTP 400"):
        await run_stream(client)
    assert route.call_count == 1


async def test_analyze_stream_empty_data_line_and_malformed_choices_are_ignored(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """Empty data and malformed choices events are skipped before valid content."""
    response = sse_response(
        "",
        {"choices": []},
        {"choices": "not-a-list"},
        {"choices": [{}]},
        {"choices": [{"delta": None}]},
        {"choices": [{"delta": {"content": VALID_CONTENT}}]},
        "[DONE]",
    )
    respx_mock.post(f"{BASE_URL}/chat/completions").mock(return_value=response)
    result = await run_stream(client)
    assert isinstance(result, AnalysisResponse)


async def test_analyze_stream_invalid_sse_json_fails_immediately(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """Malformed SSE JSON is a non-retryable LLMError."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        return_value=sse_response("{not-json}", "[DONE]"),
    )
    with pytest.raises(LLMError, match="invalid streaming JSON"):
        await run_stream(client)
    assert route.call_count == 1


async def test_analyze_stream_without_done_fails_without_retry(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """A cleanly ended stream without the completion marker is rejected."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        return_value=sse_response(
            {"choices": [{"delta": {"content": VALID_CONTENT}}]},
        ),
    )
    with pytest.raises(LLMError, match="stream ended before"):
        await run_stream(client)
    assert route.call_count == 1


async def test_analyze_stream_read_error_retries_whole_stream(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """A mid-stream read failure retries the complete request once."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=[
            httpx.Response(200, stream=BrokenSSEStream()),
            stream_response(),
        ],
    )
    response = await run_stream(client)
    assert isinstance(response, AnalysisResponse)
    assert route.call_count == 2


async def test_analyze_stream_three_read_errors_use_three_requests(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """Three stream disconnects exhaust exactly three budget units."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=[httpx.Response(200, stream=BrokenSSEStream()) for _ in range(3)],
    )
    with pytest.raises(LLMError, match="max retries"):
        await run_stream(client)
    assert route.call_count == 3


async def test_analyze_stream_content_only_does_not_emit_reasoning(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """Content-only model output is never sent to the reasoning callback."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        return_value=sse_response(
            {"choices": [{"delta": {"content": VALID_CONTENT}}]},
            "[DONE]",
        ),
    )
    chunks: list[str] = []

    def collect_reasoning(chunk: str) -> None:
        chunks.append(chunk)

    response = await client.analyze_stream(
        messages=MESSAGES,
        known_player_ids=frozenset({100}),
        role_names_ru=frozenset({ROLE_NAME}),
        on_reasoning_chunk=collect_reasoning,
    )
    assert isinstance(response, AnalysisResponse)
    assert chunks == []
    assert route.call_count == 1


async def test_analyze_stream_emits_only_dedicated_reasoning_fields(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """Dedicated reasoning fields reach the callback while content builds JSON."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        return_value=stream_response(),
    )
    chunks: list[str] = []

    def collect_reasoning(chunk: str) -> None:
        chunks.append(chunk)

    response = await client.analyze_stream(
        messages=MESSAGES,
        known_player_ids=frozenset({100}),
        role_names_ru=frozenset({ROLE_NAME}),
        on_reasoning_chunk=collect_reasoning,
    )
    assert isinstance(response, AnalysisResponse)
    assert "".join(chunks) == "A short explanation."
    assert "Player A" not in "".join(chunks)
    assert route.call_count == 1


async def test_analyze_stream_reasoning_logged_without_content_leak(
    client: LMStudioClient,
    respx_mock: respx.MockRouter,
    tmp_path: Path,
) -> None:
    """Only dedicated reasoning deltas are written to the reasoning log."""
    respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        return_value=stream_response(),
    )
    logger.remove()
    try:
        setup_logging(log_dir=tmp_path)
        await run_stream(client)
        logger.complete()
        reasoning_log = (tmp_path / "reasoning.log").read_text(encoding="utf-8")
        assert "A short " in reasoning_log
        assert "explanation." in reasoning_log
        assert "Player A" not in reasoning_log
        assert '"analysis"' not in reasoning_log
    finally:
        logger.complete()
        logger.remove()


async def test_analyze_stream_log_reasoning_chunks_false(
    client: LMStudioClient,
    respx_mock: respx.MockRouter,
) -> None:
    """Reasoning callbacks still work while per-chunk debug logging is disabled."""
    respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        return_value=stream_response(),
    )
    captured = io.StringIO()
    sink_id = logger.add(
        captured,
        level="DEBUG",
        format="{message}",
        filter=is_reasoning_log,
    )
    chunks: list[str] = []

    def collect_reasoning(chunk: str) -> None:
        chunks.append(chunk)

    try:
        response = await client.analyze_stream(
            messages=MESSAGES,
            known_player_ids=frozenset({100}),
            role_names_ru=frozenset({ROLE_NAME}),
            on_reasoning_chunk=collect_reasoning,
            log_reasoning_chunks=False,
        )
    finally:
        logger.remove(sink_id)

    assert isinstance(response, AnalysisResponse)
    assert "".join(chunks) == "A short explanation."
    assert captured.getvalue() == ""


async def test_analyze_stream_log_reasoning_chunks_true(
    client: LMStudioClient,
    respx_mock: respx.MockRouter,
) -> None:
    """The default continues to write each reasoning chunk to the log."""
    respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        return_value=stream_response(),
    )
    captured = io.StringIO()
    sink_id = logger.add(
        captured,
        level="DEBUG",
        format="{message}",
        filter=is_reasoning_log,
    )
    try:
        response = await run_stream(client)
    finally:
        logger.remove(sink_id)

    logged_reasoning = captured.getvalue()
    assert isinstance(response, AnalysisResponse)
    assert "A short " in logged_reasoning
    assert "explanation." in logged_reasoning
    assert "Player A" not in logged_reasoning


async def test_analyze_stream_schema_validation_error_does_not_retry(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """Final validation after DONE never starts another HTTP request."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        return_value=sse_response(
            {"choices": [{"delta": {"content": '{"analysis": []}'}}]},
            "[DONE]",
        ),
    )
    with pytest.raises(LLMError, match="invalid streaming response"):
        await run_stream(client)
    assert route.call_count == 1


async def test_eof_retry_caps_max_tokens_4096(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """EOF retry caps max_tokens at 4096 (was 8192 → 112s latency spike)."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=[
            sse_response({"choices": [{"delta": {"content": '{"analysis": ['}}]}, "[DONE]"),
            sse_response({"choices": [{"delta": {"content": VALID_CONTENT}}]}, "[DONE]"),
        ]
    )
    result = await run_stream(client)
    assert result.analysis[0].player_id == 100
    assert route.call_count == 2
    assert json.loads(route.calls[0].request.content)["max_tokens"] == 2048
    assert json.loads(route.calls[1].request.content)["max_tokens"] == 4096


async def test_build_payload_contains_schema(client: LMStudioClient) -> None:
    """The JSON Schema request format embeds the public response schema."""
    payload = client._build_payload(MESSAGES, response_format="json_schema")
    assert payload["response_format"]["type"] == "json_schema"
    schema = payload["response_format"]["json_schema"]
    assert schema["name"] == "AnalysisResponse"
    assert "properties" in schema["schema"]
    without_format = client._build_payload(MESSAGES, response_format=None)
    assert "response_format" not in without_format


def test_build_payload_rejects_unsupported_format(client: LMStudioClient) -> None:
    """Unsupported response formats are rejected before making a request."""
    with pytest.raises(ValueError, match="unsupported response format"):
        client._build_payload(MESSAGES, response_format="xml")


async def test_aclose_is_idempotent(client: LMStudioClient) -> None:
    """Closing an owned client twice is safe."""
    await client.aclose()
    await client.aclose()
    assert client._client.is_closed is True


async def test_operations_after_aclose_raise_domain_error() -> None:
    """All public requests fail with LLMError after the wrapper is closed."""
    external = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
    instance = LMStudioClient(base_url=BASE_URL, model=MODEL, client=external)
    await instance.aclose()
    with pytest.raises(LLMError, match="client is closed"):
        await instance.healthcheck()
    with pytest.raises(LLMError, match="client is closed"):
        await run_analysis(instance)
    with pytest.raises(LLMError, match="client is closed"):
        await run_stream(instance)
    assert external.is_closed is False
    await external.aclose()


async def test_aclose_does_not_close_or_disable_external_client() -> None:
    """The external transport remains usable after wrapper close."""
    external = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"ok": True}))
    )
    instance = LMStudioClient(base_url=BASE_URL, model=MODEL, client=external)
    await instance.aclose()
    assert external.is_closed is False
    response = await external.get("http://mock.local/health")
    assert response.json() == {"ok": True}
    await external.aclose()


SCHEMA_REJECTION_BODY = "'response_format.type' must be 'json_schema' or 'text'"


def fenced_content(content: str = VALID_CONTENT) -> str:
    """Wrap JSON content in a markdown code fence as a plain-text model answer."""
    return f"```json\n{content}\n```"


async def test_analyze_stream_uses_json_schema(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """The first stream request asks the server for a strict JSON schema."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        return_value=stream_response(),
    )
    response = await run_stream(client)
    assert isinstance(response, AnalysisResponse)
    assert route.call_count == 1
    first_body: Any = json.loads(route.calls[0].request.content)
    assert first_body["response_format"]["type"] == "json_schema"
    assert first_body["stream"] is True


async def test_analyze_stream_fallback_on_400_schema(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """A server rejecting json_schema is retried once with response_format=text."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=[
            httpx.Response(400, json={"error": SCHEMA_REJECTION_BODY}),
            stream_response(),
        ],
    )
    response = await run_stream(client)
    assert isinstance(response, AnalysisResponse)
    assert route.call_count == 2
    fallback_body: Any = json.loads(route.calls[1].request.content)
    assert fallback_body["response_format"]["type"] == "text"


async def test_analyze_stream_fallback_with_markdown_fence(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """A fenced plain-text fallback answer has its markdown fence stripped."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=[
            httpx.Response(400, json={"error": SCHEMA_REJECTION_BODY}),
            sse_response(
                {"choices": [{"delta": {"content": fenced_content()}}]},
                "[DONE]",
            ),
        ],
    )
    response = await run_stream(client)
    assert isinstance(response, AnalysisResponse)
    assert response.analysis[0].player_id == 100
    assert route.call_count == 2


async def test_analyze_stream_budget_with_fallback(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """The json_schema downgrade shares one budget of at most three requests."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=[
            httpx.Response(400, json={"error": SCHEMA_REJECTION_BODY}),
            httpx.Response(400, json={"error": SCHEMA_REJECTION_BODY}),
            stream_response(),
        ],
    )
    response = await run_stream(client)
    assert isinstance(response, AnalysisResponse)
    # Schema rejection (1) + first text attempt (2) + successful retry (3).
    assert route.call_count == 3
    assert route.call_count <= client.max_retries


async def test_eof_retry(client: LMStudioClient, respx_mock: respx.MockRouter) -> None:
    client.max_tokens = 4096
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=[
            sse_response({"choices": [{"delta": {"content": '{"analysis": ['}}]}, "[DONE]"),
            sse_response({"choices": [{"delta": {"content": VALID_CONTENT}}]}, "[DONE]"),
        ]
    )
    result = await run_stream(client)
    assert result.analysis[0].player_id == 100
    assert route.call_count == 2
    assert json.loads(route.calls[0].request.content)["max_tokens"] == 4096
    assert json.loads(route.calls[1].request.content)["max_tokens"] == 4096
    assert client.max_tokens == 4096


async def test_eof_stops_after_two_attempts(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """F1. Второй обрезанный ответ не тратит третий полный токен-бюджет."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=[
            sse_response({"choices": [{"delta": {"content": '{"analysis": ['}}]}, "[DONE]"),
            sse_response({"choices": [{"delta": {"content": '{"analysis": [{"'}}]}, "[DONE]"),
            stream_response(),
        ]
    )

    result = await run_stream(client)

    # Пустой, но валидный ответ вместо LLMError после трёх полных попыток.
    assert result.analysis == []
    assert result.game_read
    assert route.call_count == 2
    assert json.loads(route.calls[1].request.content)["max_tokens"] == 4096


async def test_eof_single_retry_succeeds(
    client: LMStudioClient, respx_mock: respx.MockRouter
) -> None:
    """F1. Один retry после EOF по-прежнему спасает анализ."""
    route = respx_mock.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=[
            sse_response({"choices": [{"delta": {"content": '{"analysis": ['}}]}, "[DONE]"),
            sse_response({"choices": [{"delta": {"content": VALID_CONTENT}}]}, "[DONE]"),
        ]
    )

    result = await run_stream(client)

    assert result.analysis[0].player_id == 100
    assert route.call_count == 2


async def test_empty_analysis_response_is_valid_contract() -> None:
    """F2. Пустой ответ (EOF-фоллбэк) валиден: 0 predictions — это OK."""
    response = empty_analysis_response()

    assert response.analysis == []
    assert response.game_read
    assert response.reasoning
