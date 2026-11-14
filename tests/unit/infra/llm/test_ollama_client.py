"""HTTP-проверки параметров Ollama и независимости клиента LM Studio."""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
import pytest
import respx

from domain.exceptions import LLMConnectionError, LLMError
from infra.llm.client import LMStudioClient
from infra.llm.ollama import OLLAMA_THINK_PARAM, OLLAMA_THINK_VALUE, OllamaClient

OLLAMA_URL = "http://localhost:11434"
LM_URL = "http://localhost:1234/v1"
MODEL = "test-model"
MESSAGES = [{"role": "user", "content": "Данные партии"}]
ANSWER = json.dumps({"analysis": [], "game_read": "Партия продолжается."}, ensure_ascii=False)


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as mocked:
        yield mocked


def ollama_response(*, thinking: str = "") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "message": {"role": "assistant", "content": ANSWER, "thinking": thinking},
            "done": True,
            "done_reason": "stop",
        },
    )


def stream_response(*, thinking: str = "") -> httpx.Response:
    events = [
        {
            "message": {"role": "assistant", "content": ANSWER[:10], "thinking": thinking},
            "done": False,
        },
        {"message": {"role": "assistant", "content": ANSWER[10:]}, "done": True},
    ]
    return httpx.Response(
        200, text="".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events)
    )


@pytest.mark.parametrize("stream", [False, True])
async def test_ollama_each_chat_request_disables_thinking(
    router: respx.MockRouter, stream: bool
) -> None:
    route = router.post(f"{OLLAMA_URL}/api/chat").mock(
        return_value=stream_response() if stream else ollama_response()
    )
    client = OllamaClient(base_url=OLLAMA_URL, model=MODEL)
    try:
        if stream:
            chunks: list[str] = []

            def sink(chunk: str) -> None:
                chunks.append(chunk)

            result = await client.analyze_stream(
                messages=MESSAGES,
                known_player_ids=frozenset(),
                role_names_ru=frozenset(),
                on_reasoning_chunk=sink,
            )
            assert not chunks
        else:
            result = await client.analyze(
                messages=MESSAGES, known_player_ids=frozenset(), role_names_ru=frozenset()
            )
        assert result.game_read == "Партия продолжается."
        request = json.loads(route.calls.last.request.content)
        assert request["think"] is False
        assert request[OLLAMA_THINK_PARAM] is OLLAMA_THINK_VALUE is False
        assert request["stream"] is stream
        assert "analysis" in request["format"]["properties"]
    finally:
        await client.aclose()


@pytest.mark.parametrize("stream", [False, True])
async def test_lm_studio_request_never_receives_ollama_think(
    router: respx.MockRouter, stream: bool
) -> None:
    if stream:
        events = [
            {"choices": [{"delta": {"content": ANSWER}}]},
        ]
        response = httpx.Response(
            200,
            text="".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )
    else:
        response = httpx.Response(200, json={"choices": [{"message": {"content": ANSWER}}]})
    route = router.post(f"{LM_URL}/chat/completions").mock(return_value=response)
    client = LMStudioClient(base_url=LM_URL, model=MODEL)
    try:
        method = client.analyze_stream if stream else client.analyze
        result = await method(
            messages=MESSAGES, known_player_ids=frozenset(), role_names_ru=frozenset()
        )
        assert result.game_read == "Партия продолжается."
        assert "think" not in json.loads(route.calls.last.request.content)
    finally:
        await client.aclose()


async def test_retries_always_include_think_false(router: respx.MockRouter) -> None:
    route = router.post(f"{OLLAMA_URL}/api/chat").mock(
        side_effect=[httpx.Response(503), ollama_response()]
    )
    client = OllamaClient(base_url=OLLAMA_URL, model=MODEL, max_retries=2, retry_backoff_sec=0)
    try:
        await client.analyze(
            messages=MESSAGES, known_player_ids=frozenset(), role_names_ru=frozenset()
        )
        assert route.call_count == 2
        assert all(json.loads(call.request.content)["think"] is False for call in route.calls)
    finally:
        await client.aclose()


async def test_unsupported_parameter_is_explicit_error_without_fallback(
    router: respx.MockRouter,
) -> None:
    route = router.post(f"{OLLAMA_URL}/api/chat").mock(
        return_value=httpx.Response(400, text="unknown field think")
    )
    client = OllamaClient(base_url=OLLAMA_URL, model=MODEL, max_retries=3, retry_backoff_sec=0)
    try:
        with pytest.raises(LLMError, match="HTTP 400"):
            await client.analyze(
                messages=MESSAGES, known_player_ids=frozenset(), role_names_ru=frozenset()
            )
        assert route.call_count == 1
        assert json.loads(route.calls[0].request.content)["think"] is False
    finally:
        await client.aclose()


@pytest.mark.parametrize("stream", [False, True])
async def test_nonempty_thinking_fails_closed(router: respx.MockRouter, stream: bool) -> None:
    route = router.post(f"{OLLAMA_URL}/api/chat").mock(
        return_value=stream_response(thinking="Скрытый текст")
        if stream
        else ollama_response(thinking="Скрытый текст")
    )
    client = OllamaClient(base_url=OLLAMA_URL, model=MODEL, max_retries=3, retry_backoff_sec=0)
    try:
        method = client.analyze_stream if stream else client.analyze
        with pytest.raises(LLMError, match="вернул thinking при think=false"):
            await method(messages=MESSAGES, known_player_ids=frozenset(), role_names_ru=frozenset())
        assert route.call_count == 1
    finally:
        await client.aclose()


async def test_invalid_output_stays_within_request_budget(router: respx.MockRouter) -> None:
    route = router.post(f"{OLLAMA_URL}/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"content": "не JSON"}, "done": True})
    )
    client = OllamaClient(base_url=OLLAMA_URL, model=MODEL, max_retries=2, retry_backoff_sec=0)
    try:
        with pytest.raises(LLMError, match="после 2 попыток"):
            await client.analyze(
                messages=MESSAGES, known_player_ids=frozenset(), role_names_ru=frozenset()
            )
        assert route.call_count == 2
        assert all(json.loads(call.request.content)["think"] is False for call in route.calls)
    finally:
        await client.aclose()


async def test_interrupted_stream_fails(router: respx.MockRouter) -> None:
    router.post(f"{OLLAMA_URL}/api/chat").mock(
        return_value=httpx.Response(
            200, text=json.dumps({"message": {"content": ANSWER}, "done": False}) + "\n"
        )
    )
    client = OllamaClient(base_url=OLLAMA_URL, model=MODEL)
    try:
        with pytest.raises(LLMError, match="оборван"):
            await client.analyze_stream(
                messages=MESSAGES, known_player_ids=frozenset(), role_names_ru=frozenset()
            )
    finally:
        await client.aclose()


async def test_healthcheck_and_client_close(router: respx.MockRouter) -> None:
    router.get(f"{OLLAMA_URL}/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": MODEL}]})
    )
    client = OllamaClient(base_url=OLLAMA_URL + "/", model=MODEL)
    await client.healthcheck()
    await client.aclose()
    await client.aclose()
    with pytest.raises(LLMError, match="закрыт"):
        await client.healthcheck()


async def test_healthcheck_model_missing(router: respx.MockRouter) -> None:
    router.get(f"{OLLAMA_URL}/api/tags").mock(return_value=httpx.Response(200, json={"models": []}))
    client = OllamaClient(base_url=OLLAMA_URL, model=MODEL)
    try:
        with pytest.raises(LLMError, match="не найдена"):
            await client.healthcheck()
    finally:
        await client.aclose()


async def test_connection_error(router: respx.MockRouter) -> None:
    router.post(f"{OLLAMA_URL}/api/chat").mock(side_effect=httpx.ConnectError("нет соединения"))
    client = OllamaClient(base_url=OLLAMA_URL, model=MODEL)
    try:
        with pytest.raises(LLMConnectionError, match="недоступен"):
            await client.analyze(
                messages=MESSAGES, known_player_ids=frozenset(), role_names_ru=frozenset()
            )
    finally:
        await client.aclose()
