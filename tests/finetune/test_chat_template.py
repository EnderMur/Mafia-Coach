"""Семантика сериализации диалога при обучении и инференсе."""

import jinja2
import pytest

from scripts.finetune.template import JINJA, OLLAMA, check_template_pair
from scripts.finetune.train_lora import tokenize_example


def test_chatml_protocol_matches_ollama():
    # Ollama range .Messages emits role/content/im_end, then assistant prefix;
    # fixed Go text checked exactly so the test fails if either side drifts.
    expected_go = '{{- range .Messages }}{{- "<|im_start|>" }}{{- .Role }}{{- "\\n" }}{{- .Content }}{{- "<|im_end|>\\n" }}{{- end }}{{- "<|im_start|>assistant\\n" }}{{- .Response }}{{- if .Response }}{{- "<|im_end|>\\n" }}{{- end }}\n'
    assert OLLAMA.read_text(encoding="utf-8") == expected_go
    messages = [{"role": "system", "content": "Правила\nигры"}, {"role": "user", "content": "Кто?"}]
    template = jinja2.Environment().from_string(JINJA.read_text(encoding="utf-8"))
    prefix = template.render(messages=messages, add_generation_prompt=True)
    ollama = (
        "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        + "<|im_start|>assistant\n"
    )
    assert prefix == ollama
    full = template.render(
        messages=messages + [{"role": "assistant", "content": "{}"}], add_generation_prompt=False
    )
    assert full == ollama + "{}<|im_end|>\n"
    check_template_pair(JINJA, OLLAMA)


def test_pair_rejects_mismatched_template(tmp_path):
    altered = tmp_path / "altered.jinja"
    altered.write_text("другой", encoding="utf-8")
    with pytest.raises(ValueError):
        check_template_pair(altered, OLLAMA)


def test_assistant_only_loss_and_no_truncation():
    class Tokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            template = jinja2.Environment().from_string(JINJA.read_text(encoding="utf-8"))
            return list(
                template.render(
                    messages=messages, add_generation_prompt=add_generation_prompt
                ).encode()
            )

    messages = [
        {"role": "system", "content": "правила"},
        {"role": "user", "content": "игра"},
        {"role": "assistant", "content": "{}"},
    ]
    row = tokenize_example(Tokenizer(), messages, 1000)
    assert -100 in row["labels"] and row["labels"][-1] != -100
    with pytest.raises(ValueError, match="усечение"):
        tokenize_example(Tokenizer(), messages, 10)


def test_server_comparison_detects_token_drift():
    import httpx

    from scripts.finetune.check_ollama_template import compare_ollama_prompt

    messages = [
        {"role": "system", "content": "правила"},
        {"role": "user", "content": "игра"},
        {"role": "assistant", "content": '{"analysis": [], "game_read": "Игра."}'},
    ]

    class Tokenizer:
        def apply_chat_template(self, rows, tokenize, add_generation_prompt):
            return [10, 11] if tokenize else "rendered"

    def handler(request):
        import json

        body = json.loads(request.content)
        if request.url.path == "/api/chat":
            assert body["think"] is False
        if body.get("_debug_render_only"):
            return httpx.Response(200, json={"_debug_info": {"rendered_template": "rendered"}})
        if request.url.path == "/api/chat":
            return httpx.Response(
                200, json={"prompt_eval_count": 2, "done": True, "message": {"content": ""}}
            )
        return httpx.Response(200, json={"context": [10, 12, 99], "prompt_eval_count": 2})

    with (
        httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test") as client,
        pytest.raises(ValueError, match="Токены"),
    ):
        compare_ollama_prompt(client, "model", Tokenizer(), messages, 100)


@pytest.mark.integration
def test_real_ollama_matches_training_tokens():
    import os
    from pathlib import Path

    from scripts.finetune.check_ollama_template import run
    from scripts.finetune.common import load_config

    model = os.environ.get("FINETUNE_OLLAMA_MODEL")
    config_path = os.environ.get("FINETUNE_CONFIG")
    sample = os.environ.get("FINETUNE_SAMPLE")
    if not model or not config_path or not sample:
        pytest.skip("Задайте FINETUNE_OLLAMA_MODEL, FINETUNE_CONFIG и FINETUNE_SAMPLE")
    result = run(
        load_config(config_path),
        os.environ.get("FINETUNE_OLLAMA_URL", "http://127.0.0.1:11434"),
        model,
        Path(sample),
        local_files_only=os.environ.get("FINETUNE_LOCAL_FILES_ONLY") == "1",
    )
    assert result["prompt_tokens"] > 0
    assert result["mode"] in {"count_only", "render_and_count", "render_and_tokens"}


def test_server_comparison_success_and_render_drift():
    import httpx

    from scripts.finetune.check_ollama_template import compare_ollama_prompt

    messages = [
        {"role": "system", "content": "системное сообщение"},
        {"role": "user", "content": "контекст"},
        {"role": "assistant", "content": '{"analysis": [], "game_read": "Игра."}'},
    ]

    class Tokenizer:
        def apply_chat_template(self, rows, tokenize, add_generation_prompt):
            return [10, 11] if tokenize else "rendered"

    def handler(request):
        import json

        body = json.loads(request.content)
        if request.url.path == "/api/chat":
            assert body["think"] is False
        if body.get("_debug_render_only"):
            return httpx.Response(200, json={"_debug_info": {"rendered_template": "rendered"}})
        if request.url.path == "/api/chat":
            return httpx.Response(
                200, json={"prompt_eval_count": 2, "done": True, "message": {"content": ""}}
            )
        return httpx.Response(200, json={"context": [10, 11, 99], "prompt_eval_count": 2})

    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test") as client:
        result = compare_ollama_prompt(client, "model", Tokenizer(), messages, 100)
        assert result["prompt_tokens"] == 2
        assert result["mode"] == "render_and_tokens"

    def bad_render(request):
        return httpx.Response(200, json={"_debug_info": {"rendered_template": "other"}})

    with (
        httpx.Client(transport=httpx.MockTransport(bad_render), base_url="http://test") as client,
        pytest.raises(ValueError, match="Рендер"),
    ):
        compare_ollama_prompt(client, "model", Tokenizer(), messages, 100)


def test_public_ollama_fallback_and_think_false():
    import json

    import httpx

    from scripts.finetune.check_ollama_template import compare_ollama_prompt

    messages = [
        {"role": "system", "content": "правила"},
        {"role": "user", "content": "игра"},
        {"role": "assistant", "content": '{"analysis": [], "game_read": "Игра."}'},
    ]
    requests = []

    class Tokenizer:
        def apply_chat_template(self, rows, tokenize, add_generation_prompt):
            return [10, 11] if tokenize else "rendered"

    def handler(request):
        requests.append((request.url.path, json.loads(request.content)))
        # Публичная версия может игнорировать неофициальный debug-флаг.
        return httpx.Response(
            200,
            json={
                "done": True,
                "prompt_eval_count": 2,
                "message": {"content": "ок", "thinking": ""},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test") as client:
        result = compare_ollama_prompt(client, "model", Tokenizer(), messages, 100)
    assert result["mode"] == "count_only"
    assert result["prompt_tokens"] == 2
    assert len(requests) == 2
    assert all(path == "/api/chat" and body["think"] is False for path, body in requests)
    assert requests[0][1]["_debug_render_only"] is True
    assert "_debug_render_only" not in requests[1][1]


def test_fallback_rejects_wrong_count_and_nonempty_thinking():
    import httpx

    from scripts.finetune.check_ollama_template import compare_ollama_prompt

    messages = [
        {"role": "system", "content": "правила"},
        {"role": "user", "content": "игра"},
        {"role": "assistant", "content": '{"analysis": [], "game_read": "Игра."}'},
    ]

    class Tokenizer:
        def apply_chat_template(self, rows, tokenize, add_generation_prompt):
            return [10, 11] if tokenize else "rendered"

    for response, message in [
        ({"done": True, "prompt_eval_count": 3, "message": {}}, "Число токенов"),
        ({"done": True, "prompt_eval_count": 2, "message": {"thinking": "текст"}}, "thinking"),
        ({"done": False, "prompt_eval_count": 2}, "не завершил"),
    ]:
        with (
            httpx.Client(
                transport=httpx.MockTransport(
                    lambda request, response=response: httpx.Response(200, json=response)
                ),
                base_url="http://test",
            ) as client,
            pytest.raises((ValueError, RuntimeError), match=message),
        ):
            compare_ollama_prompt(client, "model", Tokenizer(), messages, 100)


def test_unsupported_debug_status_falls_back():
    import json

    import httpx

    from scripts.finetune.check_ollama_template import compare_ollama_prompt

    messages = [
        {"role": "system", "content": "правила"},
        {"role": "user", "content": "игра"},
        {"role": "assistant", "content": '{"analysis": [], "game_read": "Игра."}'},
    ]

    class Tokenizer:
        def apply_chat_template(self, rows, tokenize, add_generation_prompt):
            return [10, 11] if tokenize else "rendered"

    def handler(request):
        body = json.loads(request.content)
        assert body["think"] is False
        if body.get("_debug_render_only"):
            return httpx.Response(400, json={"error": "unsupported"})
        return httpx.Response(200, json={"done": True, "prompt_eval_count": 2, "message": {}})

    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test") as client:
        assert (
            compare_ollama_prompt(client, "model", Tokenizer(), messages, 100)["mode"]
            == "count_only"
        )


def test_base_tokenizer_download_option_and_offline_error(monkeypatch):
    import sys
    from types import SimpleNamespace

    from scripts.finetune.check_ollama_template import load_base_tokenizer

    calls = []
    tokenizer = SimpleNamespace()

    def from_pretrained(model, **kwargs):
        calls.append((model, kwargs))
        return tokenizer

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=from_pretrained)),
    )
    config = {
        "base_model": "Qwen/Qwen3.5-9B",
        "adapter_dir": "does/not/exist",
        "chat_template": str(JINJA),
    }
    assert load_base_tokenizer(config) is tokenizer
    assert calls[-1] == ("Qwen/Qwen3.5-9B", {"local_files_only": False})
    load_base_tokenizer(config, local_files_only=True)
    assert calls[-1][1]["local_files_only"] is True
    assert tokenizer.chat_template == JINJA.read_text(encoding="utf-8")

    def missing(*args, **kwargs):
        raise OSError("кеш отсутствует")

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=missing)),
    )
    with pytest.raises(
        RuntimeError, match="Токенизатор Qwen/Qwen3.5-9B не найден в локальном кеше"
    ):
        load_base_tokenizer(config, local_files_only=True)
    with pytest.raises(OSError, match="кеш отсутствует"):
        load_base_tokenizer(config)


def test_cuda_warning_does_not_block_cpu(monkeypatch):
    import sys
    from types import SimpleNamespace

    from loguru import logger

    from scripts.finetune import check_ollama_template

    warnings = []
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(logger, "warning", lambda message, *args: warnings.append(message))
    check_ollama_template.warn_if_no_cuda()
    assert any("CUDA не обнаружена" in warning and "CPU" in warning for warning in warnings)


def test_render_mode_without_generate_debug():
    import json

    import httpx

    from scripts.finetune.check_ollama_template import compare_ollama_prompt

    messages = [
        {"role": "system", "content": "правила"},
        {"role": "user", "content": "игра"},
        {"role": "assistant", "content": '{"analysis": [], "game_read": "Игра."}'},
    ]

    class Tokenizer:
        def apply_chat_template(self, rows, tokenize, add_generation_prompt):
            return [10, 11] if tokenize else "rendered"

    def handler(request):
        body = json.loads(request.content)
        if request.url.path == "/api/chat" and body.get("_debug_render_only"):
            return httpx.Response(200, json={"_debug_info": {"rendered_template": "rendered"}})
        if request.url.path == "/api/chat":
            return httpx.Response(200, json={"done": True, "prompt_eval_count": 2, "message": {}})
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test") as client:
        result = compare_ollama_prompt(client, "model", Tokenizer(), messages, 100)
    assert result["mode"] == "render_and_count"


def test_run_with_public_ollama_base_tokenizer_and_no_cuda(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace

    import httpx
    from loguru import logger

    from scripts.finetune import check_ollama_template

    messages = [
        {"role": "system", "content": "правила"},
        {"role": "user", "content": "игра"},
        {"role": "assistant", "content": '{"analysis": [], "game_read": "Игра."}'},
    ]
    sample = tmp_path / "sample.jsonl"
    sample.write_text(
        json.dumps({"messages": messages}, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    calls = []
    warnings = []

    class Tokenizer:
        chat_template = ""

        def apply_chat_template(self, rows, tokenize, add_generation_prompt):
            return [10, 11] if tokenize else "rendered"

    def from_pretrained(model, **kwargs):
        calls.append((model, kwargs))
        return Tokenizer()

    def handler(request):
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "test"})
        body = json.loads(request.content)
        assert body["think"] is False
        return httpx.Response(200, json={"done": True, "prompt_eval_count": 2, "message": {}})

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=from_pretrained)),
    )
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setattr(logger, "warning", lambda message, *args: warnings.append(message))
    config = {
        "base_model": "Qwen/Qwen3.5-9B",
        "chat_template": str(JINJA),
        "ollama_template": str(OLLAMA),
        "max_seq_length": 4096,
    }
    result = check_ollama_template.run(config, "http://test", "model", sample)
    assert result["mode"] == "count_only" and result["ollama_version"] == "test"
    assert calls == [("Qwen/Qwen3.5-9B", {"local_files_only": False})]
    assert any("CUDA не обнаружена" in warning for warning in warnings)
