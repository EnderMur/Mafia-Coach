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
        if body.get("_debug_render_only"):
            return httpx.Response(200, json={"_debug_info": {"rendered_template": "rendered"}})
        if request.url.path == "/api/chat":
            return httpx.Response(200, json={"prompt_eval_count": 2})
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
    )
    assert result["prompt_tokens"] > 0


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
        if body.get("_debug_render_only"):
            return httpx.Response(200, json={"_debug_info": {"rendered_template": "rendered"}})
        if request.url.path == "/api/chat":
            return httpx.Response(200, json={"prompt_eval_count": 2})
        return httpx.Response(200, json={"context": [10, 11, 99], "prompt_eval_count": 2})

    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test") as client:
        assert (
            compare_ollama_prompt(client, "model", Tokenizer(), messages, 100)["prompt_tokens"] == 2
        )

    def bad_render(request):
        return httpx.Response(200, json={"_debug_info": {"rendered_template": "other"}})

    with (
        httpx.Client(transport=httpx.MockTransport(bad_render), base_url="http://test") as client,
        pytest.raises(ValueError, match="Рендер"),
    ):
        compare_ollama_prompt(client, "model", Tokenizer(), messages, 100)
