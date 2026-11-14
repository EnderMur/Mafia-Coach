"""Сравнить реальные рендер и токены Ollama с токенизатором обучения."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from scripts.finetune.common import load_config, read_jsonl, validate_messages
from scripts.finetune.template import check_template_pair

IDENTITY_TEMPLATE = (
    Path(__file__).resolve().parents[2] / "infra/llm/prompts/finetune_identity.gotmpl"
)


def _post(client: httpx.Client, path: str, data: dict[str, Any]) -> dict[str, Any]:
    response = client.post(path, json=data)
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise TypeError("Некорректный ответ Ollama")
    return result


def _rendered(response: dict[str, Any]) -> str:
    info = response.get("_debug_info")
    if not isinstance(info, dict) or not isinstance(info.get("rendered_template"), str):
        raise TypeError("Ollama не вернул _debug_info.rendered_template; обновите раннер")
    return str(info["rendered_template"])


def compare_ollama_prompt(
    client: httpx.Client,
    model: str,
    tokenizer: Any,
    messages: list[dict[str, str]],
    num_ctx: int,
) -> dict[str, Any]:
    """Серверный /api/chat рендер + серверная токенизация через /api/generate context.

    context устаревший API: отсутствие ответа считается ошибкой, не успехом.
    Тест не выводит на экран сообщения, промпт или идентификаторы токенов.
    """
    validate_messages(messages)
    prefix = messages[:-1]
    training_text = tokenizer.apply_chat_template(
        prefix, tokenize=False, add_generation_prompt=True
    )
    training_ids = tokenizer.apply_chat_template(prefix, tokenize=True, add_generation_prompt=True)
    options = {"num_ctx": num_ctx, "num_predict": 1, "temperature": 0}
    chat = {
        "model": model,
        "messages": prefix,
        "stream": False,
        "truncate": False,
        "options": options,
        "_debug_render_only": True,
    }
    rendered = _rendered(_post(client, "/api/chat", chat))
    if rendered != training_text:
        raise ValueError("Рендер Ollama /api/chat отличается от шаблона обучения")

    generate = {
        "model": model,
        "prompt": rendered,
        "stream": False,
        "truncate": False,
        "template": IDENTITY_TEMPLATE.read_text(encoding="utf-8").rstrip("\n"),
        "options": options,
    }
    debug = _post(client, "/api/generate", {**generate, "_debug_render_only": True})
    if _rendered(debug) != rendered:
        raise ValueError("Контрольный запрос Ollama изменил промпт")
    chat_run = _post(client, "/api/chat", {**chat, "_debug_render_only": False})
    chat_count = chat_run.get("prompt_eval_count")
    if type(chat_count) is not int or chat_count != len(training_ids):
        raise ValueError("Число токенов реального /api/chat отличается от обучения")
    result = _post(client, "/api/generate", generate)
    context, count = result.get("context"), result.get("prompt_eval_count")
    if (
        not isinstance(context, list)
        or not all(type(i) is int for i in context)
        or type(count) is not int
        or count <= 0
        or len(context) < count
    ):
        raise RuntimeError("Ollama не вернул context и prompt_eval_count; сравнение невозможно")
    if count != len(training_ids) or context[:count] != training_ids:
        raise ValueError("Токены Ollama отличаются от токенов обучения (включая BOS/усечение)")
    return {"model": model, "prompt_tokens": count}


def run(config: dict[str, Any], url: str, model: str, sample: Path) -> dict[str, Any]:
    check_template_pair(Path(config["chat_template"]), Path(config["ollama_template"]))
    records = read_jsonl(sample)
    if not records:
        raise ValueError("Файл со снимками пуст")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["adapter_dir"], local_files_only=True)
    tokenizer.chat_template = Path(config["chat_template"]).read_text(encoding="utf-8")
    with httpx.Client(base_url=url.rstrip("/"), timeout=180.0) as client:
        version = _post(client, "/api/show", {"model": model})
        if version.get("model_info") is None:
            raise RuntimeError("Ollama не подтвердил локальную модель")
        result = compare_ollama_prompt(
            client, model, tokenizer, records[0]["messages"], int(config["max_seq_length"])
        )
        result["ollama_version"] = client.get("/api/version").json().get("version")
    logger.info("Сравнение реального раннера пройдено: {}", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка промпта и токенов Ollama")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, help="Имя созданной в Ollama модели")
    parser.add_argument("--sample", required=True, help="Проверенный JSONL с сообщениями")
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    run(load_config(args.config), args.url, args.model, Path(args.sample))


if __name__ == "__main__":
    main()
