"""Сравнить реальные рендер и токены Ollama с токенизатором обучения."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from infra.llm.ollama import OLLAMA_THINK_PARAM, OLLAMA_THINK_VALUE
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


def _rendered(response: dict[str, Any]) -> str | None:
    info = response.get("_debug_info")
    if isinstance(info, dict) and isinstance(info.get("rendered_template"), str):
        return str(info["rendered_template"])
    return None


def _optional_debug(client: httpx.Client, path: str, data: dict[str, Any]) -> dict[str, Any]:
    """Старый сервер может отклонить нестандартный флаг, остальные ошибки не скрываем."""
    try:
        return _post(client, path, {**data, "_debug_render_only": True})
    except httpx.HTTPStatusError as error:
        if error.response.status_code not in (400, 404, 422):
            raise
        logger.warning(
            "Ollama не поддерживает debug-рендер {}: {}", path, error.response.status_code
        )
        return {}


def _chat_count(response: dict[str, Any], expected: int) -> None:
    if response.get("error"):
        raise RuntimeError(f"Ошибка Ollama: {response['error']}")
    if response.get("done") is not True:
        raise RuntimeError("Ollama не завершил запрос /api/chat")
    message = response.get("message")
    if not isinstance(message, dict):
        raise TypeError("В ответе Ollama отсутствует message")
    thinking = message.get("thinking")
    if thinking is not None and not isinstance(thinking, str):
        raise RuntimeError("Некорректное поле thinking в ответе Ollama")
    if thinking:
        raise RuntimeError("Ollama вернул thinking при think=false")
    count = response.get("prompt_eval_count")
    if type(count) is not int or count <= 0 or count != expected:
        raise ValueError("Число токенов реального /api/chat отличается от обучения")


def compare_ollama_prompt(
    client: httpx.Client,
    model: str,
    tokenizer: Any,
    messages: list[dict[str, str]],
    num_ctx: int,
) -> dict[str, Any]:
    """Рендер/ID при наличии debug API; иначе только сравнение числа токенов.

    Совпадение числа токенов не доказывает совпадение их ID или самого шаблона.
    Тест не выводит сообщения, промпт или идентификаторы токенов.
    """
    validate_messages(messages)
    prefix = messages[:-1]
    training_text = tokenizer.apply_chat_template(
        prefix, tokenize=False, add_generation_prompt=True
    )
    training_ids = tokenizer.apply_chat_template(prefix, tokenize=True, add_generation_prompt=True)
    if not isinstance(training_ids, list) or not training_ids:
        raise ValueError("Токенизатор не вернул список токенов обучающего промпта")
    options = {"num_ctx": num_ctx, "num_predict": 1, "temperature": 0}
    chat = {
        "model": model,
        "messages": prefix,
        "stream": False,
        "truncate": False,
        "options": options,
        OLLAMA_THINK_PARAM: OLLAMA_THINK_VALUE,
    }
    rendered = _rendered(_optional_debug(client, "/api/chat", chat))
    if rendered is not None and rendered != training_text:
        raise ValueError("Рендер Ollama /api/chat отличается от шаблона обучения")

    _chat_count(_post(client, "/api/chat", chat), len(training_ids))
    if rendered is None:
        logger.warning(
            "Ollama не предоставил debug-рендер: совпало только число токенов; "
            "совпадение шаблона и ID не доказано"
        )
        # Равное число токенов не доказывает совпадение шаблона/ID — это ослабленная проверка.
        return {
            "model": model,
            "prompt_tokens": len(training_ids),
            "mode": "count_only",
            "count_only": True,
        }

    # В некоторых версиях Ollama доступны дополнительный debug и legacy context.
    generate = {
        "model": model,
        "prompt": rendered,
        "stream": False,
        "truncate": False,
        "template": IDENTITY_TEMPLATE.read_text(encoding="utf-8").rstrip("\n"),
        "options": options,
        OLLAMA_THINK_PARAM: OLLAMA_THINK_VALUE,
    }
    generate_debug = _rendered(_optional_debug(client, "/api/generate", generate))
    if generate_debug is None:
        logger.warning("Debug /api/generate недоступен: проверены только рендер и число токенов")
        return {
            "model": model,
            "prompt_tokens": len(training_ids),
            "mode": "render_and_count",
            "count_only": False,
        }
    if generate_debug != rendered:
        raise ValueError("Контрольный запрос Ollama изменил промпт")
    result = _post(client, "/api/generate", generate)
    if result.get("error"):
        raise RuntimeError(f"Ошибка Ollama: {result['error']}")
    context, count = result.get("context"), result.get("prompt_eval_count")
    if context is None:
        logger.warning("Ollama не вернул context: проверены только рендер и число токенов")
        return {
            "model": model,
            "prompt_tokens": len(training_ids),
            "mode": "render_and_count",
            "count_only": False,
        }
    if (
        not isinstance(context, list)
        or not all(type(i) is int for i in context)
        or type(count) is not int
        or count <= 0
        or len(context) < count
    ):
        raise RuntimeError("Некорректные context или prompt_eval_count от Ollama")
    if count != len(training_ids) or context[:count] != training_ids:
        raise ValueError("Токены Ollama отличаются от токенов обучения (включая BOS/усечение)")
    return {
        "model": model,
        "prompt_tokens": count,
        "mode": "render_and_tokens",
        "count_only": False,
    }


def warn_if_no_cuda() -> None:
    """Проверка только информирует: для токенизации CUDA не нужна."""
    try:
        import torch
    except (ImportError, OSError):
        logger.warning(
            "PyTorch недоступен. Проверка шаблона возможна на CPU, "
            "но для обучения требуется совместимый torch с CUDA"
        )
        return
    try:
        available = torch.cuda.is_available()
    except (RuntimeError, OSError) as error:
        logger.warning("Не удалось проверить CUDA: {}. Проверка шаблона продолжится на CPU", error)
        return
    if not available:
        logger.warning(
            "CUDA не обнаружена. Для проверки шаблона это не критично (токенизация на CPU), "
            "но для обучения LoRA/QLoRA потребуется CUDA Toolkit и совместимый torch"
        )
        if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
            logger.warning(
                "CUDA_HOME/CUDA_PATH заданы, но torch.cuda.is_available() возвращает False"
            )


def load_base_tokenizer(config: dict[str, Any], *, local_files_only: bool = False) -> Any:
    """Загрузить токенизатор базы; при первом запуске скачивание разрешено."""
    from transformers import AutoTokenizer

    base_model = config["base_model"]
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=local_files_only)
    except OSError as error:
        if local_files_only:
            raise RuntimeError(
                f"Токенизатор {base_model} не найден в локальном кеше. "
                "Запустите с local_files_only=False (без --local-files-only) "
                "или скачайте токенизатор заранее"
            ) from error
        raise
    tokenizer.chat_template = Path(config["chat_template"]).read_text(encoding="utf-8")
    return tokenizer


def run(
    config: dict[str, Any], url: str, model: str, sample: Path, *, local_files_only: bool = False
) -> dict[str, Any]:
    warn_if_no_cuda()
    check_template_pair(Path(config["chat_template"]), Path(config["ollama_template"]))
    records = read_jsonl(sample)
    if not records:
        raise ValueError("Файл со снимками пуст")
    tokenizer = load_base_tokenizer(config, local_files_only=local_files_only)
    with httpx.Client(base_url=url.rstrip("/"), timeout=180.0) as client:
        result = compare_ollama_prompt(
            client, model, tokenizer, records[0]["messages"], int(config["max_seq_length"])
        )
        version_response = client.get("/api/version")
        if version_response.is_success:
            try:
                version = version_response.json()
                result["ollama_version"] = (
                    version.get("version") if isinstance(version, dict) else None
                )
            except ValueError:
                logger.warning("Ollama вернул некорректный ответ /api/version")
        else:
            logger.warning("Версия Ollama недоступна: HTTP {}", version_response.status_code)
    logger.info("Результат проверки Ollama: {}", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка промпта и токенов Ollama")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, help="Имя созданной в Ollama модели")
    parser.add_argument("--sample", required=True, help="Проверенный JSONL с сообщениями")
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--local-files-only", action="store_true", help="Загружать токенизатор только из кеша"
    )
    args = parser.parse_args()
    run(
        load_config(args.config),
        args.url,
        args.model,
        Path(args.sample),
        local_files_only=args.local_files_only,
    )


if __name__ == "__main__":
    main()
