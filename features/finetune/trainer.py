"""LoRA/QLoRA fine-tune runner with Unsloth and PEFT/TRL backends.

Heavy dependencies (torch, unsloth, trl, datasets, peft, bitsandbytes) are
imported lazily inside the functions that need them, so the module can be
imported — and unit-tested — without the training stack installed. When a
dependency is missing, a :class:`~domain.exceptions.TrainingError` with an
install hint is raised instead of an ``ImportError``.

The adapter is saved to ``output_dir`` together with the tokenizer and a
``training_report.json``; the CLI prints the LM Studio loading instructions
returned by :func:`lmstudio_instructions`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict

from domain.exceptions import TrainingError
from features.finetune.models import TrainingOptions

LOG_CHANNEL: str = "finetune.trainer"


class FinetuneReport(BaseModel):
    """Outcome of one fine-tune run, persisted next to the adapter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: str
    method: str
    base_model: str
    output_dir: str
    examples: int
    adapter_files: list[str]
    metrics: dict[str, float] | None = None
    lmstudio_instructions: str = ""


def train_lora(
    *,
    options: TrainingOptions,
    dataset_path: Path,
    output_dir: Path,
    max_examples: int | None = None,
) -> FinetuneReport:
    """Run LoRA/QLoRA training on a messages-format JSONL dataset.

    Args:
        options: resolved hyperparameters.
        dataset_path: JSONL file with ``messages`` records (see
            :class:`~features.finetune.models.DatasetRecord`).
        output_dir: adapter directory (created if missing).
        max_examples: optional cap on the number of training examples.

    Returns:
        A :class:`FinetuneReport` with the adapter location and instructions.

    Raises:
        TrainingError: if the dataset is missing/empty, a dependency is not
            installed, or the training run fails.
    """
    if not dataset_path.is_file():
        raise TrainingError(f"датасет не найден: {dataset_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = _load_dataset(dataset_path, max_examples)

    logger.bind(channel=LOG_CHANNEL).info(
        "training start: backend={} method={} model={} examples={}",
        options.backend,
        options.method,
        options.base_model,
        len(dataset),
    )
    if options.backend == "unsloth":
        report = _train_unsloth(options=options, dataset=dataset, output_dir=output_dir)
    elif options.backend == "peft":
        report = _train_peft(options=options, dataset=dataset, output_dir=output_dir)
    else:
        raise TrainingError(f"unsupported backend: {options.backend}")

    report = report.model_copy(
        update={"lmstudio_instructions": lmstudio_instructions(options, output_dir)}
    )
    _write_report(report, output_dir)
    logger.bind(channel=LOG_CHANNEL).info("training complete: {}", report.output_dir)
    return report


def _load_dataset(dataset_path: Path, max_examples: int | None) -> Any:
    """Read the JSONL dataset into a ``datasets.Dataset`` (lazy import)."""
    records: list[dict[str, Any]] = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if max_examples is not None and len(records) >= max_examples:
                break
    if not records:
        raise TrainingError(f"датасет пуст: {dataset_path}")
    try:
        from datasets import Dataset  # type: ignore[import-not-found]
    except ImportError as error:
        raise TrainingError(
            f"datasets не установлен: {error}.pip install -e '.[finetune]'"
        ) from error
    logger.bind(channel=LOG_CHANNEL).info(
        "dataset loaded: {} examples from {}", len(records), dataset_path
    )
    return Dataset.from_list(records)


def _train_unsloth(*, options: TrainingOptions, dataset: Any, output_dir: Path) -> FinetuneReport:
    """Run QLoRA/LoRA through Unsloth's ``FastLanguageModel`` + TRL SFTTrainer."""
    try:
        from trl import SFTConfig, SFTTrainer  # type: ignore[import-not-found]
        from unsloth import FastLanguageModel  # type: ignore[import-not-found]
    except ImportError as error:
        raise TrainingError(
            f"unsloth/trl не установлены: {error}.pip install -e '.[finetune]'"
        ) from error

    logger.bind(channel=LOG_CHANNEL).info("loading model {} (unsloth)", options.base_model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=options.base_model,
        max_seq_length=options.max_seq_length,
        dtype=None,
        load_in_4bit=options.method == "qlora",
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=options.rank,
        lora_alpha=options.alpha,
        lora_dropout=options.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=options.epochs,
        per_device_train_batch_size=options.batch_size,
        gradient_accumulation_steps=options.gradient_accumulation_steps,
        learning_rate=options.learning_rate,
        warmup_ratio=options.warmup_ratio,
        weight_decay=options.weight_decay,
        logging_steps=10,
        save_strategy="no",
        max_length=options.max_seq_length,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    metrics = trainer.train().metrics
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return _finalize_report("unsloth", options, dataset, output_dir, metrics)


def _train_peft(*, options: TrainingOptions, dataset: Any, output_dir: Path) -> FinetuneReport:
    """Run QLoRA/LoRA through transformers + PEFT + TRL (fallback backend)."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from trl import SFTConfig, SFTTrainer
    except ImportError as error:
        raise TrainingError(
            f"transformers/peft/trl не установлены: {error}.pip install -e '.[finetune]'"
        ) from error

    quantization_config = None
    if options.method == "qlora":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_use_double_quant=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(options.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        options.base_model,
        quantization_config=quantization_config,
        device_map="auto" if quantization_config is not None else None,
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=options.rank,
        lora_alpha=options.alpha,
        lora_dropout=options.dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=options.epochs,
        per_device_train_batch_size=options.batch_size,
        gradient_accumulation_steps=options.gradient_accumulation_steps,
        learning_rate=options.learning_rate,
        warmup_ratio=options.warmup_ratio,
        weight_decay=options.weight_decay,
        logging_steps=10,
        save_strategy="no",
        max_length=options.max_seq_length,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    metrics = trainer.train().metrics
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return _finalize_report("peft", options, dataset, output_dir, metrics)


def _finalize_report(
    backend: str,
    options: TrainingOptions,
    dataset: Any,
    output_dir: Path,
    metrics: dict[str, Any] | None,
) -> FinetuneReport:
    """Build the report after a backend finishes and saved its adapter."""
    adapter_files = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    return FinetuneReport(
        backend=backend,
        method=options.method,
        base_model=options.base_model,
        output_dir=str(output_dir),
        examples=len(dataset),
        adapter_files=adapter_files,
        metrics=_flatten_metrics(metrics),
    )


def _flatten_metrics(metrics: dict[str, Any] | None) -> dict[str, float] | None:
    """Convert training metrics to plain floats, dropping non-numeric values."""
    if not isinstance(metrics, dict):
        return None
    result: dict[str, float] = {}
    for key, value in metrics.items():
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return result or None


def _write_report(report: FinetuneReport, output_dir: Path) -> None:
    """Persist the training report next to the adapter."""
    report_path = output_dir / "training_report.json"
    report_path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def lmstudio_instructions(options: TrainingOptions, output_dir: Path) -> str:
    """Return the LM Studio adapter-loading instructions."""
    return (
        f"Адаптер LoRA сохранён в: {output_dir}\n\n"
        "Загрузка в LM Studio:\n"
        f"1. Откройте LM Studio → вкладка «My Models».\n"
        f"2. Выберите базовую модель ({options.base_model}).\n"
        "3. Добавьте LoRA: кнопка «Add LoRA» рядом с моделью (или меню модели → "
        "LoRA) и укажите папку адаптера.\n"
        "4. Запустите Local Server (Developer → Start Server).\n"
        "5. Проверьте GET /v1/models — модель должна отображаться с "
        "подключённым адаптером.\n\n"
        "Примечание: OpenAI-совместимый API LM Studio не принимает адаптер через "
        "параметры запроса; загрузка выполняется через UI model manager."
    )
