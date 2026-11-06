"""Fine-tune the local LLM with LoRA/QLoRA on the prepared SFT dataset.

Run ``scripts/prepare_finetune_dataset.py`` first, then::

    python scripts/finetune_lora.py

Hyperparameters come from the ``finetune:`` section of
``configs/settings.yaml`` (see :class:`app.config.FinetuneSection`); every
value can be overridden on the command line. The adapter is saved to
``data/adapters/YYYY-MM-DD_HH-MM/`` and the script prints LM Studio loading
instructions when training finishes.

Requires the optional dependencies: ``pip install -e ".[finetune]"``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from loguru import logger

from app.config import AppConfig, load_config
from domain.exceptions import TrainingError
from features.finetune.models import TrainingOptions
from features.finetune.trainer import train_lora

LOG_CHANNEL: str = "finetune.cli"

_DEFAULT_CONFIG = "configs/settings.yaml"


def _timestamped_dir(base: str) -> Path:
    """Return ``base/YYYY-MM-DD_HH-MM`` for the current moment."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    return Path(base) / stamp


def _resolve_options(args: argparse.Namespace, config: AppConfig) -> TrainingOptions:
    """Build training options from config values plus CLI overrides."""
    finetune = config.finetune
    return TrainingOptions(
        base_model=args.base_model or finetune.base_model,
        method=args.method or finetune.method,
        backend=args.backend or finetune.backend,
        rank=finetune.rank,
        alpha=finetune.alpha,
        dropout=finetune.dropout,
        learning_rate=finetune.learning_rate,
        epochs=args.epochs or finetune.epochs,
        batch_size=finetune.batch_size,
        gradient_accumulation_steps=finetune.gradient_accumulation_steps,
        max_seq_length=finetune.max_seq_length,
        warmup_ratio=finetune.warmup_ratio,
        weight_decay=finetune.weight_decay,
    )


def _build_arg_namespace(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments into a namespace (separate for testability)."""
    parser = argparse.ArgumentParser(
        description="Fine-tune the local LLM (LoRA/QLoRA) on the SFT dataset.",
    )
    parser.add_argument(
        "--config",
        default=_DEFAULT_CONFIG,
        help=f"settings YAML (default: {_DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="override finetune.dataset_path",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="override finetune.output_dir (a timestamped subdir is created)",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="override finetune.base_model",
    )
    parser.add_argument(
        "--method",
        choices=["lora", "qlora"],
        default=None,
        help="override finetune.method",
    )
    parser.add_argument(
        "--backend",
        choices=["unsloth", "peft"],
        default=None,
        help="override finetune.backend",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override finetune.epochs",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="cap the number of training examples",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, validate the dataset, and run LoRA/QLoRA training."""
    args = _build_arg_namespace(argv)

    config = load_config(args.config)
    finetune = config.finetune
    dataset = Path(args.dataset or finetune.dataset_path)
    base_output = args.output_dir or finetune.output_dir
    output_dir = _timestamped_dir(base_output)

    if not dataset.is_file():
        logger.bind(channel=LOG_CHANNEL).error(
            "dataset not found: {} — run scripts/prepare_finetune_dataset.py first",
            dataset,
        )
        return 1

    options = _resolve_options(args, config)
    try:
        report = train_lora(
            options=options,
            dataset_path=dataset,
            output_dir=output_dir,
            max_examples=args.max_examples or finetune.max_examples,
        )
    except TrainingError as error:
        logger.bind(channel=LOG_CHANNEL).error("training failed: {}", error)
        return 1

    print(
        f"Готово: адаптер сохранён в {report.output_dir} "
        f"(backend={report.backend}, method={report.method}, "
        f"examples={report.examples})",
        flush=True,
    )
    if report.metrics:
        print(f"Metrics: {report.metrics}", flush=True)
    print(report.lmstudio_instructions, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
