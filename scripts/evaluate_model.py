"""Evaluate a base model and/or a LoRA adapter on the SFT dataset.

Usage::

    python scripts/evaluate_model.py \
        --dataset data/datasets/sft_test.jsonl \
        --base-model qwen/qwen3.5-9b \
        --adapter data/adapters/2026-11-06_15-30

For every record in the dataset the script runs the model on the
``[system, user]`` messages, parses the JSON response, extracts the
``suspected_role`` for the record's ``player_id``, and compares it with
the ground truth. Metrics:

* accuracy of exact role predictions;
* accuracy by faction (town / mafia / neutral);
* average confidence on correct and incorrect predictions.

A comparative table is printed to stdout and the full report is saved as
JSON (default ``data/eval_report.json``).

Requires the optional dependencies: ``pip install -e ".[finetune]"``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from domain.exceptions import ConfigError, TrainingError
from domain.roles import load_roles
from features.finetune.evaluator import (
    EvaluationReport,
    compute_report,
    evaluate_records,
    render_table,
)
from features.finetune.models import DatasetRecord

LOG_CHANNEL: str = "finetune.evaluate"

_DEFAULT_ROLES = "configs/roles.yaml"
_DEFAULT_OUTPUT = "data/eval_report.json"
_DEFAULT_MAX_NEW_TOKENS = 2048


def _build_arg_namespace(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate a base model and/or a LoRA adapter on the SFT dataset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="path to the JSONL dataset (preferably the test split)",
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="base model name (e.g. qwen/qwen3.5-9b)",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="path to a LoRA adapter directory (optional)",
    )
    parser.add_argument(
        "--roles",
        default=_DEFAULT_ROLES,
        help=f"roles config path (default: {_DEFAULT_ROLES})",
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT,
        help=f"JSON report output path (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="cap on examples to evaluate",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=_DEFAULT_MAX_NEW_TOKENS,
        help=f"max tokens generated per example (default: {_DEFAULT_MAX_NEW_TOKENS})",
    )
    return parser.parse_args(argv)


def _load_dataset(path: Path, max_examples: int | None = None) -> list[DatasetRecord]:
    """Read JSONL dataset into :class:`DatasetRecord` objects."""
    if not path.is_file():
        raise TrainingError(f"датасет не найден: {path}")
    records: list[DatasetRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(DatasetRecord.model_validate_json(line))
            if max_examples is not None and len(records) >= max_examples:
                break
    if not records:
        raise TrainingError(f"датасет пуст: {path}")
    return records


def _build_predict_fn(
    base_model: str,
    adapter: str | None,
    max_new_tokens: int,
) -> Callable[[DatasetRecord], str]:
    """Build a predict function with lazy heavy imports.

    Raises :class:`TrainingError` if torch/transformers/peft are not installed.
    """
    try:
        import torch  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    except ImportError as error:
        raise TrainingError(
            f"torch/transformers не установлены: {error}.pip install -e '.[finetune]'"
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)

    if adapter:
        try:
            from peft import PeftModel  # type: ignore[import-not-found]
        except ImportError as error:
            raise TrainingError(
                f"peft не установлен: {error}.pip install -e '.[finetune]'"
            ) from error
        model = PeftModel.from_pretrained(model, adapter)

    model.eval()

    def predict(record: DatasetRecord) -> str:
        messages = [{"role": msg.role, "content": msg.content} for msg in record.messages[:2]]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)  # type: ignore[no-any-return]

    return predict


def _save_report(payload: dict[str, Any], path: Path) -> None:
    """Write the evaluation report as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation CLI. Returns a process exit code."""
    args = _build_arg_namespace(argv)

    try:
        registry = load_roles(args.roles)
    except ConfigError as error:
        logger.bind(channel=LOG_CHANNEL).error("roles: {}", error)
        return 1

    try:
        records = _load_dataset(Path(args.dataset), args.max_examples)
    except TrainingError as error:
        logger.bind(channel=LOG_CHANNEL).error("dataset: {}", error)
        return 1

    logger.bind(channel=LOG_CHANNEL).info(
        "evaluating {} examples (base={} adapter={})",
        len(records),
        args.base_model,
        args.adapter or "none",
    )

    base_predict = _build_predict_fn(args.base_model, None, args.max_new_tokens)
    base_samples = evaluate_records(records, predict_fn=base_predict, registry=registry)
    base_report = compute_report(
        base_samples,
        base_model=args.base_model,
        adapter=None,
        registry=registry,
    )

    adapter_report: EvaluationReport | None = None
    if args.adapter:
        adapter_predict = _build_predict_fn(args.base_model, args.adapter, args.max_new_tokens)
        adapter_samples = evaluate_records(records, predict_fn=adapter_predict, registry=registry)
        adapter_report = compute_report(
            adapter_samples,
            base_model=args.base_model,
            adapter=args.adapter,
            registry=registry,
        )

    print(render_table(base_report, adapter_report))

    report_payload: dict[str, Any] = {
        "dataset": str(args.dataset),
        "generated_at": datetime.now(UTC).isoformat(),
        "base": base_report.model_dump(),
        "adapter": adapter_report.model_dump() if adapter_report else None,
    }
    _save_report(report_payload, Path(args.output))
    logger.bind(channel=LOG_CHANNEL).info("report saved: {}", args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
