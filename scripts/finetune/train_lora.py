"""Обучение PEFT/TRL; тяжёлые зависимости импортируются только при запуске."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

from loguru import logger

from scripts.finetune.common import digest, load_config, read_jsonl, validate_messages
from scripts.finetune.prepare_dataset import check_balance


def validate_dataset(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    directory = Path(config["dataset_dir"])
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    if report.get("errors"):
        raise ValueError("Отчёт датасета содержит ошибки")
    result = {}
    for name in ("train", "val", "test"):
        path = directory / f"{name}.jsonl"
        info = report["splits"][name]
        if digest(path) != info["sha256"]:
            raise ValueError("Хеш датасета не совпадает с отчётом")
        check_balance(info["distribution"], float(config["max_class_fraction"]))
        rows = read_jsonl(path)
        if not rows or len(rows) != info["examples"]:
            raise ValueError("Выборка пуста или не соответствует отчёту")
        for row in rows:
            if set(row) != {"messages"}:
                raise ValueError("Метаданные в обучающей записи запрещены")
            validate_messages(row["messages"])
        result[name] = rows
    return result


def assess(metrics: dict[str, Any], config: dict[str, Any]) -> list[str]:
    reasons = []
    if (
        metrics.get("validation_completed") is not True
        or not isinstance(metrics.get("training_steps"), int)
        or metrics["training_steps"] < 1
    ):
        return ["Не выполнены обучение и валидационный прогон"]
    vals = metrics.get("validation_history", [])
    if len(vals) < 2 or any(not math.isfinite(x) for x in vals):
        return ["Нет достаточной истории validation loss или loss не конечен"]
    train = metrics["train_loss"]
    test = metrics["test_loss"]
    if not math.isfinite(train) or not math.isfinite(test):
        return ["Train или test loss не конечен"]
    if vals[-1] > float(config["max_val_loss"]) or vals[-1] >= vals[0]:
        reasons.append("Риск недообучения: высокий loss или нет улучшения относительно базы")
    if any(b - a > float(config["max_val_loss_increase"]) for a, b in itertools.pairwise(vals)):
        reasons.append("Риск нестабильности/переобучения: validation loss растёт")
    if vals[-1] - train > float(config["max_val_train_gap"]):
        reasons.append("Риск переобучения: большой разрыв train/validation")
    if test > float(config["max_val_loss"]) or test - vals[-1] > float(config["max_val_train_gap"]):
        reasons.append("Риск плохого обобщения: высокий test loss или разрыв test/validation")
    return reasons


def tokenize_example(
    tokenizer: Any, messages: list[dict[str, str]], max_length: int
) -> dict[str, Any]:
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if full[: len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError("Префикс обучения не совпадает с префиксом инференса")
    if len(full) > max_length:
        raise ValueError(
            f"Пример ({len(full)} токенов) превышает max_seq_length; усечение запрещено"
        )
    return {
        "input_ids": full,
        "attention_mask": [1] * len(full),
        "labels": [-100] * len(prefix) + full[len(prefix) :],
    }


def train(config: dict[str, Any]) -> dict[str, Any]:
    output = Path(config["adapter_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "quality.json").unlink(missing_ok=True)
    from scripts.finetune.template import check_template_pair

    check_template_pair(Path(config["chat_template"]), Path(config["ollama_template"]))
    rows = validate_dataset(config)
    if float(config["epochs"]) <= 0:
        raise ValueError("Число эпох должно быть положительным")
    method = config["method"]
    if method not in ("lora", "qlora"):
        raise ValueError("method должен быть lora или qlora")
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        TrainerCallback,
    )
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("Для обучения требуется CUDA GPU")
    template = Path(config["chat_template"]).read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    tokenizer.chat_template = template
    for token in ("<|im_start|>", "<|im_end|>"):
        ids = tokenizer.encode(token, add_special_tokens=False)
        if (
            len(ids) != 1
            or ids[0] == tokenizer.unk_token_id
            or ids[0] != tokenizer.convert_tokens_to_ids(token)
        ):
            raise ValueError(f"Базовый токенизатор не поддерживает {token}")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    quant = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        if method == "qlora"
        else None
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=quant,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    if method == "qlora":
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(config["rank"]),
            lora_alpha=int(config["alpha"]),
            lora_dropout=float(config["dropout"]),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    data = {
        name: Dataset.from_list(
            [
                tokenize_example(tokenizer, row["messages"], int(config["max_seq_length"]))
                for row in examples
            ]
        )
        for name, examples in rows.items()
    }
    history: list[float] = []

    class ValidationLogger(TrainerCallback):
        def on_evaluate(
            self, args: Any, state: Any, control: Any, metrics: Any = None, **kwargs: Any
        ) -> None:
            if metrics and "eval_loss" in metrics:
                history.append(float(metrics["eval_loss"]))
                logger.info("Validation loss: {:.6f}", history[-1])

    args = SFTConfig(
        output_dir=str(output / "checkpoints"),
        num_train_epochs=float(config["epochs"]),
        per_device_train_batch_size=int(config["batch_size"]),
        per_device_eval_batch_size=int(config["batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate"]),
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=1,
        report_to="none",
        fp16=True,
        seed=int(config["seed"]),
        max_length=int(config["max_seq_length"]),
        dataset_kwargs={"skip_prepare_dataset": True},
        disable_tqdm=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
        callbacks=[ValidationLogger()],
    )
    baseline = trainer.evaluate()  # baseline до обновления адаптера
    if "eval_loss" not in baseline:
        raise RuntimeError("Нет baseline validation loss")
    trainer.train()
    steps = int(trainer.state.global_step)
    if steps < 1:
        raise RuntimeError("Обучение не выполнило ни одного шага")
    final_validation = trainer.evaluate()
    if "eval_loss" not in final_validation:
        raise RuntimeError("Нет итогового validation loss")
    val_history = history.copy()
    if (
        not val_history
        or val_history[0] != float(baseline["eval_loss"])
        or val_history[-1] != float(final_validation["eval_loss"])
    ):
        raise RuntimeError("Неполная история валидации")
    train_loss = float(
        trainer.evaluate(eval_dataset=data["train"], metric_key_prefix="train_eval")[
            "train_eval_loss"
        ]
    )
    test_loss = float(
        trainer.evaluate(eval_dataset=data["test"], metric_key_prefix="test")["test_loss"]
    )
    metrics: dict[str, Any] = {
        "validation_history": val_history,
        "validation_completed": True,
        "training_steps": steps,
        "train_loss": train_loss,
        "test_loss": test_loss,
    }
    risks = assess(metrics, config)
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    (output / "chat_template.jinja").write_text(template, encoding="utf-8", newline="\n")
    result = {
        "metrics": metrics,
        "risks": risks,
        "ready_for_export": not risks,
        "ready_for_use": False,
        "status": "experimental",
        "base_model": config["base_model"],
        "template_sha256": digest(output / "chat_template.jinja"),
        "dataset_report_sha256": digest(Path(config["dataset_dir"]) / "report.json"),
    }
    (output / "quality.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if risks:
        logger.error("Адаптер не готов к экспорту: {}", "; ".join(risks))
        raise RuntimeError("Проверка качества не пройдена")
    logger.warning("Экспериментальный адаптер прошёл loss-скрининг, но не готов к использованию")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение LoRA/QLoRA")
    parser.add_argument("--config", default="configs/finetune.example.yaml")
    train(load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
