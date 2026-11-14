"""Слияние адаптера, конвертация GGUF и Modelfile без смены шаблона."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from scripts.finetune.common import digest, load_config


def export(config: dict[str, Any]) -> None:
    adapter = Path(config["adapter_dir"])
    out = Path(config["export_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "Modelfile").unlink(missing_ok=True)
    quality = json.loads((adapter / "quality.json").read_text(encoding="utf-8"))
    if (
        not quality.get("ready_for_export")
        or quality.get("risks")
        or quality.get("status") != "experimental"
        or quality.get("ready_for_use") is not False
        or quality.get("metrics", {}).get("validation_completed") is not True
    ):
        raise ValueError("Адаптер не прошёл проверку loss")
    from scripts.finetune.train_lora import assess

    if assess(quality["metrics"], config):
        raise ValueError("Метрики валидации отсутствуют или не прошли проверку loss")
    if (
        quality["base_model"] != config["base_model"]
        or digest(adapter / "chat_template.jinja") != quality["template_sha256"]
    ):
        raise ValueError("База или шаблон отличаются от обучения")
    if digest(Path(config["chat_template"])) != quality["template_sha256"]:
        raise ValueError("Шаблон инференса отличается от шаблона обучения")
    # Оба формата шаблона хранятся и проверяются как парная реализация протокола.
    from scripts.finetune.template import check_template_pair

    check_template_pair(Path(config["chat_template"]), Path(config["ollama_template"]))
    converter = Path(config["llama_cpp_dir"]) / "convert_hf_to_gguf.py"
    if not converter.is_file():
        raise FileNotFoundError(f"Не найден конвертер llama.cpp: {converter}")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged = out / "merged"
    gguf = out / "mafia-f16.gguf"
    gguf.unlink(missing_ok=True)
    logger.info("Слияние с базой {} на CPU; требуется достаточно ОЗУ", config["base_model"])
    base = AutoModelForCausalLM.from_pretrained(
        config["base_model"], torch_dtype=torch.float16, device_map="cpu"
    )
    model = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    model.save_pretrained(str(merged), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter))
    tokenizer.chat_template = (adapter / "chat_template.jinja").read_text(encoding="utf-8")
    tokenizer.save_pretrained(str(merged))
    (merged / "chat_template.jinja").write_text(
        tokenizer.chat_template, encoding="utf-8", newline="\n"
    )
    subprocess.run(
        [sys.executable, str(converter), str(merged), "--outfile", str(gguf), "--outtype", "f16"],
        check=True,
    )
    if not gguf.is_file() or gguf.stat().st_size == 0:
        raise RuntimeError("Конвертер не создал GGUF")
    go_template = Path(config["ollama_template"]).read_text(encoding="utf-8").rstrip("\n")
    (out / "Modelfile").write_text(
        f'FROM ./{gguf.name}\nTEMPLATE """{go_template}"""\nPARAMETER stop "<|im_end|>"\nPARAMETER stop "<|im_start|>"\n',
        encoding="utf-8",
    )
    logger.info("Созданы GGUF и Modelfile в {}; проверьте шаблон в LM Studio", out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Экспорт адаптера в GGUF и Ollama")
    parser.add_argument("--config", default="configs/finetune.example.yaml")
    export(load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
