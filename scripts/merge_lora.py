"""Смержить PEFT LoRA адаптер с базовой моделью для LM Studio.

LM Studio 0.3.20+ убрал/спрятал кнопку "Add LoRA" для GGUF-моделей.
Если ты не видишь её — этот скрипт делает то, что раньше делала кнопка:
вживляет веса адаптера в базу и сохраняет готовую HF-папку, которую LM Studio
уже умеет импортировать как обычную модель.

Использование (Windows / PowerShell)::

    # 1. Поставь зависимости для мержа (один раз)
    pip install transformers peft accelerate torch --upgrade

    # 2. Смержить (адаптер из скрина)
    python scripts/merge_lora.py --adapter data/adapters/2026-11-07_04-19

    # Базовая модель возьмётся из adapter_config.json, но можно переопределить:
    python scripts/merge_lora.py --adapter data/adapters/2026-11-07_04-19 --base-model Qwen/Qwen2.5-7B-Instruct

    # 3. Результат: data/merged/qwen2.5-7b-mafia-merged/
    #    Импортируй его в LM Studio (см. вывод скрипта)

Требует RAM ~ 1.2x размера модели. На 7B — около 16 Гб RAM/VRAM.
Мерж идёт на CPU, если нет CUDA — будет медленно, но сработает.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

try:
    from loguru import logger  # type: ignore[import-not-found]
except ImportError:  # fallback когда loguru не установлен
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("finetune.merge")  # type: ignore[assignment]

    class _BindStub:
        def bind(self, **_: object) -> logging.Logger:
            return logger  # type: ignore[return-value]

    logger.bind = _BindStub().bind  # type: ignore[attr-defined]

LOG_CHANNEL = "finetune.merge"


class TrainingError(RuntimeError):
    pass


try:
    from domain.exceptions import TrainingError as _DomainTrainingError  # type: ignore[import-not-found]

    TrainingError = _DomainTrainingError  # type: ignore[no-redef]
except ImportError:
    pass
_DEFAULT_OUTPUT_BASE = "data/merged"


def _resolve_base_model(adapter_dir: Path, cli_base: str | None) -> str:
    if cli_base:
        return cli_base
    cfg_path = adapter_dir / "adapter_config.json"
    if cfg_path.is_file():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            base = data.get("base_model_name_or_path")
            if isinstance(base, str) and base.strip():
                logger.bind(channel=LOG_CHANNEL).info("base_model из adapter_config.json: {}", base)
                return base.strip()
        except Exception as exc:  # noqa: BLE001
            logger.bind(channel=LOG_CHANNEL).warning("не удалось прочитать adapter_config.json: {}", exc)
    # fallback: пробуем training_report.json
    report_path = adapter_dir / "training_report.json"
    if report_path.is_file():
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
            base = data.get("base_model") or data.get("baseModel")
            if isinstance(base, str) and base.strip():
                logger.bind(channel=LOG_CHANNEL).info("base_model из training_report.json: {}", base)
                return base.strip()
        except Exception:  # noqa: BLE001
            pass
    raise TrainingError(
        "не удалось определить базовую модель. Укажи её явно: --base-model Qwen/Qwen2.5-7B-Instruct\n"
        f"  (проверь {cfg_path} -> base_model_name_or_path)"
    )


def _safe_model_name(base: str) -> str:
    # "qwen/qwen3.5-9b" -> "qwen3.5-9b", "Qwen/Qwen2.5-7B-Instruct" -> "qwen2.5-7b-instruct"
    return base.split("/")[-1].lower().replace("_", "-")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Смержить LoRA адаптер с базой (для LM Studio без Add LoRA).")
    parser.add_argument("--adapter", required=True, help="папка адаптера, как на скрине (с adapter_model.safetensors)")
    parser.add_argument("--base-model", default=None, help="HF id базы, напр. Qwen/Qwen2.5-7B-Instruct. Если не указан — берётся из adapter_config.json")
    parser.add_argument("--output", default=None, help="куда сохранить merged модель (default: data/merged/<base>-mafia-merged_TIMESTAMP/)")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto", help="тип весов при загрузке")
    args = parser.parse_args(argv)

    adapter_dir = Path(args.adapter)
    if not adapter_dir.is_dir():
        logger.bind(channel=LOG_CHANNEL).error("папка адаптера не найдена: {}", adapter_dir)
        return 1
    if not (adapter_dir / "adapter_model.safetensors").exists() and not (adapter_dir / "adapter_model.bin").exists():
        logger.bind(channel=LOG_CHANNEL).error("в {} нет adapter_model.safetensors/.bin — это точно папка адаптера?", adapter_dir)
        return 1

    try:
        base_model = _resolve_base_model(adapter_dir, args.base_model)
    except TrainingError as exc:
        logger.bind(channel=LOG_CHANNEL).error(str(exc))
        return 1

    model_tag = _safe_model_name(base_model)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(_DEFAULT_OUTPUT_BASE) / f"{model_tag}-mafia-merged_{timestamp}"

    logger.bind(channel=LOG_CHANNEL).info("мерж: base={} + adapter={} -> {}", base_model, adapter_dir, output_dir)

    # ленивые импорты — чтобы скрипт импортировался без torch
    try:
        import torch  # type: ignore[import-not-found]
        from peft import PeftModel  # type: ignore[import-not-found]
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]
    except ImportError as exc:
        logger.bind(channel=LOG_CHANNEL).error(
            "нужны зависимости для мержа: {}. Установи: pip install transformers peft accelerate torch --upgrade", exc
        )
        return 1

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(args.dtype) if args.dtype != "auto" else "auto"

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. грузим базу — на CPU, без квантизации, чтобы мерж был точным
    logger.bind(channel=LOG_CHANNEL).info("загружаю базовую модель {} ...", base_model)
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        logger.bind(channel=LOG_CHANNEL).error("не удалось загрузить токенизатор базы {}: {}", base_model, exc)
        return 1

    try:
        model_kwargs: dict = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        if torch_dtype != "auto":
            model_kwargs["torch_dtype"] = torch_dtype
        else:
            # auto — пусть transformers выберет
            pass
        # device_map не ставим — грузим на CPU, потом peft сам разберётся
        base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.bind(channel=LOG_CHANNEL).error("не удалось загрузить базу {}: {}", base_model, exc)
        logger.bind(channel=LOG_CHANNEL).error("проверь id модели и интернет. Для Qwen3.5 из конфига пример: Qwen/Qwen2.5-7B-Instruct")
        return 1

    # 2. навешиваем LoRA
    logger.bind(channel=LOG_CHANNEL).info("загружаю адаптер {} ...", adapter_dir)
    try:
        model = PeftModel.from_pretrained(base, str(adapter_dir))
    except Exception as exc:  # noqa: BLE001
        logger.bind(channel=LOG_CHANNEL).error("не удалось загрузить адаптер: {}", exc)
        return 1

    # 3. мерж
    logger.bind(channel=LOG_CHANNEL).info("мёржу веса (merge_and_unload) ... это займёт 1-5 минут")
    try:
        merged = model.merge_and_unload()
    except Exception as exc:  # noqa: BLE001
        logger.bind(channel=LOG_CHANNEL).error("ошибка мержа: {}", exc)
        return 1

    # 4. сохраняем
    logger.bind(channel=LOG_CHANNEL).info("сохраняю в {} ...", output_dir)
    try:
        merged.save_pretrained(str(output_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(output_dir))
        # сохраним чат-шаблон если был в адаптере
        chat_tpl_src = adapter_dir / "chat_template.jinja"
        if chat_tpl_src.is_file():
            (output_dir / "chat_template.jinja").write_text(chat_tpl_src.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.bind(channel=LOG_CHANNEL).error("ошибка сохранения: {}", exc)
        return 1

    print(f"\n✅ Готово: смерженная модель в {output_dir}\n")
    print("── Что дальше для LM Studio ──────────────────────────────────────")
    print("LM Studio 0.3.20+ не показывает Add LoRA для GGUF — поэтому мерж и нужен.")
    print(f"1. В LM Studio: My Models -> вверху нажми на путь к папке моделей -> Открой в проводнике")
    print(f"   Создай внутри: <publisher>/<repo>/  напр. local/mafia-{model_tag}/")
    print(f"   Скопируй туда ВСЁ содержимое из: {output_dir}")
    print(f"   Пример Windows: C:\\Users\\%USERNAME%\\.lmstudio\\models\\local\\mafia-{model_tag}\\")
    print(f"2. Перезапусти LM Studio. Модель появится как 'local/mafia-{model_tag}' -> Load")
    print(f"3. Запусти Developer -> Local Server (порт 1234). В configs/settings.yaml оставь:")
    print(f"   llm.model: \"local/mafia-{model_tag}\"  или оставь старый id, но выбери эту модель в UI")
    print(f"   Проверь: curl http://localhost:1234/v1/models")
    print(f"\n── GGUF (опционально, для экономии VRAM) ───────────────────────────")
    print(f"Если нужен .gguf вместо HF, конвертируй:")
    print(f"  git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp")
    print(f"  python convert_hf_to_gguf.py {output_dir} --outfile mafia-{model_tag}.gguf")
    print(f"  # затем: lms import mafia-{model_tag}.gguf   или скопируй в .lmstudio/models/local/mafia-{model_tag}/")
    print(f"\n── Быстрый тест без LM Studio ─────────────────────────────────────")
    print(f"  python -c \"from transformers import pipeline; p=pipeline('text-generation', model=r'{output_dir}'); print(p('Привет, кто мафия?', max_new_tokens=100))\"")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
