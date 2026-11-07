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

По умолчанию — на CPU/ОЗУ (HF 7B = ~15 ГБ, поэтому на 8-16 ГБ ОЗУ вылетает).
Для LM Studio 7B весит 5 ГБ потому что там GGUF Q4_K_M — HF всегда тяжелее.
Если ОЗУ мало (вылетает), используй квантование: --load-in-4bit (~5-6 ГБ) или --load-in-8bit (~8-9 ГБ).
Для скорости на большой карте добавь --gpu:
    python scripts/merge_lora.py --adapter data/adapters/xxx --gpu
Требует RAM ~ 1.2x размера модели в выбранном режиме. На 7B HF fp16 — ~16 Гб, 4-bit — ~6 Гб.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from domain.exceptions import TrainingError

LOG_CHANNEL = "finetune.merge"
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
            logger.bind(channel=LOG_CHANNEL).warning(
                "не удалось прочитать adapter_config.json: {}", exc
            )
    # fallback: пробуем training_report.json
    report_path = adapter_dir / "training_report.json"
    if report_path.is_file():
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
            base = data.get("base_model") or data.get("baseModel")
            if isinstance(base, str) and base.strip():
                logger.bind(channel=LOG_CHANNEL).info(
                    "base_model из training_report.json: {}", base
                )
                return base.strip()
        except Exception as exc:  # noqa: BLE001
            logger.bind(channel=LOG_CHANNEL).warning(
                "не удалось прочитать training_report.json: {}", exc
            )
    raise TrainingError(
        "не удалось определить базовую модель. Укажи её явно: --base-model Qwen/Qwen2.5-7B-Instruct\n"
        f"  (проверь {cfg_path} -> base_model_name_or_path)"
    )


def _safe_model_name(base: str) -> str:
    # "qwen/qwen3.5-9b" -> "qwen3.5-9b", "Qwen/Qwen2.5-7B-Instruct" -> "qwen2.5-7b-instruct"
    return base.split("/")[-1].lower().replace("_", "-")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Смержить LoRA адаптер с базой (для LM Studio без Add LoRA)."
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="папка адаптера, как на скрине (с adapter_model.safetensors)",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="HF id базы, напр. Qwen/Qwen2.5-7B-Instruct. Если не указан — берётся из adapter_config.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="куда сохранить merged модель (default: data/merged/<base>-mafia-merged_TIMESTAMP/)",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
        help="тип весов при загрузке",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="грузить на VRAM (device_map=auto). Быстрее в 3-5 раз, но нужно 16+ Гб VRAM для 7B",
    )
    parser.add_argument(
        "--device-map",
        default=None,
        help="переопределить accelerate device_map (напр. 'auto', 'cuda', 'cpu'). По умолчанию cpu, с --gpu -> auto",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="квантовать базу в 4-bit NF4 (~5-6 ГБ ОЗУ вместо 15 ГБ для 7B). Требует bitsandbytes. Для 8-16 ГБ ОЗУ",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="квантовать базу в 8-bit (~8-9 ГБ ОЗУ для 7B). Требует bitsandbytes",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="алиас для --load-in-4bit (совместимость)",
    )
    args = parser.parse_args(argv)

    adapter_dir = Path(args.adapter)
    if not adapter_dir.is_dir():
        logger.bind(channel=LOG_CHANNEL).error("папка адаптера не найдена: {}", adapter_dir)
        return 1
    if (
        not (adapter_dir / "adapter_model.safetensors").exists()
        and not (adapter_dir / "adapter_model.bin").exists()
    ):
        logger.bind(channel=LOG_CHANNEL).error(
            "в {} нет adapter_model.safetensors/.bin — это точно папка адаптера?", adapter_dir
        )
        return 1

    # алиас
    if args.low_memory:
        args.load_in_4bit = True

    if args.load_in_4bit and args.load_in_8bit:
        logger.bind(channel=LOG_CHANNEL).error(
            "нельзя одновременно --load-in-4bit и --load-in-8bit"
        )
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

    logger.bind(channel=LOG_CHANNEL).info(
        "мерж: base={} + adapter={} -> {}", base_model, adapter_dir, output_dir
    )

    # ленивые импорты — чтобы скрипт импортировался без torch
    try:
        import torch  # type: ignore[import-not-found]
        from peft import PeftModel  # type: ignore[import-not-found]
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]
    except ImportError as exc:
        logger.bind(channel=LOG_CHANNEL).error(
            "нужны зависимости для мержа: {}. Установи: pip install transformers peft accelerate torch --upgrade",
            exc,
        )
        return 1

    dtype_map: dict[str, Any] = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.dtype) if args.dtype != "auto" else "auto"

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. грузим базу — на CPU, без квантизации, чтобы мерж был точным
    logger.bind(channel=LOG_CHANNEL).info("загружаю базовую модель {} ...", base_model)
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        logger.bind(channel=LOG_CHANNEL).error(
            "не удалось загрузить токенизатор базы {}: {}", base_model, exc
        )
        return 1

    try:
        model_kwargs: dict[str, Any] = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        if torch_dtype != "auto":
            model_kwargs["torch_dtype"] = torch_dtype

        # квантование для слабых машин (5-6 ГБ вместо 15 ГБ)
        use_quant = args.load_in_4bit or args.load_in_8bit
        if use_quant:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                logger.bind(channel=LOG_CHANNEL).error(
                    "для --load-in-4bit/8bit нужен bitsandbytes: pip install bitsandbytes --upgrade ({})",
                    exc,
                )
                return 1
            if args.load_in_4bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                logger.bind(channel=LOG_CHANNEL).info("режим 4-bit NF4: ~5-6 ГБ ОЗУ для 7B")
            else:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                logger.bind(channel=LOG_CHANNEL).info("режим 8-bit: ~8-9 ГБ ОЗУ для 7B")
            # квантованную модель нужно раскидывать через accelerate
            if "device_map" not in model_kwargs:
                model_kwargs["device_map"] = "auto"

        # по умолчанию — CPU (безопасно), с --gpu — на VRAM
        requested_map: str | None = None
        if args.device_map is not None:
            requested_map = args.device_map
            model_kwargs["device_map"] = args.device_map
        elif args.gpu:
            requested_map = "auto"
            model_kwargs["device_map"] = "auto"
        elif use_quant and "device_map" not in model_kwargs:
            # квантование уже выставило auto выше
            requested_map = str(model_kwargs.get("device_map", "auto"))
        # иначе device_map не ставим — грузим на CPU, потом peft сам разберётся

        # диагностика GPU перед загрузкой
        if requested_map is not None:
            try:
                cuda_ok = torch.cuda.is_available()
                if not cuda_ok:
                    logger.bind(channel=LOG_CHANNEL).warning(
                        "--gpu/--device-map запрошен ({}), но torch.cuda.is_available()==False. "
                        "Проверь: pip show torch должен быть +cuXXX и nvidia-smi видит карту. "
                        "Откатываюсь на CPU — будет в ОЗУ, но смержится.",
                        requested_map,
                    )
                else:
                    props = torch.cuda.get_device_properties(0)
                    free, _total = (
                        torch.cuda.mem_get_info(0)
                        if hasattr(torch.cuda, "mem_get_info")
                        else (0, 0)
                    )
                    logger.bind(channel=LOG_CHANNEL).info(
                        "CUDA {} (VRAM {:.1f} ГБ, свободно ~{:.1f} ГБ), device_map={}",
                        torch.cuda.get_device_name(0),
                        props.total_memory / (1024**3),
                        free / (1024**3) if free else 0.0,
                        requested_map,
                    )
                    # на GPU без явного dtype fp32 сожрёт всю VRAM — форсим fp16
                    if torch_dtype == "auto":
                        model_kwargs["torch_dtype"] = torch.float16
                        logger.bind(channel=LOG_CHANNEL).info(
                            "dtype auto -> float16 для экономии VRAM (переопредели --dtype bf16/fp32 если нужно)"
                        )
            except Exception as exc:  # noqa: BLE001
                logger.bind(channel=LOG_CHANNEL).warning("не удалось проверить CUDA: {}", exc)

        logger.bind(channel=LOG_CHANNEL).info("from_pretrained kwargs: {}", model_kwargs)
        base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.bind(channel=LOG_CHANNEL).error("не удалось загрузить базу {}: {}", base_model, exc)
        logger.bind(channel=LOG_CHANNEL).error(
            "проверь id модели и интернет. Для Qwen3.5 из конфига пример: Qwen/Qwen2.5-7B-Instruct"
        )
        return 1

    # 2. навешиваем LoRA
    logger.bind(channel=LOG_CHANNEL).info("загружаю адаптер {} ...", adapter_dir)
    try:
        model = PeftModel.from_pretrained(base, str(adapter_dir))
    except Exception as exc:  # noqa: BLE001
        logger.bind(channel=LOG_CHANNEL).error("не удалось загрузить адаптер: {}", exc)
        return 1

    # логируем куда реально разложилась база
    try:
        hf_map = getattr(base, "hf_device_map", None) or getattr(model, "hf_device_map", None)
        if hf_map:
            logger.bind(channel=LOG_CHANNEL).info("hf_device_map после загрузки: {}", hf_map)
        elif hasattr(base, "device"):
            logger.bind(channel=LOG_CHANNEL).info(
                "модель на устройстве: {}", getattr(base, "device", "unknown")
            )
    except Exception as exc:  # noqa: BLE001
        logger.bind(channel=LOG_CHANNEL).debug("не удалось получить hf_device_map: {}", exc)

    # 3. мерж
    logger.bind(channel=LOG_CHANNEL).info("мёржу веса (merge_and_unload) ... это займёт 1-5 минут")
    try:
        merged = model.merge_and_unload()
    except Exception as exc:  # noqa: BLE001
        logger.bind(channel=LOG_CHANNEL).error("ошибка мержа: {}", exc)
        if "CUDA out of memory" in str(exc):
            logger.bind(channel=LOG_CHANNEL).error(
                "VRAM кончилась — 7B в fp16 нужно ~15 ГБ. Запусти без --gpu (на ОЗУ) или с квантованием: "
                "--device-map cpu или закрой LM Studio/браузер"
            )
        return 1

    # 4. сохраняем
    logger.bind(channel=LOG_CHANNEL).info("сохраняю в {} ...", output_dir)
    try:
        merged.save_pretrained(str(output_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(output_dir))
        # сохраним чат-шаблон если был в адаптере
        chat_tpl_src = adapter_dir / "chat_template.jinja"
        if chat_tpl_src.is_file():
            (output_dir / "chat_template.jinja").write_text(
                chat_tpl_src.read_text(encoding="utf-8"), encoding="utf-8"
            )
    except Exception as exc:  # noqa: BLE001
        logger.bind(channel=LOG_CHANNEL).error("ошибка сохранения: {}", exc)
        return 1

    print(f"\n✅ Готово: смерженная модель в {output_dir}\n")
    print("── Что дальше для LM Studio ──────────────────────────────────────")
    print("LM Studio 0.3.20+ не показывает Add LoRA для GGUF — поэтому мерж и нужен.")
    print(
        "1. В LM Studio: My Models -> вверху нажми на путь к папке моделей -> Открой в проводнике"
    )
    print(f"   Создай внутри: <publisher>/<repo>/  напр. local/mafia-{model_tag}/")
    print(f"   Скопируй туда ВСЁ содержимое из: {output_dir}")
    print(
        f"   Пример Windows: C:\\Users\\%USERNAME%\\.lmstudio\\models\\local\\mafia-{model_tag}\\"
    )
    print(f"2. Перезапусти LM Studio. Модель появится как 'local/mafia-{model_tag}' -> Load")
    print("3. Запусти Developer -> Local Server (порт 1234). В configs/settings.yaml оставь:")
    print(
        f'   llm.model: "local/mafia-{model_tag}"  или оставь старый id, но выбери эту модель в UI'
    )
    print("   Проверь: curl http://localhost:1234/v1/models")
    print("\n── GGUF (опционально, для экономии VRAM) ───────────────────────────")
    print("Если нужен .gguf вместо HF, конвертируй:")
    print("  git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp")
    print(f"  python convert_hf_to_gguf.py {output_dir} --outfile mafia-{model_tag}.gguf")
    print(
        f"  # затем: lms import mafia-{model_tag}.gguf   или скопируй в .lmstudio/models/local/mafia-{model_tag}/"
    )
    print("\n── Быстрый тест без LM Studio ─────────────────────────────────────")
    print(
        f"  python -c \"from transformers import pipeline; p=pipeline('text-generation', model=r'{output_dir}'); print(p('Привет, кто мафия?', max_new_tokens=100))\""
    )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
