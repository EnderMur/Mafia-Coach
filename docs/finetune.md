# Дообучение модели (fine-tune) — Mafia Coach

> Оффлайн-пайплайн: выгрузка датасета из SQLite → LoRA/QLoRA дообучение
> локальной модели → загрузка адаптера в LM Studio. **Не влияет на игру**:
> запускается отдельно, после завершения партий.

## Содержание

1. [Обзор](#1-обзор)
2. [Шаг 1: подготовка датасета](#2-шаг-1-подготовка-датасета)
3. [Шаг 2: дообучение (LoRA/QLoRA)](#3-шаг-2-дообучение-loraqlora)
4. [Шаг 3: оценка модели](#4-шаг-3-оценка-модели)
5. [Загрузка адаптера в LM Studio](#5-загрузка-адаптера-в-lm-studio)
6. [Конфигурация](#6-конфигурация)
7. [Анонимизация](#7-анонимизация)
8. [Архитектура и слои](#8-архитектура-и-слои)
9. [Ограничения и известные проблемы](#9-ограничения-и-известные-проблемы)

---

## 1. Обзор

MVP подготавливает данные для дообучения: парсер сохраняет события и
предсказания, по завершении партии `predictions.correct` заполняется сравнением
с `final_role` (`game_players`). Fine-tune-пайплайн использует эти данные:

```text
SQLite (events, predictions, predictions_history, game_players)
        │
        ▼
scripts/prepare_finetune_dataset.py    →  data/datasets/sft_train.jsonl
        │                                  (с --split: sft_train/val/test.jsonl)
        ▼
scripts/finetune_lora.py               →  data/adapters/YYYY-MM-DD_HH-MM/
        │                                  (adapter + tokenizer + training_report.json)
        ▼
scripts/evaluate_model.py              →  data/eval_report.json
        │                                  (base vs adapter: accuracy, faction, confidence)
        ▼
LM Studio (My Models → Add LoRA)      →  модель с адаптером на Local Server
```

Весь пайплайн — оффлайн. Игра (Telegram userbot, analyzer, UI) ничего не
видит и не зависит от него. Тяжёлые зависимости (torch, unsloth, trl, peft,
bitsandbytes, pyarrow) живут в optional-группе `finetune` и не грузятся
при нормальном запуске:

```bash
pip install -e ".[finetune]"   # только для дообучения
```

## 2. Шаг 1: подготовка датасета

```bash
python scripts/prepare_finetune_dataset.py \
    --db data/db/mafia.db \
    --out data/datasets/sft_train.jsonl
```

### Что делает

Для каждой предсказательной строки (таблица `predictions` по умолчанию,
`predictions_history` с `--source history`):

1. **Ground truth** — `game_players.final_role` по `(game_id, player_id)`.
   Без неё пример отбрасывается (`no_ground_truth`).
2. **Корректность** — `predictions.correct`, если заполнен; иначе сравнение
   `predicted_role` с `final_role`. С `--only-correct` оставляем только
   верные предсказания.
3. **Реконструкция состояния партии** — события с `ts <= prediction.ts`
   применяются к чистому `GameState` через `features/finetune/game_reconstructor.py`
   (толерантно: сломанные события пропускаются с WARNING, не ломают пример).
4. **Контекст в момент предсказания** — тот же `build_context` /
   `to_llm_messages`, который использует живой analyzer (системный промпт из
   `system.jinja`, пользователь — контекст партии, предыдущие предсказания
   как якорь).
5. **Assistant-ответ** — корректный JSON:
   - предсказание **правильное** → ответ модели (`suspected_role`,
     `confidence`, `reason`, `reasoning`, паттерны);
   - предсказание **неправильное** → истинная роль (`final_role`),
     фиксированная уверенность, причина исправления, `reasoning: null`,
     паттерны сбрасываются.

Итоговая строка JSONL:

```json
{
  "format_version": "1.0",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "{\"analysis\": [...], \"game_read\": \"...\", \"reasoning\": \"...\"}"}
  ],
  "game_id": 12,
  "prediction_id": 34,
  "player_id": 101,
  "ts": "2026-11-01T12:00:00Z",
  "source": "current",
  "correct": true,
  "ground_truth_role": "Мафия",
  "predicted_role": "Мафия",
  "confidence": 0.8
}
```

`messages` читает TRL/Unsloth (и Axolotl/LLaMA-Factory); остальные поля —
метаданные для отладки.

### CLI

| Параметр | Назначение |
|---|---|
| `--db` | путь к `mafia.db` (обязательно) |
| `--out` | выходной `.jsonl` (обязательно) |
| `--source` | `current` (по умолчанию) или `history` |
| `--only-correct` | оставить только предсказания с `correct = true` |
| `--min-confidence` | отбросить предсказания ниже порога (по умолчанию 0.0) |
| `--anonymize` | заменить структурированные id/имена на `player_N` (см. §6) |
| `--parquet` | также записать `.parquet` (нужен `pyarrow`) |
| `--roles` | `configs/roles.yaml` (по умолчанию) |
| `--club-patterns` | `configs/club_patterns.yaml` (по умолчанию) |
| `--max-messages` | размер окна сообщений (по умолчанию 200, как в живом analyzer) |
| `--split` | разбить датасет на train/val/test по партиям |
| `--test-ratio` | доля партий в test (по умолчанию 0.15) |
| `--val-ratio` | доля партий в validation (по умолчанию 0.10) |

### Split: train / val / test

`--split` разбивает датасет по `game_id`: все примеры одной партии
попадают строго в один сплит (examples одной партии никогда не смешиваются
между train и test). Разбиение детерминированно при фиксированном seed.

```bash
python scripts/prepare_finetune_dataset.py \
    --db data/db/mafia.db \
    --out data/datasets/sft \
    --split \
    --test-ratio 0.15 \
    --val-ratio 0.10
```

Записываются три файла:

```
data/datasets/sft_train.jsonl
data/datasets/sft_val.jsonl
data/datasets/sft_test.jsonl
```

ratio — это доля **партий**, а не примеров: `int(N_games * test_ratio)`
партий попадает в test, следующие `int(N_games * val_ratio)` — в val, остальные — в train.

С `--parquet` также записываются три `.parquet`-файла (`sft_train.parquet`,
`sft_val.parquet`, `sft_test.parquet`).

Считатель пропускаемых примеров (`no_role_predicted`, `no_ground_truth`,
`unknown_ground_truth_role`, `incorrect`, `low_confidence`,
`duplicate_history`, `player_missing`) выводится в лог и в итоговую строку:

```
Prepared 42 examples → data/datasets/sft_train.jsonl, data/datasets/sft_val.jsonl, data/datasets/sft_test.jsonl (skipped: no_ground_truth=17)
```

### Почему отдельный скрипт от `export_dataset.py`

`scripts/export_dataset.py` — сырая выгрузка для анализа, не готовый датасет.
`prepare_finetune_dataset.py` — полноценный SFT-пайплайн: reconstruction +
ground truth + assistant-ответ. Первый скрипт не трогаем.

## 3. Шаг 2: дообучение (LoRA/QLoRA)

```bash
python scripts/finetune_lora.py
```

Читает `finetune:` из `configs/settings.yaml`, CLI-параметры перекрывают
конфиг:

```bash
python scripts/finetune_lora.py \
    --dataset data/datasets/sft_train.jsonl \
    --base-model qwen/qwen3.5-9b \
    --method qlora \
    --backend unsloth \
    --epochs 3
```

### Backend

| Backend | Что делает | Когда |
|---|---|---|
| `unsloth` (по умолчанию) | `FastLanguageModel.from_pretrained` (4-bit при `qlora`) + `get_peft_model` + TRL `SFTTrainer` | быстрее и стабильнее на supported-моделях |
| `peft` (fallback) | `transformers.AutoModelForCausalLM` (+ `BitsAndBytesConfig` nf4/double-quant при `qlora`) + PEFT `get_peft_model` + TRL `SFTTrainer` | если модель не поддерживается Unsloth |

Адаптер сохраняется в `data/adapters/YYYY-MM-DD_HH-MM/` (timestamped поддиректория,
несколько прогонов не перезаписывают друг друга):

```
data/adapters/2026-11-06_15-30/
├── adapter_model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── training_report.json      # backend, method, examples, metrics
└── (файлы адаптера по выводу backend)
```

### Требования

- `pip install -e ".[finetune]"` (torch, unsloth, trl, datasets, peft,
  accelerate, bitsandbytes, pyarrow).
- Датасет существует и не пуст (`датасет не найден` / `датасет пуст` —
  `TrainingError` с подсказкой).
- При недостающей зависимости — `TrainingError` со строкой
  `pip install -e ".[finetune]"`, а не `ImportError`.
- Обучение **не запускается из игрового процесса**; CLI завершается с
  кодом 1 при ошибке.

## 4. Шаг 3: оценка модели

После дообучения можно сравнить базовую модель и модель с адаптером на
test-сплите датасета.

```bash
python scripts/evaluate_model.py \
    --dataset data/datasets/sft_test.jsonl \
    --base-model qwen/qwen3.5-9b \
    --adapter data/adapters/2026-11-06_15-30
```

### Что делает

Для каждого примера в датасете:

1. Запускает модель (базовую или с адаптером) на `messages[:2]`
   (system + user, без assistant-ответа).
2. Парсит JSON-ответ модели, находит предсказание для `player_id`
   примера.
3. Сравнивает `suspected_role` с `ground_truth_role`:
   - **accuracy (exact)** — доля точных совпадений ролей;
   - **accuracy by faction** — доля совпадений фракций
     (town / mafia / neutral);
   - **avg confidence (correct/wrong)** — средняя уверенность на
     верных и неверных предсказаниях.

### CLI

| Параметр | Назначение |
|---|---|
| `--dataset` | путь к JSONL датасету (рекомендуется test-сплит) |
| `--base-model` | имя базовой модели (обязательно) |
| `--adapter` | путь к папке LoRA-адаптеров (необязательно) |
| `--roles` | `configs/roles.yaml` (по умолчанию) |
| `--output` | путь к JSON-отчёту (по умолчанию `data/eval_report.json`) |
| `--max-examples` | лимит примеров для оценки (по умолчанию нет) |
| `--max-new-tokens` | лимит токенов на генерацию (по умолчанию 2048) |

### Пример вывода

```
Metric                          Base       Adapter
-----------------------------------------------------
Examples                        42         42
Accuracy (exact)                52.4%      61.9%
Accuracy by faction:
  town                          60.0%      65.0%
  mafia                         50.0%      55.0%
  neutral                       40.0%      50.0%
Avg confidence (correct)        0.812      0.845
Avg confidence (wrong)          0.345      0.298
```

Полный отчёт сохраняется в JSON (`data/eval_report.json`):

```json
{
  "dataset": "data/datasets/sft_test.jsonl",
  "generated_at": "2026-11-06T15:30:00Z",
  "base": {
    "base_model": "qwen/qwen3.5-9b",
    "adapter": null,
    "examples": 42,
    "accuracy_exact": 0.524,
    "accuracy_by_faction": {"town": 0.6, "mafia": 0.5, "neutral": 0.4},
    "avg_confidence_correct": 0.812,
    "avg_confidence_incorrect": 0.345
  },
  "adapter": {
    "base_model": "qwen/qwen3.5-9b",
    "adapter": "data/adapters/2026-11-06_15-30",
    "examples": 42,
    "accuracy_exact": 0.619,
    "accuracy_by_faction": {"town": 0.65, "mafia": 0.55, "neutral": 0.5},
    "avg_confidence_correct": 0.845,
    "avg_confidence_incorrect": 0.298
  }
}
```

### Требования

- `pip install -e ".[finetune]"` (torch, transformers, peft — все в
  optional-группе `finetune`).
- Датасет существует и не пуст.
- При недостающей зависимости — `TrainingError` со строкой
  `pip install -e ".[finetune]"`.

## 5. Загрузка адаптера в LM Studio

После успешного прогона CLI выводит инструкцию:

1. LM Studio → «My Models» → выбрать базовую модель (`finetune.base_model`,
   обычно та же, что в `llm.model`).
2. «Add LoRA» рядом с моделью (или меню модели → LoRA) → папка адаптера.
3. «Start Server» (Developer) → `GET /v1/models` показывает модель с адаптером.

> OpenAI-совместимый API LM Studio не принимает адаптер через параметры
> запроса — загрузка только через UI model manager.

## 6. Конфигурация

`configs/settings.example.yaml`:

```yaml
finetune:
  enabled: false                    # только для документации; скрипт запускается явно
  base_model: "qwen/qwen3.5-9b"     # та же модель, что в llm.model
  method: "qlora"                   # lora | qlora (4-bit)
  backend: "unsloth"                # unsloth | peft
  rank: 16                          # размерность LoRA
  alpha: 32                         # alpha / rank = 2
  dropout: 0.05
  learning_rate: 2.0e-4
  epochs: 3
  batch_size: 2
  gradient_accumulation_steps: 8
  max_seq_length: 4096
  warmup_ratio: 0.03
  weight_decay: 0.0
  output_dir: "data/adapters"       # data/adapters/YYYY-MM-DD_HH-MM/
  dataset_path: "data/datasets/sft_train.jsonl"
  max_examples: null                # null = без лимита; N = первые N примеров
```

`app/config.py::FinetuneSection` валидирует типы/границы (`rank` 1–256,
`epochs` 1–100 и т.д.); старый конфиг без секции `finetune` грузится
без изменений.

## 7. Анонимизация

`--anonymize` заменяет структурированные данные на `player_N`:

- **Что заменяется** — player id в metadata и assistant JSON, display names в
  контексте (список игроков, авторов сообщений, голоса, смерти, раскрытия),
  имена в `game_read`.
- **Что не трогается** — тело сообщений, `reasoning`, свободный текст.
  Перед распространением датасета проверяйте его на PII (`scripts/audit_pii.py`).
- **Стабильность** — алиасы собираются по всей БД до рендеринга: один
  игрок → один алиас внутри экспорта. Имена `user_<id>` (fallback бота при
  отсутствии username) treated как ссылка на реальный id.
- Неоднозначные имена (одно и то же display name у разных игроков)
  получают разные алиасы — люди не сливаются.

## 8. Архитектура и слои

```
features/finetune/
├── models.py               # DatasetRecord, Message, PredictionRow, TrainingOptions
├── game_reconstructor.py   # толерантная реконструкция GameState из событий
├── dataset_builder.py      # build_example: контекст → [system, user, assistant]
├── anonymizer.py           # PlayerAliasMap, anonymize_context, anonymize_payload
├── trainer.py              # train_lora (unsloth/peft), lmstudio_instructions
└── evaluator.py            # evaluate_records, compute_report, render_table (pure logic)

scripts/
├── prepare_finetune_dataset.py   # CLI: БД → data/datasets/*.jsonl
├── finetune_lora.py              # CLI: конфиг + датасет → data/adapters/YYYY-MM-DD_HH-MM/
└── evaluate_model.py             # CLI: датасет + модель/адаптер → data/eval_report.json

app/config.py               # FinetuneSection + поле AppConfig.finetune
domain/exceptions.py        # TrainingError (общее доменное исключение)
```

- `features/finetune/` зависит от `domain/` и `features/analyzer/`; не
  импортирует `app/` (trainer не знает о конфиге — `TrainingOptions`
  собирается в CLI).
- Тяжёлые импорты (torch, unsloth, …) — ленивые внутри функций; модуль
  можно импортировать и тестировать без训练-стека.
- Тесты: `tests/unit/features/finetune/`, `tests/unit/scripts/test_prepare_finetune_dataset.py`,
  `tests/unit/scripts/test_finetune_lora.py`, `tests/unit/scripts/test_evaluate_model.py`,
  `tests/unit/app/test_config_finetune.py`.

## 9. Ограничения и известные проблемы

- **Реконструкция из событий.** Pemainы регистрируются через ALIVE_LIST
  (placeholder) или `register_player` (не событие). Игрок, который не
  появлялся в alive-листе и не говорил, может отсутствовать в реконструированном
  состоянии → пример пропускается (`player_missing`).
- **Ошибочные предсказания.** Assistant-ответ строится только по ground
  truth; `reasoning` моделled-ответа сохраняется, только если предсказание
  было правильно.
- **Анонимизация не трогает свободный текст.** PII в сообщениях/рассуждениях
  остаётся — review перед распространением.
- **LM Studio не принимает адаптер через API** — только через UI.
- **Only offline.** Во время партии finetune-пайплайн не запускается.
