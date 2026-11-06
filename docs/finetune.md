# Дообучение модели (fine-tune) — Mafia Coach

> Оффлайн-пайплайн: выгрузка датасета из SQLite → LoRA/QLoRA дообучение
> локальной модели → загрузка адаптера в LM Studio. **Не влияет на игру**:
> запускается отдельно, после завершения партий.

## Содержание

1. [Обзор](#1-обзор)
2. [Шаг 1: подготовка датасета](#2-шаг-1-подготовка-датасета)
3. [Шаг 2: дообучение (LoRA/QLoRA)](#3-шаг-2-дообучение-loraqlora)
4. [Загрузка адаптера в LM Studio](#4-загрузка-адаптера-в-lm-studio)
5. [Конфигурация](#5-конфигурация)
6. [Анонимизация](#6-анонимизация)
7. [Архитектура и слои](#7-архитектура-и-слои)
8. [Ограничения и известные проблемы](#8-ограничения-и-известные-проблемы)

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
        │
        ▼
scripts/finetune_lora.py               →  data/adapters/YYYY-MM-DD_HH-MM/
        │                                  (adapter + tokenizer + training_report.json)
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

Считатель пропускаемых примеров (`no_role_predicted`, `no_ground_truth`,
`unknown_ground_truth_role`, `incorrect`, `low_confidence`,
`duplicate_history`, `player_missing`) выводится в лог и в итоговую строку:

```
Prepared 42 examples → data/datasets/sft_train.jsonl (skipped: no_ground_truth=17)
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

## 4. Загрузка адаптера в LM Studio

После успешного прогона CLI выводит инструкцию:

1. LM Studio → «My Models» → выбрать базовую модель (`finetune.base_model`,
   обычно та же, что в `llm.model`).
2. «Add LoRA» рядом с моделью (или меню модели → LoRA) → папка адаптера.
3. «Start Server» (Developer) → `GET /v1/models` показывает модель с адаптером.

> OpenAI-совместимый API LM Studio не принимает адаптер через параметры
> запроса — загрузка только через UI model manager.

## 5. Конфигурация

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

## 6. Анонимизация

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

## 7. Архитектура и слои

```
features/finetune/
├── models.py               # DatasetRecord, Message, PredictionRow, TrainingOptions
├── game_reconstructor.py   # толерантная реконструкция GameState из событий
├── dataset_builder.py      # build_example: контекст → [system, user, assistant]
├── anonymizer.py           # PlayerAliasMap, anonymize_context, anonymize_payload
└── trainer.py              # train_lora (unsloth/peft), lmstudio_instructions

scripts/
├── prepare_finetune_dataset.py   # CLI: БД → data/datasets/*.jsonl
└── finetune_lora.py              # CLI: конфиг + датасет → data/adapters/YYYY-MM-DD_HH-MM/

app/config.py               # FinetuneSection + поле AppConfig.finetune
domain/exceptions.py        # TrainingError (общее доменное исключение)
```

- `features/finetune/` зависит от `domain/` и `features/analyzer/`; не
  импортирует `app/` (trainer не знает о конфиге — `TrainingOptions`
  собирается в CLI).
- Тяжёлые импорты (torch, unsloth, …) — ленивые внутри функций; модуль
  можно импортировать и тестировать без训练-стека.
- Тесты: `tests/unit/features/finetune/`, `tests/unit/scripts/test_prepare_finetune_dataset.py`,
  `tests/unit/scripts/test_finetune_lora.py`, `tests/unit/app/test_config_finetune.py`.

## 8. Ограничения и известные проблемы

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
