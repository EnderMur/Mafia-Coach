# Дообучение модели (fine-tune) — Mafia Coach

> **Статус: экспериментально. Реального обучения и реальных метрик для нового
> пайплайна нет; артефакты не готовы к использованию.** Разделы 1–9 ниже
> описывают старый путь, а актуальные ограничения и запуск нового — в разделе
> «Пайплайн с проверкой обобщения».

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

### 5.1 Если кнопки Add LoRA нет (LM Studio 0.3.20+)

В новых сборках LM Studio кнопку убрали для GGUF-моделей. Адаптер с `adapter_model.safetensors`
напрямую не загрузить — нужно **смержить** его с базой один раз:

```powershell
pip install transformers peft accelerate torch --upgrade
python scripts/merge_lora.py --adapter data/adapters/2026-11-07_04-19
# base_model берётся из adapter_config.json; можно переопределить --base-model Qwen/Qwen2.5-7B-Instruct
```

Скрипт создаст `data/merged/<base>-mafia-merged_TIMESTAMP/` — это уже обычная HF-модель
без LoRA. Дальше два варианта:

**Вариант A — загрузить как HF (проще, без конвертации):**
1. В LM Studio: `My Models` → клик по пути к папке моделей вверху → `Открыть в проводнике`
2. Создай `C:\Users\<you>\.lmstudio\models\local\mafia-<base>\` и скопируй туда всё из `data/merged/...`
3. Перезапусти LM Studio — появится `local/mafia-<base>` → `Load` → `Start Server`
4. В `configs/settings.yaml` укажи `llm.model: "local/mafia-<base>"`

**Вариант B — конвертировать в GGUF (экономия VRAM, рекомендуется для Qwen):**
```bash
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
python convert_hf_to_gguf.py ../Mafia-Coach/data/merged/xxx --outfile mafia.gguf
# затем в LM Studio: lms import mafia.gguf
# или вручную: скопируй mafia.gguf в .lmstudio/models/local/mafia-xxx/model.gguf (два уровня папок!)
```

Проверка: `curl http://localhost:1234/v1/models` должен вернуть имя смерженной модели.
Подробнее см. вывод `python scripts/merge_lora.py --help`.

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

---

## Пайплайн с проверкой обобщения (`scripts/finetune/`)

Это отдельный, более строгий путь; предыдущие разделы описывают старые скрипты,
не совместимые с его проверками качества. Никакие изменения настроек игры не
требуются. Запуск — **Windows + Python 3.11 + NVIDIA CUDA**; Colab на Linux не
является поддерживаемой платформой этого проекта. В CI проверяется только код
на синтетических данных без скачивания моделей и GPU. В отличие от старой
реализации здесь используется PEFT + TRL; Unsloth не обязателен.

### Требования к окружению

Для **проверки шаблона** достаточно CPU, Python 3.11+, `transformers`,
токенизатора базовой модели и работающего Ollama; CUDA не обязательна.
Предупреждение об отсутствии CUDA не останавливает проверку. Первый запуск
может загрузить токенизатор из Hugging Face; для автономного запуска заранее
заполните кеш и укажите `--local-files-only` (для pytest —
`FINETUNE_LOCAL_FILES_ONLY=1`). Проверяются именно файлы `base_model` и
зафиксированный Jinja-шаблон; папка адаптера не нужна. После обучения стоит
отдельно сравнить сохранённый вместе с адаптером токенизатор с базой: ревизии
HF в конфиге не зафиксированы и могут измениться.

Для **обучения** в этом пайплайне нужны Windows, NVIDIA GPU, драйвер,
CUDA-совместимый PyTorch и поддерживаемые этим сочетанием PEFT/TRL и
bitsandbytes; для Unsloth/Triton и сборки CUDA-расширений может потребоваться
CUDA Toolkit, путь к которому задаётся `CUDA_HOME` или `CUDA_PATH`. Эти
переменные сами по себе не доказывают доступность GPU: проверьте
`torch.cuda.is_available()` и соответствие драйвера и сборки torch.

Проект требует Python 3.11+, CI проверяет 3.11, но реальный GPU-прогон в этом
репозитории ещё не выполнен: гарантированной матрицы версий torch/CUDA/Unsloth
нет. Указанные пользователем Windows/Python 3.13/torch 2.14.1 — контекст проверки
токенизатора, не доказательство работоспособности обучения. Возможные сочетания
CUDA 12.6/13.0/13.2 или Unsloth с CUDA 11.8/12.x нужно сверить с доступными
Windows-колёсами выбранных версий, драйвером, GPU и затем проверить реальным
smoke-тестом обучения. См. [релиз PyTorch 2.14.1](https://github.com/pytorch/pytorch/releases/tag/v2.14.1)
и [зависимости Unsloth](https://github.com/unslothai/unsloth/blob/main/pyproject.toml).

### Подготовка

1. Создайте виртуальное окружение, установите подходящий для драйвера CUDA
   PyTorch, `pip install -e ".[finetune]"`, а также зависимости выбранной
   версии `llama.cpp` (см. его инструкции). При несовместимости Windows с
   Unsloth установите отдельно `torch`, `transformers`, `trl`, `datasets`,
   `peft`, `accelerate`, `bitsandbytes`, `jinja2`, `pyyaml`, `loguru`,
   `pydantic>=2`. Совместимость версий и поддержка Qwen3.5 данным выпуском
   Transformers/llama.cpp должны быть проверены на рабочей машине.
2. Скопируйте `configs/finetune.example.yaml` в локальный файл вне Git,
   например `data/tmp/finetune.yaml`. Укажите реальные пути и сообщества.
   База `Qwen/Qwen3.5-9B` выбрана как предполагаемая HF-версия указанной в
   PROJECT_CONTEXT модели `qwen/qwen3.5-9b`; соответствие весов/ревизии нужно
   проверить, это **не** файл GGUF из LM Studio. Для неё
   требуется совместимый токенизатор и поддержка архитектуры в установленном
   ПО. Для 9B QLoRA ориентируйтесь на 16–24 ГБ VRAM, для LoRA — 24–32+ ГБ;
   это грубая оценка, не гарантия: длина контекста, версии и оптимизатор
   существенно влияют на расход. При нехватке памяти уменьшите контекст или
   используйте более мощную машину; слияние на CPU требует десятков ГБ ОЗУ.
3. Подготовьте локальный JSONL-файл с **экспертной разметкой**, например
   `data/datasets/annotations.jsonl`. Одна строка — один снимок:

```json
{"db":"data/db/mafia.db","game_id":1,"cutoff_event_id":123,"community":"клуб_а","format":"классика","outcome":"поражение","reply_type":"обвинение","reviewed":true,"response":{"analysis":[{"player_id":42,"player_name":"Аня","suspected_role":"Доктор","confidence":0.5,"reason":"Есть аргументы в репликах.","reasons":["Есть аргументы в репликах."],"pattern_ids":[],"pattern_descriptions":[]}],"game_read":"Идёт обсуждение."}}
```

   Это лишь иллюстрация формата, не готовый обучающий пример. `cutoff_event_id`
   — ID **публичного** события этой игры, включаемого в снимок. Время событий
   должно быть с часовым поясом и не убывать по ID; снимки после конца партии
   запрещены. `response` должен соответствовать фактическим игрокам, их именам,
   тишине и, при наличии, составу бота; один пример при известном составе
   должен покрывать всех живых игроков. Эксперт должен проверить весь текст
   входа и ответа: нельзя использовать знание финала для объяснений, выдавать
   итоговые роли как уверенные предсказания без аргументов, включать приватные
   сообщения, персональные данные или копировать скрытые поля в текст реплики.
   Установка `reviewed: true` — утверждение человека, а не автоматическая
   гарантия. `reply_type` — единая ручная категория доминирующего типа реплик
   снимка; `format` — формат игры; `outcome` — `победа`, `поражение` или
   `ничья` относительно выбранной и согласованной позиции оценки. Указывайте
   единые определения во всех сообществах.
4. Назначьте **разные** сообщества в `train_communities`, `val_communities`,
   `test_communities`. В train должны быть минимум два формата, победа и хотя
   бы поражение или ничья. Одна партия не может пересекать сплиты, но при
   копировании одной партии в разные базы автоматическое обнаружение дубликатов
   не гарантируется — это нужно проверить вручную. В каждом сплите доля любой
   целевой роли и любого типа реплики не должна превышать 70%. Типы реплик
   считаются по разметке снимков, роли — по всем предсказаниям в ответах.
   Train детерминированно прореживается без копирования примеров; val/test не
   балансируются искусственно. Если ограничение не достигнуто, подготовка
   прерывается: соберите/разметьте недостающие классы. Отчёт содержит классы
   до и после отбора, число удалённых снимков, форматы, исходы и хеши файлов.
   Непредставленные классы не выявляются автоматически: сравните отчёт с
   заранее утверждённым каталогом ролей и типов.

```powershell
python -m scripts.finetune.prepare_dataset --config data/tmp/finetune.yaml --annotations data/datasets/annotations.jsonl
python -m scripts.finetune.train_lora --config data/tmp/finetune.yaml
python -m scripts.finetune.export --config data/tmp/finetune.yaml
```

Подготовка открывает БД только для чтения и при каждом запуске удаляет старые
`train.jsonl`, `val.jsonl`, `test.jsonl`, `report.json` в `dataset_dir`. Если
результат нужен, сохраните его отдельно до повторного запуска. Вход модели
формируется из whitelist полей `message`, `vote`, `death`, `phase_change`,
`alive_list`, `role_reveal` до границы, без `game_players`, prediction history
и финала. Внутренний ID партии заменён на 0; дата/имя/ID автора относятся
к обычному контексту чата. `role_count` — публичный состав бота. Код не может
доказать, что исходная БД действительно состоит из публичных сообщений, и не
анонимизирует текст: до обучения и распространения нужна проверка источника,
согласий и PII. События `role_reveal` штатный scanner создаёт из сообщения
игрового бота, но БД не хранит подтверждение его источника. Включайте только
доверенные БД и проверяйте, что каждое раскрытие действительно было публичным
на момент снимка: тип события сам по себе не доказывает происхождение.
Исторические якоря и клубные подсказки не используются — оставшийся train/serve
skew следует учитывать в экспериментальной оценке.

### Метрики и экспорт

Обучение маскирует loss на system/user и обучает только ответ, не обрезая
примеры. Validation loss измеряется на базе до обучения, по эпохам и в конце;
train/test loss — на конечном адаптере. Высокий/неулучшающийся или растущий
validation loss, большой train/val gap и высокий test loss/gap блокируют
экспорт и сохраняют `quality.json` с рисками. Обучение требует хотя бы один
шаг, baseline и конечный validation loss; при отсутствии валидации статус не
создаётся. Даже при успешном скрининге `status=experimental`,
`ready_for_use=false`; `ready_for_export` означает только разрешение экспорта,
а не готовность к применению. Пороги примера — только
эвристика: loss не доказывает ни корректность, ни отсутствие переобучения.
Для решения о готовности дополнительно сравните базу и адаптер на **нетронутом
held-out test** по валидности JSON, точности/макро-F1 ролей и фракций,
калибровке уверенности, срезам по форматам, исходам и сообществам; проведите
слепую экспертную оценку аргументации и проверьте интервалы неопределённости.
При повторном подборе по test потребуется новый независимый test. Ни одной
реальной оценки или обучения в CI этот пайплайн не выполняет.

При прохождении скрининга export объединяет адаптер с **неквантованной** базой
на CPU через PEFT, вызывает `convert_hf_to_gguf.py --outtype f16` и создаёт
`Modelfile` с шаблоном и стоп-токенами. Установите путь `llama_cpp_dir` к
локальной совместимой версии llama.cpp. Дополнительное квантование GGUF здесь
не выполняется. Импортируйте GGUF в LM Studio, проверьте, что выбран встроенный
шаблон без UI override, и выполните пробный запрос. Для Ollama: из каталога
экспорта выполните `ollama create mafia-coach -f Modelfile`, затем проверьте
пробный запрос. Различия версий серверов, токенизатора, конвертера и системных
параметров могут менять поведение; фактическую эквивалентность нужно проверить
на целевой машине. По умолчанию данные и разметка — `data/datasets/finetune`,
адаптер и метрики — `data/adapters/finetune`, объединённая модель, GGUF и
Modelfile — `data/finetune_export`. Все эти пути, веса и GGUF игнорируются
Git; пользовательские пути нужно защитить самостоятельно. Если запуск
неудачен, старый `quality.json` удаляется до обучения, а старый Modelfile —
до конвертации, чтобы ошибка не выглядела как успешный экспорт.

### Ручная интеграционная проверка Ollama

Для каждого запроса `/api/chat`, в том числе debug и генерации,
`check_ollama_template.py` передаёт `"think": false`; параметр Modelfile не
используется. Нужны работающий Ollama с моделью, использующей созданный
`Modelfile`, локальный конфиг и проверенный JSONL-снимок. Результат зависит от
того, какую **модель и версию раннера** действительно загрузили: одинаковые
имена не гарантируют одинаковые веса и токенизаторы. Тест передаёт содержимое
снимка в Ollama, поэтому используйте доверенный локальный адрес и данные без PII.

```powershell
$env:FINETUNE_CONFIG = "data/tmp/finetune.yaml"
$env:FINETUNE_OLLAMA_MODEL = "mafia-coach"
$env:FINETUNE_SAMPLE = "data/datasets/finetune/test.jsonl"
$env:FINETUNE_OLLAMA_URL = "http://127.0.0.1:11434"
python -m pytest -q -s -m integration tests/finetune/test_chat_template.py
python -m scripts.finetune.check_ollama_template --config data/tmp/finetune.yaml --model mafia-coach --sample data/datasets/finetune/test.jsonl
# Только заранее скачанный локальный кеш, без загрузки:
python -m scripts.finetune.check_ollama_template --config data/tmp/finetune.yaml --model mafia-coach --sample data/datasets/finetune/test.jsonl --local-files-only
# Для pytest оффлайн-режим включается переменной:
$env:FINETUNE_LOCAL_FILES_ONLY = "1"
```

Скрипт логирует `mode` и число токенов:

- `count_only` (в результате `count_only=True`): публичный сервер не вернул
  `_debug_info.rendered_template` либо отклонил debug-флаг; локальная длина `apply_chat_template(..., tokenize=True,
  add_generation_prompt=True)` равна `prompt_eval_count` реального `/api/chat`.
  **Это слабая проверка:** равное количество токенов не доказывает совпадение
  их ID, рендера или шаблона. При несовпадении количества проверка завершается
  ошибкой. Сервер может проигнорировать debug-флаг и выполнить дополнительную
  генерацию; проверка не скрывает ошибки реального запроса.
- `render_and_count`: сервер вернул debug-рендер chat, который совпал с
  обучающим текстом, и количество токенов совпало; сервер не предоставил
  достаточные данные `/api/generate` для проверки ID.
- `render_and_tokens`: дополнительно через debug `/api/generate` и устаревший
  `context` совпали ID серверной токенизации с HF. Сам `/api/chat` не раскрывает
  свои ID, поэтому даже этот режим не доказывает напрямую внутренний поток ID
  chat. Несовпадение доступного рендера или ID всегда вызывает ошибку.

Усечение отключено, но `context` может зависеть от версии Ollama или повторной
токенизации границы ответа. Результат нужно сохранять вместе с версией Ollama,
именем модели и хешами GGUF/токенизатора без приватных данных. При отсутствии
переменных интеграционный pytest пропускается; при заданных — ошибки не скрываются.
Реальный GPU и LM Studio этими проверками не тестируются.

### Старые скрипты: функциональные различия и план

| Старый путь | Функциональное отличие от `scripts/finetune/` | План |
|---|---|---|
| `scripts/prepare_finetune_dataset.py` и `features/finetune/dataset_builder.py` | Автоматические цели из предсказаний и `final_role`, split по игре, опциональные Parquet и псевдонимы; нет разбиения по сообществам и текущих quality gates. Исправленная роль сопровождается объяснением из финального отчёта. | **Деприкейт для обучения**: сохранить для совместимости и ретроспективы до миграции пользователей; новые данные разметить и проверить заново. Не переиспользовать старые цели автоматически. |
| `scripts/finetune_lora.py`, `features/finetune/trainer.py` | Конфиг из настроек приложения, Unsloth или PEFT, старый отчёт; нет обязательного held-out сообщества и нового контроля шаблона. | **Деприкейт для обучения** после подтверждения нового пути на Windows/GPU; затем удалить отдельным согласованным изменением. |
| `scripts/merge_lora.py` | Самостоятельный PEFT merge с GPU/CPU/квантованием и fallback; не создаёт GGUF/Modelfile и не требует нового quality gate. | Пока **сохранить** для старых адаптеров; в дальнейшем вынести общий безопасный merge в библиотеку после теста реальных версий и согласовать совместимость CLI. |
| `scripts/export_dataset.py` | Сырой архив с приватными итогами либо упрощённый ChatML-черновик, анонимизирует только структурированные поля. | **Сохранить только как архивный экспорт**, не подавать в новый train; оценить перенос анонимизации после полного аудита PII. |
| `scripts/evaluate_model.py`, `features/finetune/evaluator.py` | Сравнение base/adapter по ролям и фракциям для старого формата; новый код считает loss и этот старый формат напрямую не принимает. | **Переиспользовать после адаптации** к новому held-out формату и проверки шаблона, затем добавить macro-F1 и срезы; пока не считать loss достаточным. |

Дублирование временное и сознательное: удаление существующих CLI нарушит
обратную совместимость, а совместное использование автоматических меток и
старого merge могло бы обойти новые ограничения. Порядок миграции: реальный
Windows/GPU smoke → адаптер оценки для нового формата → предупреждение о
деприкейте и обновление инструкций → отдельное удаление после согласования.
Старые инструменты пока не удалены и не принудительно отключены.
