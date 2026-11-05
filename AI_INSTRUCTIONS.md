# AI_INSTRUCTIONS.md — правила для ИИ-агента

> **Кому адресовано.** Это файл для ИИ-агента (Cursor / Claude Code / Copilot),
> который работает с этим репозиторием. Прочитай его полностью перед любыми правками.
> Продуктовый контекст — в `PROJECT_CONTEXT.md`.

---

## 0. Что это за проект (одним абзацем)

Локальный ассистент-тренер для игры в мафию в Telegram. Telethon-юзербот читает
сообщения, парсер структурирует игру, LM Studio (OpenAI-совместимый API) анализирует
и выдаёт JSON с ролями и уверенностью, два консольных окна показывают статистику
и поток рассуждений. Только Python 3.11+, только Windows, только русский язык.

---

## 1. Роль и границы ИИ-агента

Ты — **инженер-исполнитель**, не продукт-менеджер. Если задача неясна — задай
вопрос, а не догадывайся о бизнес-логике. Можешь принимать **технические** решения
самостоятельно (названия функций, структура модуля), но:

- НЕ добавляй фичи, которых нет в `PROJECT_CONTEXT.md` (сначала спроси).
- НЕ меняй публичные контракты (JSON-схему ответа LLM, схему БД, имена ролей)
  без обновления этого файла и `PROJECT_CONTEXT.md`.
- НЕ отправляй сообщения в Telegram (программа — только чтение).
- НЕ переводи интерфейс и промпты на другие языки.

---

## 2. Стек и окружение

- **Python 3.11+**, `asyncio` для Telegram и HTTP.
- **Windows only**, локальный запуск.
- **Зависимости:** Telethon, `httpx` (или `openai`), `pydantic` (v2), `pydantic-settings`,
  `PyYAML`, `rich` (для таблиц в консолях), `loguru` (логи), `pytest`, `ruff`, `mypy`.
- **LM Studio** на `http://localhost:1234/v1` — обязательно проверять healthcheck на старте.
- **SQLite** строго через стандартную библиотеку `sqlite3` + тонкая легковесная обёртка (без SQLAlchemy и без ORM — схема простая, лишние зависимости исключены).

---

## 3. Структура репозитория (целевая, feature-based)

```
mafia_coach/
├── app/
│   ├── __main__.py          # реальная точка входа пакета (python -m app)
│   ├── orchestrator.py      # связывает scanner → parser → analyzer → ui/memory
│   └── bootstrap.py         # загрузка конфигов, инициализация БД, сессий
│
├── domain/                  # чистые модели, без I/O
│   ├── roles.py             # enum Role, Faction; загрузка из roles.yaml
│   ├── phases.py            # enum Phase
│   ├── events.py            # dataclass'ы событий (MessageEvent, DeathEvent, …)
│   ├── game_state.py        # GameState: агрегат партии, без зависимостей на Telegram
│   ├── rules.py             # мета-правила клуба (антитаргет, линч в первые 2 дня…)
│   └── exceptions.py        # доменные исключения
│
├── infra/
│   ├── telegram/
│   │   ├── client.py        # Telethon-клиент, вход, подписка на чат
│   │   └── session.py       # работа с *.session
│   ├── llm/
│   │   ├── client.py        # LLMClient (abc) + LMStudioClient (реализация)
│   │   ├── schemas.py       # pydantic-схема ответа модели (structured output)
│   │   └── prompts/
│   │       ├── system.jinja
│   │       └── analyze.jinja
│   ├── db/
│   │   ├── connection.py
│   │   ├── migrations/
│   │   │   └── 001_init.sql
│   │   └── repositories/
│   │       ├── games.py
│   │       ├── players.py
│   │       ├── events.py
│   │       └── predictions.py
│   └── logging/
│       └── setup.py         # loguru: файлы в logs/, ротация (настройки из settings.yaml)
│
├── features/
│   ├── scanner/             # приём сообщений, фильтрация
│   ├── parser/              # классификация сообщений + извлечение сущностей
│   │   ├── classifier.py
│   │   ├── bot_messages.py  # роли, смерти, фазы из сообщений бота
│   │   └── speech.py        # речь игроков (текст + метаданные)
│   ├── analyzer/            # формирование запроса в LLM
│   │   ├── context_builder.py
│   │   ├── heuristic_hints.py
│   │   └── analyzer.py
│   ├── memory/
│   │   ├── player_profile.py
│   │   ├── style_extractor.py
│   │   └── retro.py         # сверка predictions ↔ final_role
│   └── ui/
│       ├── stats_window.py       # таблица предсказаний (output-only через IPC)
│       ├── reasoning_window.py   # поток рассуждений (output-only через IPC)
│       └── window_host.py        # запуск двух процессов-консолей на Windows
│
├── configs/
│   ├── settings.yaml
│   └── roles.yaml
│
├── rules/
│   └── Mafia_rules.txt   # официальные правила клуба (read-only)
│
├── data/
│   ├── sessions/                # *.session от Telethon (в .gitignore)
│   ├── db/                      # SQLite-файл (в .gitignore)
│   └── datasets/                # будущие датасеты для fine-tune (в .gitignore)
│
├── logs/                        # loguru (в .gitignore)
├── scripts/
│   ├── make_session.py          # одноразовое создание Telethon-сессии (читает settings.yaml)
│   ├── export_dataset.py        # выгрузка в JSONL для fine-tune
│   └── run_windows.py           # запуск двух консолей вручную
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/                # примеры сообщений, ответов LM Studio
├── docs/
│   ├── architecture.md
│   ├── parser_spec.md
│   └── llm_contract.md
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
├── README.md
└── main.py                      # корневая обёртка запуска (вызывает app.__main__)
```

### Правила раскладки

- **`domain/` не импортирует ничего из `infra/`, `features/`, `app/`.** Только stdlib
  и `pydantic`. Домен — сердце, зависимостей наружу нет.
- **`infra/`** — всё, что общается с внешним миром (Telegram, LM Studio, БД, файлы).
- **`features/`** — прикладные сценарии. Может зависеть от `domain/` и `infra/`.
- **`app/`** — сборка (bootstrap, orchestrator). Зависит от всего.
- Циклические импорты запрещены. Если появились — выноси общее в `domain/`.

---

## 4. Стиль кода

### Общее
- **Type hints везде**, включая возвращаемые типы. `mypy` в strict-режиме для `domain/`,
  `infra/`; для `features/` — strict.
- **`from __future__ import annotations`** в каждом модуле.
- Строки — **только двойные** (`black`/`ruff format`).
- Максимальная длина строки — **100**.
- Файлы и модули — `snake_case`. Классы — `PascalCase`. Константы — `UPPER_SNAKE`.
- Никаких `print()` в коде (кроме самих окон вывода). Логи — через `loguru`.

### Async
- Telegram-клиент и HTTP к LM Studio — на `asyncio`.
- Не блокируй event loop: `sqlite3` — синхронный, вызывай через
  `asyncio.to_thread` или в отдельном потоке/процессе (в зависимости от частоты).
- Стриминг ответа LM Studio реализуется через `httpx.AsyncClient.stream`.

### Ошибки
- Никаких «голых» `except:`. Ловим конкретные классы.
- Доменные ошибки — в `domain/exceptions.py` (`ParserError`, `LLMError`,
  `TelegramNotConnectedError`).
- Внешние вызовы (Telegram, LLM) — с таймаутами и retry (tenacity), но не более 3 попыток.

### Логи
- `loguru`, файлы в `logs/` с ротацией по 10MB, хранение 30 дней.
- Уровни: DEBUG — в dev, INFO — в проде. Ошибки — всегда WARNING/ERROR + traceback.
- Reasoning модели (поток) идёт **и** в окно, **и** в лог-файл `logs/reasoning.log`.

---

## 5. Контракты, которые нельзя ломать без обновления документации

### 5.1 JSON-схема ответа LLM (`infra/llm/schemas.py`)
Описывается pydantic-моделью `AnalysisResponse`. Поля — см. `PROJECT_CONTEXT.md §8`.
Меняешь поля — обнови:
- `PROJECT_CONTEXT.md §8`;
- `docs/llm_contract.md`;
- промпты в `infra/llm/prompts/`;
- фикстуры в `tests/fixtures/`.

### 5.2 Схема БД (`infra/db/migrations/`)
Миграции — только новые файлы `NNN_xxx.sql`; **старые не редактируются**.
Каждая миграция идемпотентна на пустой БД (проверяется тестом).
Контракт `games.winner_faction`: `town|mafia|neutral|mixed|unknown|NULL`;
`NULL` означает отсутствие END_GAME, `mixed` — несколько фракций победителей,
`unknown` — пустой список победителей или неразрешённая роль.

### 5.3 Имена ролей (`configs/roles.yaml`)
`id` (латиницей, snake_case) и `name_ru` (как в правилах) — константы.
Если добавляешь роль — добавь в `roles.yaml`, `docs/architecture.md`, тесты.

### 5.4 Формат событий (`domain/events.py`)
Каждое событие — frozen dataclass с `ts: datetime`, `kind: str`, `game_id: int`.
`payload` — через pydantic-модель, не «сырой dict».

### 5.5 Клубные паттерны (`configs/club_patterns.yaml`)
Файл содержит список документированных паттернов поведения игроков с выводом о
роли и уровнем уверенности. Каждый паттерн имеет поля:

- `id` — идентификатор вида ``"#1"``..``"#7"`` (единственный источник истины для `pattern_ids` в `AnalysisResponse`);
- `trigger` — тип наблюдаемого поведения (`claim_investigation`, `claim_role_direct`, `vote_against_confirmed_commissioner`, `bot_investigation_result`, `bot_lucky_survived`, `claim_peaceful_contradiction` и т.п.);
- `conclusion_role` — роль, на которую указывает паттерн (каноническое `name_ru` из `roles.yaml`);
- `confidence_tier` — один из ``0.9``, ``0.8``, ``0.6`` (см. §10);
- `description` — человек читаемое описание для колонки UI «Паттерн»;
- `requires_confirmation` — ``true`` для паттернов, которым нужен результат голосования перед применением (#1, #5).

Загрузчик — `domain.club_patterns.load_club_patterns`; валидация через pydantic
(`ClubPattern`, `ClubPatterns`). Файл не коммитится как `.txt` — актуальная
версия хранится в `configs/club_patterns.yaml`.

Обнаружение и применение паттернов — `features.analyzer.pattern_detector.PatternDetector`:

- `detect()` сканирует события игры (сообщения, голоса, раскрытия ролей) и
  возвращает `PatternMatch` (переопределение роли) и `SlotRelease` (освобождённый
  слот composition). Паттерны с `requires_confirmation=true` (#1, #5) срабатывают
  только при подтверждающем действии (голосование).
- `apply_overrides()` применяет совпадения к `PlayerPrediction`: роль, confidence
  по tiers §10, `pattern_ids` / `pattern_descriptions` из конфига. При конфликте
  паттернов приоритет у более сильного (выше confidence); конфликт фиксируется в
  `reason`.
- Освобождённый слот получает `suspected_role = null`, `confidence = 0.0`,
  `reason = "слот переопределён паттерном #N"`, `pattern_ids = ["#N"]`.
- Остальные квоты redistribuруются `_enforce_composition` среди живых не-молчаливых
  игроков без ролей. Молчаливые игроки (без сообщений и голосов) не подлежат
  переопределению паттерном.

---

## 6. Как добавлять фичи (workflow)

1. Проверь, есть ли задача в `PROJECT_CONTEXT.md §13 (дорожная карта)`. Если нет — спроси.
2. Определи слой: `domain/`, `infra/`, `features/`, `app/`.
3. Если меняется контракт (см. §5) — сначала обнови документацию, потом код.
4. Пиши тест **до** или вместе с кодом:
   - `domain/` — pure unit tests;
   - `infra/` — моки (Telethon-mock, httpx-mock, in-memory SQLite);
   - `features/` — интеграционные с фикстурами из `tests/fixtures/`.
5. Прогони локально:
   ```bash
   ruff check .
   ruff format --check .
   mypy .
   pytest -q
   ```
6. Обнови `README.md`, если затронут пользовательский сценарий.
7. Обнови `PROJECT_CONTEXT.md`, если изменились продуктовые решения.

---

## 7. Тестирование

- **pytest** обязателен. `pytest-asyncio` для async.
- Покрытие: `domain/` — 90%+, `features/` — 70%+ (по возможности).
- Фикстуры — **реалистичные сообщения** из этого клуба (см. `tests/fixtures/telegram/`):
  регистрация, ночь, день, голосование, объявления ролей ботом.
- LM Studio в тестах **никогда не вызывается по-настоящему**: используем фикстуру
  `fake_llm` (возвращает заранее заданный JSON).
- Телеграм в тестах — через фейковый клиент, без сети.

---

## 8. Документация

- **README.md** — краткая инструкция запуска (5–10 строк), скриншот двух окон (позже),
  требования, ссылка на `PROJECT_CONTEXT.md`.
- **docs/architecture.md** — схема слоёв и потоков (уже есть в PROJECT_CONTEXT,
  но здесь подробнее с диаграммой).
- **docs/parser_spec.md** — какие сообщения как классифицируются, примеры, edge cases.
- **docs/llm_contract.md** — JSON-схема, примеры запросов/ответов, промпты.
- Docstrings — обязательны для публичных функций в `domain/` и `infra/`.
  Формат — Google style.

---

## 9. Запрещено

- ❌ **Отправлять сообщения в Telegram** — только чтение.
- ❌ **Хардкодить chat_id, токены, пути** — всё через `configs/settings.yaml`
  (значения по умолчанию в коде допустимы только как `None` + предупреждение).
- ❌ **Менять `.session`-файлы** или автоматически логиниться — сессия готовится вручную.
- ❌ **Использовать иностранные языки** в UI и промптах (кроме английских имён
  переменных и технических терминов).
- ❌ **Добавлять GUI/веб** без явного согласования.
- ❌ **Ставить тяжёлые зависимости** (torch, tensorflow) в основной пакет — только
  в отдельные опциональные группы (`[project.optional-dependencies].training`).
- ❌ **Коммитить** `data/`, `logs/`, `*.session`, БД-файлы — всё в `.gitignore`.
- ❌ **Писать «магические» промпты** без шаблонизации через Jinja — все промпты
  в `infra/llm/prompts/*.jinja`.

---

## 10. Приоритеты источников роли и тиры уверенности

Когда модель выбирает роль игрока, источники расставлены по приоритету — от
высокого к низкому:

1. **Паттерн клуба** — поведение игрока совпадает с задокументированным
   паттерном из `configs/club_patterns.yaml` и подтверждается сообщениями или
   голосованием. Высший приоритет.
2. **Состав бота** — сообщение бота «Живые игроки … Из них … Всего: N» с
   точными количествами ролей (`role_count`). Роли распределяются по живым
   игрокам ровно по количеству слотов; состав бота важнее любой гипотезы.
3. **Эвристика** — правила клуба (`domain/rules.py`, например линч в первые
   два дня) и подсказки `heuristic_hints`. Носит рекомендательный характер;
   самый низкий приоритет.

Тиры уверенности (`confidence`):

- **0.9** — прямые заявления роли (игрок или бот nannte роль напрямую;
  паттерны #2 и #7).
- **0.8** — подтверждённые проверки (проверка комиссара/детектива и т.п.;
  паттерны #1 и #5).
- **0.6** — негативные выводы по поведению (уклончивость, пассивность;
  паттерны #3 и #4).
- Ниже 0.6 — только осторожные гипотезы. Если у игрока нет сообщений и
  голосов — `suspected_role = null`, `confidence = 0.0`,
  `reason = "нет данных"`.

Правило смены роли: смена роли без **НОВОГО** аргумента из сообщений,
голосований или раскрытых ролей запрещена. Прошлые предсказания — якорь:
без нового аргумента роль сохраняется, можно понизить `confidence`, но
нельзя заменить роль без причины.

## 11. Инструменты и pre-commit

`.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.0
    hooks:
      - id: mypy
```

`pyproject.toml` — конфиги для `ruff`, `mypy`, `pytest`, `black` (если используешь
вместо `ruff format`). Обязательно указывать `python_requires = ">=3.11"`.

---

## 12. Чек-лист перед завершением задачи

- [ ] Изменения соответствуют `PROJECT_CONTEXT.md` (или он обновлён).
- [ ] Слои не нарушены (`domain` не тянет `infra`).
- [ ] Type hints на месте, `mypy` зелёный.
- [ ] `ruff check` и `ruff format --check` — чисто.
- [ ] Тесты добавлены/обновлены, `pytest` зелёный.
- [ ] Обновлена документация (`README`, `docs/*`, при необходимости `PROJECT_CONTEXT.md`).
- [ ] Никаких новых «тяжёлых» зависимостей без обсуждения.
- [ ] В Telegram ничего не отправляется.
- [ ] Все новые строки UI — на русском.
- [ ] Если менялась JSON-схема или схема БД — обновлены §5.1/§5.2 и фикстуры.

---

## 13. Полезные ссылки внутри репо

- `PROJECT_CONTEXT.md` — что и зачем.
- `rules/Mafia_rules.txt` — правила клуба (read-only).
- `docs/llm_contract.md` — контракт с LM Studio.
- `configs/roles.yaml` — роли (единственный источник истины).
- `tests/fixtures/telegram/` — примеры сообщений для тестов.

---

## 14. Зафиксированные решения и оставшиеся вопросы

### 13.1 Зафиксированные решения
1. **Имя модели в LM Studio:** `qwen/qwen3.5-9b` (при развёртывании сверяется с ответом `GET /v1/models`).
2. **Сессия Telethon:** хранится в `data/sessions/mafia_coach.session`, создаётся через утилиту `scripts/make_session.py`, считывающую `api_id` и `api_hash` напрямую из `configs/settings.yaml`.
3. **Управление и выбор чата:** ввод команд (`/attach`, `/status`, `/quit`) — строго через основное консольное окно оркестратора (`main.py`). Окна `Stats Window` и `Reasoning Window` — `output-only` по лёгкому IPC.
4. **Формат датасета для fine-tune:** ChatML/SFT.
5. **Лимит контекста LLM:** `context_window_messages: 200`, суммаризация после 300 сообщений.
6. **Режим триггера LLM:** настраиваемый `trigger_mode` (`batch_debounce` / `every_message` / `queue`), по умолчанию `batch_debounce`. При `batch_debounce` game-state события (`alive_list`, смерти, фазы, голосования, раскрытия) вызывают LLM немедленно; ALIVE_LIST в фазе `REGISTRATION` не отбрасывается. JSON-схема `AnalysisResponse`, схема БД и имена ролей не меняются; состав бота и прошлые предсказания — часть промпта/постобработки.
7. **Игровой бот:** указывается пользователем через `telegram.bot_username` или `telegram.bot_id`. Бот не редактирует сообщения (парсер слушает только `events.NewMessage`).
8. **База данных:** чистый `sqlite3` (без SQLAlchemy). Сырые события (`events`) сохраняются в SQLite сразу по мере получения; `predictions` — по готовности ответа LLM.
9. **Конфигурация логов:** централизована внутри `configs/settings.yaml` (отдельный файл `logging.yaml` упразднён).

### 13.2 Открытые вопросы к владельцу (уточнять по мере необходимости)
1. **Кастомные эвристики клуба:** требуется ли жесткий справочник специфических для данного сообщества сокращений и игровых паттернов в `domain/rules.py`?
2. **Ручной ввод поправок:** планируется ли в будущих версиях ручная корректировка ролей через консоль оркестратора (например, `/mark <player> <role>`)?
