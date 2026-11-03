# Mafia Coach

**Статус:** MVP (v1) — functional, ожидает dogfooding · **Код полностью сгенерирован ИИ** · **Telegram: только чтение**

Локальный ассистент-тренер для мафии в Telegram. Читает чат как userbot (Telethon),
распознаёт фазы и события и через локальную LLM (LM Studio) выдаёт вероятностную
оценку ролей игроков с объяснением рассуждений.

Программа **не отправляет сообщений в Telegram**, не голосует и не играет за
пользователя. Это наблюдатель и тренер.

**Готовые файлы в комплекте:** `configs/settings.example.yaml`, `configs/roles.yaml`
(21 роль), `scripts/make_session.py`, `pyproject.toml`, `requirements.txt`, `LICENSE`
(MIT), `docs/architecture.md`, `docs/parser_spec.md`, `docs/llm_contract.md`,
`scripts/audit_pii.py`, `.gitignore`.

- Продуктовые требования: [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md)
- Правила разработки и контракты: [`AI_INSTRUCTIONS.md`](AI_INSTRUCTIONS.md)
- Пометка AI-generated: [`AI_DISCLOSURE.md`]
- Оговорки об ИИ-происхождении кода: [§12 «Происхождение кода и ограничения»](#12-происхождение-кода-и-ограничения)

---

## Содержание

1. [Требования](#1-требования)
2. [Быстрый старт (Windows)](#2-быстрый-старт-windows)
3. [Конфигурация](#3-конфигурация)
4. [Telethon-сессия: как получить и где хранить](#4-telethon-сессия-как-получить-и-где-хранить)
5. [Как узнать `chat_id`](#5-как-узнать-chat_id)
6. [Проверка работоспособности](#6-проверка-работоспособности)
7. [Пример вывода](#7-пример-вывода)
8. [Контракт с LLM (JSON)](#8-контракт-с-llm-json)
9. [Структура проекта](#9-структура-проекта)
10. [Устранение неполадок](#10-устранение-неполадок)
11. [Ограничения](#11-ограничения)
12. [Происхождение кода и ограничения](#12-происхождение-кода-и-ограничения)
13. [Публикация и лицензия](#13-публикация-и-лицензия)

---

## 1. Требования

| Компонент | Требование | Обязательно |
|---|---|---|
| ОС | Windows 10/11 (второе окно открывается через `CREATE_NEW_CONSOLE`) | да |
| Python | 3.11+ | да |
| [LM Studio](https://lmstudio.ai/) | Local Server на `http://localhost:1234`, модель загружена | да |
| Telegram | аккаунт с готовой `*.session` (userbot), состоящий в игровом чате | да |
| Консоль | PowerShell 5.1+ / `cmd.exe` / Git Bash / WSL (команды для всех — ниже) | — |

> ⚠️ Модель в конфиге по умолчанию — `qwen/qwen3.5-9b`. Убедитесь, что идентификатор
> в `configs/settings.yaml` **посимвольно совпадает** с тем, что загружено у вас
> в LM Studio (вкладка *My Models* → копировать имя), и с ответом `GET /v1/models`.
> При расхождении LM Studio вернёт `404 model not found`.

---

## 2. Быстрый старт (Windows)

Проект Windows-only, но команды ниже даны для трёх оболочек. Выберите одну.

### PowerShell (рекомендуется)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1        # если ругается на политику: Set-ExecutionPolicy -Scope Process RemoteSigned
python -m pip install --upgrade pip
pip install -e ".[dev]"
Copy-Item configs\settings.example.yaml configs\settings.yaml
notepad configs\settings.yaml        # заполнить chat_id, model, session_file
python main.py
```

### cmd.exe

```bat
py -3.11 -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -e ".[dev]"
copy configs\settings.example.yaml configs\settings.yaml
notepad configs\settings.yaml
python main.py
```

### Git Bash / WSL (только установка; сам запуск — из PowerShell/cmd)

```bash
py -3.11 -m venv .venv
source .venv/Scripts/activate      # в WSL: source .venv/bin/activate (нужен python3-tk, окна не откроются)
pip install -e ".[dev]"
cp configs/settings.example.yaml configs/settings.yaml
```

> **Важно:** запуск `main.py` из WSL или Git Bash не создаст два консольных окна —
> `CREATE_NEW_CONSOLE` работает только при запуске из нативной Windows-консоли.

### Если нет `pyproject.toml` / `pip install -e .` не работает

Пакет объявлен в `pyproject.toml` (см. [структуру](#9-структура-проекта)). Если файла
в вашей копии нет, используйте `requirements.txt`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

Зависимости: `telethon`, `httpx`, `pydantic>=2`, `pydantic-settings`, `pyyaml`, `jinja2`, `loguru`,
`tenacity`, `rich` (таблица Stats Window). Dev-группа: `pytest`, `pytest-asyncio`,
`ruff`, `mypy`, `pre-commit`.

---

## 3. Конфигурация

Файлы лежат в `configs/`. При первом запуске копируется `settings.example.yaml`
в `settings.yaml` — **`settings.yaml` в репозиторий не коммитится** (там `chat_id`).

### 3.1 `configs/settings.yaml`

```yaml
# ── Telegram ───────────────────────────────────────────────────────────────
telegram:
  session_file: "data/sessions/mafia_coach"   # ОБЯЗАТЕЛЬНО, БЕЗ .session (см. §4)
  api_id: null            # ОБЯЗАТЕЛЬНО: числовой app api_id из my.telegram.org (см. §4)
  api_hash: null          # ОБЯЗАТЕЛЬНО: строковый app api_hash из my.telegram.org (см. §4)
  chat_id: null           # ОБЯЗАТЕЛЬНО: число или @username (см. §5); null = ждать /attach в главном окне
  bot_username: null      # Username игрового бота (например, @TrueMafiaBot) — указывается либо он, либо bot_id
  bot_id: null            # Либо числовой ID игрового бота (например, 123456789)
  read_only: true         # захардкожено в коде; значение только для наглядности
  ignore_own_messages: true

# ── LLM (LM Studio) ────────────────────────────────────────────────────────
llm:
  base_url: "http://localhost:1234/v1"        # ОБЯЗАТЕЛЬНО
  model: "qwen/qwen3.5-9b"                    # ОБЯЗАТЕЛЬНО: идентификатор из LM Studio
  temperature: 0.3
  top_p: 0.9
  max_tokens: 1024
  request_timeout_sec: 120                    # первая генерация на CPU бывает долгой
  connect_timeout_sec: 5
  max_retries: 3                              # не больше 3 (см. §8.4)
  retry_backoff_sec: 2
  stream_reasoning: true
  reasoning_max_chars: 1024
  context_window_messages: 100                # сколько последних сообщений в промпте
  summarize_after_messages: 300               # после этого старая часть суммаризуется
  trigger_mode: "queue"              # Режимы: every_message | queue | batch_debounce
  batch_debounce_sec: 3                       # задержка накопления пачки сообщений для batch_debounce
  batch_max_messages: 10                      # макс. число сообщений в пачке перед отправкой в LLM

# ── База данных ────────────────────────────────────────────────────────────
db:
  path: "data/db/mafia.db"                    # ОБЯЗАТЕЛЬНО; каталог создаётся сам
  apply_migrations_on_start: true

# ── Роли ───────────────────────────────────────────────────────────────────
roles_file: "configs/roles.yaml"              # ОБЯЗАТЕЛЬНО

# ── Анализ ─────────────────────────────────────────────────────────────────
analysis:
  min_confidence_to_display: 0.15             # ниже — роль скрыта, игрок остаётся в таблице
  # Примечание: логика частоты вызовов управляется через llm.trigger_mode.
  # В режиме batch_debounce фазовые события (смерти, фазы, голосования)
  # принудительно сбрасывают буфер и вызывают LLM немедленно.

# ── UI ─────────────────────────────────────────────────────────────────────
ui:
  stats_window: true
  reasoning_window: true
  language: "ru"

# ── Логи ───────────────────────────────────────────────────────────────────
logging:
  level: "INFO"                                # DEBUG для отладки парсера
  dir: "logs"
  rotation: "10 MB"
  retention_days: 30
```

**Обязательные поля:** `telegram.session_file`, `telegram.api_id`, `telegram.api_hash`,
`telegram.chat_id` (либо `/attach` в рантайме), `telegram.bot_username` **или** `telegram.bot_id`,
`llm.base_url`, `llm.model`, `db.path`, `roles_file`.
Остальные имеют значения по умолчанию в коде.

### 3.2 `configs/roles.yaml`

Единственный источник истины по ролям. Валидируется при старте через pydantic:
неизвестный `faction`, дубль `id` или пустой `name_ru` → падение с понятной ошибкой.

| Поле | Тип | Обязательно | Описание |
|---|---|---|---|
| `id` | str, `snake_case`, латиница | да | стабильный идентификатор; используется в БД и промптах |
| `name_ru` | str | да | имя как в правилах клуба (попадает в JSON-ответ LLM) |
| `faction` | `town` \| `mafia` \| `neutral` | да | фракция |
| `category` | `active` \| `passive` | да | есть ли ночные действия |
| `night_actions` | list[str] | да (может быть `[]`) | `kill`, `check`, `heal`, `block`, `burn`, `swap` … |
| `can_kill` | bool | да | может убивать |
| `can_check` | bool | да | может проверять роль/фракцию |
| `can_heal` | bool | да | может лечить |
| `notes` | str | нет | эвристики для промпта |
| `bot_aliases` | list[str] | да | как роль называет бот в сообщениях (для авторазметки) |

Фрагмент (полный список — 21 роль: 10 мирных, 5 мафии, 6 нейтралов):

```yaml
roles:
  - id: don
    name_ru: "Дон"
    faction: mafia
    category: active
    night_actions: [kill]
    can_kill: true
    can_check: false
    can_heal: false
    notes: "Выбирает, кто умрёт этой ночью от Мафии."
    bot_aliases: ["Дон"]

  - id: commissioner
    name_ru: "Комиссар Каттани"
    faction: town
    category: active
    night_actions: [check, kill]
    can_kill: true        # выстрел по результатам проверки
    can_check: true
    can_heal: false
    notes: "Стреляет после проверки."
    bot_aliases: ["Комиссар Каттани", "Комиссар", "Каттани"]

  - id: civilian
    name_ru: "Мирный житель"
    faction: town
    category: passive
    night_actions: []
    can_kill: false
    can_check: false
    can_heal: false
    notes: "Голосует днём, ночью спит. Часто прикрытие для Мафии и Убийцы."
    bot_aliases: ["Мирный житель", "Мирный"]
```

> Как ИИ использует `bot_aliases`: ищет их в сообщениях бота («Вася был Мафией»),
> чтобы записать `game_players.final_role`. Список алиасов должен покрывать все
> варианты написания, иначе авторазметка потеряет игрока. Сверяйте с
> `rules/Mafia_rules.txt` и реальными сообщениями чата.

### 3.3 `configs/club_patterns.txt` (gitignored)

Файл с вашими паттернами клуба. Модель читает их как подсказки; формат —
свободный текст. Создайте свою копию и заполните наблюдениями:

```powershell
Copy-Item configs/club_patterns.example.txt configs/club_patterns.txt
```

Файл **не коммитится**: это личные наблюдения. В публичный репозиторий попадает
только пустой шаблон с иллюстративным примером. Если личного файла нет,
используется шаблон; если нет ни того ни другого — пустая строка.

Путь можно изменить полем `club_patterns_file` в `settings.yaml`. Другое
расположение автоматически не добавляется в `.gitignore`: защитите его отдельно.
Содержимое передаётся модели в составе промпта — учитывайте это при работе
с удалённым LLM-сервером.

---

## 4. Telethon-сессия: как получить и где хранить

Сессия создаётся **один раз вручную**, до первого запуска тренера. Программа сама
не логинится и не трогает `*.session` (это прямо запрещено в `AI_INSTRUCTIONS.md §9`).

### 4.1 Получить `api_id` и `api_hash`

1. Открыть <https://my.telegram.org> → **API development tools**.
2. Создать приложение (название любое, platform — Desktop).
3. Скопировать `App api_id` (число) и `App api_hash` (32 символа).
4. Это **личные секреты**: в git не коммитить, не публиковать.

### 4.2 Создать файл сессии (готовый скрипт)

Запустите готовый скрипт `scripts/make_session.py` **из PowerShell/cmd** (он считывает `api_id`, `api_hash` и путь сессии напрямую из `configs/settings.yaml`, без ручного хардкода секретов в коде):

```powershell
python scripts/make_session.py
```

Код скрипта (уже включён в репозиторий `scripts/make_session.py`):

```python
"""Одноразовое создание Telethon-сессии. Читает настройки из configs/settings.yaml."""
from pathlib import Path
import yaml
from telethon.sync import TelegramClient

with open("configs/settings.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

t_cfg = cfg.get("telegram", {})
api_id = t_cfg.get("api_id")
api_hash = t_cfg.get("api_hash")
session_file = t_cfg.get("session_file", "data/sessions/mafia_coach")

if not api_id or not api_hash:
    raise ValueError("Заполните api_id и api_hash в configs/settings.yaml перед созданием сессии")

session_path = Path(session_file)
session_path.parent.mkdir(parents=True, exist_ok=True)

with TelegramClient(str(session_path), api_id, api_hash) as client:
    me = client.get_me()
    print("OK, авторизован как:", me.username or me.first_name, "id =", me.id)
```

Скрипт спросит номер телефона, код из Telegram и пароль 2FA (если включён).
После успешного вывода появится `data/sessions/mafia_coach.session`.

> Альтернатива без своего кода — одноразовый запуск любого Telethon-скрипта или
> `telethon` CLI. Главное: файл сессии должен появиться в `data/sessions/`.

### 4.3 Где хранить и как защитить

- Каталог: `data/sessions/` — он в `.gitignore` целиком.
- **Файл `*.session` = полный доступ к аккаунту.** Не пересылать, не бэкапить в облако,
  не запускать на чужой машине.
- Один файл — один аккаунт. Для разных аккаунтов разные имена.
- Права: доступ только владельцу. На Windows рекомендован BitLocker/EFS на диск с `data/`.
- Если файл утёк — завершить все сеансы: Telegram → *Настройки → Устройства* →
  **Terminate All Other Sessions**, затем удалить файл и создать новый.
- Не запускайте одновременно два клиента на одной сессии — возможны разрывы соединения.
- Аккаунт userbot рискует блокировкой при агрессивном поведении. Здесь программа
  **только читает**, но сам userbot всё равно может нарушать ToS Telegram —
  это ответственность пользователя.

---

## 5. Как узнать `chat_id`

Три способа, от простого к точному:

1. **Через `@userinfobot` / `@username_to_id_bot`** — переслать боту любое сообщение
   из игрового чата, он вернёт `chat_id` (для супергрупп начинается с `-100...`).
2. **Через web-версию:** открыть чат в `https://web.telegram.org/a/` — в адресной строке
   будет часть вида `#-1001234567890`, это и есть id.
3. **Через Telethon** (надёжнее всего, заодно проверит доступ сессии):

   ```python
   from telethon.sync import TelegramClient
   with TelegramClient("data/sessions/mafia_coach", API_ID, API_HASH) as client:
       for d in client.iter_dialogs():
           print(d.id, "|", d.title)
   ```

Укажите id в `telegram.chat_id` **как число** (`-1001234567890`), в кавычках или без —
YAML разберёт. Альтернативно можно оставить `null` и выбрать чат командой `/attach`
в основном окне запуска оркестратора (см. `PROJECT_CONTEXT.md §3` и `§15.9`; если команда ещё не реализована —
используйте id в конфиге).

---

## 6. Проверка работоспособности

Выполняйте по шагам — так сразу понятно, что именно сломано.

### Шаг 0. Статические проверки

```powershell
ruff check .
ruff format --check .
mypy .
pytest -q
```

### Шаг 1. LM Studio отвечает

```powershell
# список загруженных моделей — сверьте имя с llm.model
Invoke-RestMethod http://localhost:1234/v1/models | ConvertTo-Json -Depth 5
```

```bash
# Git Bash / WSL / любой *nix
curl -s http://localhost:1234/v1/models | jq '.data[].id'
```

Ожидается JSON со списком `id`. Если соединение отклонено — не запущен Local Server
(в LM Studio: *Developer* → **Start Server**).

### Шаг 2. Модель реально отвечает

```powershell
$body = @{
  model = "qwen/qwen3.5-9b"
  messages = @(@{ role = "user"; content = "Ответь одним словом: работает?" })
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://localhost:1234/v1/chat/completions -Method Post `
  -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 5
```

```bash
curl -s http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen/qwen3.5-9b","messages":[{"role":"user","content":"Ответь одним словом: работает?"}]}'
```

Ожидается `choices[0].message.content`. `404` → имя модели не совпадает.
Долгая пауза → модель не загружена в память (*My Models* → **Load**).

### Шаг 3. Сессия авторизована

```powershell
python -c "from telethon.sync import TelegramClient; import yaml; c=yaml.safe_load(open('configs/settings.yaml',encoding='utf-8')); t=c['telegram']; cl=TelegramClient(t['session_file'], t['api_id'], t['api_hash']).start(); me=cl.get_me(); print('OK', me.id, me.username); cl.disconnect()"
```

Ожидается `OK <id> <username>` без запроса кода. Если просит код — сессия не создана (§4).

### Шаг 4. Запуск и приём сообщений

```powershell
python main.py
```

Признаки успеха:

| Что проверить | Ожидание |
|---|---|
| Консоль оркестратора | строки `loguru`: конфиг загружен, миграции применены, Telegram подключён, чат найден |
| Окна | открылись **два** окна: «Stats» и «Reasoning» |
| Stats Window | таблица с никами игроков текущей партии |
| Сообщение в чат | в Stats Window меняются проценты, в Reasoning Window бежит текст |
| База | появился файл `data/db/mafia.db` |

### Шаг 5. БД создалась и наполняется

```powershell
python -c "import sqlite3; c=sqlite3.connect('data/db/mafia.db'); print([r[0] for r in c.execute(\"select name from sqlite_master where type='table' order by name\")])"
python -c "import sqlite3; c=sqlite3.connect('data/db/mafia.db'); print('events:', c.execute('select count(*) from events').fetchone()[0]); print('predictions:', c.execute('select count(*) from predictions').fetchone()[0])"
```

Ожидается список таблиц `events, game_players, games, player_style, players, predictions`
и растущие счётчики.

### Шаг 6. Логи

```powershell
Get-Content logs\app.log -Tail 50
Get-Content logs\reasoning.log -Tail 50
```

Если на шаге 4 окна не открылись, но лог пишется — проблема в UI-слое, не в пайплайне.

---

## 7. Пример вывода

### 7.1 Stats Window (обновляется по событиям через IPC)

Формат — как в `features/ui/stats_window.py` (rich Table, заголовок
«Stats — партия #N, `<phase>`, день D»). Пример на основе прогона C5 v13
(партия #12), имена обезличены:

```
┌──── Stats — партия #12, night, день 3 ─────┐
│ Игрок      Роль             Уверенность   │
├────────────────────────────────────────────┤
│ player_1   Мафия            90%           │
│ player_2   Мирный житель    85%           │
│ player_3   Дон              75%           │
│ player_4   —                —             │
└────────────────────────────────────────────┘
```

- Строки — предсказания из SQLite, отсортированные по уверенности (не более 20),
  ниже порога `analysis.min_confidence_to_display` не показываются.
- Игроки без предсказаний в реальном окне не отображаются: строка `player_4`
  приведена только для наглядности («данных нет»).
- Управление — через основное окно: `/attach`, `/status`, `/reset`, `/retro`, `/quit`.

### 7.2 Reasoning Window (стриминг, русский)

Фактический вывод для non-thinking модели (канал `delta.reasoning_content` /
`delta.reasoning` пуст — это ожидаемое поведение, не ошибка):

```
(streaming reasoning unavailable — non-thinking model)
```

У thinking-модели здесь идёт поток рассуждений на русском: он пишется в
`logs/reasoning.log` (обрезка до `llm.reasoning_max_chars`) и передаётся в окно
IPC-кадром до 4096 символов. Итоговый JSON (§7.3) собирается из `delta.content`
и в окно рассуждений не попадает.

### 7.3 Ответ модели (JSON, тип `AnalysisResponse`)

Схема не менялась (см. [§8](#8-контракт-с-llm-json) и `docs/llm_contract.md`);
имена обезличены:

```json
{
  "analysis": [
    {
      "player_id": 123456789,
      "player_name": "player_1",
      "suspected_role": "Мафия",
      "confidence": 0.68,
      "alternative": { "role": "Мирный житель", "confidence": 0.22 },
      "reasons": ["уклонялся от конкретики", "голосовал против комиссара"]
    },
    {
      "player_id": 987654321,
      "player_name": "player_2",
      "suspected_role": "Дон",
      "confidence": 0.44,
      "alternative": { "role": "Мирный житель", "confidence": 0.31 },
      "reasons": ["защищала раскрытую мафию"]
    }
  ],
  "game_read": "Мафия потеряла одного игрока, город играет по инерции первого дня.",
  "reasoning": "Пошаговые рассуждения модели..."
}
```

### 7.4 Реальная статистика

После C5 v13 (game 12, 23 predictions):

```
retro: total=23 correct=3 wrong=11 unknown=9 (3/14 = 21% accuracy).
```

Низкая accuracy ожидаема для thinking-модели на 10 игроках в ранней партии.
Для dogfooding нужна модель получше (`qwen2.5-7b-instruct`) и 20–30 игроков.

Дополнительные наблюдения по прогонам C5 (строки из логов как есть):

- v17: `flush ok: 5–11 predictions, 15–30s` — один цикл анализа занимает
  15–30 с, число предсказаний за цикл — 5–11.
- v20: `game 4: total=14, positive=11, negative=3, final_role=7`.

`unknown` в retro — предсказания, для которых не нашлось `final_role`
(игрок не попал в `game_players` или роль не распознана); механика —
`docs/architecture.md` §3.

> ⚠️ Примеры §7.1–§7.3 основаны на реальных прогонах C5 и обезличены
> (`player_N`, без реальных имён и id). Числа зависят от модели и промптов;
> код сгенерирован ИИ и не проходил приёмочное тестирование.

---

## 8. Контракт с LLM (JSON)

### 8.1 Транспорт

- Endpoint: `POST {llm.base_url}/chat/completions` (OpenAI-совместимый LM Studio).
- Режим: structured output через `response_format: {"type": "json_schema", ...}`;
  при неспособности модели — fallback на `{"type": "json_object"}` + pydantic-валидация.
- Стриминг (`stream: true`) используется только для канала `reasoning`
  (Reasoning Window + `logs/reasoning.log`), итоговый JSON собирается по завершении.

### 8.2 Схема ответа (`AnalysisResponse`, `infra/llm/schemas.py`)

| Поле | Тип | Обяз. | Ограничения |
|---|---|---|---|
| `analysis` | list[`PlayerPrediction`] | да | может быть пустым `[]`, если данных нет |
| `analysis[].player_id` | int | да | должен присутствовать в текущей партии |
| `analysis[].player_name` | str | да | как в чате |
| `analysis[].suspected_role` | str | да | **строго** значение из `roles.yaml → name_ru` |
| `analysis[].confidence` | float | да | `0.0 ≤ x ≤ 1.0`, не проценты |
| `analysis[].alternative.role` | str | нет | то же ограничение, что `suspected_role` |
| `analysis[].alternative.confidence` | float | нет | `0..1`; `confidence + alternative.confidence ≤ 1.0` |
| `analysis[].reasons` | list[str] | да | 1–5 коротких пунктов, по-русски |
| `game_read` | str | да | 1–3 предложения, по-русски |
| `reasoning` | str | да | пошаговые рассуждения, по-русски |

Правила валидации при получении:

1. JSON невалиден → один повторный запрос с «верни только JSON» → затем `LLMError`.
2. `confidence` вне `[0,1]` → clamp + `WARNING`, предсказание сохраняется.
3. `suspected_role` не из `roles.yaml` → попытка матча по `bot_aliases`,
   иначе строка отбрасывается, остальные принимаются.
4. `player_id`, которого нет в партии → запись отбрасывается, `WARNING`.
5. `confidence + alternative.confidence > 1.0` → нормализация (делим на сумму).

Сумма всех предсказаний по игроку не обязана быть равна 1: это **не** распределение
вероятностей ролей, а оценка уверенности в конкретной гипотезе.

### 8.3 Промпты

Все промпты — Jinja-шаблоны в `infra/llm/prompts/` (`system.jinja`, `analyze.jinja`).
Хардкод промптов в коде запрещён. Список ролей и правила клуба подставляются
из `configs/roles.yaml` и `rules/Mafia_rules.txt`. Язык — только русский.

### 8.4 Ошибки, таймауты, ретраи

| Ситуация | Поведение |
|---|---|
| Соединение отклонено (сервер выключен) | `llm.connect_timeout_sec` = 5 с → `LLMConnectionError`, в UI баннер «LM Studio недоступен», повтор при следующем событии |
| Модель не найдена (`404`) | `LLMError: model not found` + подсказка сверить `llm.model` с `/v1/models`; ретраев нет (бессмысленно) |
| Таймаут генерации | `llm.request_timeout_sec` (120 с) → ретрай |
| `429` / `5xx` | экспоненциальный backoff от `retry_backoff_sec` (2 → 4 с; при `max_retries: 3` выполняются ровно 3 попытки и 2 паузы) |
| Все ретраи исчерпаны | `max_retries: 3` (не больше), событие помечается `analysis_failed`, партия продолжается; бюджет `max_retries` действует на ВЕСЬ вызов analyze(), включая schema-fallback и validation-retry — суммарно ≤ 3 HTTP-запросов |
| Невалидный JSON | один повторный запрос, затем отказ |
| Пустой `analysis` | валидно: значит «данных мало», в таблице показывается `—` |

Принцип: **сбой LLM никогда не останавливает чтение чата**. Все сырые сообщения и события пишутся в таблицу `events` SQLite немедленно по факту получения (независимо от батчинга или ошибок модели). Статистика просто не
обновляется при сбое, в лог пишется `ERROR` с трейсбеком. В режиме `batch_debounce` любые фазовые события (смерти, смена дня/ночи, раскрытия ролей) немедленно сбрасывают буфер и форсируют срочный вызов анализа.

---

## 9. Структура проекта

```
mafia-coach/
├── app/                    сборка: bootstrap, оркестратор, запуск двух окон
│   └── __main__.py         точка входа пакета (python -m app)
├── domain/                 чистые модели: роли, события, GameState, исключения, правила
│   ├── roles.py  phases.py  events.py  game_state.py  rules.py  exceptions.py
├── infra/
│   ├── telegram/           client.py (Telethon), session.py
│   ├── llm/                client.py (LLMClient ABC + LMStudioClient),
│   │                       schemas.py, prompts/*.jinja (system.jinja, analyze.jinja)
│   ├── db/                 connection.py, migrations/001_init.sql, repositories/
│   └── logging/            setup.py (loguru, ротация из settings.yaml)
├── features/
│   ├── scanner/            приём и фильтрация сообщений
│   ├── parser/             classifier.py, bot_messages.py, speech.py
│   ├── analyzer/           context_builder.py, heuristic_hints.py, analyzer.py
│   ├── memory/             player_profile.py, style_extractor.py, retro.py
│   └── ui/                 stats_window.py, reasoning_window.py, window_host.py
├── configs/
│   ├── settings.example.yaml   шаблон (коммитится)
│   ├── settings.yaml           ваши значения (в .gitignore)
│   └── roles.yaml              роли клуба (коммитится)
├── rules/Mafia_rules.txt    официальные правила (read-only)
├── scripts/
│   ├── make_session.py         одноразовое создание Telethon-сессии (читает settings.yaml)
│   ├── dump_chat.py            read-only выгрузка истории чата в data/dumps/
│   ├── make_fixture.py         анонимизированный тестовый fixture из дампа
│   ├── export_dataset.py       выгрузка JSONL для fine-tune
│   ├── audit_pii.py            аудит git history/index/fixtures на PII перед Public
│   └── run_windows.py          запуск двух консолей вручную
├── data/                   sessions/, db/, datasets/, dumps/, tmp/ — всё в .gitignore
├── logs/                   app.log, reasoning.log — в .gitignore
├── tests/                  unit/, integration/, fixtures/
├── docs/                   architecture.md, parser_spec.md, llm_contract.md
├── pyproject.toml          пакет, зависимости, конфиги ruff/mypy/pytest
├── requirements.txt        альтернатива для установки без -e
├── .pre-commit-config.yaml
├── .gitignore
├── LICENSE
├── README.md
├── PROJECT_CONTEXT.md
├── AI_INSTRUCTIONS.md
└── main.py                 корневая обёртка запуска (вызывает app.__main__)
```

---

## 10. Устранение неполадок

| Симптом | Причина | Что делать |
|---|---|---|
| `ConnectionRefusedError` / `All connection attempts failed` | LM Studio Local Server не запущен | LM Studio → *Developer* → **Start Server**; проверить `Invoke-RestMethod http://localhost:1234/v1/models` |
| `404 model_not_found` / `model not found` | `llm.model` не совпадает с загруженной моделью | Скопировать точное имя из `/v1/models` в `settings.yaml` |
| Ответ приходит через 60–120 с или не приходит | Модель не загружена в RAM/VRAM (грузится «на лету») | *My Models* → **Load**; уменьшить `context_window_messages`; увеличить `request_timeout_sec` |
| `LLMError: invalid JSON` повторяется | Модель плохо держит JSON Schema | Взять модель с поддержкой structured output; понизить `temperature` до 0.1–0.2 |
| `SqliteError: unable to open database file` | Нет каталога `data/db/` или нет прав | Создать каталог вручную, проверить путь `db.path` |
| `sqlite3.OperationalError: no such table` | Миграции не применились | `db.apply_migrations_on_start: true`, проверить `infra/db/migrations/`; удалить пустую БД и перезапустить |
| `ChatNotFoundError` / `Cannot find any entity` | Неверный `chat_id` или аккаунт не в чате | Перепроверить id (§5); для `@username`-чатов нужно быть участником; попробовать числовой `-100...` id |
| `AuthKeyUnregisteredError`, `SessionRevokedError` | Сессия отозвана/удалена | Создать новую сессию (§4.2) |
| `SessionPasswordNeededError` при старте | Включён 2FA, сессия не авторизована | Завершить создание сессии (§4.2), ввести пароль |
| `database is locked` | Два процесса пишут в SQLite | Не запускать два экземпляра; проверить, что окна не открывают свою БД на запись |
| Окна не открылись, но лог пишется | Запуск не из нативной Windows-консоли (WSL/Git Bash/IDE) или нет прав | Запускать `python main.py` из PowerShell/cmd; проверить логи `multiprocessing` |
| Открылось только одно окно | Падение подпроцесса | Смотреть `logs/app.log`; запустить `python scripts/run_windows.py` вручную |
| Stats Window пуст, Reasoning Window работает | Партия не распознана (нет регистрации/фаз) | Проверить `chat_id`; посмотреть `parser_spec.md`; включить `logging.level: DEBUG` |
| В таблице нет ролей, только `—` | Модель вернула пустой `analysis` или `min_confidence_to_display` слишком высок | Понизить порог до 0.05; проверить длину контекста |
| Роли не размечаются после партии (`final_role` пуст) | `bot_aliases` не совпали с текстом бота | Дополнить `bot_aliases` в `roles.yaml`; посмотреть реальные сообщения бота |
| `ModuleNotFoundError: telethon` | Не активировано venv / не установлены зависимости | `.\.venv\Scripts\Activate.ps1`; `pip install -e ".[dev]"` |
| `py` не найден | Нет Python Launcher | Использовать `python -m venv .venv` или установить Python из python.org с галочкой *py launcher* |
| `Activate.ps1 cannot be loaded` | Политика выполнения PowerShell | `Set-ExecutionPolicy -Scope Process RemoteSigned` |
| Кириллица «кракозябрами» в консоли | Кодировка консоли/файла | `chcp 65001`; в PowerShell `[Console]::OutputEncoding=[Text.Encoding]::UTF8`; сохранять YAML в UTF-8 |
| Аккаунт разлогинился после запуска | Второй клиент на той же сессии | Остановить другие клиенты, создать отдельную сессию |

---

## 11. Ограничения

- Только Windows (второе окно требует `CREATE_NEW_CONSOLE`).
- Только русский язык в UI и промптах.
- Только чтение: сообщения не отправляются, голосования не выполняются.
- Нет веб-интерфейса, GUI и API-сервера; нет многопользовательского режима.
- Нет шифрования данных: SQLite и логи лежат открытым текстом на локальном диске.
- Голосовой чат не поддерживается (игра текстовая).
- Fine-tune (LoRA) — вне v1, только оффлайн-пайплайн.
- Предсказания **вероятностные** и могут ошибаться; это тренер, а не арбитр.
- Программа не советует нарушать правила клуба (линч без аргументов в первые 2 дня,
  действия против АнтиТаргета в первую ночь и т.п.).
- LM Studio не thread-safe при малой VRAM: параллельные запросы могут замедляться
  до таймаута.

---

## 12. Происхождение кода и ограничения

**Весь код, конфиги, промпты, тесты и документация сгенерированы ИИ.** Владелец
формулировал требования и правила, реализацию писал ИИ-агент; человек не писал
код вручную построчно.

Что это значит:

- Возможны следы генерации: избыточные абстракции, неиспользуемый код,
  правдоподобные, но неверные имена моделей/методов, комментарии, расходящиеся
  с реальным поведением.
- Поведение кода **не гарантировано**: возможны ошибки в деталях Telethon, LM Studio,
  SQLite, `asyncio` и в граничных случаях парсера.
- Примеры вывода в §7 основаны на реальных прогонах C5, но обезличены
  (`player_N`); числа зависят от модели и промптов.
- `PROJECT_CONTEXT.md` и `AI_INSTRUCTIONS.md` тоже созданы при участии ИИ и содержат
  допущения (помечены как «уточнить»/«предположение»). Источник истины по
  требованиям — подтверждение владельца.

Код предоставляется **«как есть» (AS IS)**, без гарантий, явных или подразумеваемых.
Автор не отвечает за ущерб от использования или невозможности использования кода.

Минимальный аудит перед применением: `ruff check .`, `mypy .`, `pytest -q` + шаги
из [§6](#6-проверка-работоспособности), плюс ручная проверка, что в Telegram
действительно не уходит ни одного исходящего запроса.

---

## 13. Публикация и лицензия

- **Лицензия:** MIT — файл [`LICENSE`](LICENSE) в корне. Если нужна другая
  (Apache-2.0 для патентной защиты, AGPL для copyleft, или закрытая), замените файл
  целиком — это единственное место, где она объявлена.
- Перед публикацией проверьте лицензии зависимостей: Telethon (MIT), httpx (BSD-3),
  pydantic (MIT), loguru (MIT), Jinja2 (BSD-3), tenacity (Apache-2.0), rich (MIT).
- Добавьте `LICENSE` и пометку **AI-generated** в описание репозитория на GitHub.
- **Никогда не коммитить:** `data/`, `logs/`, `*.session`, `*.db`, `configs/settings.yaml`
  (там `chat_id`, `api_id`, `api_hash`), `configs/sessions/*`.
- Убедитесь, что в репозитории нет реальных данных игроков — это персональные данные
  третьих лиц.
- Использование userbot-аккаунтов может нарушать ToS Telegram; ответственность
  на пользователе.
- Программа не предназначена для обмана участников игры; её назначение — разбор
  партий и тренировка. Публикация подсказок живым игрокам во время партии
  нарушает правила клуба.

### Перед переходом Private → Public

Репозиторий сейчас **Private**; перед публикацией пройти чек-лист (шаг 12b):

1. `python scripts/audit_pii.py` — exit 0 (clean). Скрипт read-only: проверяет
   git history (все ветки), текущий index и `tests/fixtures/` на реальные имена
   из `data/dumps/`. Если exit 1 — найти и удалить: `git filter-repo` или новый
   repo (одного удаления из последнего коммита недостаточно — PII остаётся
   в истории).
2. `git log --all --oneline | Select-String "leak|pii|secret"` — пусто
   (PowerShell; в bash: `git log --all --oneline | grep -iE "leak|pii|secret"`).
3. `tests/fixtures/telegram/real_game.json` — синтетический, без реальных имён
   (уже так с 7b).
4. `README.md` §7 — примеры иллюстративные/обезличенные (`player_1`, не реальные
   имена).
5. `configs/settings.example.yaml` — `api_id: null`, `api_hash: null`,
   `chat_id: null` (уже так).
6. `LICENSE` — MIT, `Copyright (c) 2026 EnderMur` (уже).
7. GitHub repo Settings → Danger Zone → Change visibility → Public.

---

_Дата составления: 2026-10-31_
_Обновлено: 2026-11-03 (шаг 12b: документация MVP + Private→Public checklist)_
