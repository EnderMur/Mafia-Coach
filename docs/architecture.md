# Architecture — Mafia Coach

> MVP (v1), шаг 12b. Документ описывает **реальное поведение** пайплайна и служит
> картой для dogfooding. Источник истины — код и тесты: при расхождении сначала
> правится код, затем этот документ. Смежные документы:
> [`llm_contract.md`](llm_contract.md) — контракт LLM,
> [`parser_spec.md`](parser_spec.md) — спецификация парсера.

## 1. Слои

```text
[Telegram userbot (Telethon)] — read-only
        │ events.NewMessage
        ▼
infra/telegram/ (TelegramClientWrapper)
        │ raw message (text, sender_id, sender, date)
        ▼
features/parser/ (classify_message, parse_bot_message, speech)
        │ MessageKind / BotMessageParse
        ▼
features/scanner/bot_resolver.py (resolve_bot_message)
        │ list[AnyEvent]
        ▼
domain/game_state.py (GameState.apply — validate-then-append)
        │ state
        ▼
features/scanner/scanner.py (MessageScanner: flush на trigger)
        │
        ▼
features/analyzer/ (build_context → to_llm_messages → LLM)
        │
        ▼
infra/llm/ (LMStudioClient.analyze_stream)
        │ AnalysisResponse
        ▼
features/analyzer/analyzer.py (validate + persist)
        │
        ▼
infra/db/repositories/ (SQLite: events, predictions, game_players)
        │
        ▼
features/ui/ (WindowHost → subprocess: stats_window, reasoning_window)
```

`MessageScanner` — единственное stateful-звено `features/`: он владеет буфером,
debounce-таймером и маппингом `name_to_id`. Оркестратор (`app/orchestrator.py`)
владеет ресурсами (БД, Telegram, UI) и подключает их к scanner через callback'и
(`on_event`, `on_player_registered`, `on_reasoning`, `on_flush_complete`).
Парсер и резолвер — чистые функции от входных данных.

## 2. Правила слоёв

- `domain/` — pure, без infra/app/telethon. Только stdlib + pydantic + loguru;
  `roles.py` дополнительно использует PyYAML для загрузки `configs/roles.yaml`
  (единственный I/O в домене — чтение каталога ролей при старте).
- `infra/` — внешний мир: Telegram (Telethon), LLM (httpx → LM Studio), SQLite,
  filesystem, настройка loguru.
- `features/` — прикладные сценарии, могут зависеть от domain + infra.
  Чистые модули (`parser/`, `scanner/bot_resolver.py`, `scanner/trigger_policy.py`,
  `analyzer/context_builder.py`, `analyzer/heuristic_hints.py`) infra не импортируют
  и тестируются без I/O.
- `app/` — сборка: `bootstrap.py` (конфиг, миграции, healthcheck LLM),
  `orchestrator.py` (command loop, callback'и, persistence).
- Зависимости направлены вниз: `app → features → domain`; `infra` подключается
  в `features`/`app`. `domain` не знает ни о ком.
- Никаких `print()` в domain/features/infra — только loguru с `channel=...`
  (CLI-печать разрешена лишь скриптам в `scripts/`).

## 3. Поток события (end-to-end)

1. `TelegramClientWrapper.add_message_handler` → `scanner._on_event`.
   Подписан только `events.NewMessage` (правки игнорируются и логируются);
   дедупликация по `(chat_id, message_id)`; свои сообщения отсеиваются
   (`ignore_own_messages`). Handler никогда не бросает исключение наружу.
2. `_process_event`:
   - `_register_player` (если sender_id новый) → `on_player_registered` callback →
     orchestrator `_persist_player` → `players.upsert_player` +
     `game_players.add_player` (порядок важен: FK `predictions.player_id`).
   - `classify_message` (7a): `BOT_MESSAGE` / `PLAYER_SPEECH` / `IRRELEVANT`.
   - `BOT_MESSAGE`: `parse_bot_message` (7b) → `resolve_bot_message` (9a) →
     `GameState.apply` для каждого event. `on_event` callback → orchestrator
     `_persist_event` → `events.insert_event`. `AliveListEvent` дополнительно
     публикует name-only регистрации (`_persist_alive_placeholders`).
   - `PLAYER_SPEECH`: `MessageEvent` → `GameState.apply` → буфер.
   - Отклонённый доменом event (например, «already dead») не попадает ни в
     буфер, ни в БД — только лог; одно плохое сообщение не ломает партию.
3. `_maybe_flush` (9a): `TriggerPolicy.should_force_now` / `should_force_on_size`
   → `_flush`; в `batch_debounce` — (пере)запуск debounce-таймера.
4. `_flush`: re-entrancy guard `_flushing` (повторный триггер во время анализа
   отбрасывается, события копятся в буфере); `GameState.current_phase` ==
   REGISTRATION/END? skip (буфер очищается). Иначе `GameAnalyzer.analyze(persist=True)`.
5. `GameAnalyzer.analyze`:
   - `build_context` (8a): `LLMContext` (players, recent_messages, recent_deaths,
     role_count, total_players). Placeholder-игроки (`player_id <= 0`) в контекст
     не попадают.
   - `compute_hints` (8a): правила клуба (`domain/rules.py`), сейчас один hint —
     `early_lynch_no_argument` (weight −0.5).
   - `to_llm_messages`: `system.jinja` + `analyze.jinja`.
   - `LLMClient.analyze_stream`: общий бюджет HTTP-запросов `max_retries ≤ 3`;
     `_TransientEOF` — один повтор с увеличенным `max_tokens`; при отказе сервера
     от `json_schema` — fallback на `response_format=text` в том же бюджете.
   - `validate_response`: clamp confidence / drop неизвестных игроков и ролей /
     normalize пары confidence+alternative (см. `llm_contract.md`).
   - `predictions.upsert_prediction` + `predictions_history.insert_history`
     (через `asyncio.to_thread`).
   - `on_reasoning` → orchestrator `_send_reasoning` → UI;
     `on_flush_complete` → orchestrator `_send_stats` → UI.
   - Лог: `flush ok: N predictions, Xs` (или `flush failed: ...`).
6. `EndGameEvent` → `GameState.apply(_apply_end_game)` → `final_role` /
   `is_winner` для каждого игрока из отчёта; фаза → END. Финальный отчёт —
   ground truth: расхождение с более ранним `ROLE_REVEAL` логируется как WARNING.
7. `_persist_event(EndGameEvent)` → schedule retro: задержка 5 с
   (`_RETRO_DELAY_SEC`), затем ожидание `scanner.is_flushing == False`
   (до 60 с) — финальный flush должен успеть записать предсказания.
8. `_run_retro`: `_persist_player` для всех (включая END_GAME placeholders) →
   `game_players.set_final_role` → `retro_analyze` → `predictions.correct`
   (`mark_correct`). Лог: `game N retro: total=… correct=… wrong=… unknown=…`.
   Повторный запуск идемпотентен.

## 4. Backfill при /attach

1. `run()` запускает backfill фоновой задачей (таймаут 300 с), чтобы не блокировать
   command loop. `scanner.backfill(limit=5000)`: `list_recent_messages` → reverse
   (хронологический порядок).
2. Найти последний `GAME_START` (regex «Город ветров начинает», только сообщения
   бота, markdown снят).
3. Если после него есть `END_GAME` → игра завершена, **no-op** (return 0).
4. Иначе — применить все сообщения после `GAME_START` через `_process_event`
   (промежуточные flush подавлены: `_flushing = True` на время replay).
   Один `_flush_force` в конце (bypass phase guard) — если буфер непуст и
   настроен `on_flush_complete`.
5. `GAME_START` не найден → warning, return 0. `TelegramFloodWaitError` →
   sleep `wait_seconds` и одна повторная попытка.

## 5. GAME_START → reset

`PhaseChangeEvent(to_phase=REGISTRATION)` → `GameState.apply`:

- `_players = {}`, `_day_number = 0`, `_role_count = None`, `_total_players = None`.
- Log `INFO "new game detected"` (канал `game_state`).
- Scanner очищает `_registered_ids`, чтобы известные Telegram-пользователи могли
  зарегистрироваться в новой партии заново.
- Запись в `games` при этом НЕ создаётся: сброс происходит внутри текущего
  `game_id` (см. §7).

## 6. Триггеры

- `batch_debounce` — основной режим для живой игры: flush через
  `batch_debounce_sec` после последнего сообщения. Phase events (death,
  phase_change, vote, role_reveal) — force flush немедленно.
- `every_message`: flush каждое сообщение.
- `queue`: flush по размеру буфера (`batch_max_messages`; MVP-поведение,
  настоящий queue-worker — v2). Это значение по умолчанию в
  `configs/settings.example.yaml` и `app/config.py`.

## 7. Жизненный цикл партии (game_id)

- Bootstrap: `chat_id` задан → переиспользуется последняя незавершённая игра
  (`games.ended_at IS NULL`) или создаётся новая; `chat_id: null` →
  `game_id = 0` (placeholder до первого `/attach`).
- `/attach`: при `game_id == 0` создаёт игру и новый `GameState`; повторный
  attach в v1 сохраняет текущую партию (state не делится по чатам).
- `/reset`: новая запись `games` + новый `GameState` + перезапуск scanner.
- `GAME_START`: сброс state внутри того же `game_id` (§5).
- `END_GAME`: фаза → END, retro (§3). `games.finish_game` в MVP автоматически
  не вызывается — `ended_at` остаётся NULL; факт завершения фиксируется фазой
  и retro-отчётом.

## 8. Персистентность и конкурентность

- SQLite через stdlib `sqlite3` (`check_same_thread=False`); репозитории
  синхронные. В async-коде — только `await asyncio.to_thread(...)`.
- Orchestrator сериализует все обращения к БД одним `_db_lock`
  (один connection не используется из двух потоков одновременно).
- Таблицы: `players`, `games`, `game_players` (final_role, survived, is_winner),
  `events`, `predictions`, `predictions_history`, `player_style`
  (миграции `001`–`004`).
- `predictions` — текущее состояние (upsert на game_id+player_id);
  `predictions_history` — append-only, retro-метки не получает.
- Сбой LLM/БД/UI никогда не останавливает чтение чата: каждое звено —
  защитная граница с логом (`analysis_failed`, `persist event failed`, …).

## 9. UI и IPC

- `WindowHost` поднимает два subprocess-окна (`CREATE_NEW_CONSOLE` на Windows),
  обмен — JSON-строки через stdin (`IPCEnvelope`); окна output-only.
- Stats: строки из `predictions`, сортировка по confidence, порог
  `min_confidence_to_display`, максимум 20 строк; placeholder-игроки пропускаются.
  Пустой кадр отправляется и при отсутствии предсказаний — чтобы очистить
  устаревшие строки.
- Reasoning: только `delta.reasoning_content` / `delta.reasoning`; у non-thinking
  модели канал пуст → оркестратор подставляет
  `(streaming reasoning unavailable — non-thinking model)`. IPC-кадр обрезается
  до 4096 символов.

## 10. Команды оркестратора

| Команда | Действие |
|---|---|
| `/attach [id]` | выбор чата: числовой id или интерактивный список групп; затем фоновый backfill |
| `/status` | Game #id, phase, day, alive/total, топ-5 предсказаний, размер буфера |
| `/reset` | новая партия в текущем чате (новый game_id) |
| `/retro` | ручной запуск ретроспективы (идемпотентен) |
| `/quit` | graceful-остановка (EOF stdin эквивалентен) |

## 11. Инварианты MVP

- Telegram — только чтение: ни одного исходящего сообщения/голоса
  (проверяется read-only аудитом в тестах).
- Все применённые события пишутся в `events` немедленно, независимо от
  батчинга и ошибок LLM.
- Placeholder-игроки не попадают в `LLMContext` и Stats Window, но участвуют
  в retro (`game_players.final_role`).
- Один scanner на чат, один asyncio event loop; объекты не thread-safe.
- Язык UI и промптов — только русский.
