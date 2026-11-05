# LLM Contract — Mafia Coach

## 1. Endpoint

The configured endpoint is `POST {llm.base_url}/chat/completions`, using the
OpenAI-compatible local model server. The configured model identifier and
request parameters are supplied by application configuration.

Streaming is enabled only for the reasoning channel. The final structured
response is assembled after generation completes.

## 2. AnalysisResponse schema (v1)

| Field | Type | Required | Constraints |
|---|---|---:|---|
| `analysis` | list of `PlayerPrediction` | yes | May be empty |
| `analysis[].player_id` | integer | yes | Must belong to the current game |
| `analysis[].player_name` | string | yes | Non-empty display name |
| `analysis[].suspected_role` | string or null | yes | Canonical configured role name, or `null` when the player has no messages and no votes |
| `analysis[].confidence` | float | yes | Inclusive range 0.0–1.0; must be `0.0` when `suspected_role` is null |
| `analysis[].reason` | string | yes | Brief 1–2 sentence explanation of the role in Russian; `"нет данных"` when `suspected_role` is null |
| `analysis[].alternative` | `AlternativePrediction` or null | no | Optional second hypothesis |
| `analysis[].alternative.role` | string | yes when present | Canonical configured role name |
| `analysis[].alternative.confidence` | float | yes when present | Inclusive range 0.0–1.0 |
| `analysis[].reasons` | list of strings | yes | Between one and five entries |
| `analysis[].pattern_ids` | list of strings | yes | Club pattern IDs supporting the role (e.g. `["#2"]`); empty when no pattern was used |
| `analysis[].pattern_descriptions` | list of strings | yes | Human-readable descriptions of the patterns above; same length as `pattern_ids` |
| Primary and alternative confidence total | float | when alternative exists | At most 1.0 |
| `game_read` | string | yes | Non-empty summary |
| `reasoning` | string or null | no | Full explanation; empty, null, or absent is permitted |

The response and nested prediction models are immutable and reject undeclared
fields. The `analysis` collection may be empty when no usable predictions remain.

## 3. Contextual validation

JSON is parsed and validated against the Pydantic schema. Schema errors are
returned to the caller for retry handling.

1. Confidence values outside the supported range are clamped with a warning.
2. Unknown role names are passed to the optional alias resolver. If the primary
   remains unresolved but the alternative resolves, its canonical role and
   confidence become primary, `alternative` is cleared, and INFO is logged.
   Otherwise an unresolved primary drops the prediction with a warning (the
   `неизвестно` placeholder remains silent); an unresolved alternative is
   removed while retaining a valid primary.
3. Predictions for player identifiers outside the active game are dropped.
4. If primary and alternative confidence total exceeds one beyond floating
   point tolerance, both values are normalized proportionally and a warning is
   returned.
5. `suspected_role` is `null` only for players without messages and votes;
   any other null role drops the prediction with a warning.
6. `suspected_role` is `null` implies `confidence` is `0.0` and `reason` is
   `"нет данных"`; `pattern_ids` and `pattern_descriptions` must have equal
   length.
7. Contextual issues are warnings; structurally invalid JSON data is an error.

## 4. Errors and retries

| Situation | Expected handling |
|---|---|
| Connection refused | Report a connection error and retry on a later event |
| Model not found | Report a non-retryable model error |
| Request timeout | Retry according to configured policy |
| Rate limit or server error | Exponential backoff, bounded by configured retries |
| Invalid JSON | Request JSON-only output once, then report an LLM error |
| Empty analysis | Accept as a valid response |

The transport and retry implementation are separate from these static schemas.

## 5. Prompts

`system.jinja` receives the configured Russian role names and the generated JSON
Schema. It establishes the analysis task, requires Russian-language output, and
requires the response to conform to `AnalysisResponse`.

`analyze.jinja` receives game and phase identifiers, player snapshots, recent
messages, earlier predictions, and an optional summary of club heuristics. When
the bot sent an ALIVE_LIST, it also renders an explicit composition section:
`role_count` (canonical `name_ru` → count), `role_slots` (sum), `role_kinds`
(distinct roles), `total_players`, a mismatch warning, the bot's alive nicknames
and the subset backed by synthetic negative ids. The model is told to return
exactly `role_slots` predictions — one per alive player — and never exceed a
role counter. The «Прошлые предсказания (якорь)» block forbids changing a role
without a new argument. The template contains no hard-coded role catalogue and
no hard-coded «Мирный житель — 1 из 30» claim.

Role notes are filtered to roles actually present in the composition, so a small
local model is not distracted by 21 role descriptions when only 14 roles are in
the game. `GameAnalyzer` additionally loads the previous predictions from
SQLite, reverts unjustified role flips and fills missing composition slots only
when role quotas sum to the number of alive players. A mismatch is logged and
reported in `AnalysisResult.warnings`; missing slots remain unfilled.

## Reasoning channel

Reasoning for the Reasoning Window and `logs/reasoning.log` comes from
`delta.reasoning_content` or `delta.reasoning`. If no reasoning was streamed,
the analyzer falls back to the JSON `reasoning` field. Raw `delta.content` is
not forwarded to the reasoning channel. If neither source has reasoning,
the channel remains empty.

The HTTP-request budget for one `analyze()` or `analyze_stream()` call is limited
to `max_retries` (≤ 3) and includes schema fallback and validation retry. A 404
does not consume retry budget.

## 6. Version notes

This document describes the v1 structured response contract (v1.1 with the
`reason`, `pattern_ids`, `pattern_descriptions` fields and nullable
`suspected_role`). Context building, request transport, and streaming
orchestration are implemented in later layers.

## Known limitations / TODO

- `features/analyzer/context_builder.py::DEFAULT_RESPONSE_SCHEMA_JSON` is an
  embedded copy of `AnalysisResponse.model_json_schema()`. It must not import
  `infra/` because `features/analyzer` pure modules must remain infrastructure-free.
  Drift is caught by `test_default_response_schema_matches_contract` in
  `tests/unit/features/analyzer/`. **Whenever `AnalysisResponse` changes,
  update `DEFAULT_RESPONSE_SCHEMA_JSON`.**
- `predictions.reasoning` хранит полный `AnalysisResponse.reasoning` без
  усечения. Если он пуст, равен `null` или отсутствует, сохраняются
  `PlayerPrediction.reasons` соответствующего игрока через `\n`.
  `predictions_history.reasoning` отдельно сохраняет причины игрока через `\n`.
  Канал `logs/reasoning.log` использует ограничение `reasoning_max_chars`.

  Например, при `reasoning = "Общий анализ партии"` и `reasons = ["аргумент"]`
  текущая запись содержит `"Общий анализ партии"`, а история — `"аргумент"`.
  При пустом или отсутствующем `reasoning` текущая запись содержит `"аргумент"`.
- `analysis_failed` is only a log marker (`channel="llm"`, level `ERROR`), with
  no dedicated table. If a mechanism to retry an analysis later is needed, add a
  `status` column to `predictions` in step 12 or later.

### predictions split

`predictions` — текущее состояние (одна строка на game_id + player_id,
upsert при каждом flush). Используется /status, Stats Window и retro.
`predictions_history` — append-only (все flush), для будущего анализа
эволюции предсказаний и fine-tune. История не получает retro-метки correct.

## Async DB

SQLite через stdlib `sqlite3` с `check_same_thread=False`.
Репозитории остаются **синхронными** (тестируются без asyncio).
В async-контексте вызывающий оборачивает sync-метод в
`asyncio.to_thread(...)`:

```python
# analyzer.py
await asyncio.to_thread(
    self._predictions_repo.upsert_prediction,
    game_id=game_id, player_id=player_id, ...
)
```

Это работает, потому что:

- Все записи сериализованы через один asyncio event loop.
- `asyncio.to_thread` создаёт воркер-поток per-call (по умолчанию
  `ThreadPoolExecutor` с `~min(32, os.cpu_count()+4)` воркерами) —
  параллельные вызовы из разных корутин могут идти в разные
  потоки, но по факту у нас один caller (orchestrator).
- `check_same_thread=False` разрешает использование connection
  из разных потоков; serialization — ответственность caller'а.

Правило: в async-коде никогда не вызывать sync repo-методы
напрямую. Всегда `await asyncio.to_thread(...)`.
