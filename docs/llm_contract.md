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
| `analysis[].suspected_role` | string | yes | Canonical configured role name |
| `analysis[].confidence` | float | yes | Inclusive range 0.0–1.0 |
| `analysis[].alternative` | `AlternativePrediction` or null | no | Optional second hypothesis |
| `analysis[].alternative.role` | string | yes when present | Canonical configured role name |
| `analysis[].alternative.confidence` | float | yes when present | Inclusive range 0.0–1.0 |
| `analysis[].reasons` | list of strings | yes | Between one and five entries |
| Primary and alternative confidence total | float | when alternative exists | At most 1.0 |
| `game_read` | string | yes | Non-empty summary |
| `reasoning` | string | yes | Non-empty explanation |

The response and nested prediction models are immutable and reject undeclared
fields. The `analysis` collection may be empty when no usable predictions remain.

## 3. Contextual validation

JSON is parsed and validated against the Pydantic schema. Schema errors are
returned to the caller for retry handling.

1. Confidence values outside the supported range are clamped with a warning.
2. Unknown role names are passed to the optional alias resolver. Unresolved
   primary roles cause that prediction to be dropped; unresolved alternatives
   are removed while retaining the primary prediction.
3. Predictions for player identifiers outside the active game are dropped.
4. If primary and alternative confidence total exceeds one beyond floating
   point tolerance, both values are normalized proportionally and a warning is
   returned.
5. Contextual issues are warnings; structurally invalid JSON data is an error.

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
messages, earlier predictions, and an optional summary of club heuristics. The
template presents those values in labeled context sections and contains no
hard-coded role catalogue.

## Reasoning channel

Reasoning for the Reasoning Window and `logs/reasoning.log` is taken ONLY from
`delta.reasoning_content` (vLLM/Qwen-style) or `delta.reasoning` (some LM Studio
builds). `delta.content` is used exclusively to accumulate the final JSON and is
NEVER passed to the reasoning channel. If the loaded model does not emit
dedicated reasoning (a non-thinking model), the channel remains empty; this is
expected behavior, not an error.

The HTTP-request budget for one `analyze()` or `analyze_stream()` call is limited
to `max_retries` (≤ 3) and includes schema fallback and validation retry. A 404
does not consume retry budget.

## 6. Version notes

This document describes the v1 structured response contract. Context building,
request transport, and streaming orchestration are implemented in later layers.

## Known limitations / TODO

- `features/analyzer/context_builder.py::DEFAULT_RESPONSE_SCHEMA_JSON` is an
  embedded copy of `AnalysisResponse.model_json_schema()`. It must not import
  `infra/` because `features/analyzer` pure modules must remain infrastructure-free.
  Drift is caught by `test_default_response_schema_matches_contract` in
  `tests/unit/features/analyzer/`. **Whenever `AnalysisResponse` changes,
  update `DEFAULT_RESPONSE_SCHEMA_JSON`.**
- `predictions.reasoning` in the database stores `PlayerPrediction.reasons`
  joined with `\n`, not the model's full chain of thought. The full chain of thought
  is written to `logs/reasoning.log` through `channel="reasoning"` (one line per
  cycle, truncated to `reasoning_max_chars`).
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
