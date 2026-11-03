# Parser Spec — Mafia Coach

> Спецификация парсера сообщений игрового бота (шаги 7a/7b/9a, актуально на 12b).
> Реализация: `features/parser/classifier.py`, `features/parser/bot_messages.py`,
> `features/scanner/bot_resolver.py`. Все модули чистые: без I/O и async.
> Архитектура пайплайна — [`architecture.md`](architecture.md).

## 1. classify_message (features/parser/classifier.py)

| Вход | Выход | Условие |
|---|---|---|
| text=None или "" | IRRELEVANT | пустое |
| from_id == bot_id | BOT_MESSAGE | от бота |
| from_id is None | IRRELEVANT | анонимный |
| len(text.strip()) < 3 | IRRELEVANT | слишком короткое |
| иначе | PLAYER_SPEECH | речь игрока |

Правила проверяются строго в этом порядке: сообщение бота может быть коротким
(«Ночь»), поэтому проверка `bot_id` идёт до проверки длины. Порог длины —
`min_speech_length` (default 3). `MessageKind` — str-enum:
`bot_message` / `player_speech` / `irrelevant`. Некорректный `bot_id`
(не int, в т.ч. bool) → `TypeError`; отрицательный `min_speech_length` →
`ValueError`.

## 2. parse_bot_message (features/parser/bot_messages.py)

Порядок проверок (важен!):

1. END_GAME — «Игра окончена!» + Победители/Другие пользователи/Игра длилась.
2. LAST_WORDS — «Кто-то из жителей слышал, как X кричал перед смертью:…».
3. ALIVE_LIST — «Живые игроки:» / «Живые:» + «N. name» + «Из них:» + «Всего: N».
   Роли из «Из них:» приводятся к каноническим `name_ru` из `roles.yaml`
   (регистр и пробелы не важны); счётчики summed, сумма слотов и число разных
   ролей считает `context_builder`, а не парсер. Поддерживаются `N.` / `N)` /
   `N:` / `N\.` и «Живые: user_1, user_2» в одну строку.
4. REGISTRATION — «Регистрация открыта», «набор в игру».
5. GAME_START — «Город ветров начинает игру».
6. DEATH — «Кто-то был убит», «Сегодня повешен X».
7. PHASE_CHANGE (vote) — «Голосование окончено», «Пришло время».
8. VOTE — «X голосовал(а) за: Y» (multi-line).
9. PROTECTION_USED — «Кто-то потратил защиту».
10. VOTE_ABSTAIN — «X решил никого не вешать».
11. GIFT — «X подарил 💎N к Y».
12. ROLE_REVEAL — «X был Роль» (грамматическое).
13. ROLE_ACTION — «Роль сделала X» (action notifications).
14. PHASE_CHANGE (night/day) — «🌙 Наступает ночь», «🌝 Утро: N»
    (только короткие / с emoji).
15. SYSTEM — «Автоматический старт игры», «Недостаточно игроков».
16. UNKNOWN — fallback.

Пустой/None вход → `None` (не UNKNOWN). Парсер никогда не бросает исключение
на произвольной строке. Порядок load-bearing: финальный отчёт содержит
нумерованные секции и не должен путаться с ALIVE_LIST; «🌙 Кто-то был убит» —
это DEATH, а не смена фазы; «использовал защиту от линчевания» — не смерть.

Детали шаблонов:

- GAME_START: также «начинается (новая) игра», «старт игры».
- DEATH: `death_cause` ∈ `lynch` («сегодня повешен», «голосованием повешен»,
  «не выдержал… повесил») / `night_kill` («кто-то был убит», «этой ночью убили»);
  нераспознанная причина → `unknown` уже в резолвере.
- PHASE_CHANGE (night/day): guard длины — сообщения ≥ 500 символов без ведущего
  🌙/ не считаются сменой фазы; vote-объявления проверяются без guard.
- VOTE: основной шаблон «X голосовал(а) за: Y» по всем строкам; fallback —
  «проголосовал против»; итоги без пар («результаты голосования») → kind=VOTE,
  confidence 0.9, событий не производит.
- ROLE_REVEAL: три формы — «X был Роль» (conf 1.0), «X — Роль» (1.0/0.9),
  «Роль X» (0.8); грамматическая эвристика без резолва роли — 0.5, `role_id=None`
  (резолвер такое пропускает).
- ROLE_ACTION: ведущий emoji роли → conf 0.9; имя роли + глагол действия → 0.7.
- END_GAME: секции «Победители:» / «Другие пользователи:», записи «N. имя - Роль»,
  «Игра длилась: N минут»; роли остаются как написал бот (`name_ru`).

## 3. BotMessageKind (15 kinds)

| Kind | Payload fields | Event |
|---|---|---|
| REGISTRATION | — | [] (informational) |
| GAME_START | — | PhaseChangeEvent(to=REGISTRATION) |
| PHASE_CHANGE | to_phase, day_number | PhaseChangeEvent |
| DEATH | player_name, death_cause | DeathEvent |
| ROLE_REVEAL | player_name, role_id | RoleRevealEvent |
| ROLE_ACTION | role_id, target_name | [] |
| VOTE | votes [(voter, target)], voter_name, target_name | VoteEvent × N |
| LAST_WORDS | player_name, last_words | MessageEvent |
| GIFT | player_name, target_name | [] |
| ALIVE_LIST | alive_player_names, role_count (канонические `name_ru`), total_players | AliveListEvent |
| PROTECTION_USED | — | [] |
| VOTE_ABSTAIN | player_name | [] |
| SYSTEM | — | [] |
| END_GAME | winners [(name, role)], losers, duration_minutes | EndGameEvent |
| UNKNOWN | — | [] |

`BotMessageParse` — frozen pydantic-модель (`extra="forbid"`); имена остаются
строками (резолв в id — работа резолвера и `GameState`). Каждое сообщение несёт
`raw_text` и `confidence`: 1.0 для точных шаблонов, 0.5–0.9 для эвристик,
0.0 для UNKNOWN. `LAST_WORDS` без текста → `MessageEvent` с «(no words)».

## 4. Name resolution (bot_resolver.py)

Порядок:

1. `name_to_id.get(_strip_markdown(name))`.
2. `name_to_id.get(_strip_role_prefix(name, registry))`.
3. Если оба fail — warning, event пропускается.

`_strip_markdown`: `[Yulios](tg://user?id=1)` → `Yulios`, `**bold**` → `bold`.
`_strip_role_prefix`: `"👨🏼 Мирный Житель ^‿^"` → `"^‿^"`, `"( ^‿^ )"` → `"^‿^"`.

`name_to_id` строится scanner'ом по первым наблюдениям: первое написание ника
выигрывает (бот пишет ник таким, каким видел его при регистрации). Для vote
неразрешённая цель не отбрасывает событие — `target_id=None`, пишется warning;
неразрешённый voter — пропускает пару целиком.

## 5. _placeholder_player_id (game_state.py)

- `"user_8764210791"` → `8764210791` (реальный Telegram id).
- Иное имя → sha256, отрицательный int64 (`-<hash>`): первые 8 байт digest как
  unsigned big-endian → `-(n % 2**62) - 1` (стабильно между запусками).
- Placeholder не идёт в `LLMContext` (filter `player_id > 0`).
- Placeholder идёт в `game_players` для retro (final_role из END_GAME).

## 6. C3 (реальный дамп)

На 3169 bot messages (10k реальных):

- classified: 3129 (98.7%).
- unknown: 16 (0.5%).
- ROLE_REVEAL: 8, все conf=1.0 (0 ложных).
- false_positives (player messages): 0.5%.

## 7. Edge cases (закрыты)

- Markdown links: `[name](url)` → `name`.
- ZWJ-эмодзи: `👨‍💼` = `👨 + ZWJ + 💼` (ведущие emoji снимаются целиком).
- Русский падеж: `"Мафией"` → `"мафия"` (instrumental suffix strip: -ией, -ей,
  -ой, -ом, -ем, -ым, -им, -ью…); «Комиссаром Каттани» → prefix-match по реестру.
- Multi-line vote: несколько пар `X голосовал за Y`.
- Multi-line `Из них:` (бот переносит строку) — переносы схлопываются, счётчики
  «Роль - N» суммируются.
- Имена с декорациями: «🇷🇺ℹ | name | 💋» → сегмент с максимумом «настоящих»
  букв/цифр (модификаторы вроде `ᵗᵏ` в клубном теге не считаются); максимум
  3 токена имени, stop-слова («и», «был», «с», …) обрезают захват.
- В `ALIVE_LIST` имена сохраняются как написал бот (`"| Дмитрий"`,
  `"🕷️ᵗᵏ| Shinichiro🌝🖤"`); сопоставление с Telegram-никами делает
  `domain/names.py::name_match_keys` (NFKC, снятие маркеров/декораций,
  casefold). Это же сравнение используется в `GameState.apply`, резолвере
  событий и whitelist Stats Window. Placeholder-id намеренно остаются в
  `LLMContext.players`, чтобы все живые получили слот роли.
- «Кто-то», «убит», «повешен» и прочие служебные слова не считаются именами
  игроков (`_NOISE_NAME_TOKENS`, `_NON_ROLE_WORDS`).

## 8. Инварианты парсера

- `parse_bot_message` никогда не бросает исключение на любой строке
  (включая emoji-only и 1000+ символов); пустой вход → `None`.
- UNKNOWN сохраняется (conf 0.0) для статистики C3, событий не производит.
- Парсер не знает id: только текст → смысл; id резолвит scanner/GameState.
- Приоритеты веток зафиксированы тестами (`tests/unit/features/parser/`);
  менять порядок без обновления тестов и этого документа нельзя.
