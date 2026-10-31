-- Игроки (глобально, сквозь все партии)
CREATE TABLE players (
    id              INTEGER PRIMARY KEY,      -- Telegram user id
    username        TEXT,                     -- текущий username
    first_seen_at   TIMESTAMP,
    last_seen_at    TIMESTAMP,
    notes           TEXT                      -- ручные/авто метки: "блефует", "агрессивный"…
);

-- Партии
CREATE TABLE games (
    id              INTEGER PRIMARY KEY,
    chat_id         INTEGER,
    started_at      TIMESTAMP,
    ended_at        TIMESTAMP,
    player_count    INTEGER,
    winner_faction  TEXT                       -- town|mafia|neutral
);

-- Участие игрока в партии
CREATE TABLE game_players (
    id              INTEGER PRIMARY KEY,
    game_id         INTEGER REFERENCES games(id),
    player_id       INTEGER REFERENCES players(id),
    final_role      TEXT,                      -- из сообщения бота
    survived        INTEGER                    -- 0/1
);

-- События (единый поток)
CREATE TABLE events (
    id              INTEGER PRIMARY KEY,
    game_id         INTEGER REFERENCES games(id),
    ts              TIMESTAMP,
    kind            TEXT,                      -- message|death|vote|role_reveal|phase_change
    actor_id        INTEGER,                   -- если применимо
    target_id       INTEGER,                   -- если применимо
    phase           TEXT,
    day_number      INTEGER,
    payload_json    TEXT                       -- текст, метаданные
);

-- Предсказания ИИ
CREATE TABLE predictions (
    id              INTEGER PRIMARY KEY,
    game_id         INTEGER REFERENCES games(id),
    ts              TIMESTAMP,
    player_id       INTEGER REFERENCES players(id),
    predicted_role  TEXT,
    confidence      REAL,                      -- 0..1
    reasoning       TEXT,                      -- модель reasoning (для ретроспективы)
    correct         INTEGER                    -- 0/1/null — заполняется в конце партии
);

-- Профиль стиля речи
CREATE TABLE player_style (
    player_id       INTEGER REFERENCES players(id),
    tag             TEXT,                      -- "частые отмазки", "пассив"
    weight          REAL,                      -- уверенность/частота
    evidence_json   TEXT,
    PRIMARY KEY (player_id, tag)
);

CREATE INDEX IF NOT EXISTS idx_events_game_id ON events(game_id);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind);
CREATE INDEX IF NOT EXISTS idx_predictions_game_id ON predictions(game_id);
CREATE INDEX IF NOT EXISTS idx_predictions_player_id ON predictions(player_id);
CREATE INDEX IF NOT EXISTS idx_game_players_game_id ON game_players(game_id);
