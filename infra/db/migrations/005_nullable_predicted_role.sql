-- Allow ``NULL`` predicted roles: a player without messages and votes may have
-- no role hypothesis (``suspected_role = null, confidence = 0.0``) instead of
-- a hard-coded placeholder role.
-- SQLite cannot drop a NOT NULL constraint in place, so both tables are
-- rebuilt; existing rows keep their values and remain readable.
CREATE TABLE predictions_new (
    id              INTEGER PRIMARY KEY,
    game_id         INTEGER NOT NULL REFERENCES games(id),
    ts              TIMESTAMP NOT NULL,
    player_id       INTEGER NOT NULL REFERENCES players(id),
    predicted_role  TEXT,                      -- NULL = нет данных по игроку
    confidence      REAL NOT NULL,
    reasoning       TEXT,
    correct         INTEGER,
    UNIQUE (game_id, player_id)
);

INSERT INTO predictions_new (id, game_id, ts, player_id, predicted_role, confidence, reasoning, correct)
SELECT id, game_id, ts, player_id, predicted_role, confidence, reasoning, correct
FROM predictions;

DROP TABLE predictions;

ALTER TABLE predictions_new RENAME TO predictions;

CREATE INDEX IF NOT EXISTS idx_predictions_game_id ON predictions(game_id);
CREATE INDEX IF NOT EXISTS idx_predictions_player_id ON predictions(player_id);

CREATE TABLE predictions_history_new (
    id              INTEGER PRIMARY KEY,
    game_id         INTEGER NOT NULL,
    ts              TIMESTAMP NOT NULL,
    player_id       INTEGER NOT NULL,
    predicted_role  TEXT,                      -- NULL = нет данных по игроку
    confidence      REAL NOT NULL,
    reasoning       TEXT,
    correct         INTEGER
);

INSERT INTO predictions_history_new (id, game_id, ts, player_id, predicted_role, confidence, reasoning, correct)
SELECT id, game_id, ts, player_id, predicted_role, confidence, reasoning, correct
FROM predictions_history;

DROP TABLE predictions_history;

ALTER TABLE predictions_history_new RENAME TO predictions_history;

CREATE INDEX IF NOT EXISTS idx_predictions_history_game_id ON predictions_history(game_id);
CREATE INDEX IF NOT EXISTS idx_predictions_history_player_id ON predictions_history(player_id);
