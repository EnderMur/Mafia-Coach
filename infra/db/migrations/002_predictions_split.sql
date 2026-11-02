-- Split predictions: current state (upsert) + append-only history.
PRAGMA foreign_keys = OFF;

CREATE TABLE IF NOT EXISTS predictions_history (
    id INTEGER PRIMARY KEY,
    game_id INTEGER NOT NULL,
    ts TIMESTAMP NOT NULL,
    player_id INTEGER NOT NULL,
    predicted_role TEXT NOT NULL,
    confidence REAL NOT NULL,
    reasoning TEXT,
    correct INTEGER
);

INSERT INTO predictions_history (id, game_id, ts, player_id, predicted_role, confidence, reasoning, correct)
SELECT id, game_id, ts, player_id, predicted_role, confidence, reasoning, correct FROM predictions;

CREATE TABLE predictions_new (
    id INTEGER PRIMARY KEY,
    game_id INTEGER NOT NULL REFERENCES games(id),
    ts TIMESTAMP NOT NULL,
    player_id INTEGER NOT NULL REFERENCES players(id),
    predicted_role TEXT NOT NULL,
    confidence REAL NOT NULL,
    reasoning TEXT,
    correct INTEGER,
    UNIQUE (game_id, player_id)
);

INSERT INTO predictions_new (id, game_id, ts, player_id, predicted_role, confidence, reasoning, correct)
SELECT id, game_id, ts, player_id, predicted_role, confidence, reasoning, correct
FROM predictions p1
WHERE id = (
    SELECT MAX(id) FROM predictions p2
    WHERE p2.game_id = p1.game_id AND p2.player_id = p1.player_id
);

DROP TABLE predictions;
ALTER TABLE predictions_new RENAME TO predictions;
CREATE INDEX IF NOT EXISTS idx_predictions_game_id ON predictions(game_id);
CREATE INDEX IF NOT EXISTS idx_predictions_player_id ON predictions(player_id);
CREATE INDEX IF NOT EXISTS idx_predictions_history_game_id ON predictions_history(game_id);
CREATE INDEX IF NOT EXISTS idx_predictions_history_player_id ON predictions_history(player_id);
PRAGMA foreign_keys = ON;
