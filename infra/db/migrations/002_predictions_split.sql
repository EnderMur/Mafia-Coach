-- Split predictions: current state (upsert) + append-only history.
-- Rows lacking a game, player, role, or confidence cannot form a prediction.
-- Preserve otherwise valid legacy rows with missing timestamps using migration time.
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
SELECT id, game_id, COALESCE(ts, CURRENT_TIMESTAMP), player_id, predicted_role, confidence, reasoning, correct
FROM predictions
WHERE game_id IS NOT NULL AND player_id IS NOT NULL
  AND predicted_role IS NOT NULL AND confidence IS NOT NULL;

-- No table in the v1 schema references predictions, so renaming and dropping
-- the old table works with foreign key enforcement enabled throughout.
ALTER TABLE predictions RENAME TO predictions_old;

CREATE TABLE predictions (
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

INSERT INTO predictions (id, game_id, ts, player_id, predicted_role, confidence, reasoning, correct)
SELECT id, game_id, COALESCE(ts, CURRENT_TIMESTAMP), player_id, predicted_role, confidence, reasoning, correct
FROM predictions_old p1
WHERE p1.game_id IS NOT NULL AND p1.player_id IS NOT NULL
  AND p1.predicted_role IS NOT NULL AND p1.confidence IS NOT NULL
  AND p1.id = (
    SELECT MAX(id) FROM predictions_old p2
    WHERE p2.game_id = p1.game_id AND p2.player_id = p1.player_id
      AND p2.predicted_role IS NOT NULL AND p2.confidence IS NOT NULL
);

DROP TABLE predictions_old;
CREATE INDEX IF NOT EXISTS idx_predictions_game_id ON predictions(game_id);
CREATE INDEX IF NOT EXISTS idx_predictions_player_id ON predictions(player_id);
CREATE INDEX IF NOT EXISTS idx_predictions_history_game_id ON predictions_history(game_id);
CREATE INDEX IF NOT EXISTS idx_predictions_history_player_id ON predictions_history(player_id);
