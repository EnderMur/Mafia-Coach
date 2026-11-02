-- Ground truth for the fine-tune dataset: did this player win the game?
-- 1/0/NULL — NULL means "unknown" (player not mentioned in the bot's final
-- report, or the report was never received). SQLite adds the column without a
-- DEFAULT, so pre-existing rows keep NULL.
ALTER TABLE game_players ADD COLUMN is_winner INTEGER;
