CREATE UNIQUE INDEX IF NOT EXISTS idx_game_players_pair
    ON game_players(game_id, player_id);
