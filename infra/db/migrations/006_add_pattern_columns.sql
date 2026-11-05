-- Add per-player reason and pattern metadata to predictions tables.
-- ``pattern_ids`` and ``pattern_descriptions`` are stored as JSON text
-- (arrays of strings) so the stats window can render the “Паттерн” column.
-- SQLite supports ALTER TABLE ADD COLUMN, so no table rebuild is needed.

ALTER TABLE predictions ADD COLUMN reason TEXT DEFAULT '';
ALTER TABLE predictions ADD COLUMN pattern_ids TEXT DEFAULT '[]';
ALTER TABLE predictions ADD COLUMN pattern_descriptions TEXT DEFAULT '[]';

ALTER TABLE predictions_history ADD COLUMN reason TEXT DEFAULT '';
ALTER TABLE predictions_history ADD COLUMN pattern_ids TEXT DEFAULT '[]';
ALTER TABLE predictions_history ADD COLUMN pattern_descriptions TEXT DEFAULT '[]';
