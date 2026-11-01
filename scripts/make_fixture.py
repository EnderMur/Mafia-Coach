"""Create an anonymised test fixture from a raw chat dump.

Reads ``data/dumps/raw_dump.json`` (or any dump specified via ``--input``),
replaces player names/ids with ``user_N`` tokens, selects a deterministic
subset of messages, and writes the result to
``tests/fixtures/telegram/real_game.json`` (or ``--output``).

The script does NOT commit anything to git — it only writes the fixture
file.  Review the fixture manually for leftover real names before committing.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any


_DEFAULT_OUTPUT = Path("tests/fixtures/telegram/real_game.json")


def _build_name_mapping(
    messages: list[dict[str, Any]],
    bot_id: int | str,
) -> dict[str, str]:
    """Map real names/ids to anonymised ``user_N`` tokens.

    Iterates messages in chronological order; the first unseen
    ``sender_name`` / ``sender_id`` gets the next sequential number.
    Bot identifiers are excluded.
    """
    name_map: dict[str, str] = {}
    counter = 0
    for msg in messages:
        sid = msg.get("sender_id")
        sname = msg.get("sender_name", "")

        # Skip the bot
        if sid == bot_id:
            continue

        for key in (str(sname), str(sid)):
            if key and key not in name_map:
                counter += 1
                anon = f"user_{counter}"
                name_map[key] = anon

    return name_map


def _anonymise_text(
    text: str,
    name_map: dict[str, str],
) -> str:
    """Replace real names in ``text`` using word-boundary regex.

    Names are processed longest-first so that ``Иван Петров`` is
    replaced before ``Иван``.
    """
    sorted_names = sorted(name_map.keys(), key=len, reverse=True)
    for real_name in sorted_names:
        if not real_name:
            continue
        anon = name_map[real_name]
        pattern = rf"(?<![\w]){re.escape(real_name)}(?![\w])"
        text = re.sub(pattern, anon, text)
    return text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Create anonymised test fixture from raw chat dump.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to raw dump (e.g. data/dumps/raw_dump.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Output fixture JSON path",
    )
    parser.add_argument(
        "--max-bot",
        type=int,
        default=100,
        help="Maximum bot messages in sample",
    )
    parser.add_argument(
        "--max-player",
        type=int,
        default=50,
        help="Maximum player messages in sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show counts without writing file",
    )
    parser.add_argument(
        "--names-extra",
        type=Path,
        default=None,
        help="Optional JSON file with extra {real: anon} name mappings",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Load dump, anonymise, sample, and write fixture."""
    args = parse_args(argv)

    input_path: Path = args.input
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 1

    with input_path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)

    bot_id = raw.get("bot_id")
    bot_name = raw.get("bot_name", "")
    messages: list[dict[str, Any]] = raw.get("messages", [])

    if not messages:
        print("ERROR: no messages in dump", file=sys.stderr)
        return 1

    if bot_id is None:
        print("ERROR: no bot_id in dump", file=sys.stderr)
        return 1

    # Build name mapping (excludes bot)
    name_map = _build_name_mapping(messages, bot_id)

    # Load extra names if provided
    if args.names_extra and args.names_extra.exists():
        with args.names_extra.open(encoding="utf-8") as f:
            extras: dict[str, str] = json.load(f)
        for real, anon in extras.items():
            if real not in name_map:
                name_map[real] = anon

    # Sort mapping by name length (longest first) for replacement
    # (already done inside _anonymise_text)

    # Split into bot and player messages
    bot_msgs = [m for m in messages if m.get("sender_id") == bot_id]
    player_msgs = [m for m in messages if m.get("sender_id") != bot_id]

    # Deterministic sampling
    rng_bot = random.Random(args.seed)
    rng_player = random.Random(args.seed + 1)

    selected_bot = rng_bot.sample(bot_msgs, min(args.max_bot, len(bot_msgs)))
    selected_player = rng_player.sample(player_msgs, min(args.max_player, len(player_msgs)))

    # Combine and sort by timestamp
    selected = sorted(selected_bot + selected_player, key=lambda m: m.get("ts", ""))

    if args.dry_run:
        print(f"Source messages: {len(messages)}")
        print(f"Bot messages:    {len(bot_msgs)} → sample {len(selected_bot)}")
        print(f"Player messages: {len(player_msgs)} → sample {len(selected_player)}")
        print(f"Total selected:  {len(selected)}")
        print(f"Name mappings:   {len(name_map)}")
        print("(dry-run: no file written)")
        return 0

    # Anonymise each message
    anon_messages: list[dict[str, Any]] = []
    for msg in selected:
        sid = msg.get("sender_id")
        sname = msg.get("sender_name", "")
        text = msg.get("text", "")

        anon_text = _anonymise_text(text, name_map)

        if sid == bot_id:
            anon_sid: int | str = bot_id
            anon_sname = bot_name
        else:
            anon_sid = name_map.get(str(sname), name_map.get(str(sid), "user_?"))
            anon_sname = anon_sid

        anon_messages.append(
            {
                "ts": msg.get("ts", ""),
                "sender_id": anon_sid,
                "sender_name": anon_sname,
                "text": anon_text,
            }
        )

    # Build output
    output_data: dict[str, Any] = {
        "anonymized": True,
        "bot_id": bot_id,
        "bot_name": bot_name,
        "sample_seed": args.seed,
        "source_message_count": len(messages),
        "selected_bot_count": len(selected_bot),
        "selected_player_count": len(selected_player),
        "messages": anon_messages,
    }

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {len(anon_messages)} messages to {output_path}")
    print(f"  bot messages:    {len(selected_bot)}")
    print(f"  player messages: {len(selected_player)}")
    print(f"  name mappings:   {len(name_map)}")
    print()
    print(
        "WARNING: check the output file manually for leftover real names. "
        "Search for: 'был', 'была', 'повешен' — and verify names look like user_N."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
