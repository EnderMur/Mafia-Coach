"""Run the UI output windows manually, without bootstrap, Telegram, LLM, or DB."""

from __future__ import annotations

import argparse
import asyncio
import sys

from features.ui.window_host import WindowHost


async def _run(stats: bool, reasoning: bool) -> None:
    """Keep the requested windows alive until Enter, EOF, or interruption."""
    host = WindowHost(
        stats_enabled=stats,
        reasoning_enabled=reasoning,
    )
    await host.start()
    print(
        f"Windows started: stats={host.stats_running} "
        f"(pid={host.stats_pid}), "
        f"reasoning={host.reasoning_running} "
        f"(pid={host.reasoning_pid})",
        flush=True,
    )
    print("Press Ctrl+C or Enter to stop.", flush=True)
    try:
        await asyncio.to_thread(sys.stdin.readline)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        await host.shutdown()
        print("Windows stopped.", flush=True)


def main(argv: list[str] | None = None) -> int:
    """Parse window selection and run the standalone UI host."""
    parser = argparse.ArgumentParser(
        description="Run two UI windows standalone (no pipeline).",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="disable stats window",
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="disable reasoning window",
    )
    args = parser.parse_args(argv)
    if args.no_stats and args.no_reasoning:
        print("Nothing to run: both windows disabled.", flush=True)
        return 1
    try:
        asyncio.run(
            _run(
                stats=not args.no_stats,
                reasoning=not args.no_reasoning,
            )
        )
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
