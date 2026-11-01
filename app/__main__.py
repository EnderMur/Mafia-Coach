"""Package entry point for ``python -m app``."""

from __future__ import annotations

import asyncio
import sys

from app.bootstrap import bootstrap
from app.orchestrator import Orchestrator


def main() -> None:
    """Run the application and report fatal startup failures to stderr."""
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        # Ctrl+C is a graceful stop, equivalent to /quit.
        pass
    except Exception as error:  # noqa: BLE001 — fatal top-level reporting boundary
        # FATAL is the top-level fallback before logging may be initialized.
        print(f"FATAL: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)


async def _run() -> None:
    """Bootstrap the application, run the command loop, and always shut down."""
    result = await bootstrap()
    orchestrator = Orchestrator(bootstrap_result=result)
    try:
        await orchestrator.run()
    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    main()
