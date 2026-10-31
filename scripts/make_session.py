"""Одноразовое создание Telethon-сессии. Читает настройки из configs/settings.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from telethon.sync import TelegramClient  # type: ignore[import-untyped]


def main() -> None:
    """Create and authorize a local Telethon session file interactively."""
    settings_path = Path("configs/settings.yaml")
    if not settings_path.is_file():
        settings_path = Path("configs/settings.example.yaml")
        print(
            "WARNING: configs/settings.yaml не найден, читаем configs/settings.example.yaml. "
            "Скопируйте шаблон в configs/settings.yaml и укажите api_id и api_hash."
        )

    with settings_path.open("r", encoding="utf-8") as handle:
        loaded: Any = yaml.safe_load(handle)

    cfg: dict[str, Any] = loaded if isinstance(loaded, dict) else {}
    raw_tg: Any = cfg.get("telegram", {})
    t_cfg: dict[str, Any] = raw_tg if isinstance(raw_tg, dict) else {}

    api_id = t_cfg.get("api_id")
    api_hash = t_cfg.get("api_hash")
    session_file = t_cfg.get("session_file") or "data/sessions/mafia_coach"

    if not isinstance(api_id, int) or api_id <= 0 or not isinstance(api_hash, str) or not api_hash:
        raise ValueError(
            "Заполните api_id и api_hash в configs/settings.yaml перед созданием сессии"
        )

    session_path = Path(str(session_file))
    session_path.parent.mkdir(parents=True, exist_ok=True)

    with TelegramClient(str(session_path), api_id, api_hash) as client:
        me = client.get_me()
        display_name = (me.username or me.first_name) if me is not None else "unknown"
        me_id = me.id if me is not None else "unknown"
        print("OK, авторизован как:", display_name, "id =", me_id)
        print(
            "ВНИМАНИЕ: файл .session даёт полный доступ к аккаунту. "
            "Храните его локально, не передавайте третьим лицам и не добавляйте в git."
        )


if __name__ == "__main__":
    main()
