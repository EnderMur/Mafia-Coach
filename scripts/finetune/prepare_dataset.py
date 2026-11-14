"""Извлечь публичный контекст SQLite и объединить с ручной разметкой."""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

import jinja2
from loguru import logger

from domain.events import (
    AliveListEvent,
    DeathEvent,
    MessageEvent,
    PhaseChangeEvent,
    RoleRevealEvent,
    VoteEvent,
)
from domain.game_state import GameState
from domain.roles import load_roles
from features.analyzer.context_builder import build_context, to_llm_messages
from infra.llm.schemas import AnalysisResponse
from scripts.finetune.common import digest, load_config, read_jsonl, validate_messages

# Только явно разрешённые поля публичных событий. role_reveal в штатном
# scanner создаётся из сообщения игрового бота; provenance в БД не хранится.
PUBLIC: dict[str, tuple[Any, tuple[str, ...]]] = {
    "message": (MessageEvent, ("author_id", "author_name", "text")),
    "vote": (VoteEvent, ("voter_id", "voter_name", "target_id", "target_name")),
    "death": (DeathEvent, ("victim_id", "victim_name", "cause")),
    "role_reveal": (RoleRevealEvent, ("player_id", "player_name", "role_name_ru")),
    "phase_change": (PhaseChangeEvent, ("from_phase", "to_phase", "day_number")),
    "alive_list": (AliveListEvent, ("alive_player_names", "role_count", "total_players")),
}
ROOT = Path(__file__).resolve().parents[2]


def build_record(conn: sqlite3.Connection, annotation: dict[str, Any]) -> dict[str, Any]:
    """Собрать снимок до указанного события включительно, игнорируя скрытые поля."""
    registry = load_roles(ROOT / "configs/roles.yaml")
    game_id, cutoff = int(annotation["game_id"]), int(annotation["cutoff_event_id"])
    rows = conn.execute(
        "SELECT id, ts, kind, payload_json FROM events WHERE game_id=? AND id<=?",
        (game_id, cutoff),
    ).fetchall()
    cutoff_row = next((row for row in rows if row[0] == cutoff), None)
    if cutoff_row is None or cutoff_row[2] not in PUBLIC:
        raise ValueError(
            f"Игра {game_id}: граница event_id={cutoff} должна указывать "
            "на публичное событие партии"
        )

    timed_rows = []
    for event_id, ts, kind, payload_json in rows:
        try:
            current_ts = datetime.fromisoformat(ts)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Игра {game_id}: некорректный timestamp события event_id={event_id}: {ts!r}"
            ) from error
        if current_ts.tzinfo is None or current_ts.utcoffset() is None:
            raise ValueError(
                f"Игра {game_id}: у события event_id={event_id} timestamp без часового пояса"
            )
        timed_rows.append((current_ts, event_id, ts, kind, payload_json))

    # Порядок SQLite по id не гарантирует хронологию. Сравнение datetime
    # учитывает смещения часовых поясов; event_id даёт устойчивый tie-breaker.
    timed_rows.sort(key=lambda row: (row[0], row[1]))
    for previous, current in itertools.pairwise(timed_rows):
        if current[0] < previous[0]:
            raise ValueError(
                f"Игра {game_id}: нарушена хронология между событиями "
                f"event_id={previous[1]} и event_id={current[1]}"
            )

    state = GameState(game_id=game_id, registry=registry)
    previous_ts = None
    previous_event_id = None
    for current_ts, event_id, ts, kind, payload_json in timed_rows:
        if kind == "end_game":
            raise ValueError(
                f"Игра {game_id}: снимок после завершения игры (event_id={event_id}) запрещён"
            )
        if kind not in PUBLIC:
            continue
        cls, allowed = PUBLIC[kind]
        payload = json.loads(payload_json)
        public = {key: payload[key] for key in allowed if key in payload}
        if kind == "message":
            author_id = public.get("author_id")
            author = next(
                (player for player in state.players() if player.player_id == author_id), None
            )
            if author is not None and not author.alive:
                logger.debug(
                    "Пропущено сообщение после смерти игрока: game_id={}, event_id={}, author_id={}",
                    game_id,
                    event_id,
                    author_id,
                )
                # Пропуск не изменяет previous_ts и не передаёт событие агрегату.
                continue
        if previous_ts is not None and current_ts < previous_ts:
            raise ValueError(
                f"Игра {game_id}: нарушена хронология между событиями "
                f"event_id={previous_event_id} и event_id={event_id}"
            )
        previous_ts = current_ts
        previous_event_id = event_id
        event = cls.model_validate({"ts": ts, "game_id": game_id, "payload": public})
        if kind == "phase_change" and public.get("to_phase") == "end":
            raise ValueError(
                f"Игра {game_id}: снимок после завершения игры (event_id={event_id}) запрещён"
            )
        if kind == "message":
            state.register_player(public["author_id"], public["author_name"])
        elif kind == "role_reveal":
            state.register_player(public["player_id"], public["player_name"])
        elif kind == "death":
            state.register_player(public["victim_id"], public["victim_name"])
        elif kind == "vote":
            state.register_player(public["voter_id"], public["voter_name"])
            if public.get("target_id") is not None and public.get("target_name"):
                state.register_player(public["target_id"], public["target_name"])
        state.apply(event)
    context = build_context(game_state=state, registry=registry)
    # Внутренний ID партии не является сообщением чата.
    context = context.model_copy(update={"game_id": 0})
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(ROOT / "infra/llm/prompts"), autoescape=False
    )
    messages = to_llm_messages(context, registry=registry, jinja_env=env)
    response = AnalysisResponse.model_validate(annotation["response"])
    ids = {p.player_id: p for p in context.players}
    known_roles = {r.name_ru for r in registry.all_roles()}
    seen = set()
    for prediction in response.analysis:
        if prediction.player_id not in ids or prediction.player_id in seen:
            raise ValueError("Разметка содержит неизвестного или повторного игрока")
        seen.add(prediction.player_id)
        if prediction.player_name != ids[prediction.player_id].display_name:
            raise ValueError("Имя игрока в разметке не соответствует контексту")
        if prediction.suspected_role is not None and prediction.suspected_role not in known_roles:
            raise ValueError("Роль не входит в справочник")
        if (
            ids[prediction.player_id].known_role_name_ru is not None
            and prediction.suspected_role is not None
            and prediction.suspected_role != ids[prediction.player_id].known_role_name_ru
        ):
            raise ValueError("Разметка противоречит публичному раскрытию")
        if prediction.alternative and prediction.alternative.role not in known_roles:
            raise ValueError("Альтернативная роль не входит в справочник")
        if (prediction.suspected_role is None) != (
            prediction.player_id in context.silent_player_ids
        ):
            raise ValueError("Разметка противоречит правилам для молчащих игроков")
    if not response.analysis:
        raise ValueError("Пустая разметка не используется")
    if context.role_count:
        alive = {p.player_id for p in context.players if p.alive}
        if seen != alive or Counter(p.suspected_role for p in response.analysis) != Counter(
            context.role_count
        ):
            raise ValueError("Разметка не соответствует составу живых игроков")
    messages.append({"role": "assistant", "content": response.model_dump_json(exclude_none=False)})
    validate_messages(messages)
    return {"messages": messages}


def distribution(items: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    roles: Counter[str] = Counter()
    replies: Counter[str] = Counter()
    for item in items:
        answer = AnalysisResponse.model_validate_json(item["record"]["messages"][-1]["content"])
        roles.update(p.suspected_role or "нет роли" for p in answer.analysis)
        replies[item["reply_type"]] += 1
    return {"roles": dict(roles), "reply_types": dict(replies)}


def check_balance(stats: dict[str, dict[str, int]], limit: float) -> None:
    if not 0 < limit <= 0.70:
        raise ValueError("Допустимая доля класса не выше 0.70")
    for name, counts in stats.items():
        total = sum(counts.values())
        if not total or max(counts.values()) / total > limit:
            raise ValueError(f"Дисбаланс {name}: {counts}, предел {limit}")


def balance_train(items: list[dict[str, Any]], limit: float, seed: int) -> list[dict[str, Any]]:
    """Уменьшить доминирующие классы, не создавая и не копируя примеры."""
    selected = list(items)
    random.Random(seed).shuffle(selected)
    original_roles = set(distribution(selected)["roles"])
    original_types = set(distribution(selected)["reply_types"])
    if len(original_roles) < 2 or len(original_types) < 2:
        return selected

    def score(rows: list[dict[str, Any]]) -> float:
        stats = distribution(rows)
        return max(max(values.values()) / sum(values.values()) for values in stats.values())

    while selected and score(selected) > limit:
        current = score(selected)
        best = None
        best_score = current
        for index in range(len(selected)):
            candidate = selected[:index] + selected[index + 1 :]
            if not candidate:
                continue
            stats = distribution(candidate)
            if (
                set(stats["roles"]) != original_roles
                or set(stats["reply_types"]) != original_types
                or len({x["format"] for x in candidate}) < 2
                or len({x["outcome"] for x in candidate}) < 2
            ):
                continue
            new_score = score(candidate)
            if new_score < best_score:
                best, best_score = index, new_score
        if best is None:
            break
        selected.pop(best)
    return selected


def prepare(config: dict[str, Any], annotations: list[dict[str, Any]]) -> dict[str, Any]:
    out = Path(config["dataset_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for name in ("train.jsonl", "val.jsonl", "test.jsonl", "report.json"):
        (out / name).unlink(missing_ok=True)
    splits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    community_splits: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        for community in config[f"{split_name}_communities"]:
            if community in community_splits:
                raise ValueError("Сообщество встречается в нескольких выборках")
            community_splits[community] = split_name
    game_splits: dict[tuple[str, int], str] = {}
    game_meta: dict[tuple[str, int], tuple[str, str, str]] = {}
    seen_cutoffs = set()
    for a in annotations:
        if a.get("reviewed") is not True:
            raise ValueError("Каждая разметка должна быть проверена человеком")
        for field in ("community", "format", "outcome", "reply_type", "db"):
            if not isinstance(a.get(field), str) or not a[field].strip():
                raise ValueError(f"Не задано поле {field}")
        if a["outcome"] not in {"победа", "поражение", "ничья"}:
            raise ValueError("Исход должен быть победа, поражение или ничья")
        split = community_splits.get(a["community"])
        if split is None:
            raise ValueError("Сообщество не назначено выборке")
        key = (str(Path(a["db"]).resolve()), int(a["game_id"]))
        if key in game_splits and game_splits[key] != split:
            raise ValueError("Партия пересекает выборки")
        meta = (a["community"], a["format"], a["outcome"])
        if key in game_meta and game_meta[key] != meta:
            raise ValueError("Противоречивая разметка партии")
        game_splits[key], game_meta[key] = split, meta
        cutoff_key = (*key, int(a["cutoff_event_id"]))
        if cutoff_key in seen_cutoffs:
            raise ValueError("Повторная разметка снимка")
        seen_cutoffs.add(cutoff_key)
        with closing(sqlite3.connect(Path(key[0]).as_uri() + "?mode=ro", uri=True)) as conn:
            record = build_record(conn, a)
        splits[split].append(
            {
                "record": record,
                "game_key": key,
                "reply_type": a["reply_type"],
                "format": a["format"],
                "outcome": a["outcome"],
            }
        )
    if (
        len({x["format"] for x in splits["train"]}) < 2
        or not ({x["outcome"] for x in splits["train"]} & {"поражение", "ничья"})
        or "победа" not in {x["outcome"] for x in splits["train"]}
    ):
        raise ValueError("Train должен содержать минимум два формата и два исхода")
    before = distribution(splits["train"])
    count_before = len(splits["train"])
    splits["train"] = balance_train(
        splits["train"], float(config["max_class_fraction"]), int(config.get("seed", 42))
    )
    report: dict[str, Any] = {
        "splits": {},
        "train_before_balancing": before,
        "train_removed": count_before - len(splits["train"]),
    }
    errors = []
    for name in ("train", "val", "test"):
        items = splits[name]
        stats = distribution(items)
        report["splits"][name] = {
            "examples": len(items),
            "games": len({x["game_key"] for x in items}),
            "distribution": stats,
            "formats": dict(Counter(x["format"] for x in items)),
            "outcomes": dict(Counter(x["outcome"] for x in items)),
        }
        try:
            check_balance(stats, float(config["max_class_fraction"]))
        except ValueError as error:
            errors.append(f"{name}: {error}")
    out = Path(config["dataset_dir"])
    out.mkdir(parents=True, exist_ok=True)
    # При ошибке удалить устаревшие данные и не оставлять готовый датасет.
    for name in ("train", "val", "test"):
        (out / f"{name}.jsonl").unlink(missing_ok=True)
    if errors:
        report["errors"] = errors
    else:
        for name in ("train", "val", "test"):
            path = out / f"{name}.jsonl"
            path.write_text(
                "".join(json.dumps(x["record"], ensure_ascii=False) + "\n" for x in splits[name]),
                encoding="utf-8",
            )
            report["splits"][name]["sha256"] = digest(path)
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Распределение: {}", report)
    if errors:
        raise ValueError("; ".join(errors) + "; добавьте недостающие классы или отберите снимки")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Подготовка датасета из проверенной разметки и SQLite"
    )
    parser.add_argument("--config", default="configs/finetune.example.yaml")
    parser.add_argument("--annotations", required=True, help="JSONL с проверенной разметкой")
    args = parser.parse_args()
    prepare(load_config(args.config), read_jsonl(Path(args.annotations)))


if __name__ == "__main__":
    main()
