from __future__ import annotations

from typing import Any


def as_player_id(value: Any, name_to_id: dict[str, int]) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    s = str(value).strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    return name_to_id.get(s)


def extract_scores(match: dict[str, Any]) -> tuple[int, int]:
    s1 = int(match.get("s1", match.get("score_t1", 0) or 0) or 0)
    s2 = int(match.get("s2", match.get("score_t2", 0) or 0) or 0)
    return s1, s2


def normalize_rating_scope(match: dict[str, Any]) -> str:
    rating_scope = str(match.get("rating_scope", "") or "").strip().lower()
    if rating_scope not in {"overall_only", "unrated"}:
        return ""
    return rating_scope


def collect_seed_candidates(match_list: list[dict[str, Any]], name_to_id: dict[str, int]) -> tuple[set[int], set[str]]:
    player_ids: set[int] = set()
    leagues: set[str] = set()
    for match in match_list:
        for col in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"):
            pid = as_player_id(match.get(col), name_to_id)
            if pid is not None:
                player_ids.add(int(pid))
        league = str(match.get("league", "") or "").strip()
        if league:
            leagues.add(league)
    return player_ids, leagues
