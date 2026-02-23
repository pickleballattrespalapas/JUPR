# jupr_app/domain/match_processing.py

from __future__ import annotations

from jupr_app.data.sb_write import sb_update, sb_upsert

import uuid
from datetime import datetime
from typing import Any, Callable

from jupr_app.domain.ratings import calculate_hybrid_elo
from jupr_app.domain.player_activity import (
    build_player_activity_update,
    coerce_utc_datetime,
    max_activity_time,
)
from jupr_app.domain.gamification.badge_queue import enqueue_badge_eval


class NonCanonicalMatchWrite(RuntimeError):
    """Raised when `process_matches` receives non-persisted match payloads."""


class UnknownPlayerId(RuntimeError):
    """Raised when a persisted match references a player id not present in players."""

    def __init__(self, match_id: int | None, player_id: int):
        self.match_id = match_id
        self.player_id = int(player_id)
        super().__init__(f"Unknown player id {self.player_id} for match_id={self.match_id}")


class MissingMatchDatetime(ValueError):
    """Raised when match processing receives rows without deterministic datetimes."""

    def __init__(self, match_id: int | None):
        self.match_id = match_id
        super().__init__(f"Missing match datetime for match_id={self.match_id}")


def process_matches(
    match_list: list[dict[str, Any]],
    *,
    supabase_admin=None,
    supabase=None,
    club_id: str,
    name_to_id: dict[str, int],
    df_players_all,
    df_leagues,
    df_meta,
    sb_retry: Callable | None = None,
    default_k_factor: int = 32,
    min_win_delta_elo: float = 1.0,
    cap_loser_gain_elo: float | None = 16.0,
) -> dict[str, int]:
    """
    - Applies overall rating updates to players table
    - Applies league rating updates to league_ratings table (skips PopUp)
    - Recomputes projections for persisted match rows with snapshot start/end ratings

    match_list rows may contain player ids (int) or names (str). Supported score keys:
      - s1/s2 (preferred; used by live ladder + uploader)
      - score_t1/score_t2 (legacy)
    """

    if supabase_admin is None:
        supabase_admin = supabase

    if not hasattr(supabase_admin, "postgrest"):
        raise RuntimeError("process_matches requires service_role Supabase client")

    # Default retry wrapper: just run the callable
    if sb_retry is None:
        def sb_retry(fn):
            return fn()

    db_matches: list[dict[str, Any]] = []
    overall_updates: dict[int, dict[str, Any]] = {}   # pid -> {"r","w","l","mp"}
    island_updates: dict[tuple[int, str], dict[str, Any]] = {}  # (pid, league) -> {"r","start","w","l","mp"}
    last_game_updates: dict[int, datetime] = {}
    affected_players: set[int] = set()
    known_player_ids = {int(pid) for pid in df_players_all.get("id", []) if str(pid).strip() != ""}

    skipped_incomplete = 0
    skipped_empty = 0
    has_non_popup_match = False
    allowed_context_types = {"league", "ladder", "tournament", "round_robin", "moneyball", "admin"}

    def resolve_context_type(match_row: dict[str, Any], league_name: str) -> str:
        raw_context_type = str(match_row.get("context_type", "") or "").strip().lower()
        if raw_context_type in allowed_context_types:
            return raw_context_type
        if match_row.get("tournament_id") or match_row.get("tournament_game_id"):
            return "tournament"
        if league_name:
            return "league"
        return "admin"

    def parse_uuid(value: Any) -> str | None:
        if value is None:
            return None

        normalized = str(value).strip()
        if not normalized:
            return None

        try:
            return str(uuid.UUID(normalized))
        except (ValueError, TypeError, AttributeError):
            return None

    def resolve_context_id(match_row: dict[str, Any], context_type: str, league_name: str) -> str | None:
        raw_context_id = match_row.get("context_id")
        parsed_context_id = parse_uuid(raw_context_id)
        if parsed_context_id:
            return parsed_context_id
        if context_type == "tournament":
            tournament_id = match_row.get("tournament_id")
            parsed_tournament_id = parse_uuid(tournament_id)
            if parsed_tournament_id:
                return parsed_tournament_id
        return None

    def get_k(league_name: str) -> int:
        if df_meta is None or getattr(df_meta, "empty", True):
            return int(default_k_factor)
        row = df_meta[df_meta["league_name"] == league_name]
        if not row.empty:
            try:
                return int(row.iloc[0].get("k_factor", default_k_factor) or default_k_factor)
            except Exception:
                return int(default_k_factor)
        return int(default_k_factor)

    def get_player_row(pid: int):
        row = df_players_all[df_players_all["id"] == pid]
        if row.empty:
            return None
        return row.iloc[0]

    def ensure_overall_entry(pid: int):
        pid = int(pid)
        if pid in overall_updates:
            return
        pr = get_player_row(pid)
        if pr is None:
            overall_updates[pid] = {"r": 1200.0, "w": 0, "l": 0, "mp": 0}
            return
        overall_updates[pid] = {
            "r": float(pr.get("rating", 1200.0) or 1200.0),
            "w": int(pr.get("wins", 0) or 0),
            "l": int(pr.get("losses", 0) or 0),
            "mp": int(pr.get("matches_played", 0) or 0),
        }

    def get_overall_r(pid: int) -> float:
        pid = int(pid)
        if pid in overall_updates:
            return float(overall_updates[pid]["r"])
        pr = get_player_row(pid)
        if pr is None:
            return 1200.0
        return float(pr.get("rating", 1200.0) or 1200.0)

    def get_island_r(pid: int, league_name: str) -> float:
        key = (int(pid), str(league_name))
        if key in island_updates:
            return float(island_updates[key]["r"])

        if df_leagues is not None and not df_leagues.empty:
            m = df_leagues[
                (df_leagues["player_id"] == int(pid)) &
                (df_leagues["league_name"] == str(league_name))
            ]
            if not m.empty:
                return float(m.iloc[0].get("rating", 1200.0) or 1200.0)

        return get_overall_r(int(pid))

    def ensure_island_entry(pid: int, league_name: str):
        key = (int(pid), str(league_name))
        if key in island_updates:
            return
        start = float(get_island_r(int(pid), str(league_name)))
        island_updates[key] = {"r": start, "start": start, "w": 0, "l": 0, "mp": 0}

    def as_pid(x):
        """Accept int IDs OR numeric strings OR exact names. Returns int player_id or None."""
        if x is None:
            return None
        if isinstance(x, int):
            return int(x)

        s = str(x).strip()
        if not s:
            return None
        if s.isdigit():
            return int(s)

        return name_to_id.get(s)

    def apply_updates(pid: int, d_ov: float, d_isl: float, outcome, is_popup: bool, league_name: str) -> float:
        """
        outcome:
          - True  => this player won
          - False => this player lost
          - None  => tie/unknown (no W/L change)
        """
        pid = int(pid)
        ensure_overall_entry(pid)

        overall_updates[pid]["r"] += float(d_ov)
        overall_updates[pid]["mp"] += 1
        if outcome is True:
            overall_updates[pid]["w"] += 1
        elif outcome is False:
            overall_updates[pid]["l"] += 1

        if not bool(is_popup):
            ensure_island_entry(pid, league_name)
            key = (pid, league_name)
            island_updates[key]["r"] += float(d_isl)
            island_updates[key]["mp"] += 1
            if outcome is True:
                island_updates[key]["w"] += 1
            elif outcome is False:
                island_updates[key]["l"] += 1

        return float(overall_updates[pid]["r"])

    # -------------------------
    # Main match loop
    # -------------------------
    for m in match_list:
        if int(m.get("id") or 0) <= 0:
            raise NonCanonicalMatchWrite(
                "process_matches requires persisted matches with numeric IDs; canonical writes must go through record_match/submit_match first"
            )

        p1 = as_pid(m.get("t1_p1"))
        p2 = as_pid(m.get("t1_p2"))
        p3 = as_pid(m.get("t2_p1"))
        p4 = as_pid(m.get("t2_p2"))

        if any(pid is None for pid in (p1, p2, p3, p4)):
            skipped_incomplete += 1
            continue

        p1, p2, p3, p4 = int(p1), int(p2), int(p3), int(p4)
        match_id = int(m.get("id") or 0)
        for pid in (p1, p2, p3, p4):
            if int(pid) not in known_player_ids:
                raise UnknownPlayerId(match_id=match_id, player_id=int(pid))

        s1 = int(m.get("s1", m.get("score_t1", 0) or 0) or 0)
        s2 = int(m.get("s2", m.get("score_t2", 0) or 0) or 0)
        if (s1 + s2) <= 0:
            skipped_empty += 1
            continue

        league_name = str(m.get("league", "") or "").strip()
        week_tag = str(m.get("week_tag", "") or "")
        match_type = str(m.get("match_type", "") or "")
        is_popup = bool(m.get("is_popup", False)) or (match_type == "PopUp")
        if not is_popup:
            has_non_popup_match = True

        dt_val = m.get("date", None)
        match_dt = coerce_utc_datetime(dt_val)
        if match_dt is None:
            raise MissingMatchDatetime(match_id=match_id)
        dt_val = match_dt.isoformat()

        ro1, ro2, ro3, ro4 = get_overall_r(p1), get_overall_r(p2), get_overall_r(p3), get_overall_r(p4)

        do1, do2 = calculate_hybrid_elo(
            (ro1 + ro2) / 2.0,
            (ro3 + ro4) / 2.0,
            s1,
            s2,
            k_factor=float(default_k_factor),
            min_win_delta=float(min_win_delta_elo),
            cap_loser_gain=cap_loser_gain_elo,
        )

        di1, di2 = 0.0, 0.0
        if not is_popup:
            k_val = get_k(league_name)
            ri1, ri2, ri3, ri4 = (
                get_island_r(p1, league_name),
                get_island_r(p2, league_name),
                get_island_r(p3, league_name),
                get_island_r(p4, league_name),
            )
            di1, di2 = calculate_hybrid_elo(
                (ri1 + ri2) / 2.0,
                (ri3 + ri4) / 2.0,
                s1,
                s2,
                k_factor=float(k_val),
                min_win_delta=float(min_win_delta_elo),
                cap_loser_gain=cap_loser_gain_elo,
            )

        if s1 == s2:
            t1_outcome = None
            t2_outcome = None
        else:
            t1_outcome = (s1 > s2)
            t2_outcome = (s2 > s1)

        end_r1 = apply_updates(p1, do1, di1, t1_outcome, is_popup, league_name)
        end_r2 = apply_updates(p2, do1, di1, t1_outcome, is_popup, league_name)
        end_r3 = apply_updates(p3, do2, di2, t2_outcome, is_popup, league_name)
        end_r4 = apply_updates(p4, do2, di2, t2_outcome, is_popup, league_name)

        for pid in (p1, p2, p3, p4):
            last_game_updates[pid] = max_activity_time(last_game_updates.get(pid), match_dt)
            affected_players.add(int(pid))

        stored_elo_delta = abs(do1) if (t1_outcome is True) else abs(do2)

        db_matches.append(
            {
                "id": int(m.get("id") or 0),
                "club_id": club_id,
                "date": dt_val,
                "league": league_name,
                "t1_p1": p1,
                "t1_p2": p2,
                "t2_p1": p3,
                "t2_p2": p4,
                "score_t1": s1,
                "score_t2": s2,
                "elo_delta": float(stored_elo_delta),
                "match_type": match_type,
                "week_tag": week_tag,
                "t1_p1_r": float(ro1),
                "t1_p2_r": float(ro2),
                "t2_p1_r": float(ro3),
                "t2_p2_r": float(ro4),
                "t1_p1_r_end": float(end_r1),
                "t1_p2_r_end": float(end_r2),
                "t2_p1_r_end": float(end_r3),
                "t2_p2_r_end": float(end_r4),
                "context_type": m.get("context_type"),
                "context_id": m.get("context_id"),
                "tournament_id": m.get("tournament_id"),
                "tournament_game_id": m.get("tournament_game_id"),
            }
        )

    queued_badge_events: list[dict[str, Any]] = []
    if db_matches:
        CHUNK_M = 300
        for i in range(0, len(db_matches), CHUNK_M):
            chunk = db_matches[i : i + CHUNK_M]
            for match_row in chunk:
                context_type = resolve_context_type(match_row, str(match_row.get("league") or "").strip())
                context_id = resolve_context_id(match_row, context_type, str(match_row.get("league") or "").strip())
                queued_badge_events.append({
                    "context_id": str(context_id or "overall"),
                    "match_id": str(match_row.get("id")),
                    "player_ids": [int(match_row["t1_p1"]), int(match_row["t1_p2"]), int(match_row["t2_p1"]), int(match_row["t2_p2"])],
                    "payload": {
                        "match_id": str(match_row.get("id")),
                        "score_t1": int(match_row["score_t1"]),
                        "score_t2": int(match_row["score_t2"]),
                        "t1_p1": int(match_row["t1_p1"]),
                        "t1_p2": int(match_row["t1_p2"]),
                        "t2_p1": int(match_row["t2_p1"]),
                        "t2_p2": int(match_row["t2_p2"]),
                        "t1_p1_r": float(match_row["t1_p1_r"]),
                        "t1_p2_r": float(match_row["t1_p2_r"]),
                        "t2_p1_r": float(match_row["t2_p1_r"]),
                        "t2_p2_r": float(match_row["t2_p2_r"]),
                    },
                })

    # -------------------------
    # Update overall player rows
    # -------------------------
    def update_player_row(row, activity_update: dict):
        pid = int(row["id"])
        payload = {
            "rating": float(row["rating"]),
            "wins": int(row["wins"]),
            "losses": int(row["losses"]),
            "matches_played": int(row["matches_played"]),
        }
        if activity_update:
            payload.update(activity_update)
        res = sb_update(
            supabase_admin,
            "players",
            payload,
            filters={"club_id": club_id, "id": pid},
            derived_from_match_history=True,
        )
        if not res.data:
            raise UnknownPlayerId(match_id=None, player_id=pid)

    for pid, stats in overall_updates.items():
        row = {
            "id": int(pid),
            "rating": float(stats["r"]),
            "wins": int(stats["w"]),
            "losses": int(stats["l"]),
            "matches_played": int(stats["mp"]),
        }
        existing_last_game_at = None
        pr = get_player_row(int(pid))
        if pr is not None:
            existing_last_game_at = pr.get("last_game_at")
        latest_match_at = last_game_updates.get(int(pid))
        activity_update = build_player_activity_update(existing_last_game_at, latest_match_at)
        sb_retry(lambda row=row, activity_update=activity_update: update_player_row(row, activity_update))

    # -------------------------
    # Update league ratings (atomic upsert)
    # -------------------------
    if island_updates:
        for (pid, league_name), stats in island_updates.items():
            payload = {
                "club_id": club_id,
                "player_id": int(pid),
                "league_name": str(league_name),
                "rating": float(stats["r"]),
                "wins": int(stats["w"]),
                "losses": int(stats["l"]),
                "matches_played": int(stats["mp"]),
                "starting_rating": float(stats.get("start", 1200.0)),
                "is_active": True,
                "inactive_at": None,
            }

            sb_retry(lambda payload=payload: sb_upsert(
                supabase_admin,
                "league_ratings",
                payload,
                conflict="club_id,player_id,league_name",
                derived_from_match_history=True,
            ))

    # -------------------------
    # Enqueue per-match badge jobs (facts are updated in badge_worker)
    # -------------------------
    if supabase_admin is not None and queued_badge_events and has_non_popup_match:
        for event in queued_badge_events:
            enqueue_badge_eval(
                supabase_admin,
                club_id=str(club_id),
                event_type="match_recorded",
                player_ids=event["player_ids"],
                context_id=event["context_id"],
                match_id=event["match_id"],
                payload=event["payload"],
            )

    return {
        "inserted": len(db_matches),
        "skipped_incomplete": int(skipped_incomplete),
        "skipped_empty": int(skipped_empty),
    }
