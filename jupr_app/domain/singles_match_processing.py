from __future__ import annotations

from datetime import datetime, timezone
import logging
import math
from typing import Any, Callable

from jupr_app.domain.matches import (
    as_player_id,
    build_match_row,
    compute_outcomes,
    compute_team_deltas,
    extract_scores,
    insert_match_chunks_with_rating_scope_fallback,
    is_popup_match,
    normalize_rating_scope,
    should_update_island,
)
from jupr_app.domain.notifications.player_profile_update_repo import queue_player_updates_for_affected_subscribers
from jupr_app.domain.match_processing import (
    build_active_league_metadata_expectations,
    has_managed_league_metadata,
)
from jupr_app.domain.player_activity import build_player_activity_update, coerce_utc_datetime, max_activity_time

logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float = 1200.0) -> float:
    if value in (None, ""):
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_positive_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return max(0.0, float(value))
    except Exception:
        return 0.0


def _finite_float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _is_singles_match(match: dict[str, Any]) -> bool:
    return str(match.get("match_format") or match.get("format") or "").strip().lower() == "singles"


def _fallback_player_frame_lookup(df_players_all: Any, pid: int) -> dict[str, Any] | None:
    try:
        row = df_players_all[df_players_all["id"] == int(pid)]
        if row.empty:
            return None
        return dict(row.iloc[0])
    except Exception:
        return None


def _league_frame_rows(
    df_leagues: Any,
    *,
    player_id: int,
    league_name: str,
) -> list[dict[str, Any]]:
    if isinstance(df_leagues, list):
        return [
            dict(row)
            for row in df_leagues
            if str(row.get("player_id")) == str(int(player_id))
            and str(row.get("league_name") or "").casefold()
            == str(league_name).casefold()
        ]
    if df_leagues is None or getattr(df_leagues, "empty", True):
        return []
    try:
        selected = df_leagues[
            (df_leagues["player_id"].astype(str) == str(int(player_id)))
            & (
                df_leagues["league_name"].astype(str).str.casefold()
                == str(league_name).casefold()
            )
        ]
    except Exception:
        return []
    return [dict(row) for _, row in selected.iterrows()]


def _fetch_league_rows(
    supabase: Any,
    *,
    club_id: str,
    player_ids: set[int],
) -> list[dict[str, Any]]:
    if not player_ids:
        return []
    try:
        rows = (
            supabase.table("league_ratings")
            .select(
                "id,player_id,league_name,rating,starting_rating,wins,losses,"
                "matches_played,is_active,inactive_at"
            )
            .eq("club_id", str(club_id))
            .in_("player_id", sorted(int(pid) for pid in player_ids))
            .execute()
            .data
            or []
        )
    except Exception as exc:
        raise RuntimeError(
            "Authoritative league roster baselines are unavailable; "
            "no singles match was written."
        ) from exc
    return [dict(row) for row in rows]


def _fetch_player_rows(supabase: Any, *, club_id: str, player_ids: set[int]) -> dict[int, dict[str, Any]]:
    if not player_ids:
        return {}
    try:
        rows = (
            supabase.table("players")
            .select(
                "id,name,rating,wins,losses,matches_played,last_game_at,"
                "inactive_at,active,singles_rating,singles_wins,singles_losses,"
                "singles_matches_played,singles_last_game_at,"
                "singles_replay_baseline"
            )
            .eq("club_id", str(club_id))
            .in_("id", sorted(int(pid) for pid in player_ids))
            .execute()
            .data
            or []
        )
    except Exception as exc:
        raise RuntimeError(
            "Authoritative singles player baselines are unavailable; "
            "no singles match was written."
        ) from exc
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            result[int(row.get("id"))] = dict(row)
        except Exception:
            continue
    return result


def _singles_replay_baseline(row: dict[str, Any] | None) -> dict[str, Any]:
    baseline = (row or {}).get("singles_replay_baseline")
    return dict(baseline) if isinstance(baseline, dict) else {}


def _seed_singles_rating(row: dict[str, Any] | None) -> float:
    row = row or {}
    if row.get("singles_rating") not in (None, ""):
        raw_rating = row.get("singles_rating")
    else:
        baseline = _singles_replay_baseline(row)
        raw_rating = baseline.get("rating")
        if raw_rating in (None, ""):
            raise RuntimeError(
                "A preserved singles replay baseline is required before the "
                "first managed singles match."
            )
    try:
        rating = float(raw_rating)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("The singles rating seed is invalid.") from exc
    if isinstance(raw_rating, bool) or not math.isfinite(rating):
        raise RuntimeError("The singles rating seed is invalid.")
    return rating


def _seed_stat(row: dict[str, Any] | None, key: str) -> int:
    try:
        return int(float((row or {}).get(key) or 0))
    except Exception:
        return 0


def _update_player_singles_row(supabase: Any, *, club_id: str, player_id: int, payload: dict[str, Any], sb_retry: Callable) -> None:
    def _update(update_payload: dict[str, Any]):
        return (
            supabase.table("players")
            .update(update_payload)
            .eq("club_id", str(club_id))
            .eq("id", int(player_id))
            .execute()
        )

    result = sb_retry(lambda: _update(payload))
    updated_ids = {
        int(row.get("id"))
        for row in (getattr(result, "data", None) or [])
        if row.get("id") is not None
    }
    if updated_ids != {int(player_id)}:
        raise RuntimeError(
            "Singles player aggregate update did not affect exactly one "
            f"authoritative row for player {int(player_id)}."
        )


def process_singles_matches(
    match_list: list[dict[str, Any]],
    *,
    supabase: Any,
    club_id: str,
    name_to_id: dict[str, int],
    df_players_all: Any = None,
    df_leagues: Any = None,
    df_meta: Any = None,
    sb_retry: Callable | None = None,
    default_k_factor: int = 32,
    min_win_delta_elo: float = 1.0,
    cap_loser_gain_elo: float | None = 16.0,
    build_write_plan_only: bool = False,
) -> dict[str, Any]:
    """Persist and rate one-on-one singles matches using players.singles_* fields.

    Singles rating is intentionally separate from doubles/overall JUPR. It writes official
    match-history rows with match_format='singles', but only updates singles rating counters.
    """

    if sb_retry is None:
        def sb_retry(fn):
            return fn()

    prepared_singles_rows: list[tuple[datetime, int, dict[str, Any]]] = []
    for row_index, row in enumerate(match_list or []):
        if not _is_singles_match(row):
            continue
        match_dt = coerce_utc_datetime(row.get("date")) or datetime.now(timezone.utc)
        prepared_singles_rows.append((match_dt, row_index, row))
    prepared_singles_rows.sort(key=lambda item: (item[0], item[1]))

    db_matches: list[dict[str, Any]] = []
    player_updates: dict[int, dict[str, Any]] = {}
    league_updates: dict[tuple[int, str], dict[str, Any]] = {}
    last_game_updates: dict[int, datetime] = {}
    affected_players: set[int] = set()
    successful_match_dates: list[str] = []
    managed_league_names: set[str] = set()

    skipped_incomplete = 0
    skipped_empty = 0
    skipped_unrated = 0
    bonus_match_count = 0
    bonus_player_elo_total = 0.0

    candidate_ids: set[int] = set()
    for _match_dt, _row_index, match in prepared_singles_rows:
        for key in ("t1_p1", "t2_p1"):
            pid = as_player_id(match.get(key), name_to_id)
            if pid is not None:
                candidate_ids.add(int(pid))
    live_players = _fetch_player_rows(supabase, club_id=str(club_id), player_ids=candidate_ids)
    missing_player_ids = sorted(candidate_ids - set(live_players))
    if missing_player_ids:
        raise RuntimeError(
            "Authoritative singles player rows are incomplete; "
            f"no singles match was written: {missing_player_ids[:10]}"
        )

    # The legacy direct-write path has no atomic compare-and-swap wrapper, so
    # refresh its league rows immediately before calculating final values. The
    # atomic path deliberately uses the caller's snapshot as its expected CAS
    # state instead.
    effective_df_leagues = df_leagues
    if not build_write_plan_only:
        effective_df_leagues = _fetch_league_rows(
            supabase,
            club_id=str(club_id),
            player_ids=candidate_ids,
        )

    def player_row(pid: int) -> dict[str, Any] | None:
        if int(pid) in live_players:
            return live_players[int(pid)]
        return _fallback_player_frame_lookup(df_players_all, int(pid))

    def ensure_entry(pid: int) -> None:
        pid = int(pid)
        if pid in player_updates:
            return
        row = player_row(pid)
        start = _seed_singles_rating(row)
        player_updates[pid] = {
            "r": start,
            "start": start,
            "w": _seed_stat(row, "singles_wins"),
            "l": _seed_stat(row, "singles_losses"),
            "mp": _seed_stat(row, "singles_matches_played"),
        }

    def get_singles_r(pid: int) -> float:
        ensure_entry(pid)
        return float(player_updates[int(pid)]["r"])

    def apply_update(pid: int, delta: float, outcome: bool | None, *, bonus_elo: float = 0.0) -> float:
        pid = int(pid)
        ensure_entry(pid)
        player_updates[pid]["r"] += float(delta) + _safe_positive_float(bonus_elo)
        player_updates[pid]["mp"] += 1
        if outcome is True:
            player_updates[pid]["w"] += 1
        elif outcome is False:
            player_updates[pid]["l"] += 1
        return float(player_updates[pid]["r"])

    def league_k_factor(league_name: str) -> int:
        if df_meta is None or getattr(df_meta, "empty", True):
            return int(default_k_factor)
        try:
            selected = df_meta[
                df_meta["league_name"].astype(str).str.casefold()
                == str(league_name).casefold()
            ]
            if len(selected) == 1:
                return int(
                    selected.iloc[0].get("k_factor", default_k_factor)
                    or default_k_factor
                )
        except Exception:
            pass
        return int(default_k_factor)

    def ensure_league_entry(pid: int, league_name: str) -> None:
        key = (int(pid), str(league_name))
        if key in league_updates:
            return
        rows = _league_frame_rows(
            effective_df_leagues,
            player_id=int(pid),
            league_name=str(league_name),
        )
        if len(rows) > 1:
            raise RuntimeError(
                "Official singles publish found duplicate league-rating rows "
                f"for player {int(pid)} in {league_name}."
            )
        current = rows[0] if rows else None
        seed = _seed_singles_rating(player_row(int(pid)))
        baseline = (
            _finite_float_or_none(current.get("rating"))
            if current is not None
            else None
        )
        if baseline is None:
            baseline = seed
        starting_rating = (
            _finite_float_or_none(current.get("starting_rating"))
            if current is not None
            else None
        )
        if starting_rating is None:
            starting_rating = baseline
        league_updates[key] = {
            "r": baseline,
            "start": starting_rating,
            "w": _seed_stat(current, "wins"),
            "l": _seed_stat(current, "losses"),
            "mp": _seed_stat(current, "matches_played"),
        }

    def league_rating(pid: int, league_name: str) -> float:
        ensure_league_entry(int(pid), str(league_name))
        return float(league_updates[(int(pid), str(league_name))]["r"])

    def apply_league_update(
        pid: int,
        league_name: str,
        delta: float,
        outcome: bool | None,
        *,
        bonus_elo: float = 0.0,
    ) -> None:
        ensure_league_entry(int(pid), str(league_name))
        state = league_updates[(int(pid), str(league_name))]
        state["r"] += float(delta) + _safe_positive_float(bonus_elo)
        state["mp"] += 1
        if outcome is True:
            state["w"] += 1
        elif outcome is False:
            state["l"] += 1

    for match_dt, _row_index, match in prepared_singles_rows:
        p1_raw = as_player_id(match.get("t1_p1"), name_to_id)
        p2_raw = as_player_id(match.get("t2_p1"), name_to_id)
        if p1_raw is None or p2_raw is None:
            skipped_incomplete += 1
            continue
        p1, p2 = int(p1_raw), int(p2_raw)
        if p1 == p2:
            skipped_incomplete += 1
            continue

        score_t1, score_t2 = extract_scores(match)
        if (score_t1 + score_t2) <= 0:
            skipped_empty += 1
            continue
        rating_scope = normalize_rating_scope(match)
        is_unrated = rating_scope == "unrated"
        league_name = str(match.get("league") or "Singles").strip() or "Singles"
        official_league = has_managed_league_metadata(df_meta, league_name)
        if official_league:
            managed_league_names.add(league_name)
        is_popup = is_popup_match(
            str(match.get("match_type") or ""),
            bool(match.get("is_popup", False)),
        )
        update_league_rating = official_league and should_update_island(
            is_popup=is_popup,
            rating_scope=rating_scope,
        )
        if official_league:
            # Membership and its league-format baseline must exist even for an
            # unrated or overall-only result.
            ensure_league_entry(p1, league_name)
            ensure_league_entry(p2, league_name)
        winner_bonus_elo = 0.0 if is_unrated else _safe_positive_float(match.get("rating_bonus_elo", match.get("winner_bonus_elo")))
        dt_val = match_dt.isoformat()
        r1, r2 = get_singles_r(p1), get_singles_r(p2)
        d1, d2 = (0.0, 0.0) if is_unrated else compute_team_deltas(
            r1,
            r2,
            score_t1,
            score_t2,
            k_factor=float(default_k_factor),
            min_win_delta=float(min_win_delta_elo),
            cap_loser_gain=cap_loser_gain_elo,
        )
        p1_outcome, p2_outcome = compute_outcomes(score_t1, score_t2)
        p1_bonus = winner_bonus_elo if p1_outcome is True else 0.0
        p2_bonus = winner_bonus_elo if p2_outcome is True else 0.0

        league_d1, league_d2 = 0.0, 0.0
        if update_league_rating:
            league_d1, league_d2 = compute_team_deltas(
                league_rating(p1, league_name),
                league_rating(p2, league_name),
                score_t1,
                score_t2,
                k_factor=float(league_k_factor(league_name)),
                min_win_delta=float(min_win_delta_elo),
                cap_loser_gain=cap_loser_gain_elo,
            )

        if is_unrated:
            end_r1, end_r2 = r1, r2
            stored_delta = 0.0
            skipped_unrated += 1
        else:
            end_r1 = apply_update(p1, d1, p1_outcome, bonus_elo=p1_bonus)
            end_r2 = apply_update(p2, d2, p2_outcome, bonus_elo=p2_bonus)
            if update_league_rating:
                apply_league_update(
                    p1,
                    league_name,
                    league_d1,
                    p1_outcome,
                    bonus_elo=p1_bonus,
                )
                apply_league_update(
                    p2,
                    league_name,
                    league_d2,
                    p2_outcome,
                    bonus_elo=p2_bonus,
                )
            for pid in (p1, p2):
                last_game_updates[pid] = max_activity_time(last_game_updates.get(pid), match_dt)
                affected_players.add(int(pid))
            if winner_bonus_elo > 0 and (p1_outcome is True or p2_outcome is True):
                bonus_match_count += 1
                bonus_player_elo_total += winner_bonus_elo
            stored_delta = (abs(d1) if p1_outcome is True else abs(d2)) + (winner_bonus_elo if (p1_outcome is True or p2_outcome is True) else 0.0)

        db_match = build_match_row(
            club_id=str(club_id),
            dt_val=dt_val,
            league_name=league_name,
            pids=(p1, None, p2, None),
            scores=(int(score_t1), int(score_t2)),
            stored_elo_delta=stored_delta,
            match_type=str(match.get("match_type") or "Singles"),
            week_tag=(
                "Singles"
                if match.get("week_tag") is None
                else str(match.get("week_tag"))
            ),
            start_ratings=(r1, None, r2, None),
            end_ratings=(end_r1, None, end_r2, None),
            context={**match, "match_format": "singles"},
            rating_scope=rating_scope,
            match_format="singles",
        )
        db_match["singles_replay_managed"] = True
        db_matches.append(db_match)
        successful_match_dates.append(dt_val)

    if build_write_plan_only:
        planned_player_updates: list[dict[str, Any]] = []
        for pid, stats in sorted(player_updates.items()):
            if pid not in affected_players:
                continue
            if int(pid) not in live_players:
                raise RuntimeError(
                    f"Official publish requires one authoritative player snapshot for player {int(pid)}."
                )
            current = dict(live_players[int(pid)])
            latest_match_at = last_game_updates.get(int(pid))
            singles_last_game_at = max_activity_time(
                current.get("singles_last_game_at")
                or _singles_replay_baseline(current).get("last_game_at"),
                latest_match_at,
            )
            activity_update = build_player_activity_update(current.get("last_game_at"), latest_match_at)
            expected = {
                "singles_rating": _seed_singles_rating(current),
                "singles_wins": _seed_stat(current, "singles_wins"),
                "singles_losses": _seed_stat(current, "singles_losses"),
                "singles_matches_played": _seed_stat(current, "singles_matches_played"),
                "singles_last_game_at": current.get("singles_last_game_at"),
                "last_game_at": current.get("last_game_at"),
                "inactive_at": current.get("inactive_at"),
                "active": bool(current.get("active", True)) if current.get("active") is not None else None,
            }
            after = {
                "singles_rating": float(stats["r"]),
                "singles_wins": int(stats["w"]),
                "singles_losses": int(stats["l"]),
                "singles_matches_played": int(stats["mp"]),
                "singles_last_game_at": (
                    singles_last_game_at.isoformat()
                    if singles_last_game_at
                    else None
                ),
                **activity_update,
            }
            planned_player_updates.append(
                {"player_id": int(pid), "rating_mode": "singles", "expected": expected, "after": after}
            )
        planned_league_updates: list[dict[str, Any]] = []
        for (pid, league_name), stats in sorted(league_updates.items()):
            rows = _league_frame_rows(
                effective_df_leagues,
                player_id=int(pid),
                league_name=str(league_name),
            )
            if len(rows) > 1:
                raise RuntimeError(
                    "Official singles publish found duplicate league-rating "
                    f"rows for player {int(pid)} in {league_name}."
                )
            current = rows[0] if rows else None
            expected = None
            if current is not None:
                expected_rating = _finite_float_or_none(current.get("rating"))
                expected_starting_rating = _finite_float_or_none(
                    current.get("starting_rating")
                )
                expected = {
                    "id": int(current["id"]),
                    "rating": expected_rating,
                    "wins": _seed_stat(current, "wins"),
                    "losses": _seed_stat(current, "losses"),
                    "matches_played": _seed_stat(current, "matches_played"),
                    "starting_rating": expected_starting_rating,
                    "is_active": (
                        bool(current.get("is_active"))
                        if current.get("is_active") is not None
                        else None
                    ),
                    "inactive_at": current.get("inactive_at"),
                }
            planned_league_updates.append(
                {
                    "player_id": int(pid),
                    "league_name": str(league_name),
                    "expected": expected,
                    "after": {
                        "rating": float(stats["r"]),
                        "wins": int(stats["w"]),
                        "losses": int(stats["l"]),
                        "matches_played": int(stats["mp"]),
                        "starting_rating": float(stats["start"]),
                        "is_active": True,
                        "inactive_at": None,
                    },
                }
            )

        official_league_names = {
            str(league_name) for _, league_name in league_updates
        }
        league_metadata_expectations = build_active_league_metadata_expectations(
            df_meta,
            club_id=str(club_id),
            league_names=official_league_names,
            default_k_factor=int(default_k_factor),
            expected_match_format="singles",
        )
        return {
            "inserted": len(db_matches),
            "match_format": "singles",
            "skipped_incomplete": int(skipped_incomplete),
            "skipped_empty": int(skipped_empty),
            "skipped_unrated": int(skipped_unrated),
            "winner_bonus_summary": {
                "match_count": int(bonus_match_count),
                "player_elo_total": float(bonus_player_elo_total),
            },
            "write_plan": {
                "match_rows": db_matches,
                "player_updates": planned_player_updates,
                "league_rating_updates": planned_league_updates,
                "league_metadata_expectations": (
                    league_metadata_expectations
                ),
            },
            "side_effect_context": {
                "affected_player_ids": sorted(affected_players),
                "successful_match_dates": successful_match_dates,
                "has_badge_eligible_match": False,
                "match_payloads": [
                    {
                        "league": str(row.get("league") or ""),
                        "date": str(row.get("date") or ""),
                        "score_t1": row.get("score_t1"),
                        "score_t2": row.get("score_t2"),
                    }
                    for row in db_matches
                ],
            },
        }

    if managed_league_names:
        league_labels = ", ".join(sorted(managed_league_names, key=str.casefold))
        raise RuntimeError(
            "Managed league matches require the atomic match-entry path; "
            f"no singles match was written for: {league_labels}."
        )

    if db_matches:
        insert_match_chunks_with_rating_scope_fallback(db_matches=db_matches, supabase=supabase, sb_retry=sb_retry)

    for (pid, league_name), stats in sorted(league_updates.items()):
        rows = _league_frame_rows(
            effective_df_leagues,
            player_id=int(pid),
            league_name=str(league_name),
        )
        if len(rows) > 1:
            raise RuntimeError(
                "Official singles publish found duplicate league-rating rows "
                f"for player {int(pid)} in {league_name}."
            )
        payload = {
            "club_id": str(club_id),
            "player_id": int(pid),
            "league_name": str(league_name),
            "rating": float(stats["r"]),
            "wins": int(stats["w"]),
            "losses": int(stats["l"]),
            "matches_played": int(stats["mp"]),
            "starting_rating": float(stats["start"]),
            "is_active": True,
            "inactive_at": None,
        }
        if rows:
            row_id = rows[0].get("id")
            if row_id is None:
                raise RuntimeError(
                    "Official singles league roster row is missing its stable id."
                )
            sb_retry(
                lambda payload=payload, row_id=int(row_id): supabase.table(
                    "league_ratings"
                )
                .update(payload)
                .eq("club_id", str(club_id))
                .eq("id", row_id)
                .execute()
            )
        else:
            sb_retry(
                lambda payload=payload: supabase.table("league_ratings")
                .insert(payload)
                .execute()
            )

    for pid, stats in player_updates.items():
        if pid not in affected_players:
            continue
        existing_row = player_row(int(pid)) or {}
        existing_last_game_at = existing_row.get("last_game_at")
        latest_match_at = last_game_updates.get(int(pid))
        singles_last_game_at = max_activity_time(
            existing_row.get("singles_last_game_at")
            or _singles_replay_baseline(existing_row).get("last_game_at"),
            latest_match_at,
        )
        activity_update = build_player_activity_update(existing_last_game_at, latest_match_at)
        payload = {
            "singles_rating": float(stats["r"]),
            "singles_wins": int(stats["w"]),
            "singles_losses": int(stats["l"]),
            "singles_matches_played": int(stats["mp"]),
            "singles_last_game_at": (
                singles_last_game_at.isoformat()
                if singles_last_game_at
                else None
            ),
        }
        if activity_update:
            payload.update(activity_update)
        sb_retry(lambda pid=pid, payload=payload: _update_player_singles_row(supabase, club_id=str(club_id), player_id=int(pid), payload=payload, sb_retry=sb_retry))

    player_update_queue: dict[str, Any] = {"mode": "skipped", "affected_players": len(affected_players), "week_windows": 0, "queued": 0, "already_queued": 0, "no_active_subscription": 0, "failed": 0}
    if db_matches and affected_players:
        try:
            queue_summary = queue_player_updates_for_affected_subscribers(
                supabase,
                club_id=str(club_id),
                affected_player_ids=sorted(affected_players),
                match_dates=successful_match_dates,
            )
            player_update_queue = {"mode": "queued", **queue_summary}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Player update queueing failed after singles match processing: %s", exc)
            player_update_queue = {**player_update_queue, "mode": "error", "error": str(exc)}

    return {
        "inserted": len(db_matches),
        "match_format": "singles",
        "skipped_incomplete": int(skipped_incomplete),
        "skipped_empty": int(skipped_empty),
        "skipped_unrated": int(skipped_unrated),
        "winner_bonus_summary": {"match_count": int(bonus_match_count), "player_elo_total": float(bonus_player_elo_total)},
        "player_update_queue": player_update_queue,
    }
