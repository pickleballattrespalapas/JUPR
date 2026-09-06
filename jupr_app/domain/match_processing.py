from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Callable

import pandas as pd

from jupr_app.domain.player_activity import build_player_activity_update, coerce_utc_datetime, max_activity_time
from jupr_app.domain.player_ratings_source import build_seed_rating_maps, current_seed_rating
from jupr_app.domain.gamification.badge_queue import enqueue_badge_eval
from jupr_app.domain.gamification.badge_worker import process_badge_eval_queue
from jupr_app.domain.gamification.live_awards import run_live_badge_awards
from jupr_app.domain.notifications.player_profile_update_repo import queue_player_updates_for_affected_subscribers
from jupr_app.domain.matches import (
    as_player_id,
    build_match_row,
    collect_seed_candidates,
    compute_outcomes,
    compute_team_deltas,
    extract_scores,
    insert_match_chunks_with_rating_scope_fallback,
    is_popup_match,
    normalize_rating_scope,
    should_update_island,
)

logger = logging.getLogger(__name__)

_RESERVED_LEAGUE_NAMES = frozenset({"", "overall", "popup", "singles"})


def is_official_league_name(value: Any) -> bool:
    """Return whether ``value`` is a candidate managed-league name."""

    try:
        normalized = str(value or "").strip().casefold()
    except Exception:
        return False
    return normalized not in _RESERVED_LEAGUE_NAMES


def has_managed_league_metadata(df_meta: Any, league_name: Any) -> bool:
    """Return whether the loaded metadata snapshot owns ``league_name``.

    A non-reserved label alone is not enough: tournaments and other event
    contexts also use the legacy ``matches.league`` display field.  Requiring
    a metadata row keeps the roster invariant scoped to actual managed
    leagues while the later CAS expectation validates status and format.
    """

    if not is_official_league_name(league_name):
        return False
    if df_meta is None or getattr(df_meta, "empty", True):
        return False
    try:
        names = df_meta["league_name"].astype(str).str.strip().str.casefold()
        return bool((names == str(league_name).strip().casefold()).any())
    except Exception:
        return False


def _safe_positive_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return max(0.0, float(value))
    except Exception:
        return 0.0


def _json_value(value: Any) -> Any:
    """Convert DataFrame scalars into deterministic JSON-safe values."""

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _frame_rows(frame: Any, **conditions: Any) -> list[dict[str, Any]]:
    if frame is None or getattr(frame, "empty", True):
        return []
    selected = frame
    try:
        for field, expected in conditions.items():
            selected = selected[selected[field].astype(str) == str(expected)]
    except Exception:
        return []
    return [
        {str(key): _json_value(value) for key, value in dict(row).items()}
        for _, row in selected.iterrows()
    ]


def build_active_league_metadata_expectations(
    df_meta: Any,
    *,
    club_id: str,
    league_names: set[str] | list[str] | tuple[str, ...],
    default_k_factor: int,
    expected_match_format: str = "doubles",
) -> list[dict[str, Any]]:
    """Build the exact lifecycle and format snapshot required by the atomic RPC."""

    clean_match_format = str(expected_match_format or "").strip().casefold()
    if clean_match_format not in {"doubles", "singles"}:
        raise ValueError("League match format must be doubles or singles.")

    required_names = sorted(
        {
            str(name or "").strip()
            for name in league_names
            if is_official_league_name(name)
        },
        key=str.casefold,
    )
    expectations: list[dict[str, Any]] = []
    for league_name in required_names:
        if df_meta is None or getattr(df_meta, "empty", True):
            raise RuntimeError(
                f"Active league metadata for {league_name} is unavailable. "
                "Nothing was written."
            )
        try:
            selected = df_meta[
                df_meta["league_name"].astype(str).str.casefold()
                == league_name.casefold()
            ]
        except Exception as exc:
            raise RuntimeError(
                f"Active league metadata for {league_name} is unavailable. "
                "Nothing was written."
            ) from exc
        rows = [
            {
                str(key): _json_value(value)
                for key, value in dict(row).items()
            }
            for _, row in selected.iterrows()
        ]
        if len(rows) != 1:
            raise RuntimeError(
                f"Expected one active league record for {league_name}; "
                f"found {len(rows)}. Nothing was written."
            )
        current = rows[0]
        current_match_format = str(
            current.get("match_format") or "doubles"
        ).strip().casefold()
        if current_match_format != clean_match_format:
            raise ValueError(
                f"{league_name} is a {current_match_format} league and cannot "
                f"accept {clean_match_format} matches."
            )
        ended_at = current.get("ended_at")
        status = str(current.get("status") or "").strip()
        is_active = current.get("is_active")
        if isinstance(is_active, str):
            is_active = is_active.strip().casefold() not in {
                "0",
                "false",
                "no",
                "off",
            }
        if is_active is not None:
            is_active = bool(is_active)
        lifecycle_active = (
            ended_at in (None, "")
            and (
                (
                    status.casefold() in {"active", "running", "live"}
                    and is_active is not False
                )
                or (not status and is_active is True)
            )
        )
        if not lifecycle_active:
            raise ValueError(
                f"{league_name} is no longer an active official league. "
                "Reload before entering league-rated matches."
            )
        required_fields = {
            "id",
            "club_id",
            "league_name",
            "k_factor",
            "status",
            "is_active",
            "ended_at",
        }
        if not required_fields.issubset(current):
            raise RuntimeError(
                f"The active league snapshot for {league_name} is incomplete. "
                "Nothing was written."
            )
        expectations.append(
            {
                "league_name": str(current.get("league_name") or league_name),
                "expected": {
                    "id": current.get("id"),
                    "club_id": str(current.get("club_id") or club_id),
                    "league_name": str(
                        current.get("league_name") or league_name
                    ),
                    "k_factor": int(
                        current.get("k_factor", default_k_factor)
                        or default_k_factor
                    ),
                    "status": current.get("status"),
                    "is_active": current.get("is_active"),
                    "ended_at": current.get("ended_at"),
                    "match_format": current_match_format,
                },
            }
        )
    return expectations


def process_matches(
    match_list: list[dict[str, Any]],
    *,
    supabase,
    club_id: str,
    name_to_id: dict[str, int],
    df_players_all,
    df_leagues,
    df_meta,
    sb_retry: Callable | None = None,
    default_k_factor: int = 32,
    min_win_delta_elo: float = 1.0,
    cap_loser_gain_elo: float | None = 16.0,
    build_write_plan_only: bool = False,
) -> dict[str, Any]:
    if sb_retry is None:
        def sb_retry(fn):
            return fn()

    db_matches: list[dict[str, Any]] = []
    overall_updates: dict[int, dict[str, Any]] = {}
    island_updates: dict[tuple[int, str], dict[str, Any]] = {}
    last_game_updates: dict[int, datetime] = {}
    affected_players: set[int] = set()
    match_payloads: list[dict[str, Any]] = []
    successful_match_dates: list[str] = []
    managed_league_names: set[str] = set()

    skipped_incomplete = 0
    skipped_empty = 0
    skipped_unrated = 0
    has_badge_eligible_match = False
    bonus_match_count = 0
    bonus_player_elo_total = 0.0

    candidate_player_ids, candidate_league_names = collect_seed_candidates(match_list, name_to_id)
    overall_seed_map, league_seed_map, ratings_from_live_tables = build_seed_rating_maps(
        supabase=supabase,
        club_id=str(club_id),
        player_ids=candidate_player_ids,
        league_names=candidate_league_names,
        df_players_all=df_players_all,
        df_leagues=df_leagues,
    )
    if not ratings_from_live_tables:
        logger.warning("process_matches is using fallback rating DataFrames; live seed rating query failed.")

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
        return None if row.empty else row.iloc[0]

    def ensure_overall_entry(pid: int):
        pid = int(pid)
        if pid in overall_updates:
            return
        pr = get_player_row(pid)
        if pr is None:
            overall_updates[pid] = {"r": float(overall_seed_map.get(pid, 1200.0)), "w": 0, "l": 0, "mp": 0}
            return
        overall_updates[pid] = {
            "r": float(overall_seed_map.get(pid, pr.get("rating", 1200.0) or 1200.0)),
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
            return float(overall_seed_map.get(pid, 1200.0))
        return float(overall_seed_map.get(pid, pr.get("rating", 1200.0) or 1200.0))

    def get_island_r(pid: int, league_name: str) -> float:
        key = (int(pid), str(league_name))
        if key in island_updates:
            return float(island_updates[key]["r"])
        return current_seed_rating(
            player_id=int(pid),
            league_name=str(league_name),
            overall_map=overall_seed_map,
            league_map=league_seed_map,
            default_rating=get_overall_r(int(pid)),
        )

    def ensure_island_entry(pid: int, league_name: str):
        key = (int(pid), str(league_name))
        if key in island_updates:
            return
        start = float(get_island_r(int(pid), str(league_name)))
        island_updates[key] = {"r": start, "start": start, "w": 0, "l": 0, "mp": 0}

    def apply_updates(
        pid: int,
        d_ov: float,
        d_isl: float,
        outcome,
        *,
        update_island: bool,
        league_name: str,
        rating_bonus_elo: float = 0.0,
    ) -> float:
        pid = int(pid)
        bonus = _safe_positive_float(rating_bonus_elo)
        ensure_overall_entry(pid)
        overall_updates[pid]["r"] += float(d_ov) + bonus
        overall_updates[pid]["mp"] += 1
        if outcome is True:
            overall_updates[pid]["w"] += 1
        elif outcome is False:
            overall_updates[pid]["l"] += 1
        if update_island:
            ensure_island_entry(pid, league_name)
            key = (pid, league_name)
            island_updates[key]["r"] += float(d_isl) + bonus
            island_updates[key]["mp"] += 1
            if outcome is True:
                island_updates[key]["w"] += 1
            elif outcome is False:
                island_updates[key]["l"] += 1
        return float(overall_updates[pid]["r"])

    for m in match_list:
        pids = tuple(as_player_id(m.get(col), name_to_id) for col in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"))
        if any(pid is None for pid in pids):
            skipped_incomplete += 1
            continue
        p1, p2, p3, p4 = tuple(int(pid) for pid in pids)

        s1, s2 = extract_scores(m)
        if (s1 + s2) <= 0:
            skipped_empty += 1
            continue

        league_name = str(m.get("league", "") or "").strip()
        week_tag = str(m.get("week_tag", "") or "")
        match_type = str(m.get("match_type", "") or "")
        rating_scope = normalize_rating_scope(m)
        is_popup = is_popup_match(match_type, bool(m.get("is_popup", False)))
        is_unrated = rating_scope == "unrated"
        managed_league = has_managed_league_metadata(df_meta, league_name)
        if managed_league:
            managed_league_names.add(league_name)
        update_island = managed_league and should_update_island(
            is_popup=is_popup,
            rating_scope=rating_scope,
        )
        if managed_league:
            # League membership is a structural invariant, not a side effect of
            # whether this particular result is league-rated.  Unrated and
            # overall-only official league games must still leave every
            # participant on the league roster with a baseline rating row.
            for player_id in (p1, p2, p3, p4):
                ensure_island_entry(player_id, league_name)
        winner_bonus_elo = 0.0 if is_unrated else _safe_positive_float(m.get("rating_bonus_elo", m.get("winner_bonus_elo")))
        if (not is_popup) and (not is_unrated):
            has_badge_eligible_match = True

        match_dt = coerce_utc_datetime(m.get("date")) or datetime.now(timezone.utc)
        dt_val = match_dt.isoformat()
        ro1, ro2, ro3, ro4 = get_overall_r(p1), get_overall_r(p2), get_overall_r(p3), get_overall_r(p4)

        do1, do2 = (0.0, 0.0) if is_unrated else compute_team_deltas(
            (ro1 + ro2) / 2.0,
            (ro3 + ro4) / 2.0,
            s1,
            s2,
            k_factor=float(default_k_factor),
            min_win_delta=float(min_win_delta_elo),
            cap_loser_gain=cap_loser_gain_elo,
        )
        di1, di2 = 0.0, 0.0
        league_k_factor: float | None = None
        if update_island:
            k_val = get_k(league_name)
            league_k_factor = float(k_val)
            ri1, ri2, ri3, ri4 = get_island_r(p1, league_name), get_island_r(p2, league_name), get_island_r(p3, league_name), get_island_r(p4, league_name)
            di1, di2 = compute_team_deltas(
                (ri1 + ri2) / 2.0,
                (ri3 + ri4) / 2.0,
                s1,
                s2,
                k_factor=float(k_val),
                min_win_delta=float(min_win_delta_elo),
                cap_loser_gain=cap_loser_gain_elo,
            )

        t1_outcome, t2_outcome = compute_outcomes(s1, s2)
        t1_bonus = winner_bonus_elo if t1_outcome is True else 0.0
        t2_bonus = winner_bonus_elo if t2_outcome is True else 0.0

        if is_unrated:
            end_r1, end_r2, end_r3, end_r4 = ro1, ro2, ro3, ro4
            stored_elo_delta = 0.0
        else:
            end_r1 = apply_updates(p1, do1, di1, t1_outcome, update_island=update_island, league_name=league_name, rating_bonus_elo=t1_bonus)
            end_r2 = apply_updates(p2, do1, di1, t1_outcome, update_island=update_island, league_name=league_name, rating_bonus_elo=t1_bonus)
            end_r3 = apply_updates(p3, do2, di2, t2_outcome, update_island=update_island, league_name=league_name, rating_bonus_elo=t2_bonus)
            end_r4 = apply_updates(p4, do2, di2, t2_outcome, update_island=update_island, league_name=league_name, rating_bonus_elo=t2_bonus)
            for pid in (p1, p2, p3, p4):
                last_game_updates[pid] = max_activity_time(last_game_updates.get(pid), match_dt)
                affected_players.add(int(pid))
            if winner_bonus_elo > 0 and (t1_outcome is True or t2_outcome is True):
                bonus_match_count += 1
                bonus_player_elo_total += winner_bonus_elo * 2.0
            stored_elo_delta = (abs(do1) if (t1_outcome is True) else abs(do2)) + (winner_bonus_elo if (t1_outcome is True or t2_outcome is True) else 0.0)

        if is_unrated:
            skipped_unrated += 1

        db_matches.append(
            build_match_row(
                club_id=club_id,
                dt_val=dt_val,
                league_name=league_name,
                pids=(p1, p2, p3, p4),
                scores=(s1, s2),
                stored_elo_delta=stored_elo_delta,
                match_type=match_type,
                week_tag=week_tag,
                start_ratings=(ro1, ro2, ro3, ro4),
                end_ratings=(end_r1, end_r2, end_r3, end_r4),
                context=m,
                rating_scope=rating_scope,
                overall_k_factor=float(default_k_factor),
                min_win_delta_elo=float(min_win_delta_elo),
                cap_loser_gain_elo=cap_loser_gain_elo,
                league_k_factor=league_k_factor,
            )
        )
        match_payloads.append({"league": league_name, "date": dt_val, "score_t1": s1, "score_t2": s2})
        successful_match_dates.append(dt_val)

    if build_write_plan_only:
        player_updates: list[dict[str, Any]] = []
        for pid, stats in sorted(overall_updates.items()):
            rows = _frame_rows(df_players_all, id=int(pid))
            if len(rows) != 1:
                raise RuntimeError(
                    f"Official publish requires one existing player snapshot for player {int(pid)}."
                )
            current = rows[0]
            latest_match_at = last_game_updates.get(int(pid))
            activity_update = build_player_activity_update(current.get("last_game_at"), latest_match_at)
            expected = {
                "rating": float(current.get("rating", 1200.0) or 1200.0),
                "wins": int(current.get("wins", 0) or 0),
                "losses": int(current.get("losses", 0) or 0),
                "matches_played": int(current.get("matches_played", 0) or 0),
                "last_game_at": current.get("last_game_at"),
                "inactive_at": current.get("inactive_at"),
                "active": bool(current.get("active", True)) if current.get("active") is not None else None,
            }
            after = {
                "rating": float(stats["r"]),
                "wins": int(stats["w"]),
                "losses": int(stats["l"]),
                "matches_played": int(stats["mp"]),
                **activity_update,
            }
            player_updates.append(
                {"player_id": int(pid), "rating_mode": "doubles", "expected": expected, "after": after}
            )

        league_rating_updates: list[dict[str, Any]] = []
        for (pid, league_name), stats in sorted(island_updates.items()):
            rows = _frame_rows(
                df_leagues,
                player_id=int(pid),
                league_name=str(league_name),
            )
            if len(rows) > 1:
                raise RuntimeError(
                    f"Official publish found duplicate league-rating rows for player {int(pid)} in {league_name}."
                )
            current = rows[0] if rows else None
            expected = None
            if current is not None:
                expected = {
                    "id": int(current["id"]),
                    "rating": float(current.get("rating", 1200.0) or 1200.0),
                    "wins": int(current.get("wins", 0) or 0),
                    "losses": int(current.get("losses", 0) or 0),
                    "matches_played": int(current.get("matches_played", 0) or 0),
                    "starting_rating": (
                        float(current["starting_rating"])
                        if current.get("starting_rating") is not None
                        else None
                    ),
                    "is_active": (
                        bool(current.get("is_active")) if current.get("is_active") is not None else None
                    ),
                    "inactive_at": current.get("inactive_at"),
                }
            after = {
                "rating": float(stats["r"]),
                "wins": int(stats["w"]) + int((current or {}).get("wins", 0) or 0),
                "losses": int(stats["l"]) + int((current or {}).get("losses", 0) or 0),
                "matches_played": int(stats["mp"]) + int((current or {}).get("matches_played", 0) or 0),
                "starting_rating": (
                    float(current["starting_rating"])
                    if current and current.get("starting_rating") is not None
                    else float(stats.get("start", 1200.0))
                ),
                "is_active": True,
                "inactive_at": None,
            }
            league_rating_updates.append(
                {
                    "player_id": int(pid),
                    "league_name": str(league_name),
                    "expected": expected,
                    "after": after,
                }
            )
        league_metadata_expectations = (
            build_active_league_metadata_expectations(
                df_meta,
                club_id=str(club_id),
                league_names={str(name) for _, name in island_updates},
                default_k_factor=int(default_k_factor),
                expected_match_format="doubles",
            )
        )
        return {
            "inserted": len(db_matches),
            "skipped_incomplete": int(skipped_incomplete),
            "skipped_empty": int(skipped_empty),
            "skipped_unrated": int(skipped_unrated),
            "winner_bonus_summary": {
                "match_count": int(bonus_match_count),
                "player_elo_total": float(bonus_player_elo_total),
            },
            "write_plan": {
                "match_rows": db_matches,
                "player_updates": player_updates,
                "league_rating_updates": league_rating_updates,
                "league_metadata_expectations": league_metadata_expectations,
            },
            "side_effect_context": {
                "affected_player_ids": sorted(affected_players),
                "successful_match_dates": successful_match_dates,
                "has_badge_eligible_match": bool(has_badge_eligible_match),
                "match_payloads": match_payloads,
            },
        }

    if managed_league_names:
        league_labels = ", ".join(sorted(managed_league_names, key=str.casefold))
        raise RuntimeError(
            "Managed league matches require the atomic match-entry path; "
            f"no match was written for: {league_labels}."
        )

    badge_summary: dict[str, Any] = {"mode": "skipped", "awarded_count": 0, "candidate_count": 0, "badge_ids": []}
    if db_matches:
        insert_match_chunks_with_rating_scope_fallback(db_matches=db_matches, supabase=supabase, sb_retry=sb_retry)
        if supabase is not None and has_badge_eligible_match:
            enqueue_result = enqueue_badge_eval(
                supabase,
                club_id=str(club_id),
                event_type="match_recorded",
                player_ids=sorted(affected_players),
                payload={"match_count": len(db_matches), "matches": match_payloads[:10]},
            )
            worker_result = None
            should_fallback = not bool(enqueue_result.get("queued"))
            if not should_fallback:
                try:
                    worker_result = process_badge_eval_queue(
                        supabase,
                        str(club_id),
                        max_jobs=1,
                        time_budget_seconds=2,
                    )
                    should_fallback = bool(worker_result.get("errored")) or (int(worker_result.get("processed") or 0) == 0 and int(worker_result.get("errored") or 0) > 0)
                    badge_summary = {"mode": "queue", **worker_result}
                except Exception as exc:  # noqa: BLE001
                    should_fallback = True
                    logger.warning("Badge queue worker failed during match processing: %s", exc)
            else:
                logger.warning("Badge queue enqueue unavailable during match processing; falling back inline. reason=%s", enqueue_result.get("reason"))
            if should_fallback:
                try:
                    badge_summary = run_live_badge_awards(supabase, club_id=str(club_id), player_ids=sorted(affected_players), event_type="match_recorded")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Inline live badge fallback failed after match processing: %s", exc)
                    badge_summary = {"mode": "inline_error", "error": str(exc)}

    def update_player_row(row, activity_update: dict):
        pid = int(row["id"])
        payload = {"rating": float(row["rating"]), "wins": int(row["wins"]), "losses": int(row["losses"]), "matches_played": int(row["matches_played"])}
        if activity_update:
            payload.update(activity_update)
        res = supabase.table("players").update(payload).eq("club_id", club_id).eq("id", pid).execute()
        if not res.data:
            supabase.table("players").insert({"club_id": club_id, "id": pid, **payload}).execute()

    for pid, stats in overall_updates.items():
        row = {"id": int(pid), "rating": float(stats["r"]), "wins": int(stats["w"]), "losses": int(stats["l"]), "matches_played": int(stats["mp"])}
        pr = get_player_row(int(pid))
        existing_last_game_at = None if pr is None else pr.get("last_game_at")
        latest_match_at = last_game_updates.get(int(pid))
        activity_update = build_player_activity_update(existing_last_game_at, latest_match_at)
        sb_retry(lambda row=row, activity_update=activity_update: update_player_row(row, activity_update))

    if island_updates:
        for (pid, league_name), stats in island_updates.items():
            payload = {"club_id": club_id, "player_id": int(pid), "league_name": str(league_name), "rating": float(stats["r"]), "wins": int(stats["w"]), "losses": int(stats["l"]), "matches_played": int(stats["mp"])}
            existing = sb_retry(lambda pid=pid, league_name=league_name: supabase.table("league_ratings").select("id,wins,losses,matches_played,starting_rating").eq("club_id", club_id).eq("player_id", int(pid)).eq("league_name", str(league_name)).limit(1).execute())
            if existing.data:
                cur = existing.data[0]
                payload["wins"] += int(cur.get("wins", 0) or 0)
                payload["losses"] += int(cur.get("losses", 0) or 0)
                payload["matches_played"] += int(cur.get("matches_played", 0) or 0)
                payload["is_active"] = True
                payload["inactive_at"] = None
                payload["starting_rating"] = float(cur["starting_rating"]) if cur.get("starting_rating") is not None else float(stats.get("start", 1200.0))
                sb_retry(lambda payload=payload, rid=int(cur["id"]): supabase.table("league_ratings").update(payload).eq("id", rid).execute())
            else:
                payload["starting_rating"] = float(stats.get("start", 1200.0))
                payload["is_active"] = True
                payload["inactive_at"] = None
                sb_retry(lambda payload=payload: supabase.table("league_ratings").insert(payload).execute())

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
            logger.warning("Player update queueing failed after match processing: %s", exc)
            player_update_queue = {**player_update_queue, "mode": "error", "error": str(exc)}
    return {
        "inserted": len(db_matches),
        "skipped_incomplete": int(skipped_incomplete),
        "skipped_empty": int(skipped_empty),
        "skipped_unrated": int(skipped_unrated),
        "winner_bonus_summary": {
            "match_count": int(bonus_match_count),
            "player_elo_total": float(bonus_player_elo_total),
        },
        "badge_summary": badge_summary,
        "player_update_queue": player_update_queue,
    }
