from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import math
import unicodedata
from typing import Any, Callable, Dict, Optional

import pandas as pd

from jupr_app.domain.constants import DEFAULT_K_FACTOR
from jupr_app.domain.matches import compute_outcomes, compute_team_deltas
from jupr_app.domain.player_activity import coerce_utc_datetime, max_activity_time
from jupr_app.domain.ratings import calculate_hybrid_elo

FULL_RESET_LABEL = "ALL (Full System Reset)"
_RESERVED_LEAGUE_NAMES = frozenset({"", "overall", "popup", "singles"})
_LEAGUE_RATING_STORAGE_QUANTUM = Decimal("0.0001")


class ReplayLeaseLostError(RuntimeError):
    """Raised before a replay write when its durable lease is no longer valid."""


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _league_rating_storage_value(value: Any) -> float:
    """Match the numeric(10,4) league-rating columns before strict RPC checks."""

    try:
        parsed = Decimal(str(value))
        if not parsed.is_finite():
            raise InvalidOperation
        return float(
            parsed.quantize(
                _LEAGUE_RATING_STORAGE_QUANTUM,
                rounding=ROUND_HALF_UP,
            )
        )
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Replay History produced a non-finite league-rating value."
        ) from exc


def _normalize_league_name(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    normalized = normalized.replace("’", "'")
    return " ".join(normalized.split())


def _managed_singles_replay_plan(
    *,
    rows: list[dict[str, Any]],
    players: list[dict[str, Any]],
    club_id: str,
    managed_leagues: dict[str, dict[str, Any]] | None = None,
    existing_league_ratings: dict[tuple[int, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Rebuild post-baseline singles and official-league projections."""

    state: dict[int, dict[str, Any]] = {}
    league_state: dict[tuple[int, str], dict[str, Any]] = {}
    managed_league_rows = dict(managed_leagues or {})
    existing_league_rows = dict(existing_league_ratings or {})
    player_rows: dict[int, dict[str, Any]] = {}
    for player in players:
        pid = _safe_int(player.get("id"))
        if pid is None:
            raise RuntimeError(
                "Managed singles replay loaded a player without a stable identity."
            )
        if pid in player_rows:
            raise RuntimeError(
                f"Managed singles replay loaded duplicate player identity {pid}."
            )
        player_rows[pid] = dict(player)

    snapshots: list[dict[str, Any]] = []
    snapshots_by_league: dict[str, list[dict[str, Any]]] = {}
    active_rated = 0
    active_unrated = 0
    deleted = 0
    invalid = 0

    def ensure_player(pid: int) -> dict[str, Any]:
        if pid in state:
            return state[pid]
        player = player_rows.get(pid)
        if player is None:
            raise RuntimeError(
                f"Managed singles replay player {pid} is unavailable."
            )
        baseline = player.get("singles_replay_baseline")
        required_keys = {"rating", "wins", "losses", "matches_played"}
        if not isinstance(baseline, dict) or not required_keys.issubset(baseline):
            raise RuntimeError(
                f"Managed singles replay baseline for player {pid} is incomplete."
            )
        rating = _safe_float(baseline.get("rating"), float("nan"))
        wins = _safe_int(baseline.get("wins"))
        losses = _safe_int(baseline.get("losses"))
        matches_played = _safe_int(baseline.get("matches_played"))
        if (
            not math.isfinite(rating)
            or wins is None
            or losses is None
            or matches_played is None
            or min(wins, losses, matches_played) < 0
        ):
            raise RuntimeError(
                f"Managed singles replay baseline for player {pid} is invalid."
            )
        baseline_last_game_at = baseline.get("last_game_at")
        parsed_last_game_at = coerce_utc_datetime(baseline_last_game_at)
        if baseline_last_game_at not in (None, "") and parsed_last_game_at is None:
            raise RuntimeError(
                f"Managed singles replay baseline for player {pid} has an invalid last-game timestamp."
            )
        state[pid] = {
            "r": rating,
            "w": wins,
            "l": losses,
            "mp": matches_played,
            "last_game_at": parsed_last_game_at,
        }
        return state[pid]

    def managed_league_for(row: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
        league_key = _normalize_league_name(row.get("league")).casefold()
        if league_key in _RESERVED_LEAGUE_NAMES:
            return None
        metadata = managed_league_rows.get(league_key)
        if metadata is None:
            return None
        match_format = str(metadata.get("match_format") or "").strip().casefold()
        if match_format != "singles":
            raise RuntimeError(
                "A replay-managed singles row conflicts with its managed "
                f"league match format: {metadata.get('league_name') or league_key}."
            )
        return league_key, metadata

    def ensure_league_player(
        pid: int,
        league_key: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        key = (int(pid), str(league_key))
        if key in league_state:
            return league_state[key]
        player_state = ensure_player(int(pid))
        existing = existing_league_rows.get(key)
        existing_start = _safe_float(
            existing.get("starting_rating") if existing else None,
            float("nan"),
        )
        existing_rating = _safe_float(
            existing.get("rating") if existing else None,
            float("nan"),
        )
        if math.isfinite(existing_start):
            seed = float(existing_start)
        elif math.isfinite(existing_rating):
            seed = float(existing_rating)
        else:
            seed = float(player_state["r"])
        league_state[key] = {
            "r": seed,
            "start": seed,
            "w": 0,
            "l": 0,
            "mp": 0,
            "league_name": str(
                (existing or {}).get("league_name")
                or metadata.get("league_name")
                or ""
            ).strip(),
            "is_active": bool((existing or {}).get("is_active", True)),
            "inactive_at": (existing or {}).get("inactive_at"),
        }
        return league_state[key]

    # A full replay restores the entire club projection, not just players still
    # referenced by current managed rows. This clears contributions left behind
    # by a supported participant correction or a physically removed managed row.
    for player_id in sorted(player_rows):
        ensure_player(player_id)

    for row in rows:
        if str(row.get("match_format") or "").strip().lower() != "singles":
            raise RuntimeError(
                "A replay-managed singles row has an invalid match format."
            )
        match_id = _safe_int(row.get("id"))
        if match_id is None:
            raise RuntimeError(
                "A replay-managed singles row has no stable match identity."
            )
        p1 = _safe_int(row.get("t1_p1"))
        p2 = _safe_int(row.get("t2_p1"))
        if (
            p1 is None
            or p2 is None
            or p1 == p2
            or row.get("t1_p2") is not None
            or row.get("t2_p2") is not None
        ):
            raise RuntimeError(
                "A replay-managed singles row has invalid player identities."
            )
        p1_state = ensure_player(p1)
        p2_state = ensure_player(p2)
        if row.get("deleted_at") not in (None, ""):
            deleted += 1
            continue

        s1 = _safe_int(row.get("score_t1"))
        s2 = _safe_int(row.get("score_t2"))
        if s1 is None or s2 is None or s1 < 0 or s2 < 0 or s1 == s2 or (s1 + s2) <= 0:
            raise RuntimeError(
                "A replay-managed singles row has an invalid final score."
            )

        rating_scope = str(row.get("rating_scope") or "").strip().casefold()
        is_popup = (
            str(row.get("match_type") or "").strip().casefold()
            == "popup"
        )
        managed_league = managed_league_for(row)
        league_p1_state: dict[str, Any] | None = None
        league_p2_state: dict[str, Any] | None = None
        if (
            managed_league is not None
            and not is_popup
            and rating_scope not in {"overall_only", "unrated"}
        ):
            league_key, league_metadata = managed_league
            league_p1_state = ensure_league_player(
                p1, league_key, league_metadata
            )
            league_p2_state = ensure_league_player(
                p2, league_key, league_metadata
            )

        r1 = float(p1_state["r"])
        r2 = float(p2_state["r"])
        is_unrated = rating_scope == "unrated"
        if is_unrated:
            d1 = d2 = 0.0
            end_r1, end_r2 = r1, r2
            stored_delta = 0.0
            active_unrated += 1
        else:
            match_dt = coerce_utc_datetime(row.get("date"))
            if match_dt is None:
                raise RuntimeError(
                    "A replay-managed rated singles row has an invalid match date."
                )
            d1, d2 = compute_team_deltas(
                r1,
                r2,
                s1,
                s2,
                k_factor=float(DEFAULT_K_FACTOR),
                min_win_delta=1.0,
                cap_loser_gain=16.0,
            )
            p1_outcome, p2_outcome = compute_outcomes(s1, s2)
            winner_bonus = max(0.0, _safe_float(row.get("rating_bonus_elo"), 0.0))
            p1_bonus = winner_bonus if p1_outcome is True else 0.0
            p2_bonus = winner_bonus if p2_outcome is True else 0.0
            end_r1 = r1 + float(d1) + p1_bonus
            end_r2 = r2 + float(d2) + p2_bonus
            p1_state["r"] = end_r1
            p2_state["r"] = end_r2
            for player_state, won in (
                (p1_state, p1_outcome),
                (p2_state, p2_outcome),
            ):
                player_state["mp"] += 1
                if won is True:
                    player_state["w"] += 1
                elif won is False:
                    player_state["l"] += 1
                player_state["last_game_at"] = max_activity_time(
                    player_state.get("last_game_at"),
                    match_dt,
                )
            stored_delta = (
                abs(float(d1)) if p1_outcome is True else abs(float(d2))
            ) + (winner_bonus if (p1_outcome is True or p2_outcome is True) else 0.0)
            active_rated += 1

            if (
                league_p1_state is not None
                and league_p2_state is not None
            ):
                league_d1, league_d2 = compute_team_deltas(
                    float(league_p1_state["r"]),
                    float(league_p2_state["r"]),
                    s1,
                    s2,
                    k_factor=float(league_metadata.get("k_factor") or DEFAULT_K_FACTOR),
                    min_win_delta=1.0,
                    cap_loser_gain=16.0,
                )
                for league_player_state, delta, won, bonus in (
                    (league_p1_state, league_d1, p1_outcome, p1_bonus),
                    (league_p2_state, league_d2, p2_outcome, p2_bonus),
                ):
                    league_player_state["r"] += float(delta) + float(bonus)
                    league_player_state["mp"] += 1
                    if won is True:
                        league_player_state["w"] += 1
                    elif won is False:
                        league_player_state["l"] += 1

        snapshot = {
            "id": match_id,
            "club_id": str(club_id),
            "elo_delta": float(stored_delta),
            "t1_p1_r": r1,
            "t1_p2_r": None,
            "t2_p1_r": r2,
            "t2_p2_r": None,
            "t1_p1_r_end": float(end_r1),
            "t1_p2_r_end": None,
            "t2_p1_r_end": float(end_r2),
            "t2_p2_r_end": None,
        }
        snapshots.append(snapshot)
        row_league_key = _normalize_league_name(row.get("league")).casefold()
        snapshots_by_league.setdefault(row_league_key, []).append(snapshot)

    player_updates: list[dict[str, Any]] = []
    for pid in sorted(player_rows):
        last_game_at = coerce_utc_datetime(state[pid].get("last_game_at"))
        player_updates.append(
            {
                "id": int(pid),
                "club_id": str(club_id),
                "singles_rating": float(state[pid]["r"]),
                "singles_wins": int(state[pid]["w"]),
                "singles_losses": int(state[pid]["l"]),
                "singles_matches_played": int(state[pid]["mp"]),
                "singles_last_game_at": (
                    last_game_at.isoformat() if last_game_at else None
                ),
            }
        )
    league_ratings: list[dict[str, Any]] = []
    for (pid, _league_key), league_player_state in sorted(
        league_state.items(),
        key=lambda item: (item[0][1], item[0][0]),
    ):
        league_ratings.append(
            {
                "club_id": str(club_id),
                "player_id": int(pid),
                "league_name": str(league_player_state["league_name"]),
                "rating": float(league_player_state["r"]),
                "wins": int(league_player_state["w"]),
                "losses": int(league_player_state["l"]),
                "matches_played": int(league_player_state["mp"]),
                "starting_rating": float(league_player_state["start"]),
                "is_active": bool(league_player_state["is_active"]),
                "inactive_at": league_player_state["inactive_at"],
            }
        )
    return {
        "snapshots": snapshots,
        "snapshots_by_league": snapshots_by_league,
        "player_updates": player_updates,
        "league_ratings": league_ratings,
        "matches_scanned": len(rows),
        "active_rated": active_rated,
        "active_unrated": active_unrated,
        "deleted": deleted,
        "invalid": invalid,
    }


def replay_history(
    *,
    supabase,
    club_id: str,
    df_meta: Optional[pd.DataFrame],
    target_reset: str,
    progress_cb: Optional[Callable[[float], None]] = None,
    write_fence: Optional[Dict[str, str]] = None,
    before_write_batch: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """
    Replays match history in chronological order and:

      - rewrites match snapshot columns (RPC bulk UPDATE)
      - reconciles match-evidenced league_ratings without deleting roster-only rows
      - updates players table ONLY when doing FULL reset (RPC bulk UPDATE)

    No per-row .execute() loops for writes.
    No upsert() for partial updates (avoids NOT NULL failures like matches.league).
    """
    import random
    import time

    import httpx

    READ_PAGE_SIZE = 1000
    WRITE_BATCH_SIZE = 500
    RETRIES = 5

    RPC_MATCH_SNAPSHOTS = "bulk_update_match_snapshots"
    RPC_PLAYERS_STATS = "bulk_update_players_stats"
    RPC_PLAYER_SINGLES_STATS = "bulk_update_player_singles_stats"
    RPC_FENCED_WRITE_BATCH = "apply_replay_write_batch_atomic"
    RPC_FENCED_LEAGUE_RATINGS = "apply_replay_league_rating_rows_atomic"

    # ------------------------------
    # Helpers
    # ------------------------------
    def _norm_league(val: Any) -> str:
        return _normalize_league_name(val)

    def _chunk(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
        if not rows:
            return []
        return [rows[i : i + size] for i in range(0, len(rows), size)]

    def _require_rpc_count(response: Any, *, expected: int, label: str) -> int:
        try:
            actual = int(response.data)
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{label} did not return a verifiable update count."
            ) from exc
        if actual != int(expected):
            raise RuntimeError(
                f"{label} updated {actual} of {int(expected)} expected rows."
            )
        return actual

    def _retry(fn, *, label: str = ""):
        for attempt in range(RETRIES):
            try:
                return fn()
            except (
                httpx.ReadError,
                httpx.ConnectError,
                httpx.WriteError,
                httpx.RemoteProtocolError,
                httpx.TimeoutException,
            ):
                if attempt == RETRIES - 1:
                    raise
                time.sleep(0.35 * (2**attempt) + random.random() * 0.15)

    def _is_replay_fence_error(exc: Exception) -> bool:
        parts = [str(exc)]
        args = getattr(exc, "args", ())
        if args and isinstance(args[0], dict):
            parts.extend(str(value) for value in args[0].values())
        parts.extend(
            str(getattr(exc, key, "") or "")
            for key in ("code", "message", "details", "hint")
        )
        searchable = " ".join(parts).upper()
        return (
            "JUPR_REPLAY_WRITE_FENCE_LOST" in searchable
            or "REPLAY_LEASE_LOST" in searchable
        )

    def _load_exact_pages(
        query_factory: Callable[[int, int], Any],
        *,
        label: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        expected_count: int | None = None
        while True:
            start = len(rows)
            end = start + READ_PAGE_SIZE - 1
            response = _retry(
                lambda s=start, e=end: query_factory(s, e).execute(),
                label=f"{label}_{start}_{end}",
            )
            raw_count = getattr(response, "count", None)
            try:
                page_count = int(raw_count)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"{label} did not return an exact row count."
                ) from exc
            if page_count < 0:
                raise RuntimeError(f"{label} returned an invalid row count.")
            if expected_count is None:
                expected_count = page_count
            elif page_count != expected_count:
                raise RuntimeError(
                    f"{label} changed while Replay History was reading it."
                )

            page = [dict(row) for row in (response.data or [])]
            if len(rows) + len(page) > expected_count:
                raise RuntimeError(f"{label} returned more rows than its exact count.")
            rows.extend(page)
            if len(rows) == expected_count:
                return rows
            if not page:
                raise RuntimeError(
                    f"{label} ended after {len(rows)} of {expected_count} rows."
                )

    # ------------------------------
    # Reset mode
    # ------------------------------
    target_reset_raw = str(target_reset).strip()
    full_reset = target_reset_raw == FULL_RESET_LABEL
    target_league_norm = _norm_league(target_reset_raw)
    fence_params: dict[str, str] | None = None
    if write_fence is not None:
        if not isinstance(write_fence, dict):
            raise ValueError("Replay write fence must be an exact mapping.")
        fence_params = {
            "p_job_id": str(write_fence.get("job_id") or "").strip(),
            "p_club_id": str(club_id),
            "p_lease_token": str(
                write_fence.get("lease_token") or ""
            ).strip(),
            "p_worker_id": str(write_fence.get("worker_id") or "").strip(),
            "p_target_reset": target_reset_raw,
        }
        if not all(fence_params.values()):
            raise ValueError(
                "Replay write fence requires job, club, lease token, worker, "
                "and target."
            )

    def _before_mutation() -> None:
        if before_write_batch is not None:
            before_write_batch()

    def _fenced_rpc(params: dict[str, Any], *, label: str) -> Any:
        if fence_params is None:
            raise RuntimeError("Replay fenced RPC called without a write fence.")
        try:
            return _retry(
                lambda: supabase.rpc(
                    RPC_FENCED_WRITE_BATCH,
                    {**fence_params, **params},
                ).execute(),
                label=label,
            )
        except Exception as exc:
            if _is_replay_fence_error(exc):
                raise ReplayLeaseLostError(
                    "Replay mutation was rejected because this worker no "
                    "longer owns the active database lease."
                ) from exc
            raise

    def _fenced_league_rating_rpc(
        rows: list[dict[str, Any]],
        *,
        label: str,
    ) -> Any:
        if fence_params is None:
            raise RuntimeError("Replay fenced RPC called without a write fence.")
        try:
            return _retry(
                lambda: supabase.rpc(
                    RPC_FENCED_LEAGUE_RATINGS,
                    {**fence_params, "p_rows": rows},
                ).execute(),
                label=label,
            )
        except Exception as exc:
            if _is_replay_fence_error(exc):
                raise ReplayLeaseLostError(
                    "Replay mutation was rejected because this worker no "
                    "longer owns the active database lease."
                ) from exc
            raise

    # ---------------------------------------------------------
    # Load players (only what we need)
    # ---------------------------------------------------------
    try:
        all_players = _load_exact_pages(
            lambda start, end: (
                supabase.table("players")
                .select(
                    "id,starting_rating,rating,singles_replay_baseline",
                    count="exact",
                )
                .eq("club_id", club_id)
                .order("id", desc=False)
                .range(start, end)
            ),
            label="select_players_with_singles_baseline",
        )
    except Exception as exc:
        raise RuntimeError(
            "Managed singles player baselines are required for Replay History."
        ) from exc
    if any("singles_replay_baseline" not in player for player in all_players):
        raise RuntimeError(
            "Managed singles player baselines are incomplete."
        )

    try:
        existing_league_ratings = _load_exact_pages(
            lambda start, end: (
                supabase.table("league_ratings")
                .select(
                    "player_id,league_name,rating,wins,losses,"
                    "matches_played,starting_rating,is_active,inactive_at",
                    count="exact",
                )
                .eq("club_id", club_id)
                .order("player_id", desc=False)
                .order("league_name", desc=False)
                .range(start, end)
            ),
            label="select_existing_league_ratings",
        )
    except Exception as exc:
        raise RuntimeError(
            "Existing league-rating membership is required for Replay History."
        ) from exc

    existing_league_rows: dict[tuple[int, str], dict[str, Any]] = {}
    for existing_rating in existing_league_ratings:
        player_id = _safe_int(existing_rating.get("player_id"))
        league_name = _norm_league(existing_rating.get("league_name"))
        if player_id is None or not league_name:
            raise RuntimeError(
                "Replay History loaded an invalid existing league-rating row."
            )
        key = (int(player_id), league_name.casefold())
        if key in existing_league_rows:
            raise RuntimeError(
                "Replay History loaded duplicate normalized league-rating rows."
            )
        existing_league_rows[key] = dict(existing_rating)

    valid_player_ids: set[int] = set()

    for p in all_players:
        try:
            pid = int(p["id"])
        except Exception:
            continue

        valid_player_ids.add(pid)

    # ---------------------------------------------------------
    # Build K-factor map (normalized league names)
    # ---------------------------------------------------------
    k_map: Dict[str, int] = {}
    managed_leagues: dict[str, dict[str, Any]] = {}
    if df_meta is not None and not df_meta.empty:
        for _, r in df_meta.iterrows():
            try:
                lg_name = _norm_league(r["league_name"])
                k_map[lg_name.casefold()] = int(
                    r.get("k_factor", DEFAULT_K_FACTOR) or DEFAULT_K_FACTOR
                )
            except Exception:
                pass
            league_key = _norm_league(r.get("league_name")).casefold()
            if league_key in _RESERVED_LEAGUE_NAMES:
                continue
            if league_key in managed_leagues:
                raise RuntimeError(
                    "Replay History loaded duplicate normalized league metadata: "
                    f"{_norm_league(r.get('league_name'))}."
                )
            managed_leagues[league_key] = {
                "league_name": _norm_league(r.get("league_name")),
                "match_format": str(r.get("match_format") or "").strip().casefold(),
                "k_factor": int(
                    _safe_int(r.get("k_factor")) or DEFAULT_K_FACTOR
                ),
            }

    target_managed_league = managed_leagues.get(target_league_norm.casefold())
    target_managed_singles = bool(
        target_managed_league
        and target_managed_league.get("match_format") == "singles"
    )

    def k_for(lg: str) -> int:
        return int(
            k_map.get(_norm_league(lg).casefold(), DEFAULT_K_FACTOR)
        )

    # ---------------------------------------------------------
    # Initialize overall player state (in-memory)
    # ---------------------------------------------------------
    p_map: Dict[int, Dict[str, Any]] = {}
    for p in all_players:
        try:
            pid = int(p["id"])
        except Exception:
            continue

        base = p.get("starting_rating")
        if base is None:
            base = p.get("rating", 1200.0)

        try:
            p_map[pid] = {"r": float(base or 1200.0), "w": 0, "l": 0, "mp": 0}
        except Exception:
            p_map[pid] = {"r": 1200.0, "w": 0, "l": 0, "mp": 0}

    def ensure_player(pid: int) -> None:
        if pid not in p_map:
            p_map[pid] = {"r": 1200.0, "w": 0, "l": 0, "mp": 0}

    def gr(pid: int) -> float:
        ensure_player(int(pid))
        return float(p_map[int(pid)]["r"])

    island_map: Dict[tuple[int, str], Dict[str, Any]] = {}

    def gir(pid: int, lg: str, *, seed_rating: Optional[float] = None) -> float:
        pid_i = int(pid)
        lg_s = _norm_league(lg)
        ensure_player(pid_i)

        key = (pid_i, lg_s)
        if key not in island_map:
            existing = existing_league_rows.get((pid_i, lg_s.casefold()))
            existing_start = _safe_float(
                existing.get("starting_rating") if existing else None,
                float("nan"),
            )
            existing_rating = _safe_float(
                existing.get("rating") if existing else None,
                float("nan"),
            )
            if math.isfinite(existing_start):
                base_r = float(existing_start)
            elif math.isfinite(existing_rating):
                base_r = float(existing_rating)
            elif seed_rating is not None:
                base_r = float(seed_rating)
            else:
                base_r = float(p_map[pid_i]["r"])
            island_map[key] = {
                "r": float(base_r),
                "start": float(base_r),
                "w": 0,
                "l": 0,
                "mp": 0,
                "league_name": str(
                    (existing or {}).get("league_name") or lg_s
                ),
                "is_active": bool((existing or {}).get("is_active", True)),
                "inactive_at": (existing or {}).get("inactive_at"),
            }
        return float(island_map[key]["r"])

    # ---------------------------------------------------------
    # Read matches with pagination
    # ---------------------------------------------------------
    match_cols = (
        "id,date,league,match_type,match_format,"
        "t1_p1,t1_p2,t2_p1,t2_p2,"
        "score_t1,score_t2,deleted_at,rating_scope"
    )

    matches_to_update: list[dict[str, Any]] = []
    skipped_incomplete_scope = 0
    matches_scanned_total = 0
    last_game_at_by_pid: dict[int, Any] = {
        player_id: None for player_id in valid_player_ids
    }

    expected_match_count: int | None = None
    offset = 0
    while True:
        start = offset
        end = offset + READ_PAGE_SIZE - 1

        def _load_match_page():
            return (
                supabase.table("matches")
                .select(match_cols, count="exact")
                .eq("club_id", club_id)
                .is_("deleted_at", None)
                .order("date", desc=False)
                .order("id", desc=False)
                .range(start, end)
                .execute()
            )

        page_resp = _retry(
            _load_match_page,
            label=f"select_matches_{start}_{end}",
        )
        try:
            page_count = int(page_resp.count)
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Replay History match scan did not return an exact row count."
            ) from exc
        if expected_match_count is None:
            expected_match_count = page_count
        elif page_count != expected_match_count:
            raise RuntimeError(
                "Replay History matches changed while they were being read."
            )

        page = page_resp.data or []
        if not page:
            if offset == expected_match_count:
                break
            raise RuntimeError(
                "Replay History match scan ended before its exact row count."
            )
        if offset + len(page) > expected_match_count:
            raise RuntimeError(
                "Replay History match scan exceeded its exact row count."
            )

        matches_scanned_total += len(page)

        for m in page:
            if m.get("deleted_at") is not None:
                continue
            # Player activity is a projection of every active, scored match,
            # including unrated and singles rows. Compute it before rating
            # scope/format filtering so exclusion replay restores the prior
            # active timestamp (or NULL when no scored match remains).
            activity_score_t1 = _safe_int(m.get("score_t1"))
            activity_score_t2 = _safe_int(m.get("score_t2"))
            activity_time = coerce_utc_datetime(m.get("date"))
            if (
                activity_score_t1 is not None
                and activity_score_t2 is not None
                and (activity_score_t1 + activity_score_t2) > 0
                and activity_time is not None
            ):
                for player_value in (
                    m.get("t1_p1"),
                    m.get("t1_p2"),
                    m.get("t2_p1"),
                    m.get("t2_p2"),
                ):
                    activity_player_id = _safe_int(player_value)
                    if activity_player_id in last_game_at_by_pid:
                        last_game_at_by_pid[activity_player_id] = (
                            max_activity_time(
                                last_game_at_by_pid[activity_player_id],
                                activity_time,
                            )
                        )
            rating_scope = str(
                m.get("rating_scope", "") or ""
            ).strip().casefold()
            if "rating_scope" in m and rating_scope == "unrated":
                continue
            lg = _norm_league(m.get("league", "") or "")
            in_scope = full_reset or (lg == target_league_norm)

            p1, p2, p3, p4 = m.get("t1_p1"), m.get("t1_p2"), m.get("t2_p1"), m.get("t2_p2")
            if None in (p1, p2, p3, p4):
                if (
                    p2 is None
                    and p4 is None
                    and (
                        str(m.get("match_type") or "").strip().casefold()
                        == "singles"
                        or str(m.get("match_format") or "")
                        .strip()
                        .casefold()
                        == "singles"
                    )
                ):
                    continue
                if in_scope:
                    skipped_incomplete_scope += 1
                continue

            try:
                p1, p2, p3, p4 = int(p1), int(p2), int(p3), int(p4)
            except Exception:
                if in_scope:
                    skipped_incomplete_scope += 1
                continue

            s1 = int(m.get("score_t1", 0) or 0)
            s2 = int(m.get("score_t2", 0) or 0)

            sr1, sr2, sr3, sr4 = gr(p1), gr(p2), gr(p3), gr(p4)

            # league-specific only if in_scope and not PopUp
            do_league = (
                str(m.get("match_type") or "").strip().casefold()
                != "popup"
                and rating_scope != "overall_only"
                and in_scope
            )
            if do_league:
                ir1 = gir(p1, lg, seed_rating=sr1)
                ir2 = gir(p2, lg, seed_rating=sr2)
                ir3 = gir(p3, lg, seed_rating=sr3)
                ir4 = gir(p4, lg, seed_rating=sr4)

                di1, di2 = calculate_hybrid_elo(
                    (ir1 + ir2) / 2,
                    (ir3 + ir4) / 2,
                    s1,
                    s2,
                    k_factor=k_for(lg),
                )
            else:
                di1 = di2 = 0.0

            do1, do2 = calculate_hybrid_elo(
                (sr1 + sr2) / 2,
                (sr3 + sr4) / 2,
                s1,
                s2,
                k_factor=DEFAULT_K_FACTOR,
            )

            win = s1 > s2

            # overall updates
            for pid, delta, won_flag in [
                (p1, do1, win),
                (p2, do1, win),
                (p3, do2, not win),
                (p4, do2, not win),
            ]:
                ensure_player(pid)
                p_map[pid]["r"] += float(delta)
                p_map[pid]["mp"] += 1
                if won_flag:
                    p_map[pid]["w"] += 1
                else:
                    p_map[pid]["l"] += 1

            # league updates
            if do_league:
                for pid, delta, won_flag in [
                    (p1, di1, win),
                    (p2, di1, win),
                    (p3, di2, not win),
                    (p4, di2, not win),
                ]:
                    key = (int(pid), lg)
                    if key not in island_map:
                        gir(int(pid), lg, seed_rating=float(gr(pid)))

                    island_map[key]["r"] += float(delta)
                    island_map[key]["mp"] += 1
                    if won_flag:
                        island_map[key]["w"] += 1
                    else:
                        island_map[key]["l"] += 1

            er1, er2, er3, er4 = gr(p1), gr(p2), gr(p3), gr(p4)

            if in_scope:
                stored_elo_delta = abs(do1) if win else abs(do2)
                matches_to_update.append(
                    {
                        "id": int(m["id"]),
                        "club_id": club_id,
                        "elo_delta": float(stored_elo_delta),
                        "t1_p1_r": float(sr1),
                        "t1_p2_r": float(sr2),
                        "t2_p1_r": float(sr3),
                        "t2_p2_r": float(sr4),
                        "t1_p1_r_end": float(er1),
                        "t1_p2_r_end": float(er2),
                        "t2_p1_r_end": float(er3),
                        "t2_p2_r_end": float(er4),
                    }
                )

        offset += len(page)
        if offset == expected_match_count:
            break

    singles_plan = {
        "snapshots": [],
        "snapshots_by_league": {},
        "player_updates": [],
        "league_ratings": [],
        "matches_scanned": 0,
        "active_rated": 0,
        "active_unrated": 0,
        "deleted": 0,
        "invalid": 0,
    }
    if full_reset or target_managed_singles:
        try:
            singles_rows = _load_exact_pages(
                lambda start, end: (
                    supabase.table("matches")
                    .select(
                        "id,date,league,match_type,match_format,rating_scope,"
                        "t1_p1,t1_p2,t2_p1,t2_p2,"
                        "score_t1,score_t2,rating_bonus_elo,deleted_at,"
                        "singles_replay_managed",
                        count="exact",
                    )
                    .eq("club_id", club_id)
                    .eq("singles_replay_managed", True)
                    .order("date", desc=False)
                    .order("id", desc=False)
                    .range(start, end)
                ),
                label="select_managed_singles_matches",
            )
        except Exception as exc:
            raise RuntimeError(
                "Managed singles Replay History data is unavailable."
            ) from exc
        singles_plan = _managed_singles_replay_plan(
            rows=[dict(row) for row in singles_rows],
            players=[dict(player) for player in all_players],
            club_id=str(club_id),
            managed_leagues=managed_leagues,
            existing_league_ratings=existing_league_rows,
        )
        if full_reset:
            matches_to_update.extend(singles_plan["snapshots"])
        else:
            matches_to_update.extend(
                singles_plan["snapshots_by_league"].get(
                    target_league_norm.casefold(),
                    [],
                )
            )

    # ---------------------------------------------------------
    # Build league_ratings rows
    # ---------------------------------------------------------
    new_rows: list[dict[str, Any]] = []
    for (pid, lg), s in island_map.items():
        if pid not in valid_player_ids:
            continue

        new_rows.append(
            {
                "club_id": club_id,
                "player_id": int(pid),
                "league_name": str(s.get("league_name") or lg),
                "rating": float(s["r"]),
                "wins": int(s["w"]),
                "losses": int(s["l"]),
                "matches_played": int(s["mp"]),
                "starting_rating": float(s["start"]),
                "is_active": bool(s["is_active"]),
                "inactive_at": s["inactive_at"],
            }
        )
    if full_reset:
        new_rows.extend(singles_plan["league_ratings"])
    elif target_managed_singles:
        new_rows.extend(
            row
            for row in singles_plan["league_ratings"]
            if _norm_league(row.get("league_name")).casefold()
            == target_league_norm.casefold()
        )

    rebuilt_keys: set[tuple[int, str]] = set()
    for row in new_rows:
        row_key = (
            int(row["player_id"]),
            _norm_league(row.get("league_name")).casefold(),
        )
        if row_key in rebuilt_keys:
            raise RuntimeError(
                "Replay History produced duplicate normalized league-rating rows."
            )
        rebuilt_keys.add(row_key)

    reset_without_active_evidence = 0
    for row_key, existing in existing_league_rows.items():
        if row_key in rebuilt_keys:
            continue
        if not full_reset and row_key[1] != target_league_norm.casefold():
            continue
        matches_played = _safe_int(existing.get("matches_played"))
        if matches_played is None or matches_played < 0:
            raise RuntimeError(
                "Replay History loaded invalid existing league-rating counters."
            )
        if matches_played == 0:
            continue
        starting_rating = _safe_float(
            existing.get("starting_rating"),
            float("nan"),
        )
        if not math.isfinite(starting_rating):
            starting_rating = _safe_float(existing.get("rating"), float("nan"))
        if not math.isfinite(starting_rating):
            raise RuntimeError(
                "Replay History cannot reset an unevidenced league rating "
                "without a finite starting rating."
            )
        new_rows.append(
            {
                "club_id": club_id,
                "player_id": int(row_key[0]),
                "league_name": str(existing.get("league_name") or ""),
                "rating": float(starting_rating),
                "wins": 0,
                "losses": 0,
                "matches_played": 0,
                "starting_rating": float(starting_rating),
                "is_active": bool(existing.get("is_active", True)),
                "inactive_at": existing.get("inactive_at"),
            }
        )
        rebuilt_keys.add(row_key)
        reset_without_active_evidence += 1

    # The durable writer verifies exact values after PostgreSQL stores these
    # columns as numeric(10,4). Normalize at the application boundary so the
    # write and its strict readback compare the same representation.
    for row in new_rows:
        row["rating"] = _league_rating_storage_value(row.get("rating"))
        row["starting_rating"] = _league_rating_storage_value(
            row.get("starting_rating")
        )

    # ---------------------------------------------------------
    # Writes
    # ---------------------------------------------------------
    players_updated = False

    # 1) FULL reset: update players via RPC bulk UPDATE
    if full_reset:
        players_updated = True

        player_updates: list[dict[str, Any]] = []
        for pid, s in p_map.items():
            if pid not in valid_player_ids:
                continue
            last_game_at = coerce_utc_datetime(
                last_game_at_by_pid.get(int(pid))
            )
            player_updates.append(
                {
                    "id": int(pid),
                    "club_id": club_id,
                    "rating": float(s["r"]),
                    "wins": int(s["w"]),
                    "losses": int(s["l"]),
                    "matches_played": int(s["mp"]),
                    "last_game_at": (
                        last_game_at.isoformat() if last_game_at else None
                    ),
                }
            )

        for batch in _chunk(player_updates, WRITE_BATCH_SIZE):
            _before_mutation()
            if fence_params is not None:
                response = _fenced_rpc(
                    {
                        "p_write_kind": "players_stats",
                        "p_rows": batch,
                    },
                    label="rpc_fenced_bulk_update_players_stats",
                )
            else:
                response = _retry(
                    lambda b=batch: supabase.rpc(
                        RPC_PLAYERS_STATS, {"rows": b}
                    ).execute(),
                    label="rpc_bulk_update_players_stats",
                )
            _require_rpc_count(
                response,
                expected=len(batch),
                label="Overall player replay",
            )

        for batch in _chunk(singles_plan["player_updates"], WRITE_BATCH_SIZE):
            _before_mutation()
            if fence_params is not None:
                response = _fenced_rpc(
                    {
                        "p_write_kind": "player_singles_stats",
                        "p_rows": batch,
                    },
                    label="rpc_fenced_bulk_update_player_singles_stats",
                )
            else:
                response = _retry(
                    lambda b=batch: supabase.rpc(
                        RPC_PLAYER_SINGLES_STATS, {"rows": b}
                    ).execute(),
                    label="rpc_bulk_update_player_singles_stats",
                )
            _require_rpc_count(
                response,
                expected=len(batch),
                label="Managed singles player replay",
            )

    # 2) league_ratings: upsert only match-evidenced projections and explicit
    # stale-stat resets. Unevidenced zero-counter roster rows are structural
    # membership, not disposable replay output, so they remain untouched.
    for batch in _chunk(new_rows, WRITE_BATCH_SIZE):
        _before_mutation()
        if fence_params is not None:
            response = _fenced_league_rating_rpc(
                batch,
                label="rpc_fenced_upsert_league_ratings",
            )
        else:
            response = _retry(
                lambda b=batch: supabase.table("league_ratings")
                .upsert(
                    b,
                    on_conflict="club_id,player_id,league_name",
                    returning="minimal",
                )
                .execute(),
                label="upsert_league_ratings",
            )
        if fence_params is not None:
            _require_rpc_count(
                response,
                expected=len(batch),
                label="League-rating replay",
            )

    # 3) match snapshots: update via RPC bulk UPDATE (never touches league)
    total = max(1, len(matches_to_update))
    rewritten = 0
    updated_rows_total = 0

    for batch in _chunk(matches_to_update, WRITE_BATCH_SIZE):
        _before_mutation()
        if fence_params is not None:
            resp = _fenced_rpc(
                {
                    "p_write_kind": "match_snapshots",
                    "p_rows": batch,
                },
                label="rpc_fenced_bulk_update_match_snapshots",
            )
        else:
            resp = _retry(
                lambda b=batch: supabase.rpc(
                    RPC_MATCH_SNAPSHOTS, {"rows": b}
                ).execute(),
                label="rpc_bulk_update_match_snapshots",
            )
        updated_rows_total += _require_rpc_count(
            resp,
            expected=len(batch),
            label="Match snapshot replay",
        )

        rewritten += len(batch)
        if progress_cb:
            progress_cb(rewritten / total)

    return {
        "target_reset": target_reset,
        "players_updated": players_updated,
        "skipped_incomplete": int(skipped_incomplete_scope),
        "matches_rewritten": int(len(matches_to_update)),
        "matches_snapshots_updated_rows": int(updated_rows_total),
        "league_ratings_rows": int(len(new_rows)),
        "league_ratings_reset_without_active_evidence": int(
            reset_without_active_evidence
        ),
        "matches_scanned_total": int(matches_scanned_total),
        "activity_players_updated": (
            int(len(valid_player_ids)) if full_reset else 0
        ),
        "activity_players_with_matches": (
            int(
                sum(
                    value is not None
                    for value in last_game_at_by_pid.values()
                )
            )
            if full_reset
            else 0
        ),
        "activity_players_without_matches": (
            int(
                sum(
                    value is None
                    for value in last_game_at_by_pid.values()
                )
            )
            if full_reset
            else 0
        ),
        "singles_replay_supported": bool(full_reset or target_managed_singles),
        "singles_players_updated": (
            int(len(singles_plan["player_updates"])) if full_reset else 0
        ),
        "singles_matches_rewritten": int(
            len(singles_plan["snapshots"])
            if full_reset
            else len(
                singles_plan["snapshots_by_league"].get(
                    target_league_norm.casefold(),
                    [],
                )
            )
        ),
        "singles_matches_scanned_total": int(singles_plan["matches_scanned"]),
        "singles_active_rated": int(singles_plan["active_rated"]),
        "singles_active_unrated": int(singles_plan["active_unrated"]),
        "singles_deleted_skipped": int(singles_plan["deleted"]),
        "singles_invalid_skipped": int(singles_plan["invalid"]),
    }
