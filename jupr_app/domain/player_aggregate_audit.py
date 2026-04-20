from __future__ import annotations

from typing import Any

import pandas as pd

from jupr_app.domain.gamification.match_facts import build_player_match_facts
from jupr_app.domain.match_filters import apply_match_filters_with_audit, normalize_player_id

_PLAYER_SLOT_COLUMNS = ["t1_p1", "t1_p2", "t2_p1", "t2_p2"]


def compute_player_aggregate_reconciliation(
    ctx: Any,
    *,
    player_id: int,
    club_id: str,
    league_id: str | None = None,
    filtered_matches: pd.DataFrame | None = None,
    match_audit: Any | None = None,
) -> dict[str, Any]:
    pid = int(player_id)
    club = str(club_id)

    players_row = _lookup_player_row(getattr(ctx, "df_players_all", pd.DataFrame()), pid)

    df_matches = getattr(ctx, "df_matches", pd.DataFrame())
    player_raw = _filter_player_rows(df_matches, pid)

    context_filters: dict[str, Any] = {"club_id": club, "exclude_popups": True}
    if league_id:
        context_filters["league_name"] = str(league_id).strip()

    if filtered_matches is None or match_audit is None:
        filtered_matches, match_audit = apply_match_filters_with_audit(player_raw, context_filters)
    else:
        filtered_matches = _filter_player_rows(filtered_matches, pid)

    facts = build_player_match_facts(
        ctx,
        df_matches_override=filtered_matches,
        club_id_override=club,
    )
    facts_player = facts[facts["player_id"] == pid].copy() if "player_id" in facts.columns else pd.DataFrame()
    wins = facts_player[facts_player["win"] == True].copy() if "win" in facts_player.columns else pd.DataFrame()
    losses = facts_player[facts_player["win"] == False].copy() if "win" in facts_player.columns else pd.DataFrame()

    dup_rows = 0
    if filtered_matches is not None and not filtered_matches.empty:
        match_id_col = "id" if "id" in filtered_matches.columns else "match_id" if "match_id" in filtered_matches.columns else None
        if match_id_col:
            dup_rows = int(filtered_matches.duplicated(subset=[match_id_col], keep=False).sum())

    diagnostics: dict[str, Any] = {
        "player_id": pid,
        "players_table_wins": int(players_row.get("wins", 0)),
        "players_table_losses": int(players_row.get("losses", 0)),
        "players_table_matches_played": int(players_row.get("matches_played", 0)),
        "raw_player_match_rows": int(len(player_raw)),
        "filtered_player_match_rows": int(len(filtered_matches.index) if isinstance(filtered_matches, pd.DataFrame) else 0),
        "filtered_match_win_rows": int(len(wins)),
        "filtered_match_loss_rows": int(len(losses)),
        "filtered_match_distinct_match_ids": _distinct_count(facts_player, "match_id"),
        "filtered_match_distinct_win_match_ids": _distinct_count(wins, "match_id"),
        "filtered_match_distinct_loss_match_ids": _distinct_count(losses, "match_id"),
        "excluded_match_count_by_filter_step": _step_removed_counts(match_audit),
        "popup_match_count_for_player": _count_mask(player_raw, lambda d: d.get("match_type", "").fillna("").astype(str).str.upper() == "POPUP"),
        "tournament_context_match_count_for_player": _count_mask(
            player_raw,
            lambda d: (
                d.get("context_type", "").fillna("").astype(str).str.upper() == "TOURNAMENT"
            )
            | d.get("tournament_id", pd.Series(index=d.index)).notna()
            | (d.get("match_type", "").fillna("").astype(str).str.upper() == "TOURNAMENT"),
        ),
        "invalid_or_void_match_count_for_player": _invalid_void_count(player_raw),
        "invalid_or_missing_score_match_count_for_player": _score_invalid_count(player_raw),
        "matches_missing_required_player_slots_for_facts": _missing_required_slots_count(filtered_matches),
        "filtered_duplicate_match_rows_for_player": int(dup_rows),
    }

    diagnostics["wins_delta"] = diagnostics["players_table_wins"] - diagnostics["filtered_match_distinct_win_match_ids"]
    diagnostics["losses_delta"] = diagnostics["players_table_losses"] - diagnostics["filtered_match_distinct_loss_match_ids"]
    diagnostics["matches_delta"] = diagnostics["players_table_matches_played"] - diagnostics["filtered_match_distinct_match_ids"]
    diagnostics["aggregate_out_of_sync_warning"] = bool(
        diagnostics["wins_delta"] != 0 or diagnostics["losses_delta"] != 0 or diagnostics["matches_delta"] != 0
    )

    return diagnostics


def _lookup_player_row(df_players: pd.DataFrame | None, player_id: int) -> dict[str, int]:
    defaults = {"wins": 0, "losses": 0, "matches_played": 0}
    if df_players is None or df_players.empty or "id" not in df_players.columns:
        return defaults
    hit = df_players[df_players["id"].astype(int) == int(player_id)]
    if hit.empty:
        return defaults
    row = hit.iloc[0]
    return {
        "wins": int(pd.to_numeric(row.get("wins", 0), errors="coerce") or 0),
        "losses": int(pd.to_numeric(row.get("losses", 0), errors="coerce") or 0),
        "matches_played": int(pd.to_numeric(row.get("matches_played", 0), errors="coerce") or 0),
    }


def _filter_player_rows(df_matches: pd.DataFrame | None, player_id: int) -> pd.DataFrame:
    if df_matches is None or df_matches.empty:
        return pd.DataFrame()
    df = df_matches.copy()
    mask = pd.Series(False, index=df.index)
    for col in [c for c in _PLAYER_SLOT_COLUMNS if c in df.columns]:
        mask = mask | df[col].map(lambda value: normalize_player_id(value) == int(player_id))
    return df[mask].copy()


def _distinct_count(df: pd.DataFrame, col: str) -> int:
    if df is None or df.empty or col not in df.columns:
        return 0
    return int(df[col].dropna().nunique())


def _step_removed_counts(audit: Any | None) -> dict[str, int]:
    if audit is None:
        return {}
    steps = getattr(audit, "steps", []) or []
    return {str(step.step_name): int(len(step.removed_match_ids or [])) for step in steps}


def _count_mask(df: pd.DataFrame, mask_builder) -> int:
    if df is None or df.empty:
        return 0
    try:
        mask = mask_builder(df)
        return int(mask.fillna(False).astype(bool).sum())
    except Exception:
        return 0


def _invalid_void_count(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    mask = pd.Series(False, index=df.index)
    for col in ["is_valid", "is_void", "voided", "is_voided", "invalid", "is_invalid", "deleted", "is_deleted"]:
        if col not in df.columns:
            continue
        if col == "is_valid":
            mask = mask | (~df[col].fillna(True).astype(bool))
        else:
            mask = mask | df[col].fillna(False).astype(bool)
    return int(mask.sum())


def _score_invalid_count(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    if "score_t1" not in df.columns or "score_t2" not in df.columns:
        return 0
    s1 = pd.to_numeric(df["score_t1"], errors="coerce").fillna(0).astype(int)
    s2 = pd.to_numeric(df["score_t2"], errors="coerce").fillna(0).astype(int)
    return int(((s1 + s2) <= 0).sum())


def _missing_required_slots_count(df: pd.DataFrame | None) -> int:
    if df is None or df.empty:
        return 0
    if "t1_p1" not in df.columns or "t2_p1" not in df.columns:
        return 0
    p1 = df["t1_p1"].map(normalize_player_id)
    p3 = df["t2_p1"].map(normalize_player_id)
    return int((p1.isna() | p3.isna()).sum())
