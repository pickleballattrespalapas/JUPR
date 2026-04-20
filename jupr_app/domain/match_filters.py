from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class MatchFilterAuditStep:
    step_name: str
    before_count: int
    after_count: int
    removed_match_ids: list[str] = field(default_factory=list)


@dataclass
class MatchFilterAudit:
    raw_match_ids: list[str] = field(default_factory=list)
    final_match_ids: list[str] = field(default_factory=list)
    steps: list[MatchFilterAuditStep] = field(default_factory=list)


def apply_match_filters(df_matches: pd.DataFrame, context_filters: dict | None) -> pd.DataFrame:
    return _apply_match_filters(df_matches, context_filters, audit=None)


def apply_match_filters_with_audit(
    df_matches: pd.DataFrame, context_filters: dict | None
) -> tuple[pd.DataFrame, MatchFilterAudit]:
    audit = MatchFilterAudit()
    filtered = _apply_match_filters(df_matches, context_filters, audit=audit)
    return filtered, audit


def summarize_match_filter_audit(audit: MatchFilterAudit) -> dict[str, object]:
    return {
        "raw_count": len(audit.raw_match_ids),
        "final_count": len(audit.final_match_ids),
        "steps": [
            {
                "step_name": step.step_name,
                "before_count": int(step.before_count),
                "after_count": int(step.after_count),
                "removed_count": int(step.before_count - step.after_count),
                "removed_match_ids": list(step.removed_match_ids),
            }
            for step in audit.steps
        ],
    }


def resolve_match_id_column(df_matches: pd.DataFrame) -> str:
    for col in ("match_id", "id", "matchId", "matchID"):
        if col in df_matches.columns:
            return col
    raise ValueError(
        "Match filter audit requires a match id column. "
        f"Expected one of match_id/id/matchId/matchID, found {list(df_matches.columns)}."
    )


def _normalize_context_value(value) -> str:
    return str(value or "").strip().upper()


def is_tournament_match(row: dict | pd.Series) -> bool:
    context_type = _normalize_context_value(row.get("context_type"))
    match_type = _normalize_context_value(row.get("match_type"))
    return (
        context_type == "TOURNAMENT"
        or row.get("tournament_id") is not None
        or match_type == "TOURNAMENT"
    )


def is_popup_event_match(row: dict | pd.Series) -> bool:
    if is_tournament_match(row):
        return False
    match_type = _normalize_context_value(row.get("match_type"))
    context_type = _normalize_context_value(row.get("context_type"))
    return match_type == "POPUP" or context_type == "EVENT"


def _apply_match_filters(
    df_matches: pd.DataFrame, context_filters: dict | None, audit: MatchFilterAudit | None
) -> pd.DataFrame:
    if df_matches is None or df_matches.empty:
        return pd.DataFrame()

    df = df_matches.copy()
    context_filters = context_filters or {}

    match_id_col = None
    if audit is not None:
        match_id_col = resolve_match_id_column(df)
        audit.raw_match_ids = _match_id_list(df, match_id_col)

    club_id = context_filters.get("club_id")
    if club_id is not None and "club_id" in df.columns:
        before = df
        df = df[df["club_id"].astype(str) == str(club_id)].copy()
        _record_audit_step(audit, "club_id", before, df, match_id_col)

    league_name = context_filters.get("league_name")
    if league_name:
        before = df
        df["league"] = df.get("league", "").fillna("").astype(str).str.strip()
        df = df[df["league"] == str(league_name).strip()].copy()
        _record_audit_step(audit, "league_name", before, df, match_id_col)

    if context_filters.get("exclude_popups", False) and "match_type" in df.columns:
        before = df
        df = df[df["match_type"].fillna("") != "PopUp"].copy()
        _record_audit_step(audit, "exclude_popups", before, df, match_id_col)

    if "is_valid" in df.columns:
        before = df
        df = df[df["is_valid"].fillna(True).astype(bool)].copy()
        _record_audit_step(audit, "is_valid", before, df, match_id_col)

    if "context_type" in df.columns:
        before = df
        df = df[df["context_type"].map(_normalize_context_value) != "TOURNAMENT"].copy()
        _record_audit_step(audit, "exclude_context_type_tournament", before, df, match_id_col)

    if "tournament_id" in df.columns:
        before = df
        df = df[df["tournament_id"].isna()].copy()
        _record_audit_step(audit, "tournament_id_is_null", before, df, match_id_col)

    if "match_type" in df.columns:
        before = df
        df = df[df["match_type"].map(_normalize_context_value) != "TOURNAMENT"].copy()
        _record_audit_step(audit, "exclude_match_type_tournament", before, df, match_id_col)

    for col in [
        "is_void",
        "voided",
        "is_voided",
        "invalid",
        "is_invalid",
        "deleted",
        "is_deleted",
    ]:
        if col in df.columns:
            before = df
            df = df[~df[col].fillna(False).astype(bool)].copy()
            _record_audit_step(audit, f"exclude_{col}", before, df, match_id_col)

    start_date = context_filters.get("start_date")
    end_date = context_filters.get("end_date")
    df["date_dt"] = pd.to_datetime(df.get("date", None), utc=True, errors="coerce")
    if start_date is not None:
        start_dt = pd.to_datetime(start_date, utc=True, errors="coerce")
        if pd.notna(start_dt):
            before = df
            df = df[df["date_dt"] >= start_dt].copy()
            _record_audit_step(audit, "start_date", before, df, match_id_col)
    if end_date is not None:
        end_dt = pd.to_datetime(end_date, utc=True, errors="coerce")
        if pd.notna(end_dt):
            before = df
            df = df[df["date_dt"] <= end_dt].copy()
            _record_audit_step(audit, "end_date", before, df, match_id_col)

    eligible_ids = context_filters.get("eligible_player_ids")
    if eligible_ids:
        eligible_set = {int(pid) for pid in eligible_ids}
        player_cols = [c for c in ["t1_p1", "t1_p2", "t2_p1", "t2_p2"] if c in df.columns]
        if player_cols:
            before = df
            mask = pd.Series(True, index=df.index)
            for col in player_cols:
                values = df[col]
                keep = values.isna() | values.map(lambda x: normalize_player_id(x) in eligible_set)
                mask &= keep
            df = df[mask].copy()
            _record_audit_step(audit, "eligible_player_ids", before, df, match_id_col)

    if audit is not None and match_id_col is not None:
        audit.final_match_ids = _match_id_list(df, match_id_col)

    return df


def _match_id_list(df: pd.DataFrame, match_id_col: str) -> list[str]:
    if df is None or df.empty:
        return []
    return df[match_id_col].astype(str).tolist()


def _record_audit_step(
    audit: MatchFilterAudit | None,
    step_name: str,
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    match_id_col: str | None,
) -> None:
    if audit is None or match_id_col is None:
        return
    before_ids = _match_id_list(before_df, match_id_col)
    after_ids = _match_id_list(after_df, match_id_col)
    after_set = set(after_ids)
    removed_ids = [mid for mid in before_ids if mid not in after_set]
    audit.steps.append(
        MatchFilterAuditStep(
            step_name=step_name,
            before_count=len(before_ids),
            after_count=len(after_ids),
            removed_match_ids=removed_ids,
        )
    )


def normalize_player_id(value) -> int | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        pid = int(value)
        if pid <= 0:
            return None
        return pid
    except Exception:
        return None


def normalize_score(value) -> int:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return 0
        return int(value)
    except Exception:
        return 0
