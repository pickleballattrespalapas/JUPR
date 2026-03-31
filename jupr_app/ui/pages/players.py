import html
import json
import logging
import math
import re
import textwrap
from urllib.parse import urlencode

import streamlit as st
import pandas as pd
from streamlit.components.v1 import html as st_html

from jupr_app.domain.gamification.badge_copy import build_badge_copy_plain
from jupr_app.ui.components.badge_cards import render_inline_badge_text
from jupr_app.ui.helpers import (
    qp_get,
    display_requirement_text,
)
from jupr_app.ui.layout import page_shell
from jupr_app.domain.gamification.profile import (
    build_gamification_summary,
)
from jupr_app.domain.gamification.requirements import load_requirements_map
from jupr_app.domain.gamification.top_performer_awards import TOP_PERFORMER_BADGE_IDS
from jupr_app.domain.gamification.trophies import get_player_tournament_trophies
from jupr_app.domain.live_social import (
    SOCIAL_TABLES_INSTALL_MESSAGE,
    is_missing_social_tables_error,
)

logger = logging.getLogger(__name__)

try:
    import altair as alt
except Exception:
    alt = None

# Optional: only needed for "league replay" trend charts
try:
    from jupr_app.domain.ratings import calculate_hybrid_elo
    from jupr_app.domain.constants import DEFAULT_K_FACTOR, MIN_WIN_DELTA_ELO, CAP_LOSER_GAIN_ELO
    _LEAGUE_REPLAY_AVAILABLE = True
except Exception:
    calculate_hybrid_elo = None
    DEFAULT_K_FACTOR = 32
    MIN_WIN_DELTA_ELO = 1.0
    CAP_LOSER_GAIN_ELO = 16.0
    _LEAGUE_REPLAY_AVAILABLE = False


@st.cache_data(ttl=30)
def fetch_player_matches(_supabase, club_id: str, pid: int, limit: int = 600) -> pd.DataFrame:
    """
    Tries to fetch snapshot columns (t*_r / t*_r_end). Falls back gracefully if missing.
    """
    base_select = (
        "id,date,league,match_type,score_t1,score_t2,"
        "t1_p1,t1_p2,t2_p1,t2_p2,"
        "elo_delta"
    )

    snap_select = (
        base_select
        + ",t1_p1_r,t1_p1_r_end,t1_p2_r,t1_p2_r_end,"
          "t2_p1_r,t2_p1_r_end,t2_p2_r,t2_p2_r_end"
    )

    def _run(select_cols: str):
        resp = (
            _supabase.table("matches")
            .select(select_cols)
            .eq("club_id", str(club_id))
            .or_(f"t1_p1.eq.{pid},t1_p2.eq.{pid},t2_p1.eq.{pid},t2_p2.eq.{pid}")
            .order("date", desc=True)
            .order("id", desc=True)
            .limit(int(limit))
            .execute()
        )
        return pd.DataFrame(resp.data or [])

    try:
        return _run(snap_select)
    except Exception:
        return _run(base_select)


@st.cache_data(ttl=60)
def fetch_player_badges(_supabase, club_id: str, pid: int) -> pd.DataFrame:
    try:
        resp = (
            _supabase.table("player_badges")
            .select("player_id,badge_id,earned_at,context_type,context_id,match_id,value_num,value_json")
            .eq("club_id", str(club_id))
            .eq("player_id", int(pid))
            .execute()
        )
        pb_df = pd.DataFrame(resp.data or [])
    except Exception:
        logger.exception("Failed to load player_badges")
        return pd.DataFrame()

    if pb_df.empty or "badge_id" not in pb_df.columns:
        return pd.DataFrame()

    badge_ids = pb_df["badge_id"].dropna().astype(str).unique().tolist()
    if not badge_ids:
        return pd.DataFrame()

    try:
        b_resp = (
            _supabase.table("badges")
            .select(
                "badge_id,name,prestige,category,is_stackable,is_active,rarity,tier,"
                "icon_key,scope"
            )
            .in_("badge_id", badge_ids)
            .execute()
        )
        badges_df = pd.DataFrame(b_resp.data or [])
    except Exception:
        logger.exception("Failed to load badges definitions")
        return pd.DataFrame()

    if badges_df.empty:
        return pd.DataFrame()

    requirements_map = load_requirements_map()
    badges_df["requirements"] = badges_df["badge_id"].map(requirements_map).fillna("Requirements TBD")

    return pb_df.merge(badges_df, on="badge_id", how="left")


# Keep this cache short so newly finalized tournament podium trophies appear quickly.
@st.cache_data(ttl=60)
def fetch_player_tournament_trophies(_supabase, club_id: str, pid: int) -> list[dict]:
    return get_player_tournament_trophies(_supabase, club_id, pid)


@st.cache_data(ttl=120)
def fetch_badge_definitions(_supabase) -> pd.DataFrame:
    try:
        resp = (
            _supabase.table("badges")
            .select(
                "badge_id,name,prestige,category,is_stackable,is_active,rarity,"
                "tier,icon_key,scope,created_at"
            )
            .execute()
        )
        df = pd.DataFrame(resp.data or [])
        if not df.empty:
            requirements_map = load_requirements_map()
            df["requirements"] = df["badge_id"].map(requirements_map).fillna("Requirements TBD")
        return df
    except Exception:
        logger.exception("Failed to load badge definitions")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def fetch_player_stories(_supabase, club_id: str, pid: int, limit: int = 6) -> pd.DataFrame:
    try:
        resp = (
            _supabase.table("player_stories")
            .select("story_type,context_id,created_at,title,body,importance,match_id")
            .eq("club_id", str(club_id))
            .eq("player_id", int(pid))
            .order("created_at", desc=True)
            .limit(int(limit) * 3)
            .execute()
        )
        return pd.DataFrame(resp.data or [])
    except Exception:
        logger.exception("Failed to load player stories")
        return pd.DataFrame()


BADGE_ICONS = {
    "participant": "🎟️",
    "dedicated_participant_50": "🧭",
    "lifetime_participant_200": "🏅",
    "mountain_climber": "🧗",
    "breakthrough": "🚀",
    "above_expectations": "⭐",
    "clutch_performer": "⚡",
    "dominant_run": "🔥",
    "high_output": "💥",
    "battle_tested": "🛡️",
    "consistency": "🎯",
    "giant_slayer": "🗡️",
    "upset_champion": "👑",
    "league_champion": "🥇",
    "league_runner_up": "🥈",
    "league_third_place": "🥉",
    "tournament_champion": "🥇",
    "tournament_runner_up": "🥈",
    "tournament_third_place": "🥉",
    "podium": "🏅",
}


def badge_icon(badge_id: str, category: str | None = None) -> str:
    return BADGE_ICONS.get(str(badge_id), "🏆")


def _season_sort_key(league_name: str) -> tuple[int, int] | None:
    name = str(league_name or "").strip()
    if not name:
        return None
    lowered = name.lower()
    season_order = {"winter": 1, "spring": 2, "summer": 3, "fall": 4}
    season_rank = None
    for season, rank in season_order.items():
        if season in lowered:
            season_rank = rank
            break
    years = [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", lowered)]
    year = max(years) if years else None
    if year is None and season_rank is None:
        return None
    return (year or 0, season_rank or 0)


def _parse_week_num(week_tag: str | None) -> int | None:
    if week_tag is None:
        return None
    match = re.search(r"(\d+)", str(week_tag))
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _build_league_display_label(league_name: str, season_label: str | None) -> str:
    league_name = str(league_name or "").strip()
    season_label = str(season_label or "").strip()
    if not season_label:
        return league_name
    if season_label.lower() in league_name.lower():
        return league_name
    return f"{league_name} • {season_label}"


def _inactive_league_frame(
    df_meta: pd.DataFrame | None,
    df_leagues: pd.DataFrame | None,
    df_matches: pd.DataFrame | None,
) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    inactive_df = pd.DataFrame()
    if df_meta is not None and not df_meta.empty and "league_name" in df_meta.columns:
        meta = df_meta.copy()
        meta["league_name"] = meta["league_name"].astype(str).str.strip()
        inactive_mask = pd.Series(False, index=meta.index)
        if "is_active" in meta.columns:
            inactive_mask |= meta["is_active"].fillna(False) == False
        if "status" in meta.columns:
            inactive_mask |= (
                meta["status"]
                .fillna("")
                .astype(str)
                .str.lower()
                .isin({"archived", "completed", "complete", "done"})
            )
        end_col = next(
            (col for col in ["end_date", "ended_at", "end_at", "season_end", "final_date"] if col in meta.columns),
            None,
        )
        if end_col:
            end_dates = pd.to_datetime(meta[end_col], errors="coerce", utc=True)
            inactive_mask |= end_dates < now
            meta["sort_date"] = end_dates
        else:
            meta["sort_date"] = pd.NaT

        inactive_df = meta.loc[inactive_mask].copy()
        if not inactive_df.empty:
            season_col = next(
                (col for col in ["season_label", "season", "season_name"] if col in inactive_df.columns),
                None,
            )
            inactive_df["season_label"] = (
                inactive_df[season_col].fillna("").astype(str).str.strip() if season_col else ""
            )
            inactive_df = inactive_df[["league_name", "season_label", "sort_date"]]

    if inactive_df.empty and df_leagues is not None and not df_leagues.empty and "league_name" in df_leagues.columns:
        leagues = df_leagues.copy()
        leagues["league_name"] = leagues["league_name"].astype(str).str.strip()
        if "is_active" in leagues.columns:
            league_active = leagues.groupby("league_name")["is_active"].apply(lambda s: bool(s.fillna(False).any()))
            inactive_names = league_active[league_active == False].index.tolist()
        else:
            inactive_names = []
        if inactive_names:
            inactive_df = pd.DataFrame({"league_name": inactive_names})
        else:
            inactive_df = pd.DataFrame()
        inactive_df["season_label"] = ""
        inactive_df["sort_date"] = pd.NaT

    if inactive_df.empty:
        return inactive_df

    inactive_df = inactive_df[inactive_df["league_name"].str.upper() != "OVERALL"].copy()

    if (
        df_matches is not None
        and not df_matches.empty
        and "league" in df_matches.columns
        and "date" in df_matches.columns
    ):
        match_df = df_matches.copy()
        match_df["league"] = match_df["league"].fillna("").astype(str).str.strip()
        match_df["date_dt"] = pd.to_datetime(match_df["date"], errors="coerce", utc=True)
        last_dates = match_df.dropna(subset=["date_dt"]).groupby("league")["date_dt"].max()
        inactive_df["match_date"] = inactive_df["league_name"].map(last_dates)
        inactive_df["sort_date"] = inactive_df["sort_date"].combine_first(inactive_df["match_date"])

    inactive_df["season_sort"] = inactive_df["league_name"].map(_season_sort_key)
    inactive_df["display_label"] = inactive_df.apply(
        lambda r: _build_league_display_label(r["league_name"], r.get("season_label", "")), axis=1
    )
    return inactive_df


def _social_skill_levels_from_summary(summary_json: object) -> list[str]:
    payload = summary_json if isinstance(summary_json, dict) else {}
    tags = payload.get("event_tags") if isinstance(payload, dict) else {}
    skill_levels = []
    if isinstance(tags, dict):
        raw_levels = tags.get("skill_levels")
        if isinstance(raw_levels, str):
            raw_levels = [raw_levels]
        if isinstance(raw_levels, (list, tuple, set)):
            for value in raw_levels:
                text = str(value or "").strip()
                if text and text not in skill_levels:
                    skill_levels.append(text)
    if not skill_levels:
        return ["All"]
    if "All" in skill_levels:
        return ["All"]
    return skill_levels


@st.cache_data(ttl=60)
def fetch_player_social_event_history(_supabase, club_id: str, pid: int, limit: int = 100) -> pd.DataFrame:
    try:
        events_resp = (
            _supabase.table("live_events")
            .select("id,name,event_type,event_date,submitted_by,status,result_mode,summary_json")
            .eq("club_id", str(club_id))
            .eq("result_mode", "social_unrated")
            .eq("status", "saved")
            .order("event_date", desc=True)
            .limit(max(int(limit), 1))
            .execute()
        )
    except Exception as exc:
        if is_missing_social_tables_error(exc):
            return pd.DataFrame([{"_missing_social_tables": True, "_social_message": SOCIAL_TABLES_INSTALL_MESSAGE}])
        raise

    events = events_resp.data or []
    event_ids = [str(row.get("id")) for row in events if row.get("id")]
    if not event_ids:
        return pd.DataFrame()

    try:
        participants_resp = (
            _supabase.table("live_event_participants")
            .select("id,event_id,club_person_id,linked_player_id")
            .in_("event_id", event_ids)
            .execute()
        )
        participants_df = pd.DataFrame(participants_resp.data or [])
        if participants_df.empty:
            return pd.DataFrame()

        person_match_df = participants_df[participants_df.get("linked_player_id") == int(pid)].copy()
        if person_match_df.empty and "club_person_id" in participants_df.columns:
            club_people_resp = (
                _supabase.table("club_people")
                .select("id,linked_player_id")
                .eq("club_id", str(club_id))
                .eq("linked_player_id", int(pid))
                .execute()
            )
            club_people_df = pd.DataFrame(club_people_resp.data or [])
            if not club_people_df.empty and "id" in club_people_df.columns:
                cp_ids = club_people_df["id"].dropna().astype(str).tolist()
                if cp_ids:
                    person_match_df = participants_df[
                        participants_df.get("club_person_id").astype(str).isin(cp_ids)
                    ].copy()

        if person_match_df.empty:
            return pd.DataFrame()
        target_participant_ids = set(person_match_df["id"].dropna().astype(str).tolist())

        matches_resp = (
            _supabase.table("live_event_matches")
            .select(
                "event_id,played_on,t1_p1_participant_id,t1_p2_participant_id,t2_p1_participant_id,t2_p2_participant_id,score_t1,score_t2"
            )
            .in_("event_id", event_ids)
            .execute()
        )
    except Exception as exc:
        if is_missing_social_tables_error(exc):
            return pd.DataFrame([{"_missing_social_tables": True, "_social_message": SOCIAL_TABLES_INSTALL_MESSAGE}])
        raise

    events_by_id = {str(row.get("id")): row for row in events if row.get("id")}
    history_rows: list[dict] = []
    for event_id in sorted(person_match_df["event_id"].dropna().astype(str).unique().tolist()):
        event_row = events_by_id.get(str(event_id))
        if not event_row:
            continue
        summary_json = event_row.get("summary_json")
        skill_tags = _social_skill_levels_from_summary(summary_json)
        history_rows.append(
            {
                "event_id": str(event_id),
                "Date": event_row.get("event_date"),
                "Event": str(event_row.get("name") or "Social Event").strip() or "Social Event",
                "Event Type": str(event_row.get("event_type") or "social_unrated").strip() or "social_unrated",
                "Skill Tags": ", ".join(skill_tags),
                "_skill_tags_list": skill_tags,
                "Matches": 0,
                "Wins": 0,
                "Losses": 0,
                "Diff": 0,
                "Submitted By": str(event_row.get("submitted_by") or "").strip(),
            }
        )

    if not history_rows:
        return pd.DataFrame()

    history_df = pd.DataFrame(history_rows)
    matches_df = pd.DataFrame(matches_resp.data or [])
    if matches_df.empty:
        return history_df.sort_values(["Date", "event_id"], ascending=[False, False]).reset_index(drop=True)

    for _, row in matches_df.iterrows():
        s1 = int(row.get("score_t1") or 0)
        s2 = int(row.get("score_t2") or 0)
        if (s1 + s2) <= 0 or s1 == s2:
            continue
        event_id = str(row.get("event_id") or "")
        if not event_id or event_id not in set(history_df["event_id"].astype(str)):
            continue

        team1_ids = {str(row.get("t1_p1_participant_id") or ""), str(row.get("t1_p2_participant_id") or "")}
        team2_ids = {str(row.get("t2_p1_participant_id") or ""), str(row.get("t2_p2_participant_id") or "")}
        on_team1 = bool(team1_ids & target_participant_ids)
        on_team2 = bool(team2_ids & target_participant_ids)
        if not on_team1 and not on_team2:
            continue

        idx = history_df.index[history_df["event_id"].astype(str) == event_id]
        if len(idx) == 0:
            continue
        i = idx[0]
        history_df.at[i, "Matches"] = int(history_df.at[i, "Matches"]) + 1
        if on_team1:
            won = s1 > s2
            diff = s1 - s2
        else:
            won = s2 > s1
            diff = s2 - s1
        if won:
            history_df.at[i, "Wins"] = int(history_df.at[i, "Wins"]) + 1
        else:
            history_df.at[i, "Losses"] = int(history_df.at[i, "Losses"]) + 1
        history_df.at[i, "Diff"] = int(history_df.at[i, "Diff"]) + int(diff)

    history_df["Date"] = pd.to_datetime(history_df["Date"], utc=True, errors="coerce")
    history_df = history_df.sort_values(["Date", "event_id"], ascending=[False, False]).reset_index(drop=True)
    return history_df


@st.cache_data(ttl=60)
def fetch_player_social_participation(_supabase, club_id: str, pid: int) -> dict:
    history_df = fetch_player_social_event_history(_supabase, club_id, pid, limit=400)
    if history_df.empty:
        return {"available": True, "history_df": history_df}
    if bool(history_df.get("_missing_social_tables", pd.Series(dtype=bool)).fillna(False).any()):
        return {
            "available": False,
            "message": str(history_df.iloc[0].get("_social_message") or SOCIAL_TABLES_INSTALL_MESSAGE),
            "history_df": pd.DataFrame(),
        }

    history_df = history_df.copy()
    history_df["Matches"] = pd.to_numeric(history_df.get("Matches"), errors="coerce").fillna(0).astype(int)
    history_df["Wins"] = pd.to_numeric(history_df.get("Wins"), errors="coerce").fillna(0).astype(int)
    history_df["Losses"] = pd.to_numeric(history_df.get("Losses"), errors="coerce").fillna(0).astype(int)
    history_df["Diff"] = pd.to_numeric(history_df.get("Diff"), errors="coerce").fillna(0).astype(int)

    events = int(len(history_df))
    matches = int(history_df["Matches"].sum())
    wins = int(history_df["Wins"].sum())
    losses = int(history_df["Losses"].sum())
    diff = int(history_df["Diff"].sum())
    last_dt = pd.to_datetime(history_df.get("Date"), utc=True, errors="coerce").max()
    last_appearance = last_dt.strftime("%Y-%m-%d") if pd.notna(last_dt) else "—"

    buckets: dict[str, dict[str, int | str]] = {}
    for _, row in history_df.iterrows():
        skill_levels = row.get("_skill_tags_list")
        if isinstance(skill_levels, str):
            skill_levels = [skill_levels]
        if not isinstance(skill_levels, list) or not skill_levels:
            skill_levels = ["All"]
        if "All" in skill_levels:
            target_levels = ["All"]
        else:
            target_levels = [str(level).strip() for level in skill_levels if str(level).strip()] or ["All"]
        for level in target_levels:
            bucket = buckets.setdefault(
                level,
                {"Skill Level": level, "Events": 0, "Matches": 0, "Wins": 0, "Losses": 0, "Diff": 0},
            )
            bucket["Events"] += 1
            bucket["Matches"] += int(row.get("Matches") or 0)
            bucket["Wins"] += int(row.get("Wins") or 0)
            bucket["Losses"] += int(row.get("Losses") or 0)
            bucket["Diff"] += int(row.get("Diff") or 0)

    skill_breakdown_df = (
        pd.DataFrame(list(buckets.values()))
        .sort_values(["Skill Level"], ascending=[True])
        .reset_index(drop=True)
        if buckets
        else pd.DataFrame(columns=["Skill Level", "Events", "Matches", "Wins", "Losses", "Diff"])
    )

    return {
        "available": True,
        "history_df": history_df,
        "summary": {
            "events": events,
            "matches": matches,
            "wins": wins,
            "losses": losses,
            "record": f"{wins}-{losses}",
            "diff": diff,
            "last_appearance": last_appearance,
        },
        "skill_breakdown_df": skill_breakdown_df,
    }


def build_inactive_league_options(
    df_meta: pd.DataFrame | None,
    df_leagues: pd.DataFrame | None,
    df_matches: pd.DataFrame | None,
) -> pd.DataFrame:
    inactive_df = _inactive_league_frame(df_meta, df_leagues, df_matches)
    if inactive_df.empty:
        return inactive_df

    with_dates = inactive_df[inactive_df["sort_date"].notna()].copy()
    without_dates = inactive_df[inactive_df["sort_date"].isna()].copy()
    if not with_dates.empty:
        with_dates = with_dates.sort_values("sort_date", ascending=False)
    if not without_dates.empty:
        without_dates = without_dates.sort_values(
            by="season_sort",
            key=lambda s: s.apply(lambda v: v if v is not None else (0, 0)),
            ascending=False,
        )
    return pd.concat([with_dates, without_dates], ignore_index=True)


def _filter_league_matches(df_matches: pd.DataFrame, league_name: str) -> pd.DataFrame:
    df = df_matches.copy()
    df["league"] = df.get("league", "").fillna("").astype(str).str.strip()
    df = df[df["league"] == str(league_name).strip()].copy()
    if "match_type" in df.columns:
        df["match_type"] = df.get("match_type", "").fillna("").astype(str).str.strip()
        df = df[df["match_type"] != "PopUp"].copy()
    df["score_t1"] = pd.to_numeric(df.get("score_t1", 0), errors="coerce").fillna(0).astype(int)
    df["score_t2"] = pd.to_numeric(df.get("score_t2", 0), errors="coerce").fillna(0).astype(int)
    df = df[(df["score_t1"] + df["score_t2"]) > 0].copy()
    week_src = df.get("week_num")
    if week_src is not None:
        df["week_num"] = pd.to_numeric(week_src, errors="coerce")
    else:
        df["week_num"] = df.get("week_tag", "").map(_parse_week_num)
    return df


def _parse_value_json(raw: object) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def filter_player_league_trophies(
    player_badges: pd.DataFrame | None,
    player_id: int,
    league_id: str,
) -> pd.DataFrame:
    if player_badges is None or player_badges.empty:
        return pd.DataFrame()

    league_key = str(league_id).strip()
    df = player_badges.copy()
    if "player_id" in df.columns:
        df = df[pd.to_numeric(df["player_id"], errors="coerce").fillna(-1).astype(int) == int(player_id)].copy()
    if df.empty:
        return pd.DataFrame()

    def _match_league(row: pd.Series) -> bool:
        value_json = _parse_value_json(row.get("value_json"))
        league_val = value_json.get("league_id") or value_json.get("league")
        if league_val is not None and str(league_val).strip() == league_key:
            return True
        context_type = str(row.get("context_type", "")).strip()
        context_id = str(row.get("context_id", "")).strip()
        if context_type == "league" and context_id:
            if context_id == league_key:
                return True
            if context_id.startswith(f"{league_key}:"):
                return True
        return False

    df = df[df.apply(_match_league, axis=1)].copy()
    if "earned_at" in df.columns:
        df["earned_at_dt"] = pd.to_datetime(df.get("earned_at"), utc=True, errors="coerce")
        df = df.sort_values(["earned_at_dt"], ascending=False, na_position="last")
    return df


def _normalize_league_id(value: object | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _extract_league_id(row: pd.Series) -> str | None:
    value_json = _parse_value_json(row.get("value_json"))
    league_val = value_json.get("league_id") or value_json.get("league")
    if league_val is not None:
        return _normalize_league_id(league_val)
    context_type = str(row.get("context_type", "")).strip()
    context_id = str(row.get("context_id", "")).strip()
    if context_type == "league" and context_id:
        return _normalize_league_id(context_id.split(":")[0])
    return None


def _extract_tournament_label(row: pd.Series) -> str | None:
    value_json = _parse_value_json(row.get("value_json"))
    tournament_val = value_json.get("tournament_name") or value_json.get("tournament_id")
    if tournament_val is not None:
        return _normalize_league_id(tournament_val)
    return None


def _format_earned_at(value: object | None) -> str:
    if value is None:
        return ""
    earned_at_dt = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(earned_at_dt):
        return ""
    return earned_at_dt.date().isoformat()


def _format_top_performer_metric(category_key: str | None, metric_value: object | None) -> str:
    if metric_value is None:
        return ""
    try:
        value = float(metric_value)
    except Exception:
        return str(metric_value)
    if category_key == "highest_rating":
        return f"{value:.3f}"
    if category_key == "most_improved":
        return f"{value:+.3f}"
    if category_key == "best_win_pct":
        return f"{value:.1f}%"
    if category_key == "most_wins":
        return f"{int(round(value))}"
    return str(metric_value)


def _trophy_display_name(row: pd.Series) -> str:
    badge_id = str(row.get("badge_id") or "").strip()
    value_json = _parse_value_json(row.get("value_json"))
    if _is_top_performer_badge(badge_id):
        category_label = value_json.get("category_label")
        badge_name = row.get("badge_name") or row.get("name") or row.get("title")
        title = _format_top_performer_title(badge_name, category_label)
        rank = _format_top_performer_rank(value_json)
        return f"{title} {rank}".strip()
    for key in ["badge_name", "name", "title"]:
        value = row.get(key)
        if value is None:
            continue
        cleaned = str(value).strip()
        if cleaned:
            return cleaned
    return "Trophy"


def _is_top_performer_badge(badge_id: str | None) -> bool:
    badge_key = str(badge_id or "").strip()
    if not badge_key:
        return False
    if badge_key.startswith("top_performer_"):
        return True
    return badge_key in set(TOP_PERFORMER_BADGE_IDS.values())


def _format_top_performer_title(badge_name: str | None, category_label: str | None) -> str:
    base = str(category_label or badge_name or "Top Performer").strip()
    if not base:
        base = "Top Performer"
    if base.lower().startswith("top performer"):
        return base
    return f"Top Performer: {base}"


def _format_top_performer_rank(value_json: dict) -> str:
    rank = value_json.get("rank")
    if rank is None:
        return ""
    try:
        return f"#{int(rank)}"
    except Exception:
        return f"#{rank}"


def _decorate_trophies_with_leagues(
    trophies: pd.DataFrame,
    league_labels: dict[str, str],
) -> pd.DataFrame:
    df = trophies if isinstance(trophies, pd.DataFrame) else pd.DataFrame(trophies)
    if df.empty:
        if "prestige_num" not in df.columns:
            df["prestige_num"] = pd.Series(dtype="int64")
        if "league_name" not in df.columns:
            df["league_name"] = pd.Series(dtype="object")
        if "earned_at_display" not in df.columns:
            df["earned_at_display"] = pd.Series(dtype="object")
        return df
    df = df.copy()
    if "league_id" not in df.columns:
        df["league_id"] = pd.NA
    df["league_id"] = df.apply(_extract_league_id, axis=1).fillna(df["league_id"])
    df["league_label"] = df["league_id"].map(league_labels).fillna(df["league_id"])
    if "context_type" in df.columns:
        tournament_labels = df.apply(_extract_tournament_label, axis=1)
        df.loc[df["context_type"].astype(str).str.strip() == "tournament", "league_label"] = tournament_labels
    df["league_label"] = df["league_label"].fillna("League")
    if "earned_at" not in df.columns:
        df["earned_at"] = pd.NaT
    df["earned_at_dt"] = pd.to_datetime(df["earned_at"], utc=True, errors="coerce")
    if "prestige" in df.columns:
        prestige_series = df["prestige"]
    else:
        prestige_series = pd.Series([0] * len(df), index=df.index)
    df["prestige_num"] = (
        pd.to_numeric(prestige_series, errors="coerce")
        .fillna(0)
        .astype(int)
    )
    return df


def get_player_trophies(
    player_badges: pd.DataFrame | None,
    player_id: int,
    completed_league_ids: set[str],
    league_id: str | None = None,
    completed_only: bool = True,
) -> pd.DataFrame:
    if player_badges is None or player_badges.empty:
        return pd.DataFrame()
    df = player_badges.copy()
    if "player_id" in df.columns:
        df = df[pd.to_numeric(df["player_id"], errors="coerce").fillna(-1).astype(int) == int(player_id)].copy()
    if df.empty:
        return pd.DataFrame()
    df = _decorate_trophies_with_leagues(df, {})
    if completed_only:
        if "context_type" in df.columns:
            df = df[
                df["league_id"].isin(completed_league_ids)
                | (df["context_type"].astype(str).str.strip() == "tournament")
            ].copy()
        else:
            df = df[df["league_id"].isin(completed_league_ids)].copy()
    if league_id:
        league_key = str(league_id).strip()
        df = df[df["league_id"] == league_key].copy()
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values(["prestige_num", "earned_at_dt"], ascending=[False, False], na_position="last")
    return df


def get_player_trophy_case(
    player_badges: pd.DataFrame | None,
    player_id: int,
    completed_league_ids: set[str],
    limit: int = 8,
    completed_only: bool = True,
) -> pd.DataFrame:
    trophies = get_player_trophies(
        player_badges,
        player_id,
        completed_league_ids,
        league_id=None,
        completed_only=completed_only,
    )
    if trophies.empty:
        return trophies
    return trophies.head(int(limit))


@st.cache_data(ttl=300)
def build_league_snapshot_map(_supabase, club_id: str, league_name: str, df_meta: pd.DataFrame | None, df_players_all: pd.DataFrame | None) -> dict:
    """
    Optional “full restore”: replay league-island Elo across matches in that league.
    Returns snap_map[match_id][player_id] = (start_elo, end_elo)
    Only runs if domain imports exist; otherwise returns {}.
    """
    if not _LEAGUE_REPLAY_AVAILABLE:
        return {}

    lg = str(league_name or "").strip()
    if not lg:
        return {}

    base_select = "id,date,league,match_type,score_t1,score_t2,t1_p1,t1_p2,t2_p1,t2_p2"
    snap_select = base_select + ",t1_p1_r,t1_p2_r,t2_p1_r,t2_p2_r"

    rows = []
    used_snap_select = True
    try:
        resp = (
            _supabase.table("matches")
            .select(snap_select)
            .eq("club_id", str(club_id))
            .order("date", desc=False)
            .order("id", desc=False)
            .execute()
        )
        rows = resp.data or []
    except Exception:
        used_snap_select = False
        resp = (
            _supabase.table("matches")
            .select(base_select)
            .eq("club_id", str(club_id))
            .order("date", desc=False)
            .order("id", desc=False)
            .execute()
        )
        rows = resp.data or []

    if not rows:
        return {}

    df = pd.DataFrame(rows)
    if df.empty:
        return {}

    df["league"] = df.get("league", "").fillna("").astype(str).str.strip()
    df["match_type"] = df.get("match_type", "").fillna("").astype(str).str.strip()

    df = df[df["league"] == lg].copy()
    if df.empty:
        return {}

    # exclude PopUp only; allow NULL/blank match_type
    df = df[df["match_type"] != "PopUp"].copy()
    if df.empty:
        return {}

    df["date"] = pd.to_datetime(df.get("date", None), utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        return {}

    # K factor from meta
    k_val = int(DEFAULT_K_FACTOR)
    try:
        if df_meta is not None and not df_meta.empty and "league_name" in df_meta.columns:
            hit = df_meta[df_meta["league_name"].astype(str).str.strip() == lg]
            if not hit.empty:
                k_val = int(hit.iloc[0].get("k_factor", DEFAULT_K_FACTOR) or DEFAULT_K_FACTOR)
    except Exception:
        k_val = int(DEFAULT_K_FACTOR)

    # Seed from current overall Elo if needed
    overall_seed = {}
    try:
        if df_players_all is not None and not df_players_all.empty:
            overall_seed = dict(zip(df_players_all["id"].astype(int), df_players_all["rating"].astype(float)))
    except Exception:
        overall_seed = {}

    island = {}    # pid -> league elo
    snap_map = {}  # match_id -> {pid: (start,end)}

    def _safe_int(x, default=None):
        try:
            if x is None or str(x).strip() == "":
                return default
            return int(x)
        except Exception:
            return default

    def seed_from_row(row, pid: int) -> float:
        pid = int(pid)
        if used_snap_select:
            try:
                if pid == _safe_int(row.get("t1_p1")):
                    v = row.get("t1_p1_r", None)
                elif pid == _safe_int(row.get("t1_p2")):
                    v = row.get("t1_p2_r", None)
                elif pid == _safe_int(row.get("t2_p1")):
                    v = row.get("t2_p1_r", None)
                elif pid == _safe_int(row.get("t2_p2")):
                    v = row.get("t2_p2_r", None)
                else:
                    v = None
                if v is not None and str(v).strip() != "":
                    return float(v)
            except Exception:
                pass
        return float(overall_seed.get(pid, 1200.0))

    def get_r(row, pid: int) -> float:
        pid = int(pid)
        if pid not in island:
            island[pid] = seed_from_row(row, pid)
        return float(island[pid])

    df = df.sort_values(["date", "id"], ascending=[True, True])

    for _, m in df.iterrows():
        try:
            mid = int(m["id"])
            p1, p2, p3, p4 = int(m["t1_p1"]), int(m["t1_p2"]), int(m["t2_p1"]), int(m["t2_p2"])
            s1 = int(m.get("score_t1", 0) or 0)
            s2 = int(m.get("score_t2", 0) or 0)
        except Exception:
            continue

        if (s1 + s2) <= 0:
            continue

        r1, r2, r3, r4 = get_r(m, p1), get_r(m, p2), get_r(m, p3), get_r(m, p4)

        d1, d2 = calculate_hybrid_elo(
            (r1 + r2) / 2.0,
            (r3 + r4) / 2.0,
            s1,
            s2,
            k_factor=int(k_val),
            min_win_delta=float(MIN_WIN_DELTA_ELO),
            cap_loser_gain=float(CAP_LOSER_GAIN_ELO),
        )

        island[p1] = r1 + float(d1)
        island[p2] = r2 + float(d1)
        island[p3] = r3 + float(d2)
        island[p4] = r4 + float(d2)

        snap_map[mid] = {
            p1: (r1, island[p1]),
            p2: (r2, island[p2]),
            p3: (r3, island[p3]),
            p4: (r4, island[p4]),
        }

    return snap_map


def render(ctx):
    PUBLIC_MODE = bool(getattr(ctx, "public_mode", False))
    mode_label = "Public" if PUBLIC_MODE else "Admin"
    page_shell("🔍 Player Search", "Find players and view ratings.", mode_label=mode_label)

    df_players_all = ctx.df_players_all
    df_leagues = getattr(ctx, "df_leagues", None)
    df_meta = getattr(ctx, "df_meta", None)

    if df_players_all is None or df_players_all.empty:
        st.info("No players found.")
        return

    players_df = df_players_all.copy()
    if "inactive_at" in players_df.columns:
        players_df = players_df[players_df["inactive_at"].isna()].copy()
    elif "active" in players_df.columns:
        players_df = players_df[players_df["active"] == True].copy()

    if players_df.empty:
        st.info("No active players.")
        return

    players_df["id"] = players_df["id"].astype(int)

    pid_q = qp_get("pid", "").strip()
    pid_sig = f"pid:{pid_q}" if pid_q else ""
    last_sig = st.session_state.get("player_pid_sig_applied", "")

    if pid_q.isdigit() and pid_sig != last_sig:
        pid_int = int(pid_q)
        hit = players_df[players_df["id"] == pid_int]
        if not hit.empty:
            st.session_state["player_search_id"] = int(hit.iloc[0]["id"])
            try:
                st.query_params.pop("pid", None)
            except Exception:
                pass
        st.session_state["player_pid_sig_applied"] = pid_sig

    players_df = players_df.sort_values("name").copy()
    options = [""] + players_df["id"].tolist()

    def _fmt(x):
        if x == "":
            return ""
        r = players_df[players_df["id"] == int(x)]
        if r.empty:
            return f"#{x}"
        return f"{str(r.iloc[0]['name'])}  (#{int(x)})"

    pick_id = st.selectbox(
        "Select a player",
        options=options,
        format_func=_fmt,
        key="player_search_id",
    )

    if pick_id == "":
        st.info("Select a player to view details.")
        return

    pid = int(pick_id)
    row = players_df[players_df["id"] == pid].iloc[0]
    pick_name = str(row["name"])
    _supabase = ctx.supabase
    club_id = ctx.club_id

    try:
        current_overall_elo = float(row.get("rating", 1200.0) or 1200.0)
    except Exception:
        current_overall_elo = 1200.0
    current_jupr = current_overall_elo / 400.0

    c1, c2 = st.columns(2)
    c1.metric("Player", pick_name)
    c2.metric("Overall JUPR", f"{current_jupr:.3f}")

    tape_tab, ratings_tab, social_tab = st.tabs(["Trophy Room", "Ratings", "Social"])

    with tape_tab:
        debug_render = False
        if bool(getattr(ctx, "admin_logged_in", False)):
            debug_render = st.toggle("Debug badge render", value=False)

        def _debug_html_warning(label: str, fn_name: str, text: str) -> None:
            if not debug_render:
                return
            if "<div" in text or "badge-card" in text:
                snippet = textwrap.shorten(text.replace("\n", " "), width=140, placeholder="…")
                st.warning(f"Badge render debug ({label}) via {fn_name}: {snippet}")

        def badge_markdown(text: str, *, label: str) -> None:
            _debug_html_warning(label, "markdown", text)
            st.markdown(text)

        def badge_write(text: str, *, label: str) -> None:
            _debug_html_warning(label, "write", text)
            st.write(text)

        def badge_caption(text: str, *, label: str) -> None:
            _debug_html_warning(label, "caption", text)
            st.caption(text)

        def badge_code(text: str, *, label: str) -> None:
            _debug_html_warning(label, "code", text)
            st.code(text)

        badge_markdown("### Trophy Room", label="badges.header")

        badge_css = """
        .badge-summary {
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
            align-items: stretch;
            margin-bottom: 0.75rem;
        }
        .badge-stat {
            background: var(--panel);
            border: 1px solid var(--border);
            box-shadow: var(--shadow, none);
            border-radius: 0.75rem;
            padding: 0.75rem 0.9rem;
            min-width: 120px;
            color: var(--text-primary);
        }
        .badge-stat-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-secondary);
        }
        .badge-stat-value {
            font-size: 1.6rem;
            font-weight: 700;
        }
        .badge-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            align-items: center;
        }
        .badge-chip {
            display: inline-flex;
            gap: 0.35rem;
            align-items: center;
            padding: 0.25rem 0.5rem;
            border-radius: 999px;
            border: 1px solid var(--border);
            background: var(--pill-bg);
            font-size: 0.8rem;
            max-width: 180px;
            color: var(--text-primary);
        }
        .trophy-section {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
            margin-bottom: 0.6rem;
        }
        .trophy-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-secondary);
        }
        .trophy-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            align-items: stretch;
        }
        .trophy-league {
            font-size: 0.78rem;
            color: var(--text-muted);
        }
        .trophy-chip {
            display: inline-flex;
            gap: 0.45rem;
            align-items: flex-start;
            padding: 0.35rem 0.6rem;
            border-radius: 0.75rem;
            border: 1px solid var(--border);
            background: var(--panel);
            box-shadow: var(--shadow, none);
            font-size: 0.8rem;
            max-width: 320px;
            color: var(--text-primary);
        }
        .trophy-text {
            display: flex;
            flex-direction: column;
            gap: 0.1rem;
            min-width: 0;
        }
        .trophy-title {
            font-weight: 600;
            font-size: 0.85rem;
        }
        .trophy-body {
            font-size: 0.7rem;
            color: var(--text-muted);
        }
        .trophy-case-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.75rem;
        }
        .trophy-case-card {
            border-radius: 0.85rem;
            border: 1px solid var(--border);
            background: var(--panel);
            box-shadow: var(--shadow, none);
            padding: 0.8rem 0.85rem;
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
            color: var(--text-primary);
        }
        .trophy-case-header {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            font-weight: 600;
        }
        .trophy-case-meta {
            font-size: 0.75rem;
            color: var(--text-muted);
        }
        .badge-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.75rem;
        }
        .featured-grid .badge-card:nth-child(n+4) {
            display: none;
        }
        @media (max-width: 900px) {
            .featured-grid .badge-card:nth-child(n+3) {
                display: none;
            }
        }
        @media (max-width: 640px) {
            .featured-grid .badge-card:nth-child(n+2) {
                display: none;
            }
        }
        .badge-card {
            border-radius: 0.8rem;
            border: 1px solid var(--border);
            background: var(--panel);
            box-shadow: var(--shadow, none);
            padding: 0.7rem 0.8rem;
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
            color: var(--text-primary);
        }
        .badge-card.silhouette {
            background: var(--panel);
            opacity: 0.7;
        }
        .badge-card-header {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            font-weight: 600;
        }
        .badge-subtext {
            font-size: 0.75rem;
            color: var(--text-muted);
        }
        .badge-subtext p {
            margin: 0;
        }
        .badge-subtext ul,
        .badge-subtext ol {
            margin: 0 0 0 1rem;
            padding: 0;
        }
        .badge-subtext li {
            margin: 0;
        }
        .truncate-1 {
            display: -webkit-box;
            -webkit-line-clamp: 1;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        .truncate-2 {
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
    """

        def _estimate_badge_height(cleaned: str) -> int:
            card_count = cleaned.count("badge-card")
            if card_count <= 0:
                return 120
            cards_per_row = 3 if "featured-grid" in cleaned else 4
            rows = max(1, math.ceil(card_count / cards_per_row))
            return 110 + rows * 150

        def render_badge_html(html_block: str, *, label: str, height: int | None = None) -> None:
            cleaned = textwrap.dedent(html_block).strip()
            _debug_html_warning(label, "st_html", cleaned)
            doc = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <style>{badge_css}</style>
  </head>
  <body>{cleaned}</body>
</html>"""
            resolved_height = height if height is not None else _estimate_badge_height(cleaned)
            st_html(doc, height=resolved_height, scrolling=False)
        badge_defs = getattr(ctx, "df_badges", None)
        if badge_defs is None or (isinstance(badge_defs, pd.DataFrame) and badge_defs.empty):
            badge_defs = fetch_badge_definitions(_supabase)

        player_badges = getattr(ctx, "df_player_badges", None)
        if player_badges is None or (isinstance(player_badges, pd.DataFrame) and player_badges.empty):
            try:
                player_badges = fetch_player_badges(_supabase, club_id, pid)
            except Exception:
                logger.exception("Failed to fetch badges for player view")
                player_badges = pd.DataFrame()

        summary = build_gamification_summary(pid, badge_defs, player_badges)
        prestige_total = summary.get("prestige_total", 0)
        collected_unique = summary.get("collected_unique_count", 0)
        total_active = summary.get("total_active_badge_types", 0)

        unlocked_badges = summary.get("unlocked_badges", [])
        locked_badges = summary.get("locked_badges", [])

        df_matches = getattr(ctx, "df_matches", None)
        inactive_leagues = build_inactive_league_options(df_meta, df_leagues, df_matches)
        league_options = inactive_leagues["league_name"].tolist() if not inactive_leagues.empty else []
        league_labels = dict(zip(inactive_leagues["league_name"], inactive_leagues["display_label"]))
        completed_league_ids = set(league_options)

        badge_markdown("#### Career Trophy Case", label="trophies.case.header")
        trophy_case = get_player_trophy_case(player_badges, pid, completed_league_ids, limit=8)
        trophy_case = _decorate_trophies_with_leagues(trophy_case, league_labels)
        if trophy_case.empty:
            badge_caption(
                "No trophies yet. Win podium finishes or top performer awards in completed leagues.",
                label="trophies.case.empty",
            )
        else:
            trophy_cards = []
            for _, row in trophy_case.iterrows():
                row_player_id = row.get("player_id")
                if row_player_id is not None and int(row_player_id) != int(pid):
                    logger.warning(
                        "Skipping trophy for non-owner player_id=%s on player_id=%s view",
                        row_player_id,
                        pid,
                    )
                    continue
                icon = "🏆"
                value_json = _parse_value_json(row.get("value_json"))
                badge_name = _trophy_display_name(row)
                if _is_top_performer_badge(row.get("badge_id")):
                    trophy_title = _format_top_performer_title(badge_name, value_json.get("category_label"))
                    rank_label = _format_top_performer_rank(value_json)
                else:
                    trophy_title = badge_name
                    rank_label = ""
                metric_display = value_json.get("metric_display")
                if not metric_display:
                    metric_display = _format_top_performer_metric(
                        value_json.get("category_key"),
                        value_json.get("metric_value"),
                    )
                if rank_label:
                    metric_display = f"{rank_label} • {metric_display}" if metric_display else rank_label
                elif not metric_display and value_json.get("rank") is not None:
                    metric_display = f"#{value_json.get('rank')}"
                league_label = str(row.get("league_label") or "League")
                earned_at_label = _format_earned_at(row.get("earned_at"))
                earned_at_label = f"Earned {earned_at_label}" if earned_at_label else ""
                metric_line = (
                    f"<div class=\"trophy-case-meta truncate-1\">{html.escape(metric_display)}</div>"
                    if metric_display
                    else ""
                )
                card = f"""
                <div class="trophy-case-card">
                    <div class="trophy-case-header">
                        <span>{html.escape(icon)}</span>
                        <span class="truncate-1">{html.escape(trophy_title)}</span>
                    </div>
                    {metric_line}
                    <div class="trophy-case-meta truncate-1">{html.escape(league_label)}</div>
                    <div class="trophy-case-meta">{html.escape(earned_at_label)}</div>
                </div>
                """
                trophy_cards.append(card)
            if trophy_cards:
                render_badge_html(
                    f"<div class='trophy-case-grid'>{''.join(trophy_cards)}</div>",
                    label="trophies.case.grid",
                    height=190 + len(trophy_cards) * 30,
                )
            else:
                badge_caption(
                    "No trophies yet. Win podium finishes or top performer awards in completed leagues.",
                    label="trophies.case.empty",
                )

        tournament_trophies = fetch_player_tournament_trophies(_supabase, club_id, pid)
        if debug_render:
            badge_code(
                f"Tournament podium badge rows: {len(tournament_trophies)}",
                label="trophies.tournaments.debug",
            )

        st.subheader("🏆 Tournament Trophies")
        if not tournament_trophies:
            badge_caption("No tournament trophies yet.", label="trophies.tournaments.empty")
        else:
            podium_labels = {
                1: "🥇 Champion",
                2: "🥈 Runner-up",
                3: "🥉 Bronze",
            }
            trophy_cards = []
            for trophy in tournament_trophies:
                placement = trophy.get("placement")
                medal_label = podium_labels.get(placement, "🏅 Podium")
                tournament_name = trophy.get("tournament_name")
                if not tournament_name:
                    tournament_id = trophy.get("tournament_id")
                    if tournament_id:
                        tournament_name = f"Tournament {str(tournament_id)[:8]}"
                    else:
                        tournament_name = "Tournament"
                teammate_line = trophy.get("teammate_names")
                earned_at_label = _format_earned_at(trophy.get("earned_at"))
                earned_at_label = f"Awarded {earned_at_label}" if earned_at_label else ""
                teammate_html = (
                    f"<div class=\"trophy-body truncate-1\">{html.escape(str(teammate_line))}</div>"
                    if teammate_line
                    else ""
                )
                earned_html = (
                    f"<div class=\"trophy-body\">{html.escape(earned_at_label)}</div>"
                    if earned_at_label
                    else ""
                )
                card = f"""
                <div class="trophy-chip">
                    <span>{html.escape(medal_label)}</span>
                    <div class="trophy-text">
                        <div class="trophy-title truncate-1">{html.escape(str(tournament_name))}</div>
                        {teammate_html}
                        {earned_html}
                    </div>
                </div>
                """
                trophy_cards.append(card)

            height = 130 + math.ceil(len(trophy_cards) / 2) * 90
            render_badge_html(
                f"<div class='trophy-chip-row'>{''.join(trophy_cards)}</div>",
                label="trophies.tournaments.grid",
                height=height,
            )

        top_prestige_key = f"top_prestige_{pid}"
        if top_prestige_key in st.session_state:
            top_prestige = st.session_state[top_prestige_key]
        else:
            prestige_sorted = sorted(
                unlocked_badges,
                key=lambda b: (
                    int(b.get("prestige", 0) or 0),
                    pd.to_datetime(b.get("last_earned_at"), utc=True, errors="coerce"),
                ),
                reverse=True,
            )
            top_prestige = prestige_sorted[:5]
            st.session_state[top_prestige_key] = top_prestige

        chip_items = []
        for badge in top_prestige:
            icon = badge_icon(badge.get("badge_id"), badge.get("category"))
            stack = badge.get("stack_count", 1)
            stack_text = f" ×{stack}" if stack and stack > 1 else ""
            chip_items.append(
                f"<span class='badge-chip'><span>{html.escape(icon)}</span>"
                f"<span class='truncate-1'>{html.escape(str(badge.get('name', 'Badge')))}{stack_text}</span></span>"
            )

        summary_html = f"""
        <div class="badge-summary">
            <div class="badge-stat">
                <div class="badge-stat-label">Prestige</div>
                <div class="badge-stat-value">{int(prestige_total)}</div>
            </div>
            <div class="badge-stat">
                <div class="badge-stat-label">Collection</div>
                <div class="badge-stat-value">{collected_unique}/{total_active}</div>
            </div>
            <div class="badge-stat" style="flex:1; min-width: 220px;">
                <div class="badge-stat-label">Top Prestige</div>
                <div class="badge-chip-row">{''.join(chip_items) or "<span class='badge-subtext'>No reels yet.</span>"}</div>
            </div>
        </div>
    """
        render_badge_html(summary_html, label="badges.summary")

        if not unlocked_badges and not locked_badges:
            badge_caption("No badges available yet.", label="badges.empty")
        else:
            badge_markdown("#### Featured Cuts", label="badges.featured.header")
            prestige_sorted = sorted(
                unlocked_badges,
                key=lambda b: (
                    int(b.get("prestige", 0) or 0),
                    pd.to_datetime(b.get("last_earned_at"), utc=True, errors="coerce"),
                ),
                reverse=True,
            )
            non_participant = [b for b in prestige_sorted if b.get("badge_id") != "participant"]
            if len(non_participant) >= 3:
                featured = non_participant[:3]
            else:
                featured = non_participant[:]
                remaining_slots = 3 - len(featured)
                if remaining_slots > 0:
                    participant_badges = [
                        b for b in prestige_sorted if b.get("badge_id") == "participant"
                    ]
                    featured.extend(participant_badges[:remaining_slots])
            if not featured:
                badge_caption(
                    "The trophy room is quiet—new reels arrive after the next run.",
                    label="badges.featured.empty",
                )
            else:
                featured_cards = []
                for badge in featured:
                    copy_plain = build_badge_copy_plain(badge)
                    icon = badge_icon(badge.get("badge_id"), badge.get("category"))
                    stack = badge.get("stack_count", 1)
                    stack_text = f" ×{stack}" if stack and stack > 1 else ""
                    requirements_html = render_inline_badge_text(copy_plain.req_text)
                    featured_cards.append(
                        f"""
                    <div class="badge-card">
                        <div class="badge-card-header">
                            <span>{html.escape(icon)}</span>
                            <span class="truncate-1">{html.escape(str(badge.get('name', 'Badge')))}{stack_text}</span>
                        </div>
                        <div class="badge-subtext">Prestige {int(badge.get('prestige', 0) or 0)}</div>
                        <div class="badge-subtext truncate-2">{requirements_html}</div>
                    </div>
                    """
                    )
                render_badge_html(
                    f"<div class='badge-grid featured-grid'>{''.join(featured_cards)}</div>",
                    label="badges.featured.grid",
                )

            with st.expander("Open Cabinet", expanded=False):
                filter_cols = st.columns(2)
                show_unlocked = filter_cols[0].checkbox("Unlocked", value=True, key="badge_filter_unlocked")
                show_locked = filter_cols[0].checkbox("Locked", value=True, key="badge_filter_locked")

                all_badges = []
                for badge in unlocked_badges:
                    badge_copy = dict(badge)
                    badge_copy["status"] = "unlocked"
                    all_badges.append(badge_copy)
                for badge in locked_badges:
                    badge_copy = dict(badge)
                    badge_copy["status"] = "locked"
                    all_badges.append(badge_copy)

                categories = sorted({b.get("category") or "Other" for b in all_badges})
                rarities = sorted({b.get("rarity") or "common" for b in all_badges})
                selected_categories = filter_cols[1].multiselect(
                    "Category",
                    categories,
                    default=categories,
                    key="badge_filter_categories",
                )
                selected_rarities = filter_cols[1].multiselect(
                    "Rarity",
                    rarities,
                    default=rarities,
                    key="badge_filter_rarities",
                )

                def _visible(badge: dict) -> bool:
                    category = badge.get("category") or "Other"
                    rarity = badge.get("rarity") or "common"
                    if badge.get("status") == "unlocked" and not show_unlocked:
                        return False
                    if badge.get("status") == "locked" and not show_locked:
                        return False
                    if category not in selected_categories:
                        return False
                    if rarity not in selected_rarities:
                        return False
                    return True

                visible_badges = [b for b in all_badges if _visible(b)]
                if not visible_badges:
                    badge_caption("No badges match the filters.", label="badges.filters.empty")
                else:
                    card_items = []
                    for badge in visible_badges:
                        status = badge.get("status")
                        copy_plain = build_badge_copy_plain(badge)
                        name = html.escape(str(badge.get("name", "Badge")))
                        prestige = int(badge.get("prestige", 0) or 0)
                        icon = badge_icon(badge.get("badge_id"), badge.get("category"))
                        stack = badge.get("stack_count", 1)
                        stack_text = f" ×{stack}" if stack and stack > 1 else ""
                        if status == "locked":
                            requirements_html = render_inline_badge_text(copy_plain.req_text)
                            card_items.append(
                                f"""
                            <div class="badge-card silhouette">
                                <div class="badge-card-header">
                                    <span>⬛</span>
                                    <span class="truncate-1">{name}{stack_text}</span>
                                </div>
                                <div class="badge-subtext">Prestige {prestige}</div>
                                <div class="badge-subtext truncate-2">{requirements_html}</div>
                            </div>
                            """
                            )
                        else:
                            requirements_html = render_inline_badge_text(copy_plain.req_text)
                            card_items.append(
                                f"""
                            <div class="badge-card">
                                <div class="badge-card-header">
                                    <span>{html.escape(icon)}</span>
                                    <span class="truncate-1">{name}{stack_text}</span>
                                </div>
                                <div class="badge-subtext">Prestige {prestige}</div>
                                <div class="badge-subtext truncate-2">{requirements_html}</div>
                            </div>
                            """
                            )
                    render_badge_html(
                        f"<div class='badge-grid'>{''.join(card_items)}</div>",
                        label="badges.cabinet.grid",
                    )

                details_view = st.toggle("Details view", value=False, key="badge_details_view")
                if details_view and unlocked_badges:
                    summary_df = pd.DataFrame(unlocked_badges)
                    summary_df["last_earned_at_dt"] = pd.to_datetime(
                        summary_df.get("last_earned_at", None), utc=True, errors="coerce"
                    )
                    summary_df = summary_df.sort_values(
                        ["last_earned_at_dt", "prestige"], ascending=[False, False]
                    )
                    show_df = summary_df[
                        ["name", "category", "prestige", "stack_count", "last_earned_at_dt"]
                    ].rename(
                        columns={
                            "name": "Badge",
                            "category": "Category",
                            "prestige": "Prestige",
                            "stack_count": "Count",
                            "last_earned_at_dt": "Last Earned",
                        }
                    )
                    st.dataframe(
                        show_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Prestige": st.column_config.NumberColumn(format="%d"),
                            "Last Earned": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
                        },
                    )

                    admin_debug = False
                    if bool(getattr(ctx, "admin_logged_in", False)):
                        admin_debug = st.toggle("Show debug columns", value=False, key="badge_debug_columns")

                    if isinstance(player_badges, pd.DataFrame) and not player_badges.empty:
                        pb_df = player_badges.copy()
                        pb_df = pb_df[pb_df.get("player_id") == int(pid)].copy()
                        pb_df["earned_at_dt"] = pd.to_datetime(
                            pb_df.get("earned_at", None), utc=True, errors="coerce"
                        )
                        for badge in summary_df.itertuples(index=False):
                            badge_id = getattr(badge, "badge_id", "")
                            badge_name = getattr(badge, "name", "Badge")
                            stack = getattr(badge, "stack_count", 1)
                            stack_text = f" x{stack}" if stack and stack > 1 else ""
                            icon = badge_icon(badge_id, getattr(badge, "category", None))
                            with st.expander(f"{icon} {badge_name}{stack_text}", expanded=False):
                                requirements = display_requirement_text(
                                    getattr(badge, "requirements", None)
                                )
                                st.markdown(requirements)
                                rows = pb_df[pb_df.get("badge_id") == badge_id].copy()
                                rows = rows.sort_values("earned_at_dt", ascending=False)
                                cols = ["earned_at_dt", "match_id"]
                                if admin_debug:
                                    cols.append("context_id")
                                show_rows = rows[cols].rename(
                                    columns={
                                        "earned_at_dt": "Earned",
                                        "match_id": "Match",
                                        "context_id": "Context",
                                    }
                                )
                                st.dataframe(
                                    show_rows,
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config={
                                        "Earned": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
                                    },
                                )
            st.subheader("Story Cards")
            story_df = fetch_player_stories(_supabase, club_id, pid, limit=6)
            if story_df.empty:
                st.caption("No new stories in the tape room yet.")
            else:
                story_df = story_df.drop_duplicates(subset=["story_type", "context_id"], keep="first")
                story_df = story_df.sort_values("created_at", ascending=False)
                highlights = story_df[story_df["story_type"].str.startswith("highlight", na=False)].head(3)
                foreshadow = story_df[story_df["story_type"].str.startswith("foreshadow", na=False)].head(3)
                highlight_col, foreshadow_col = st.columns(2)
                with highlight_col:
                    st.markdown("**Highlights**")
                    if highlights.empty:
                        st.caption("No highlights yet.")
                    else:
                        for _, row in highlights.iterrows():
                            title = html.escape(str(row.get("title") or "Highlight"))
                            body = html.escape(str(row.get("body") or ""))
                            st.markdown(f"**{title}**")
                            st.caption(body)
                with foreshadow_col:
                    st.markdown("**Foreshadowing**")
                    if foreshadow.empty:
                        st.caption("No foreshadowing yet.")
                    else:
                        for _, row in foreshadow.iterrows():
                            title = html.escape(str(row.get("title") or "Foreshadowing"))
                            body = html.escape(str(row.get("body") or ""))
                            st.markdown(f"**{title}**")
                            st.caption(body)

    def render_ratings_tab():
        # -------------------------
        # Restore: Ratings by active league (table)
        # -------------------------
        st.markdown("### Ratings by active league")

        active_leagues = []
        if df_meta is not None and isinstance(df_meta, pd.DataFrame) and not df_meta.empty:
            if "is_active" in df_meta.columns and "league_name" in df_meta.columns:
                active_leagues = (
                    df_meta[df_meta["is_active"] == True]["league_name"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .tolist()
                )

        lr_rows = pd.DataFrame()
        if df_leagues is not None and isinstance(df_leagues, pd.DataFrame) and not df_leagues.empty:
            if "player_id" in df_leagues.columns:
                lr_rows = df_leagues[df_leagues["player_id"].astype(int) == int(pid)].copy()

        if not lr_rows.empty:
            if "league_name" in lr_rows.columns:
                lr_rows["league_name"] = lr_rows["league_name"].astype(str).str.strip()

            if active_leagues and "league_name" in lr_rows.columns:
                lr_rows = lr_rows[lr_rows["league_name"].isin(active_leagues)].copy()

            if "is_active" in lr_rows.columns:
                lr_rows = lr_rows[lr_rows["is_active"] == True].copy()

            if lr_rows.empty:
                st.caption("No active league ratings found for this player.")
            else:
                if "rating" in lr_rows.columns:
                    lr_rows["League JUPR"] = lr_rows["rating"].astype(float) / 400.0

                cols = ["league_name", "League JUPR", "wins", "losses", "matches_played"]
                cols = [c for c in cols if c in lr_rows.columns]

                if "League JUPR" in lr_rows.columns:
                    lr_rows = lr_rows.sort_values("League JUPR", ascending=False)

                st.dataframe(
                    lr_rows[cols].rename(
                        columns={"league_name": "League", "wins": "W", "losses": "L", "matches_played": "MP"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                    column_config={"League JUPR": st.column_config.NumberColumn(format="%.3f")},
                )
        else:
            st.caption("No league ratings table entries found for this player yet.")

        st.divider()

        matches = fetch_player_matches(_supabase, club_id, pid, limit=600)

        if matches.empty:
            st.info("No matches recorded for this player.")
            return

        def _safe_int(x, default=None):
            try:
                if x is None or str(x).strip() == "":
                    return default
                return int(x)
            except Exception:
                return default

        def _safe_float(x, default=None):
            try:
                if x is None or str(x).strip() == "":
                    return default
                return float(x)
            except Exception:
                return default

        def score_for_player(r):
            try:
                t1p1 = _safe_int(r.get("t1_p1"))
                t1p2 = _safe_int(r.get("t1_p2"))
                s1 = _safe_int(r.get("score_t1"), 0) or 0
                s2 = _safe_int(r.get("score_t2"), 0) or 0
            except Exception:
                return ""
            if t1p1 == pid or t1p2 == pid:
                return f"{s1}-{s2}"
            return f"{s2}-{s1}"

        def result_for_player(r):
            try:
                t1p1 = _safe_int(r.get("t1_p1"))
                t1p2 = _safe_int(r.get("t1_p2"))
                s1 = _safe_int(r.get("score_t1"), 0) or 0
                s2 = _safe_int(r.get("score_t2"), 0) or 0
            except Exception:
                return ""

            if s1 == s2:
                return "DRAW"
            on_t1 = pid in {t1p1, t1p2}
            winner = "WIN" if s1 > s2 else "LOSS"
            if not on_t1:
                winner = "WIN" if s2 > s1 else "LOSS"
            return winner

        def explain_link(r):
            try:
                t1p1 = _safe_int(r.get("t1_p1"))
                t1p2 = _safe_int(r.get("t1_p2"))
                t2p1 = _safe_int(r.get("t2_p1"))
                t2p2 = _safe_int(r.get("t2_p2"))
                s1 = _safe_int(r.get("score_t1"), 0) or 0
                s2 = _safe_int(r.get("score_t2"), 0) or 0
            except Exception:
                return ""

            if t1p1 == pid or t1p2 == pid:
                partner = t1p1 if t1p2 == pid else t1p2
                opp1, opp2 = t2p1, t2p2
                sy, so = s1, s2
            elif t2p1 == pid or t2p2 == pid:
                partner = t2p1 if t2p2 == pid else t2p2
                opp1, opp2 = t1p1, t1p2
                sy, so = s2, s1
            else:
                return ""

            if partner is None or opp1 is None or opp2 is None:
                return ""

            params = {
                "page": "match_explorer",
                "ctx": "OVERALL",
                "me": int(pid),
                "partner": int(partner),
                "opp1": int(opp1),
                "opp2": int(opp2),
                "sy": int(sy),
                "so": int(so),
            }
            if bool(ctx.public_mode):
                params["public"] = 1
            return f"/?{urlencode(params)}"

        def get_overall_snap(r: dict, pid_: int):
            pid_ = int(pid_)
            t1p1 = _safe_int(r.get("t1_p1"))
            t1p2 = _safe_int(r.get("t1_p2"))
            t2p1 = _safe_int(r.get("t2_p1"))
            t2p2 = _safe_int(r.get("t2_p2"))

            if t1p1 == pid_:
                return _safe_float(r.get("t1_p1_r")), _safe_float(r.get("t1_p1_r_end"))
            if t1p2 == pid_:
                return _safe_float(r.get("t1_p2_r")), _safe_float(r.get("t1_p2_r_end"))
            if t2p1 == pid_:
                return _safe_float(r.get("t2_p1_r")), _safe_float(r.get("t2_p1_r_end"))
            if t2p2 == pid_:
                return _safe_float(r.get("t2_p2_r")), _safe_float(r.get("t2_p2_r_end"))
            return None, None

        def signed_delta_from_elo_delta(r: dict, pid_: int):
            pid_ = int(pid_)
            raw = _safe_float(r.get("elo_delta"), None)
            if raw is None:
                return None

            s1 = _safe_int(r.get("score_t1"), 0) or 0
            s2 = _safe_int(r.get("score_t2"), 0) or 0
            if s1 == s2:
                return 0.0

            t1 = {_safe_int(r.get("t1_p1")), _safe_int(r.get("t1_p2"))}
            t2 = {_safe_int(r.get("t2_p1")), _safe_int(r.get("t2_p2"))}
            on_t1 = pid_ in t1
            on_t2 = pid_ in t2
            if not on_t1 and not on_t2:
                return None

            winner_team = 1 if s1 > s2 else 2
            my_team = 1 if on_t1 else 2
            return abs(float(raw)) if winner_team == my_team else -abs(float(raw))

        # Normalize date + league strings
        matches = matches.copy()
        matches["date_dt"] = pd.to_datetime(matches.get("date", None), errors="coerce", utc=True)
        matches = matches.dropna(subset=["date_dt"]).copy()
        matches["league"] = matches.get("league", "").fillna("").astype(str).str.strip()
        matches["match_type"] = matches.get("match_type", "").fillna("").astype(str).str.strip()

        # Build overall series rows
        processed = []
        for _, r0 in matches.iterrows():
            r = dict(r0)

            start_elo, end_elo = get_overall_snap(r, pid)
            after_jupr = None
            delta_jupr = None

            if start_elo is not None and end_elo is not None:
                try:
                    delta_jupr = (float(end_elo) - float(start_elo)) / 400.0
                    after_jupr = float(end_elo) / 400.0
                except Exception:
                    pass
            else:
                d_elo = signed_delta_from_elo_delta(r, pid)
                if d_elo is not None:
                    delta_jupr = float(d_elo) / 400.0

            processed.append(
                {
                    "id": _safe_int(r.get("id")),
                    "Date": r.get("date_dt"),
                    "League": str(r.get("league", "") or "").strip(),
                    "match_type": str(r.get("match_type", "") or "").strip(),
                    "Score": score_for_player(r),
                    "Result": result_for_player(r),
                    "Overall Δ": delta_jupr,
                    "Overall After": after_jupr,
                    "Explain": explain_link(r),
                }
            )

        df = pd.DataFrame(processed)
        if df.empty:
            st.info("No matches available.")
            return

        df = df.sort_values(["Date", "id"], ascending=[True, True]).reset_index(drop=True)
        df["Overall Δ"] = pd.to_numeric(df["Overall Δ"], errors="coerce")
        df["Overall After"] = pd.to_numeric(df["Overall After"], errors="coerce")

        # Backfill overall-after if needed
        if df["Overall After"].notna().any():
            for i in range(len(df)):
                if pd.isna(df.loc[i, "Overall After"]):
                    if i > 0 and pd.notna(df.loc[i - 1, "Overall After"]) and pd.notna(df.loc[i, "Overall Δ"]):
                        df.loc[i, "Overall After"] = float(df.loc[i - 1, "Overall After"]) + float(df.loc[i, "Overall Δ"])
            for i in range(len(df) - 2, -1, -1):
                if pd.isna(df.loc[i, "Overall After"]):
                    if pd.notna(df.loc[i + 1, "Overall After"]) and pd.notna(df.loc[i + 1, "Overall Δ"]):
                        df.loc[i, "Overall After"] = float(df.loc[i + 1, "Overall After"]) - float(df.loc[i + 1, "Overall Δ"])
        else:
            df_rev = df.sort_values(["Date", "id"], ascending=[False, False]).reset_index(drop=True)
            running = 0.0
            after_vals = []
            for i in range(len(df_rev)):
                after_vals.append(float(current_jupr) - float(running))
                d = df_rev.loc[i, "Overall Δ"]
                if pd.notna(d):
                    running += float(d)
            df_rev["Overall After"] = after_vals
            df = df_rev.sort_values(["Date", "id"], ascending=[True, True]).reset_index(drop=True)

        # -------------------------
        # Restore: tabs for Overall + each league
        # -------------------------
        leagues_in_matches = sorted(
            [x for x in df["League"].fillna("").astype(str).str.strip().unique().tolist() if x and x.upper() != "OVERALL"]
        )
        tab_labels = ["Overall"] + [f"League: {lg}" for lg in leagues_in_matches]
        tabs = st.tabs(tab_labels)

        def render_chart_and_table(view_df: pd.DataFrame, title_prefix: str, *, league_trend: bool = False, league_name: str = ""):
            st.subheader(f"{title_prefix} JUPR Trend")

            chart_df = view_df.copy().dropna(subset=["Overall After"]).sort_values(["Date", "id"]).reset_index(drop=True)
            if chart_df.empty:
                st.info("No chartable rating data in this view.")
            else:
                chart_df["Match #"] = range(1, len(chart_df) + 1)
                chart_df["DateStr"] = pd.to_datetime(chart_df["Date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
                chart_df["DeltaStr"] = chart_df["Overall Δ"].map(lambda x: f"{float(x):+.4f}" if pd.notna(x) else "")
                chart_df["AfterStr"] = chart_df["Overall After"].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "")

                tail = chart_df.tail(60).copy()

                # Optional full restore: show league replay trend (if available)
                if league_trend and league_name and _LEAGUE_REPLAY_AVAILABLE:
                    snap_map = build_league_snapshot_map(_supabase, club_id, league_name, df_meta, df_players_all)
                    if snap_map:
                        # Build a league-after series from snap_map for this player
                        tmp = view_df.copy()
                        tmp["League After"] = pd.NA
                        tmp["League Δ"] = pd.NA
                        for i in range(len(tmp)):
                            mid = tmp.iloc[i].get("id", None)
                            if mid is None:
                                continue
                            hit = snap_map.get(int(mid), {}).get(int(pid), None)
                            if hit:
                                ls, le = hit
                                tmp.at[tmp.index[i], "League Δ"] = (float(le) - float(ls)) / 400.0
                                tmp.at[tmp.index[i], "League After"] = float(le) / 400.0

                        tmp2 = tmp.dropna(subset=["League After"]).sort_values(["Date", "id"]).reset_index(drop=True)
                        if not tmp2.empty:
                            tmp2["Match #"] = range(1, len(tmp2) + 1)
                            tmp2["DateStr"] = pd.to_datetime(tmp2["Date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
                            tmp2["DeltaStr"] = tmp2["League Δ"].map(lambda x: f"{float(x):+.4f}" if pd.notna(x) else "")
                            tmp2["AfterStr"] = tmp2["League After"].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "")
                            tail = tmp2.tail(60).copy()
                            y_col = "League After"
                            y_title = "League JUPR After Match"
                        else:
                            y_col = "Overall After"
                            y_title = "JUPR After Match (Overall)"
                    else:
                        y_col = "Overall After"
                        y_title = "JUPR After Match (Overall)"
                else:
                    y_col = "Overall After"
                    y_title = "JUPR After Match (Overall)"

                if alt is not None:
                    line = (
                        alt.Chart(tail)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("Match #:Q", axis=alt.Axis(tickMinStep=1), title="Match Order"),
                            y=alt.Y(
                                f"{y_col}:Q",
                                axis=alt.Axis(format=".3f"),
                                title=y_title,
                                scale=alt.Scale(zero=False),
                            ),
                            tooltip=[
                                alt.Tooltip("DateStr:N", title="Date"),
                                alt.Tooltip("League:N", title="League"),
                                alt.Tooltip("Score:N", title="Score"),
                                alt.Tooltip("AfterStr:N", title="After"),
                                alt.Tooltip("DeltaStr:N", title="Δ"),
                            ],
                        )
                        .interactive()
                    )
                    st.altair_chart(line, use_container_width=True)
                else:
                    st.line_chart(tail.set_index("Match #")[y_col])

            st.divider()
            st.subheader(f"{title_prefix} Match History")

            show = view_df.sort_values(["Date", "id"], ascending=[False, False]).copy()
            show["date"] = pd.to_datetime(show["Date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
            show["delta_raw"] = pd.to_numeric(show["Overall Δ"], errors="coerce")
            show["Overall Δ"] = show["delta_raw"].map(lambda x: f"{float(x):+.4f}" if pd.notna(x) else "")
            show["Overall After"] = show["Overall After"].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "")

            # Render Overall history with a native LinkColumn so Explain links are clickable.
            if title_prefix == "Overall":
                show = show.rename(columns={"Explain": "EXPLAIN"})
                show = show[["date", "League", "Score", "Result", "match_type", "Overall Δ", "Overall After", "EXPLAIN"]]
                st.dataframe(
                    show,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "EXPLAIN": st.column_config.LinkColumn("Explain", display_text="Explain"),
                    },
                )
                return

            def result_badge(result: str) -> str:
                label = str(result or "").strip().upper() or "—"
                normalized = label.upper()
                if normalized in {"W", "WIN", "WON"}:
                    variant = "win"
                elif normalized in {"L", "LOSS", "LOST"}:
                    variant = "loss"
                else:
                    variant = "draw"
                return f"<span class='jupr-result-badge {variant}'>{label}</span>"

            def delta_span(delta_str: str, delta_raw: float | None) -> str:
                if not delta_str:
                    return ""
                kind = "zero"
                try:
                    delta_val = float(delta_raw)
                except (TypeError, ValueError):
                    delta_val = 0.0
                if delta_val > 0:
                    kind = "pos"
                elif delta_val < 0:
                    kind = "neg"
                return f"<span class='jupr-delta {kind}'>{delta_str}</span>"

            show["Result"] = show["Result"].map(result_badge)
            show["Overall Δ"] = show.apply(lambda row: delta_span(row["Overall Δ"], row["delta_raw"]), axis=1)
            show["Explain"] = show["Explain"].map(
                lambda url: f"<a href='{url}' target='_self'>Explain</a>" if url else ""
            )

            show = show[["date", "League", "Score", "Result", "match_type", "Overall Δ", "Overall After", "Explain"]]

            html_table = show.to_html(index=False, escape=False)

            st.markdown(
                f"""
                <div class="match-history-table">
                  {html_table}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with tabs[0]:
            render_chart_and_table(df, "Overall", league_trend=False)

        for i, lg in enumerate(leagues_in_matches, start=1):
            with tabs[i]:
                df_lg = df[df["League"].astype(str).str.strip() == lg].copy()
                # Show league replay trend if available; otherwise overall trend filtered to that league’s matches.
                render_chart_and_table(df_lg, f"League: {lg}", league_trend=True, league_name=lg)

    def render_social_tab():
        st.markdown("### Social / Community")
        try:
            social_data = fetch_player_social_participation(_supabase, club_id, pid)
        except Exception:
            logger.exception("Failed to load social profile data")
            st.info("No social RR history yet.")
            return

        if not social_data.get("available", True):
            st.info(
                "Social profile data is unavailable until Club Social tables are installed."
            )
            return

        history_df = social_data.get("history_df")
        if history_df is None or history_df.empty:
            st.info("No social RR history yet.")
            return

        summary = social_data.get("summary") or {}
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Social Events", f"{int(summary.get('events', 0))}")
        m2.metric("Social Matches", f"{int(summary.get('matches', 0))}")
        m3.metric("Social Record", str(summary.get("record") or "0-0"))
        m4.metric("Social Diff", f"{int(summary.get('diff', 0)):+d}")
        m5.metric("Last Social Appearance", str(summary.get("last_appearance") or "—"))

        st.markdown("#### Skill-Level Breakdown")
        skill_df = social_data.get("skill_breakdown_df")
        if isinstance(skill_df, pd.DataFrame) and not skill_df.empty:
            st.dataframe(skill_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No skill-level tags found for this player's social events.")

        chart_df = history_df.copy()
        chart_df["Date"] = pd.to_datetime(chart_df.get("Date"), utc=True, errors="coerce")
        chart_df = chart_df.dropna(subset=["Date"]).copy()
        if not chart_df.empty:
            st.markdown("#### Social Match Activity")
            daily = (
                chart_df.groupby(chart_df["Date"].dt.date)["Matches"].sum().reset_index().rename(columns={"Date": "Day"})
            )
            daily["Cumulative Matches"] = daily["Matches"].cumsum()
            daily["Day"] = pd.to_datetime(daily["Day"], errors="coerce")
            if alt is not None:
                chart = (
                    alt.Chart(daily)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("Day:T", title="Date"),
                        y=alt.Y("Cumulative Matches:Q", title="Cumulative Social Matches"),
                        tooltip=[
                            alt.Tooltip("Day:T", title="Date"),
                            alt.Tooltip("Matches:Q", title="Matches"),
                            alt.Tooltip("Cumulative Matches:Q", title="Cumulative"),
                        ],
                    )
                    .interactive()
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.line_chart(daily.set_index("Day")["Cumulative Matches"])

        st.markdown("#### Recent Social Events")
        show_df = history_df.copy()
        show_df["Date"] = pd.to_datetime(show_df.get("Date"), utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
        cols = ["Date", "Event", "Event Type", "Skill Tags", "Matches", "Wins", "Losses", "Diff", "Submitted By"]
        cols = [c for c in cols if c in show_df.columns]
        st.dataframe(show_df[cols], use_container_width=True, hide_index=True)

    with ratings_tab:
        render_ratings_tab()

    with social_tab:
        render_social_tab()
