import time
import pandas as pd

from jupr_app.domain.player_activity import add_activity_columns
from jupr_app.domain.gamification.badge_descriptions import BADGE_DESCRIPTIONS_MD
from jupr_app.domain.gamification.badge_registry import badge_schema_by_id
from jupr_app.domain.gamification.requirements import load_requirements_map
from jupr_app.data.schema_preflight import ensure_badge_schema_preflight
import re
from postgrest.exceptions import APIError


PLAYER_BADGES_BASE_COLUMNS = [
    "id",
    "club_id",
    "player_id",
    "badge_id",
    "earned_at",
    "context_type",
    "context_id",
    "match_id",
    "value_num",
    "value_json",
]
PLAYER_BADGES_OPTIONAL_COLUMNS = [
    "awarded_by",
    "rule_version",
    "eval_run_id",
    "revoked_at",
    "revoked_by",
    "revoke_reason",
]
MERGED_PLAYER_MARKER = "(MERGED into "


def _missing_player_badges_columns(exc: APIError) -> set[str]:
    message = str(exc)
    missing = {col for col in PLAYER_BADGES_OPTIONAL_COLUMNS if col in message}
    if missing:
        return missing
    matches = re.findall(r"player_badges\\.([a-zA-Z0-9_]+)", message)
    return {col for col in matches if col in PLAYER_BADGES_OPTIONAL_COLUMNS}


def _ensure_player_badges_columns(df: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        "awarded_by": "engine",
        "rule_version": None,
        "eval_run_id": None,
        "revoked_at": None,
        "revoked_by": None,
        "revoke_reason": None,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    return df


def _fetch_player_badges(supabase, club_id: str) -> tuple[pd.DataFrame, bool, str | None]:
    schema_degraded = False
    schema_degraded_reason = None
    select_cols = ",".join(PLAYER_BADGES_BASE_COLUMNS + PLAYER_BADGES_OPTIONAL_COLUMNS)
    try:
        pb_resp = (
            supabase.table("player_badges")
            .select(select_cols)
            .eq("club_id", club_id)
            .execute()
        )
    except APIError as exc:
        missing = _missing_player_badges_columns(exc)
        if getattr(exc, "code", None) == "42703" and missing:
            schema_degraded = True
            schema_degraded_reason = (
                "player_badges missing columns "
                f"{', '.join(sorted(missing))}; apply migrations/20260625_badge_recompute_runs.sql and "
                "migrations/20260630_player_badges_revocation.sql."
            )
            legacy_select = ",".join(PLAYER_BADGES_BASE_COLUMNS)
            pb_resp = (
                supabase.table("player_badges")
                .select(legacy_select)
                .eq("club_id", club_id)
                .execute()
            )
        else:
            raise
    df_player_badges = pd.DataFrame(pb_resp.data or [])
    df_player_badges = _ensure_player_badges_columns(df_player_badges)
    return df_player_badges, schema_degraded, schema_degraded_reason


def _fetch_matches(
    supabase,
    club_id: str,
    match_limit: int,
) -> tuple[pd.DataFrame, bool, str | None]:
    schema_degraded = False
    schema_degraded_reason = None
    try:
        m_resp = (
            supabase.table("matches")
            .select("*")
            .eq("club_id", club_id)
            .is_("deleted_at", None)
            .order("id", desc=True)
            .limit(int(match_limit))
            .execute()
        )
    except APIError as exc:
        message = str(exc)
        if getattr(exc, "code", None) == "42703" and "deleted_at" in message:
            schema_degraded = True
            schema_degraded_reason = (
                "matches.deleted_at column is missing; apply "
                "supabase/migrations/20260424_matches_soft_delete.sql to enable soft-delete filtering."
            )
            m_resp = (
                supabase.table("matches")
                .select("*")
                .eq("club_id", club_id)
                .order("id", desc=True)
                .limit(int(match_limit))
                .execute()
            )
        else:
            raise
    return pd.DataFrame(m_resp.data or []), schema_degraded, schema_degraded_reason


def _drop_merged_players(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "name" not in df.columns:
        return df
    merged_mask = (
        df["name"]
        .fillna("")
        .astype(str)
        .str.contains(MERGED_PLAYER_MARKER, case=False, regex=False)
    )
    if not merged_mask.any():
        return df
    return df.loc[~merged_mask].copy()


def load_data(supabase, club_id: str, match_limit: int = 5000):
    """
    Loads club-scoped tables and returns:
      df_players_all, df_players_active, df_leagues, df_matches, df_meta, df_badges,
      df_player_badges, name_to_id, id_to_name, schema_degraded, schema_degraded_reason

    No Streamlit calls here. Raise exceptions to be handled by UI.
    """
    club_id = str(club_id)
    ensure_badge_schema_preflight(supabase)

    max_retries = 3
    last_err = None

    for attempt in range(max_retries):
        try:
            schema_degraded = False
            schema_degraded_reason = None
            # Players
            p_resp = (
                supabase.table("players")
                .select("*")
                .eq("club_id", club_id)
                .execute()
            )
            df_players_all = pd.DataFrame(p_resp.data or [])

            df_players_all = add_activity_columns(df_players_all)
            df_players_all = _drop_merged_players(df_players_all)

            # Active players (inactive_at is authoritative when present)
            if not df_players_all.empty and "inactive_at" in df_players_all.columns:
                df_players_active = df_players_all[df_players_all["inactive_at"].isna()].copy()
            elif not df_players_all.empty and "active" in df_players_all.columns:
                df_players_active = df_players_all[df_players_all["active"] == True].copy()
            else:
                df_players_active = df_players_all.copy()

            # League ratings
            l_resp = (
                supabase.table("league_ratings")
                .select("id,player_id,league_name,rating,starting_rating,wins,losses,matches_played,is_active")
                .eq("club_id", club_id)
                .execute()
            )

            df_leagues = pd.DataFrame(l_resp.data or [])

            # Matches
            (
                df_matches,
                matches_schema_degraded,
                matches_schema_degraded_reason,
            ) = _fetch_matches(supabase, club_id, match_limit)
            if matches_schema_degraded:
                schema_degraded = True
                schema_degraded_reason = matches_schema_degraded_reason

            # Metadata
            meta_resp = (
                supabase.table("leagues_metadata")
                .select("*")
                .eq("club_id", club_id)
                .execute()
            )
            df_meta = pd.DataFrame(meta_resp.data or [])

            # Badges (global definitions)
            try:
                # Support older badge schema variants that lack state-related columns.
                badges_resp = (
                    supabase.table("badges")
                    .select(
                        "badge_id,name,prestige,category,is_stackable,is_active,rarity,"
                        "tier,icon_key,scope,state,state_changed_at,state_change_reason,eval_triggers,created_at"
                    )
                    .execute()
                )
            except APIError as exc:
                message = str(exc)
                if getattr(exc, "code", None) == "42703" or "badges.state does not exist" in message:
                    badges_resp = (
                        supabase.table("badges")
                        .select(
                            "badge_id,name,prestige,category,is_stackable,is_active,rarity,"
                            "tier,icon_key,scope,created_at"
                        )
                        .execute()
                    )
                else:
                    raise
            df_badges = pd.DataFrame(badges_resp.data or [])
            if not df_badges.empty and "badge_id" in df_badges.columns:
                if "state" not in df_badges.columns:
                    df_badges["state"] = "live"
                if "eval_triggers" not in df_badges.columns:
                    df_badges["eval_triggers"] = [["match_recorded", "match_updated"]] * len(df_badges)
                requirements_map = load_requirements_map()
                df_badges["requirements"] = (
                    df_badges["badge_id"].astype(str).map(requirements_map).fillna("Requirements TBD")
                )
                df_badges["description_md"] = (
                    df_badges["badge_id"].astype(str).map(BADGE_DESCRIPTIONS_MD).fillna("")
                )
                schema_map = badge_schema_by_id()
                df_badges["badge_status"] = df_badges["badge_id"].astype(str).map(
                    lambda bid: schema_map.get(str(bid)).status if str(bid) in schema_map else "live"
                )
                df_badges["badge_award_timing"] = df_badges["badge_id"].astype(str).map(
                    lambda bid: schema_map.get(str(bid)).award_timing if str(bid) in schema_map else "live"
                )
                df_badges["badge_scope"] = df_badges["badge_id"].astype(str).map(
                    lambda bid: schema_map.get(str(bid)).scope if str(bid) in schema_map else None
                )

            # Player badges (club-scoped)
            (
                df_player_badges,
                badges_schema_degraded,
                badges_schema_degraded_reason,
            ) = _fetch_player_badges(
                supabase,
                club_id,
            )
            if badges_schema_degraded:
                schema_degraded = True
                if schema_degraded_reason and badges_schema_degraded_reason:
                    schema_degraded_reason = (
                        f"{schema_degraded_reason} Also: {badges_schema_degraded_reason}"
                    )
                elif badges_schema_degraded_reason:
                    schema_degraded_reason = badges_schema_degraded_reason

            # Mappings
            if (
                not df_players_all.empty
                and "id" in df_players_all.columns
                and "name" in df_players_all.columns
            ):
                try:
                    ids = df_players_all["id"].astype(int)
                except Exception:
                    ids = df_players_all["id"]
                names = df_players_all["name"].astype(str)
                id_to_name = dict(zip(ids, names))
                name_to_id = dict(zip(names, ids))
            else:
                id_to_name, name_to_id = {}, {}
                df_players_all = pd.DataFrame(
                    columns=[
                        "id",
                        "name",
                        "rating",
                        "wins",
                        "losses",
                        "matches_played",
                        "active",
                        "last_game_at",
                        "inactive_at",
                    ]
                )
                df_players_active = df_players_all.copy()

            # Optional helper cols for match display
            if not df_matches.empty and id_to_name:
                for col_src, col_out in [
                    ("t1_p1", "p1"),
                    ("t1_p2", "p2"),
                    ("t2_p1", "p3"),
                    ("t2_p2", "p4"),
                ]:
                    if col_src in df_matches.columns:
                        df_matches[col_out] = df_matches[col_src].map(id_to_name)

            return (
                df_players_all,
                df_players_active,
                df_leagues,
                df_matches,
                df_meta,
                df_badges,
                df_player_badges,
                name_to_id,
                id_to_name,
                schema_degraded,
                schema_degraded_reason,
            )

        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            raise last_err
