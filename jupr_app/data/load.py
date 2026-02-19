import time
import json
import hashlib
import pandas as pd

from jupr_app.domain.player_activity import add_activity_columns
from jupr_app.domain.gamification.badge_descriptions import BADGE_DESCRIPTIONS_MD
from jupr_app.domain.gamification.badge_registry import badge_schema_by_id
from jupr_app.domain.gamification.requirements import load_requirements_map
from jupr_app.data.schema_preflight import (
    check_required_upsert_indexes,
    ensure_badge_schema_preflight,
)
from services.match_pipeline import submit_match


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


def _resolve_loader_context(match_row: dict) -> tuple[str, str | None]:
    context_type = str(match_row.get("context_type") or "").strip().lower()
    if context_type not in {"league", "ladder", "tournament", "admin"}:
        if match_row.get("league"):
            context_type = "league"
        elif match_row.get("tournament_id"):
            context_type = "tournament"
        else:
            context_type = "admin"

    context_id = match_row.get("context_id")
    if context_id is not None and str(context_id).strip() != "":
        return context_type, str(context_id)
    if context_type == "league":
        league = str(match_row.get("league") or "").strip()
        return context_type, (league or None)
    if context_type == "tournament":
        tournament_id = match_row.get("tournament_id")
        if tournament_id is not None and str(tournament_id).strip() != "":
            return context_type, str(tournament_id)
    return context_type, None


def _loader_idempotency_key(club_id: str, match_row: dict) -> str:
    original_id = match_row.get("id", match_row.get("match_id"))
    external_id = match_row.get("external_id", match_row.get("source_match_id"))
    signature_payload = {
        "club_id": str(club_id),
        "source": "data_loader",
        "original_id": str(original_id) if original_id is not None else None,
        "external_id": str(external_id) if external_id is not None else None,
        "date": str(match_row.get("date") or match_row.get("created_at") or ""),
        "league": str(match_row.get("league") or ""),
        "context_type": str(match_row.get("context_type") or ""),
        "context_id": str(match_row.get("context_id") or ""),
        "t1_p1": int(match_row.get("t1_p1") or 0),
        "t1_p2": int(match_row.get("t1_p2") or 0),
        "t2_p1": int(match_row.get("t2_p1") or 0),
        "t2_p2": int(match_row.get("t2_p2") or 0),
        "score_t1": int(match_row.get("score_t1") or 0),
        "score_t2": int(match_row.get("score_t2") or 0),
    }
    normalized = json.dumps(signature_payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"loader:{club_id}:{original_id}:{digest}"


def submit_matches_from_loader(supabase, club_id: str, matches: list[dict], chunk_size: int = 500) -> int:
    _ = supabase
    club_id = str(club_id)
    submitted = 0

    for chunk_index, start in enumerate(range(0, len(matches), max(1, int(chunk_size))), start=1):
        chunk = matches[start : start + max(1, int(chunk_size))]
        try:
            for row in chunk:
                match_row = dict(row)
                context_type, context_id = _resolve_loader_context(match_row)
                idempotency_key = _loader_idempotency_key(club_id, match_row)
                match_row["idempotency_key"] = idempotency_key
                match_row["club_id"] = club_id

                submit_match(
                    club_id=club_id,
                    context_type=context_type,
                    context_id=context_id,
                    match_payload=match_row,
                    idempotency_key=idempotency_key,
                )
                submitted += 1
        except Exception as exc:
            raise RuntimeError(f"Failed loader matches submit chunk {chunk_index}: {exc}") from exc

    return submitted


def _fetch_player_badges(supabase, club_id: str) -> pd.DataFrame:
    select_cols = ",".join(PLAYER_BADGES_BASE_COLUMNS + PLAYER_BADGES_OPTIONAL_COLUMNS)
    pb_resp = (
        supabase.table("player_badges")
        .select(select_cols)
        .eq("club_id", club_id)
        .execute()
    )
    df_player_badges = pd.DataFrame(pb_resp.data or [])
    return df_player_badges


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
    schema_degraded, schema_degraded_reason = check_required_upsert_indexes(supabase)

    max_retries = 3
    last_err = None

    for attempt in range(max_retries):
        try:
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
            m_resp = (
                supabase.table("matches")
                .select("*")
                .eq("club_id", club_id)
                .order("id", desc=True)
                .limit(int(match_limit))
                .execute()
            )
            df_matches = pd.DataFrame(m_resp.data or [])

            # Metadata
            meta_resp = (
                supabase.table("leagues_metadata")
                .select("*")
                .eq("club_id", club_id)
                .execute()
            )
            df_meta = pd.DataFrame(meta_resp.data or [])

            # Badges (global definitions)
            badges_resp = (
                supabase.table("badges")
                .select(
                    "badge_id,name,prestige,category,is_stackable,is_active,rarity,"
                    "tier,icon_key,scope,state,state_changed_at,state_change_reason,eval_triggers,created_at"
                )
                .execute()
            )
            df_badges = pd.DataFrame(badges_resp.data or [])
            if not df_badges.empty and "badge_id" in df_badges.columns:
                if "state" not in df_badges.columns:
                    raise RuntimeError("Schema mismatch: badges.state missing.")
                if "eval_triggers" not in df_badges.columns:
                    raise RuntimeError("Schema mismatch: badges.eval_triggers missing.")
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
            df_player_badges = _fetch_player_badges(
                supabase,
                club_id,
            )

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
