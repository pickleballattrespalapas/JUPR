import time
import pandas as pd

from jupr_app.domain.player_activity import add_activity_columns
from jupr_app.domain.gamification.requirements import load_requirements_map


def load_data(supabase, club_id: str, match_limit: int = 5000):
    """
    Loads club-scoped tables and returns:
      df_players_all, df_players_active, df_leagues, df_matches, df_meta, df_badges,
      df_player_badges, name_to_id, id_to_name

    No Streamlit calls here. Raise exceptions to be handled by UI.
    """
    club_id = str(club_id)

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
                    "tier,icon_key,lore,hint,scope,created_at"
                )
                .execute()
            )
            df_badges = pd.DataFrame(badges_resp.data or [])
            if not df_badges.empty and "badge_id" in df_badges.columns:
                requirements_map = load_requirements_map()
                df_badges["requirements"] = (
                    df_badges["badge_id"].astype(str).map(requirements_map).fillna("Requirements TBD")
                )

            # Player badges (club-scoped)
            pb_resp = (
                supabase.table("player_badges")
                .select(
                    "id,club_id,player_id,badge_id,earned_at,context_type,context_id,"
                    "match_id,value_num,value_json"
                )
                .eq("club_id", club_id)
                .execute()
            )
            df_player_badges = pd.DataFrame(pb_resp.data or [])

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
            )

        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            raise last_err
