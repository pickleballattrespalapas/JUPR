import streamlit as st
import pandas as pd

from jupr_app.ui.url import qp_get
from jupr_app.ui.helpers import build_player_profile_link
from jupr_app.ui.public_links import build_public_url, public_link_button


def render(ctx):
    # Always use 4-space indentation in this file.
    df_players = ctx.df_players_active
    df_players_all = ctx.df_players_all
    df_leagues = ctx.df_leagues
    df_meta = ctx.df_meta
    id_to_name = ctx.id_to_name
    supabase = ctx.supabase
    PUBLIC_MODE = bool(ctx.public_mode)
    club_id = str(ctx.club_id)

    st.header("🏆 Leaderboards")

    # -------------------------
    # Available leagues
    # -------------------------
    if df_meta is not None and not df_meta.empty and "is_active" in df_meta.columns:
        active_meta = df_meta[df_meta["is_active"] == True]
        available_leagues = ["OVERALL"] + sorted(
            active_meta["league_name"].dropna().unique().tolist()
        )
    else:
        available_leagues = ["OVERALL"]

    # Preselect league from URL if present
    pre = st.session_state.get("preselect_league", "") or qp_get("league", "")
    default_idx = available_leagues.index(pre) if (pre and pre in available_leagues) else 0

    target_league = st.selectbox(
        "Select View",
        available_leagues,
        index=default_idx,
        key="lb_league",
    )

    # Keep URL in sync (do not indent other blocks under this try/except)
    try:
        st.query_params["page"] = "leaderboards"
        st.query_params["league"] = target_league
        if PUBLIC_MODE:
            st.query_params["public"] = "1"
    except Exception:
        pass

    # -------------------------
    # Share link + open button (works in admin + public)
    # -------------------------
    public_url = build_public_url(page="leaderboards", params={"league": target_league})
    st.caption("Public standings link")
    st.text_input("", value=public_url, label_visibility="collapsed")
    public_link_button("Open Public Standings", public_url)

    # -------------------------
    # Min games requirement (league views only)
    # -------------------------
    min_games_req = 0
    if target_league != "OVERALL" and df_meta is not None and not df_meta.empty:
        cfg = df_meta[df_meta["league_name"] == target_league]
        if not cfg.empty:
            try:
                min_games_req = int(cfg.iloc[0].get("min_games", 0) or 0)
            except Exception:
                min_games_req = 0

    inactive_hidden = 0

    # -------------------------
    # Build display_df
    # -------------------------
    display_df = None

    if target_league == "OVERALL":
        display_df = df_players.copy() if df_players is not None else pd.DataFrame()

        # Normalize expected columns
        if not display_df.empty:
            if "name" not in display_df.columns and "id" in display_df.columns:
                display_df["name"] = display_df["id"].map(id_to_name)

            if "starting_rating" not in display_df.columns:
                display_df["starting_rating"] = display_df.get("rating", 1200.0)

            for c in ["wins", "losses", "matches_played", "rating"]:
                if c not in display_df.columns:
                    display_df[c] = 0

    else:
        # League-specific view: prefer df_leagues (preloaded), else fetch from league_ratings
        display_df = pd.DataFrame()

        if df_leagues is not None and not df_leagues.empty and "league_name" in df_leagues.columns:
            display_df = df_leagues[df_leagues["league_name"] == target_league].copy()

        # If df_leagues doesn't have it (or is empty), fetch directly from league_ratings
        if display_df.empty:
            try:
                lr_resp = (
                    supabase.table("league_ratings")
                    .select(
                        "player_id,league_name,rating,starting_rating,wins,losses,matches_played,is_active"
                    )
                    .eq("club_id", club_id)
                    .eq("league_name", target_league)
                    .execute()
                )
                display_df = (
                    pd.DataFrame(lr_resp.data)
                    if lr_resp and lr_resp.data is not None
                    else pd.DataFrame()
                )
            except Exception as e:
                admin_logged_in = bool(
                    getattr(ctx, "admin_logged_in", st.session_state.get("admin_logged_in", False))
                )
                if (not PUBLIC_MODE) and admin_logged_in:
                    st.warning(f"Could not fetch league_ratings for {target_league}: {e}")
                display_df = pd.DataFrame()

        # Per-league inactive filtering if available
        if not display_df.empty:
            if "is_active" in display_df.columns:
                inactive_hidden = int((display_df["is_active"] == False).sum())
                display_df = display_df[display_df["is_active"] == True].copy()

            # Attach names and normalize required columns
            if "name" not in display_df.columns:
                display_df["name"] = display_df["player_id"].map(id_to_name)

            if "starting_rating" not in display_df.columns:
                display_df["starting_rating"] = display_df["rating"]

            for c in ["wins", "losses", "matches_played", "rating"]:
                if c not in display_df.columns:
                    display_df[c] = 0

        if inactive_hidden > 0:
            st.caption(
                f"{inactive_hidden} inactive player(s) hidden from Standings/Top Performers for this league."
            )

    # -------------------------
    # Render guard
    # -------------------------
    if display_df is None or display_df.empty or "rating" not in display_df.columns:
        st.info("No data.")
        return

    display_df["JUPR"] = display_df["rating"].astype(float) / 400.0
    mp = display_df["matches_played"].replace(0, 1).astype(float)
    display_df["Win %"] = (display_df["wins"].astype(float) / mp) * 100.0

    def calc_gain(row):
        cur = float(row.get("rating", 0) or 0)
        start = float(row.get("starting_rating", cur) or cur)
        return cur - start

    display_df["rating_gain"] = display_df.apply(calc_gain, axis=1)

    # -------------------------
    # Top performers (league views)
    # -------------------------
    if target_league != "OVERALL":
        qualified_df = display_df[
            display_df["matches_played"].astype(int) >= int(min_games_req)
        ].copy()

        if not qualified_df.empty:
            st.markdown(f"### 🏅 Top Performers (Min {min_games_req} Games)")
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown("**👑 Highest Rating**")
                top = qualified_df.sort_values("rating", ascending=False).head(5)
                for _, r in top.iterrows():
                    st.markdown(f"**{float(r['JUPR']):.3f}** - {r['name']}")

            with c2:
                st.markdown("**🔥 Most Improved**")
                top = qualified_df.sort_values("rating_gain", ascending=False).head(5)
                for _, r in top.iterrows():
                    st.markdown(f"**{(float(r['rating_gain'])/400.0):+.3f}** - {r['name']}")

            with c3:
                st.markdown("**🎯 Best Win %**")
                top = qualified_df.sort_values("Win %", ascending=False).head(5)
                for _, r in top.iterrows():
                    st.markdown(f"**{float(r['Win %']):.1f}%** - {r['name']}")

            with c4:
                st.markdown("**🚜 Most Wins**")
                top = qualified_df.sort_values("wins", ascending=False).head(5)
                for _, r in top.iterrows():
                    st.markdown(f"**{int(r['wins'])} Wins** - {r['name']}")

            st.divider()

    # -------------------------
    # Standings table
    # -------------------------
    st.markdown("### 📊 Standings")
    final_view = display_df.sort_values("rating", ascending=False).copy()

    # Canonical pid column for links
    if target_league == "OVERALL":
        final_view["_pid"] = final_view["id"].astype(int) if ("id" in final_view.columns) else None
    else:
        final_view["_pid"] = (
            final_view["player_id"].astype(int) if ("player_id" in final_view.columns) else None
        )

    final_view["RankNum"] = range(1, len(final_view) + 1)

    def _rank_badge(r):
        r = int(r)
        if r == 1:
            return "🥇"
        if r == 2:
            return "🥈"
        if r == 3:
            return "🥉"
        return str(r)

    final_view["Rank"] = final_view["RankNum"].apply(_rank_badge)
    final_view["Gain"] = (final_view["rating_gain"].astype(float) / 400.0).map("{:+.3f}".format)

    def _name_link(row):
        nm = str(row.get("name", "—") or "—")
        pid = row.get("_pid", None)
        if pid is None or (isinstance(pid, float) and pd.isna(pid)):
            return nm
        url = build_player_profile_link(int(pid), public=PUBLIC_MODE)
        safe_nm = nm.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"<a href='{url}' target='_self'>{safe_nm}</a>"

    final_view["Player"] = final_view.apply(_name_link, axis=1)

    final_view["JUPR"] = final_view["rating"].astype(float) / 400.0
    mp2 = final_view["matches_played"].replace(0, 1).astype(float)
    final_view["Win %"] = (final_view["wins"].astype(float) / mp2) * 100.0

    tbl = final_view[
        ["Rank", "Player", "JUPR", "Gain", "matches_played", "wins", "losses", "Win %"]
    ].copy()

    tbl["JUPR"] = tbl["JUPR"].map(lambda x: f"{float(x):.3f}")
    tbl["Win %"] = tbl["Win %"].map(lambda x: f"{float(x):.1f}%")
    tbl = tbl.rename(columns={"matches_played": "MP", "wins": "W", "losses": "L"})

    html = tbl.to_html(index=False, escape=False)

    st.markdown(
        f"""
        <div class="lbtable">
          <style>
            .lbtable {{ width: 100%; overflow-x: auto; }}
            .lbtable table {{ width: 100%; border-collapse: collapse; }}
            .lbtable th, .lbtable td {{
              padding: 8px;
              border-bottom: 1px solid rgba(0,0,0,0.08);
              text-align: left;
              vertical-align: middle;
              white-space: nowrap;
            }}
            .lbtable th {{ font-weight: 700; }}
            .lbtable a {{ text-decoration: underline; }}
          </style>
          {html}
        </div>
        """,
        unsafe_allow_html=True,
    )
