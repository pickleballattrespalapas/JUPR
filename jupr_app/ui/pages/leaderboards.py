import html
import streamlit as st
import pandas as pd

from jupr_app.ui.url import qp_get
from jupr_app.ui.helpers import build_player_profile_link
from jupr_app.ui.public_links import build_public_url, public_link_button
from jupr_app.ui.layout import page_shell
from jupr_app.ui.theme_clean import callout
from jupr_app.ui.theme import color_for_delta


def render_top_performers_cards(
    top_perf_dict=None,
    qualified_df=None,
    title="Top Performers (Min 6 Games)",
):
    if top_perf_dict is None:
        if qualified_df is None or qualified_df.empty:
            return

        def _build_entries(df, sort_key, value_fn):
            top = df.sort_values(sort_key, ascending=False).head(5)
            entries = []
            for _, r in top.iterrows():
                entries.append(
                    {
                        "value": value_fn(r),
                        "name": str(r.get("name", "")),
                    }
                )
            return entries

        top_perf_dict = [
            {
                "label": "Highest Rating",
                "entries": _build_entries(
                    qualified_df,
                    "rating",
                    lambda r: f"{float(r['JUPR']):.3f}",
                ),
            },
            {
                "label": "Most Improved",
                "entries": _build_entries(
                    qualified_df,
                    "rating_gain",
                    lambda r: f"{(float(r['rating_gain'])/400.0):+.3f}",
                ),
            },
            {
                "label": "Best Win %",
                "entries": _build_entries(
                    qualified_df,
                    "Win %",
                    lambda r: f"{float(r['Win %']):.1f}%",
                ),
            },
            {
                "label": "Most Wins",
                "entries": _build_entries(
                    qualified_df,
                    "wins",
                    lambda r: f"{int(r['wins'])}",
                ),
            },
        ]

    if not top_perf_dict:
        return

    accent = st.get_option("theme.primaryColor") or "#4C78A8"

    st.markdown(f"### 🏅 {title}")
    st.markdown(
        """
        <style>
        .tp-cards {
            margin-top: 8px;
        }
        .tp-cards .tp-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 14px 16px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.18);
            border-top: 3px solid var(--tp-accent);
            min-height: 196px;
        }
        .tp-cards .tp-label {
            font-size: 11px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: rgba(255,255,255,0.55);
            margin-bottom: 8px;
        }
        .tp-cards .tp-value {
            font-size: 26px;
            font-weight: 700;
            color: var(--tp-accent);
            margin-bottom: 4px;
        }
        .tp-cards .tp-name {
            font-size: 14px;
            color: rgba(255,255,255,0.82);
            margin-bottom: 10px;
        }
        .tp-cards .tp-list {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .tp-cards .tp-list-item {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: rgba(255,255,255,0.55);
        }
        .tp-cards .tp-list-value {
            font-weight: 600;
            color: rgba(255,255,255,0.7);
            margin-right: 10px;
            white-space: nowrap;
        }
        .tp-cards .tp-list-name {
            color: rgba(255,255,255,0.6);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            max-width: 150px;
            text-align: right;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f'<div class="tp-cards" style="--tp-accent: {accent};">', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, card in zip(cols, top_perf_dict):
        entries = card.get("entries", [])
        if not entries:
            continue
        primary = entries[0]
        secondary = entries[1:5]
        list_items = "".join(
            f'<div class="tp-list-item"><span class="tp-list-value">{html.escape(entry["value"])}</span>'
            f'<span class="tp-list-name">{html.escape(entry["name"])}</span></div>'
            for entry in secondary
        )
        card_html = f"""
        <div class="tp-card">
            <div class="tp-label">{html.escape(str(card.get("label", "")))}</div>
            <div class="tp-value">{html.escape(primary["value"])}</div>
            <div class="tp-name">{html.escape(primary["name"])}</div>
            <div class="tp-list">{list_items}</div>
        </div>
        """
        with col:
            st.markdown(card_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


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

    admin_logged_in = bool(
        getattr(ctx, "admin_logged_in", st.session_state.get("admin_logged_in", False))
    )

    mode_label = "Public" if PUBLIC_MODE else "Admin"
    page_shell("🏆 Leaderboards", "Standings and top performers by league.", mode_label=mode_label)

    # -------------------------
    # Available leagues
    # -------------------------
    available_leagues = ["OVERALL"]

    try:
        if df_meta is not None and not df_meta.empty and "is_active" in df_meta.columns:
            meta = df_meta.copy()
            if "league_name" in meta.columns:
                meta["league_name"] = meta["league_name"].fillna("").astype(str).str.strip()
                active_meta = meta[meta["is_active"] == True].copy()
                leagues = (
                    active_meta["league_name"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .unique()
                    .tolist()
                )
                leagues = [x for x in leagues if x and x.upper() != "OVERALL"]
                available_leagues = ["OVERALL"] + sorted(leagues)
    except Exception:
        available_leagues = ["OVERALL"]

    # Preselect league from URL if present
    pre = (st.session_state.get("preselect_league", "") or qp_get("league", "") or "").strip()
    default_idx = 0
    if pre and pre in available_leagues:
        default_idx = available_leagues.index(pre)

    target_league = st.selectbox(
        "Select View",
        available_leagues,
        index=default_idx,
        key="lb_league",
    )

    # Keep URL in sync
    try:
        st.query_params["page"] = "leaderboards"
        st.query_params["league"] = target_league
        if PUBLIC_MODE:
            st.query_params["public"] = "1"
        else:
            # Avoid leaving stale public=1 around in admin mode
            try:
                st.query_params.pop("public", None)
            except Exception:
                pass
    except Exception:
        pass

    # -------------------------
    # Share link + open button (ADMIN ONLY)
    # -------------------------
    public_url = build_public_url(page="leaderboards", params={"league": target_league})
    if not PUBLIC_MODE:
        with st.container(border=True):
            st.markdown("### Public standings")
            st.caption("Share this link with players.")
            st.text_input("", value=public_url, label_visibility="collapsed")
            public_link_button("Open Public Standings", public_url)

    # -------------------------
    # Min games requirement (league views only)
    # -------------------------
    min_games_req = 0
    if target_league != "OVERALL":
        try:
            if df_meta is not None and not df_meta.empty and "league_name" in df_meta.columns:
                cfg = df_meta.copy()
                cfg["league_name"] = cfg["league_name"].fillna("").astype(str).str.strip()
                hit = cfg[cfg["league_name"] == str(target_league).strip()]
                if not hit.empty:
                    min_games_req = int(hit.iloc[0].get("min_games", 0) or 0)
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
        # League-specific view: prefer df_leagues (preloaded), else fetch league_ratings
        display_df = pd.DataFrame()

        if df_leagues is not None and not df_leagues.empty and "league_name" in df_leagues.columns:
            tmp = df_leagues.copy()
            tmp["league_name"] = tmp["league_name"].fillna("").astype(str).str.strip()
            display_df = tmp[tmp["league_name"] == str(target_league).strip()].copy()

        if display_df.empty:
            try:
                lr_resp = (
                    supabase.table("league_ratings")
                    .select("player_id,league_name,rating,starting_rating,wins,losses,matches_played,is_active")
                    .eq("club_id", club_id)
                    .eq("league_name", str(target_league).strip())
                    .execute()
                )
                display_df = pd.DataFrame(lr_resp.data or [])
            except Exception as e:
                if (not PUBLIC_MODE) and admin_logged_in:
                    st.warning(f"Could not fetch league_ratings for {target_league}: {e}")
                display_df = pd.DataFrame()

        # Per-league inactive filtering if available
        if not display_df.empty:
            if "is_active" in display_df.columns:
                try:
                    inactive_hidden = int((display_df["is_active"] == False).sum())
                except Exception:
                    inactive_hidden = 0
                display_df = display_df[display_df["is_active"] == True].copy()

            if "name" not in display_df.columns:
                display_df["name"] = display_df["player_id"].map(id_to_name)

            if "starting_rating" not in display_df.columns:
                display_df["starting_rating"] = display_df.get("rating", 1200.0)

            for c in ["wins", "losses", "matches_played", "rating"]:
                if c not in display_df.columns:
                    display_df[c] = 0

        # ADMIN ONLY message (never show in public mode)
        if inactive_hidden > 0 and (not PUBLIC_MODE):
            callout(
                "info",
                "Heads up",
                f"{inactive_hidden} inactive player(s) hidden from Standings/Top Performers for this league.",
            )

    # -------------------------
    # Render guard
    # -------------------------
    if display_df is None or display_df.empty or "rating" not in display_df.columns:
        st.info("No data.")
        return

    # Defensive numeric conversions
    display_df = display_df.copy()
    display_df["rating"] = pd.to_numeric(display_df["rating"], errors="coerce").fillna(0.0)
    display_df["starting_rating"] = pd.to_numeric(display_df.get("starting_rating", display_df["rating"]), errors="coerce").fillna(display_df["rating"])
    display_df["wins"] = pd.to_numeric(display_df.get("wins", 0), errors="coerce").fillna(0).astype(int)
    display_df["losses"] = pd.to_numeric(display_df.get("losses", 0), errors="coerce").fillna(0).astype(int)
    display_df["matches_played"] = pd.to_numeric(display_df.get("matches_played", 0), errors="coerce").fillna(0).astype(int)

    display_df["JUPR"] = display_df["rating"].astype(float) / 400.0
    mp = display_df["matches_played"].replace(0, 1).astype(float)
    display_df["Win %"] = (display_df["wins"].astype(float) / mp) * 100.0
    display_df["rating_gain"] = (display_df["rating"] - display_df["starting_rating"]).astype(float)

    # -------------------------
    # Top performers (league views)
    # -------------------------
    if target_league != "OVERALL":
        qualified_df = display_df[display_df["matches_played"].astype(int) >= int(min_games_req)].copy()

        if not qualified_df.empty:
            render_top_performers_cards(
                qualified_df=qualified_df,
                title=f"Top Performers (Min {min_games_req} Games)",
            )
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
        final_view["_pid"] = final_view["player_id"].astype(int) if ("player_id" in final_view.columns) else None

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
    final_view["Gain"] = final_view["rating_gain"].astype(float) / 400.0

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

    tbl = final_view[["Rank", "Player", "JUPR", "Gain", "matches_played", "wins", "losses", "Win %"]].copy()
    tbl["JUPR"] = tbl["JUPR"].map(lambda x: f"{float(x):.3f}")
    tbl["Gain"] = tbl["Gain"].map(
        lambda x: (
            f"<span style='color: {color_for_delta(x)};'>{float(x):+.3f}</span>"
            if pd.notna(x)
            else ""
        )
    )
    tbl["Win %"] = tbl["Win %"].map(lambda x: f"{float(x):.1f}%")
    tbl = tbl.rename(columns={"matches_played": "MP", "wins": "W", "losses": "L"})

    html = tbl.to_html(index=False, escape=False)

    st.markdown(
        f"""
        <div class="lbtable">
          {html}
        </div>
        """,
        unsafe_allow_html=True,
    )
