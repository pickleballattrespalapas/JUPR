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
                    lambda r: f"{float(r['Win %']):.1f}%"
                    if pd.notna(r["Win %"])
                    else "—",
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
        .tp-cards .tp-card {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 14px 16px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.18);
            border-top: 3px solid var(--tp-accent);
        }
        .tp-cards .tp-label {
            font-size: 12px;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: rgba(255,255,255,0.55);
            margin-bottom: 6px;
        }
        .tp-cards .tp-value {
            font-size: 26px;
            font-weight: 700;
            color: var(--tp-accent);
            margin-bottom: 2px;
        }
        .tp-cards .tp-name {
            font-size: 14px;
            color: rgba(255,255,255,0.8);
            margin-bottom: 8px;
        }
        .tp-cards .tp-list {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .tp-cards .tp-list-item {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: rgba(255,255,255,0.55);
        }
        .tp-cards .tp-list-value {
            font-weight: 600;
            color: rgba(255,255,255,0.65);
            margin-right: 8px;
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

    qp_player = (qp_get("player", "") or "").strip()
    qp_pid_raw = (qp_get("pid", "") or "").strip()
    selected_pid_param = None
    if qp_pid_raw.isdigit():
        selected_pid_param = int(qp_pid_raw)

    if qp_player and not st.session_state.get("lb_search"):
        st.session_state["lb_search"] = qp_player

    st.text_input("Find yourself", key="lb_search")

    qualified_default = True if PUBLIC_MODE else False
    show_qualified_only = False
    show_inactive = False
    if target_league != "OVERALL":
        show_qualified_only = st.checkbox(
            "Qualified only",
            key="lb_qualified_only",
            value=qualified_default,
        )
        if not PUBLIC_MODE:
            show_inactive = st.checkbox(
                "Show inactive",
                key="lb_show_inactive",
                value=False,
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
                if PUBLIC_MODE or (not show_inactive):
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

    # Defensive numeric conversions + canonical columns
    display_df = display_df.copy()
    if target_league == "OVERALL":
        display_df["_pid"] = display_df["id"].astype(int) if ("id" in display_df.columns) else None
    else:
        display_df["_pid"] = (
            display_df["player_id"].astype(int) if ("player_id" in display_df.columns) else None
        )

    if "name" in display_df.columns:
        display_df["name"] = display_df["name"].astype(str)
    else:
        display_df["name"] = ""
    display_df["rating"] = pd.to_numeric(display_df["rating"], errors="coerce").fillna(0.0)
    display_df["starting_rating"] = pd.to_numeric(
        display_df.get("starting_rating", display_df["rating"]), errors="coerce"
    ).fillna(display_df["rating"])
    display_df["wins"] = (
        pd.to_numeric(display_df.get("wins", 0), errors="coerce").fillna(0).astype(int)
    )
    display_df["losses"] = (
        pd.to_numeric(display_df.get("losses", 0), errors="coerce").fillna(0).astype(int)
    )
    display_df["matches_played"] = (
        pd.to_numeric(display_df.get("matches_played", 0), errors="coerce")
        .fillna(0)
        .astype(int)
    )

    if "is_active" not in display_df.columns:
        display_df["is_active"] = pd.NA

    display_df["JUPR"] = display_df["rating"].astype(float) / 400.0
    display_df["rating_gain"] = (display_df["rating"] - display_df["starting_rating"]).astype(float)
    display_df["Gain"] = display_df["rating_gain"].astype(float) / 400.0
    display_df["Win %"] = display_df.apply(
        lambda r: (
            (float(r["wins"]) / float(r["matches_played"]) * 100.0)
            if int(r["matches_played"]) > 0
            else pd.NA
        ),
        axis=1,
    )

    if target_league != "OVERALL":
        if "matches_played" in display_df.columns:
            try:
                display_df["Qualified"] = display_df["matches_played"].astype(int) >= int(
                    min_games_req or 0
                )
            except Exception:
                display_df["Qualified"] = False
        else:
            display_df["Qualified"] = False

    if target_league != "OVERALL" and "Qualified" not in display_df.columns:
        display_df["Qualified"] = False

    final_view = display_df.sort_values("rating", ascending=False, kind="mergesort").copy()
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
    final_view["Gap"] = (final_view["rating"].shift(1) - final_view["rating"]) / 400.0

    # -------------------------
    # Player-first: My Snapshot
    # -------------------------
    selected_row = None
    selected_pid = None
    search_text = (st.session_state.get("lb_search") or "").strip()

    if selected_pid_param is not None and "_pid" in final_view.columns:
        hit = final_view[final_view["_pid"] == int(selected_pid_param)]
        if not hit.empty:
            selected_row = hit.iloc[0]
            selected_pid = int(selected_row["_pid"])

    if selected_row is None and search_text:
        matches = final_view[
            final_view["name"].astype(str).str.contains(search_text, case=False, na=False)
        ].copy()
        matches = matches[pd.notna(matches["_pid"])].copy()
        if len(matches) == 1:
            selected_row = matches.iloc[0]
            selected_pid = int(selected_row["_pid"]) if pd.notna(selected_row["_pid"]) else None
        elif len(matches) > 1:
            options = matches["_pid"].tolist()
            pick_pid = st.selectbox(
                "Select player",
                options,
                key="lb_pick_player",
                format_func=lambda x: str(
                    matches.loc[matches["_pid"] == x, "name"].iloc[0]
                ),
            )
            pick_row = matches[matches["_pid"] == pick_pid]
            if not pick_row.empty:
                selected_row = pick_row.iloc[0]
                selected_pid = int(selected_row["_pid"]) if pd.notna(selected_row["_pid"]) else None
        else:
            st.info("No matching player found.")

    if selected_row is not None:
        win_pct_display = (
            f"{float(selected_row['Win %']):.1f}%"
            if pd.notna(selected_row["Win %"])
            else "—"
        )
        gain_value = float(selected_row["Gain"]) if pd.notna(selected_row["Gain"]) else 0.0
        gain_color = color_for_delta(gain_value)
        with st.container(border=True):
            st.markdown("### 👤 My Snapshot")
            st.markdown(
                f"""
                <div style="display:flex; flex-wrap:wrap; gap:16px;">
                    <div><strong>Rank</strong><br>{selected_row['Rank']} (#{int(selected_row['RankNum'])})</div>
                    <div><strong>Name</strong><br>{html.escape(str(selected_row['name']))}</div>
                    <div><strong>JUPR</strong><br>{float(selected_row['JUPR']):.3f}</div>
                    <div><strong>Gain</strong><br><span style="color:{gain_color};">{gain_value:+.3f}</span></div>
                    <div><strong>MP</strong><br>{int(selected_row['matches_played'])}</div>
                    <div><strong>W-L</strong><br>{int(selected_row['wins'])}-{int(selected_row['losses'])}</div>
                    <div><strong>Win %</strong><br>{win_pct_display}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if selected_pid is not None:
                player_url = build_player_profile_link(int(selected_pid), public=PUBLIC_MODE)
                try:
                    st.link_button("View full player page", player_url)
                except Exception:
                    st.markdown(f"[View full player page]({player_url})")

        selected_idx = int(selected_row["RankNum"]) - 1
        start_idx = max(selected_idx - 3, 0)
        end_idx = min(selected_idx + 4, len(final_view))
        neighborhood = final_view.iloc[start_idx:end_idx].copy()
        neighborhood = neighborhood[
            ["Rank", "name", "JUPR", "Gain", "matches_played", "wins", "losses", "Win %"]
        ].rename(
            columns={
                "name": "Player",
                "matches_played": "MP",
                "wins": "W",
                "losses": "L",
            }
        )
        neighborhood["JUPR"] = neighborhood["JUPR"].map(lambda x: f"{float(x):.3f}")
        neighborhood["Gain"] = neighborhood["Gain"].map(lambda x: f"{float(x):+.3f}")
        neighborhood["Win %"] = neighborhood["Win %"].map(
            lambda x: f"{float(x):.1f}%" if pd.notna(x) else "—"
        )

        st.markdown("#### Around Me")
        st.caption("Your neighborhood in the standings.")
        def _gain_style_cell(v):
            try:
                if pd.isna(v):
                    return ""
            except Exception:
                return ""
            try:
                return f"color: {color_for_delta(float(v))}"
            except Exception:
                return ""

        neighborhood_style = neighborhood.style.applymap(_gain_style_cell, subset=["Gain"])
        st.dataframe(neighborhood_style, use_container_width=True, hide_index=True)

        link_params = {"league": target_league}
        if selected_pid is not None:
            link_params["pid"] = str(selected_pid)
        else:
            link_params["player"] = str(selected_row["name"])

        share_url = build_public_url(page="leaderboards", params=link_params)
        st.text_input(
            "Copy link to this view",
            value=share_url,
            label_visibility="collapsed",
        )
        st.divider()

    # -------------------------
    # Top performers (league views)
    # -------------------------
    if target_league != "OVERALL":
        qualified_df = display_df[display_df["matches_played"].astype(int) >= int(min_games_req)].copy()

        if not qualified_df.empty:
            with st.expander(
                "Top Performers",
                expanded=not PUBLIC_MODE,
            ):
                render_top_performers_cards(
                    qualified_df=qualified_df,
                    title=f"Top Performers (Min {min_games_req} Games)",
                )
            st.divider()

    # -------------------------
    # Standings table
    # -------------------------
    st.markdown("### 📊 Standings")

    standings = final_view.copy()
    if target_league != "OVERALL":
        if "Qualified" not in standings.columns:
            standings["Qualified"] = False
        if show_qualified_only and "Qualified" in standings.columns:
            standings = standings[standings["Qualified"] == True].copy()

    standings["Player"] = standings["name"].astype(str)
    if "Qualified" in standings.columns:
        standings["Qualified"] = standings["Qualified"].fillna(False).astype(bool)

    columns = ["Rank", "Player", "JUPR", "Gain", "Gap"]
    if target_league != "OVERALL" and "Qualified" in standings.columns:
        columns.append("Qualified")
    columns.extend(["matches_played", "wins", "losses", "Win %"])
    standings = standings[columns].rename(
        columns={
            "matches_played": "MP",
            "wins": "W",
            "losses": "L",
        }
    )

    if "Qualified" in standings.columns:
        standings["Qualified"] = standings["Qualified"].map(lambda x: "✓" if x else "")

    def _gain_style(v):
        try:
            if pd.isna(v):
                return ""
        except Exception:
            return ""
        try:
            return f"color: {color_for_delta(float(v))}"
        except Exception:
            return ""

    standings_style = standings.style.format(
        {
            "JUPR": lambda x: f"{float(x):.3f}",
            "Gain": lambda x: f"{float(x):+.3f}" if pd.notna(x) else "",
            "Gap": lambda x: f"{float(x):.3f}" if pd.notna(x) else "",
            "Win %": lambda x: f"{float(x):.1f}%" if pd.notna(x) else "—",
        }
    ).applymap(_gain_style, subset=["Gain"])

    if standings.empty:
        st.info("No data.")
    else:
        st.dataframe(standings_style, use_container_width=True, hide_index=True)

    # Keep URL in sync with selected player
    try:
        if selected_pid is not None:
            st.query_params["pid"] = str(selected_pid)
            try:
                st.query_params.pop("player", None)
            except Exception:
                pass
        else:
            try:
                st.query_params.pop("pid", None)
                st.query_params.pop("player", None)
            except Exception:
                pass
    except Exception:
        pass
