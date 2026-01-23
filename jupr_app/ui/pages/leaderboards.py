import html
import urllib.parse
import streamlit as st
import pandas as pd

from jupr_app.ui.url import qp_get
from jupr_app.ui.public_links import build_public_url, public_link_button
from jupr_app.ui.layout import page_shell
from jupr_app.ui.theme_clean import callout


def _delta_color(delta, up_color, flat_color, down_color):
    try:
        if pd.isna(delta):
            return flat_color
    except Exception:
        return flat_color
    try:
        delta = float(delta)
    except Exception:
        return flat_color
    if delta > 0:
        return up_color
    if delta < 0:
        return down_color
    return flat_color


def _format_win_pct(value):
    if pd.notna(value):
        return f"{float(value):.1f}%"
    return "—"


def _player_profile_url(pid, public_mode, ctx):
    if pd.isna(pid):
        return None
    try:
        player_id = int(pid)
    except Exception:
        return None
    params = {"page": "players", "pid": str(player_id)}
    if public_mode:
        params["public"] = "1"
        if getattr(ctx, "club_id", None):
            params["club_id"] = str(ctx.club_id)
    query = urllib.parse.urlencode(params, quote_via=urllib.parse.quote_plus)
    return f"/?{query}"


def _build_player_link(pid, name, public_mode, ctx):
    safe_name = html.escape(str(name))
    url = _player_profile_url(pid, public_mode, ctx)
    if not url:
        return safe_name
    return f'<a href="{url}" style="color: inherit; text-decoration: none;">{safe_name}</a>'


def _render_hero_section_html(top3_rows, ctx):
    if top3_rows is None or top3_rows.empty:
        return ""

    accent = st.get_option("theme.primaryColor") or "#6AA6FF"
    hero_cards = []
    for idx, (_, row) in enumerate(top3_rows.iterrows(), start=1):
        rank_class = "is-top" if idx == 1 else ("is-second" if idx == 2 else "is-third")
        rank_style = f"color:{accent};" if idx != 1 else "color: rgba(255,255,255,0.8);"
        name_html = _build_player_link(row["_pid"], row["name"], bool(ctx.public_mode), ctx)
        if idx == 1:
            name_html = f'<span class="lb-hero-accent">{name_html}</span>'
        hero_cards.append(
            f"""
            <div class="lb-hero-card {rank_class}">
                <div class="lb-hero-rank" style="{rank_style}">{idx}</div>
                <div class="lb-hero-name {'is-top' if idx == 1 else ''}">{name_html}</div>
                <div class="lb-hero-rating">{float(row['JUPR']):.3f}</div>
                <div class="lb-subtitle">{int(row['matches_played'])} games • {_format_win_pct(row['Win %'])} win %</div>
            </div>
            """
        )

    card_order = [
        hero_cards[1] if len(hero_cards) > 1 else "",
        hero_cards[0] if len(hero_cards) > 0 else "",
        hero_cards[2] if len(hero_cards) > 2 else "",
    ]

    return f"""
    <section class="lb-hero" style="--lb-accent: {accent};">
        <div class="lb-hero-band">
            <div class="lb-hero-grid">
                {''.join(card_order)}
            </div>
        </div>
    </section>
    """


def _render_story_highlights_html(highlights, ctx):
    if not highlights:
        return ""
    cards_html = "".join(
        f"""
        <div class="lb-story-card">
            <div class="lb-story-icon">{html.escape(card.get('icon', ''))}</div>
            <div class="lb-story-title">{html.escape(card.get('title', ''))}</div>
            <div class="lb-story-text">{card.get('text', '')}</div>
            <div class="lb-muted" style="font-size:12px;">{html.escape(card.get('secondary', ''))}</div>
        </div>
        """
        for card in highlights[:4]
    )
    return f'<div class="lb-story-grid">{cards_html}</div>'


def render_top_performers_cards(
    top_perf_dict=None,
    qualified_df=None,
    title="Top Performers (Min 6 Games)",
    compact_view=False,
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

    st.markdown(f"### {title}")
    st.markdown(
        """
        <style>
        .tp-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: 0 6px 16px rgba(0,0,0,0.16);
            border-top: 3px solid var(--tp-accent);
        }
        .tp-label {
            font-size: 12px;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: rgba(255,255,255,0.55);
            margin-bottom: 6px;
        }
        .tp-value {
            font-size: 24px;
            font-weight: 700;
            color: var(--tp-accent);
            margin-bottom: 2px;
        }
        .tp-name {
            font-size: 14px;
            color: rgba(255,255,255,0.8);
            margin-bottom: 8px;
        }
        .tp-list {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .tp-list-item {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: rgba(255,255,255,0.55);
        }
        .tp-list-value {
            font-weight: 600;
            color: rgba(255,255,255,0.65);
            margin-right: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    column_count = 2 if compact_view else 4
    cols = st.columns(column_count)
    for idx, card in enumerate(top_perf_dict):
        col = cols[idx % column_count]
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
        <div class="tp-card" style="--tp-accent: {accent};">
            <div class="tp-label">{html.escape(str(card.get("label", "")))}</div>
            <div class="tp-value">{html.escape(primary["value"])}</div>
            <div class="tp-name">{html.escape(primary["name"])}</div>
            <div class="tp-list">{list_items}</div>
        </div>
        """
        with col:
            st.markdown(card_html, unsafe_allow_html=True)


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
    page_shell("Leaderboards", "Standings and standout moments by league.", mode_label=mode_label)
    st.markdown(
        """
        <style>
        .lb-wrap {
            max-width: 1120px;
            margin: 0 auto;
            padding-bottom: 24px;
        }
        .lb-card {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.16);
        }
        .lb-row {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }
        .lb-kpi {
            flex: 1 1 140px;
            min-width: 120px;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 10px 12px;
        }
        .lb-kpi .lb-kpi-label {
            font-size: 11px;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: rgba(255,255,255,0.6);
            margin-bottom: 4px;
        }
        .lb-kpi .lb-kpi-value {
            font-size: 18px;
            font-weight: 700;
        }
        .lb-title {
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 2px;
        }
        .lb-subtitle {
            font-size: 13px;
            color: rgba(255,255,255,0.65);
        }
        .lb-muted {
            color: rgba(255,255,255,0.6);
        }
        .lb-hero {
            border-radius: 22px;
            padding: 24px;
            background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.01));
            border: 1px solid rgba(255,255,255,0.08);
        }
        .lb-hero-band {
            width: 100%;
        }
        .lb-hero-grid {
            display: flex;
            gap: 16px;
            align-items: stretch;
            flex-wrap: nowrap;
        }
        .lb-hero-card {
            flex: 1 1 0;
            background: rgba(255,255,255,0.03);
            border-radius: 18px;
            padding: 18px;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .lb-hero-card.is-top {
            flex: 1.3 1 0;
            background: rgba(255,255,255,0.04);
        }
        .lb-hero-card.is-second,
        .lb-hero-card.is-third {
            flex: 1 1 0;
        }
        .lb-hero-rank {
            font-size: 38px;
            font-weight: 700;
            line-height: 1;
            margin-bottom: 8px;
        }
        .lb-hero-name {
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 6px;
        }
        .lb-hero-name.is-top {
            font-size: 26px;
        }
        .lb-hero-rating {
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 6px;
        }
        .lb-hero-accent {
            display: inline-block;
            border-bottom: 2px solid var(--lb-accent);
            padding-bottom: 2px;
        }
        .lb-story-grid {
            display: grid;
            gap: 14px;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        }
        .lb-story-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 16px;
        }
        .lb-story-icon {
            font-size: 16px;
            margin-bottom: 10px;
            color: rgba(255,255,255,0.6);
        }
        .lb-story-title {
            font-size: 12px;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: rgba(255,255,255,0.6);
            margin-bottom: 6px;
        }
        .lb-story-text {
            font-size: 14px;
            color: rgba(255,255,255,0.85);
            margin-bottom: 6px;
        }
        .lb-standings-card {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 14px 16px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .lb-standings-top {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            align-items: center;
        }
        .lb-standings-rank {
            font-weight: 700;
            font-size: 16px;
            min-width: 38px;
        }
        .lb-standings-name {
            font-size: 15px;
            font-weight: 600;
        }
        .lb-standings-rating {
            font-size: 16px;
            font-weight: 700;
            text-align: right;
        }
        .lb-standings-stats {
            font-size: 12px;
            color: rgba(255,255,255,0.6);
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .lb-badge {
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.18);
            color: rgba(255,255,255,0.65);
        }
        .lb-controls {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 16px;
        }
        .lb-actions a,
        .lb-actions button {
            width: 100%;
            justify-content: center;
        }
        @media (max-width: 640px) {
            .lb-wrap {
                padding: 0 4px 24px;
            }
            .lb-card {
                padding: 16px;
            }
            .lb-row {
                flex-direction: column;
            }
            .lb-kpi {
                min-width: 0;
            }
            .lb-hero-grid {
                flex-wrap: wrap;
            }
            .lb-hero-card.is-top {
                order: 1;
                flex: 1 1 100%;
            }
            .lb-hero-card.is-second,
            .lb-hero-card.is-third {
                order: 2;
                flex: 1 1 47%;
            }
            .lb-story-grid {
                display: flex;
                overflow-x: auto;
                padding-bottom: 4px;
            }
            .lb-story-card {
                min-width: 220px;
            }
            .stButton > button,
            .stLinkButton > a {
                width: 100% !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    delta_up = "#6DBE7C"
    delta_flat = "#8D94A3"
    delta_down = "#C08A3C"

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

    st.session_state.setdefault("lb_league", available_leagues[default_idx])
    st.session_state.setdefault("lb_view_mode", "Story View")
    st.session_state.setdefault("lb_show_inactive", False)
    target_league = st.session_state.get("lb_league", available_leagues[default_idx])
    if target_league not in available_leagues:
        target_league = available_leagues[default_idx]
        st.session_state["lb_league"] = target_league

    qp_player = (qp_get("player", "") or "").strip()
    qp_pid_raw = (qp_get("pid", "") or "").strip()
    selected_pid_param = None
    if qp_pid_raw.isdigit():
        selected_pid_param = int(qp_pid_raw)

    if qp_player and not st.session_state.get("lb_search"):
        st.session_state["lb_search"] = qp_player

    show_inactive = False
    if target_league != "OVERALL":
        if not PUBLIC_MODE:
            show_inactive = bool(st.session_state.get("lb_show_inactive", False))

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
    inactive_notice = None

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
            inactive_notice = (
                f"{inactive_hidden} inactive player(s) hidden from standings for this league."
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

    final_view["Rank"] = final_view["RankNum"].astype(int).astype(str)
    final_view["Gap"] = (final_view["rating"].shift(1) - final_view["rating"]) / 400.0

    # -------------------------
    # Hero podium
    # -------------------------
    podium = final_view.head(3).copy()
    st.markdown("### Who’s on top right now?")
    if not podium.empty:
        hero_html = _render_hero_section_html(podium, ctx)
        if hero_html:
            st.markdown(hero_html, unsafe_allow_html=True)

    # -------------------------
    # Storytelling highlights
    # -------------------------
    view_mode = st.session_state.get("lb_view_mode", "Story View")
    show_highlights = view_mode == "Story View"
    if show_highlights:
        highlights = []
        recent_delta_col = None
        for col in [
            "rating_gain_recent",
            "recent_gain",
            "recent_delta",
            "delta_recent",
            "rating_delta_recent",
            "rating_delta_week",
            "rating_delta_7d",
        ]:
            if col in display_df.columns:
                recent_delta_col = col
                break

        if recent_delta_col is not None:
            recent_delta = pd.to_numeric(display_df[recent_delta_col], errors="coerce")
            riser = display_df.loc[recent_delta.idxmax()] if recent_delta.notna().any() else None
            slide = display_df.loc[recent_delta.idxmin()] if recent_delta.notna().any() else None

            if riser is not None and float(recent_delta.max()) > 0:
                delta_val = float(recent_delta.max()) / 400.0
                highlights.append(
                    {
                        "icon": "▲",
                        "title": "Biggest Riser",
                        "text": f"{_build_player_link(riser['_pid'], riser['name'], PUBLIC_MODE, ctx)} climbed {delta_val:+.3f} JUPR across {int(riser['matches_played'])} games.",
                        "secondary": f"Rating now {float(riser['JUPR']):.3f}",
                    }
                )
            if slide is not None and float(recent_delta.min()) < 0:
                delta_val = float(recent_delta.min()) / 400.0
                highlights.append(
                    {
                        "icon": "▼",
                        "title": "Toughest Slide",
                        "text": f"{_build_player_link(slide['_pid'], slide['name'], PUBLIC_MODE, ctx)} slipped {delta_val:+.3f} JUPR across {int(slide['matches_played'])} games.",
                        "secondary": f"Rating now {float(slide['JUPR']):.3f}",
                    }
                )

        active_df = display_df.copy()
        if "matches_played" in active_df.columns:
            active_df = active_df.sort_values("matches_played", ascending=False, kind="mergesort")
            if not active_df.empty and int(active_df.iloc[0]["matches_played"]) > 0:
                most_active = active_df.iloc[0]
                highlights.append(
                    {
                        "icon": "●",
                        "title": "Most Active",
                        "text": f"{_build_player_link(most_active['_pid'], most_active['name'], PUBLIC_MODE, ctx)} logged {int(most_active['matches_played'])} games.",
                        "secondary": f"Win rate {_format_win_pct(most_active['Win %'])}",
                    }
                )

        best_win_df = display_df.copy()
        if min_games_req > 0:
            best_win_df = best_win_df[
                best_win_df["matches_played"].astype(int) >= int(min_games_req)
            ].copy()
        best_win_df = best_win_df[pd.notna(best_win_df["Win %"])].copy()
        if not best_win_df.empty:
            best_win_df = best_win_df.sort_values("Win %", ascending=False, kind="mergesort")
            best_win = best_win_df.iloc[0]
            highlights.append(
                {
                    "icon": "◎",
                    "title": "Best Win %",
                    "text": f"{_build_player_link(best_win['_pid'], best_win['name'], PUBLIC_MODE, ctx)} holds {_format_win_pct(best_win['Win %'])} over {int(best_win['matches_played'])} games.",
                    "secondary": f"Rating {float(best_win['JUPR']):.3f}",
                }
            )

        if len(highlights) >= 2:
            st.markdown("### This Week at Tres Palapas")
            st.markdown(
                '<div class="lb-subtitle">Based on recent JUPR activity</div>',
                unsafe_allow_html=True,
            )
            story_html = _render_story_highlights_html(highlights, ctx)
            if story_html:
                st.markdown(story_html, unsafe_allow_html=True)

    # -------------------------
    # Controls
    # -------------------------
    st.markdown("### Full Standings")
    control_cols = st.columns([2.2, 1.2, 1.2])
    with control_cols[0]:
        try:
            target_league = st.segmented_control(
                "League",
                available_leagues,
                default=target_league,
                key="lb_league",
            )
        except Exception:
            target_league = st.radio(
                "League",
                available_leagues,
                index=available_leagues.index(target_league),
                horizontal=True,
                key="lb_league",
            )
    with control_cols[1]:
        try:
            view_mode = st.segmented_control(
                "View",
                ["Story View", "Stats View"],
                default=view_mode,
                key="lb_view_mode",
            )
        except Exception:
            view_mode = st.radio(
                "View",
                ["Story View", "Stats View"],
                index=0 if view_mode == "Story View" else 1,
                horizontal=True,
                key="lb_view_mode",
            )
    with control_cols[2]:
        st.text_input("Find player", key="lb_search")
    if target_league != "OVERALL" and not PUBLIC_MODE:
        st.checkbox("Show inactive", key="lb_show_inactive", value=False)

    # -------------------------
    # Share link + open button (ADMIN ONLY)
    # -------------------------
    public_url = build_public_url(page="leaderboards", params={"league": target_league})
    if not PUBLIC_MODE:
        with st.container(border=True):
            st.markdown("#### Public standings")
            st.caption("Share this link with players.")
            st.text_input("", value=public_url, label_visibility="collapsed")
            public_link_button("Open Public Standings", public_url)
    if inactive_notice and not PUBLIC_MODE:
        callout("info", "Heads up", inactive_notice)

    # -------------------------
    # Standings list
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
                format_func=lambda x: str(matches.loc[matches["_pid"] == x, "name"].iloc[0]),
            )
            pick_row = matches[matches["_pid"] == pick_pid]
            if not pick_row.empty:
                selected_row = pick_row.iloc[0]
                selected_pid = int(selected_row["_pid"]) if pd.notna(selected_row["_pid"]) else None
        else:
            st.info("No matching player found.")

    if selected_row is not None:
        win_pct_display = _format_win_pct(selected_row["Win %"])
        gain_value = float(selected_row["Gain"]) if pd.notna(selected_row["Gain"]) else 0.0
        gain_color = _delta_color(gain_value, delta_up, delta_flat, delta_down)
        st.markdown("#### My Snapshot")
        st.markdown(
            f"""
            <div class="lb-card">
                <div class="lb-row" style="align-items:center; justify-content:space-between;">
                    <div>
                        <div class="lb-title">{_build_player_link(selected_row['_pid'], selected_row['name'], PUBLIC_MODE, ctx)}</div>
                        <div class="lb-subtitle">Rank {int(selected_row['RankNum'])}</div>
                    </div>
                    <div style="font-size:20px; font-weight:700;">
                        {float(selected_row['JUPR']):.3f}
                        <div class="lb-muted" style="font-size:12px; font-weight:500;">JUPR</div>
                    </div>
                </div>
                <div class="lb-row" style="margin-top:12px;">
                    <div class="lb-kpi">
                        <div class="lb-kpi-label">Δ Rating</div>
                        <div class="lb-kpi-value" style="color:{gain_color};">{gain_value:+.3f}</div>
                    </div>
                    <div class="lb-kpi">
                        <div class="lb-kpi-label">Win %</div>
                        <div class="lb-kpi-value">{win_pct_display}</div>
                    </div>
                    <div class="lb-kpi">
                        <div class="lb-kpi-label">Games</div>
                        <div class="lb-kpi-value">{int(selected_row['matches_played'])}</div>
                    </div>
                    <div class="lb-kpi">
                        <div class="lb-kpi-label">W-L</div>
                        <div class="lb-kpi-value">{int(selected_row['wins'])}-{int(selected_row['losses'])}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if selected_pid is not None:
            player_url = _player_profile_url(int(selected_pid), PUBLIC_MODE, ctx)
        else:
            player_url = None
        if player_url:
            try:
                st.link_button("View full player page", player_url)
            except Exception:
                st.markdown(f"[View full player page]({player_url})")

        selected_idx = int(selected_row["RankNum"]) - 1
        start_idx = max(selected_idx - 2, 0)
        end_idx = min(selected_idx + 3, len(final_view))
        neighborhood = final_view.iloc[start_idx:end_idx].copy()
        st.markdown("##### Around Me")
        for _, row in neighborhood.iterrows():
            gain_val = float(row["Gain"]) if pd.notna(row["Gain"]) else 0.0
            gain_color = _delta_color(gain_val, delta_up, delta_flat, delta_down)
            st.markdown(
                f"""
                <div class="lb-standings-card" style="min-width: 200px;">
                    <div class="lb-standings-top">
                        <div class="lb-standings-rank">#{int(row['RankNum'])}</div>
                        <div class="lb-standings-name">{_build_player_link(row['_pid'], row['name'], PUBLIC_MODE, ctx)}</div>
                        <div class="lb-standings-rating">{float(row['JUPR']):.3f}</div>
                    </div>
                    <div class="lb-standings-stats">
                        <span>{int(row['matches_played'])} games</span>
                        <span>{_format_win_pct(row['Win %'])} win %</span>
                        <span style="color:{gain_color};">{gain_val:+.3f} Δ</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

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

    standings = final_view.copy()
    if target_league != "OVERALL":
        if "Qualified" not in standings.columns:
            standings["Qualified"] = False
    if "Qualified" in standings.columns:
        standings["Qualified"] = standings["Qualified"].fillna(False).astype(bool)

    st.session_state.setdefault("lb_limit", 50)
    limit = int(st.session_state.get("lb_limit", 50))
    if limit < 50:
        limit = 50
        st.session_state["lb_limit"] = 50

    for _, row in standings.head(limit).iterrows():
        gain_val = float(row["Gain"]) if pd.notna(row["Gain"]) else 0.0
        gain_color = _delta_color(gain_val, delta_up, delta_flat, delta_down)
        badges = []
        if target_league != "OVERALL" and not row.get("Qualified", True):
            badges.append("Min games not met")
        is_active_value = row.get("is_active")
        if pd.notna(is_active_value) and not bool(is_active_value):
            badges.append("Inactive")
        if row.get("matches_played", 0) == 0:
            badges.append("New")
        badge_html = "".join(f'<span class="lb-badge">{html.escape(b)}</span>' for b in badges)
        st.markdown(
            f"""
            <div class="lb-standings-card">
                <div class="lb-standings-top">
                    <div class="lb-standings-rank">#{int(row['RankNum'])}</div>
                    <div class="lb-standings-name">{_build_player_link(row['_pid'], row['name'], PUBLIC_MODE, ctx)}</div>
                    <div class="lb-standings-rating">{float(row['JUPR']):.3f}</div>
                </div>
                <div class="lb-standings-stats">
                    <span>{int(row['matches_played'])} games</span>
                    <span>{_format_win_pct(row['Win %'])} win %</span>
                    <span style="color:{gain_color};">{gain_val:+.3f} Δ</span>
                </div>
                <div class="lb-row" style="gap:6px;">{badge_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if len(standings) > limit:
        if st.button("Load more", key="lb_load_more"):
            st.session_state["lb_limit"] = limit + 50

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
