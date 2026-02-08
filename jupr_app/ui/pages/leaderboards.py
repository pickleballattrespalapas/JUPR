import html
import logging
import re
import textwrap
import urllib.parse
import streamlit as st
import pandas as pd
import altair as alt

from jupr_app.domain.awards import build_top_performer_entries
from jupr_app.ui.url import qp_get
from jupr_app.ui.public_links import build_public_url, public_link_button
from jupr_app.ui.layout import page_shell
from jupr_app.ui.theme_clean import callout
from jupr_app.ui.pages.players import badge_icon
from jupr_app.ui.helpers import build_badge_story, sanitize_story_text
from jupr_app.domain.gamification.requirements import load_requirements_map
from jupr_app.domain.story_stats import (
    build_best_partner_map,
    build_rival_map,
)


logger = logging.getLogger(__name__)
MAX_STORY_BADGES = 2
_STORY_HTML_RE = re.compile(
    r"</?(div|span|br|p|ul|li|strong|em|a|img|table|tr|td|th)\b",
    re.IGNORECASE,
)


def _safe_text(value: object) -> str:
    return html.escape("" if value is None else str(value))


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


def _format_pct(value: float | None) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100.0:.1f}%"
    except Exception:
        return "—"


def _build_rival_story(rival: dict, id_to_name: dict[int, str]) -> str:
    rival_id = rival.get("opponent_id")
    rival_name = id_to_name.get(int(rival_id), f"#{rival_id}") if rival_id is not None else "Unknown"
    games = int(rival.get("games", 0))
    win_pct = float(rival.get("win_pct", 0.0))
    win_pct_str = _format_pct(win_pct)
    loss_pct_str = _format_pct(1.0 - win_pct if games else 0.0)
    return (
        f"Rival: {rival_name} ({games} games, {win_pct_str} win%). "
        f"You and {rival_name} are {win_pct_str}-{loss_pct_str} over {games} games."
    )


def _build_partner_story(partner: dict, id_to_name: dict[int, str]) -> str:
    partner_id = partner.get("partner_id")
    partner_name = id_to_name.get(int(partner_id), f"#{partner_id}") if partner_id is not None else "Unknown"
    games = int(partner.get("games", 0))
    win_pct_str = _format_pct(partner.get("win_pct", 0.0))
    return (
        f"Best partner: {partner_name} ({games} games, {win_pct_str}). "
        f"Your best results come with {partner_name}: {win_pct_str} over {games} games."
    )


def _story_has_html(raw: str | None) -> bool:
    if not raw:
        return False
    return bool(_STORY_HTML_RE.search(str(raw)))


def _log_story_html_warning(
    raw_story: str | None,
    player_id: object,
    source: str,
    admin_logged_in: bool,
) -> None:
    if not admin_logged_in or not raw_story:
        return
    if not _story_has_html(raw_story):
        return
    snippet = re.sub(r"\s+", " ", str(raw_story).strip())[:120]
    logger.warning(
        "Leaderboards story HTML detected (player_id=%s, %s): %s",
        player_id,
        source,
        snippet,
    )


def _extract_story_from_row(row: pd.Series | dict) -> tuple[str | None, str | None]:
    for key in ("story", "story_text", "story_html"):
        try:
            if key not in row:
                continue
            value = row.get(key)
        except Exception:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            if value is None:
                continue
        text = str(value).strip()
        if text:
            return text, f"df_column:{key}"
    return None, None


def _compute_story_text(
    row: pd.Series | dict,
    earned_badges: list[dict],
    id_to_name: dict[int, str],
    rival_map: dict[int, dict],
    partner_map: dict[int, dict],
    admin_logged_in: bool,
) -> tuple[str, str | None]:
    raw, source = _extract_story_from_row(row)
    player_id = None
    if hasattr(row, "get"):
        player_id = row.get("_pid") or row.get("id") or row.get("player_id")
    if raw and _story_has_html(raw):
        _log_story_html_warning(raw, player_id, source or "df_column:unknown", admin_logged_in)
        raw = None
        source = None
    if not raw:
        raw = build_badge_story(row, earned_badges or [])
        source = "generated:build_badge_story"
        if not raw:
            raw = build_badge_story(row, [])
            if raw:
                source = "generated:build_badge_story:fallback"

    inserts = []
    if pd.notna(player_id):
        try:
            pid_int = int(player_id)
        except Exception:
            pid_int = None
        if pid_int is not None:
            rival = rival_map.get(pid_int) if rival_map else None
            if rival:
                inserts.append(_build_rival_story(rival, id_to_name))
            partner = partner_map.get(pid_int) if partner_map else None
            if partner:
                inserts.append(_build_partner_story(partner, id_to_name))
    if inserts:
        base = raw or ""
        raw = f"{base} {' '.join(inserts)}".strip()
        source = f"{source}+rival_partner" if source else "generated:rival_partner"

    raw = sanitize_story_text(raw or "")
    return raw, source


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
    safe_name = _safe_text(name)
    url = _player_profile_url(pid, public_mode, ctx)
    if not url:
        return safe_name
    return f'<a href="{url}" target="_self" class="lb-link">{safe_name}</a>'


def select_leaderboard_players(
    df_players_active: pd.DataFrame | None,
    df_players_all: pd.DataFrame | None,
    view_option: str,
) -> pd.DataFrame | None:
    if view_option == "See all":
        return df_players_all
    return df_players_active


def _fetch_story_badges(ctx, player_ids):
    if not player_ids:
        return pd.DataFrame()

    pb_df = getattr(ctx, "df_player_badges", None)
    badges_df = getattr(ctx, "df_badges", None)

    if (
        pb_df is not None
        and badges_df is not None
        and not pb_df.empty
        and not badges_df.empty
    ):
        pb_copy = pb_df.copy()
        if "club_id" in pb_copy.columns:
            pb_copy = pb_copy[pb_copy["club_id"].astype(str) == str(ctx.club_id)]
        pb_copy = pb_copy[pb_copy["player_id"].isin(player_ids)]
        if pb_copy.empty:
            return pd.DataFrame()
        return pb_copy.merge(badges_df, on="badge_id", how="left")

    try:
        resp = (
            ctx.supabase.table("player_badges")
            .select(
                "player_id,badge_id,earned_at,"
                "badges:badges(badge_id,name,prestige,category,icon_key,rarity)"
            )
            .eq("club_id", str(ctx.club_id))
            .in_("player_id", player_ids)
            .execute()
        )
    except Exception:
        logger.exception("Failed to fetch story badges")
        return pd.DataFrame()

    data = resp.data or []
    if not data:
        return pd.DataFrame()

    story_df = pd.json_normalize(data, sep=".")
    column_map = {
        "badges.badge_id": "badge_id",
        "badges.name": "name",
        "badges.prestige": "prestige",
        "badges.category": "category",
        "badges.icon_key": "icon_key",
        "badges.rarity": "rarity",
    }
    story_df = story_df.rename(columns=column_map)
    if "badge_id" in story_df.columns and "requirements" not in story_df.columns:
        requirements_map = load_requirements_map()
        story_df["requirements"] = story_df["badge_id"].map(requirements_map).fillna("Requirements TBD")
    return story_df


def _build_story_badge_map(badges_df: pd.DataFrame) -> dict[int, list[dict]]:
    if badges_df is None or badges_df.empty:
        return {}

    story_df = badges_df.copy()
    if "badge_id" not in story_df.columns or "player_id" not in story_df.columns:
        return {}

    earned_col = "earned_at" if "earned_at" in story_df.columns else "created_at"
    story_df["earned_at_dt"] = pd.to_datetime(
        story_df.get(earned_col, None), utc=True, errors="coerce"
    )
    story_df["prestige"] = pd.to_numeric(story_df.get("prestige", 0), errors="coerce").fillna(0)

    story_df = story_df.sort_values(
        ["player_id", "badge_id", "earned_at_dt"], ascending=[True, True, False]
    ).drop_duplicates(subset=["player_id", "badge_id"], keep="first")

    story_df = story_df.sort_values(
        ["prestige", "earned_at_dt"], ascending=[False, False]
    )

    badges_by_player: dict[int, list[dict]] = {}
    for _, row in story_df.iterrows():
        try:
            pid = int(row["player_id"])
        except Exception:
            continue
        badges_by_player.setdefault(pid, []).append(
            {
                "badge_id": row.get("badge_id"),
                "name": row.get("name", "Badge"),
                "prestige": int(row.get("prestige", 0) or 0),
                "category": row.get("category"),
                "icon_key": row.get("icon_key"),
                "rarity": row.get("rarity"),
                "earned_at_dt": row.get("earned_at_dt"),
            }
        )

    return badges_by_player


def _verify_story_badges(badges_by_player, story_player_ids, badges_df, admin_logged_in):
    if not admin_logged_in or not story_player_ids:
        return
    if badges_df is None or badges_df.empty:
        return
    try:
        eligible = badges_df[badges_df["player_id"].isin(story_player_ids)]
        if eligible.empty:
            return
        if not any(badges_by_player.get(pid) for pid in story_player_ids):
            logger.warning("Story View badge map empty despite eligible player badges.")
    except Exception:
        logger.exception("Story View badge verification failed")


def render_top_performers_cards(
    top_perf_dict=None,
    qualified_df=None,
    title="Top Performers (Min 6 Games)",
    compact_view=False,
    public_mode=False,
    ctx=None,
):
    if top_perf_dict is None:
        entries = build_top_performer_entries(qualified_df, limit=5)
        if not entries:
            return
        top_perf_dict = []
        for category in entries:
            rows = []
            for entry in category.get("entries", []):
                name = entry.get("name", "")
                pid = entry.get("player_id")
                name_html = (
                    _build_player_link(pid, name, public_mode, ctx)
                    if ctx is not None
                    else html.escape(str(name))
                )
                rows.append(
                    {
                        "value": entry.get("metric_display", "—"),
                        "name": str(name),
                        "name_html": name_html,
                    }
                )
            top_perf_dict.append(
                {
                    "label": category.get("label"),
                    "entries": rows,
                }
            )

    if not top_perf_dict:
        return

    accent = st.get_option("theme.primaryColor") or "var(--accent)"

    st.markdown(f"### {title}")
    st.markdown(
        """
        <style>
        .tp-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: var(--shadow);
            border-top: 3px solid var(--tp-accent);
        }
        .tp-label {
            font-size: 12px;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: var(--text-muted);
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
            color: var(--text-primary);
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
            color: var(--text-muted);
        }
        .tp-list-value {
            font-weight: 600;
            color: var(--text-secondary);
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
            f'<span class="tp-list-name">{entry.get("name_html", html.escape(entry["name"]))}</span></div>'
            for entry in secondary
        )
        card_html = f"""
        <div class="tp-card" style="--tp-accent: {accent};">
            <div class="tp-label">{html.escape(str(card.get("label", "")))}</div>
            <div class="tp-value">{html.escape(primary["value"])}</div>
            <div class="tp-name">{primary.get("name_html", html.escape(primary["name"]))}</div>
            <div class="tp-list">{list_items}</div>
        </div>
        """
        with col:
            st.markdown(card_html, unsafe_allow_html=True)


def render(ctx):
    # Always use 4-space indentation in this file.
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

    view_choice = st.radio("Show", ["Active", "See all"], index=0, horizontal=True)

    df_players = select_leaderboard_players(ctx.df_players_active, df_players_all, view_choice)
    st.markdown(
        """
        <style>
        .lb-wrap {
            max-width: 1120px;
            margin: 0 auto;
            padding-bottom: 24px;
        }
        .lb-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: var(--shadow);
        }
        .lb-link {
            color: inherit;
            text-decoration: none;
        }
        .lb-link:hover {
            text-decoration: underline;
            opacity: 0.95;
        }
        .lb-row {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }
        .lb-kpi {
            flex: 1 1 140px;
            min-width: 120px;
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 10px 12px;
        }
        .lb-kpi .lb-kpi-label {
            font-size: 11px;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: var(--text-muted);
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
            color: var(--text-muted);
        }
        .lb-muted {
            color: var(--text-muted);
        }
        .lb-standings-card {
            background: var(--panel);
            border: 1px solid var(--border);
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
            flex: 0 0 42px;
        }
        .lb-standings-name {
            font-size: 18px;
            font-weight: 650;
            color: var(--accent);
            letter-spacing: 0.2px;
            flex: 1 1 240px;
            min-width: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .lb-standings-name a {
            color: inherit;
        }
        .lb-standings-rating {
            font-size: 16px;
            font-weight: 700;
            text-align: right;
            flex: 0 0 72px;
        }
        .lb-standings-stats {
            font-size: 12px;
            color: var(--text-muted);
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .lb-story-text {
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-top: 6px;
        }
        .lb-badge {
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 999px;
            border: 1px solid var(--border-strong);
            color: var(--text-muted);
            background: var(--pill-bg);
        }
        .lb-badge-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            align-items: center;
        }
        .lb-story-badge {
            font-size: 15px;
            line-height: 1;
            padding: 2px 6px;
            border-radius: 8px;
            background: var(--pill-bg);
            border: 1px solid var(--border);
        }
        .lb-controls {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 16px;
        }
        .lb-table-wrap {
            overflow-x: auto;
            border-radius: 14px;
            border: 1px solid var(--border);
            background: var(--panel);
        }
        .lb-table {
            width: 100%;
            border-collapse: collapse;
            min-width: 720px;
        }
        .lb-table th,
        .lb-table td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
            font-size: 13px;
        }
        .lb-table th {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            background: var(--table-stripe);
            position: sticky;
            top: 0;
        }
        .lb-table tbody tr:hover {
            background: var(--accent-soft);
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
            .lb-standings-name {
                white-space: normal;
                display: -webkit-box;
                -webkit-line-clamp: 2;
                -webkit-box-orient: vertical;
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
    delta_up = "var(--delta-pos)"
    delta_flat = "var(--delta-zero)"
    delta_down = "var(--delta-neg)"

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

    qp_player = (qp_get("player", "") or "").strip()
    qp_pid_raw = (qp_get("pid", "") or "").strip()
    selected_pid_param = None
    if qp_pid_raw.isdigit():
        selected_pid_param = int(qp_pid_raw)

    if qp_player and not st.session_state.get("lb_search"):
        st.session_state["lb_search"] = qp_player

    active_ids = None
    if getattr(ctx, "df_players_active", None) is not None and not ctx.df_players_active.empty:
        if "id" in ctx.df_players_active.columns:
            active_ids = set(ctx.df_players_active["id"].astype(int).tolist())

    target_league = st.session_state.get("lb_league", available_leagues[default_idx])
    if target_league not in available_leagues:
        target_league = available_leagues[default_idx]
        st.session_state["lb_league"] = target_league

    view_mode = st.session_state.get("lb_view_mode", "Story View")

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

    target_league = st.session_state.get("lb_league", "OVERALL")
    if target_league not in available_leagues:
        target_league = available_leagues[default_idx]
        st.session_state["lb_league"] = target_league
    view_mode = st.session_state.get("lb_view_mode", "Story View")

    try:
        st.query_params["page"] = "leaderboards"
        st.query_params["league"] = target_league
        if PUBLIC_MODE:
            st.query_params["public"] = "1"
        else:
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
                if PUBLIC_MODE:
                    display_df = display_df[display_df["is_active"] == True].copy()

            if view_choice == "Active" and active_ids is not None and "player_id" in display_df.columns:
                before = len(display_df)
                display_df = display_df[display_df["player_id"].astype(int).isin(active_ids)].copy()
                inactive_hidden += max(0, before - len(display_df))

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
    for legacy_col in ("story", "story_text", "story_html"):
        if legacy_col in final_view.columns:
            final_view = final_view.drop(columns=[legacy_col])

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

    selected_story_snapshot = None
    if selected_row is not None:
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

        if view_mode == "Story View":
            selected_story_snapshot = selected_row
        else:
            st.markdown("#### Player Snapshot")
            player_url = (
                _player_profile_url(int(selected_pid), PUBLIC_MODE, ctx)
                if selected_pid is not None
                else None
            )
            header_cols = st.columns([3, 1])
            with header_cols[0]:
                st.markdown(f"**{selected_row['name']}** (Rank {int(selected_row['RankNum'])})")
            with header_cols[1]:
                if player_url:
                    try:
                        st.link_button("Open profile", player_url)
                    except Exception:
                        st.markdown(f"[Open profile]({player_url})")
            metric_cols = st.columns(4)
            metric_cols[0].metric("JUPR", f"{float(selected_row['JUPR']):.3f}")
            metric_cols[1].metric(
                "Rating Δ",
                f"{float(selected_row['Gain']):+.3f}" if pd.notna(selected_row["Gain"]) else "—",
            )
            metric_cols[2].metric(
                "Win %",
                _format_win_pct(selected_row["Win %"]),
            )
            metric_cols[3].metric("Games", int(selected_row["matches_played"]))

    standings = final_view.copy()
    if target_league != "OVERALL":
        if "Qualified" not in standings.columns:
            standings["Qualified"] = False
    if "Qualified" in standings.columns:
        standings["Qualified"] = standings["Qualified"].fillna(False).astype(bool)

    for col, default in {
        "wins": 0,
        "losses": 0,
        "matches_played": 0,
        "rating_gain": 0.0,
        "JUPR": 0.0,
        "Win %": pd.NA,
    }.items():
        if col not in standings.columns:
            standings[col] = default

    min_games_for_awards = 0
    if target_league != "OVERALL":
        min_games_for_awards = int(min_games_req or 0)

    if target_league != "OVERALL" and min_games_for_awards > 0:
        qualified_df = standings[standings["matches_played"] >= min_games_for_awards].copy()
        top_perf_title = f"Top Performers (Min {min_games_for_awards} Games)"
        if qualified_df.empty:
            st.markdown(f"### {top_perf_title}")
            st.caption("Not enough games recorded yet for awards.")
        else:
            render_top_performers_cards(
                qualified_df=qualified_df,
                title=top_perf_title,
                public_mode=PUBLIC_MODE,
                ctx=ctx,
            )

    story_debug_rows = []
    if view_mode == "Stats View":
        st.markdown("#### Standings Table")
        table_df = standings.copy()
        table_df["Rank"] = table_df["RankNum"].astype(int)
        table_df["Player"] = table_df["name"].astype(str)
        table_df["Rating Δ"] = table_df["Gain"]
        table_df["Win %"] = pd.to_numeric(table_df["Win %"], errors="coerce")
        table_df["Games"] = table_df["matches_played"].astype(int)
        table_df["Wins"] = table_df["wins"].astype(int)
        table_df["Losses"] = table_df["losses"].astype(int)
        table_df["Rating (JUPR)"] = table_df["JUPR"].astype(float)
        table_df = table_df[
            [
                "_pid",
                "Rank",
                "Player",
                "Rating (JUPR)",
                "Games",
                "Wins",
                "Losses",
                "Win %",
                "Rating Δ",
            ]
        ]
        table_rows = []
        for _, row in table_df.iterrows():
            player_link = _build_player_link(row["_pid"], row["Player"], PUBLIC_MODE, ctx)
            rating_val = row.get("Rating (JUPR)")
            rating_display = f"{float(rating_val):.3f}" if pd.notna(rating_val) else "—"
            win_pct_display = _format_win_pct(row.get("Win %"))
            rating_delta = row.get("Rating Δ")
            rating_delta_display = (
                f"{float(rating_delta):+.3f}" if pd.notna(rating_delta) else "—"
            )
            table_rows.append(
                "<tr>"
                f"<td>{int(row['Rank'])}</td>"
                f"<td>{player_link}</td>"
                f"<td>{html.escape(rating_display)}</td>"
                f"<td>{int(row['Games'])}</td>"
                f"<td>{int(row['Wins'])}</td>"
                f"<td>{int(row['Losses'])}</td>"
                f"<td>{html.escape(win_pct_display)}</td>"
                f"<td>{html.escape(rating_delta_display)}</td>"
                "</tr>"
            )

        table_html = """
        <div class="lb-table-wrap">
            <table class="lb-table">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Player</th>
                        <th>Rating (JUPR)</th>
                        <th>Games</th>
                        <th>Wins</th>
                        <th>Losses</th>
                        <th>Win %</th>
                        <th>Rating Δ</th>
                    </tr>
                </thead>
                <tbody>
        """
        table_html += "".join(table_rows) if table_rows else ""
        table_html += """
                </tbody>
            </table>
        </div>
        """
        st.markdown(table_html, unsafe_allow_html=True)

        st.markdown("#### Standings Analytics")
        chart_data = standings.copy()
        chart_data = chart_data[pd.notna(chart_data["Win %"])].copy()
        chart_data["Player"] = chart_data["name"].astype(str)
        chart_data["Games"] = chart_data["matches_played"].astype(int)
        chart_data["Rating"] = chart_data["JUPR"].astype(float)
        chart_data["WinPct"] = chart_data["Win %"].astype(float)
        chart_data["RatingDelta"] = chart_data["Gain"].astype(float)

        scatter = (
            alt.Chart(chart_data)
            .mark_circle(opacity=0.75)
            .encode(
                x=alt.X("Rating", title="JUPR Rating"),
                y=alt.Y("WinPct", title="Win %"),
                size=alt.Size("Games", title="Games"),
                tooltip=[
                    alt.Tooltip("Player", title="Player"),
                    alt.Tooltip("Rating", title="Rating", format=".3f"),
                    alt.Tooltip("WinPct", title="Win %", format=".1f"),
                    alt.Tooltip("Games", title="Games"),
                    alt.Tooltip("RatingDelta", title="Rating Δ", format="+.3f"),
                ],
            )
            .properties(height=320)
            .interactive()
        )
        st.altair_chart(scatter, use_container_width=True)

        rating_hist = (
            alt.Chart(chart_data)
            .mark_bar(opacity=0.8)
            .encode(
                x=alt.X("Rating", bin=alt.Bin(maxbins=20), title="JUPR Rating"),
                y=alt.Y("count()", title="Players"),
                tooltip=[alt.Tooltip("count()", title="Players")],
            )
            .properties(height=220)
        )
        st.altair_chart(rating_hist, use_container_width=True)

        min_games_chart = int(min_games_req or 0)
        top_candidates = standings.copy()
        top_candidates = top_candidates[top_candidates["matches_played"] >= max(min_games_chart, 1)]
        top_candidates = top_candidates[pd.notna(top_candidates["Win %"])]
        top_candidates = top_candidates.sort_values("Win %", ascending=False).head(10)
        if not top_candidates.empty:
            top_candidates["Player"] = top_candidates["name"].astype(str)
            top_candidates["WinPct"] = top_candidates["Win %"].astype(float)
            win_bar = (
                alt.Chart(top_candidates)
                .mark_bar()
                .encode(
                    x=alt.X("WinPct", title="Win %", axis=alt.Axis(format=".1f")),
                    y=alt.Y("Player", sort="-x", title="Player"),
                    tooltip=[
                        alt.Tooltip("Player", title="Player"),
                        alt.Tooltip("WinPct", title="Win %", format=".1f"),
                        alt.Tooltip("matches_played", title="Games"),
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(win_bar, use_container_width=True)
        else:
            st.caption("Not enough games recorded yet for Top Win %.")
    else:
        st.session_state.setdefault("lb_limit", 50)
        limit = int(st.session_state.get("lb_limit", 50))
        if limit < 50:
            limit = 50
            st.session_state["lb_limit"] = 50

        if admin_logged_in:
            if st.button("Refresh leaderboard cache", key="lb_refresh_cache"):
                try:
                    st.cache_data.clear()
                    st.cache_resource.clear()
                except Exception:
                    logger.exception("Failed to clear leaderboard caches")

        story_badges_by_player = {}
        story_badges_df = pd.DataFrame()
        story_player_ids = []
        show_story_badges = True
        story_player_ids = (
            standings.head(limit)["_pid"].dropna().astype(int).unique().tolist()
        )
        story_badges_df = _fetch_story_badges(ctx, story_player_ids)
        story_badges_by_player = _build_story_badge_map(story_badges_df)
        _verify_story_badges(
            story_badges_by_player,
            story_player_ids,
            story_badges_df,
            admin_logged_in,
        )
        story_rivals_by_player = {}
        story_partners_by_player = {}
        df_matches = getattr(ctx, "df_matches", None)
        if (
            df_matches is not None
            and isinstance(df_matches, pd.DataFrame)
            and not df_matches.empty
            and story_player_ids
        ):
            eligible_ids = []
            if df_players is not None and not df_players.empty and "id" in df_players.columns:
                eligible_ids = df_players["id"].dropna().astype(int).tolist()
            context_filters = {
                "club_id": club_id,
                "league_name": None if target_league == "OVERALL" else target_league,
                "exclude_popups": target_league != "OVERALL",
                "eligible_player_ids": eligible_ids,
            }
            story_rivals_by_player = build_rival_map(
                story_player_ids,
                context_filters,
                df_matches,
            )
            story_partners_by_player = build_best_partner_map(
                story_player_ids,
                context_filters,
                df_matches,
            )
        if admin_logged_in and story_badges_by_player:
            if not st.session_state.get("lb_story_sanity_logged"):
                sanity_story = ""
                for _, row in standings.head(limit).iterrows():
                    pid = row.get("_pid")
                    if pd.notna(pid) and story_badges_by_player.get(int(pid)):
                        sanity_story = build_badge_story(
                            row, story_badges_by_player.get(int(pid), [])
                        )
                        break
                if not sanity_story:
                    logger.warning(
                        "Story View sanity check failed: no story generated for badges."
                    )
                st.session_state["lb_story_sanity_logged"] = True

        if selected_story_snapshot is not None:
            story_badges = []
            snapshot_player_id = selected_story_snapshot.get("_pid")
            if pd.notna(snapshot_player_id):
                try:
                    story_badges = story_badges_by_player.get(int(snapshot_player_id), [])
                except Exception:
                    story_badges = []
            story_text, story_source = _compute_story_text(
                selected_story_snapshot,
                story_badges,
                id_to_name,
                story_rivals_by_player,
                story_partners_by_player,
                admin_logged_in,
            )
            safe_story = html.escape(story_text)
            story_html = f'<div class="lb-story-text">{safe_story}</div>'
            win_pct_display = _format_win_pct(selected_story_snapshot["Win %"])
            gain_value = (
                float(selected_story_snapshot["Gain"])
                if pd.notna(selected_story_snapshot["Gain"])
                else 0.0
            )
            gain_color = _delta_color(gain_value, delta_up, delta_flat, delta_down)
            st.markdown("#### My Snapshot")
            card_html = f"""
                <div class="lb-card">
                    <div class="lb-row" style="align-items:center; justify-content:space-between;">
                        <div>
                            <div class="lb-title">{_build_player_link(selected_story_snapshot['_pid'], selected_story_snapshot['name'], PUBLIC_MODE, ctx)}</div>
                            <div class="lb-subtitle">Rank {int(selected_story_snapshot['RankNum'])}</div>
                        </div>
                        <div style="font-size:20px; font-weight:700;">
                            {float(selected_story_snapshot['JUPR']):.3f}
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
                            <div class="lb-kpi-value">{int(selected_story_snapshot['matches_played'])}</div>
                        </div>
                        <div class="lb-kpi">
                            <div class="lb-kpi-label">W-L</div>
                            <div class="lb-kpi-value">{int(selected_story_snapshot['wins'])}-{int(selected_story_snapshot['losses'])}</div>
                        </div>
                    </div>
                    {story_html}
                </div>
                """
            st.markdown(textwrap.dedent(card_html), unsafe_allow_html=True)
            story_debug_rows.append(
                {
                    "pid": snapshot_player_id,
                    "name": selected_story_snapshot.get("name", ""),
                    "story_source": story_source or "",
                    "story_has_html": _story_has_html(story_text),
                    "story_preview": story_text[:120],
                }
            )

        for _, row in standings.head(limit).iterrows():
            gain_val = float(row["Gain"]) if pd.notna(row["Gain"]) else 0.0
            gain_color = _delta_color(gain_val, delta_up, delta_flat, delta_down)
            status_badges = []
            if target_league != "OVERALL" and not row.get("Qualified", True):
                status_badges.append("Min games not met")
            is_active_value = row.get("is_active")
            if pd.notna(is_active_value) and not bool(is_active_value):
                status_badges.append("Inactive")
            if row.get("matches_played", 0) == 0:
                status_badges.append("New")
            badge_html = "".join(
                f'<span class="lb-badge">{html.escape(b)}</span>' for b in status_badges
            )

            story_badges_html = ""
            story_text_html = ""
            player_id = row.get("_pid")
            story_badges = []
            if pd.notna(player_id):
                try:
                    story_badges = story_badges_by_player.get(int(player_id), [])
                except Exception:
                    story_badges = []
            if story_badges:
                badge_parts = []
                for badge in story_badges[:MAX_STORY_BADGES]:
                    icon = badge_icon(badge.get("badge_id"), badge.get("category"))
                    name = badge.get("name", "Badge")
                    prestige = badge.get("prestige", 0)
                    category = badge.get("category")
                    title_parts = [str(name)]
                    if category:
                        title_parts.append(str(category))
                    if prestige and int(prestige) > 0:
                        title_parts.append(f"Prestige {int(prestige)}")
                    title = " • ".join(title_parts)
                    badge_parts.append(
                        f'<span class="lb-story-badge" title="{_safe_text(title)}">'
                        f"{_safe_text(icon)}</span>"
                    )
                story_badges_html = f'<div class="lb-badge-strip">{"".join(badge_parts)}</div>'
            elif row.get("matches_played", 0) == 0:
                story_badges_html = (
                    '<div class="lb-badge-strip"><span class="lb-badge">New</span></div>'
                )
            story_text, story_source = _compute_story_text(
                row,
                story_badges,
                id_to_name,
                story_rivals_by_player,
                story_partners_by_player,
                admin_logged_in,
            )
            safe_story = html.escape(story_text)
            story_text_html = f'<div class="lb-story-text">{safe_story}</div>'
            card_html = f"""
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
                {story_badges_html}
                {story_text_html}
                <div class="lb-row" style="gap:6px;">{badge_html}</div>
            </div>
            """
            st.markdown(textwrap.dedent(card_html), unsafe_allow_html=True)
            story_debug_rows.append(
                {
                    "pid": player_id,
                    "name": row.get("name", ""),
                    "story_source": story_source or "",
                    "story_has_html": _story_has_html(story_text),
                    "story_preview": story_text[:120],
                }
            )

        if len(standings) > limit:
            if st.button("Load more", key="lb_load_more"):
                st.session_state["lb_limit"] = limit + 50

    if admin_logged_in:
        with st.expander("Story Debug"):
            debug_df = pd.DataFrame(
                story_debug_rows,
                columns=[
                    "pid",
                    "name",
                    "story_source",
                    "story_has_html",
                    "story_preview",
                ],
            )
            st.dataframe(debug_df, use_container_width=True)
        try:
            import subprocess

            sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        except Exception:
            sha = "unknown"
        st.caption(f"Build: {sha}")

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
