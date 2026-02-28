import html
import logging
import urllib.parse
from dataclasses import dataclass
import streamlit as st
import pandas as pd

from jupr_app.domain.awards import build_top_performer_entries
from jupr_app.ui.url import qp_get
from jupr_app.ui.public_links import build_public_url, public_link_button
from jupr_app.ui.layout import page_shell
from jupr_app.ui.theme_clean import callout
from jupr_app.ui.pages.players import badge_icon


logger = logging.getLogger(__name__)
MAX_BADGES_PER_PLAYER = 3


@dataclass(frozen=True)
class LeaderboardBadge:
    badge_id: object
    name: str
    prestige: int = 0
    category: str | None = None
    icon_key: str | None = None
    rarity: str | None = None
    earned_at_dt: pd.Timestamp | None = None

def _safe_text(value: object) -> str:
    return html.escape("" if value is None else str(value))


def _format_win_pct(value):
    if pd.notna(value):
        return f"{float(value):.1f}%"
    return "—"


def safe_int(x: object) -> int | None:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        return int(float(x))
    except Exception:
        return None


def safe_float(x: object) -> float | None:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return None


def fmt_pct(x: object) -> str | None:
    val = safe_float(x)
    if val is None:
        return None
    return f"{val:.1f}%"


def fmt_delta(x: object) -> str | None:
    val = safe_float(x)
    if val is None:
        return None
    return f"{val:+.3f}"


def _delta_class(delta_value: float | None) -> str:
    if delta_value is None:
        return "zero"
    if delta_value > 0:
        return "pos"
    if delta_value < 0:
        return "neg"
    return "zero"


def render_leaderboard_row(row, rank: int, badges: list[LeaderboardBadge]) -> None:
    profile_url = row.get("_profile_url")
    player_name = str(row.get("name", "Player") or "Player")
    wins = safe_int(row.get("wins")) or 0
    losses = safe_int(row.get("losses")) or 0
    rating = safe_float(row.get("JUPR"))
    matches_played = safe_int(row.get("matches_played"))
    if matches_played is None:
        matches_played = wins + losses

    win_pct = fmt_pct(row.get("Win %"))
    if win_pct is None and matches_played > 0:
        win_pct = fmt_pct((wins / matches_played) * 100.0)

    delta_value = safe_float(row.get("Gain"))
    delta_text = fmt_delta(delta_value)

    with st.container():
        left_col, right_col = st.columns([0.7, 0.3])
        if profile_url:
            left_col.markdown(
                f"<p class='lb-player-name'><strong>#{rank} <a href='{profile_url}' target='_self'>{_safe_text(player_name)}</a></strong></p>",
                unsafe_allow_html=True,
            )
        else:
            left_col.markdown(
                f"<p class='lb-player-name'><strong>#{rank} {_safe_text(player_name)}</strong></p>",
                unsafe_allow_html=True,
            )
        left_col.markdown(f"<p class='lb-wl'>W-L: {wins}–{losses}</p>", unsafe_allow_html=True)

        rating_display = f"{rating:.3f}" if rating is not None else "—"
        delta_markup = ""
        if delta_text:
            delta_css = _delta_class(delta_value)
            delta_markup = f"<p class='lb-delta'><span class='jupr-delta {delta_css}'>Δ {delta_text}</span></p>"
        right_col.markdown(
            f"<div class='lb-rating-block'><p class='lb-rating'>{rating_display}</p>{delta_markup}</div>",
            unsafe_allow_html=True,
        )

        secondary = []
        if matches_played is not None:
            secondary.append(f"Games: {matches_played}")
        if win_pct:
            secondary.append(f"Win%: {win_pct}")
        if secondary:
            st.markdown(f"<p class='lb-secondary-stats'>{' • '.join(secondary)}</p>", unsafe_allow_html=True)

        ordered_badges = sorted(
            badges,
            key=lambda b: (-int(getattr(b, "prestige", 0) or 0), str(getattr(b, "name", "")).lower()),
        )
        visible_badges = ordered_badges[:MAX_BADGES_PER_PLAYER]
        if visible_badges:
            badge_tokens = []
            for badge in visible_badges:
                icon = badge_icon(badge.badge_id, badge.category)
                label = f"{icon} {badge.name}".strip()
                badge_tokens.append(f"<span class='lb-chip'>{_safe_text(label)}</span>")
            overflow = len(ordered_badges) - len(visible_badges)
            if overflow > 0:
                badge_tokens.append(f"<span class='lb-chip lb-chip-overflow'>+{overflow}</span>")
            st.markdown(
                f"<div class='lb-badges'>{''.join(badge_tokens)}</div>",
                unsafe_allow_html=True,
            )

        st.divider()

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


def _fetch_leaderboard_badges(ctx, player_ids):
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
        merged = pb_copy.merge(badges_df, on="badge_id", how="left")
        for col in ("name", "category", "icon_key", "rarity"):
            if col in merged.columns:
                merged[col] = merged[col].fillna("").astype(str).str.replace(r"<[^>]+>", "", regex=True)
        sort_cols = [col for col in ("player_id", "earned_at", "badge_id") if col in merged.columns]
        if sort_cols:
            merged = merged.sort_values(sort_cols, ascending=[True] * len(sort_cols)).reset_index(drop=True)
        return merged

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
        logger.exception("Failed to fetch leaderboard badges")
        return pd.DataFrame()

    data = resp.data or []
    if not data:
        return pd.DataFrame()

    badges_flat = pd.json_normalize(data, sep=".")
    column_map = {
        "badges.badge_id": "badge_id",
        "badges.name": "name",
        "badges.prestige": "prestige",
        "badges.category": "category",
        "badges.icon_key": "icon_key",
        "badges.rarity": "rarity",
    }
    badges_flat = badges_flat.rename(columns=column_map)
    for col in ("name", "category", "icon_key", "rarity"):
        if col in badges_flat.columns:
            badges_flat[col] = badges_flat[col].fillna("").astype(str).str.replace(r"<[^>]+>", "", regex=True)
    sort_cols = [col for col in ("player_id", "earned_at", "badge_id") if col in badges_flat.columns]
    if sort_cols:
        badges_flat = badges_flat.sort_values(sort_cols, ascending=[True] * len(sort_cols)).reset_index(drop=True)
    return badges_flat


def _build_badge_map(badges_df: pd.DataFrame) -> dict[int, list[LeaderboardBadge]]:
    if badges_df is None or badges_df.empty:
        return {}

    badge_rows = badges_df.copy()
    if "badge_id" not in badge_rows.columns or "player_id" not in badge_rows.columns:
        return {}

    earned_col = "earned_at" if "earned_at" in badge_rows.columns else "created_at"
    badge_rows["earned_at_dt"] = pd.to_datetime(
        badge_rows.get(earned_col, None), utc=True, errors="coerce"
    )
    badge_rows["prestige"] = pd.to_numeric(badge_rows.get("prestige", 0), errors="coerce").fillna(0)

    badge_rows = badge_rows.sort_values(
        ["player_id", "badge_id", "earned_at_dt"], ascending=[True, True, False]
    ).drop_duplicates(subset=["player_id", "badge_id"], keep="first")

    badge_rows = badge_rows.sort_values(
        ["player_id", "prestige", "earned_at_dt", "badge_id"],
        ascending=[True, False, False, True],
    )

    badges_by_player: dict[int, list[LeaderboardBadge]] = {}
    for _, row in badge_rows.iterrows():
        try:
            pid = int(row["player_id"])
        except Exception:
            continue
        name = str(row.get("name", "Badge") or "Badge").strip() or "Badge"
        category = str(row.get("category", "") or "").strip() or None
        icon_key = str(row.get("icon_key", "") or "").strip() or None
        rarity = str(row.get("rarity", "") or "").strip() or None
        badges_by_player.setdefault(pid, []).append(
            LeaderboardBadge(
                badge_id=row.get("badge_id"),
                name=name,
                prestige=int(row.get("prestige", 0) or 0),
                category=category,
                icon_key=icon_key,
                rarity=rarity,
                earned_at_dt=row.get("earned_at_dt"),
            )
        )

    return badges_by_player


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
        .lb-player-name {
            margin: 0 0 1px;
            font-size: 19px;
            line-height: 1.15;
        }
        .lb-player-name a {
            color: inherit;
            text-decoration: none;
        }
        .lb-player-name a:hover {
            text-decoration: underline;
        }
        .lb-wl {
            margin: 0;
            font-size: 27px;
            font-weight: 760;
            line-height: 1.05;
        }
        .lb-rating-block {
            min-height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: flex-end;
            text-align: right;
        }
        .lb-rating {
            margin: 0;
            font-size: 38px;
            font-weight: 800;
            line-height: 1;
            text-align: right;
            letter-spacing: 0.01em;
        }
        .lb-delta {
            margin: 2px 0 0;
            text-align: right;
            font-size: 13px;
            line-height: 1.2;
        }
        .lb-secondary-stats {
            margin: 2px 0 0;
            font-size: 12px;
            color: var(--text-muted);
            line-height: 1.15;
        }
        .lb-badges {
            margin: 3px 0 0;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 5px;
        }
        .lb-chip {
            display: inline-flex;
            align-items: center;
            padding: 1px 8px;
            border-radius: 999px;
            border: 1px solid var(--border-strong);
            background: var(--pill-bg);
            color: var(--text-muted);
            font-size: 11px;
            line-height: 1.25;
            white-space: nowrap;
        }
        .lb-chip-overflow {
            font-weight: 600;
        }
        .stDivider {
            margin: 0.35rem 0 0.5rem;
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

    # -------------------------
    # Controls
    # -------------------------
    st.markdown("### Full Standings")
    control_cols = st.columns([2.5, 1.5])
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
        st.text_input("Find player", key="lb_search")

    target_league = st.session_state.get("lb_league", "OVERALL")
    if target_league not in available_leagues:
        target_league = available_leagues[default_idx]
        st.session_state["lb_league"] = target_league
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

        st.markdown("#### Player Snapshot")
        snap_cols = st.columns(4)
        snap_cols[0].metric("Rating", f"{float(selected_row['JUPR']):.3f}")
        snap_cols[1].metric("W-L", f"{int(selected_row['wins'])}-{int(selected_row['losses'])}")
        snap_cols[2].metric("Games", int(selected_row["matches_played"]))
        snap_cols[3].metric(
            "Win %",
            _format_win_pct(selected_row["Win %"]),
        )

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

    leaderboard_player_ids = standings.head(limit)["_pid"].dropna().astype(int).unique().tolist()
    badges_df = _fetch_leaderboard_badges(ctx, leaderboard_player_ids)
    badges_by_player = _build_badge_map(badges_df)

    for _, row in standings.head(limit).iterrows():
        player_id = row.get("_pid")
        player_badges: list[LeaderboardBadge] = []
        if pd.notna(player_id):
            try:
                player_badges = badges_by_player.get(int(player_id), [])
            except Exception:
                player_badges = []

        row_payload = row.copy()
        row_payload["_profile_url"] = _player_profile_url(row.get("_pid"), PUBLIC_MODE, ctx)
        render_leaderboard_row(
            row_payload,
            rank=safe_int(row.get("RankNum")) or 0,
            badges=player_badges,
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
