from __future__ import annotations

import urllib.parse

import pandas as pd
import streamlit as st


def qp_get(key: str, default: str = "") -> str:
    """Streamlit query params can be str or list depending on version."""
    try:
        v = st.query_params.get(key, default)
    except Exception:
        return default
    if isinstance(v, list):
        return v[0] if v else default
    return str(v) if v is not None else default


def _public_base_url() -> str:
    """
    Base URL for share links.

    Priority:
      1) st.secrets["PUBLIC_BASE_URL"]
      2) st.get_url() (Streamlit newer versions)
      3) localhost (dev fallback)
    """
    # 1) Prefer secrets
    try:
        base = str(st.secrets.get("PUBLIC_BASE_URL", "") or "").strip().rstrip("/")
        if base:
            return base
    except Exception:
        pass

    # 2) Fallback: infer from current URL (Streamlit ≥ 1.27)
    try:
        u = st.get_url()
        if u:
            return u.split("?", 1)[0].rstrip("/")
    except Exception:
        pass

    # 3) Last resort: localhost (useful in dev)
    return "http://localhost:8501"


def build_standings_link(league_name: str, public: bool = True) -> str:
    """Shareable URL for public leaderboards, pre-selected to a league."""
    base = _public_base_url()
    params = {"page": "leaderboards", "league": str(league_name)}
    if public:
        params["public"] = "1"
    q = urllib.parse.urlencode(params, quote_via=urllib.parse.quote_plus)
    return f"{base}/?{q}"


def build_player_profile_link(player_id: int, public: bool = False) -> str:
    """Deep link to Player Search page with a player preselected."""
    base = _public_base_url()
    params = {"page": "players", "pid": str(int(player_id))}
    if public:
        params["public"] = "1"
    q = urllib.parse.urlencode(params, quote_via=urllib.parse.quote_plus)
    return f"{base}/?{q}"


def build_match_explorer_link(
    ctx: str,
    me: int,
    partner: int,
    opp1: int,
    opp2: int,
    sy: int,
    so: int,
    public: bool = False,
) -> str:
    """
    Deep link to Match Explorer prefilled for a specific perspective.
    Uses numeric IDs to avoid name encoding issues.
    """
    base = _public_base_url()
    params = {
        "page": "match_explorer",
        "ctx": str(ctx),
        "me": str(int(me)),
        "partner": str(int(partner)),
        "opp1": str(int(opp1)),
        "opp2": str(int(opp2)),
        "sy": str(int(sy)),
        "so": str(int(so)),
    }
    if public:
        params["public"] = "1"
    q = urllib.parse.urlencode(params, quote_via=urllib.parse.quote_plus)
    return f"{base}/?{q}"


def build_badge_story(player_row: dict | pd.Series, earned_badges: list[dict]) -> str:
    name = _coerce_text(_row_value(player_row, ["name", "player", "player_name"]), "Player")
    games = _coerce_int(_row_value(player_row, ["games", "matches_played", "matches", "games_played"]), 0)
    win_ratio = _normalize_win_pct(_row_value(player_row, ["win_pct", "Win %", "win_pct_pct"]))
    delta = _coerce_float(_row_value(player_row, ["delta", "Gain", "rating_gain", "rating_delta"]))

    badges = _dedupe_badges(earned_badges or [])
    headline_badges = _select_headline_badges(badges)

    if not headline_badges:
        if games <= 0:
            return "New to the standings—play your first matches to begin earning badges."
        if games <= 5:
            return "New to the leaderboard—log a few matches to start earning badges."
        return (
            f"Active this season with {games} games logged—"
            "badges will start appearing as the reel fills."
        )

    badge_sentence = _build_badge_sentence(name, headline_badges)
    stats_sentence = _build_stats_sentence(games, win_ratio, delta)
    category_hook = _category_hook(headline_badges[0].get("category"))
    if category_hook:
        stats_sentence = f"{category_hook} {stats_sentence[0].lower()}{stats_sentence[1:]}"

    story = f"{badge_sentence} {stats_sentence}".strip()
    if len(story) > 180:
        story = badge_sentence
    if len(story) > 180 and len(headline_badges) > 1:
        short_badge_sentence = _build_badge_sentence(name, headline_badges[:1])
        story = short_badge_sentence
    if not story:
        if games <= 0:
            return "New to the standings—play your first matches to begin earning badges."
        games_str = f"{games} game" + ("s" if games != 1 else "")
        return f"Active this season with {games_str} logged."
    return story


def _row_value(row: dict | pd.Series, keys: list[str]):
    if row is None:
        return None
    for key in keys:
        if hasattr(row, "get"):
            value = row.get(key)
        else:
            try:
                value = row[key]
            except Exception:
                value = None
        if not _is_missing(value):
            return value
    return None


def _is_missing(value) -> bool:
    try:
        return value is None or pd.isna(value)
    except Exception:
        return value is None


def _coerce_float(value) -> float | None:
    if _is_missing(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _coerce_int(value, default: int = 0) -> int:
    if _is_missing(value):
        return default
    try:
        return int(value)
    except Exception:
        return default


def _coerce_text(value, default: str) -> str:
    if _is_missing(value):
        return default
    text = str(value).strip()
    return text or default


def _normalize_win_pct(value) -> float | None:
    val = _coerce_float(value)
    if val is None:
        return None
    if val > 1.0:
        return val / 100.0
    return val


def _dedupe_badges(badges: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for badge in badges:
        badge_id = badge.get("badge_id") or badge.get("id") or badge.get("code") or badge.get("name")
        key = str(badge_id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(badge)
    return unique


def _badge_sort_key(badge: dict) -> tuple:
    prestige = _coerce_int(badge.get("prestige"), 0)
    earned_at = badge.get("earned_at_dt") or badge.get("earned_at")
    earned_dt = pd.to_datetime(earned_at, utc=True, errors="coerce")
    earned_ts = earned_dt.timestamp() if pd.notna(earned_dt) else 0.0
    return (-prestige, -earned_ts)


def _select_headline_badges(badges: list[dict]) -> list[dict]:
    if not badges:
        return []
    ordered = sorted(badges, key=_badge_sort_key)
    return ordered[:2]


def _build_badge_sentence(name: str, badges: list[dict]) -> str:
    badge_names = [str(b.get("name", "Badge")) for b in badges if b.get("name")]
    if not badge_names:
        return f"{name} has earned a badge this season."
    if len(badge_names) == 1:
        return f"{name} has earned {badge_names[0]} this season."
    return f"{name} has earned {badge_names[0]} and {badge_names[1]} this season."


def _build_stats_sentence(games: int, win_ratio: float | None, delta: float | None) -> str:
    if games <= 0:
        return "No games logged yet."

    games_str = f"{games} game" + ("s" if games != 1 else "")
    win_pct = f"{win_ratio * 100:.0f}%" if win_ratio is not None else None

    if delta is not None and win_pct is not None:
        if delta >= 0.10:
            return (
                f"They're climbing with a {delta:+.3f} rating change and a {win_pct} "
                f"win rate over {games_str}."
            )
        if delta <= -0.10:
            return f"Despite a small dip lately, they've logged {games_str} with a {win_pct} win rate."
        return (
            f"Steady with a {delta:+.3f} rating change and a {win_pct} win rate over {games_str}."
        )

    if delta is not None:
        if delta >= 0.10:
            return f"They're climbing with a {delta:+.3f} rating change over {games_str}."
        if delta <= -0.10:
            return (
                f"Despite a small dip lately, they've logged {games_str} with a "
                f"{delta:+.3f} rating change."
            )
        return f"Steady with a {delta:+.3f} rating change over {games_str}."

    if win_pct is not None:
        if win_ratio is not None and win_ratio >= 0.70:
            form = "Hot form"
        elif win_ratio is not None and win_ratio >= 0.55:
            form = "Solid form"
        else:
            form = "Grinding"
        return f"{form} with a {win_pct} win rate over {games_str}."

    return f"Active with {games_str} logged."


def _category_hook(category: str | None) -> str:
    if not category:
        return ""
    category = str(category).strip()
    hooks = {
        "Prestige / Rarity": "A marquee badge highlight,",
        "Momentum & Progress": "Momentum is building,",
        "Dominance & Quality": "Strong results back it up,",
        "Participation": "Plenty of activity to build on,",
        "Performance vs Expectation": "A standout result to build on,",
    }
    return hooks.get(category, "")
