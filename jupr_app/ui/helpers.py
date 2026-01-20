import streamlit as st
import urllib.parse


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
