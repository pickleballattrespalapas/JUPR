from __future__ import annotations

MATCH_COLORS = {
    "win": "#1F7A6D",
    "loss": "#5E6F82",
    "draw": "#B9A874",
    "delta_pos": "#2FAE9A",
    "delta_neg": "#7D8A97",
    "delta_zero": "#B9A874",
    "text_light": "#1E2933",
    "text_dark": "#E6EEF3",
    "hover_light": "#F3F7F9",
    "hover_dark": "#1C232A",
    "border": "#D5DEE5",
}


def _normalize_result(result_str: str | None) -> str:
    if result_str is None:
        return ""
    return str(result_str).strip().upper()


def color_for_result(result_str: str | None) -> str | None:
    result = _normalize_result(result_str)
    if result in {"W", "WIN", "WON"}:
        return MATCH_COLORS["win"]
    if result in {"L", "LOSS", "LOST"}:
        return MATCH_COLORS["loss"]
    if result in {"D", "DRAW", "PUSH", "EVEN", "TIE"}:
        return MATCH_COLORS["draw"]
    return None


def color_for_delta(delta_float: float | int | None) -> str | None:
    if delta_float is None:
        return None
    try:
        delta_val = float(delta_float)
    except (TypeError, ValueError):
        return None
    if delta_val > 0:
        return MATCH_COLORS["delta_pos"]
    if delta_val < 0:
        return MATCH_COLORS["delta_neg"]
    return MATCH_COLORS["delta_zero"]


def delta_sign_label(delta_float: float | int | None) -> str:
    if delta_float is None:
        return ""
    try:
        delta_val = float(delta_float)
    except (TypeError, ValueError):
        return ""
    if delta_val > 0:
        return "+"
    if delta_val < 0:
        return "-"
    return ""
