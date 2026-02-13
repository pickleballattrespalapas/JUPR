from __future__ import annotations

from typing import Literal

import streamlit as st

ThemeMode = Literal["light", "dark"]
THEME_SESSION_KEY = "ui_theme_mode"

_THEME_TOKENS: dict[ThemeMode, dict[str, str]] = {
    "light": {
        "bg": "#F8FAFC",
        "card_bg": "#FFFFFF",
        "text_primary": "#0F172A",
        "text_secondary": "#64748B",
        "border_subtle": "#DCE3EE",
    },
    "dark": {
        "bg": "#0B1220",
        "card_bg": "#121A2A",
        "text_primary": "#E5E7EB",
        "text_secondary": "#94A3B8",
        "border_subtle": "#243047",
    },
}


def get_theme_mode(default: ThemeMode = "dark") -> ThemeMode:
    mode = str(st.session_state.get(THEME_SESSION_KEY, default)).lower()
    if mode not in _THEME_TOKENS:
        mode = default
    st.session_state[THEME_SESSION_KEY] = mode
    return mode  # type: ignore[return-value]


def set_theme_mode(mode: str) -> ThemeMode:
    normalized = "dark" if str(mode).lower() == "dark" else "light"
    st.session_state[THEME_SESSION_KEY] = normalized
    return normalized


def get_theme_tokens(mode: ThemeMode | None = None) -> dict[str, str]:
    resolved_mode: ThemeMode = mode or get_theme_mode()
    return _THEME_TOKENS[resolved_mode]
