from __future__ import annotations

import streamlit as st

from jupr_app.ui.theme_tokens import ThemeMode, get_theme_mode, set_theme_mode


def render_theme_toggle(*, key: str = "global_theme_toggle", label: str = "Dark mode") -> ThemeMode:
    current_mode = get_theme_mode()
    is_dark = st.toggle(label, value=(current_mode == "dark"), key=key)
    return set_theme_mode("dark" if is_dark else "light")
