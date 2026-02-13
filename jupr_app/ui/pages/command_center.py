from __future__ import annotations

import streamlit as st

from jupr_app.ui.theme import MATCH_COLORS


def _inject_command_center_css(theme_mode: str) -> None:
    st.markdown(
        f"""
        <style>
          .cc-root {{
            --cc-bg: var(--bg);
            --cc-panel: var(--panel);
            --cc-text: var(--text-primary);
            --cc-muted: var(--text-muted);
            --cc-border: var(--border);
            --cc-shadow: var(--shadow-lg);
            --cc-accent-soft: var(--accent-soft);
            --cc-accent-border: var(--accent-border);
          }}

          .cc-root[data-theme='dark'] {{
            --cc-bg: #0B1220;
            --cc-panel: #121A2A;
            --cc-text: #E5E7EB;
            --cc-muted: #94A3B8;
            --cc-border: #243047;
            --cc-shadow: 0 16px 36px rgba(0, 0, 0, 0.55);
            --cc-accent-soft: rgba(59, 130, 246, 0.16);
            --cc-accent-border: rgba(59, 130, 246, 0.35);
          }}

          .cc-root[data-theme='light'] {{
            --cc-bg: #F8FAFC;
            --cc-panel: #FFFFFF;
            --cc-text: #0F172A;
            --cc-muted: #64748B;
            --cc-border: #DCE3EE;
            --cc-shadow: 0 14px 32px rgba(15, 23, 42, 0.12);
            --cc-accent-soft: rgba(47, 111, 237, 0.10);
            --cc-accent-border: rgba(47, 111, 237, 0.28);
          }}

          .cc-root .block-container {{
            max-width: 100%;
            padding: 0.75rem 1.2rem 1.5rem;
          }}

          .cc-shell {{
            display: grid;
            grid-template-columns: repeat(12, minmax(0, 1fr));
            gap: 1rem;
            width: 100%;
          }}

          .cc-header,
          .cc-hero {{
            grid-column: 1 / -1;
            border: 1px solid var(--cc-border);
            background: var(--cc-panel);
            box-shadow: var(--cc-shadow);
            border-radius: 16px;
            color: var(--cc-text);
          }}

          .cc-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 1.2rem;
          }}

          .cc-brand {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
          }}

          .cc-logo {{
            width: 36px;
            height: 36px;
            border-radius: 10px;
            display: grid;
            place-items: center;
            border: 1px solid var(--cc-accent-border);
            background: var(--cc-accent-soft);
            font-size: 1.05rem;
          }}

          .cc-brand h1 {{ margin: 0; font-size: 1.05rem; }}
          .cc-brand p {{ margin: 0.1rem 0 0; color: var(--cc-muted); font-size: 0.85rem; }}

          .cc-actions {{
            display: flex;
            align-items: center;
            gap: 0.65rem;
          }}

          .cc-badge {{
            padding: 0.35rem 0.65rem;
            border-radius: 999px;
            border: 1px solid var(--cc-accent-border);
            background: var(--cc-accent-soft);
            font-size: 0.76rem;
            font-weight: 700;
          }}

          .cc-hero {{
            padding: 1.25rem;
          }}
          .cc-hero h2 {{ margin: 0; font-size: 1.35rem; }}
          .cc-hero p {{ margin: 0.45rem 0 0; color: var(--cc-muted); }}
          .cc-pill {{
            display: inline-block;
            margin-top: 0.8rem;
            padding: 0.25rem 0.55rem;
            border-radius: 999px;
            font-size: 0.75rem;
            color: #fff;
            background: {MATCH_COLORS['win']};
          }}
        </style>

        <div class="cc-root" data-theme="{theme_mode}">
          <div class="cc-shell">
            <div class="cc-header">
              <div class="cc-brand">
                <div class="cc-logo">🌵</div>
                <div>
                  <h1>JUPR Club Command Center</h1>
                  <p>Admin operations hub</p>
                </div>
              </div>
              <div class="cc-actions">
                <span class="cc-badge">ADMIN</span>
              </div>
            </div>

            <div class="cc-hero">
              <h2>Record Match Result</h2>
              <p>Placeholder hero card for the upcoming guided result entry workflow.</p>
              <span class="cc-pill">Coming soon</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render(ctx) -> None:
    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.error("Admin login required.")
        st.stop()

    default_theme = st.session_state.get("cc_theme_mode", "dark")
    col_l, col_r = st.columns([0.85, 0.15])
    with col_r:
        is_dark = st.toggle("Dark theme", value=(default_theme == "dark"), key="cc_theme_toggle")
    with col_l:
        st.caption(" ")

    st.session_state["cc_theme_mode"] = "dark" if is_dark else "light"
    _inject_command_center_css(st.session_state["cc_theme_mode"])
