from __future__ import annotations

import html

import streamlit as st

from jupr_app.ui.theme_clean import topbar


def _inject_elevation_motion_css() -> None:
    st.markdown(
        """
        <style>
          :root {
            --elev-0: none;
            --elev-1: 0 1px 2px rgba(15, 23, 42, 0.08), 0 1px 1px rgba(15, 23, 42, 0.04);
            --elev-2: 0 6px 18px rgba(15, 23, 42, 0.12), 0 2px 6px rgba(15, 23, 42, 0.08);
            --elev-3: 0 14px 36px rgba(15, 23, 42, 0.18), 0 4px 12px rgba(15, 23, 42, 0.10);

            --motion-fast: 120ms;
            --motion-standard: 180ms;
            --motion-medium: 240ms;
            --motion-easing-standard: cubic-bezier(0.4, 0.0, 0.2, 1);
          }

          html[data-theme="dark"] {
            --elev-1: 0 1px 2px rgba(0, 0, 0, 0.45), 0 1px 1px rgba(0, 0, 0, 0.28);
            --elev-2: 0 8px 20px rgba(0, 0, 0, 0.52), 0 3px 8px rgba(0, 0, 0, 0.36);
            --elev-3: 0 16px 40px rgba(0, 0, 0, 0.62), 0 6px 14px rgba(0, 0, 0, 0.42);
          }

          @media (prefers-color-scheme: dark) {
            :root {
              --elev-1: 0 1px 2px rgba(0, 0, 0, 0.45), 0 1px 1px rgba(0, 0, 0, 0.28);
              --elev-2: 0 8px 20px rgba(0, 0, 0, 0.52), 0 3px 8px rgba(0, 0, 0, 0.36);
              --elev-3: 0 16px 40px rgba(0, 0, 0, 0.62), 0 6px 14px rgba(0, 0, 0, 0.42);
            }
          }

          .elev-1,
          .elev-2,
          .elev-3 {
            transition:
              box-shadow var(--motion-standard) var(--motion-easing-standard),
              transform var(--motion-fast) var(--motion-easing-standard);
            will-change: box-shadow, transform;
          }

          .elev-1 {
            box-shadow: var(--elev-1);
          }

          .elev-2 {
            box-shadow: var(--elev-2);
          }

          .elev-3 {
            box-shadow: var(--elev-3);
          }

          .elev-1:hover,
          .elev-2:hover,
          .elev-3:hover {
            transform: translateY(-1px);
          }

          .elev-1:hover {
            box-shadow: var(--elev-2);
          }

          .elev-2:hover,
          .elev-3:hover {
            box-shadow: var(--elev-3);
          }

          .elev-1:active,
          .elev-2:active,
          .elev-3:active {
            transform: translateY(0);
            box-shadow: var(--elev-1);
            transition-duration: var(--motion-fast);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_shell(
    title: str,
    subtitle: str = "",
    *,
    mode_label: str = "",
    right_html: str = "",
) -> None:
    """
    Render a consistent page chrome wrapper (topbar + spacing).
    """
    _inject_elevation_motion_css()

    resolved_right_html = right_html
    if not resolved_right_html and mode_label:
        resolved_right_html = f'<span class="jupr-pill neutral">{html.escape(mode_label)}</span>'

    topbar(title, subtitle, right_html=resolved_right_html)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
