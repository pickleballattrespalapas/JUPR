from __future__ import annotations

from datetime import datetime
from html import escape

import streamlit as st


def render_weekly_recap(recap: dict, *, print_view: bool, title_override: str | None = None) -> None:
    week_start = recap.get("week_start")
    week_end = recap.get("week_end")
    title = title_override or "Tres Palapas Weekly Recap"
    theme = (recap.get("meta", {}) or {}).get("print_theme", "Classic")
    is_baja = theme == "Baja Flyer"

    date_label = ""
    if week_start and week_end:
        try:
            start_dt = datetime.fromisoformat(str(week_start))
            end_dt = datetime.fromisoformat(str(week_end))
            date_label = f"{start_dt.strftime('%b %d')} – {end_dt.strftime('%b %d')}"
        except Exception:
            date_label = f"{week_start} – {week_end}"

    numbers = recap.get("numbers", {})
    spotlight = recap.get("spotlight", [])
    around = recap.get("around_club", {})
    looking_ahead = recap.get("looking_ahead", []) or []

    league_items = around.get("leagues", []) or []
    rr_items = around.get("round_robins", []) or []

    spotlight_items = []
    for item in (spotlight or []):
        if not isinstance(item, dict):
            continue
        label = (item or {}).get("label", "")
        display = (item or {}).get("display", "")
        description = (item or {}).get("description", "") or ""
        if is_baja:
            spotlight_items.append(
                "<li class=\"spotlight-card\">"
                f"<div class=\"spotlight-label\">{label}</div>"
                f"<div class=\"spotlight-desc\">{escape(description)}</div>"
                f"<div class=\"spotlight-winners\">{display}</div>"
                "</li>"
            )
        else:
            desc_html = f"<span class=\"award-desc\">{escape(description)}</span><br/>" if description else ""
            spotlight_items.append(
                "<li>"
                f"<strong>{label}</strong>: {desc_html}"
                f"{display}"
                "</li>"
            )
    spotlight_html = "".join(spotlight_items)
    leagues_html = "".join(
        _render_event_block((item or {}).get("league_name", "League"), (item or {}).get("highlights", []))
        for item in (league_items or [])
        if isinstance(item, dict)
    )
    rr_html = "".join(
        _render_event_block((item or {}).get("event_name", "Pop-Up Event"), (item or {}).get("highlights", []))
        for item in (rr_items or [])
        if isinstance(item, dict)
    )
    looking_ahead_html = "".join(
        f"<li>{item}</li>" for item in (looking_ahead or [])
        if str(item).strip() != ""
    )

    print_mode_notice = "<!-- PRINT MODE -->" if print_view else ""
    print_view_css = ".no-print { display: none !important; }" if print_view else ""
    theme_class = " theme-baja" if is_baja else ""
    header_html = (
        f"""
        <div class=\"hero-bar\"></div>
        <div class=\"weekly-header\">
          <div class=\"weekly-title-group\">
            <div class=\"weekly-title\">{title}</div>
            <div class=\"weekly-subtitle\">Tres Palapas Baja Pickleball Resort • Los Barriles, BCS</div>
          </div>
          <div class=\"weekly-range\">{date_label}</div>
        </div>
        """
        if is_baja
        else f"""
        <div class=\"weekly-header\">
          <div class=\"weekly-title\">{title}</div>
          <div class=\"weekly-range\">{date_label}</div>
        </div>
        """
    )
    baja_theme_css = """
      .theme-baja {
        --baja-sunset: #FF6A3D;
        --baja-ocean: #0EA5A4;
        --baja-sand: #F6E7C6;
        --ink: #111827;
        --muted: #475569;
        --border: #E5E7EB;
        color: var(--ink);
      }
      .theme-baja .hero-bar {
        height: 6px;
        border-radius: 999px;
        margin-bottom: 10px;
        background: linear-gradient(90deg, var(--baja-sunset), var(--baja-ocean));
      }
      .theme-baja .weekly-header {
        border-bottom: 1px solid var(--border);
        padding-bottom: 10px;
        margin-bottom: 14px;
      }
      .theme-baja .weekly-title {
        font-size: 32px;
        font-weight: 800;
        letter-spacing: -0.02em;
      }
      .theme-baja .weekly-title-group {
        display: flex;
        flex-direction: column;
        gap: 4px;
      }
      .theme-baja .weekly-subtitle {
        font-size: 13px;
        color: var(--muted);
        font-weight: 600;
      }
      .theme-baja .weekly-range {
        font-size: 12px;
        font-weight: 700;
        background: var(--baja-sand);
        border: 1px solid var(--baja-ocean);
        padding: 4px 10px;
        border-radius: 999px;
        color: var(--ink);
      }
      .theme-baja .numbers-strip {
        gap: 10px;
      }
      .theme-baja .number-card {
        background: var(--baja-sand);
        border: 1px solid var(--baja-ocean);
        border-radius: 999px;
        padding: 10px 6px;
      }
      .theme-baja .number-label {
        color: var(--muted);
      }
      .theme-baja .section h3 {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding-left: 10px;
        border-left: 4px solid var(--baja-sunset);
        color: var(--ink);
      }
      .theme-baja .spotlight-list {
        list-style: none;
        margin-left: 0;
        padding-left: 0;
      }
      .theme-baja .spotlight-card {
        background: var(--baja-sand);
        border: 1px solid var(--border);
        border-left: 4px solid var(--baja-ocean);
        border-radius: 10px;
        padding: 8px 10px;
        margin-bottom: 6px;
      }
      .theme-baja .spotlight-label {
        font-weight: 700;
      }
      .theme-baja .spotlight-desc {
        color: var(--muted);
        font-size: 12.5px;
      }
      .theme-baja .spotlight-winners {
        margin-top: 4px;
        font-size: 13px;
      }
      .theme-baja .event-name {
        font-variant: small-caps;
        letter-spacing: 0.05em;
        border-bottom: 2px solid var(--baja-ocean);
        display: inline-block;
        padding-bottom: 2px;
      }
      .theme-baja a {
        text-decoration: underline;
      }
      @media print {
        .theme-baja {
          background: #ffffff;
          -webkit-print-color-adjust: exact;
          print-color-adjust: exact;
        }
        .weekly-recap.theme-baja {
          box-shadow: none;
          background: #ffffff;
          border: 1px solid var(--border);
        }
      }
    """
    theme_css = baja_theme_css if is_baja else ""

    html = f"""
    <style>
      {print_view_css}
      .weekly-recap {{
        font-family: 'Inter', sans-serif;
        color: #111827;
        max-width: 900px;
        margin: 0 auto;
        padding: 16px 20px 24px;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        background: #ffffff;
      }}
      .weekly-header {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        border-bottom: 2px solid #111827;
        padding-bottom: 8px;
        margin-bottom: 12px;
      }}
      .weekly-title {{
        font-size: 28px;
        font-weight: 700;
      }}
      .weekly-range {{
        font-size: 14px;
        font-weight: 600;
        color: #374151;
      }}
      .numbers-strip {{
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 8px;
        margin-bottom: 16px;
      }}
      .number-card {{
        background: #f3f4f6;
        padding: 10px 8px;
        border-radius: 10px;
        text-align: center;
      }}
      .number-value {{
        font-size: 20px;
        font-weight: 700;
      }}
      .number-label {{
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #6b7280;
      }}
      .section {{
        margin-top: 14px;
      }}
      .section h3 {{
        font-size: 18px;
        margin: 0 0 6px;
      }}
      .subsection h4 {{
        font-size: 14px;
        margin: 8px 0 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6b7280;
      }}
      .event-block {{
        margin-bottom: 6px;
      }}
      .event-name {{
        font-weight: 600;
        font-size: 13px;
      }}
      ul.compact {{
        margin: 4px 0 6px 18px;
        padding: 0;
      }}
      ul.compact li {{
        margin-bottom: 3px;
        font-size: 13px;
      }}
      .award-desc {{
        color: #374151;
        font-size: 12.5px;
      }}
      .looking-ahead li {{
        font-size: 13px;
      }}
      @media print {{
        body {{
          margin: 0;
        }}
        .weekly-recap {{
          border: none;
          margin: 0;
          padding: 0.4in 0.5in;
          box-shadow: none;
        }}
        .no-print {{
          display: none !important;
        }}
      }}
      {theme_css}
    </style>
    {print_mode_notice}
    <div class=\"weekly-recap{theme_class}\">
      {header_html}
      <div class=\"numbers-strip\">
        <div class=\"number-card\"><div class=\"number-value\">{numbers.get('matches', 0)}</div><div class=\"number-label\">Matches</div></div>
        <div class=\"number-card\"><div class=\"number-value\">{numbers.get('players', 0)}</div><div class=\"number-label\">Players</div></div>
        <div class=\"number-card\"><div class=\"number-value\">{numbers.get('leagues', 0)}</div><div class=\"number-label\">Leagues</div></div>
        <div class=\"number-card\"><div class=\"number-value\">{numbers.get('round_robins', 0)}</div><div class=\"number-label\">Pop-Ups</div></div>
        <div class=\"number-card\"><div class=\"number-value\">{numbers.get('new_faces', 0)}</div><div class=\"number-label\">New Faces</div></div>
      </div>
      <div class=\"section\">
        <h3>Spotlight Reel</h3>
        <ul class=\"compact spotlight-list\">
          {spotlight_html}
        </ul>
      </div>
      <div class=\"section\">
        <h3>Around the Club</h3>
        <div class=\"subsection\">
          <h4>Leagues</h4>
          {leagues_html}
        </div>
        <div class=\"subsection\">
          <h4>Round Robins</h4>
          {rr_html}
        </div>
      </div>
      <div class=\"section\">
        <h3>Looking Ahead</h3>
        <ul class=\"compact looking-ahead\">
          {looking_ahead_html}
        </ul>
      </div>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)


def _render_event_block(name: str, highlights: list[dict]) -> str:
    safe = [h for h in (highlights or []) if isinstance(h, dict)]
    items = "".join(f"<li>{h.get('display','')}</li>" for h in safe if str(h.get("display","")).strip() != "")
    return f"<div class='event-block'><div class='event-name'>{name}</div><ul class='compact'>{items}</ul></div>"
