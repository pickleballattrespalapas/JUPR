from __future__ import annotations

from datetime import datetime

import streamlit as st

ACCENT_CLASS_BY_KEY = {
    "TOP_PERFORMER_WEEK": "baja-top",
    "BIGGEST_JUMP_WEEK": "baja-jump",
    "GIANT_SLAYER_WEEK": "baja-slayer",
    "GRIND_WEEK": "baja-grind",
    "PERFECT_RUN": "baja-perfect",
}


def _inject_baja_styles(print_view: bool) -> None:
    print_view_css = ".no-print { display: none !important; }" if print_view else ""
    st.markdown(
        """
<style>
  """ + print_view_css + """
  .weekly-recap {
    font-family: 'Inter', sans-serif;
    color: black;
    max-width: 900px;
    margin: 0 auto;
    padding: 16px 20px 24px;
    border: 1px solid gainsboro;
    border-radius: 12px;
    background: white;
  }
  .weekly-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    border-bottom: 2px solid black;
    padding-bottom: 8px;
    margin-bottom: 12px;
  }
  .weekly-title { font-size: 28px; font-weight: 700; }
  .weekly-range { font-size: 14px; font-weight: 600; color: dimgray; }
  .numbers-strip {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
    margin-bottom: 16px;
  }
  .number-card { background: whitesmoke; padding: 10px 8px; border-radius: 10px; text-align: center; }
  .number-value { font-size: 20px; font-weight: 700; }
  .number-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: gray; }
  .section { margin-top: 14px; }
  .baja-card {
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 8px 22px rgba(0,0,0,0.08);
  }

  .baja-top {
    background: linear-gradient(135deg, #FFF3E6, #FFE0C2);
    border-left: 6px solid #FF7A00;
  }

  .baja-jump {
    background: linear-gradient(135deg, #E6FAF7, #C8F3EE);
    border-left: 6px solid #00B3A4;
  }

  .baja-slayer {
    background: linear-gradient(135deg, #F3E8FF, #E3D4FF);
    border-left: 6px solid #7E57C2;
  }

  .baja-grind {
    background: linear-gradient(135deg, #F1F1F1, #E2E2E2);
    border-left: 6px solid #444444;
  }

  .baja-perfect {
    background: linear-gradient(135deg, #FFF0F5, #FFD6E8);
    border-left: 6px solid #D63384;
  }

  .baja-league {
    background: linear-gradient(135deg, #EAF4FF, #D4E8FF);
    border-left: 6px solid #1976D2;
  }

  .baja-roundrobin {
    background: linear-gradient(135deg, #E8FFF4, #D2F7E7);
    border-left: 6px solid #1B9E77;
  }

  .baja-pop {
    background: linear-gradient(135deg, #FFFBE6, #FFF1B8);
    border-left: 6px solid #C48F00;
  }

  .baja-tournament {
    background: linear-gradient(135deg, #FFF4E6, #FFE8CC);
    border-left: 6px solid #E67700;
  }

  .baja-podium {
    background: linear-gradient(135deg, #FFE7A3, #FFD166);
    border: 2px solid #E0A100;
    border-radius: 14px;
    padding: 28px;
    margin-bottom: 24px;
    box-shadow: 0 10px 26px rgba(0,0,0,0.1);
  }

  .podium-grid {
    display: flex;
    justify-content: center;
    align-items: flex-end;
    gap: 20px;
    margin-top: 15px;
  }

  .podium-col {
    text-align: center;
  }

  .podium-1 {
    font-size: 1.25rem;
    font-weight: 700;
    color: #7A5600;
  }

  .podium-2 {
    font-size: 1rem;
    font-weight: 600;
    color: #5D5D5D;
  }

  .podium-3 {
    font-size: 1rem;
    font-weight: 600;
    color: #78512A;
  }

  .podium-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    opacity: 0.7;
    margin-bottom: 4px;
  }

  .section-title {
    font-weight: 700;
    font-size: 1.15rem;
    margin-bottom: 10px;
  }

  .section-subtitle {
    font-weight: 600;
    margin-bottom: 6px;
  }

  .baja-title { font-weight: 700; font-size: 1.1rem; margin-bottom: 6px; }
  .baja-desc { font-size: 0.9rem; color: dimgrey; margin-bottom: 8px; }
  .baja-player { margin-left: 10px; }
  @media print {
    body { margin: 0; }
    .weekly-recap { border: none; margin: 0; padding: 0.4in 0.5in; box-shadow: none; }
    .no-print { display: none !important; }
  }
</style>
""",
        unsafe_allow_html=True,
    )


def render_award_card(title: str, players: list[str], description: str, accent_class: str) -> None:
    player_lines = "".join(f"<div class='baja-player'>• {player}</div>" for player in (players or []))
    st.markdown(
        f"""
<div class="baja-card {accent_class}">
  <div class="baja-title">{title}</div>
  <div class="baja-desc">{description}</div>
  {player_lines}
</div>
""",
        unsafe_allow_html=True,
    )


def render_section_card(title: str, rows: list[str], css_class: str) -> None:
    content = "".join(f"<div>• {row}</div>" for row in rows)
    st.markdown(
        f"""
<div class="baja-card {css_class}">
  <div class="section-title">{title}</div>
  {content}
</div>
""",
        unsafe_allow_html=True,
    )


def render_podium_layout(podium_rows: list[dict]) -> str:
    podium_by_place = {
        int(item.get("placement", 0) or 0): str(item.get("display_name", "") or "")
        for item in podium_rows
    }
    first = podium_by_place.get(1, "")
    second = podium_by_place.get(2, "")
    third = podium_by_place.get(3, "")

    return f"""
<div class="podium-grid">
  <div class="podium-col">
    <div class="podium-label">2nd</div>
    <div class="podium-2">🥈 {second}</div>
  </div>
  <div class="podium-col">
    <div class="podium-label">1st</div>
    <div class="podium-1">🥇 {first}</div>
  </div>
  <div class="podium-col">
    <div class="podium-label">3rd</div>
    <div class="podium-3">🥉 {third}</div>
  </div>
</div>
"""


def render_tournament_podium(tournament: dict) -> None:
    title = str(tournament.get("tournament_name") or "Tournament")
    podium_rows = tournament.get("podium", []) or []

    st.markdown(
        f"""
<div class="baja-podium">
  <div class="section-title">🏆 {title}</div>
  {render_podium_layout(podium_rows)}
</div>
""",
        unsafe_allow_html=True,
    )


def render_weekly_recap(recap: dict, *, print_view: bool, title_override: str | None = None) -> None:
    week_start = recap.get("week_start") or recap.get("start_date")
    week_end = recap.get("week_end") or recap.get("end_date")
    title = title_override or "Tres Palapas Weekly Recap"

    date_label = ""
    if week_start and week_end:
        try:
            start_dt = datetime.fromisoformat(str(week_start))
            end_dt = datetime.fromisoformat(str(week_end))
            date_label = f"{start_dt.strftime('%b %d')} – {end_dt.strftime('%b %d')}"
        except Exception:
            date_label = f"{week_start} – {week_end}"

    numbers = recap.get("numbers", {})
    spotlight = recap.get("spotlight", []) or []
    around = recap.get("around_club", {})
    tournaments = recap.get("tournaments", []) or []
    looking_ahead = [str(item).strip() for item in (recap.get("looking_ahead", []) or []) if str(item).strip()]

    _inject_baja_styles(print_view)

    st.markdown(
        f"""
<div class="weekly-recap">
  <div class="weekly-header">
    <div class="weekly-title">{title}</div>
    <div class="weekly-range">{date_label}</div>
  </div>
  <div class="numbers-strip">
    <div class="number-card"><div class="number-value">{numbers.get('matches', 0)}</div><div class="number-label">Matches</div></div>
    <div class="number-card"><div class="number-value">{numbers.get('players', 0)}</div><div class="number-label">Players</div></div>
    <div class="number-card"><div class="number-value">{numbers.get('leagues', 0)}</div><div class="number-label">Leagues</div></div>
    <div class="number-card"><div class="number-value">{numbers.get('round_robins', 0)}</div><div class="number-label">Pop-Ups</div></div>
    <div class="number-card"><div class="number-value">{numbers.get('new_faces', 0)}</div><div class="number-label">New Faces</div></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("### Spotlight Reel")
    col1, col2 = st.columns(2)
    for idx, item in enumerate(sorted(spotlight, key=lambda award: int(award.get("order", 999)))):
        if not item.get("include", True):
            continue
        target_col = col1 if idx % 2 == 0 else col2
        with target_col:
            render_award_card(
                item.get("label", "Award"),
                item.get("players", []) or [],
                item.get("description", ""),
                ACCENT_CLASS_BY_KEY.get(item.get("key"), "baja-top"),
            )

    around_sections: list[tuple[str, list[str], str]] = []

    for item in around.get("leagues", []) or []:
        title_text = str(item.get("league_name", "League"))
        rows = [
            str((highlight or {}).get("display", "")).strip()
            for highlight in item.get("highlights", []) or []
            if str((highlight or {}).get("display", "")).strip()
        ]
        if not rows:
            continue
        if "pop" in title_text.lower():
            around_sections.append((title_text, rows, "baja-pop"))
        else:
            around_sections.append((title_text, rows, "baja-league"))

    for item in around.get("round_robins", []) or []:
        title_text = str(item.get("event_name", "Pop-Up Event"))
        rows = [
            str((highlight or {}).get("display", "")).strip()
            for highlight in item.get("highlights", []) or []
            if str((highlight or {}).get("display", "")).strip()
        ]
        if not rows:
            continue
        if "pop" in title_text.lower():
            around_sections.append((title_text, rows, "baja-pop"))
        else:
            around_sections.append((title_text, rows, "baja-roundrobin"))

    st.markdown("### Around the Club")
    around_col1, around_col2 = st.columns(2)
    for idx, (title_text, rows, css_class) in enumerate(around_sections):
        target_col = around_col1 if idx % 2 == 0 else around_col2
        with target_col:
            render_section_card(title_text, rows, css_class)

    if tournaments:
        st.markdown("### Tournaments")
        for tournament in tournaments:
            render_tournament_podium(tournament)

    if looking_ahead:
        st.markdown("### Looking Ahead")
        for item in looking_ahead:
            st.markdown(f"• {item}")
