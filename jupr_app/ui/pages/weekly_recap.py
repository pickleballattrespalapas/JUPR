from __future__ import annotations

import streamlit as st
from postgrest.exceptions import APIError

from jupr_app.ui.components.weekly_recap_layout import render_weekly_recap
from jupr_app.ui.layout import page_shell
from jupr_app.ui.public_links import navigate_same_tab
from jupr_app.ui.url import qp_get


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_recap_pdf(recap: dict, week_start: str, week_end: str) -> bytes:
    lines = [
        "JUPR Weekly Recap",
        f"Week: {week_start} to {week_end}",
        "",
        "Highlights",
    ]
    highlights = [str(item).strip() for item in (recap.get("highlights", []) or []) if str(item).strip()]
    if not highlights:
        for item in (recap.get("spotlight", []) or []):
            players = [str(player).strip() for player in (item.get("players", []) or []) if str(player).strip()]
            if players:
                highlights.append(f"{item.get('label', 'Award')}: {', '.join(players)}")

    for item in highlights[:5]:
        text = str(item).strip()
        if text:
            lines.append(f"- {text}")

    lines.append("")
    lines.append("Looking Ahead")
    for item in recap.get("looking_ahead", [])[:5]:
        text = str(item).strip()
        if text:
            lines.append(f"- {text}")

    text_commands = ["BT", "/F1 12 Tf", "72 770 Td", "14 TL"]
    for idx, line in enumerate(lines):
        safe_line = _pdf_escape(line).encode("latin-1", "replace").decode("latin-1")
        text_commands.append(f"({safe_line}) Tj")
        if idx < len(lines) - 1:
            text_commands.append("T*")
    text_commands.append("ET")
    content = "\n".join(text_commands).encode("latin-1")

    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n",
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        f"5 0 obj\n<< /Length {len(content)} >>\nstream\n".encode("latin-1") + content + b"\nendstream\nendobj\n",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj)

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(
        (
            "trailer\n"
            f"<< /Size {len(offsets)} /Root 1 0 R >>\n"
            "startxref\n"
            f"{xref_offset}\n"
            "%%EOF\n"
        ).encode("latin-1")
    )
    return bytes(pdf)


def _get_api_error_code(exc: APIError) -> str | None:
    code = getattr(exc, "code", None)
    if code:
        return code
    if exc.args and isinstance(exc.args[0], dict):
        return exc.args[0].get("code")
    return None


def _handle_missing_table(exc: APIError) -> bool:
    code = _get_api_error_code(exc)
    if code in {"PGRST205", "42P01"}:
        st.error("Weekly recaps table not found. Apply migration migrations/20260207_weekly_recaps.sql in Supabase.")
        return True
    return False


def render(ctx):
    supabase = ctx.supabase
    club_id = ctx.club_id

    mode_label = "Public" if bool(ctx.public_mode) else "Admin"
    page_shell("🗞️ Tres Palapas Weekly Recap", "Club-wide weekly recap.", mode_label=mode_label)

    print_mode = qp_get("print", "0").lower() in ("1", "true", "yes", "y")

    if print_mode:
        st.markdown("<style>header{visibility:hidden;} footer{visibility:hidden;} </style>", unsafe_allow_html=True)

    try:
        response = (
            supabase.table("weekly_recaps")
            .select("week_start,week_end,status,final_json")
            .eq("club_id", club_id)
            .eq("status", "published")
            .order("week_start", desc=True)
            .execute()
        )
    except APIError as exc:
        if _handle_missing_table(exc):
            return
        raise
    published = response.data or []
    if not published:
        st.info("No published recaps yet.")
        return

    selected_row = published[0]
    if not print_mode and len(published) > 1:
        week_options = [row["week_start"] for row in published]
        selected_week = st.selectbox("Select week", options=week_options, format_func=str)
        selected_row = next((row for row in published if row["week_start"] == selected_week), published[0])

    recap = selected_row.get("final_json") or {}
    render_weekly_recap(recap, print_view=print_mode)

    if not print_mode:
        st.caption("Tip: use your browser print dialog for a bulletin-board-ready PDF.")
        if st.button("Open Print-Friendly View"):
            navigate_same_tab(
                page="weekly_recap",
                params={"print": "1"},
                public_mode=True,
                source_label="weekly_recap:print_view",
            )
        pdf_bytes = _build_recap_pdf(
            recap,
            str(selected_row.get("week_start") or ""),
            str(selected_row.get("week_end") or ""),
        )
        st.download_button(
            "Download Weekly Recap PDF",
            data=pdf_bytes,
            file_name=f"weekly_recap_{selected_row.get('week_start') or 'recap'}.pdf",
            mime="application/pdf",
        )

    if (
        (not print_mode)
        and (not bool(ctx.public_mode))
        and supabase
        and club_id
    ):
        try:
            draft_check = (
                supabase.table("weekly_recaps")
                .select("week_start")
                .eq("club_id", club_id)
                .eq("status", "draft")
                .eq("week_start", selected_row["week_start"])
                .execute()
            )
        except APIError as exc:
            if _handle_missing_table(exc):
                return
            raise
        if draft_check.data:
            st.info("Draft exists for this week; public view shows published only.")
