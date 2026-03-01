from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from jupr_app.ui.layout import page_shell


def _safe_int(value) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(value)
    except Exception:
        return None


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _truncate_name(name: str, width: int) -> str:
    text = str(name or "").strip()
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def _approx_text_w(text: str, font_size: float) -> float:
    return len(str(text)) * float(font_size) * 0.52


def _previous_month_window(now_utc: datetime) -> tuple[datetime, datetime]:
    first_of_this_month = now_utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end_dt = first_of_this_month
    start_dt = (first_of_this_month - timedelta(days=1)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return start_dt, end_dt


def _previous_month_subtitle(now_utc: datetime) -> str:
    start_dt, _ = _previous_month_window(now_utc)
    return start_dt.strftime("%B %Y")


def _build_rating_map(df_players_all: pd.DataFrame) -> dict[int, float]:
    rating_col = "rating" if "rating" in df_players_all.columns else ("elo" if "elo" in df_players_all.columns else None)
    if not rating_col or "id" not in df_players_all.columns:
        return {}
    tmp = df_players_all[["id", rating_col]].copy()
    tmp["id"] = pd.to_numeric(tmp["id"], errors="coerce")
    tmp[rating_col] = pd.to_numeric(tmp[rating_col], errors="coerce")
    tmp = tmp.dropna(subset=["id"]).copy()
    return {int(pid): float(val) for pid, val in zip(tmp["id"].astype(int), tmp[rating_col]) if pd.notna(val)}


def _build_ranked_rows(df_matches: pd.DataFrame, id_to_name: dict[int, str], rating_map: dict[int, float]) -> list[dict]:
    expanded = _expand_player_matches(df_matches)
    if expanded.empty:
        return []

    stats = (
        expanded.groupby("player_id", as_index=False)
        .agg(wins=("wins", "sum"), losses=("losses", "sum"), games=("games", "sum"))
    )
    stats = stats[stats["games"] >= 10].copy()
    if stats.empty:
        return []

    rows: list[dict] = []
    for _, item in stats.iterrows():
        pid = int(item["player_id"])
        wins = int(item["wins"])
        losses = int(item["losses"])
        games = int(item["games"])
        rating_elo = float(rating_map.get(pid, 1200.0))
        jupr = rating_elo / 400.0
        rows.append(
            {
                "player_id": pid,
                "name": id_to_name.get(pid, f"#{pid}"),
                "jupr": jupr,
                "wins": wins,
                "losses": losses,
                "games": games,
            }
        )

    top = sorted(rows, key=lambda r: (r["jupr"], r["games"], r["wins"]), reverse=True)[:50]
    for idx, row in enumerate(top, start=1):
        row["rank"] = idx
        row["jupr_str"] = f"{row['jupr']:.3f}"
        row["wl_str"] = f"{row['wins']}-{row['losses']}"
    return top


def _build_page_content(rows: list[dict], title: str, subtitle: str) -> bytes:
    page_w = 792.0
    page_h = 612.0
    m = 36.0
    title_y = page_h - m
    subtitle_y = title_y - 26.0
    table_top_y = subtitle_y - 24.0

    x0 = m
    x1 = x0 + 55.0
    x2 = x0 + 430.0
    x3 = x0 + 530.0
    x4 = x0 + 620.0
    x5 = page_w - m

    header_h = 18.0
    row_h = 10.0
    body_font = 9.5
    n_rows = max(50, len(rows))
    table_h = header_h + n_rows * row_h
    if table_top_y - table_h < m:
        row_h = 9.5
        body_font = 9.0
        table_h = header_h + n_rows * row_h

    y_header_top = table_top_y
    y_header_bottom = y_header_top - header_h
    table_bottom = y_header_bottom - n_rows * row_h

    commands: list[str] = []

    commands.append("q")
    commands.append("0.3 w")
    commands.append(f"{x0:.2f} {y_header_top:.2f} m {x5:.2f} {y_header_top:.2f} l S")
    commands.append(f"{x0:.2f} {y_header_bottom:.2f} m {x5:.2f} {y_header_bottom:.2f} l S")
    commands.append("Q")

    text: list[str] = ["BT"]
    safe_title = _pdf_escape(title)
    safe_subtitle = _pdf_escape(subtitle)
    text.extend([f"/F2 22 Tf", f"1 0 0 1 {m:.2f} {title_y:.2f} Tm", f"({safe_title}) Tj"])
    text.extend([f"/F1 18 Tf", f"1 0 0 1 {m:.2f} {subtitle_y:.2f} Tm", f"({safe_subtitle}) Tj"])

    header_font = 11.0
    header_center_y = y_header_top - (header_h / 2.0) - (header_font / 3.0)
    text.append(f"/F2 {header_font:.1f} Tf")
    headers = [
        ("Rank", x1 - 4.0, "right", x0, x1),
        ("Player", x1 + 4.0, "left", x1, x2),
        ("JUPR", x3 - 4.0, "right", x2, x3),
        ("W-L", x4 - 4.0, "right", x3, x4),
    ]
    for label, anchor, align, _, _ in headers:
        w = _approx_text_w(label, header_font)
        x = anchor - w if align == "right" else anchor
        text.extend([f"1 0 0 1 {x:.2f} {header_center_y:.2f} Tm", f"({_pdf_escape(label)}) Tj"])

    text.append(f"/F1 {body_font:.1f} Tf")
    for idx in range(n_rows):
        if idx >= len(rows):
            continue
        row = rows[idx]
        y_text = y_header_bottom - idx * row_h - (row_h / 2.0) - (body_font / 3.0)

        rank = str(row.get("rank", ""))
        player = _truncate_name(str(row.get("name", "")), 32)
        jupr = str(row.get("jupr_str", f"{float(row.get('jupr', 3.0)):.3f}"))
        wl = str(row.get("wl_str", f"{int(row.get('wins', 0))}-{int(row.get('losses', 0))}"))

        rank_x = (x1 - 4.0) - _approx_text_w(rank, body_font)
        jupr_x = (x3 - 4.0) - _approx_text_w(jupr, body_font)
        wl_x = (x4 - 4.0) - _approx_text_w(wl, body_font)

        text.extend([f"1 0 0 1 {rank_x:.2f} {y_text:.2f} Tm", f"({_pdf_escape(rank)}) Tj"])
        text.extend([f"1 0 0 1 {x1 + 4.0:.2f} {y_text:.2f} Tm", f"({_pdf_escape(player)}) Tj"])
        text.extend([f"1 0 0 1 {jupr_x:.2f} {y_text:.2f} Tm", f"({_pdf_escape(jupr)}) Tj"])
        text.extend([f"1 0 0 1 {wl_x:.2f} {y_text:.2f} Tm", f"({_pdf_escape(wl)}) Tj"])

    if not rows:
        y_msg = y_header_bottom - row_h - (body_font / 3.0)
        msg = "No eligible active players (min 10 games in previous calendar month)."
        text.extend([f"1 0 0 1 {x1 + 4.0:.2f} {y_msg:.2f} Tm", f"({_pdf_escape(msg)}) Tj"])

    text.append("ET")
    commands.extend(text)
    return "\n".join(commands).encode("latin-1", "replace")


def _build_top_players_pdf(rows: list[dict], title: str, subtitle: str) -> bytes:
    objects: list[bytes] = []
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objects.append(b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n")
    objects.append(b"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
    objects.append(b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>\nendobj\n")

    content = _build_page_content(rows, title, subtitle)
    objects.append(
        b"5 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 792 612] /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents 6 0 R >>\nendobj\n"
    )
    objects.append(
        f"6 0 obj\n<< /Length {len(content)} >>\nstream\n".encode("latin-1") + content + b"\nendstream\nendobj\n"
    )
    objects[1] = b"2 0 obj\n<< /Type /Pages /Kids [5 0 R] /Count 1 >>\nendobj\n"

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


def _expand_player_matches(df_matches: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, match in df_matches.iterrows():
        p1 = _safe_int(match.get("t1_p1"))
        p2 = _safe_int(match.get("t1_p2"))
        p3 = _safe_int(match.get("t2_p1"))
        p4 = _safe_int(match.get("t2_p2"))
        s1 = _safe_int(match.get("score_t1")) or 0
        s2 = _safe_int(match.get("score_t2")) or 0

        if any(pid is None for pid in (p1, p2, p3, p4)):
            continue
        if (s1 + s2) <= 0:
            continue

        t1_win = s1 > s2
        t2_win = s2 > s1
        for pid, team in ((p1, 1), (p2, 1), (p3, 2), (p4, 2)):
            win = 1 if (t1_win and team == 1) or (t2_win and team == 2) else 0
            loss = 1 if (t1_win and team == 2) or (t2_win and team == 1) else 0
            rows.append({"player_id": int(pid), "wins": int(win), "losses": int(loss), "games": 1})

    return pd.DataFrame(rows)


def render(ctx):
    mode_label = "Public" if bool(ctx.public_mode) else "Admin"
    page_shell("🧾 Top Active Players PDF", "Printable Top 50 active players (previous calendar month).", mode_label=mode_label)

    if bool(ctx.public_mode) or (not bool(getattr(ctx, "admin_logged_in", False))):
        st.error("Admin login required.")
        return

    now_utc = datetime.now(timezone.utc)
    start_dt, end_dt = _previous_month_window(now_utc)

    df_matches = ctx.df_matches.copy() if ctx.df_matches is not None else pd.DataFrame()
    df_players_all = ctx.df_players_all.copy() if ctx.df_players_all is not None else pd.DataFrame()
    id_to_name = dict(getattr(ctx, "id_to_name", {}) or {})

    if df_matches.empty:
        st.info("No active players found in the previous calendar month.")
        return

    if "date_dt" in df_matches.columns:
        df_matches["date_dt"] = pd.to_datetime(df_matches["date_dt"], errors="coerce", utc=True)
    else:
        df_matches["date_dt"] = pd.to_datetime(df_matches.get("date"), errors="coerce", utc=True)

    df_matches["score_t1"] = pd.to_numeric(df_matches.get("score_t1"), errors="coerce").fillna(0)
    df_matches["score_t2"] = pd.to_numeric(df_matches.get("score_t2"), errors="coerce").fillna(0)

    df_recent = df_matches[(df_matches["score_t1"] + df_matches["score_t2"]) > 0].copy()
    df_recent = df_recent[(df_recent["date_dt"] >= start_dt) & (df_recent["date_dt"] < end_dt)].copy()

    if df_recent.empty:
        st.info("No active players found in the previous calendar month.")
        return

    rating_map = _build_rating_map(df_players_all)
    top_rows = _build_ranked_rows(df_recent, id_to_name, rating_map)

    if top_rows:
        preview_df = pd.DataFrame(top_rows)
        st.dataframe(preview_df[["rank", "name", "jupr_str", "wins", "losses", "wl_str"]], use_container_width=True, hide_index=True)
    else:
        st.info("No eligible active players (min 10 games in previous calendar month).")

    subtitle = _previous_month_subtitle(now_utc)
    pdf_bytes = _build_top_players_pdf(top_rows, "Tres Palapas -- Top 50 Players", subtitle)

    st.download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name="top_50_active_players.pdf",
        mime="application/pdf",
    )
