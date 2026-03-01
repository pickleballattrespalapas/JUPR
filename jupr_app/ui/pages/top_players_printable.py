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
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def _build_top_players_pdf(rows: list[dict], title: str, generated_at_iso: str) -> bytes:
    line_rows = [
        title,
        "Active = match recorded in last 30 days",
        f"Generated: {generated_at_iso}",
        "",
        "Rank  Player                         Rating    W-L",
        "----  ------------------------------  --------  -----",
    ]

    for row in rows:
        rank = int(row.get("rank", 0) or 0)
        name = _truncate_name(str(row.get("name", "")), 30)
        rating = float(row.get("rating", 1200.0) or 1200.0)
        wins = int(row.get("wins", 0) or 0)
        losses = int(row.get("losses", 0) or 0)
        record = f"{wins}-{losses}"
        line_rows.append(f"{rank:>4}  {name:<30}  {rating:>8.2f}  {record:>5}")

    lines_per_page = 52
    pages: list[list[str]] = [line_rows[i : i + lines_per_page] for i in range(0, len(line_rows), lines_per_page)]
    if not pages:
        pages = [[title, "No rows available"]]

    objects: list[bytes] = []
    page_obj_nums: list[int] = []

    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")

    objects.append(b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n")

    font_obj_num = 3
    objects.append(b"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>\nendobj\n")

    next_obj_num = 4
    for page_lines in pages:
        text_commands = ["BT", "/F1 10 Tf", "54 760 Td", "12 TL"]
        for idx, line in enumerate(page_lines):
            safe_line = _pdf_escape(str(line)).encode("latin-1", "replace").decode("latin-1")
            text_commands.append(f"({safe_line}) Tj")
            if idx < len(page_lines) - 1:
                text_commands.append("T*")
        text_commands.append("ET")
        content = "\n".join(text_commands).encode("latin-1")

        page_obj_num = next_obj_num
        content_obj_num = next_obj_num + 1
        next_obj_num += 2

        page_obj = (
            f"{page_obj_num} 0 obj\n"
            "<< /Type /Page /Parent 2 0 R "
            "/MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_obj_num} 0 R >> >> "
            f"/Contents {content_obj_num} 0 R >>\n"
            "endobj\n"
        ).encode("latin-1")
        content_obj = (
            f"{content_obj_num} 0 obj\n<< /Length {len(content)} >>\nstream\n".encode("latin-1")
            + content
            + b"\nendstream\nendobj\n"
        )

        objects.append(page_obj)
        objects.append(content_obj)
        page_obj_nums.append(page_obj_num)

    kids_refs = " ".join(f"{num} 0 R" for num in page_obj_nums)
    objects[1] = f"2 0 obj\n<< /Type /Pages /Kids [{kids_refs}] /Count {len(page_obj_nums)} >>\nendobj\n".encode("latin-1")

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
    page_shell("🧾 Top Active Players PDF", "Printable Top 50 active players (last 30 days).", mode_label=mode_label)

    if bool(ctx.public_mode) or (not bool(getattr(ctx, "admin_logged_in", False))):
        st.error("Admin login required.")
        return

    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(days=30)

    df_matches = ctx.df_matches.copy() if ctx.df_matches is not None else pd.DataFrame()
    df_players_all = ctx.df_players_all.copy() if ctx.df_players_all is not None else pd.DataFrame()
    id_to_name = dict(getattr(ctx, "id_to_name", {}) or {})

    if df_matches.empty:
        st.info("No active players found in the last 30 days.")
        return

    if "date_dt" in df_matches.columns:
        df_matches["date_dt"] = pd.to_datetime(df_matches["date_dt"], errors="coerce", utc=True)
    else:
        df_matches["date_dt"] = pd.to_datetime(df_matches.get("date"), errors="coerce", utc=True)

    df_matches["score_t1"] = pd.to_numeric(df_matches.get("score_t1"), errors="coerce").fillna(0)
    df_matches["score_t2"] = pd.to_numeric(df_matches.get("score_t2"), errors="coerce").fillna(0)

    df_matches = df_matches[(df_matches["score_t1"] + df_matches["score_t2"]) > 0].copy()
    df_matches = df_matches[df_matches["date_dt"] >= cutoff].copy()

    if df_matches.empty:
        st.info("No active players found in the last 30 days.")
        return

    expanded = _expand_player_matches(df_matches)
    if expanded.empty:
        st.info("No active players found in the last 30 days.")
        return

    stats = (
        expanded.groupby("player_id", as_index=False)
        .agg(wins=("wins", "sum"), losses=("losses", "sum"), games=("games", "sum"))
    )

    rating_col = "rating" if "rating" in df_players_all.columns else ("elo" if "elo" in df_players_all.columns else None)
    rating_map: dict[int, float] = {}
    if rating_col and "id" in df_players_all.columns:
        tmp = df_players_all[["id", rating_col]].copy()
        tmp["id"] = pd.to_numeric(tmp["id"], errors="coerce")
        tmp[rating_col] = pd.to_numeric(tmp[rating_col], errors="coerce")
        tmp = tmp.dropna(subset=["id"]).copy()
        rating_map = {int(pid): float(val) for pid, val in zip(tmp["id"].astype(int), tmp[rating_col]) if pd.notna(val)}

    rows: list[dict] = []
    for _, item in stats.iterrows():
        pid = int(item["player_id"])
        wins = int(item["wins"])
        losses = int(item["losses"])
        games = int(item["games"])
        rating = float(rating_map.get(pid, 1200.0))
        rows.append(
            {
                "player_id": pid,
                "name": id_to_name.get(pid, f"#{pid}"),
                "rating": rating,
                "wins": wins,
                "losses": losses,
                "games": games,
            }
        )

    top = sorted(rows, key=lambda r: (r["rating"], r["games"], r["wins"]), reverse=True)[:50]

    for idx, row in enumerate(top, start=1):
        row["rank"] = idx
        row["record_str"] = f"{row['wins']}-{row['losses']}"

    top_df = pd.DataFrame(top)
    if top_df.empty:
        st.info("No active players found in the last 30 days.")
        return

    st.dataframe(
        top_df[["rank", "name", "rating", "wins", "losses", "record_str"]],
        use_container_width=True,
        hide_index=True,
    )

    generated_at_iso = now_utc.replace(microsecond=0).isoformat()
    pdf_bytes = _build_top_players_pdf(top, "Top 50 Active Players", generated_at_iso)

    st.download_button(
        "Download Top 50 Active Players PDF",
        data=pdf_bytes,
        file_name="top_50_active_players.pdf",
        mime="application/pdf",
    )
