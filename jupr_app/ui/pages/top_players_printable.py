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


def _previous_month_subtitle(now_utc: datetime) -> str:
    first_day_current_month = now_utc.replace(day=1)
    previous_month_date = first_day_current_month - timedelta(days=1)
    return previous_month_date.strftime("%B %Y")


def _build_rating_map(df_players_all: pd.DataFrame) -> dict[int, float]:
    rating_col = "rating" if "rating" in df_players_all.columns else ("elo" if "elo" in df_players_all.columns else None)
    if not rating_col or "id" not in df_players_all.columns:
        return {}
    tmp = df_players_all[["id", rating_col]].copy()
    tmp["id"] = pd.to_numeric(tmp["id"], errors="coerce")
    tmp[rating_col] = pd.to_numeric(tmp[rating_col], errors="coerce")
    tmp = tmp.dropna(subset=["id"]).copy()
    return {int(pid): float(val) for pid, val in zip(tmp["id"].astype(int), tmp[rating_col]) if pd.notna(val)}


def _detect_league_column(df_matches: pd.DataFrame) -> str | None:
    candidates = [
        "league_id",
        "league_key",
        "league_name",
        "league",
        "event_id",
        "session_id",
        "division_id",
    ]
    for col in candidates:
        if col in df_matches.columns:
            return col
    return None


def _build_league_label_map(ctx, league_col: str, values: list[str]) -> dict[str, str]:
    labels = {v: v for v in values}
    metadata_names = {
        "league_id": ["df_leagues", "df_sessions", "df_events"],
        "league_key": ["df_leagues", "df_sessions", "df_events"],
        "league_name": ["df_leagues", "df_sessions", "df_events"],
        "league": ["df_leagues", "df_sessions", "df_events"],
        "event_id": ["df_events", "df_sessions", "df_leagues"],
        "session_id": ["df_sessions", "df_events", "df_leagues"],
        "division_id": ["df_sessions", "df_events", "df_leagues"],
    }
    label_candidates = ["league_label", "league_name", "name", "title", "session_name", "event_name", "division_name"]
    id_candidates = [league_col, "league_id", "league_key", "event_id", "session_id", "division_id", "id"]

    for frame_name in metadata_names.get(league_col, []):
        df_meta = getattr(ctx, frame_name, None)
        if df_meta is None or not isinstance(df_meta, pd.DataFrame) or df_meta.empty:
            continue
        id_col = next((c for c in id_candidates if c in df_meta.columns), None)
        label_col = next((c for c in label_candidates if c in df_meta.columns), None)
        if not id_col or not label_col:
            continue
        tmp = df_meta[[id_col, label_col]].copy()
        tmp[id_col] = tmp[id_col].astype(str)
        tmp[label_col] = tmp[label_col].astype(str).str.strip()
        for _, row in tmp.iterrows():
            key = str(row[id_col])
            if key in labels and row[label_col]:
                labels[key] = row[label_col]
    return labels


def list_leagues(ctx, df_recent: pd.DataFrame) -> tuple[list[dict], str | None]:
    league_col = _detect_league_column(df_recent)
    if not league_col:
        return [{"league_id": "all", "league_label": "All Matches"}], None

    series = df_recent[league_col].fillna("Unknown").astype(str).str.strip()
    values = sorted(v for v in series.unique().tolist() if v)
    if not values:
        values = ["Unknown"]

    label_map = _build_league_label_map(ctx, league_col, values)
    leagues = [{"league_id": value, "league_label": label_map.get(value, value)} for value in values]
    leagues.sort(key=lambda item: item["league_label"].lower())
    return leagues, league_col


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


def _build_page_content(rows: list[dict], title: str, subtitle: str, league_label: str) -> bytes:
    page_w = 792.0
    page_h = 612.0
    m = 36.0
    title_y = page_h - m
    subtitle_y = title_y - 26.0
    league_y = subtitle_y - 22.0
    table_top_y = league_y - 22.0

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

    for i in range(n_rows):
        if (i + 1) % 2 == 0:
            row_top = y_header_bottom - i * row_h
            row_bottom = row_top - row_h
            commands.append("q")
            commands.append("0.95 0.95 0.95 rg")
            commands.append(f"{x0:.2f} {row_bottom:.2f} {(x5 - x0):.2f} {row_h:.2f} re f")
            commands.append("Q")

    commands.append("q")
    commands.append("0.5 w")
    for y in [y_header_top, y_header_bottom] + [y_header_bottom - i * row_h for i in range(1, n_rows + 1)]:
        commands.append(f"{x0:.2f} {y:.2f} m {x5:.2f} {y:.2f} l S")
    for x in (x0, x1, x2, x3, x4, x5):
        commands.append(f"{x:.2f} {y_header_top:.2f} m {x:.2f} {table_bottom:.2f} l S")
    commands.append("Q")

    text: list[str] = ["BT"]
    safe_title = _pdf_escape(title)
    safe_subtitle = _pdf_escape(subtitle)
    safe_league = _pdf_escape(league_label)

    text.extend([f"/F2 22 Tf", f"1 0 0 1 {m:.2f} {title_y:.2f} Tm", f"({safe_title}) Tj"])
    text.extend([f"/F1 18 Tf", f"1 0 0 1 {m:.2f} {subtitle_y:.2f} Tm", f"({safe_subtitle}) Tj"])
    text.extend([f"/F2 14 Tf", f"1 0 0 1 {m:.2f} {league_y:.2f} Tm", f"({safe_league}) Tj"])

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
        msg = "No eligible active players (min 10 games in last 30 days)."
        text.extend([f"1 0 0 1 {x1 + 4.0:.2f} {y_msg:.2f} Tm", f"({_pdf_escape(msg)}) Tj"])

    text.append("ET")
    commands.extend(text)
    return "\n".join(commands).encode("latin-1", "replace")


def _build_top_players_pdf(league_pages: list[dict], title: str, subtitle: str) -> bytes:
    objects: list[bytes] = []
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objects.append(b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n")
    objects.append(b"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
    objects.append(b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>\nendobj\n")

    next_obj = 5
    page_refs: list[int] = []
    for league_page in league_pages:
        content = _build_page_content(league_page.get("rows", []), title, subtitle, league_page.get("league_label", "League"))
        page_obj_num = next_obj
        content_obj_num = next_obj + 1
        next_obj += 2

        page_obj = (
            f"{page_obj_num} 0 obj\n"
            "<< /Type /Page /Parent 2 0 R "
            "/MediaBox [0 0 792 612] "
            "/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> "
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
        page_refs.append(page_obj_num)

    kids_refs = " ".join(f"{ref} 0 R" for ref in page_refs)
    objects[1] = f"2 0 obj\n<< /Type /Pages /Kids [{kids_refs}] /Count {len(page_refs)} >>\nendobj\n".encode("latin-1")

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

    df_recent = df_matches[(df_matches["score_t1"] + df_matches["score_t2"]) > 0].copy()
    df_recent = df_recent[df_recent["date_dt"] >= cutoff].copy()

    if df_recent.empty:
        st.info("No active players found in the last 30 days.")
        return

    leagues, league_col = list_leagues(ctx, df_recent)
    league_label_to_id = {item["league_label"]: item["league_id"] for item in leagues}
    selected_labels = st.multiselect("Leagues", options=list(league_label_to_id.keys()), default=list(league_label_to_id.keys()))

    if not selected_labels:
        st.info("Select at least one league.")
        return

    rating_map = _build_rating_map(df_players_all)

    pages: list[dict] = []
    preview_rows: list[dict] = []
    for label in selected_labels:
        league_id = league_label_to_id[label]
        if league_col:
            league_df = df_recent[df_recent[league_col].fillna("Unknown").astype(str).str.strip() == str(league_id)].copy()
        else:
            league_df = df_recent.copy()

        top_rows = _build_ranked_rows(league_df, id_to_name, rating_map)
        pages.append({"league_label": label, "rows": top_rows})

        if top_rows and not preview_rows:
            preview_rows = top_rows

    if preview_rows:
        preview_df = pd.DataFrame(preview_rows)
        st.dataframe(preview_df[["rank", "name", "jupr_str", "wins", "losses", "wl_str"]], use_container_width=True, hide_index=True)
    else:
        st.info("No eligible active players (min 10 games in last 30 days) for selected leagues.")

    subtitle = _previous_month_subtitle(now_utc)
    pdf_bytes = _build_top_players_pdf(pages, "Tres Palapas -- Top 50 Players", subtitle)

    st.download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name="top_50_active_players_by_league.pdf",
        mime="application/pdf",
    )
