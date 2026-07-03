from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

WEEKLY_RECAP_SELECT = "club_id,week_start,week_end,status,final_json"
PUBLIC_RECAP_KEYS = {
    "week_start",
    "week_end",
    "start_date",
    "end_date",
    "numbers",
    "numbers_cards",
    "spotlight",
    "around_club",
    "tournaments",
    "looking_ahead",
    "highlights",
}


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _clean_text(value: Any) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()


def _clean_text_list(values: Any, *, limit: int = 12) -> list[str]:
    if not isinstance(values, list):
        return []
    return [_clean_text(item) for item in values[:limit] if _clean_text(item)]


def _sanitize_spotlight(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    result: list[dict[str, Any]] = []
    for item in items[:20]:
        if not isinstance(item, dict):
            continue
        result.append(
            {
                "key": _clean_text(item.get("key")),
                "label": _clean_text(item.get("label") or "Award"),
                "players": _clean_text_list(item.get("players"), limit=5),
                "description": _clean_text(item.get("description")),
                "order": int(item.get("order") or 999),
                "include": bool(item.get("include", True)),
            }
        )
    return result


def _sanitize_highlight_rows(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in items[:10]:
        if isinstance(item, dict):
            rows.append({"display": _clean_text(item.get("display"))})
        else:
            rows.append({"display": _clean_text(item)})
    return [row for row in rows if row.get("display")]


def _sanitize_around_club(around: Any) -> dict[str, Any]:
    if not isinstance(around, dict):
        return {"leagues": [], "round_robins": [], "community_events": []}

    leagues = []
    for item in (around.get("leagues") or [])[:12]:
        if not isinstance(item, dict):
            continue
        leagues.append(
            {
                "league_name": _clean_text(item.get("league_name") or "League"),
                "highlights": _sanitize_highlight_rows(item.get("highlights")),
            }
        )

    round_robins = []
    for item in (around.get("round_robins") or [])[:12]:
        if not isinstance(item, dict):
            continue
        round_robins.append(
            {
                "event_name": _clean_text(item.get("event_name") or "Pop-Up Event"),
                "highlights": _sanitize_highlight_rows(item.get("highlights")),
            }
        )

    community_events = around.get("community_events")
    if community_events is None:
        community_events = around.get("social_round_robins")
    public_events = []
    for item in (community_events or [])[:12]:
        if not isinstance(item, dict):
            continue
        public_events.append(
            {
                "event_name": _clean_text(item.get("event_name") or "Community Event"),
                "event_type_label": _clean_text(item.get("event_type_label") or "Community Event"),
                "skill_level": _clean_text(item.get("skill_level")),
                "highlights": _sanitize_highlight_rows(item.get("highlights")),
            }
        )

    return {"leagues": leagues, "round_robins": round_robins, "community_events": public_events}


def _sanitize_tournaments(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    tournaments: list[dict[str, Any]] = []
    for item in items[:20]:
        if not isinstance(item, dict):
            continue
        podium = []
        for podium_row in (item.get("podium") or [])[:3]:
            if not isinstance(podium_row, dict):
                continue
            podium.append(
                {
                    "placement": int(podium_row.get("placement") or 0),
                    "display_name": _clean_text(podium_row.get("display_name") or podium_row.get("name")),
                }
            )
        tournaments.append({"tournament_name": _clean_text(item.get("tournament_name") or "Tournament"), "podium": podium})
    return tournaments


def sanitize_public_recap_json(raw: Any, *, week_start: str | None = None, week_end: str | None = None) -> dict[str, Any]:
    source = _parse_json_object(raw)
    recap: dict[str, Any] = {key: source.get(key) for key in PUBLIC_RECAP_KEYS if key in source}
    if week_start and not recap.get("week_start"):
        recap["week_start"] = week_start
    if week_end and not recap.get("week_end"):
        recap["week_end"] = week_end
    if not recap.get("start_date") and recap.get("week_start"):
        recap["start_date"] = recap.get("week_start")
    if not recap.get("end_date") and recap.get("week_end"):
        recap["end_date"] = recap.get("week_end")

    numbers = recap.get("numbers") if isinstance(recap.get("numbers"), dict) else {}
    recap["numbers"] = {str(key): value for key, value in numbers.items() if isinstance(value, (int, float, str, type(None)))}

    cards = []
    recap_cards = recap.get("numbers_cards") if isinstance(recap.get("numbers_cards"), list) else []
    for card in recap_cards[:12]:
        if not isinstance(card, dict):
            continue
        cards.append({"key": _clean_text(card.get("key")), "label": _clean_text(card.get("label")), "value": card.get("value", 0)})
    recap["numbers_cards"] = cards
    recap["spotlight"] = _sanitize_spotlight(recap.get("spotlight"))
    recap["around_club"] = _sanitize_around_club(recap.get("around_club"))
    recap["tournaments"] = _sanitize_tournaments(recap.get("tournaments")) if "tournaments" in recap else []
    recap["looking_ahead"] = _clean_text_list(recap.get("looking_ahead"), limit=10)
    recap["highlights"] = _clean_text_list(recap.get("highlights"), limit=10)
    return recap


def _recap_summary(recap: dict[str, Any]) -> dict[str, Any]:
    numbers = recap.get("numbers") if isinstance(recap.get("numbers"), dict) else {}
    spotlight = [item for item in (recap.get("spotlight") or []) if isinstance(item, dict) and item.get("include", True)]
    headline = None
    if spotlight:
        players = spotlight[0].get("players") or []
        headline = f"{spotlight[0].get('label', 'Spotlight')}: {', '.join(players[:3])}" if players else str(spotlight[0].get("label") or "Spotlight")
    if not headline:
        highlights = recap.get("highlights") or []
        headline = highlights[0] if highlights else None
    return {
        "matches": numbers.get("matches", 0),
        "players": numbers.get("players", 0),
        "leagues": numbers.get("leagues", 0),
        "headline": headline,
    }


def _published_rows(supabase: Any, *, club_id: str) -> list[dict[str, Any]]:
    try:
        rows = _safe_rows(
            supabase.table("weekly_recaps")
            .select(WEEKLY_RECAP_SELECT)
            .eq("club_id", str(club_id))
            .eq("status", "published")
            .order("week_start", desc=True)
            .execute()
        )
    except Exception:
        return []
    return rows


def build_public_weekly_recaps(supabase: Any, *, club_id: str, week_start: str | None = None) -> dict[str, Any]:
    """Return published weekly recaps only, with a selected public-safe recap detail."""

    rows = _published_rows(supabase, club_id=str(club_id))
    summaries: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    requested = str(week_start or "").strip()

    for row in rows:
        start = str(row.get("week_start") or "").strip()
        end = str(row.get("week_end") or "").strip()
        recap = sanitize_public_recap_json(row.get("final_json"), week_start=start, week_end=end)
        summary = {
            "week_start": start,
            "week_end": end,
            "updated_at": _json_safe(row.get("updated_at") or row.get("published_at") or row.get("created_at")),
            "summary": _recap_summary(recap),
        }
        summaries.append(summary)
        if selected is None and (not requested or requested == start):
            selected = {**summary, "recap": recap}

    return {"recaps": summaries, "selected_recap": selected}


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_weekly_recap_pdf_bytes(recap: dict[str, Any], *, week_start: str, week_end: str) -> bytes:
    """Build a tiny dependency-free public PDF summary for a published recap."""

    lines = ["JUPR Weekly Recap", f"Week: {week_start} to {week_end}", "", "Highlights"]
    highlights = [str(item).strip() for item in (recap.get("highlights", []) or []) if str(item).strip()]
    if not highlights:
        for item in (recap.get("spotlight", []) or []):
            players = [str(player).strip() for player in (item.get("players", []) or []) if str(player).strip()]
            if players:
                highlights.append(f"{item.get('label', 'Award')}: {', '.join(players)}")

    for item in highlights[:6]:
        if item:
            lines.append(f"- {item}")

    lines.append("")
    lines.append("Looking Ahead")
    for item in (recap.get("looking_ahead", []) or [])[:6]:
        text = str(item).strip()
        if text:
            lines.append(f"- {text}")

    text_commands = ["BT", "/F1 12 Tf", "72 770 Td", "14 TL"]
    for idx, line in enumerate(lines[:42]):
        safe_line = _pdf_escape(str(line)).encode("latin-1", "replace").decode("latin-1")
        text_commands.append(f"({safe_line}) Tj")
        if idx < len(lines[:42]) - 1:
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
    pdf.extend(("trailer\n" f"<< /Size {len(offsets)} /Root 1 0 R >>\n" "startxref\n" f"{xref_offset}\n" "%%EOF\n").encode("latin-1"))
    return bytes(pdf)
