from __future__ import annotations

import json
import textwrap
from datetime import date, datetime
from typing import Any

WEEKLY_RECAP_SELECT = "club_id,week_start,week_end,status,final_json"
DEFAULT_RECAP_PAGE_SIZE = 8
MAX_RECAP_PAGE_SIZE = 12
PUBLIC_NUMBER_KEYS = {
    "matches",
    "players",
    "leagues",
    "round_robins",
    "community_events",
    "social_round_robins",
    "new_faces",
}
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


def _default_numbers_cards(numbers: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"key": "matches", "label": "Matches", "value": numbers.get("matches", 0)},
        {"key": "players", "label": "Players", "value": numbers.get("players", 0)},
        {"key": "leagues", "label": "Leagues", "value": numbers.get("leagues", 0)},
        {"key": "round_robins", "label": "Pop-Ups", "value": numbers.get("round_robins", 0)},
        {
            "key": "community_events",
            "label": "Community Events",
            "value": numbers.get("community_events", numbers.get("social_round_robins", 0)),
        },
        {"key": "new_faces", "label": "New Faces", "value": numbers.get("new_faces", 0)},
    ]


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
    recap["numbers"] = {
        str(key): value
        for key, value in numbers.items()
        if str(key) in PUBLIC_NUMBER_KEYS and isinstance(value, (int, float, str, type(None)))
    }

    cards_by_key: dict[str, dict[str, Any]] = {}
    recap_cards = recap.get("numbers_cards") if isinstance(recap.get("numbers_cards"), list) else []
    for card in recap_cards[:12]:
        if not isinstance(card, dict):
            continue
        key = _clean_text(card.get("key"))
        value = card.get("value", 0)
        if key not in PUBLIC_NUMBER_KEYS or not isinstance(value, (int, float, str, type(None))):
            continue
        cards_by_key[key] = {"key": key, "label": _clean_text(card.get("label")), "value": value}
    complete_cards = []
    for default_card in _default_numbers_cards(recap["numbers"]):
        supplied = cards_by_key.get(str(default_card["key"]))
        complete_cards.append(
            {
                "key": default_card["key"],
                "label": (supplied or {}).get("label") or default_card["label"],
                "value": (supplied or {}).get("value", default_card["value"]),
            }
        )
    recap["numbers_cards"] = complete_cards
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


def _published_rows(supabase: Any, *, club_id: str, offset: int, limit: int) -> list[dict[str, Any]]:
    try:
        query = (
            supabase.table("weekly_recaps")
            .select(WEEKLY_RECAP_SELECT)
            .eq("club_id", str(club_id))
            .eq("status", "published")
            .order("week_start", desc=True)
        )
        if hasattr(query, "range"):
            query = query.range(int(offset), int(offset + limit - 1))
            return _safe_rows(query.execute())
        rows = _safe_rows(query.execute())
        rows = rows[int(offset) : int(offset + limit)]
    except Exception:
        return []
    return rows


def _published_row(supabase: Any, *, club_id: str, week_start: str) -> dict[str, Any] | None:
    requested = str(week_start or "").strip()
    if not requested:
        return None
    try:
        query = (
            supabase.table("weekly_recaps")
            .select(WEEKLY_RECAP_SELECT)
            .eq("club_id", str(club_id))
            .eq("status", "published")
            .eq("week_start", requested)
        )
        if hasattr(query, "limit"):
            query = query.limit(1)
        rows = _safe_rows(query.execute())
    except Exception:
        return None
    return rows[0] if rows else None


def _public_recap_row(row: dict[str, Any], *, include_recap: bool) -> dict[str, Any]:
    start = str(row.get("week_start") or "").strip()
    end = str(row.get("week_end") or "").strip()
    recap = sanitize_public_recap_json(row.get("final_json"), week_start=start, week_end=end)
    result = {
        "week_start": start,
        "week_end": end,
        "updated_at": _json_safe(row.get("updated_at") or row.get("published_at") or row.get("created_at")),
        "summary": _recap_summary(recap),
    }
    if include_recap:
        result["recap"] = recap
    return result


def build_public_weekly_recaps(
    supabase: Any,
    *,
    club_id: str,
    week_start: str | None = None,
    page: int = 1,
    page_size: int = DEFAULT_RECAP_PAGE_SIZE,
) -> dict[str, Any]:
    """Return published weekly recaps only, with a selected public-safe recap detail."""

    normalized_page = max(1, int(page or 1))
    normalized_page_size = min(MAX_RECAP_PAGE_SIZE, max(1, int(page_size or DEFAULT_RECAP_PAGE_SIZE)))
    offset = (normalized_page - 1) * normalized_page_size
    rows = _published_rows(
        supabase,
        club_id=str(club_id),
        offset=offset,
        limit=normalized_page_size + 1,
    )
    has_next = len(rows) > normalized_page_size
    page_rows = rows[:normalized_page_size]
    summaries: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    requested = str(week_start or "").strip()

    for row in page_rows:
        summary = _public_recap_row(row, include_recap=False)
        summaries.append(summary)
        if selected is None and (not requested or requested == summary["week_start"]):
            selected = _public_recap_row(row, include_recap=True)

    if requested and selected is None:
        requested_row = _published_row(supabase, club_id=str(club_id), week_start=requested)
        if requested_row:
            selected = _public_recap_row(requested_row, include_recap=True)

    return {
        "recaps": summaries,
        "selected_recap": selected,
        "pagination": {
            "page": normalized_page,
            "page_size": normalized_page_size,
            "has_previous": normalized_page > 1,
            "has_next": has_next,
        },
    }


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _weekly_recap_pdf_lines(recap: dict[str, Any], *, week_start: str, week_end: str) -> list[str]:
    lines = ["JUPR Weekly Recap", f"Week: {week_start} to {week_end}", ""]

    lines.append("Weekly Numbers")
    for card in recap.get("numbers_cards", []) or []:
        label = str(card.get("label") or card.get("key") or "Metric").strip()
        lines.append(f"- {label}: {card.get('value', 0)}")

    lines.extend(["", "Spotlight Reel"])
    spotlight = [item for item in (recap.get("spotlight", []) or []) if item.get("include", True)]
    if not spotlight:
        lines.append("- No spotlight reel published for this week.")
    for item in spotlight:
        players = ", ".join(str(player).strip() for player in (item.get("players", []) or []) if str(player).strip())
        label = str(item.get("label") or "Award").strip()
        description = str(item.get("description") or "").strip()
        lines.append(f"- {label}{f': {players}' if players else ''}")
        if description:
            lines.append(f"  {description}")

    lines.extend(["", "Around the Club"])
    around = recap.get("around_club") if isinstance(recap.get("around_club"), dict) else {}
    around_groups = (
        ("leagues", "league_name", "League"),
        ("round_robins", "event_name", "Pop-Up Event"),
        ("community_events", "event_name", "Community Event"),
    )
    around_count = 0
    for group_key, title_key, fallback in around_groups:
        for item in around.get(group_key, []) or []:
            around_count += 1
            title = str(item.get(title_key) or item.get("event_type_label") or fallback).strip()
            lines.append(f"- {title}")
            for highlight in item.get("highlights", []) or []:
                display = str((highlight or {}).get("display") or "").strip()
                if display:
                    lines.append(f"  - {display}")
    if not around_count:
        lines.append("- No around-the-club highlights published for this week.")

    lines.extend(["", "Tournaments"])
    tournaments = recap.get("tournaments", []) or []
    if not tournaments:
        lines.append("- No tournament podiums published for this week.")
    for tournament in tournaments:
        lines.append(f"- {str(tournament.get('tournament_name') or 'Tournament').strip()}")
        for podium_row in sorted(tournament.get("podium", []) or [], key=lambda row: int(row.get("placement") or 99)):
            lines.append(f"  {int(podium_row.get('placement') or 0)}. {str(podium_row.get('display_name') or '').strip()}")

    lines.extend(["", "Looking Ahead"])
    looking_ahead = [str(item).strip() for item in (recap.get("looking_ahead", []) or []) if str(item).strip()]
    if not looking_ahead:
        lines.append("- No looking-ahead notes published for this week.")
    else:
        lines.extend(f"- {item}" for item in looking_ahead)

    wrapped: list[str] = []
    for line in lines:
        if not line:
            wrapped.append("")
            continue
        subsequent = "  " if line.startswith("-") else ""
        wrapped.extend(textwrap.wrap(line, width=82, subsequent_indent=subsequent) or [""])
    return wrapped


def build_weekly_recap_pdf_bytes(recap: dict[str, Any], *, week_start: str, week_end: str) -> bytes:
    """Build a dependency-free, multi-page PDF of the complete published recap."""

    lines = _weekly_recap_pdf_lines(recap, week_start=week_start, week_end=week_end)
    page_lines = [lines[index : index + 48] for index in range(0, len(lines), 48)] or [["JUPR Weekly Recap"]]
    objects: dict[int, bytes] = {}
    page_ids = [4 + (index * 2) for index in range(len(page_lines))]
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("latin-1")
    objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    for index, current_lines in enumerate(page_lines):
        page_id = page_ids[index]
        content_id = page_id + 1
        text_commands = ["BT", "/F1 10 Tf", "54 756 Td", "13 TL"]
        for line_index, line in enumerate(current_lines):
            safe_line = _pdf_escape(str(line)).encode("latin-1", "replace").decode("latin-1")
            text_commands.append(f"({safe_line}) Tj")
            if line_index < len(current_lines) - 1:
                text_commands.append("T*")
        text_commands.append("ET")
        content = "\n".join(text_commands).encode("latin-1")
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode("latin-1")
        objects[content_id] = f"<< /Length {len(content)} >>\nstream\n".encode("latin-1") + content + b"\nendstream"

    max_object_id = max(objects)
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0] * (max_object_id + 1)
    for object_id in range(1, max_object_id + 1):
        offsets[object_id] = len(pdf)
        pdf.extend(f"{object_id} 0 obj\n".encode("latin-1"))
        pdf.extend(objects[object_id])
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {max_object_id + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(("trailer\n" f"<< /Size {max_object_id + 1} /Root 1 0 R >>\n" "startxref\n" f"{xref_offset}\n" "%%EOF\n").encode("latin-1"))
    return bytes(pdf)
