"""Explicit public projection for report snapshots and email-safe links."""
from urllib.parse import urlsplit
import re


def text(value, limit=3000):
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def safe_url(value):
    value = str(value or "").strip()
    if any(c.isspace() or ord(c) < 32 for c in value):
        return ""
    try:
        parsed = urlsplit(value)
        return value if parsed.scheme in {"https", "http"} and parsed.hostname and not parsed.username and not parsed.password else ""
    except ValueError:
        return ""


def report_href(value):
    value = str(value or "")
    if any(c.isspace() or ord(c) < 32 for c in value) or any(c in value for c in ("..", "\\", ":", "#")):
        return ""
    segment = r"[A-Za-z0-9_.%~-]+"
    routes = [rf"leagues/{segment}/standings", rf"leagues/{segment}/weekly-history(?:\?week=[1-9][0-9]*)?",
              rf"round-robin-generator/shared/{segment}/standings", rf"tournament-team-results/{segment}/{segment}",
              rf"tournament-results\?tournament={segment}", rf"/interclub/{segment}"]
    if any(re.fullmatch(pattern, value) for pattern in routes):
        return value
    return ""


def sanitize_report(raw):
    if not isinstance(raw, dict):
        return {}
    result = {k: text(raw.get(k), 3000 if k in {"opening", "event_notes"} else 160) for k in ("club_name", "club_slug", "headline", "opening", "event_notes")}
    result["version"] = 1 if raw.get("version") == 1 else 0
    result["event_results"] = [{**{k: text(e.get(k), 600) for k in ("id", "title", "kind", "date", "summary")}, "details": [text(line, 600) for line in e.get("details", [])[:30]], "href": report_href(e.get("href"))} for e in raw.get("event_results", [])[:100] if isinstance(e, dict)]
    for event, source in zip(result["event_results"], [e for e in raw.get("event_results", [])[:100] if isinstance(e, dict)]):
        event["weekly_reports"] = [{"label": text(link.get("label"), 80), "href": report_href(link.get("href"))} for link in source.get("weekly_reports", [])[:20] if isinstance(link, dict) and report_href(link.get("href"))]
    result["season_races"] = [{"title": text(r.get("title"), 240), "lines": [text(line, 500) for line in r.get("lines", [])[:8]], "href": report_href(r.get("href"))} for r in raw.get("season_races", [])[:60] if isinstance(r, dict)]
    for k in ("achievements", "new_faces"):
        result[k] = [text(line, 500) for line in raw.get(k, [])[:100]]
    result["upcoming"] = [{"text": text(e.get("text"), 240), "url": safe_url(e.get("url"))} for e in raw.get("upcoming", [])[:10] if isinstance(e, dict) and text(e.get("text"))]
    return result


def report_lines(report):
    lines = [report.get("headline") or "This week on the courts"]
    if report.get("opening"):
        lines.extend([report["opening"], ""])
    if report.get("event_results"):
        lines.append("Event results")
        for event in report["event_results"]:
            lines.extend([event["title"], event.get("summary") or "", *[f"- {line}" for line in event.get("details", [])]])
    if report.get("event_notes"):
        lines.extend(["Club notes", report["event_notes"]])
    if report.get("season_races"):
        lines.append("Season awards race")
        for race in report["season_races"]:
            lines.extend([race["title"], *[f"- {line}" for line in race.get("lines", [])]])
    if report.get("achievements"):
        lines.extend(["New achievements", *[f"- {line}" for line in report["achievements"]]])
    if report.get("new_faces"):
        lines.extend(["Welcome to the courts", ", ".join(report["new_faces"])])
    if report.get("upcoming"):
        lines.extend(["Coming up", *[f"- {item['text']}{': ' + item['url'] if item.get('url') else ''}" for item in report["upcoming"]]])
    return lines
