"""Snapshot-based editorial selection. Hiding a story never rewrites game results."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import re

from .weekly_report_public import safe_url, report_href, text

SECTIONS = {
    "opening": "From the courts", "numbers": "The week in numbers",
    "events": "Event results", "spotlight": "Player highlights",
    "races": "Season awards race", "achievements": "New achievements",
    "welcome": "Welcome to the courts", "notes": "Around the club",
    "custom": "Club stories", "upcoming": "Coming up",
}
EMAIL_LIMITS = {"events": 3, "spotlight": 2, "races": 2, "achievements": 3}


def stable_id(kind, identity):
    if kind == "spotlight":
        return f"spotlight-{identity}"
    return f"{kind}-{sha256(str(identity).encode()).hexdigest()[:20]}"


def image_url(value):
    value = str(value or "")
    if re.fullmatch(r"data:image/(?:png|jpeg|webp);base64,[A-Za-z0-9+/=]+", value) and len(value) <= 300000:
        return value
    url = safe_url(value)
    return url if url.startswith("https://") else ""


def clean_item(raw):
    if not isinstance(raw, dict):
        return {}
    result = {
        "id": text(raw.get("id"), 120), "title": text(raw.get("title"), 240),
        "text": text(raw.get("text")), "kind": text(raw.get("kind"), 80),
        "lines": [text(line, 600) for line in (raw.get("lines") or [])[:30]],
        "href": report_href(raw.get("href")), "url": safe_url(raw.get("url")),
        "image_url": image_url(raw.get("image_url")), "alt": text(raw.get("alt"), 240),
        "link_label": text(raw.get("link_label"), 100) or "Learn more",
        "links": [{"label": text(link.get("label"), 80), "href": report_href(link.get("href"))}
                  for link in (raw.get("links") or [])[:20] if isinstance(link, dict) and report_href(link.get("href"))],
    }
    # Optional fields leave saved legacy editions unchanged.
    for key, limit in (("caption", 600), ("credit", 160)):
        if raw.get(key):
            result[key] = text(raw[key], limit)
    if isinstance(raw.get("layout"), str) and raw["layout"] in {"plain", "photo", "callout", "promotion"}:
        result["layout"] = raw["layout"]
    if isinstance(raw.get("image_fit"), str) and raw["image_fit"] in {"contain", "cover"}:
        result["image_fit"] = raw["image_fit"]
    return result


def catalog(recap):
    """Stable item identities come from source IDs, never display positions."""
    report = recap.get("report") or {}
    groups = {key: [] for key in SECTIONS}
    def add(section, identity, **item):
        groups[section].append(clean_item({"id": stable_id(section, identity), **item}))
    add("opening", "opening", text=report.get("opening", ""))
    for card in recap.get("numbers_cards") or []:
        add("numbers", card.get("key") or card.get("label"), title=card.get("label"), text=str(card.get("value", 0)))
    for e in report.get("event_results") or []:
        add("events", (e.get("kind"), e.get("id") or (e.get("title"), e.get("date"))),
            title=e.get("title"), text=e.get("summary"), lines=e.get("details"), href=e.get("href"),
            kind=e.get("kind"), links=e.get("weekly_reports"))
    for s in recap.get("spotlight") or []:
        if s.get("include", True) and s.get("players"):
            add("spotlight", s.get("key") or s.get("label"), title=s.get("label"), text=s.get("description"), lines=s.get("players"), kind=s.get("key"))
    for race in report.get("season_races") or []:
        add("races", (race.get("title"), race.get("href")), title=race.get("title"), lines=race.get("lines"), href=race.get("href"))
    for line in report.get("achievements") or []:
        add("achievements", line, text=line)
    for name in report.get("new_faces") or []:
        add("welcome", name, text=name)
    add("notes", "notes", text=report.get("event_notes", ""))
    for item in report.get("upcoming") or []:
        add("upcoming", (item.get("text"), item.get("url")), text=item.get("text"), url=item.get("url"), link_label=item.get("text"))
    for line in recap.get("looking_ahead") or []:
        if str(line).strip():
            add("upcoming", line, text=line)
    return groups


def normalize_config(raw, *, template=False):
    raw = raw if isinstance(raw, dict) else {}
    raw_order = raw.get("order") if isinstance(raw.get("order"), list) else []
    order = list(dict.fromkeys([key for key in raw_order if isinstance(key, str) and key in SECTIONS] + list(SECTIONS)))
    configs = raw.get("sections") if isinstance(raw.get("sections"), dict) else {}
    sections = {}
    for key in SECTIONS:
        src = configs.get(key) if isinstance(configs.get(key), dict) else {}
        section = {"enabled": src.get("enabled") is not False, "email_enabled": src.get("email_enabled") is not False,
                   "title": text(src.get("title"), 100) or SECTIONS[key], "detail": "brief" if src.get("detail") == "brief" else "full"}
        for selection in ("item_ids", "email_item_ids"):
            if (not template or key in {"numbers", "spotlight"}) and isinstance(src.get(selection), list):
                section[selection] = list(dict.fromkeys(text(v, 120) for v in src[selection][:150]))
        sections[key] = section
    custom = raw.get("custom_blocks") if isinstance(raw.get("custom_blocks"), list) else []
    blocks = [] if template else [clean_item(item) for item in custom[:12] if isinstance(item, dict)]
    result = {"version": 1, "order": order, "sections": sections, "custom_blocks": blocks,
              "step": 0 if template else max(0, min(4, int(raw.get("step", 0) or 0)))}
    if isinstance(raw.get("email_order"), list):
        result["email_order"] = list(dict.fromkeys([key for key in raw["email_order"] if isinstance(key, str) and key in SECTIONS] + order))
    return result


def build_editions(recap, raw):
    config = normalize_config(raw)
    groups = catalog(recap)
    groups["custom"] = config["custom_blocks"]
    editions = {}
    for channel in ("edition", "email_edition"):
        blocks = []
        for key in (config.get("email_order", config["order"]) if channel == "email_edition" else config["order"]):
            settings = config["sections"][key]
            if not settings["enabled"] or (channel == "email_edition" and not settings["email_enabled"]):
                continue
            available = {item["id"]: item for item in groups[key]}
            ids = settings.get("item_ids", list(available))
            if channel == "email_edition":
                selected = set(ids)
                ids = [i for i in settings.get("email_item_ids", ids[:EMAIL_LIMITS.get(key, 150)]) if i in selected]
            items = [deepcopy(available[i]) for i in ids if i in available]
            for item in items:
                if settings["detail"] == "brief":
                    # League results often live entirely in details; retain the winner.
                    item["lines"] = item["lines"][:1]
                if channel == "email_edition" and key == "spotlight":
                    item["lines"] = item["lines"][:1]
            items = [i for i in items if i["title"] or i["text"] or i["lines"] or i["image_url"]]
            if items:
                blocks.append({"key": key, "title": settings["title"], "items": items})
        editions[channel] = blocks
    return editions


def apply_builder(recap, config):
    recap.setdefault("report", {}).update(build_editions(recap, config))
    return recap


def public_edition_report(raw):
    result = {key: text(raw.get(key), 160) for key in ("headline", "club_name", "club_slug", "scope_label")}
    result["version"] = 1
    for channel in ("edition", "email_edition"):
        result[channel] = [{"key": block["key"], "title": text(block.get("title"), 100),
                            "items": [clean_item(item) for item in block.get("items", [])[:150]]}
                           for block in raw.get(channel, [])[:len(SECTIONS)] if isinstance(block, dict) and block.get("key") in SECTIONS]
    return result


def edition_lines(report):
    lines = [report.get("headline") or "This week on the courts"]
    if report.get("scope_label") and report["scope_label"] != "Whole club":
        lines.append(f"League: {report['scope_label']}")
    for block in report.get("edition", []):
        lines.extend(["", block["title"]])
        for item in block["items"]:
            if item["title"]:
                lines.append(item["title"])
            if item["text"]:
                lines.extend(item["text"].splitlines())
            lines.extend(f"- {line}" for line in item["lines"])
            if item["url"]:
                lines.append(item["url"])
            if item["image_url"] and item["alt"]:
                lines.append(item["alt"])
            if item.get("caption"):
                lines.append(item["caption"])
            if item.get("credit"):
                lines.append(f"Photo: {item['credit']}")
    return lines
