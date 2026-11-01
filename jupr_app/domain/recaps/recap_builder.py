"""Snapshot-based editorial selection. Hiding a story never rewrites game results."""
from __future__ import annotations

from copy import deepcopy
from collections import Counter, defaultdict
from hashlib import sha256
import json
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
TEXT_LIMITS = {"title": 240, "text": 3000, "kind": 80, "caption": 600, "credit": 160,
               "credit_label": 80, "results_label": 100, "link_label": 100, "alt": 240}
DISPLAY_FIELDS = ("date", "scope", "footer_title", "footer_link")


def text_override_key(section, item):
    if section == "upcoming":
        return "upcoming:" + json.dumps([text(item["text"], 240), item.get("url", "").strip()], ensure_ascii=False, separators=(",", ":"))
    return item["id"]


def clean_text_override(raw):
    raw = raw if isinstance(raw, dict) else {}
    patch = {key: text(raw[key], limit) for key, limit in TEXT_LIMITS.items() if isinstance(raw.get(key), str)}
    for field, count, limit in (("lines", 30, 600), ("link_labels", 20, 80)):
        values = raw.get(field)
        if isinstance(values, dict):
            patch[field] = {text(key, 600): text(value, limit) for key, value in list(values.items())[:count] if isinstance(value, str)}
    return patch


def clean_display_text(raw):
    return {key: text(raw[key], 160) for key in DISPLAY_FIELDS if isinstance(raw.get(key), str)} if isinstance(raw, dict) else {}


def apply_text_override(item, patch):
    item = deepcopy(item)
    item.update({key: value for key, value in patch.items() if key in TEXT_LIMITS})
    # Bind detail edits to their source text so selecting a different winner
    # or refreshing changed standings cannot transfer an old player's wording.
    item["lines"] = [patch.get("lines", {}).get(line, line) for line in item["lines"]]
    for link in item.get("links", []):
        link["label"] = patch.get("link_labels", {}).get(link["href"], link["label"])
    return item


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
        "lines": [text(line, 600) for line in (raw.get("lines") or [])[:(600 if raw.get("id") == "ladder-court-winners" else 30)]],
        "href": report_href(raw.get("href")), "url": safe_url(raw.get("url")),
        "image_url": image_url(raw.get("image_url")), "alt": text(raw.get("alt"), 240),
        "link_label": text(raw.get("link_label"), 100) if isinstance(raw.get("link_label"), str) else "Learn more",
        "links": [{"label": text(link.get("label"), 80), "href": report_href(link.get("href"))}
                  for link in (raw.get("links") or [])[:20] if isinstance(link, dict) and report_href(link.get("href"))],
    }
    if raw.get("kind") == "Ladder court" and isinstance(raw.get("court_players"), list):
        result["court_players"] = [{"id": text(p.get("id"), 120), "name": text(p.get("name"), 240)}
                                   for p in raw["court_players"][:5] if isinstance(p, dict) and p.get("name")]
    # Optional fields leave saved legacy editions unchanged.
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(raw.get("date") or "")):
        result["date"] = str(raw["date"])
    for key, limit in (("caption", 600), ("credit", 160)):
        if raw.get(key):
            result[key] = text(raw[key], limit)
    for key in ("credit_label", "results_label"):
        if isinstance(raw.get(key), str):
            result[key] = text(raw[key], TEXT_LIMITS[key])
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
    events = report.get("event_results") or []
    identities = [(e.get("kind"), e.get("id") or (e.get("title"), e.get("date"))) for e in events]
    counts = Counter(identities)
    for e, identity in zip(events, identities):
        legacy_identity = (e.get("kind"), (e.get("legacy_title", e.get("title")), e.get("date")))
        # Old snapshots omitted draw IDs. Distinguish their recorded evidence,
        # never list positions, so reordering cannot swap the selected podium.
        evidence_identity = (legacy_identity, e.get("href"), e.get("summary"), tuple(e.get("details") or []))
        aliases = []
        if counts[identity] > 1:
            aliases.append(stable_id("events", identity))
            identity = evidence_identity
        elif e.get("legacy_title") is not None:
            aliases.extend([stable_id("events", legacy_identity), stable_id("events", evidence_identity)])
        add("events", identity,
            title=e.get("title"), text=e.get("summary"), lines=e.get("details"), href=e.get("href"),
            kind=e.get("kind"), date=e.get("date"), links=e.get("weekly_reports"), court_players=e.get("court_players"))
        if aliases:
            groups["events"][-1]["selection_aliases"] = aliases
    # Repeated copies of the very same evidence need only one checkbox.
    groups["events"] = list({item["id"]: item for item in groups["events"]}.values())
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


def resolve_builder_selections(config, groups):
    """Expand legacy shared selections without transferring ambiguous wording."""
    config = deepcopy(config)
    for section, items in groups.items():
        aliases = defaultdict(list)
        current = {item["id"] for item in items}
        for item in items:
            for alias in item.get("selection_aliases", []):
                aliases[alias].append(item["id"])
        settings = config["sections"][section]
        for field in ("item_ids", "email_item_ids"):
            if field in settings:
                settings[field] = list(dict.fromkeys(
                    target for value in settings[field]
                    for target in ([value] if value in current else aliases.get(value, [value]))))
        overrides = config.get("text_overrides", {})
        for alias, targets in aliases.items():
            if len(targets) == 1 and alias not in current and alias in overrides:
                overrides.setdefault(targets[0], deepcopy(overrides[alias]))
    return config


def normalize_config(raw, *, template=False):
    raw = raw if isinstance(raw, dict) else {}
    raw_order = raw.get("order") if isinstance(raw.get("order"), list) else []
    order = list(dict.fromkeys([key for key in raw_order if isinstance(key, str) and key in SECTIONS] + list(SECTIONS)))
    configs = raw.get("sections") if isinstance(raw.get("sections"), dict) else {}
    sections = {}
    for key in SECTIONS:
        src = configs.get(key) if isinstance(configs.get(key), dict) else {}
        section = {"enabled": src.get("enabled") is not False, "email_enabled": src.get("email_enabled") is not False,
                   "title": text(src["title"], 100) if isinstance(src.get("title"), str) else SECTIONS[key], "detail": "brief" if src.get("detail") == "brief" else "full"}
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
    if not template:
        if isinstance(raw.get("text_overrides"), dict):
            result["text_overrides"] = {key: clean_text_override(patch) for key, patch in list(raw["text_overrides"].items())[:1500] if isinstance(key, str) and len(key) <= 5000 and isinstance(patch, dict)}
        if isinstance(raw.get("display_text"), dict):
            result["display_text"] = clean_display_text(raw["display_text"])
    return result


def compact_ladder_courts(items):
    courts = [item for item in items if item.get("kind") == "Ladder court"]
    if not courts:
        return items
    rows, recognized = {}, set()
    for item in courts:
        status = "winner" if item["text"].startswith("Winner:") else "tie" if item["text"].startswith("Tied for first:") else "leader"
        players = item.get("court_players") or [
            {"id": "", "name": match.group(1)} for line in item["lines"]
            if (match := re.fullmatch(r"(.*) — \d+–\d+ · point difference [+-]\d+", line))]
        if not players and status == "winner":
            players = [{"id": "", "name": item["text"][len("Winner:"):].strip()}]
        for player in players:
            if not player["name"]:
                continue
            key = (status, player.get("id") or player["name"])
            row = rows.setdefault(key, {"name": player["name"], "status": status, "ids": set()})
            row["ids"].add(item["id"])
            recognized.add(item["id"])
    lines = []
    for row in rows.values():
        count = len(row["ids"])
        suffix = " (tied for first)" if row["status"] == "tie" else " (incomplete court leader)" if row["status"] == "leader" else ""
        lines.append(row["name"] + (f" ×{count}" if count > 1 else "") + suffix)
    lines.extend(item["text"] or item["title"] for item in courts if item["id"] not in recognized)
    summary = clean_item({"id": "ladder-court-winners", "title": "Ladder court winners", "lines": lines})
    return [summary if item is courts[0] else item for item in items if item.get("kind") != "Ladder court" or item is courts[0]]


def build_editions(recap, raw):
    config = normalize_config(raw)
    groups = catalog(recap)
    config = resolve_builder_selections(config, groups)
    groups["custom"] = config["custom_blocks"]
    editions = {}
    for channel in ("edition", "email_edition"):
        blocks = []
        for key in (config.get("email_order", config["order"]) if channel == "email_edition" else config["order"]):
            settings = config["sections"][key]
            if not settings["enabled"] or (channel == "email_edition" and not settings["email_enabled"]):
                continue
            available = {item["id"]: item if item.get("kind") == "Ladder court" else apply_text_override(item, config.get("text_overrides", {}).get(text_override_key(key, item), {})) for item in groups[key]}
            ids = settings.get("item_ids", list(available))
            if channel == "email_edition":
                selected = set(ids)
                ids = [i for i in settings.get("email_item_ids", ids[:EMAIL_LIMITS.get(key, 150)]) if i in selected]
            items = [deepcopy(available[i]) for i in ids if i in available]
            for item in items:
                if settings["detail"] == "brief" and key != "spotlight" and item.get("kind") != "Ladder court":
                    # League results often live entirely in details; retain the winner.
                    item["lines"] = item["lines"][:1]
                item["lines"] = [line for line in item["lines"] if line]
            items = [i for i in items if i["title"] or i["text"] or i["lines"] or i["image_url"]]
            if key == "events":
                items = [apply_text_override(item, config.get("text_overrides", {}).get(item["id"], {}))
                         if item["id"] == "ladder-court-winners" else item for item in compact_ladder_courts(items)]
            if items:
                blocks.append({"key": key, "title": settings["title"], "items": items})
        editions[channel] = blocks
    return editions


def apply_builder(recap, config):
    recap.setdefault("report", {}).update(build_editions(recap, config))
    if isinstance(config.get("display_text"), dict):
        recap["report"]["display_text"] = clean_display_text(config["display_text"])
    return recap


def public_edition_report(raw):
    result = {key: text(raw.get(key), 160) for key in ("headline", "club_name", "club_slug", "scope_label")}
    result["version"] = 1
    if isinstance(raw.get("display_text"), dict):
        result["display_text"] = clean_display_text(raw["display_text"])
    for channel in ("edition", "email_edition"):
        result[channel] = [{"key": block["key"], "title": text(block.get("title"), 100),
                            "items": [clean_item(item) for item in block.get("items", [])[:150]]}
                           for block in raw.get(channel, [])[:len(SECTIONS)] if isinstance(block, dict) and block.get("key") in SECTIONS]
    return result


def edition_lines(report):
    lines = [report.get("headline", "This week on the courts")]
    display = report.get("display_text", {})
    scope = display.get("scope", f"League: {report['scope_label']}" if report.get("scope_label") and report["scope_label"] != "Whole club" else "")
    if scope:
        lines.append(scope)
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
                lines.append(f"{item.get('credit_label', 'Photo:')} {item['credit']}".strip())
    return lines
