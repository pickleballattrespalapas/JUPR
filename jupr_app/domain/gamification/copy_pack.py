from __future__ import annotations

from jupr_app.domain.gamification.presentation import badge_requirement

import hashlib
import json
from pathlib import Path
import re
from typing import Any


_COPY_PACK_CACHE: dict[str, Any] | None = None
_TEMPLATE_RE = re.compile(r"{([^{}]+)}")


def load_copy_pack() -> dict[str, Any]:
    global _COPY_PACK_CACHE
    if _COPY_PACK_CACHE is not None:
        return _COPY_PACK_CACHE

    path = Path(__file__).with_name("copy_pack.yaml")
    if not path.exists():
        _COPY_PACK_CACHE = {"style_guide": {}, "badges": {}}
        return _COPY_PACK_CACHE

    with path.open("r", encoding="utf-8") as handle:
        raw = handle.read()
    data = _parse_simple_yaml(raw)
    if not isinstance(data, dict):
        data = {"style_guide": {}, "badges": {}}
    _COPY_PACK_CACHE = data
    return data


def _parse_simple_yaml(raw: str) -> dict[str, Any]:
    lines = [line.rstrip("\n") for line in raw.splitlines() if line.strip() != ""]
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    def _parse_value(value: str) -> Any:
        value = value.strip()
        if value == "":
            return None
        try:
            return json.loads(value)
        except Exception:
            return value

    def _collect_block(start_index: int, base_indent: int) -> tuple[str, int]:
        block_lines: list[str] = []
        idx = start_index + 1
        min_indent = None
        while idx < len(lines):
            next_line = lines[idx]
            next_indent = len(next_line) - len(next_line.lstrip(" "))
            if next_indent <= base_indent:
                break
            if min_indent is None or next_indent < min_indent:
                min_indent = next_indent
            block_lines.append(next_line)
            idx += 1
        if min_indent is None:
            return "", idx - 1
        cleaned = [line[min_indent:] if len(line) >= min_indent else "" for line in block_lines]
        return "\n".join(cleaned), idx - 1

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if content.startswith("- "):
            if not isinstance(parent, list):
                new_list: list[Any] = []
                if isinstance(parent, dict):
                    raise ValueError("Malformed YAML: list without key")
                parent = new_list
            item_value = content[2:].strip()
            if item_value in {"|", ">"}:
                block, idx = _collect_block(idx, indent)
                parent.append(block)
            elif item_value == "":
                new_item: dict[str, Any] = {}
                parent.append(new_item)
                stack.append((indent, new_item))
            else:
                parent.append(_parse_value(item_value))
            idx += 1
            continue

        if ":" in content:
            key, remainder = content.split(":", 1)
            key = key.strip()
            value = remainder.strip()
            if value in {"|", ">"}:
                block, idx = _collect_block(idx, indent)
                if isinstance(parent, dict):
                    parent[key] = block
                else:
                    raise ValueError("Malformed YAML: mapping entry in list")
                idx += 1
                continue
            if value == "":
                container: Any = {}
                if idx + 1 < len(lines):
                    next_line = lines[idx + 1]
                    next_indent = len(next_line) - len(next_line.lstrip(" "))
                    if next_line.strip().startswith("- ") and next_indent > indent:
                        container = []
                if isinstance(parent, dict):
                    parent[key] = container
                else:
                    raise ValueError("Malformed YAML: mapping entry in list")
                stack.append((indent, container))
            else:
                if isinstance(parent, dict):
                    parent[key] = _parse_value(value)
                else:
                    raise ValueError("Malformed YAML: mapping entry in list")
            idx += 1
            continue

        idx += 1

    return root


def pick_variant(variants: list[str], seed_str: str) -> str:
    if not variants:
        return ""
    seed = seed_str or ""
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(variants)
    return str(variants[idx])


def render_template(text: str, data: dict[str, Any]) -> str:
    if not text:
        return ""

    def _format_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            rendered = f"{value:.2f}".rstrip("0").rstrip(".")
            return rendered
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        return str(value)

    def _replace(match: re.Match) -> str:
        key = match.group(1).strip()
        if not key:
            return ""
        return _format_value(data.get(key))

    rendered = _TEMPLATE_RE.sub(_replace, text)
    cleaned_lines = []
    for line in rendered.splitlines():
        cleaned = " ".join(line.split())
        if cleaned:
            cleaned_lines.append(cleaned)
    return "\n".join(cleaned_lines)


def get_badge_copy(badge_id: str) -> dict[str, Any]:
    pack = load_copy_pack()
    badges = pack.get("badges", {}) if isinstance(pack, dict) else {}
    badge_copy = badges.get(str(badge_id), {}) if isinstance(badges, dict) else {}
    return {
        "name": badge_copy.get("name", ""),
        "lore": badge_requirement(badge_id),
        "hint": badge_requirement(badge_id),
        "rarity": badge_copy.get("rarity", "common"),
        "tier": badge_copy.get("tier", None),
        "icon_key": badge_copy.get("icon_key", None),
        "scope": badge_copy.get("scope", "overall"),
        "tape_excerpts": [badge_requirement(badge_id)],
        "highlight": badge_copy.get("highlight", {}) or {},
        "foreshadow": badge_copy.get("foreshadow", {}) or {},
    }


def assert_no_banned_words(text: str) -> None:
    pack = load_copy_pack()
    style = pack.get("style_guide", {}) if isinstance(pack, dict) else {}
    banned_words = style.get("banned_words", []) if isinstance(style, dict) else []
    if not banned_words:
        return
    joined = (text or "").lower()
    for word in banned_words:
        if not word:
            continue
        assert str(word).lower() not in joined, f"forbidden word found: {word}"
