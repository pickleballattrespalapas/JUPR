from __future__ import annotations

from pathlib import Path
import re

_REQUIREMENTS_CACHE: dict[str, str] | None = None

_HEADING_RE = re.compile(r"^#{2,3}\s+(.+)$")
_REQ_RE = re.compile(r"^(Unlock|Status):\s*(.+)$", re.IGNORECASE)


def _requirements_path() -> Path | None:
    source = Path(__file__).resolve()
    for parent in source.parents:
        candidate = parent / "docs" / "badge_requirements.md"
        if candidate.exists():
            return candidate
    return None


def _extract_badge_id(heading_line: str) -> str | None:
    match = _HEADING_RE.match(heading_line)
    if not match:
        return None
    content = match.group(1).strip()
    if not content:
        return None
    for separator in (" — ", " – ", " - ", "—", "–", "-"):
        if separator in content:
            return content.split(separator, 1)[0].strip()
    return content.split()[0].strip()


def _parse_requirements(text: str) -> dict[str, str]:
    requirements: dict[str, str] = {}
    lines = text.splitlines()
    current_badge_id: str | None = None
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if _HEADING_RE.match(line):
            current_badge_id = _extract_badge_id(line)
            index += 1
            continue
        if not current_badge_id or current_badge_id in requirements:
            index += 1
            continue
        req_match = _REQ_RE.match(line)
        if not req_match:
            index += 1
            continue
        requirement_lines = [req_match.group(2).strip()]
        lookahead = index + 1
        while lookahead < len(lines):
            next_line = lines[lookahead].strip()
            if not next_line:
                break
            if _HEADING_RE.match(next_line):
                break
            if next_line.startswith("-"):
                requirement_lines.append(next_line.lstrip("-").strip())
            lookahead += 1
        requirement_text = " ".join(" ".join(requirement_lines).split())
        if requirement_text:
            requirements[current_badge_id] = requirement_text
        index = lookahead
    return requirements


def load_requirements_map() -> dict[str, str]:
    global _REQUIREMENTS_CACHE
    if _REQUIREMENTS_CACHE is not None:
        return _REQUIREMENTS_CACHE
    requirements: dict[str, str] = {}
    requirements_path = _requirements_path()
    if requirements_path and requirements_path.exists():
        requirements = _parse_requirements(requirements_path.read_text(encoding="utf-8"))
    _REQUIREMENTS_CACHE = requirements
    return requirements


def load_requirements() -> dict[str, str]:
    return load_requirements_map()


def requirement_for(badge_id: str) -> str:
    requirements = load_requirements_map()
    return requirements.get(str(badge_id), "Requirements TBD")


def audit_missing_badge_requirements(badge_defs) -> list[dict]:
    """Return missing requirement entries for the badge catalog.

    To resolve missing entries, add or update the badge heading in docs/badge_requirements.md:
    "## <badge_id> — <Badge Name>" followed by an "Unlock:" line with the requirement text.
    """

    def _get_field(row, *fields):
        for field in fields:
            if isinstance(row, dict) and field in row:
                return row.get(field)
            if hasattr(row, field):
                return getattr(row, field)
        return None

    def _iter_badges(defs):
        if defs is None:
            return []
        if hasattr(defs, "itertuples"):
            return list(defs.itertuples(index=False))
        return list(defs)

    def _first_value(row, *fields):
        for field in fields:
            value = _get_field(row, field)
            if value is not None and str(value).strip() != "":
                return value
        return None

    missing: list[dict] = []
    for row in _iter_badges(badge_defs):
        badge_id = _first_value(row, "badge_id", "slug", "id")
        if badge_id is None:
            continue
        badge_id_str = str(badge_id)
        returned_text = requirement_for(badge_id_str)
        cleaned = str(returned_text or "").strip()
        if not cleaned or cleaned == "Requirements TBD":
            missing.append(
                {
                    "badge_id": badge_id_str,
                    "slug": _first_value(row, "slug"),
                    "name": str(_get_field(row, "name") or "Badge"),
                    "category": _get_field(row, "category"),
                    "source_hint": _first_value(row, "source_hint", "source"),
                    "returned_text": cleaned or "Requirements TBD",
                }
            )
    return missing


def missing_badge_ids(all_badge_ids: list[str]) -> list[str]:
    requirements = load_requirements_map()
    return [str(badge_id) for badge_id in all_badge_ids if str(badge_id) not in requirements]


if __name__ == "__main__":
    import sys

    ids = [arg.strip() for arg in sys.argv[1:] if arg.strip()]
    if ids:
        print(missing_badge_ids(ids))
