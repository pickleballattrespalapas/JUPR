from __future__ import annotations

from pathlib import Path
import re

_REQUIREMENTS_CACHE: dict[str, str] | None = None

_HEADING_RE = re.compile(r"^#{2,3}\s+([^\s]+)\s+[—–-]\s+.+$")
_REQ_RE = re.compile(r"^(Unlock|Status):\s*(.+)$", re.IGNORECASE)


def _requirements_paths() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[3]
    return [
        repo_root / "docs" / "badge_requirements.md",
        Path(__file__).resolve().parent / "badge_requirements.md",
    ]


def _parse_requirements(text: str) -> dict[str, str]:
    requirements: dict[str, str] = {}
    current_badge_id: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            current_badge_id = heading_match.group(1).strip()
            continue
        if not current_badge_id:
            continue
        req_match = _REQ_RE.match(line)
        if not req_match:
            continue
        if current_badge_id in requirements:
            continue
        requirement_text = req_match.group(2).strip()
        if not requirement_text:
            continue
        requirements[current_badge_id] = " ".join(requirement_text.split())
    return requirements


def load_requirements() -> dict[str, str]:
    global _REQUIREMENTS_CACHE
    if _REQUIREMENTS_CACHE is not None:
        return _REQUIREMENTS_CACHE
    requirements: dict[str, str] = {}
    for path in _requirements_paths():
        if path.exists():
            requirements = _parse_requirements(path.read_text(encoding="utf-8"))
            break
    _REQUIREMENTS_CACHE = requirements
    return requirements


def requirement_for(badge_id: str) -> str:
    requirements = load_requirements()
    return requirements.get(str(badge_id), "Requirements TBD")
