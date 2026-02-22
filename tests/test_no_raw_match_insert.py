from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "jupr_app"


def test_no_raw_match_table_insert_calls() -> None:
    violations: list[str] = []
    for py_file in SOURCE_ROOT.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        if 'table("matches").insert(' in text or "table('matches').insert(" in text:
            violations.append(str(py_file.relative_to(REPO_ROOT)))

    assert not violations, f"Direct matches insert detected in: {', '.join(violations)}"
