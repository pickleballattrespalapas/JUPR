from __future__ import annotations

from pathlib import Path

from scripts.check_parity_closure_program import check_program, partial_keys


ROOT = Path(__file__).resolve().parents[1]


def test_parity_closure_program_covers_every_partial_page_once() -> None:
    assert check_program() == []


def test_current_closure_wave_is_45_pages() -> None:
    text = (ROOT / "docs" / "next_streamlit_parity_matrix.md").read_text(
        encoding="utf-8"
    )
    assert len(partial_keys(text)) == 45


def test_program_rejects_a_missing_page(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.md"
    program = tmp_path / "program.md"
    matrix.write_text("| `alpha` | A | Public | `Partial` | No | Gate | Next |\n")
    program.write_text("# Empty\n")

    errors = check_program(matrix, program)

    assert errors == [
        "Partial pages missing closure contracts: alpha",
        "Expected the current closure wave to contain 45 Partial pages, found 1. "
        "Update the program deliberately when matrix statuses change.",
    ]

