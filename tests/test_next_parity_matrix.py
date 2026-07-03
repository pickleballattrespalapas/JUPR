from __future__ import annotations

from scripts.check_next_parity_matrix import check_matrix


def test_next_parity_matrix_covers_streamlit_page_registry() -> None:
    assert check_matrix() == []
