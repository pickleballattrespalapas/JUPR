from __future__ import annotations

import pandas as pd

import streamlit_sortable


def test_sort_items_uses_drag_order(monkeypatch):
    items = ["A", "B", "C"]

    def fake_court_board(_payload, key=None):
        return {
            "courts": [
                {
                    "court_id": "Court 1",
                    "players": [
                        {"player_id": "2", "name": "C"},
                        {"player_id": "0", "name": "A"},
                        {"player_id": "1", "name": "B"},
                    ],
                }
            ]
        }

    monkeypatch.setattr(streamlit_sortable, "court_board", fake_court_board)

    result = streamlit_sortable.sort_items(items, key="test")

    assert result == ["C", "A", "B"]


def test_sort_items_falls_back_to_rank_editor(monkeypatch):
    items = ["A", "B", "C"]

    def raising_court_board(_payload, key=None):
        raise RuntimeError("component unavailable")

    def fake_data_editor(df, **_kwargs):
        assert list(df["Player"]) == items
        return pd.DataFrame(
            [
                {"Rank": 2, "Player": "A"},
                {"Rank": 1, "Player": "B"},
                {"Rank": 3, "Player": "C"},
            ]
        )

    monkeypatch.setattr(streamlit_sortable, "court_board", raising_court_board)
    monkeypatch.setattr(streamlit_sortable.st, "caption", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(streamlit_sortable.st, "data_editor", fake_data_editor)
    monkeypatch.setattr(streamlit_sortable.st, "column_config", streamlit_sortable.st.column_config)

    result = streamlit_sortable.sort_items(items, key="test")

    assert result == ["B", "A", "C"]
