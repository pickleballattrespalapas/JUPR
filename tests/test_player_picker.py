import pandas as pd

from jupr_app.ui.components.player_picker import build_player_picker_df, filter_player_picker_df


def test_build_player_picker_df_alphabetized_and_display_name_preferred():
    df = pd.DataFrame(
        [
            {"id": 2, "name": "zoe alpha", "display_name": ""},
            {"id": 1, "name": "robert novotny", "display_name": "Rob Novotny"},
            {"id": 3, "name": "amy bee"},
        ]
    )
    out = build_player_picker_df(df)
    assert out["id"].tolist() == [3, 1, 2]
    assert out.loc[out["id"] == 1, "option_label"].iloc[0] == "Rob Novotny (#1)"


def test_filter_player_picker_df_case_insensitive_and_multi_token():
    df = pd.DataFrame(
        [
            {"id": 1, "display_label": "Rob Novotny", "option_label": "Rob Novotny (#1)", "search_text": "rob novotny", "sort_label": "rob novotny"},
            {"id": 2, "display_label": "Robin Smith", "option_label": "Robin Smith (#2)", "search_text": "robin smith", "sort_label": "robin smith"},
        ]
    )
    out = filter_player_picker_df(df, "ROB nov")
    assert out["id"].tolist() == [1]


def test_duplicate_names_distinguished_by_id():
    df = pd.DataFrame(
        [
            {"id": 1, "name": "Alex Kim"},
            {"id": 2, "name": "Alex Kim"},
        ]
    )
    out = build_player_picker_df(df)
    assert "Alex Kim (#1)" in out["option_label"].tolist()
    assert "Alex Kim (#2)" in out["option_label"].tolist()


def test_blank_query_returns_all_rows():
    df = pd.DataFrame([{"id": 1, "name": "A"}, {"id": 2, "name": "B"}])
    built = build_player_picker_df(df)
    out = filter_player_picker_df(built, "")
    assert out["id"].tolist() == [1, 2]
