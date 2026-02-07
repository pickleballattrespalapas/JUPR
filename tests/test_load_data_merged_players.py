import pandas as pd

from jupr_app.data.load import _drop_merged_players


def test_drop_merged_players_filters_marker():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": [
                "Alice",
                "Bob (MERGED into Alice #1)",
                None,
            ],
        }
    )

    filtered = _drop_merged_players(df)

    assert filtered["id"].tolist() == [1, 3]
    assert "Bob (MERGED into Alice #1)" not in filtered["name"].fillna("").tolist()
