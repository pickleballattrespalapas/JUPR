import pandas as pd

from jupr_app.domain.match_admin import preview_week_tag_update


def test_preview_week_tag_update_collects_counts_and_tags():
    df = pd.DataFrame(
        [
            {"id": 1, "week_tag": "Week 1"},
            {"id": 2, "week_tag": "Week 1"},
            {"id": 3, "week_tag": "Week 2"},
        ]
    )

    preview = preview_week_tag_update(df, [1, 3], "Week 4")

    assert preview["count"] == 2
    assert preview["new_tag"] == "Week 4"
    assert preview["old_tags"] == ["Week 1", "Week 2"]
