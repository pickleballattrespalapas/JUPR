import pandas as pd

from jupr_app.domain.live_ladder import compute_next_order_from_movement


def test_compute_next_order_from_movement_preserves_missing_players_from_current_order():
    current_order = [1, 2, 3, 4, 5, 6]
    movement_df = pd.DataFrame(
        {
            "court": [1, 1, 1, 2, 2],
            "player_id": [1, 2, 3, 4, 5],
        }
    )

    next_order = compute_next_order_from_movement(
        current_order=current_order,
        movement_df=movement_df,
        players_per_court=3,
    )

    assert set(next_order) == set(current_order)
    assert next_order[-1] == 6
