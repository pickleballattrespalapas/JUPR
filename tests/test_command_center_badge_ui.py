from jupr_app.ui.pages import command_center


def test_build_condition_payload_numeric():
    payload = command_center._build_condition_payload(
        {
            "fact_key": "matches_played",
            "operator": ">=",
            "value_type": "numeric",
            "value_numeric": 10,
        }
    )

    assert payload == {
        "fact_key": "matches_played",
        "operator": ">=",
        "value_numeric": 10.0,
        "value_boolean": None,
    }


def test_build_condition_payload_boolean():
    payload = command_center._build_condition_payload(
        {
            "fact_key": "is_champion",
            "operator": "is",
            "value_type": "boolean",
            "value_boolean": True,
        }
    )

    assert payload == {
        "fact_key": "is_champion",
        "operator": "is",
        "value_numeric": None,
        "value_boolean": True,
    }


def test_build_condition_payload_rejects_invalid_numeric():
    payload = command_center._build_condition_payload(
        {
            "fact_key": "matches_played",
            "operator": ">=",
            "value_type": "boolean",
            "value_boolean": True,
        }
    )

    assert payload is None
