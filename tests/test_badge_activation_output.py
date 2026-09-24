import json

import pytest

from scripts.parse_badge_activation_output import ACTIVATION, parse_activation_output


def test_fly_machine_selection_banner_does_not_hide_successful_activation():
    result = {"activation": ACTIVATION, "ok": True, "existing_awards_unchanged": True,
              "classic_inserted": 671, "total_awards_after": 7894}
    output = "No machine specified, using 2872960c342798 in region dfw\r\n" + json.dumps(result) + "\r\n"
    assert parse_activation_output(output) == result


def test_completed_activation_is_recognized_without_repeating_awards():
    result = {"activation": ACTIVATION, "ok": True, "already_applied": True}
    assert parse_activation_output(json.dumps(result)) == result


@pytest.mark.parametrize("output", ["", "Connecting to machine", "{}", json.dumps({"activation": ACTIVATION, "ok": False}),
    json.dumps({"activation": "different-release", "ok": True, "already_applied": True}),
    json.dumps({"activation": ACTIVATION, "ok": True}), "{}\n{}"])
def test_missing_failed_ambiguous_or_unpreserved_results_are_rejected(output):
    with pytest.raises(ValueError):
        parse_activation_output(output)
