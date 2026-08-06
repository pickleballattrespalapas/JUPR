from __future__ import annotations

import pytest

pytest.importorskip("postgrest")

from jupr_app.services.admin_tournament_setup_service import (
    _communication_acknowledgement_summary,
)


def communication_impact() -> dict:
    detail = {
        "impact_id": "division:event1:scheduled_day_ids:communication",
        "impact_type": "SCHEDULE_COMMUNICATION",
        "entity_label": "Men's 3.5+",
        "field": "scheduled_day_ids",
        "current_value": ["day1", "day2"],
        "proposed_value": ["day1"],
        "requires_acknowledgement": True,
        "affected_registrations": [
            {
                "registration_id": "r1",
                "selection_id": "s1",
                "display_name": "Alex Player",
                "email": "alex@example.com",
            }
        ],
    }
    return {"communication_impact_details": [detail]}


def test_communication_acknowledgement_is_required_and_fingerprint_bound() -> None:
    impact = communication_impact()
    detail = impact["communication_impact_details"][0]
    with pytest.raises(ValueError, match="acknowledge the communication impact"):
        _communication_acknowledgement_summary(impact, {})

    plan = {
        "impact_id": detail["impact_id"],
        "status": "ACKNOWLEDGED",
        "acknowledged": True,
        "action": "NOTIFY_AFFECTED",
        "current_value": detail["current_value"],
        "proposed_value": detail["proposed_value"],
        "affected_registrations": detail["affected_registrations"],
    }
    summary = _communication_acknowledgement_summary(
        impact,
        {"communication_change_acknowledgements": {detail["impact_id"]: plan}},
    )
    assert summary[0]["action"] == "NOTIFY_AFFECTED"
    assert summary[0]["affected_registration_count"] == 1

    stale_plan = {**plan, "proposed_value": ["day2"]}
    with pytest.raises(ValueError, match="proposed value changed"):
        _communication_acknowledgement_summary(
            impact,
            {"communication_change_acknowledgements": {detail["impact_id"]: stale_plan}},
        )


def test_no_notice_acknowledgement_is_valid() -> None:
    impact = communication_impact()
    detail = impact["communication_impact_details"][0]
    plan = {
        "impact_id": detail["impact_id"],
        "status": "ACKNOWLEDGED",
        "acknowledged": True,
        "action": "ACKNOWLEDGE_NO_NOTICE",
        "current_value": detail["current_value"],
        "proposed_value": detail["proposed_value"],
        "affected_registrations": detail["affected_registrations"],
    }
    summary = _communication_acknowledgement_summary(
        impact,
        {"communication_change_acknowledgements": {detail["impact_id"]: plan}},
    )
    assert summary[0]["action"] == "ACKNOWLEDGE_NO_NOTICE"


def test_required_age_data_blocks_communication_acknowledgement() -> None:
    impact = communication_impact()
    detail = impact["communication_impact_details"][0]
    missing = {
        "registration_id": "r2",
        "selection_id": "s2",
        "display_name": "Missing Partner Age",
        "proposed_value": {
            "age_label": "Needs age information",
            "assignment_issue_type": "MISSING_AGE_DATA",
            "assignment_issue": "Complete partner age before assigning this entry to an age group.",
        },
    }
    detail["impact_type"] = "AGE_GROUPING_COMMUNICATION"
    detail["requires_data_completion"] = True
    detail["data_completion_registrations"] = [missing]
    detail["affected_registrations"] = [*detail["affected_registrations"], missing]
    plan = {
        "impact_id": detail["impact_id"],
        "status": "ACKNOWLEDGED",
        "acknowledged": True,
        "action": "NOTIFY_AFFECTED",
        "current_value": detail["current_value"],
        "proposed_value": detail["proposed_value"],
        "affected_registrations": detail["affected_registrations"],
    }

    with pytest.raises(ValueError, match="complete required age information"):
        _communication_acknowledgement_summary(
            impact,
            {"communication_change_acknowledgements": {detail["impact_id"]: plan}},
        )
