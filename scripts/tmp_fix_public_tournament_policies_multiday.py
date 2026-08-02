from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one match, found {count}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


for service_path in (
    "jupr_app/services/public_tournament_registration_service.py",
    "jupr_app/services/public_tournament_roster_service.py",
):
    replace_once(
        service_path,
        '''        "refund_policy_markdown": _clean_text(row.get("refund_policy_markdown"), limit=4000),
        "sponsor_markdown": _clean_text(row.get("sponsor_markdown"), limit=4000),
''',
        '''        "refund_policy_markdown": _clean_text(row.get("refund_policy_markdown"), limit=4000),
        "weather_policy_markdown": _clean_text(row.get("weather_policy_markdown"), limit=4000),
        "sponsor_markdown": _clean_text(row.get("sponsor_markdown"), limit=4000),
''',
    ) if "registration_service" in service_path else None

# The roster settings projection intentionally exposes the same public policies.
replace_once(
    "jupr_app/services/public_tournament_roster_service.py",
    '''        "partner_board_enabled": _safe_bool(row.get("partner_board_enabled")),
    }
''',
    '''        "partner_board_enabled": _safe_bool(row.get("partner_board_enabled")),
        "rules_markdown": _clean_text(row.get("rules_markdown"), limit=4000),
        "refund_policy_markdown": _clean_text(row.get("refund_policy_markdown"), limit=4000),
        "weather_policy_markdown": _clean_text(row.get("weather_policy_markdown"), limit=4000),
        "sponsor_markdown": _clean_text(row.get("sponsor_markdown"), limit=4000),
    }
''',
)

for service_path in (
    "jupr_app/services/public_tournament_registration_service.py",
    "jupr_app/services/public_tournament_roster_service.py",
):
    replace_once(
        service_path,
        '''        "registration_day_id": str(row.get("registration_day_id") or ""),
        "label": _clean_text(row.get("label") or row.get("division_name") or "Division", limit=160),
''',
        '''        "registration_day_id": str(row.get("registration_day_id") or ""),
        "scheduled_day_ids": [
            _clean_text(value, limit=160)
            for value in (
                row.get("scheduled_day_ids")
                if isinstance(row.get("scheduled_day_ids"), list)
                else []
            )
            if _clean_text(value, limit=160)
        ]
        or [str(row.get("registration_day_id") or "")],
        "label": _clean_text(row.get("label") or row.get("division_name") or "Division", limit=160),
''',
    )

replace_once(
    "tests/test_public_tournament_registration_service.py",
    '''                "refund_policy_markdown": "No refunds after draw publication.",
                "sponsor_markdown": "Presented by Rally House.",
''',
    '''                "refund_policy_markdown": "No refunds after draw publication.",
                "weather_policy_markdown": "Unsafe conditions may delay or reschedule play.",
                "sponsor_markdown": "Presented by Rally House.",
''',
)
replace_once(
    "tests/test_public_tournament_registration_service.py",
    '''                "registration_day_id": "day1",
                "sort_order": 1,
''',
    '''                "registration_day_id": "day1",
                "scheduled_day_ids": ["day1", "day2"],
                "sort_order": 1,
''',
)
replace_once(
    "tests/test_public_tournament_registration_service.py",
    '''    assert payload["settings"]["sponsor_markdown"] == "Presented by Rally House."
''',
    '''    assert payload["settings"]["sponsor_markdown"] == "Presented by Rally House."
    assert payload["settings"]["weather_policy_markdown"] == "Unsafe conditions may delay or reschedule play."
    assert payload["events"][0]["scheduled_day_ids"] == ["day1", "day2"]
''',
)

replace_once(
    "tests/test_public_tournament_roster_service.py",
    '''    assert payload["settings"]["registration_close_at"] == "2026-08-25T23:00:00Z"
''',
    '''    assert payload["settings"]["registration_close_at"] == "2026-08-25T23:00:00Z"
    assert payload["settings"]["weather_policy_markdown"] == "Unsafe conditions may delay or reschedule play."
    assert payload["events"][0]["scheduled_day_ids"] == ["day1", "day2"]
''',
)

contract = ROOT / "tests/test_api_contract_tournament_setup_flow_policies_multiday.py"
text = contract.read_text(encoding="utf-8")
addition = '''


def test_public_services_project_weather_and_multi_day_fields() -> None:
    registration_service = (ROOT / "jupr_app/services/public_tournament_registration_service.py").read_text()
    roster_service = (ROOT / "jupr_app/services/public_tournament_roster_service.py").read_text()
    for source in (registration_service, roster_service):
        assert '"weather_policy_markdown"' in source
        assert '"scheduled_day_ids"' in source
'''
if "def test_public_services_project_weather_and_multi_day_fields" not in text:
    contract.write_text((text.rstrip() + addition).rstrip() + "\n", encoding="utf-8")
