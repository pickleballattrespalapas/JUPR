import json
import pytest
from scripts.check_badge_worker_progress import verify


def event(**updates):
    return {"event": "badge.worker.cycle_completed", "deployment_sha": "a" * 40,
            "worker_id": "123:1:uuid", "program_errors": 0, "queue_errors": 0,
            "program_processed": 0, "queue_processed": 1, **updates}


def test_success_is_bound_to_machine_and_release_and_does_not_echo_private_fields():
    line = json.dumps({"message": json.dumps(event(payload="secret"))})
    result = verify(line, sha="a" * 40, machine_id="123")
    assert result["queue_processed"] == 1
    assert "secret" not in json.dumps(result)


@pytest.mark.parametrize("updates", [{"deployment_sha": "b" * 40}, {"worker_id": "999:1:uuid"},
                                    {"event": "badge.worker.cycle_failed"}, {"queue_errors": 1},
                                    {"program_errors": None}])
def test_running_or_failed_worker_is_not_ready(updates):
    with pytest.raises(ValueError):
        verify(json.dumps(event(**updates)), sha="a" * 40, machine_id="123")


def test_later_failure_invalidates_earlier_success():
    lines = "\n".join(map(json.dumps, [event(), event(event="badge.worker.cycle_failed")]))
    with pytest.raises(ValueError):
        verify(lines, sha="a" * 40, machine_id="123")
