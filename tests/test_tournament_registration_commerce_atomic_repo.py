from jupr_app.domain import tournament_registration_repo as repo


TOURNAMENT = "00000000-0000-4000-8000-000000000001"
DAY = "00000000-0000-4000-8000-000000000002"
EVENT = "00000000-0000-4000-8000-000000000003"
KEY_ONE = "00000000-0000-4000-8000-000000000004"
KEY_TWO = "00000000-0000-4000-8000-000000000005"


class _Response:
    def __init__(self, data):
        self.data = data


class _Rpc:
    def __init__(self, response):
        self.response = response

    def execute(self):
        return _Response(self.response)


class _Supabase:
    def __init__(self, response):
        self.response = response
        self.rpc_calls = []
        self.table_calls = []

    def rpc(self, name, params):
        self.rpc_calls.append((name, params))
        return _Rpc(self.response)

    def table(self, name):
        self.table_calls.append(name)
        raise AssertionError(f"non-atomic table write attempted: {name}")


def _patch_validation(monkeypatch, *, existing=None):
    monkeypatch.setattr(
        repo, "_get_existing_registration_by_email", lambda *_args: existing
    )
    monkeypatch.setattr(
        repo,
        "list_registration_days",
        lambda *_args: [{"id": DAY, "enabled": True}],
    )
    monkeypatch.setattr(
        repo,
        "list_event_options",
        lambda *_args: [
            {
                "id": EVENT,
                "registration_day_id": DAY,
                "status": "open",
                "enabled": True,
                "division_name": "Open",
            }
        ],
    )
    monkeypatch.setattr(
        repo, "validate_selection_against_skill", lambda **_kwargs: (True, None)
    )


def _payload():
    return {
        "_registration_id": "reg_atomic",
        "display_name": "Atomic Player",
        "email": "atomic@example.com",
        "selections": [
            {
                "event_option_id": EVENT,
                "registration_day_id": DAY,
                "partner_mode": "NONE",
            }
        ],
    }


def _commerce():
    return {
        "club_id": "club",
        "quote": {
            "quote_fingerprint": "server-quote",
            "request_fingerprint": "server-request",
        },
        "operation_idempotency_key": KEY_ONE,
        "order_idempotency_key": KEY_TWO,
        "request_fingerprint": "atomic-request",
        "actor_label": "a***@example.com",
        "source": "test",
    }


def test_new_registration_and_commerce_use_one_atomic_rpc(monkeypatch):
    _patch_validation(monkeypatch)
    supabase = _Supabase(
        {
            "ok": True,
            "registration_id": "reg_atomic",
            "selection_count": 1,
            "commerce_order": {"status": "ACTIVE"},
        }
    )

    result = repo.save_registration(
        supabase,
        tournament_id=TOURNAMENT,
        payload=_payload(),
        commerce_transaction=_commerce(),
    )

    assert result["registration_id"] == "reg_atomic"
    assert result["commerce_order"]["status"] == "ACTIVE"
    assert len(supabase.rpc_calls) == 1
    rpc_name, params = supabase.rpc_calls[0]
    assert rpc_name == repo.PUBLIC_REGISTRATION_COMMERCE_CREATE_RPC
    assert params["p_quote_snapshot"]["quote_fingerprint"] == "server-quote"
    assert params["p_operation_idempotency_key"] == KEY_ONE
    assert not supabase.table_calls


def test_registration_edit_and_commerce_use_one_version_checked_rpc(
    monkeypatch,
):
    existing = {
        "id": "reg_atomic",
        "tournament_id": TOURNAMENT,
        "email": "atomic@example.com",
        "status": "confirmed",
        "submitted_at": "2026-07-01T00:00:00Z",
    }
    _patch_validation(monkeypatch, existing=existing)
    monkeypatch.setattr(
        repo, "get_registration_by_id", lambda *_args: existing
    )
    supabase = _Supabase(
        {
            "ok": True,
            "registration_id": "reg_atomic",
            "updated_at": "2026-07-02T00:00:00Z",
            "selection_count": 1,
            "commerce_order": {"current_revision": 2},
            "idempotent_replay": False,
        }
    )
    commerce = {
        **_commerce(),
        "expected_order_updated_at": "2026-07-01T00:00:00Z",
    }

    result = repo.save_registration(
        supabase,
        tournament_id=TOURNAMENT,
        payload=_payload(),
        expected_registration_id="reg_atomic",
        expected_updated_at="2026-07-01T00:00:00Z",
        expected_selection_versions=[],
        atomic_edit=True,
        commerce_transaction=commerce,
    )

    assert result["commerce_order"]["current_revision"] == 2
    rpc_name, params = supabase.rpc_calls[0]
    assert rpc_name == repo.PUBLIC_REGISTRATION_COMMERCE_EDIT_RPC
    assert params["p_expected_registration_updated_at"] == (
        "2026-07-01T00:00:00Z"
    )
    assert params["p_expected_order_updated_at"] == "2026-07-01T00:00:00Z"
    assert params["p_quote_snapshot"]["quote_fingerprint"] == "server-quote"
