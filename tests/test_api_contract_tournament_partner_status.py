from jupr_app.domain import tournament_partner_service as svc
from tests.test_tournament_partner_requests import _FakeQuery, _FakeSupabase, _storage


def test_confirming_legacy_partner_finishes_without_resubmitting_manual_identity(monkeypatch):
    storage = _storage()
    first, second = storage["tournament_registration_selections"][:2]
    first.update(partner_name=None, partner_email=None)
    second.update(partner_mode="HAS_PARTNER", partner_name="Mary Bauman", partner_email=None, show_on_partner_board=True)
    execute = _FakeQuery.execute
    writes = []

    def with_manual_partner_trigger(query):
        if query.table_name == "tournament_registration_selections" and query._update_payload is not None:
            for row in query.storage[query.table_name]:
                if not all(str(row.get(k)) == str(v) for k, v in query.filters):
                    continue
                writes.append((row["id"], dict(query._update_payload)))
                # PostgreSQL's UPDATE OF trigger fires even when HAS_PARTNER
                # is assigned its current value. The legacy email is absent.
                if query._update_payload.get("partner_mode") == "HAS_PARTNER" and row.get("partner_name") and not row.get("partner_email"):
                    raise RuntimeError("JUPR_MANUAL_PARTNER_EMAIL_REQUIRED")
        return execute(query)

    monkeypatch.setattr(_FakeQuery, "execute", with_manual_partner_trigger)
    result = svc.admin_replace_partner_link(
        _FakeSupabase(storage), tournament_id="tour-1", event_option_id="event-wd-35",
        selection_id=first["id"], partner_selection_id=second["id"], unpaired_mode="NONE",
        admin_user_id="admin@example.com",
    )
    assert result["outcome"] == "paired"
    assert len(storage["tournament_registration_team_links"]) == 1
    assert len(storage["tournament_registration_team_members"]) == 2
    assert first["partner_mode"] == second["partner_mode"] == "HAS_PARTNER"
    assert first["show_on_partner_board"] is second["show_on_partner_board"] is False
    assert second["partner_name"] == "Mary Bauman"
    assert second["partner_email"] is None
    assert (second["id"], {"show_on_partner_board": False}) in writes

    before = len(writes)
    svc._update_selection_partner_mode(_FakeSupabase(storage), selection_id=second["id"], partner_mode="HAS_PARTNER")
    assert len(writes) == before, "An already complete partner state must not write again"
