from __future__ import annotations

from postgrest.exceptions import APIError

from jupr_app.domain.admin.role_assignments import (
    count_super_admin_assignments,
    delete_role_assignment,
    has_other_super_admin_support,
    is_role_table_missing_error,
    list_role_assignments,
    normalize_email,
    upsert_role_assignment,
)
from jupr_app.domain.admin.roles import ROLE_CLUB_OWNER, ROLE_SUPER_ADMIN, normalize_role, can_manage_roles


def test_normalize_email_and_role():
    assert normalize_email("  Joe@Example.COM ") == "joe@example.com"
    assert normalize_role(" organizer ") == "organizer"
    assert normalize_role("bad-role") == "read_only"


def test_only_super_admin_can_manage_roles():
    assert can_manage_roles(ROLE_SUPER_ADMIN) is True
    assert can_manage_roles(ROLE_CLUB_OWNER) is False


def test_missing_table_error_detection():
    assert is_role_table_missing_error(APIError({"message": "x", "code": "42P01"})) is True
    assert is_role_table_missing_error(APIError({"message": "x", "code": "PGRST205"})) is True


def test_super_admin_count_and_safety_checks():
    rows = [
        {"email": "a@example.com", "role": "super_admin"},
        {"email": "b@example.com", "role": "club_owner"},
    ]
    assert count_super_admin_assignments(rows) == 1
    assert has_other_super_admin_support(rows=rows, target_email="a@example.com", admin_allowlist=set()) is False
    assert has_other_super_admin_support(rows=rows, target_email="a@example.com", admin_allowlist={"joe@example.com"}) is True


class _Query:
    def __init__(self):
        self.filters = []
        self.payload = None
        self.on_conflict = None

    def select(self, *_args, **_kwargs):
        return self

    def order(self, *_args, **_kwargs):
        return self

    def eq(self, key, val):
        self.filters.append((key, val))
        return self

    def upsert(self, payload, on_conflict=None):
        self.payload = payload
        self.on_conflict = on_conflict
        return self

    def delete(self):
        return self

    def execute(self):
        return type("Resp", (), {"data": []})()


class _Supabase:
    def __init__(self):
        self.query = _Query()

    def table(self, _name):
        return self.query


def test_role_assignment_queries_are_club_scoped():
    sb = _Supabase()
    list_role_assignments(sb, "club-a")
    assert ("club_id", "club-a") in sb.query.filters

    sb = _Supabase()
    upsert_role_assignment(sb, "club-a", "a@example.com", "scorekeeper", user_id="u1")
    assert sb.query.payload["club_id"] == "club-a"
    assert sb.query.on_conflict == "club_id,email"

    sb = _Supabase()
    delete_role_assignment(sb, "club-a", "a@example.com")
    assert ("club_id", "club-a") in sb.query.filters
    assert ("email", "a@example.com") in sb.query.filters
