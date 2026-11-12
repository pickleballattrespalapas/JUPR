from copy import deepcopy
from datetime import date, datetime
from types import SimpleNamespace
import re

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from jupr_app.domain.adaptive_play_engine import (
    create_generator_preview, start_generator_event, save_generator_round,
    complete_generator_event, generator_event_standings,
)
from jupr_app.services.round_robin_history_service import list_round_robin_history, get_round_robin_history
from services.api import public_round_robin_history_routes as routes


def field(row, key):
    value = row
    for part in re.split(r"->>?", key):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def split_terms(expression):
    terms, start, depth = [], 0, 0
    for i, char in enumerate(expression):
        depth += (char == "(") - (char == ")")
        if char == "," and depth == 0:
            terms.append(expression[start:i]); start = i + 1
    return [*terms, expression[start:]]


def predicate(row, expression):
    for operator in ("and", "or"):
        if expression.startswith(operator + "("):
            values = [predicate(row, term) for term in split_terms(expression[len(operator) + 1:-1])]
            return all(values) if operator == "and" else any(values)
    key, op, expected = expression.split(".", 2)
    value = field(row, key)
    if op == "not":
        return not predicate(row, f"{key}.{expected}")
    if op == "is": return value is None if expected == "null" else str(value).lower() == expected
    if value is None: return False  # SQL comparisons with NULL are not true.
    return (str(value).lower() == expected) if op == "eq" else (str(value).lower() != expected)


class Query:
    def __init__(self, db):
        self.db, self.tests, self.orders = db, [], []
        self.columns, self.start, self.end = "*", 0, 10000
    def select(self, columns): self.columns = columns; return self
    def eq(self, key, value): self.tests.append(lambda r: field(r, key) == value); return self
    def in_(self, key, values): self.tests.append(lambda r: field(r, key) in values); return self
    def or_(self, expression): self.tests.append(lambda r: predicate(r, f"or({expression})")); return self
    def gte(self, key, value): self.tests.append(lambda r: datetime.fromisoformat(field(r, key)) >= datetime.fromisoformat(value)); return self
    def lt(self, key, value): self.tests.append(lambda r: datetime.fromisoformat(field(r, key)) < datetime.fromisoformat(value)); return self
    def order(self, key, desc=False): self.orders.append((key, desc)); return self
    def range(self, start, end): self.start, self.end = start, end; return self
    def limit(self, count): return self.range(0, count - 1)
    def execute(self):
        if self.db.fail: raise RuntimeError("private database error")
        self.db.reads.append(self.columns)
        rows = [r for r in self.db.rows if all(test(r) for test in self.tests)]
        for key, desc in reversed(self.orders): rows.sort(key=lambda r: str(field(r, key) or ""), reverse=desc)
        rows = rows[self.start:self.end + 1]
        def project(row):
            return {(col.split(":", 1)[0] if ":" in col else col): field(row, col.split(":", 1)[-1])
                    for col in self.columns.split(",")}
        return SimpleNamespace(data=[project(r) for r in rows])


class DB:
    def __init__(self, rows): self.rows, self.reads, self.fail = rows, [], False
    def table(self, name): assert name == "live_sessions"; return Query(self)


@pytest.fixture
def row():
    event = start_generator_event(create_generator_preview(
        generator_kind="round_robin", play_format="doubles", title="Tuesday round robin",
        participant_names=["A", "B", "C", "D"], total_rounds=1))
    match = event["rounds"][0]["matches"][0]
    event = save_generator_round(event, round_number=1,
        scores=[{"match_id": match["id"], "score_a": 11, "score_b": 0}])
    event = complete_generator_event(event)
    return {"club_id": "club", "session_key": "rr-one", "title": event["name"],
        "status": "completed", "created_at": "2026-11-10T18:00:00+00:00", "completed_at": "2026-11-10T20:00:00+00:00",
        "created_by_email": "organizer@example.test", "edit_token_hash": "private-edit-hash",
        "state": {"mode": "admin_play_generator", "generator_kind": "round_robin", "play_format": "doubles",
                  "generator_sharing": {"enabled": True, "handoff_nonce": "private-handoff"},
                  "page_state": {"event": event, "participant_count": 4}}}


def test_archive_visibility_is_applied_before_pagination(row):
    hidden = []
    for change in ["club", "active", "archived", "private", "ladder", "draft", "other-mode"]:
        item = deepcopy(row); item["session_key"] = change
        if change == "club": item["club_id"] = "other"
        elif change in {"active", "archived"}: item["status"] = change
        elif change == "private": item["state"].pop("generator_sharing")
        elif change == "ladder": item["state"]["generator_kind"] = "ladder"
        elif change == "draft": item["state"]["page_state"]["event"]["hubSupplementalDraft"] = True
        else: item["state"]["mode"] = "public_club_social"
        hidden.append(item)
    approved = deepcopy(row); approved["session_key"] = "rr-two"
    approved["state"].pop("generator_sharing"); approved["state"]["generator_submission"] = {"status": "approved"}
    public = deepcopy(row); public["session_key"] = "rr-three"
    public["state"].pop("generator_sharing"); public["state"]["mode"] = "public_play_generator"
    published = deepcopy(row); published["session_key"] = "rr-four"
    published["state"].pop("generator_sharing"); published["state"]["official_publish"] = {"published_at": "2026-11-10"}
    db = DB([*hidden, row, approved, public, published])
    result = list_round_robin_history(db, club_id="club", limit=2)
    assert len(result["sessions"]) == 2 and result["has_more"]
    second = list_round_robin_history(db, club_id="club", offset=2, limit=2)
    assert not second["has_more"]
    assert {r["session_key"] for r in result["sessions"] + second["sessions"]} == {"rr-one", "rr-two", "rr-three", "rr-four"}
    assert all("state," not in selection and "*" not in selection for selection in db.reads)
    for item in hidden:
        with pytest.raises(LookupError): get_round_robin_history(db, club_id="club", session_key=item["session_key"])


def test_results_preserve_scores_standings_and_omit_secrets(row):
    original = deepcopy(row)
    result = get_round_robin_history(DB([row]), club_id="club", session_key="rr-one")["session"]
    assert result["standings"] == generator_event_standings(row["state"]["page_state"]["event"])
    assert result["rounds"][0]["matches"][0]["score_b"] == 0
    assert sorted(result["rounds"][0]["matches"][0]["side_a"] + result["rounds"][0]["matches"][0]["side_b"]) == ["A", "B", "C", "D"]
    assert not any(secret in str(result) for secret in ["private-edit-hash", "private-handoff", "organizer@example.test", "generator_sharing", "moneyballPrivate"])
    assert row == original
    # Nested court schedules have the same public score projection.
    event_round = row["state"]["page_state"]["event"]["rounds"][0]
    event_round["courts"] = [{"matches": event_round.pop("matches")}]
    assert get_round_robin_history(DB([row]), club_id="club", session_key="rr-one")["session"]["rounds"] == result["rounds"]


def test_type_dates_and_errors(row):
    moneyball = deepcopy(row); moneyball["session_key"] = "mb"
    moneyball["state"]["page_state"]["event"]["eventFormat"] = "moneyball_expectations"
    db = DB([row, moneyball])
    assert list_round_robin_history(db, club_id="club", kind="moneyball")["sessions"][0]["session_key"] == "mb"
    assert list_round_robin_history(db, club_id="club", kind="round_robin")["sessions"][0]["session_key"] == "rr-one"
    assert not list_round_robin_history(db, club_id="club", from_date=date(2026, 11, 11))["sessions"]
    assert len(list_round_robin_history(db, club_id="club", from_date=date(2026, 11, 10), to_date=date(2026, 11, 10))["sessions"]) == 2
    early = deepcopy(row); early["created_at"] = "2026-11-10T02:00:00+00:00"
    assert not list_round_robin_history(DB([early]), club_id="club", from_date=date(2026, 11, 10))["sessions"]
    assert list_round_robin_history(DB([early]), club_id="club", to_date=date(2026, 11, 9))["sessions"]
    with pytest.raises(ValueError): list_round_robin_history(db, club_id="club", from_date=date(2026, 11, 11), to_date=date(2026, 11, 10))
    db.fail = True
    with pytest.raises(RuntimeError): list_round_robin_history(db, club_id="club")


def test_moneyball_uses_frozen_scoring_and_keeps_private_setup_off_the_wire(row):
    from jupr_app.domain.moneyball_expectations import freeze_moneyball
    from test_moneyball_expectations import preview, save
    event = preview()
    event = start_generator_event(freeze_moneyball(event, {p["id"]: 3 + i * .15 for i, p in enumerate(event["participants"])}))
    for number in range(1, 8): event = save(event, number)
    event = complete_generator_event(event)
    row["state"]["page_state"]["event"] = event
    result = get_round_robin_history(DB([row]), club_id="club", session_key="rr-one")["session"]
    assert result["kind"] == "moneyball" and result["standings_sort"] == "moneyball_expectations"
    assert result["standings"] == generator_event_standings(event)
    assert all(standing["moneyballFinal"] for standing in result["standings"])
    assert "moneyballPrivate" not in str(result) and "lock_sha256" not in str(result)


def test_routes_are_read_only_require_published_club_and_fail_clearly(row, monkeypatch):
    db = DB([row])
    def shell(_db, slug):
        if slug != "alpha": raise HTTPException(404, "Club not published")
        return {"club_id": "club", "document": {"visibility": "unlisted"}}
    monkeypatch.setattr(routes, "published_shell", shell)
    app = FastAPI(); routes.install_public_round_robin_history_routes(app, get_supabase_client=lambda: db)
    client = TestClient(app); root = "/clubs/alpha/round-robins"
    assert client.get(root).status_code == 200
    assert client.get(root).headers["cache-control"] == "no-store"
    assert "noindex" in client.get(root).headers["x-robots-tag"]
    assert client.get(root + "/rr-one").status_code == 200
    assert client.get(root + "/missing").status_code == 404
    assert client.get("/clubs/unpublished/round-robins").status_code == 404
    for query in ["kind=other", "offset=-1", "limit=1000", "from_date=bad", "from_date=2026-11-11&to_date=2026-11-10"]:
        assert client.get(root + "?" + query).status_code == 422
    for action in ["scores", "submit", "publish", "sharing"]:
        assert client.post(root + "/rr-one/" + action, json={}).status_code == 404
    db.fail = True
    assert client.get(root).status_code == 503
    assert "private database error" not in client.get(root).text
