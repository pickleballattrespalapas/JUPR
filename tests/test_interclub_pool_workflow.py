"""HTTP workflow coverage with a stateful adapter; SQL atomicity is tested separately.

These checks exercise real route matching, validation, preview matching, profile
enrichment and subsequent reloads. The adapter only supplies database outcomes;
it does not stand in for the SQL transaction tests.
"""
from copy import deepcopy
from types import SimpleNamespace
from urllib.parse import urlsplit
from uuid import uuid4

import pytest

from services.api import interclub_player_pool_routes as routes
from tests.test_api_contract_interclub_player_pool import setup  # noqa: F401


@pytest.fixture
def workflow(setup):
    client, state = setup
    state["tables"][routes.MEMBERS].clear()
    state["tables"]["players"].extend([
        dict(id=3, club_id="alpha", name="Taylor Shared", rating=1440, gender="female", active=True),
        dict(id=4, club_id="alpha", name="Taylor Shared", rating=1640, gender="male", active=True),
        dict(id=5, club_id="alpha", name="Sam Verbal", rating=1452, gender="male", active=True),
    ])
    state["tables"]["players"][0].update(rating=1524, gender="female")
    state["details"] = {
        1: dict(player_id=1, league_rating=4.02, eligible_divisions=["4.0"]),
        3: dict(player_id=3, league_rating=3.6, eligible_divisions=["3.5"]),
        4: dict(player_id=4, league_rating=4.1, eligible_divisions=["4.0"]),
        5: dict(player_id=5, league_rating=3.63, eligible_divisions=["3.5"]),
    }
    original_rpc = state["db"].rpc
    state["mutation"] = None

    def rpc(name, params):
        if name == "pcs_interclub_pool_search_players":
            state["calls"].append((name, params))
            query = " ".join(params["p_query"].split()).casefold()
            return SimpleNamespace(execute=lambda: SimpleNamespace(data=[
                {key: row.get(key) for key in ("id", "name", "rating", "gender")}
                for row in state["tables"]["players"]
                if row.get("active") and row["club_id"] == params["p_club_id"]
                and query in " ".join(row["name"].split()).casefold()
            ]))
        if name == "pcs_interclub_pool_player_details":
            state["calls"].append((name, params))
            return SimpleNamespace(execute=lambda: SimpleNamespace(data=[
                deepcopy(state["details"][pid]) for pid in params["p_player_ids"]
                if pid in state["details"]
            ]))
        if name == "pcs_interclub_pool_bulk_add" and state["mutation"]:
            state["calls"].append((name, params))
            return SimpleNamespace(execute=lambda: SimpleNamespace(data=state["mutation"](params)))
        return original_rpc(name, params)

    state["db"].rpc = rpc
    return client, state


def bulk_calls(state):
    return [params for name, params in state["calls"] if name == "pcs_interclub_pool_bulk_add"]


def test_mixed_admin_batch_preview_add_and_reload_keep_profile_ratings_and_pending_approval(workflow):
    client, state = workflow
    body = {"members": [
        {"player_id": "1"},  # Existing profile, no email needed for a verbal commitment.
        {"name": "sam verbal"},
        {"name": "New Verbal Player", "notes": "Confirmed in person"},
    ]}
    preview = client.post(state["base"] + "/pool/bulk-preview", json=body)
    assert preview.status_code == 200
    rows = preview.json()["rows"]
    assert [row["status"] for row in rows] == ["matched", "matched", "new"]
    assert [row["name"] for row in rows] == ["Alex Alpha", "Sam Verbal", "New Verbal Player"]
    assert rows[0]["rating"] == 3.81
    assert rows[0]["league_rating"] == 4.02
    assert rows[0]["eligible_divisions"] == ["4.0"]
    assert all(row["email"] == "" for row in rows)
    assert not bulk_calls(state), "Preview must not add players or assert meet availability."

    directory_before = deepcopy(state["tables"]["players"])
    responses_before = deepcopy(state["tables"][routes.RESPONSES])

    def add_transaction(params):
        assert params["p_club_id"] == "alpha"
        assert params["p_season_id"] == state["season"]["id"]
        assert params["p_actor_id"] == state["user"].user_id
        assert [entry["player_id"] for entry in params["p_members"]] == ["1", "5", None]
        # Simulate an atomic database response after the season has started.
        for entry in params["p_members"]:
            state["tables"][routes.MEMBERS].append({
                **state["member"], **entry, "id": str(uuid4()), "revision": 1,
                "player_id": int(entry["player_id"]) if entry["player_id"] else None,
                "late_join": True, "approval_status": "pending", "approval_reason": None,
                "token_nonce": str(uuid4()),
            })
        return {"added_count": 3, "skipped_count": 0}

    state["mutation"] = add_transaction
    saved = client.post(state["base"] + "/pool/bulk-add", json=body)
    assert saved.status_code == 200
    assert saved.json()["added_count"] == 3 and saved.json()["skipped_count"] == 0
    pool = saved.json()["pool"]
    assert len(pool["members"]) == 3
    assert all(member["approval_status"] == "pending" and member["late_join"] for member in pool["members"])
    assert pool["members"][0]["rating"] == 3.81
    assert pool["members"][0]["league_rating"] == 4.02
    assert pool["members"][0]["eligible_divisions"] == ["4.0"]
    assert pool["members"][2]["rating"] is None
    assert "token_nonce" not in saved.text
    assert all(urlsplit(member["manage_url"]).fragment.startswith("token=") for member in pool["members"])

    reloaded = client.get(state["base"] + "/pool")
    assert reloaded.status_code == 200
    assert reloaded.json()["members"] == pool["members"]
    repeat = client.post(state["base"] + "/pool/bulk-preview", json=body)
    assert repeat.status_code == 200
    assert repeat.json()["duplicate_count"] == 3 and repeat.json()["ready_count"] == 0
    assert len(bulk_calls(state)) == 1
    assert state["tables"]["players"] == directory_before, "Pool intake never changes existing ratings."
    assert state["tables"][routes.RESPONSES] == responses_before, "Joining the pool is not a meet RSVP."


def test_ambiguous_names_block_commit_then_distinct_profiles_can_share_name_and_email(workflow):
    client, state = workflow
    body = {"members": [{"name": "Taylor Shared"}]}
    preview = client.post(state["base"] + "/pool/bulk-preview", json=body)
    assert preview.status_code == 200
    row = preview.json()["rows"][0]
    assert row["status"] == "ambiguous"
    assert {candidate["id"] for candidate in row["candidates"]} == {"3", "4"}
    assert {candidate["rating"] for candidate in row["candidates"]} == {3.6, 4.1}
    assert client.post(state["base"] + "/pool/bulk-add", json=body).status_code == 422
    assert not bulk_calls(state)

    for email in (None, "household@example.test"):
        resolved = {"members": [{"name": "Taylor Shared", "player_id": str(pid),
                                  **({"email": email} if email else {})} for pid in (3, 4)]}
        preview = client.post(state["base"] + "/pool/bulk-preview", json=resolved)
        assert preview.status_code == 200
        assert [entry["status"] for entry in preview.json()["rows"]] == ["matched", "matched"], (
            "Explicitly choosing two distinct profiles must preserve two people, including without email."
        )
        assert preview.json()["ready_count"] == 2 and preview.json()["duplicate_count"] == 0


def test_public_candidate_and_admin_pool_use_same_club_and_rating_details(workflow):
    client, state = workflow
    signup = client.get(state["signup"] + "/players", params={"q": "Alex"})
    assert signup.status_code == 200
    assert signup.json()["linked_player"] is None
    candidate = signup.json()["players"][0]
    assert candidate["id"] == "1" and candidate["rating"] == 3.81
    assert candidate["league_rating"] == 4.02 and candidate["eligible_divisions"] == ["4.0"]
    assert "email" not in signup.text and "private" not in signup.text

    member = {**state["member"], "player_id": 1, "approval_status": "approved", "late_join": False}
    state["tables"][routes.MEMBERS].append(member)
    pool = client.get(state["base"] + "/pool")
    assert pool.status_code == 200
    stored = pool.json()["members"][0]
    assert {key: stored[key] for key in ("rating", "league_rating", "gender", "eligible_divisions")} == {
        key: candidate[key] for key in ("rating", "league_rating", "gender", "eligible_divisions")
    }
    forbidden = client.get(state["signup"] + "/players", params={"q": "Private Beta"})
    assert forbidden.status_code == 200 and forbidden.json()["players"] == []
    assert signup.headers["cache-control"] == "no-store"
    assert signup.headers["referrer-policy"] == "no-referrer"


def test_same_profile_repeated_with_different_names_and_emails_still_skips_duplicate(workflow):
    client, state = workflow
    response = client.post(state["base"] + "/pool/bulk-preview", json={"members": [
        {"player_id": "1", "name": "Alex", "email": "one@example.test"},
        {"player_id": "1", "name": "Alex Alpha", "email": "two@example.test"},
        {"name": "New Household One", "email": "shared@example.test"},
        {"name": "New Household Two", "email": "shared@example.test"},
    ]})
    assert response.status_code == 200
    assert [row["status"] for row in response.json()["rows"]] == ["matched", "duplicate", "new", "new"]
    assert response.json()["ready_count"] == 3 and response.json()["duplicate_count"] == 1
    assert not bulk_calls(state)


def test_admin_can_explicitly_decline_profile_matching_for_a_namesake(workflow):
    client, state = workflow
    response = client.post(state["base"] + "/pool/bulk-preview", json={"members": [
        {"name": "Taylor Shared", "player_id": None},
    ]})
    assert response.status_code == 200
    row = response.json()["rows"][0]
    assert row["status"] == "new" and row["player_id"] is None
    assert row["candidates"] == [] and row["rating"] is None
    assert response.json()["ambiguous_count"] == 0
    assert not bulk_calls(state)


def test_bulk_contact_inheritance_respects_explicit_email_and_verified_contact_preferences(workflow):
    client, state = workflow
    contact = dict(club_id="alpha", player_id=1, request_status="active",
                   email="Alex@Example.Test", email_normalized="alex@example.test",
                   verified_at="2026-01-01T00:00:00Z", unsubscribed_at=None,
                   preferences_json={})

    def preview_email(contacts, fields=None):
        state["tables"]["player_profile_update_subscriptions"] = contacts
        response = client.post(state["base"] + "/pool/bulk-preview", json={"members": [
            {"player_id": "1", **(fields or {})},
        ]})
        assert response.status_code == 200
        return response.json()["rows"][0]["email"]

    assert preview_email([contact]) == "alex@example.test"
    assert preview_email([contact], {"email": ""}) == ""
    assert preview_email([contact], {"email": None}) == ""
    assert preview_email([contact], {"email": "new@example.test"}) == "new@example.test"
    for patch in (
        {"verified_at": None},
        {"unsubscribed_at": "2026-02-01T00:00:00Z"},
        {"preferences_json": {"optional_emails_enabled": False}},
        {"preferences_json": {"unsubscribe_scope": "global"}},
        {"request_status": "pending"},
        {"club_id": "beta"},
    ):
        assert preview_email([{**contact, **patch}]) == "", patch
    assert preview_email([contact, {**contact, "email_normalized": "other@example.test"}]) == ""
    assert not bulk_calls(state), "Looking up saved contact information must not send or add anything."
