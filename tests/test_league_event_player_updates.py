from copy import deepcopy
from datetime import date
from types import SimpleNamespace

import pytest

from jupr_app.domain.notifications import league_event_updates as events
from jupr_app.domain.notifications import player_update_sender as sender
from jupr_app.domain.notifications.player_update_email_template import build_player_update_email_text
from jupr_app.domain.recaps.player_weekly_digest import compute_player_weekly_digest
from tests.test_admin_match_log_service import FakeSupabase
from tests.test_player_session_digest import DB, context, game
from tests.test_player_update_session_scope import sender_case  # noqa: F401


def _round(number, ids, *, session_id="event-a", club_id="club-1", players=(1, 2, 3, 4)):
    matches = [{"id": mid, "club_id": club_id, "date": "2026-11-04",
        "context_type": "league_live_session", "context_id": f"{session_id}-{mid}",
        "match_format": "doubles", "rating_scope": "both", "deleted_at": None,
        **dict(zip(("t1_p1", "t1_p2", "t2_p1", "t2_p2"), players))} for mid in ids]
    operation = {"id": f"{session_id}-round-{number}", "club_id": club_id, "session_id": session_id,
        "round_number": number, "status": "completed", "completion_audited_at": "2026-11-04T19:00:00Z",
        "published_match_ids": ids, "match_context_ids": [row["context_id"] for row in matches],
        "request_json": {"matches": matches},
        "rating_before_json": [{"id": pid, "matches_played": (number - 1) * 3} for pid in players],
        "rating_after_json": [{"id": pid, "matches_played": number * 3} for pid in players]}
    return operation, matches


def _database(*, rounds=1, total=2):
    sb = FakeSupabase({"league_live_sessions": [{"id": "event-a", "club_id": "club-1",
        "status": "active", "total_rounds": total, "completed_at": None}],
        "league_live_publish_operations": [], "matches": [], "badge_eval_queue": [],
        "player_profile_update_subscriptions": [{"id": f"sub-{pid}", "club_id": "club-1",
            "player_id": pid, "email": f"player-{pid}@example.test", "request_status": "active"}
            for pid in range(1, 6)]})
    for number in range(1, rounds + 1):
        _add_round(sb, number, list(range(number * 100, number * 100 + 3)))
    return sb


def _add_round(sb, number, ids, **kwargs):
    operation, matches = _round(number, ids, **kwargs)
    sb.tables["league_live_publish_operations"].append(operation)
    sb.tables["matches"].extend(matches)
    sb.tables["badge_eval_queue"].append({"id": f"badge-{ids[0]}", "club_id": operation["club_id"],
        "event_type": "match_recorded", "match_id": str(ids[0]), "status": "done", "attempts": 1})
    return operation


def _queue(sb, session_id="event-a"):
    return events.queue_league_event_player_updates(sb, club_id="club-1", session_id=session_id)


def _resolve(sb):
    return events.resolve_league_event_update(sb, club_id="club-1", session_id="event-a")


def test_rounds_share_one_pending_update_including_departures_late_arrivals_and_retries():
    sb = _database()
    assert _queue(sb)["queued"] == 4
    assert _resolve(sb) is None
    initial = deepcopy(sb.tables["player_profile_update_outbox"])
    _add_round(sb, 2, [200, 201, 202], players=(1, 2, 3, 5))
    result = _queue(sb)
    assert result["queued"] == 1 and result["already_queued"] == 4
    rows = sb.tables["player_profile_update_outbox"]
    assert len(rows) == 5
    assert rows[:4] == initial, "Later rounds do not replace or reopen pending/sent rows"
    assert {row["player_id"] for row in rows} == {1, 2, 3, 4, 5}
    assert len({row["queue_operation_key"] for row in rows}) == 1
    assert all(row["digest_snapshot_json"]["league_live_session_id"] == "event-a" for row in rows)
    rows[0]["send_status"] = "sent"
    assert _queue(sb)["already_queued"] == 5
    assert rows[0]["send_status"] == "sent" and len(rows) == 5
    resolved = _resolve(sb)
    assert set(resolved["match_ids"]) == {"100", "101", "102", "200", "201", "202"}
    assert resolved["badge_refs"] == []
    assert len(resolved["session_evidence"]["player_updates"]) == 8


def test_same_day_other_event_and_other_club_are_separate():
    sb = _database(rounds=2)
    _queue(sb)
    _add_round(sb, 1, [800], session_id="event-b")
    _add_round(sb, 1, [900], club_id="other-club")
    assert _queue(sb, "event-b")["queued"] == 4
    assert len(sb.tables["player_profile_update_outbox"]) == 8
    assert len({row["queue_operation_key"] for row in sb.tables["player_profile_update_outbox"]}) == 2
    assert set(_resolve(sb)["match_ids"]) == {"100", "101", "102", "200", "201", "202"}


@pytest.mark.parametrize("status", ["intent", "publishing", "published", "reconciling", "retryable", "recovery_required"])
def test_no_email_while_any_round_is_incomplete_even_after_finish(status):
    sb = _database(rounds=2)
    sb.tables["league_live_sessions"][0].update(status="complete", completed_at="2026-11-04T20:00:00Z")
    sb.tables["league_live_publish_operations"][1]["status"] = status
    assert _resolve(sb) is None


def test_finish_early_releases_only_published_rounds_and_missing_audit_still_waits():
    sb = _database(rounds=1, total=3)
    assert _resolve(sb) is None
    sb.tables["league_live_sessions"][0].update(status="complete", completed_at="2026-11-04T20:00:00Z")
    assert _resolve(sb)["match_ids"] == ["100", "101", "102"]
    sb.tables["league_live_publish_operations"][0]["completion_audited_at"] = None
    assert _resolve(sb) is None


@pytest.mark.parametrize("problem", ["missing_game", "deleted_game", "wrong_context", "wrong_club", "short_receipt", "overlap"])
def test_incomplete_or_wrong_event_receipts_never_send_partial_results(problem):
    sb = _database(rounds=2)
    if problem == "missing_game":
        sb.tables["matches"].pop()
    elif problem == "deleted_game":
        sb.tables["matches"][-1]["deleted_at"] = "2026-11-04"
    elif problem == "wrong_context":
        sb.tables["matches"][-1]["context_id"] = "other-event-game"
    elif problem == "wrong_club":
        sb.tables["matches"][-1]["club_id"] = "other-club"
    elif problem == "short_receipt":
        sb.tables["league_live_publish_operations"][-1]["published_match_ids"].pop()
    else:
        sb.tables["league_live_publish_operations"][-1]["published_match_ids"][0] = 100
    with pytest.raises(ValueError):
        _resolve(sb)


def test_automatic_resolution_ignores_and_preserves_retired_badge_jobs(monkeypatch):
    monkeypatch.setattr(events, "ensure_notification_badge_job", lambda **_: pytest.fail("No badge job creation or retry"))
    sb = _database(rounds=2)
    sb.tables["badge_eval_queue"][0].update(status="error", attempts=3)
    scope = _resolve(sb)
    assert scope["badge_refs"] == []
    assert sb.tables["badge_eval_queue"][0]["status"] == "error"
    assert sb.tables["badge_eval_queue"][0]["attempts"] == 3


def test_sender_waits_without_claiming_then_builds_one_complete_event(sender_case):
    case = sender_case
    case.outbox["digest_snapshot_json"]["league_live_session_id"] = "event-a"
    sb = _database()
    ctx = SimpleNamespace(supabase=sb, club_id="club-1")
    first = sender.send_pending_player_update_emails(ctx, public_base_url="https://example.test")
    assert first["waiting_for_event"] == 1 and first["sent"] == 0
    assert case.calls.claims == case.calls.computes == case.calls.renders == []
    _add_round(sb, 2, [200, 201, 202])
    second = sender.send_pending_player_update_emails(ctx, public_base_url="https://example.test")
    assert second["sent"] == 1 and second["waiting_for_event"] == 0
    assert len(case.calls.claims) == len(case.calls.computes) == 1
    assert case.calls.computes[0]["match_ids"] == ["100", "101", "102", "200", "201", "202"]
    assert case.calls.computes[0]["start_date"] == date(2026, 11, 4)
    assert case.calls.computes[0]["end_date"] == date(2026, 11, 4)
    assert case.calls.smtp == []


def test_event_read_failure_leaves_email_unclaimed_for_automatic_retry(sender_case):
    case = sender_case
    case.outbox["digest_snapshot_json"]["league_live_session_id"] = "event-a"
    class Unavailable:
        def table(self, _name):
            raise RuntimeError("Database unavailable")
    result = sender.send_pending_player_update_emails(SimpleNamespace(supabase=Unavailable(), club_id="club-1"))
    assert result["errors"] == 1 and result["sent"] == 0
    assert case.outbox["send_status"] == "pending"
    assert case.calls.claims == case.calls.computes == case.calls.smtp == []


def test_reported_two_rounds_render_six_games_five_wins_and_combined_rating(monkeypatch):
    monkeypatch.setattr("jupr_app.domain.recaps.player_weekly_digest.get_player_tournament_trophies", lambda *_args: [])
    # Use unrounded snapshots so the 4-decimal movement is tested too.
    values = [5.01502, 4.976, 5.008, 5.01032, 5.0134, 5.026, 5.03752]
    games = [game(100 + i, score=(9, 11) if i == 0 else (11, 5),
        before=values[i] * 400, after=values[i + 1] * 400,
        when="2026-11-04T19:00:00Z") for i in range(6)]
    db = DB(games + [game(900, score=(15, 0), when="2026-11-04T21:00:00Z")])
    digest = compute_player_weekly_digest(context(db), 1, date(2026, 11, 4), date(2026, 11, 4),
        match_ids=[row["id"] for row in games])
    assert digest["summary"] == {"overall_jupr_before": 5.015, "overall_jupr_after": 5.038,
        "overall_delta": .0225, "matches_played": 6, "wins": 5, "losses": 1, "record": "5-1"}
    rendered = build_player_update_email_text(digest)
    assert "5-1" in rendered and "15-0" not in rendered
    assert len(digest["matches_played_rows"]) == 6
