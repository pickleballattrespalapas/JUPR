"""Exercise the recap's real PostgREST client against the deployed table contract."""
from datetime import date, datetime, timezone

import httpx
import pandas as pd
import pytest
from supabase import ClientOptions, create_client

from jupr_app.domain.recaps.weekly_recap import _filter_matches, _load_matches
from scripts.verify_weekly_recap_generation import require_read


# public.matches columns verified in production on 2026-10-16.
MATCH_COLUMNS = set("""id club_id date league t1_p1 t1_p2 t2_p1 t2_p2 score_t1 score_t2
elo_delta match_type week_tag t1_p1_r t1_p2_r t2_p1_r t2_p2_r t1_p1_r_end
t1_p2_r_end t2_p1_r_end t2_p2_r_end elo_delta_t1 elo_delta_t2 context_type context_id
tournament_id tournament_game_id idempotency_key deleted_at deleted_by deleted_source
delete_note updated_at updated_by correction_note rating_bonus_elo rating_bonus_reason
match_format singles_replay_managed rating_scope notes row_version""".split())


@pytest.mark.parametrize("method", ["POST", "PATCH", "PUT", "DELETE"])
def test_live_report_verification_cannot_mutate_data(method):
    with pytest.raises(RuntimeError, match="database reads only"):
        require_read(httpx.Request(method, "https://fixture.supabase.co/rest/v1/weekly_recaps"))
    require_read(httpx.Request("GET", "https://fixture.supabase.co/rest/v1/matches"))


def test_recap_loads_all_valid_games_with_the_deployed_match_schema():
    source = [{"id": i, "club_id": "club", "date": "2026-10-11T12:00:00+00:00",
               "league": "Fall", "match_type": "League", "score_t1": 11, "score_t2": 7,
               "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "deleted_at": None}
              for i in range(1, 7)]
    source[4]["deleted_at"] = "2026-10-12T12:00:00+00:00"
    source[5]["club_id"] = "other"
    reads = []

    def serve(request):
        assert request.method == "GET"
        assert request.url.path == "/rest/v1/matches"
        query = request.url.params
        columns = query["select"].split(",")
        unknown = set(columns) - MATCH_COLUMNS
        if unknown:
            return httpx.Response(400, json={"code": "42703", "message": f"Unknown columns: {sorted(unknown)}", "hint": None, "details": None})
        assert query["club_id"] == "eq.club"
        assert query["deleted_at"] == "is.null"
        assert query["order"] == "id.asc"
        rows = [r for r in source if r["club_id"] == "club" and r["deleted_at"] is None]
        offset = int(query.get("offset", "0"))
        reads.append(offset)
        # A server cap below the requested page size must not truncate history.
        return httpx.Response(200, json=[{c: r.get(c) for c in columns} for r in rows[offset:offset + 2]])

    with httpx.Client(transport=httpx.MockTransport(serve)) as transport:
        db = create_client("https://fixture.supabase.co", "test-key", options=ClientOptions(httpx_client=transport))
        frame = _load_matches(pd.DataFrame(), db, "club", date(2026, 10, 10), date(2026, 10, 16),
                              datetime(2026, 10, 10, tzinfo=timezone.utc), datetime(2026, 10, 17, tzinfo=timezone.utc),
                              include_tournaments=True)
    assert frame["id"].astype(int).tolist() == [1, 2, 3, 4]
    assert reads == [0, 2, 4]
    # An excluded game in a caller's cached frame cannot re-enter the report.
    filtered = _filter_matches(pd.DataFrame(source[:5]), include_tournaments=True)
    assert filtered["id"].tolist() == [1, 2, 3, 4]
