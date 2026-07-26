import json
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request
from uuid import UUID

import pytest

from scripts.prepare_parity_tournament_live_fixture import (
    CLEANUP_REPORT_NAME,
    EXPECTED_CLUB_ID,
    EXPECTED_SUPABASE_ORIGIN,
    EXPECTED_SUPABASE_PROJECT_REF,
    EXERCISE_SCORE_A,
    EXERCISE_SCORE_B,
    FixtureError,
    MANIFEST_NAME,
    ORIGINAL_SCORE_A,
    ORIGINAL_SCORE_B,
    cleanup_fixture,
    prepare_fixture,
)


def _base_env() -> dict[str, str]:
    return {
        "STAGING_SUPABASE_URL": EXPECTED_SUPABASE_ORIGIN,
        "STAGING_SUPABASE_PROJECT_REF": EXPECTED_SUPABASE_PROJECT_REF,
        "STAGING_SUPABASE_SERVICE_ROLE_KEY": "service-role-test-secret",
        "CANDIDATE_SHA": "a" * 40,
        "GITHUB_RUN_ID": "12345",
        "GITHUB_RUN_ATTEMPT": "2",
    }


def _uuid_factory() -> Iterator[UUID]:
    for value in range(1, 100):
        yield UUID(int=value)


class FakePostgrest:
    def __init__(self) -> None:
        self.tables: dict[str, list[dict[str, object]]] = {
            "tournaments": [],
            "tournament_event_draws": [],
            "tournament_teams": [],
            "tournament_games": [],
            "tournament_podium": [],
            "matches": [],
            "tournament_admin_operations": [],
            "tournament_registration_settings": [],
            "tournament_registration_days": [],
            "tournament_event_options": [],
            "tournament_registrations": [],
            "tournament_registration_selections": [],
            "tournament_registration_partner_requests": [],
            "tournament_registration_team_links": [],
            "tournament_registration_team_members": [],
        }
        self.calls: list[tuple[str, str]] = []
        self.delete_order: list[str] = []
        self.fail_post_table: str | None = None

    @staticmethod
    def _matches(row: dict[str, object], query: dict[str, list[str]]) -> bool:
        for name, values in query.items():
            if name == "select":
                continue
            if len(values) != 1 or not values[0].startswith("eq."):
                raise AssertionError(f"Unsupported fake filter: {name}={values}")
            expected = values[0][3:]
            if str(row.get(name) or "") != expected:
                return False
        return True

    def __call__(self, request: Request) -> tuple[int, bytes]:
        parsed = urlparse(request.full_url)
        assert f"{parsed.scheme}://{parsed.netloc}" == EXPECTED_SUPABASE_ORIGIN
        prefix = "/rest/v1/"
        assert parsed.path.startswith(prefix)
        table = parsed.path[len(prefix) :]
        assert table in self.tables
        method = request.get_method()
        self.calls.append((method, table))
        assert request.headers.get("Authorization") == "Bearer service-role-test-secret"
        assert request.headers.get("Apikey") == "service-role-test-secret"
        query = parse_qs(parsed.query, keep_blank_values=True)

        if method == "GET":
            rows = [
                deepcopy(row)
                for row in self.tables[table]
                if self._matches(row, query)
            ]
            return 200, json.dumps(rows).encode()
        if method == "POST":
            if self.fail_post_table == table:
                return 500, b'{"message":"intentionally hidden"}'
            payload = json.loads((request.data or b"null").decode())
            rows = payload if isinstance(payload, list) else [payload]
            assert all(isinstance(row, dict) for row in rows)
            if table in {"tournament_teams", "tournament_games"}:
                assert all(row.get("club_id") == EXPECTED_CLUB_ID for row in rows)
            self.tables[table].extend(deepcopy(rows))
            return 201, json.dumps(rows).encode()
        if method == "DELETE":
            deleted: list[dict[str, object]] = []
            retained: list[dict[str, object]] = []
            for row in self.tables[table]:
                (deleted if self._matches(row, query) else retained).append(row)
            self.tables[table] = retained
            self.delete_order.extend([table] * len(deleted))
            return 200, json.dumps(deleted).encode()
        raise AssertionError(f"Unsupported fake method: {method}")


def _prepare(
    tmp_path: Path,
    fake: FakePostgrest,
    *,
    env: dict[str, str] | None = None,
) -> tuple[dict[str, object], Path]:
    github_env = tmp_path / "github-env"
    ids = _uuid_factory()
    prepared = prepare_fixture(
        report_dir=tmp_path,
        github_env=github_env,
        env=env or _base_env(),
        transport=fake,
        uuid_factory=lambda: next(ids),
    )
    return prepared, github_env


def test_prepare_exports_unique_fixture_and_cleanup_deletes_exact_rows(
    tmp_path: Path,
) -> None:
    fake = FakePostgrest()
    prepared, github_env = _prepare(tmp_path, fake)

    assert prepared["status"] == "prepared"
    assert prepared["club_id"] == EXPECTED_CLUB_ID
    assert len(fake.tables["tournaments"]) == 1
    assert len(fake.tables["tournament_event_draws"]) == 1
    assert len(fake.tables["tournament_teams"]) == 4
    assert len(fake.tables["tournament_games"]) == 1
    assert all(
        row["player1_id"] is None and row["player2_id"] is None
        for row in fake.tables["tournament_teams"]
    )
    game = fake.tables["tournament_games"][0]
    assert (game["score_a"], game["score_b"]) == (
        ORIGINAL_SCORE_A,
        ORIGINAL_SCORE_B,
    )

    exported = github_env.read_text(encoding="utf-8")
    assert f"JUPR_TOURNAMENT_LIVE_TOURNAMENT_ID={prepared['tournament_id']}" in exported
    assert f"JUPR_TOURNAMENT_LIVE_DRAW_ID={prepared['draw_id']}" in exported
    assert f"JUPR_TOURNAMENT_LIVE_GAME_ID={prepared['game_id']}" in exported
    assert f"JUPR_TOURNAMENT_LIVE_ORIGINAL_SCORE_A={ORIGINAL_SCORE_A}" in exported
    assert f"JUPR_TOURNAMENT_LIVE_ORIGINAL_SCORE_B={ORIGINAL_SCORE_B}" in exported
    assert f"JUPR_TOURNAMENT_LIVE_EXERCISE_SCORE_A={EXERCISE_SCORE_A}" in exported
    assert f"JUPR_TOURNAMENT_LIVE_EXERCISE_SCORE_B={EXERCISE_SCORE_B}" in exported
    assert "JUPR_TOURNAMENT_LIVE_ALLOW_MUTATION_E2E=1" in exported

    manifest_text = (tmp_path / MANIFEST_NAME).read_text(encoding="utf-8")
    assert "service-role-test-secret" not in manifest_text
    assert json.loads(manifest_text)["verified_counts"]["official_matches"] == 0

    cleaned = cleanup_fixture(
        report_dir=tmp_path,
        env=_base_env(),
        transport=fake,
    )
    assert cleaned["status"] == "cleaned"
    assert cleaned["deleted_counts"] == {
        "tournaments": 1,
        "draws": 1,
        "teams": 4,
        "games": 1,
    }
    assert fake.delete_order == [
        "tournament_games",
        "tournament_teams",
        "tournament_teams",
        "tournament_teams",
        "tournament_teams",
        "tournament_event_draws",
        "tournaments",
    ]
    assert all(
        not fake.tables[name]
        for name in (
            "tournaments",
            "tournament_event_draws",
            "tournament_teams",
            "tournament_games",
        )
    )
    cleanup_text = (tmp_path / CLEANUP_REPORT_NAME).read_text(encoding="utf-8")
    assert "service-role-test-secret" not in cleanup_text
    assert json.loads(cleanup_text)["verified_remaining_core_rows"] == 0


def test_prepare_refuses_wrong_project_before_manifest_or_request(tmp_path: Path) -> None:
    fake = FakePostgrest()
    env = _base_env()
    env["STAGING_SUPABASE_URL"] = "https://production-ref.supabase.co"
    env["STAGING_SUPABASE_PROJECT_REF"] = "production-ref"
    ids = _uuid_factory()

    with pytest.raises(FixtureError, match="non-allowlisted Supabase origin"):
        prepare_fixture(
            report_dir=tmp_path,
            github_env=tmp_path / "github-env",
            env=env,
            transport=fake,
            uuid_factory=lambda: next(ids),
        )

    assert fake.calls == []
    assert not (tmp_path / MANIFEST_NAME).exists()


def test_collision_is_detected_before_any_mutation_and_manifest_is_retained(
    tmp_path: Path,
) -> None:
    fake = FakePostgrest()
    tournament_id = str(UUID(int=1))
    fake.tables["tournaments"].append({"id": tournament_id, "club_id": "other"})
    ids = _uuid_factory()

    with pytest.raises(FixtureError, match="already exists"):
        prepare_fixture(
            report_dir=tmp_path,
            github_env=tmp_path / "github-env",
            env=_base_env(),
            transport=fake,
            uuid_factory=lambda: next(ids),
        )

    assert not any(method in {"POST", "DELETE"} for method, _table in fake.calls)
    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["status"] == "planned"
    assert manifest["tournament_id"] == tournament_id


def test_partial_prepare_is_removed_from_manifest_owned_ids(tmp_path: Path) -> None:
    fake = FakePostgrest()
    fake.fail_post_table = "tournament_teams"
    ids = _uuid_factory()

    with pytest.raises(FixtureError, match="HTTP 500"):
        prepare_fixture(
            report_dir=tmp_path,
            github_env=tmp_path / "github-env",
            env=_base_env(),
            transport=fake,
            uuid_factory=lambda: next(ids),
        )

    assert len(fake.tables["tournaments"]) == 1
    assert len(fake.tables["tournament_event_draws"]) == 1
    fake.fail_post_table = None
    cleaned = cleanup_fixture(
        report_dir=tmp_path,
        env=_base_env(),
        transport=fake,
    )
    assert cleaned["status"] == "cleaned"
    assert cleaned["deleted_counts"] == {
        "tournaments": 1,
        "draws": 1,
        "teams": 0,
        "games": 0,
    }


@pytest.mark.parametrize(
    "reference_field",
    ("tournament_id", "tournament_game_id"),
)
def test_cleanup_refuses_each_official_match_reference(
    tmp_path: Path,
    reference_field: str,
) -> None:
    fake = FakePostgrest()
    prepared, _github_env = _prepare(tmp_path, fake)
    reference_value = (
        prepared["tournament_id"]
        if reference_field == "tournament_id"
        else prepared["game_id"]
    )
    fake.tables["matches"].append(
        {
            "id": "match-1",
            reference_field: reference_value,
            "is_deleted": True,
        }
    )
    with pytest.raises(FixtureError, match="official match"):
        cleanup_fixture(
            report_dir=tmp_path,
            env=_base_env(),
            transport=fake,
        )
    assert not fake.delete_order
    failure = json.loads(
        (tmp_path / CLEANUP_REPORT_NAME).read_text(encoding="utf-8")
    )
    assert failure["status"] == "cleanup_refused"
    assert "official match" in failure["failure"]


def test_cleanup_refuses_unrestored_score(tmp_path: Path) -> None:
    fake = FakePostgrest()
    _prepared, _github_env = _prepare(tmp_path, fake)
    fake.tables["tournament_games"][0]["score_b"] = EXERCISE_SCORE_B
    with pytest.raises(FixtureError, match="not restored"):
        cleanup_fixture(
            report_dir=tmp_path,
            env=_base_env(),
            transport=fake,
        )
    assert not fake.delete_order


def test_cleanup_refuses_nonterminal_operation_and_retains_completed_evidence(
    tmp_path: Path,
) -> None:
    fake = FakePostgrest()
    prepared, _github_env = _prepare(tmp_path, fake)
    fake.tables["tournament_admin_operations"].append(
        {
            "operation_key": "operation-active",
            "club_id": EXPECTED_CLUB_ID,
            "surface": "operations",
            "entity_id": prepared["draw_id"],
            "action": "tournament_live_score",
            "status": "recovery_required",
        }
    )
    with pytest.raises(FixtureError, match="unresolved"):
        cleanup_fixture(
            report_dir=tmp_path,
            env=_base_env(),
            transport=fake,
        )
    assert not fake.delete_order

    fake.tables["tournament_admin_operations"][0]["status"] = "completed"
    cleaned = cleanup_fixture(
        report_dir=tmp_path,
        env=_base_env(),
        transport=fake,
    )
    assert cleaned["retained_completed_operation_keys"] == ["operation-active"]
    assert fake.tables["tournament_admin_operations"]


def test_cleanup_refuses_unowned_tournament_child(tmp_path: Path) -> None:
    fake = FakePostgrest()
    prepared, _github_env = _prepare(tmp_path, fake)
    fake.tables["tournament_registration_partner_requests"].append(
        {
            "id": "unowned-request",
            "tournament_id": prepared["tournament_id"],
        }
    )

    with pytest.raises(FixtureError, match="unowned tournament child"):
        cleanup_fixture(
            report_dir=tmp_path,
            env=_base_env(),
            transport=fake,
        )
    assert not fake.delete_order


def test_cleanup_without_manifest_is_sanitized_noop_without_credentials(
    tmp_path: Path,
) -> None:
    fake = FakePostgrest()
    report = cleanup_fixture(report_dir=tmp_path, env={}, transport=fake)

    assert report["status"] == "not_prepared"
    assert fake.calls == []
    assert json.loads(
        (tmp_path / CLEANUP_REPORT_NAME).read_text(encoding="utf-8")
    )["deleted_counts"] == {
        "tournaments": 0,
        "draws": 0,
        "teams": 0,
        "games": 0,
    }
