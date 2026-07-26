from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request
from uuid import UUID

import pytest

from scripts.prepare_parity_match_exclusion_fixture import (
    CLEANUP_REPORT_NAME,
    EXPECTED_AUTH_ISSUER,
    EXPECTED_REPOSITORY,
    EXPECTED_SOURCE_CLUB_ID,
    EXPECTED_SUPABASE_ORIGIN,
    EXPECTED_SUPABASE_PROJECT_REF,
    FIXTURE_SOURCE,
    MANIFEST_NAME,
    PLAYER_BASELINE,
    FixtureError,
    cleanup_fixture,
    prepare_fixture,
)

SERVICE_ROLE_KEY = "never-log-service-role"
ADMIN_EMAIL = "staging-admin@example.invalid"
ADMIN_USER_ID = "70000000-0000-4000-8000-000000000001"


def _jwt_segment(payload: object) -> str:
    raw = json.dumps(payload, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _token(
    *,
    user_id: str = ADMIN_USER_ID,
    email: str = ADMIN_EMAIL,
    issuer: str = EXPECTED_AUTH_ISSUER,
    audience: object = "authenticated",
) -> str:
    return ".".join(
        (
            _jwt_segment({"alg": "ES256", "typ": "JWT"}),
            _jwt_segment(
                {
                    "sub": user_id,
                    "email": email,
                    "iss": issuer,
                    "aud": audience,
                }
            ),
            "test-signature",
        )
    )


def _base_env() -> dict[str, str]:
    return {
        "STAGING_SUPABASE_URL": EXPECTED_SUPABASE_ORIGIN,
        "STAGING_SUPABASE_PROJECT_REF": EXPECTED_SUPABASE_PROJECT_REF,
        "STAGING_SUPABASE_SERVICE_ROLE_KEY": SERVICE_ROLE_KEY,
        "STAGING_ADMIN_BEARER_TOKEN": _token(),
        "STAGING_ADMIN_EMAIL": ADMIN_EMAIL,
        "CANDIDATE_SHA": "a" * 40,
        "GITHUB_REPOSITORY": EXPECTED_REPOSITORY,
        "GITHUB_REF": "refs/heads/staging",
        "GITHUB_RUN_ID": "12345",
        "GITHUB_RUN_ATTEMPT": "2",
    }


def _uuid_factory() -> Iterator[UUID]:
    for value in range(1, 200):
        yield UUID(int=value)


class FakePostgrest:
    def __init__(self) -> None:
        self.tables: dict[str, list[dict[str, object]]] = {
            "clubs": [],
            "admin_role_assignments": [
                {
                    "id": 1,
                    "club_id": EXPECTED_SOURCE_CLUB_ID,
                    "email": ADMIN_EMAIL,
                    "role": "club_owner",
                    "user_id": ADMIN_USER_ID,
                }
            ],
            "players": [],
            "matches": [],
            "league_ratings": [],
            "player_badges": [],
            "match_exclusion_operations": [],
            "replay_jobs": [],
            "match_exclusion_badge_progress": [],
        }
        self.calls: list[tuple[str, str]] = []
        self.delete_order: list[str] = []
        self.delete_queries: list[dict[str, list[str]]] = []
        self.fail_post_table: str | None = None
        self.first_post_hook = None
        self._next_ids = {
            "admin_role_assignments": 10,
            "players": 100,
            "matches": 200,
            "league_ratings": 300,
        }

    @staticmethod
    def _matches(
        row: dict[str, object],
        query: dict[str, list[str]],
    ) -> bool:
        for name, values in query.items():
            if name in {"select", "order", "limit"}:
                continue
            if len(values) != 1:
                raise AssertionError(f"Unsupported fake filter: {name}={values}")
            expression = values[0]
            actual = row.get(name)
            if isinstance(actual, bool):
                actual_text = "true" if actual else "false"
            else:
                actual_text = "" if actual is None else str(actual)
            if expression.startswith("eq."):
                if actual_text != expression[3:]:
                    return False
                continue
            if expression.startswith("in.(") and expression.endswith(")"):
                expected_values = set(expression[4:-1].split(","))
                if actual_text not in expected_values:
                    return False
                continue
            raise AssertionError(f"Unsupported fake filter: {name}={values}")
        return True

    def _assign_defaults(self, table: str, row: dict[str, object]) -> None:
        if "id" not in row:
            if table == "player_badges":
                row["id"] = "90000000-0000-4000-8000-%012d" % (
                    len(self.tables[table]) + 1
                )
            elif table == "match_exclusion_badge_progress":
                row["id"] = "91000000-0000-4000-8000-%012d" % (
                    len(self.tables[table]) + 1
                )
            else:
                row["id"] = self._next_ids[table]
                self._next_ids[table] += 1
        if table == "matches":
            row.setdefault("row_version", 1)
            row.setdefault("deleted_at", None)
        if table == "clubs":
            row.setdefault("is_active", True)

    def __call__(self, request: Request) -> tuple[int, bytes]:
        parsed = urlparse(request.full_url)
        assert f"{parsed.scheme}://{parsed.netloc}" == EXPECTED_SUPABASE_ORIGIN
        assert parsed.path.startswith("/rest/v1/")
        table = parsed.path.removeprefix("/rest/v1/")
        assert table in self.tables
        method = request.get_method()
        self.calls.append((method, table))
        assert request.headers.get("Authorization") == f"Bearer {SERVICE_ROLE_KEY}"
        assert request.headers.get("Apikey") == SERVICE_ROLE_KEY
        query = parse_qs(parsed.query, keep_blank_values=True)

        if method == "GET":
            rows = [
                deepcopy(row)
                for row in self.tables[table]
                if self._matches(row, query)
            ]
            return 200, json.dumps(rows).encode()
        if method == "POST":
            if self.first_post_hook is not None:
                hook = self.first_post_hook
                self.first_post_hook = None
                hook()
            if self.fail_post_table == table:
                return 500, b'{"message":"sensitive external failure"}'
            payload = json.loads((request.data or b"null").decode())
            rows = payload if isinstance(payload, list) else [payload]
            inserted: list[dict[str, object]] = []
            for value in rows:
                assert isinstance(value, dict)
                row = deepcopy(value)
                self._assign_defaults(table, row)
                self.tables[table].append(row)
                inserted.append(deepcopy(row))
            return 201, json.dumps(inserted).encode()
        if method == "DELETE":
            self.delete_queries.append(query)
            deleted: list[dict[str, object]] = []
            retained: list[dict[str, object]] = []
            for row in self.tables[table]:
                (deleted if self._matches(row, query) else retained).append(row)
            self.tables[table] = retained
            self.delete_order.extend([table] * len(deleted))
            return 200, json.dumps(deleted).encode()
        raise AssertionError(f"Unsupported method: {method}")


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


def _install_terminal_evidence(
    fake: FakePostgrest,
    prepared: dict[str, object],
) -> None:
    club_id = str(prepared["club_id"])
    player_ids = [int(value) for value in prepared["player_ids"]]
    match_ids = {
        name: int(value)
        for name, value in dict(prepared["match_ids"]).items()
    }
    keys = {
        name: str(value)
        for name, value in dict(prepared["idempotency_keys"]).items()
    }
    operation_specs = (
        (
            "81000000-0000-4000-8000-000000000001",
            "82000000-0000-4000-8000-000000000001",
            keys["duplicate_cleanup"],
            "duplicate_cleanup",
            match_ids["duplicate_target"],
        ),
        (
            "81000000-0000-4000-8000-000000000002",
            "82000000-0000-4000-8000-000000000002",
            keys["direct_exclusion"],
            "exclude",
            match_ids["distinct"],
        ),
    )
    for operation_id, replay_id, key, mode, excluded_id in operation_specs:
        fake.tables["match_exclusion_operations"].append(
            {
                "id": operation_id,
                "club_id": club_id,
                "mode": mode,
                "idempotency_key": key,
                "status": "succeeded",
                "recovery_stage": None,
                "replay_job_id": replay_id,
                "source": FIXTURE_SOURCE,
                "delete_note": f"Disposable staging fixture {club_id}",
                "excluded_match_ids": [excluded_id],
                "affected_player_ids": player_ids,
                "finished_at": "2026-07-26T12:00:00Z",
            }
        )
        fake.tables["replay_jobs"].append(
            {
                "id": replay_id,
                "club_id": club_id,
                "target_reset": "ALL (Full System Reset)",
                "status": "succeeded",
                "source": FIXTURE_SOURCE,
                "finished_at": "2026-07-26T12:00:00Z",
                "attempt_count": 1,
                "lease_token": None,
                "leased_by": None,
                "lease_expires_at": None,
                "heartbeat_at": None,
            }
        )
        for player_id in player_ids:
            progress = {
                "operation_id": operation_id,
                "club_id": club_id,
                "player_id": player_id,
                "status": "succeeded",
                "finished_at": "2026-07-26T12:00:00Z",
            }
            fake._assign_defaults("match_exclusion_badge_progress", progress)
            fake.tables["match_exclusion_badge_progress"].append(progress)

    for row in fake.tables["matches"]:
        if int(row["id"]) in {
            match_ids["duplicate_target"],
            match_ids["distinct"],
        }:
            row["deleted_at"] = "2026-07-26T12:00:00Z"
            row["row_version"] = 2
    for player in fake.tables["players"]:
        player["last_game_at"] = "2099-01-05T18:00:00Z"

    fake.tables["league_ratings"].append(
        {
            "id": 301,
            "club_id": club_id,
            "player_id": player_ids[0],
            "league_name": "fixture",
        }
    )
    fake.tables["player_badges"].append(
        {
            "id": "90000000-0000-4000-8000-000000000001",
            "club_id": club_id,
            "player_id": player_ids[0],
            "badge_id": "participant",
        }
    )


def test_prepare_writes_planned_manifest_before_first_mutation_and_exports_only_ids(
    tmp_path: Path,
) -> None:
    fake = FakePostgrest()
    manifest_path = tmp_path / MANIFEST_NAME

    def assert_planned_manifest() -> None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["status"] == "planned"
        assert manifest["player_ids"] == []
        assert manifest["match_ids"] == {}

    fake.first_post_hook = assert_planned_manifest
    prepared, github_env = _prepare(tmp_path, fake)

    assert prepared["status"] == "prepared"
    assert prepared["club_id"].startswith("jupr_parity_mex_")
    assert len(prepared["player_ids"]) == 4
    assert set(prepared["match_ids"]) == {
        "duplicate_keep",
        "duplicate_target",
        "distinct",
    }
    assert len(fake.tables["clubs"]) == 1
    assert len(
        [
            row
            for row in fake.tables["admin_role_assignments"]
            if row["club_id"] == prepared["club_id"]
        ]
    ) == 1
    assert len(fake.tables["players"]) == 4
    assert len(fake.tables["matches"]) == 3
    assert all(
        row["singles_replay_baseline"] == PLAYER_BASELINE
        for row in fake.tables["players"]
    )

    exported = github_env.read_text(encoding="utf-8")
    assert (
        f"JUPR_MATCH_EXCLUSION_FIXTURE_CLUB_ID={prepared['club_id']}" in exported
    )
    assert "JUPR_MATCH_EXCLUSION_ALLOW_MUTATION_E2E=1" in exported
    assert "JUPR_MATCH_EXCLUSION_DUPLICATE_IDEMPOTENCY_KEY=" in exported
    assert SERVICE_ROLE_KEY not in exported
    assert ADMIN_EMAIL not in exported
    assert ADMIN_USER_ID not in exported
    assert _token() not in exported

    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert SERVICE_ROLE_KEY not in manifest_text
    assert ADMIN_EMAIL not in manifest_text
    assert ADMIN_USER_ID not in manifest_text
    assert _token() not in manifest_text
    assert json.loads(manifest_text)["verified_counts"] == {
        "clubs": 1,
        "temporary_roles": 1,
        "players": 4,
        "rated_doubles_matches": 3,
        "canonical_duplicate_targets": 1,
        "distinct_targets": 1,
        "operations": 0,
        "replay_jobs": 0,
        "badge_progress": 0,
    }


@pytest.mark.parametrize(
    ("env_patch", "message"),
    [
        (
            {"STAGING_SUPABASE_URL": "https://production.example.invalid"},
            "non-allowlisted Supabase origin",
        ),
        (
            {"STAGING_SUPABASE_PROJECT_REF": "production-ref"},
            "non-allowlisted Supabase project",
        ),
        (
            {"GITHUB_REPOSITORY": "somebody/else"},
            "allowlisted repository",
        ),
        (
            {"GITHUB_REF": "refs/heads/feature"},
            "non-allowlisted Git ref",
        ),
        ({"CANDIDATE_SHA": "abc"}, "exact lowercase commit SHA"),
    ],
)
def test_prepare_refuses_non_allowlisted_environment_before_manifest_or_request(
    tmp_path: Path,
    env_patch: dict[str, str],
    message: str,
) -> None:
    fake = FakePostgrest()
    env = {**_base_env(), **env_patch}
    with pytest.raises(FixtureError, match=message):
        _prepare(tmp_path, fake, env=env)
    assert fake.calls == []
    assert not (tmp_path / MANIFEST_NAME).exists()


@pytest.mark.parametrize(
    ("env_patch", "message"),
    [
        ({"STAGING_ADMIN_BEARER_TOKEN": "not-a-jwt"}, "bearer"),
        (
            {"STAGING_ADMIN_BEARER_TOKEN": _token(user_id=str(UUID(int=55)))},
            "already-bound",
        ),
        (
            {"STAGING_ADMIN_BEARER_TOKEN": _token(email="other@example.invalid")},
            "bearer",
        ),
        (
            {
                "STAGING_ADMIN_BEARER_TOKEN": _token(
                    issuer="https://wrong.invalid/auth/v1"
                )
            },
            "bearer",
        ),
        (
            {"STAGING_ADMIN_BEARER_TOKEN": _token(audience="anon")},
            "bearer",
        ),
    ],
)
def test_prepare_refuses_invalid_session_identity_before_manifest(
    tmp_path: Path,
    env_patch: dict[str, str],
    message: str,
) -> None:
    fake = FakePostgrest()
    env = {**_base_env(), **env_patch}
    with pytest.raises(FixtureError, match=message):
        _prepare(tmp_path, fake, env=env)
    if message == "bearer":
        assert fake.calls == []
        assert not (tmp_path / MANIFEST_NAME).exists()
    else:
        assert all(method == "GET" for method, _table in fake.calls)
        assert json.loads((tmp_path / MANIFEST_NAME).read_text())["status"] == "planned"


def test_prepare_requires_exact_bound_source_assignment(tmp_path: Path) -> None:
    fake = FakePostgrest()
    fake.tables["admin_role_assignments"][0]["user_id"] = str(UUID(int=99))
    with pytest.raises(FixtureError, match="exactly one already-bound"):
        _prepare(tmp_path, fake)
    assert all(method == "GET" for method, _table in fake.calls)
    assert json.loads((tmp_path / MANIFEST_NAME).read_text())["status"] == "planned"


def test_prepare_failure_is_sanitized_and_manifest_enables_partial_cleanup(
    tmp_path: Path,
) -> None:
    fake = FakePostgrest()
    fake.fail_post_table = "players"
    with pytest.raises(
        FixtureError,
        match="Disposable player creation failed with HTTP 500",
    ) as caught:
        _prepare(tmp_path, fake)
    assert "sensitive external failure" not in str(caught.value)
    assert len(fake.tables["clubs"]) == 1
    assert json.loads((tmp_path / MANIFEST_NAME).read_text())["status"] == "planned"

    fake.fail_post_table = None
    cleaned = cleanup_fixture(
        report_dir=tmp_path,
        env=_base_env(),
        transport=fake,
    )
    assert cleaned["status"] == "cleaned"
    assert not fake.tables["clubs"]
    assert not [
        row
        for row in fake.tables["admin_role_assignments"]
        if row["club_id"] != EXPECTED_SOURCE_CLUB_ID
    ]


def test_cleanup_requires_terminal_ledgers_and_then_removes_exact_fixture_rows(
    tmp_path: Path,
) -> None:
    fake = FakePostgrest()
    prepared, _github_env = _prepare(tmp_path, fake)
    _install_terminal_evidence(fake, prepared)

    cleaned = cleanup_fixture(
        report_dir=tmp_path,
        env=_base_env(),
        transport=fake,
    )

    assert cleaned["status"] == "cleaned"
    assert cleaned["deleted_counts"] == {
        "player_badges": 1,
        "league_ratings": 1,
        "matches": 3,
        "players": 4,
        "temporary_roles": 1,
        "clubs": 1,
    }
    assert cleaned["verified_remaining_core_rows"] == 0
    assert len(cleaned["retained_operation_ids"]) == 2
    assert len(cleaned["retained_replay_job_ids"]) == 2
    assert cleaned["retained_badge_progress_count"] == 8
    assert fake.delete_order == [
        "player_badges",
        "league_ratings",
        "matches",
        "matches",
        "matches",
        "players",
        "players",
        "players",
        "players",
        "admin_role_assignments",
        "clubs",
    ]
    assert fake.delete_queries
    assert all(
        len(query.get("id", [])) == 1
        and (
            (
                query["id"][0].startswith("in.(")
                and query["id"][0].endswith(")")
            )
            or query["id"][0]
            == f"eq.{prepared['club_id']}"
        )
        for query in fake.delete_queries
    )
    assert fake.tables["match_exclusion_operations"]
    assert fake.tables["replay_jobs"]
    assert fake.tables["match_exclusion_badge_progress"]
    assert not fake.tables["clubs"]
    assert not fake.tables["players"]
    assert not fake.tables["matches"]
    assert len(fake.tables["admin_role_assignments"]) == 1

    cleanup_text = (tmp_path / CLEANUP_REPORT_NAME).read_text(encoding="utf-8")
    for sensitive in (SERVICE_ROLE_KEY, ADMIN_EMAIL, ADMIN_USER_ID, _token()):
        assert sensitive not in cleanup_text


@pytest.mark.parametrize(
    ("table", "field", "value", "message"),
    [
        (
            "match_exclusion_operations",
            "status",
            "pending_replay",
            "operation is not terminal",
        ),
        (
            "replay_jobs",
            "status",
            "running",
            "replay job lacks exact terminal lease evidence",
        ),
        (
            "match_exclusion_badge_progress",
            "status",
            "failed",
            "badge reconciliation is not terminal",
        ),
    ],
)
def test_cleanup_refuses_nonterminal_recovery_without_any_delete(
    tmp_path: Path,
    table: str,
    field: str,
    value: str,
    message: str,
) -> None:
    fake = FakePostgrest()
    prepared, _github_env = _prepare(tmp_path, fake)
    _install_terminal_evidence(fake, prepared)
    fake.tables[table][0][field] = value
    if table == "replay_jobs" and value == "running":
        fake.tables[table][0]["finished_at"] = None
    if table == "match_exclusion_badge_progress":
        fake.tables[table][0]["finished_at"] = None

    with pytest.raises(FixtureError, match=message):
        cleanup_fixture(
            report_dir=tmp_path,
            env=_base_env(),
            transport=fake,
        )

    assert fake.delete_order == []
    report = json.loads((tmp_path / CLEANUP_REPORT_NAME).read_text())
    assert report["status"] == "cleanup_refused"
    assert fake.tables["clubs"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("attempt_count", 2),
        ("lease_token", "83000000-0000-4000-8000-000000000001"),
        ("leased_by", "stale-worker"),
        ("lease_expires_at", "2099-01-01T00:00:00Z"),
        ("heartbeat_at", "2026-07-26T12:00:00Z"),
    ],
)
def test_cleanup_requires_exact_terminal_replay_lease_evidence(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    fake = FakePostgrest()
    prepared, _github_env = _prepare(tmp_path, fake)
    _install_terminal_evidence(fake, prepared)
    fake.tables["replay_jobs"][0][field] = value

    with pytest.raises(
        FixtureError,
        match="replay job lacks exact terminal lease evidence",
    ):
        cleanup_fixture(
            report_dir=tmp_path,
            env=_base_env(),
            transport=fake,
        )

    assert fake.delete_order == []


def test_cleanup_requires_all_player_activity_to_match_active_fixture_history(
    tmp_path: Path,
) -> None:
    fake = FakePostgrest()
    prepared, _github_env = _prepare(tmp_path, fake)
    _install_terminal_evidence(fake, prepared)
    fake.tables["players"][0]["last_game_at"] = "2099-01-05T18:05:00Z"

    with pytest.raises(
        FixtureError,
        match="last_game_at does not match the latest active scored fixture match",
    ):
        cleanup_fixture(
            report_dir=tmp_path,
            env=_base_env(),
            transport=fake,
        )

    assert fake.delete_order == []


def test_cleanup_refuses_unowned_isolated_club_row_without_any_delete(
    tmp_path: Path,
) -> None:
    fake = FakePostgrest()
    prepared, _github_env = _prepare(tmp_path, fake)
    fake.tables["matches"].append(
        {
            "id": 999,
            "club_id": prepared["club_id"],
            "context_type": "unknown",
            "context_id": "unknown",
            "notes": "not owned",
            "match_format": "doubles",
            "rating_scope": "overall_only",
            "t1_p1": prepared["player_ids"][0],
            "t1_p2": prepared["player_ids"][1],
            "t2_p1": prepared["player_ids"][2],
            "t2_p2": prepared["player_ids"][3],
            "row_version": 1,
            "deleted_at": None,
        }
    )
    with pytest.raises(FixtureError, match="unowned match row"):
        cleanup_fixture(
            report_dir=tmp_path,
            env=_base_env(),
            transport=fake,
        )
    assert fake.delete_order == []
    assert fake.tables["clubs"]


def test_cleanup_refuses_incomplete_badge_progress_without_any_delete(
    tmp_path: Path,
) -> None:
    fake = FakePostgrest()
    prepared, _github_env = _prepare(tmp_path, fake)
    _install_terminal_evidence(fake, prepared)
    fake.tables["match_exclusion_badge_progress"].pop()

    with pytest.raises(FixtureError, match="progress is incomplete"):
        cleanup_fixture(
            report_dir=tmp_path,
            env=_base_env(),
            transport=fake,
        )

    assert fake.delete_order == []
    assert fake.tables["clubs"]


def test_cleanup_without_manifest_is_a_sanitized_noop(tmp_path: Path) -> None:
    fake = FakePostgrest()
    cleaned = cleanup_fixture(
        report_dir=tmp_path,
        env={},
        transport=fake,
    )
    assert cleaned["status"] == "not_prepared"
    assert fake.calls == []
    assert (tmp_path / CLEANUP_REPORT_NAME).exists()


def test_cli_requires_github_env_for_prepare() -> None:
    from scripts import prepare_parity_match_exclusion_fixture as fixture

    assert fixture.main(["prepare", "--report-dir", "artifacts"]) == 2
