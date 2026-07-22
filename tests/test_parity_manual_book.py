import re
from pathlib import Path

from scripts.check_parity_manual_book import (
    BOOK_PATH,
    BOOK_ROW_RE,
    FIXTURE_ROW_RE,
    FLAG_ROW_RE,
    MANUAL_MUTATION_ROW_RE,
    MIGRATION_ROW_RE,
    WAVE_ROW_RE,
    check_book,
    check_book_complete,
)

VERCEL_ID = "dpl_staging123"
VERCEL_ORIGIN = "https://jupr-a1b2c3d4-pickleballattrespalapas1.vercel.app"
FLY_IMAGE = "registry.fly.io/juprleagues-api-staging@sha256:123"


def test_manual_book_covers_every_partial_page_and_manifest_entry_once() -> None:
    assert check_book() == []


def test_manual_book_guard_rejects_missing_duplicate_stale_and_manifest_rows(
    tmp_path: Path,
) -> None:
    text = BOOK_PATH.read_text(encoding="utf-8")
    text = text.replace("| `leaderboards` | `Pending` | — | N/A | — |\n", "")
    text = text.replace(
        "| `fixture:support-intake` | — | — | — | `Pending` |\n", ""
    )
    text = text.replace(
        "| `manual:support-intake` | — | — | — | `Pending` | — |\n", ""
    )
    text += "\n| `home` | `Pass` | stale | N/A | test |\n"
    text += "| `players` | `Pending` | duplicate | N/A | test |\n"
    book = tmp_path / "book.md"
    book.write_text(text, encoding="utf-8")

    errors = check_book(book_path=book)
    assert any("leaderboards" in error for error in errors)
    assert any("home" in error for error in errors)
    assert any("players" in error for error in errors)
    assert any("support-intake" in error for error in errors)
    assert any("Deferred manual mutation ledger is missing" in error for error in errors)


def test_completion_mode_rejects_pending_placeholders_and_unresolved_prerequisites() -> None:
    errors = check_book_complete()
    assert any("Candidate identity is incomplete" in error for error in errors)
    assert any("Page evidence is not Pass" in error for error in errors)
    assert any("Automated wave is not Pass" in error for error in errors)
    assert any("Fixture cleanup is not Pass" in error for error in errors)
    assert any("order-27-tournament-ops" in error for error in errors)
    assert any("unchecked" in error for error in errors)


def _completed_book(text: str, candidate_sha: str) -> str:
    candidate_values = {
        "Application candidate Git SHA": f"`{candidate_sha}`",
        "Final stacked PR": "PR-1029",
        "Vercel preview URL": "https://jupr-git-staging-pickleballattrespalapas1.vercel.app",
        "Vercel deployment ID": VERCEL_ID,
        "Vercel immutable deployment origin": VERCEL_ORIGIN,
        "Fly staging image ref": FLY_IMAGE,
        "Deployment identity preflight artifact": (
            f"candidate={candidate_sha}; vercel={VERCEL_ID}; fly={FLY_IMAGE}; artifact=run-1029"
        ),
        "Schema inventory / migration head evidence": "schema-run-1",
        "Streamlit fallback URL / build": "streamlit-build-1",
        "Staging role accounts exercised": "admin, club-owner, viewer",
        "Session start / end": "2026-07-20T01:00Z / 2026-07-20T03:00Z",
        "Primary operator": "operator-1",
        "Witness / reviewer": "reviewer-1",
    }
    for field, value in candidate_values.items():
        text = text.replace(f"| {field} | — |", f"| {field} | {value} |")
    text = text.replace("- [ ]", "- [x]")
    text = text.replace(
        "UNRESOLVED — uniquely versioned canonical Tournament Ops operation-ledger migration required",
        "`supabase/migrations/20260719204600_tournament_ops_operations.sql`",
    )
    text = text.replace(
        "UNRESOLVED — dedicated staging-only Tournament Ops write gate required; broad tournament visibility is insufficient",
        "`JUPR_ENABLE_STAGING_NEXT_ADMIN_TOURNAMENT_OPS_WRITES=1`",
    )
    text = text.replace("BLOCKED — keep Tournament Ops writes closed", "Close Tournament Ops gate")
    text = text.replace("BLOCKED — no write until dedicated gate exists", "Close Tournament Ops gate")
    text = text.replace("preserve unresolved jobs", "preserve open jobs")

    text = BOOK_ROW_RE.sub(
        lambda match: (
            f"| `{match.group('key')}` | `Pass` | evidence-{match.group('key')} | "
            f"{match.group('recovery') if match.group('recovery').strip() == 'N/A' else 'Verified: recovery-1'} "
            "| operator-1 |"
        ),
        text,
    )
    text = WAVE_ROW_RE.sub(
        lambda match: (
            f"| `{match.group('wave')}` | {match.group('command')} | {match.group('inputs')} "
            "| `Pass` | run-1 | operator-1 |"
        ),
        text,
    )
    text = MANUAL_MUTATION_ROW_RE.sub(
        lambda match: (
            f"| `manual:{match.group('surface')}` | Verified: method=PATCH; "
            f"path=/manual/{match.group('surface')}; "
            f"resource={match.group('surface')}-fixture; prestate=state-original "
            "| Verified: status=200; projection=state=exercised,readback=true; "
            f"artifact=write-{match.group('surface')} "
            "| Verified: method=PATCH; "
            f"path=/manual/{match.group('surface')}/restore; status=200; "
            "projection=state=restored,readback=true; "
            f"artifact=restore-{match.group('surface')} "
            "| `Pass` | operator=operator-1; witness=reviewer-1 |"
        ),
        text,
    )
    text = FIXTURE_ROW_RE.sub(
        lambda match: (
            f"| `fixture:{match.group('scope')}` | ids-and-operation-key | owner-1 "
            "| Verified: cleanup-evidence-1 | `Pass` |"
        ),
        text,
    )
    text = MIGRATION_ROW_RE.sub(
        lambda match: (
            f"| `migration:{match.group('key')}` | {match.group('required')} | Verified "
            f"| migration-evidence-1 | {match.group('rollback')} | owner-1 |"
        ),
        text,
    )
    text = FLAG_ROW_RE.sub(
        lambda match: (
            f"| `flag:{match.group('key')}` | {match.group('staging')} | {match.group('production')} "
            f"| {match.group('disable')} | Verified disabled: flag-evidence-1 | owner-1 |"
        ),
        text,
    )
    return re.sub(r"\n{3,}", "\n\n", text)


def test_completion_mode_accepts_bound_all_pass_evidence(tmp_path: Path) -> None:
    candidate_sha = "a" * 40
    book = tmp_path / "book.md"
    book.write_text(
        _completed_book(BOOK_PATH.read_text(encoding="utf-8"), candidate_sha),
        encoding="utf-8",
    )
    assert check_book_complete(
        book_path=book,
        candidate_sha=candidate_sha,
        vercel_deployment_id=VERCEL_ID,
        vercel_deployment_origin=VERCEL_ORIGIN,
        fly_image_ref=FLY_IMAGE,
    ) == []


def test_completion_mode_rejects_automated_review_for_manual_only_staging_writes(
    tmp_path: Path,
) -> None:
    candidate_sha = "a" * 40
    completed = _completed_book(BOOK_PATH.read_text(encoding="utf-8"), candidate_sha)
    completed = re.sub(
        r"operator=operator-1; witness=reviewer-1",
        "operator=operator-1; review=automated",
        completed,
    )
    book = tmp_path / "book.md"
    book.write_text(completed, encoding="utf-8")

    errors = check_book_complete(
        book_path=book,
        candidate_sha=candidate_sha,
        vercel_deployment_id=VERCEL_ID,
        vercel_deployment_origin=VERCEL_ORIGIN,
        fly_image_ref=FLY_IMAGE,
    )
    assert any("distinct human identities" in error for error in errors)


def test_completion_mode_rejects_candidate_sha_mismatch(tmp_path: Path) -> None:
    candidate_sha = "a" * 40
    book = tmp_path / "book.md"
    book.write_text(
        _completed_book(BOOK_PATH.read_text(encoding="utf-8"), candidate_sha),
        encoding="utf-8",
    )
    errors = check_book_complete(
        book_path=book,
        candidate_sha="b" * 40,
        vercel_deployment_id=VERCEL_ID,
        vercel_deployment_origin=VERCEL_ORIGIN,
        fly_image_ref=FLY_IMAGE,
    )
    assert any("does not match required SHA" in error for error in errors)


def test_completion_mode_rejects_fabricated_bindings_and_false_success_prose(tmp_path: Path) -> None:
    candidate_sha = "a" * 40
    completed = _completed_book(BOOK_PATH.read_text(encoding="utf-8"), candidate_sha)
    completed = completed.replace("| Verified | migration-evidence-1 |", "| Not applied | migration-evidence-1 |", 1)
    completed = completed.replace("Verified disabled: flag-evidence-1", "Production enabled", 1)
    completed = completed.replace("Verified: recovery-1", "Failed", 1)
    completed = completed.replace("Verified: cleanup-evidence-1", "Failed", 1)
    completed = completed.replace(f"vercel={VERCEL_ID}", "vercel=dpl_fabricated", 1)
    book = tmp_path / "book.md"
    book.write_text(completed, encoding="utf-8")

    errors = check_book_complete(
        book_path=book,
        candidate_sha=candidate_sha,
        vercel_deployment_id=VERCEL_ID,
        vercel_deployment_origin=VERCEL_ORIGIN,
        fly_image_ref=FLY_IMAGE,
    )
    assert any("state must be exactly" in error for error in errors)
    assert any("Verified disabled" in error for error in errors)
    assert any("Recovery proof must" in error for error in errors)
    assert any("cleanup must be" in error for error in errors)
    assert any("different Vercel deployment ID" in error for error in errors)


def test_completion_mode_rejects_dispatch_identifier_mismatch(tmp_path: Path) -> None:
    candidate_sha = "a" * 40
    book = tmp_path / "book.md"
    book.write_text(_completed_book(BOOK_PATH.read_text(encoding="utf-8"), candidate_sha), encoding="utf-8")
    errors = check_book_complete(
        book_path=book,
        candidate_sha=candidate_sha,
        vercel_deployment_id="dpl_other",
        vercel_deployment_origin=(
            "https://jupr-other1234-pickleballattrespalapas1.vercel.app"
        ),
        fly_image_ref="registry.fly.io/other@sha256:456",
    )
    assert any("Vercel deployment ID does not match" in error for error in errors)
    assert any("immutable Vercel deployment origin does not match" in error for error in errors)
    assert any("Fly staging image ref does not match" in error for error in errors)


def test_completion_mode_rejects_reserved_prose_in_every_completion_field_family(
    tmp_path: Path,
) -> None:
    candidate_sha = "a" * 40
    base = _completed_book(BOOK_PATH.read_text(encoding="utf-8"), candidate_sha)
    substitutions = (
        ("| Primary operator | operator-1 |", "| Primary operator | Pending approval |"),
        ("evidence-leaderboards", "Blocked evidence"),
        ("Candidate SHA and clean integrated Order-28 tree", "Unresolved inputs"),
        ("prestate=state-original", "prestate=Blocked baseline"),
        ("ids-and-operation-key", "Failed fixture IDs"),
        ("migration-evidence-1", "Not applied migration evidence"),
        ("Verified disabled: flag-evidence-1", "Verified disabled: Failed flag evidence"),
    )
    for index, (old, new) in enumerate(substitutions):
        assert old in base
        book = tmp_path / f"reserved-{index}.md"
        book.write_text(base.replace(old, new, 1), encoding="utf-8")
        errors = check_book_complete(
            book_path=book,
            candidate_sha=candidate_sha,
            vercel_deployment_id=VERCEL_ID,
            vercel_deployment_origin=VERCEL_ORIGIN,
            fly_image_ref=FLY_IMAGE,
        )
        assert any("reserved incomplete/failure prose" in error for error in errors)


def test_completion_mode_rejects_unstructured_or_weak_manual_mutation_evidence(
    tmp_path: Path,
) -> None:
    candidate_sha = "a" * 40
    base = _completed_book(BOOK_PATH.read_text(encoding="utf-8"), candidate_sha)
    substitutions = (
        (
            "Verified: method=PATCH; path=/manual/support-intake; "
            "resource=support-intake-fixture; prestate=state-original",
            "Verified: anything",
            "route must use",
        ),
        (
            "Verified: status=200; projection=state=exercised,readback=true; "
            "artifact=write-support-intake",
            "Verified: status=404; projection=state=exercised; artifact=write-1",
            "write must use",
        ),
        (
            "Verified: status=200; projection=state=exercised,readback=true; "
            "artifact=write-support-intake",
            "Verified: status=204; projection=state=exercised; artifact=write-1",
            "write must use",
        ),
        (
            "Verified: method=PATCH; path=/manual/support-intake/restore; "
            "status=200; projection=state=restored,readback=true; "
            "artifact=restore-support-intake",
            "Verified: method=PATCH; path=/manual/support-intake/restore; "
            "status=200; projection=false; artifact=restore-1",
            "recovery must use",
        ),
        (
            "operator=operator-1; witness=reviewer-1",
            "operator=operator-1; witness=operator-1",
            "distinct human identities",
        ),
    )
    for index, (old, new, expected_error) in enumerate(substitutions):
        assert old in base
        book = tmp_path / f"manual-structure-{index}.md"
        book.write_text(base.replace(old, new, 1), encoding="utf-8")
        errors = check_book_complete(
            book_path=book,
            candidate_sha=candidate_sha,
            vercel_deployment_id=VERCEL_ID,
            vercel_deployment_origin=VERCEL_ORIGIN,
            fly_image_ref=FLY_IMAGE,
        )
        assert any(expected_error in error for error in errors)
