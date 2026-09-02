from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_staging_header_shows_the_exact_deployment_commit_identity() -> None:
    layout = (ROOT / "apps/web/app/layout.tsx").read_text(encoding="utf-8")
    header = (ROOT / "apps/web/components/PublicSiteHeader.tsx").read_text(
        encoding="utf-8"
    )

    assert "process.env.VERCEL_GIT_COMMIT_SHA" in layout
    assert "/^[0-9a-f]{40}$/.test(value)" in layout
    assert "const stagingBuildSha = isStaging ? deploymentGitSha() : null" in layout
    assert "stagingBuildSha={stagingBuildSha}" in layout

    assert "{isStaging ? (" in header
    assert "stagingBuildSha?.slice(0, 7).toUpperCase()" in header
    assert 'BUILD {shortBuildSha || "UNAVAILABLE"}' in header
    assert 'data-staging-build-sha={stagingBuildSha || "unavailable"}' in header
    assert "Staging deployment commit ${stagingBuildSha}" in header


def test_build_identity_markup_is_nested_inside_the_staging_only_branch() -> None:
    layout = (ROOT / "apps/web/app/layout.tsx").read_text(encoding="utf-8")
    header = (ROOT / "apps/web/components/PublicSiteHeader.tsx").read_text(
        encoding="utf-8"
    )

    assert "const stagingBuildSha = isStaging ? deploymentGitSha() : null" in layout
    staging_branch_start = header.index("{isStaging ? (")
    staging_branch_end = header.index(") : null}", staging_branch_start)
    staging_markup = header[staging_branch_start:staging_branch_end]
    assert 'BUILD {shortBuildSha || "UNAVAILABLE"}' in staging_markup
    assert header.count('BUILD {shortBuildSha || "UNAVAILABLE"}') == 1
