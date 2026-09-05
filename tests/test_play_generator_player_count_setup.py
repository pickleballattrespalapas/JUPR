from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_generator_setup_uses_player_count_and_previous_picker_pattern() -> None:
    component = read("apps/web/components/GeneratorRosterSetup.tsx")
    public = read("apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorWorkspace.tsx")
    admin = read("apps/web/app/admin/play-generators/GeneratorWorkspace.tsx")

    for text in (public, admin):
        assert "GeneratorRosterSetup" in text
        assert "targetCount" in text
        assert "automaticSetup.totalRounds" in text
        assert "automaticSetup.courtCount" in text
        assert "Planned rounds" not in text
        assert "Available courts" not in text
        assert "Club player ID optional" not in text

    assert "Number of players" in component
    assert "Automatic setup" in component
    assert "Search club players" in component
    assert "Type at least 2 letters, then add a player" in component
    assert "Players ({participantNames.length} of {targetCount})" in component
    assert "One player per line, in starting order" in component
    assert "The order below sets the starting order and who gets the first bye" in component


def test_generator_setup_has_deterministic_auto_shape_rules() -> None:
    component = read("apps/web/components/GeneratorRosterSetup.tsx")
    assert "count % 2 === 0 ? count - 1 : count" in component
    assert "const uniquePartnerPairs = (count * (count - 1)) / 2" in component
    assert "const partnerPairsPerRound = courtCount * 2" in component
    assert "const courtCount = ladderCourtCount(count, playFormat)" in component
    assert "totalRounds: 4" in component
    assert "Math.min(50" in component
    assert "recommendedMixedCourtSetup" in component
    assert "mixedRoundCount" in component
    assert "Doubles courts" in component
    assert "Singles courts" in component


def test_admin_preserves_official_links_only_for_complete_linked_roster() -> None:
    admin = read("apps/web/app/admin/play-generators/GeneratorWorkspace.tsx")
    assert "const allLinked = orderedIds.every" in admin
    assert "player_ids: allLinked ? orderedIds : []" in admin


def test_public_passes_selected_player_links_by_name() -> None:
    public = read("apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorWorkspace.tsx")
    assert "participant_player_ids: participantPlayerIds" in public
    assert "linkedPlayerIds[normalizeRosterName(name)]" in public
