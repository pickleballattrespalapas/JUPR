from jupr_app.domain.tournament_public_references import (
    build_public_tournament_reference,
    public_tournament_reference_matches,
)


def test_public_tournament_reference_is_stable_and_hides_source_identifier():
    reference = build_public_tournament_reference(
        tournament_id="tournament-internal-1",
        namespace="partner-board-selection",
        source_id="selection-internal-99",
    )

    assert reference.startswith("tr_")
    assert "selection-internal-99" not in reference
    assert reference == build_public_tournament_reference(
        tournament_id="tournament-internal-1",
        namespace="partner-board-selection",
        source_id="selection-internal-99",
    )
    assert public_tournament_reference_matches(
        reference,
        tournament_id="tournament-internal-1",
        namespace="partner-board-selection",
        source_id="selection-internal-99",
    )


def test_public_tournament_reference_is_bound_to_tournament_namespace_and_source():
    reference = build_public_tournament_reference(
        tournament_id="tournament-1",
        namespace="partner-board-selection",
        source_id="selection-1",
    )

    assert not public_tournament_reference_matches(
        reference,
        tournament_id="tournament-2",
        namespace="partner-board-selection",
        source_id="selection-1",
    )
    assert not public_tournament_reference_matches(
        reference,
        tournament_id="tournament-1",
        namespace="roster-entry",
        source_id="selection-1",
    )
    assert not public_tournament_reference_matches(
        reference,
        tournament_id="tournament-1",
        namespace="partner-board-selection",
        source_id="selection-2",
    )
