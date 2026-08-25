from __future__ import annotations

import re
from pathlib import Path


PANEL = Path("apps/web/app/admin/league-manager/live/LeagueLiveRoundPanel.tsx")


def _source() -> str:
    return PANEL.read_text(encoding="utf-8")


def _visible_label_position(source: str, label: str) -> int:
    """Find a label expressed as JSX text or as a quoted step label."""

    candidates = (
        f">{label}<",
        f'"{label}"',
        f"'{label}'",
        f"`{label}`",
    )
    positions = [source.find(candidate) for candidate in candidates]
    positions = [position for position in positions if position >= 0]
    assert positions, f"Missing visible League Live workflow label: {label!r}"
    return min(positions)


def _assert_any(source: str, alternatives: tuple[str, ...], requirement: str) -> None:
    assert any(value in source for value in alternatives), requirement


def test_live_session_has_six_visible_ordered_operator_steps() -> None:
    source = _source()
    labels = (
        "Setup",
        "Players",
        "Courts and Preview",
        "Score Entry with Review",
        "Movement",
        "Repeat or Finish",
    )

    positions = [_visible_label_position(source, label) for label in labels]
    assert positions == sorted(positions), "League Live steps must appear in operator order"
    assert re.search(r"<nav\b[^>]*aria-label=", source), (
        "The step sequence must be exposed as accessible navigation"
    )
    assert "aria-current" in source, "The active League Live step must be identified"


def test_players_step_pastes_comma_or_newline_names_and_rejects_duplicates() -> None:
    source = _source()

    assert "Paste players" in source
    assert re.search(r"<textarea\b", source), "Paste players must use a multiline input"
    assert re.search(r"\b(?:parse|split)\w*(?:Player|Name)", source, re.I), (
        "Pasted names need a dedicated parser rather than court-text parsing"
    )
    regex_splitters = re.findall(r"split\(\s*/([^/]*)/[gimsuy]*\s*\)", source)
    parses_comma_and_newline = bool(
        re.search(r"replace\(\s*/,/g\s*,\s*[\"']\\n[\"']\s*\).*?split\(\s*[\"']\\n[\"']\s*\)", source, re.S)
        or any("," in pattern and r"\n" in pattern for pattern in regex_splitters)
    )
    assert parses_comma_and_newline, "Paste parsing must accept both commas and newlines"
    assert "duplicate" in source.lower(), "Duplicate pasted names need an operator-visible error"
    assert "new Set" in source or "new Map" in source
    _assert_any(
        source,
        (".toLowerCase()", ".toLocaleLowerCase()"),
        "Duplicate detection and player matching must be case-insensitive",
    )


def test_players_step_resolves_existing_players_and_creates_missing_non_roster_players() -> None:
    source = _source()

    assert re.search(r"\b(?:resolve|match)\w*(?:Player|Name)", source, re.I), (
        "The Players step must resolve pasted names against existing club players"
    )
    assert re.search(r"\b(?:existing|matched)\b", source, re.I), (
        "The Players step must visibly distinguish resolved existing players"
    )
    assert re.search(r"\b(?:missing|new player|not found)\b", source, re.I), (
        "The Players step must visibly distinguish names that need player creation"
    )
    _assert_any(
        source,
        ("createMissingPlayer", "createMissingPlayers", "createGuest"),
        "Missing pasted names need an explicit player-creation path",
    )
    assert re.search(r"starting\s+JUPR", source, re.I), (
        "Every new player must receive an explicit starting JUPR"
    )
    assert re.search(r"non[- ]roster", source, re.I), (
        "The Players step must explain that existing and newly created non-roster players can attend"
    )


def test_flex_starts_fresh_while_set_resumes_saved_roster_and_courts() -> None:
    source = _source()

    assert 'participationMode === "flex"' in source
    assert "setAttendeePlayerIds([])" in source
    assert 'setCourts([{ court: "1", formatType: "4-player", playerNames: "" }])' in source
    assert re.search(r"new Flex session resets attendance", source, re.I)
    assert re.search(r"rebuilds rating-seeded pods/courts", source, re.I)

    assert 'participationMode === "set"' in source
    assert re.search(r"Set participation", source, re.I)
    assert re.search(r"retain(?:s)? (?:its )?saved pods and positions", source, re.I)


def test_score_entry_requires_an_explicit_team_and_score_review_before_submit() -> None:
    source = _source()

    assert "Score Entry with Review" in source
    _assert_any(
        source,
        ("Review scores", "Review entered scores"),
        "Operators need a distinct score-review action",
    )
    _assert_any(
        source,
        ("Edit scores", "Back to score entry"),
        "The review must let the operator return to edit a score",
    )
    _assert_any(
        source,
        ("Publish reviewed scores", "Confirm scores", "Scores reviewed"),
        "The review must require an explicit confirmation",
    )
    assert re.search(r"(?:scoreReview|scoresReviewed|scoresConfirmed)", source), (
        "Reviewed-score state must be represented separately from score validity"
    )
    assert re.search(r"set(?:ScoreReview|ScoresReviewed|ScoresConfirmed)\s*\(\s*false\s*\)", source), (
        "Editing a score must invalidate the previous review"
    )
    assert re.search(r"disabled=\{[^}]*!(?:scoreReview|scoresReviewed|scoresConfirmed)", source), (
        "Publishing or advancing must stay disabled until scores are reviewed"
    )


def test_movement_then_repeat_or_finish_are_explicit_steps() -> None:
    source = _source()

    assert "Movement" in source
    assert "Preview movement" in source
    assert "Next-round movement plan" in source
    assert "Repeat or Finish" in source
    _assert_any(
        source,
        ("Start next round", "Repeat round workflow"),
        "After movement, the operator needs an explicit path to the next round",
    )
    _assert_any(
        source,
        ("Finish session", "Complete session"),
        "After movement, the operator needs an explicit path to finish the session",
    )
    assert re.search(r"(?:set\w*Step|goTo\w*Step)\s*\(", source), (
        "Repeat/finish actions must drive the guided workflow state"
    )


def test_next_round_reuses_published_context_and_opens_score_entry() -> None:
    source = _source()
    start_next_round = re.search(
        r"async function startNextRound\(\)\s*\{(?P<body>.*?)\n  \}\n\n  function updateCourt",
        source,
        re.S,
    )
    assert start_next_round, "Start next round needs a dedicated guarded transition"
    body = start_next_round.group("body")

    assert "nextRoundMatchDate(roundHistory, lastPublishedRound, matchDate)" in body, (
        "The next round must inherit the already confirmed league-night date"
    )
    assert "await requestRoundPreview(nextRound)" in body, (
        "The saved moved courts must be turned into the next scoring preview automatically"
    )
    assert "setScores(emptyScoresForPreview(payload, matchStructure))" in body
    assert "setWorkflowStep(4)" in body, (
        "Starting the next round must open Score Entry instead of restarting Setup"
    )
    assert "Enter scores when play begins" in body
    assert "previousRoundWasPublished" in source
    assert "await requestRoundPreview(sessionCurrentRound, savedNextCourts)" in source, (
        "Reloading an already advanced session must also recover directly into score entry"
    )
    assert "Round ${sessionCurrentRound} resumed with the approved movement" in source


def test_uncertain_writes_do_not_advance_the_workflow_as_if_they_succeeded() -> None:
    source = _source()

    assert source.count('if (completion.status !== "success") return completion;') >= 2, (
        "Session creation and round publishing must not advance on an uncertain write"
    )
    assert "setRoundPublished(true)" in source and "setWorkflowStep(6)" in source, (
        "A successful exact round-publish recovery must update the final workflow step"
    )


def test_paste_edits_are_re_resolved_and_ambiguous_player_names_fail_closed() -> None:
    source = _source()

    assert "pasteResolutionCurrent" in source
    assert "pastedSelectedPlayerIds" in source
    assert "changePastedPlayerText" in source
    assert "!pasteResolutionCurrent" in source
    assert 'status: "ambiguous"' in source
    assert "matches multiple club player IDs" in source
    assert "· ID {player.id}" in source


def test_round_context_scores_and_courts_are_validated_before_publish() -> None:
    source = _source()

    assert '/^(0|[1-9]\\d*)$/' in source, "Both score fields must contain explicit base-10 integers"
    assert "roundContextValid" in source
    assert "matchDateValid" in source
    assert "Court numbers must be contiguous starting at 1" in source
    assert "invalidateRoundDraft" in source
    assert "Date required" in source


def test_persisted_sessions_support_pause_resume_final_round_and_local_draft_recovery() -> None:
    source = _source()

    assert "persistedPublishedRoundNumber" in source
    assert "resumePausedSession" in source
    assert "pauseActiveSession" in source
    assert "Resume session" in source
    assert "Pause session" in source
    assert "StoredLeagueLiveRoundDraft" in source
    assert "beforeunload" in source
    assert "restoreStoredRoundDraft" in source
    assert "guestPlayers" in source


def test_movement_review_shows_authoritative_post_change_courts_and_bench() -> None:
    source = _source()

    assert "Authoritative next-round courts" in source
    assert "Approved next-round roster and courts" in source
    assert "movementPlan.next_courts" in source
    assert "movementPlan.bench" in source
