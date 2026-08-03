from __future__ import annotations

from pathlib import Path
from textwrap import indent


def read(path: str) -> str:
    return Path(path).read_text()


def write(path: str, text: str) -> None:
    Path(path).write_text(text)


def replace_once(path: str, old: str, new: str) -> None:
    text = read(path)
    if old not in text:
        raise SystemExit(f"Expected source fragment not found in {path}: {old[:120]!r}")
    write(path, text.replace(old, new, 1))


def replace_block(path: str, start: str, end: str, replacement: str) -> None:
    text = read(path)
    start_index = text.find(start)
    if start_index < 0:
        raise SystemExit(f"Start marker not found in {path}: {start!r}")
    end_index = text.find(end, start_index)
    if end_index < 0:
        raise SystemExit(f"End marker not found in {path}: {end!r}")
    write(path, text[:start_index] + replacement + text[end_index:])


def insert_after_in_function(
    path: str,
    function_start: str,
    needle: str,
    replacement: str,
) -> None:
    text = read(path)
    function_index = text.find(function_start)
    if function_index < 0:
        raise SystemExit(f"Function marker not found in {path}: {function_start!r}")
    needle_index = text.find(needle, function_index)
    if needle_index < 0:
        raise SystemExit(f"Scoped source fragment not found in {path}: {needle!r}")
    write(
        path,
        text[:needle_index] + replacement + text[needle_index + len(needle):],
    )


DOMAIN = "jupr_app/domain/adaptive_play_engine.py"
replace_once(
    DOMAIN,
    (
        '    row = _get_round(next_event, round_number)\n'
        '    if str(row.get("status")) not in {"active", "preview"}:\n'
        '        raise ValueError("Only an active round can be marked played.")\n'
    ),
    (
        '    row = _get_round(next_event, round_number)\n'
        '    round_status = str(row.get("status"))\n'
        '    if round_status == "played":\n'
        '        return next_event\n'
        '    if round_status not in {"active", "preview"}:\n'
        '        raise ValueError("Only an active round can be marked played.")\n'
    ),
)

ADMIN_SERVICE = "jupr_app/services/admin_play_generator_service.py"
replace_once(
    ADMIN_SERVICE,
    "    mark_generator_round_played,\n",
    "    mark_generator_round_played,\n    normalize_scoring_mode,\n",
)
replace_block(
    ADMIN_SERVICE,
    "def mark_play_generator_round_played(\n",
    "def skip_play_generator_round(\n",
    '''def mark_play_generator_round_played(
    supabase: Any,
    *,
    club_id: str,
    session_key: str,
    round_number: int,
    expected_version: str,
    actor_email: str,
    actor_role: str,
    source: str,
) -> dict[str, Any]:
    before = _live_row(supabase, club_id=str(club_id), session_key=str(session_key))
    event = mark_generator_round_played(
        _event_from_state(_state(before)),
        round_number=int(round_number),
    )
    event = advance_generator_event(event)
    row_status = "completed" if str(event.get("status") or "") == "completed" else None
    updated = _persist_event(
        supabase,
        before=before,
        event=event,
        expected_version=expected_version,
        status=row_status,
    )
    session = _session_payload(updated)
    _audit(
        supabase,
        club_id=club_id,
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="mark_play_generator_round_played",
        entity_id=session_key,
        before_json={"session": _session_payload(before)},
        after_json={"session": session, "round_number": int(round_number)},
        source=source,
    )
    return {"ok": True, "mode": "play_generator_round_played", "session": session}


''',
)
insert_after_in_function(
    ADMIN_SERVICE,
    "def publish_play_generator_matches(\n",
    '    event = _event_from_state(state)\n',
    (
        '    event = _event_from_state(state)\n'
        '    if normalize_scoring_mode(event.get("scoringMode")) != "scored":\n'
        '        raise ValueError(\n'
        '            "Official publishing is unavailable for unscored Round-Robin sessions."\n'
        '        )\n'
    ),
)

PUBLIC_SERVICE = "jupr_app/services/public_play_generator_service.py"
replace_block(
    PUBLIC_SERVICE,
    "def mark_public_play_generator_round_played(\n",
    "def skip_public_play_generator_round(\n",
    '''def mark_public_play_generator_round_played(
    supabase: Any,
    *,
    club_id: str,
    session_key: str,
    round_number: int,
    edit_token: str,
    expected_version: int,
    idempotency_key: str,
    requester_hash: str,
) -> dict[str, Any]:
    def mutate(event: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        next_event = mark_generator_round_played(
            event,
            round_number=int(round_number),
        )
        next_event = advance_generator_event(next_event)
        if str(next_event.get("status") or "") == "completed":
            now = _now_iso()
            return next_event, {"status": "completed", "completed_at": now}
        return next_event, {}

    return _run_mutation(
        supabase,
        club_id=club_id,
        session_key=session_key,
        edit_token=edit_token,
        expected_version=expected_version,
        idempotency_key=idempotency_key,
        requester_hash=requester_hash,
        action="played",
        request_payload={"round_number": int(round_number)},
        mutate=mutate,
    )


''',
)

ADMIN_RUNNER = "apps/web/app/admin/play-generators/GeneratorRoundRunner.tsx"
replace_block(
    ADMIN_RUNNER,
    "  async function markRoundPlayed(): Promise<void> {\n",
    "  async function skipRound(): Promise<void> {\n",
    '''  async function markRoundPlayed(): Promise<void> {
    if (!session || !round || generatorKind !== "round_robin" || scoredSession) return;
    setBusy(true);
    setMessage(null);
    try {
      const playedPayload = await requestJson<MutationResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(
          sessionKey
        )}/rounds/${roundNumber}/played`,
        {
          method: "POST",
          body: JSON.stringify({
            expected_version: session.version,
            idempotency_key: operationKey("played")
          })
        }
      );
      if (!playedPayload.session) {
        throw new Error("Round marked played without a refreshed session.");
      }
      applySession(playedPayload.session);
      if (playedPayload.session.status === "completed") {
        setMessage("Session completed.");
        router.refresh();
        return;
      }
      const nextRound = playedPayload.session.current_round_number || roundNumber + 1;
      router.push(roundPath(generatorKind, sessionKey, nextRound));
      router.refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to mark the round played.");
    } finally {
      setBusy(false);
    }
  }


''',
)

PUBLIC_RUNNER = "apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner.tsx"
replace_block(
    PUBLIC_RUNNER,
    "  async function markRoundPlayed(): Promise<void> {\n",
    "  async function skipRound(): Promise<void> {\n",
    '''  async function markRoundPlayed(): Promise<void> {
    if (!session || !round || generatorKind !== "round_robin" || scoredSession) return;
    setBusy(true);
    setMessage(null);
    try {
      const playedPayload = await requestJson<MutationResponse>(
        `/clubs/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(
          sessionKey
        )}/rounds/${roundNumber}/played`,
        {
          method: "POST",
          body: JSON.stringify({
            edit_token: editToken,
            expected_version: Number(session.version),
            idempotency_key: operationKey("played")
          })
        }
      );
      if (!playedPayload.session) {
        throw new Error("Round marked played without a refreshed session.");
      }
      applySession(playedPayload.session);
      if (playedPayload.session.status === "completed") {
        setMessage("Session completed.");
        router.refresh();
        return;
      }
      const nextRound = playedPayload.session.current_round_number || roundNumber + 1;
      router.push(roundPath(generatorKind, clubId, sessionKey, nextRound));
      router.refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to mark the round played.");
    } finally {
      setBusy(false);
    }
  }


''',
)


def add_runner_completion(path: str, *, public: bool) -> None:
    copy_text = (
        "All scheduled rounds are complete. This public session remains unrated."
        if public
        else "All scheduled rounds are complete. Review the saved session history below."
    )
    marker = "      </article>\n\n      <article style={cardStyle}>"
    replacement = (
        "      </article>\n\n"
        '      {session.status === "completed" ? (\n'
        '        <article style={{ ...cardStyle, background: "#ecfdf5", borderColor: "#86efac" }}>\n'
        '          <h2 style={{ marginTop: 0 }}>Session complete</h2>\n'
        f'          <p style={{{{ marginBottom: 0, color: "#166534" }}}}>{copy_text}</p>\n'
        "        </article>\n"
        "      ) : null}\n\n"
        "      <article style={cardStyle}>"
    )
    replace_once(path, marker, replacement)


add_runner_completion(ADMIN_RUNNER, public=False)
add_runner_completion(PUBLIC_RUNNER, public=True)

admin_text = read(ADMIN_RUNNER)
panel_start = admin_text.find(
    '      <article style={cardStyle}>\n'
    '        <h2 style={{ marginTop: 0 }}>Official results</h2>'
)
panel_end = admin_text.find("\n\n      {message ? (", panel_start)
if panel_start < 0 or panel_end < 0:
    raise SystemExit("Admin official-results panel boundary was not found.")
panel = admin_text[panel_start:panel_end]
wrapped_panel = (
    "      {scoredSession ? (\n"
    + indent(panel, "  ")
    + "\n      ) : null}"
)
write(ADMIN_RUNNER, admin_text[:panel_start] + wrapped_panel + admin_text[panel_end:])

STANDINGS_FILES = (
    "apps/web/app/admin/play-generators/GeneratorStandings.tsx",
    "apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorStandings.tsx",
)
for standings_path in STANDINGS_FILES:
    table_line = (
        "      <PlayGeneratorStandingsTable "
        "rows={session.standings || []} sortMode={sortMode} />\n"
    )
    completion = (
        table_line
        + '      {session.status === "completed" ? (\n'
        + '        <article style={{ ...cardStyle, background: "#ecfdf5", borderColor: "#86efac" }}>\n'
        + '          <h2 style={{ marginTop: 0 }}>Session complete</h2>\n'
        + '          <p style={{ marginBottom: 0, color: "#166534" }}>\n'
        + "            The final cumulative standings are preserved above.\n"
        + "          </p>\n"
        + "        </article>\n"
        + "      ) : null}\n"
    )
    replace_once(standings_path, table_line, completion)

TEST_CONTENT = r'''from pathlib import Path

import pytest

from jupr_app.domain.adaptive_play_engine import (
    advance_generator_event,
    create_generator_preview,
    generator_event_standings,
    history_before_round,
    mark_generator_round_played,
    save_generator_round,
    start_generator_event,
)

ROOT = Path(__file__).resolve().parents[1]


def _matches(round_row: dict) -> list[dict]:
    if round_row.get("matches"):
        return list(round_row.get("matches") or [])
    return [
        match
        for court in round_row.get("courts") or []
        for match in court.get("matches") or []
    ]


def _score_round(event: dict, round_number: int) -> dict:
    round_row = next(row for row in event["rounds"] if row["number"] == round_number)
    return save_generator_round(
        event,
        round_number=round_number,
        scores=[
            {"match_id": match["id"], "score_a": 11, "score_b": 7}
            for match in _matches(round_row)
        ],
    )


def _function_block(text: str, start: str, end: str) -> str:
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[start_index:end_index]


def test_scoring_mode_changes_preview_fingerprint_and_defaults_to_scored() -> None:
    common = dict(
        generator_kind="round_robin",
        play_format="singles",
        title="Fingerprint",
        participant_names=["A", "B", "C", "D"],
        total_rounds=2,
        court_count=2,
    )
    scored = create_generator_preview(**common)
    unscored = create_generator_preview(**common, scoring_mode="unscored")
    assert scored["scoringMode"] == "scored"
    assert unscored["scoringMode"] == "unscored"
    assert scored["previewFingerprint"] != unscored["previewFingerprint"]


def test_scored_round_robin_lifecycle_saves_standings_and_completes() -> None:
    event = start_generator_event(
        create_generator_preview(
            generator_kind="round_robin",
            play_format="singles",
            title="Scored lifecycle",
            participant_names=["A", "B", "C", "D"],
            total_rounds=2,
            court_count=2,
        )
    )
    event = _score_round(event, 1)
    assert event["rounds"][0]["status"] == "saved"
    assert any(row["matches"] > 0 for row in generator_event_standings(event))
    event = advance_generator_event(event)
    assert event["currentRoundNumber"] == 2
    assert event["rounds"][1]["status"] == "active"
    event = _score_round(event, 2)
    event = advance_generator_event(event)
    assert event["status"] == "completed"


def test_unscored_round_played_is_idempotent_and_preserves_history() -> None:
    event = start_generator_event(
        create_generator_preview(
            generator_kind="round_robin",
            play_format="singles",
            title="Unscored lifecycle",
            participant_names=["A", "B", "C"],
            total_rounds=2,
            court_count=1,
            scoring_mode="unscored",
        )
    )
    with pytest.raises(ValueError, match="Round Played"):
        save_generator_round(event, round_number=1, scores=[])
    played = mark_generator_round_played(event, round_number=1)
    replay = mark_generator_round_played(played, round_number=1)
    assert replay == played
    assert played["rounds"][0]["status"] == "played"
    assert sum(history_before_round(played, 2)["games"].values()) == 2
    event = advance_generator_event(played)
    assert event["currentRoundNumber"] == 2
    event = mark_generator_round_played(event, round_number=2)
    event = advance_generator_event(event)
    assert event["status"] == "completed"
    assert generator_event_standings(event) == []


def test_ladder_rejects_unscored_mode() -> None:
    with pytest.raises(ValueError, match="requires scored rounds"):
        create_generator_preview(
            generator_kind="ladder",
            play_format="doubles",
            title="Bad ladder",
            participant_names=["A", "B", "C", "D"],
            total_rounds=3,
            court_count=1,
            scoring_mode="unscored",
        )


def test_round_played_api_is_one_durable_navigation_action() -> None:
    admin_routes = (ROOT / "services/api/admin_play_generator_routes.py").read_text()
    public_routes = (ROOT / "services/api/public_play_generator_routes.py").read_text()
    admin_service = (ROOT / "jupr_app/services/admin_play_generator_service.py").read_text()
    public_service = (ROOT / "jupr_app/services/public_play_generator_service.py").read_text()

    admin_block = _function_block(
        admin_service,
        "def mark_play_generator_round_played(",
        "def skip_play_generator_round(",
    )
    public_block = _function_block(
        public_service,
        "def mark_public_play_generator_round_played(",
        "def skip_public_play_generator_round(",
    )
    for block in (admin_block, public_block):
        assert "mark_generator_round_played" in block
        assert "advance_generator_event" in block

    assert "run_durable_admin_operation" in admin_routes
    assert 'operation_type="mark_round_played"' in admin_routes
    assert "idempotency_key=payload.idempotency_key" in admin_routes
    assert 'action="played"' in public_service
    assert "idempotency_key=idempotency_key" in public_service
    assert "/played" in admin_routes
    assert "/played" in public_routes
    assert "scoring_mode" in admin_routes
    assert "scoring_mode" in public_routes
    assert (
        "Official publishing is unavailable for unscored Round-Robin sessions."
        in admin_service
    )


def test_public_and_staff_component_navigation_contracts() -> None:
    admin_setup = (
        ROOT / "apps/web/app/admin/play-generators/GeneratorWorkspace.tsx"
    ).read_text()
    public_setup = (
        ROOT
        / "apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorWorkspace.tsx"
    ).read_text()
    admin_runner = (
        ROOT / "apps/web/app/admin/play-generators/GeneratorRoundRunner.tsx"
    ).read_text()
    public_runner = (
        ROOT
        / "apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner.tsx"
    ).read_text()

    for setup in (admin_setup, public_setup):
        assert "Unscored — mark each round played" in setup
        assert "standingsSort" in setup
        assert "scoringMode" in setup

    admin_ui = _function_block(
        admin_runner,
        "  async function markRoundPlayed",
        "  async function skipRound",
    )
    public_ui = _function_block(
        public_runner,
        "  async function markRoundPlayed",
        "  async function skipRound",
    )
    for block in (admin_ui, public_ui):
        assert "/played" in block
        assert "/advance" not in block
        assert "current_round_number" in block

    for runner in (admin_runner, public_runner):
        assert "Round Played" in runner
        assert "View standings and continue" in runner
        assert "Session complete" in runner

    assert "Boolean(editToken)" in public_runner
    assert "{scoredSession ? (" in admin_runner
    assert "Official results" in admin_runner


def test_standings_pages_own_scored_progression_and_completion() -> None:
    admin = (
        ROOT / "apps/web/app/admin/play-generators/GeneratorStandings.tsx"
    ).read_text()
    public = (
        ROOT
        / "apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorStandings.tsx"
    ).read_text()
    for text in (admin, public):
        assert "Continue to Round" in text
        assert "/advance" in text
        assert "This unscored Round-Robin does not use standings." in text
        assert "Session complete" in text
'''

write("tests/test_round_robin_unscored_flow.py", TEST_CONTENT)

DOCS = "docs/round_robin_scoring_modes.md"
docs = read(DOCS)
durability_section = '''
## Durability and completion

**Round Played** is one durable, idempotent operation: it records the round and
advances to the next round, or completes the session on the final round.
Retrying the same operation cannot advance twice. Skipped rounds remain distinct
from played rounds.

Scored sessions always move from the saved round results to the full cumulative
standings before the organizer can continue. The final standings lead to a clear
session-complete state. Official match publishing is available only for scored
staff sessions.
'''
if "## Durability and completion" not in docs:
    write(DOCS, docs.rstrip() + "\n\n" + durability_section.lstrip())

CHECKLIST = '''# Round-Robin Generator manual acceptance checklist

Use the exact staging candidate and deployment named in the acceptance handoff.

## PRR-12 — Scored flow

- [ ] Create a scored Round Robin.
- [ ] Confirm ranking can be set to Total wins, Total points, or Point differential.
- [ ] Start the session and enter all current-round scores.
- [ ] Save scores and confirm the individual Round Results are shown.
- [ ] Select **View standings and continue**.
- [ ] Confirm the full cumulative Standings are shown before any next-round action.
- [ ] Confirm Standings include rank, games played, wins, losses, points for,
      points against, and differential.
- [ ] Continue from Standings and confirm the next round opens.
- [ ] Confirm every scored round page retains a Standings link.

## PRR-13 — Unscored flow

- [ ] Create an unscored Round Robin.
- [ ] Confirm ranking controls disappear during setup.
- [ ] Confirm schedule preview, CSV, and PDF remain available.
- [ ] Start the session and confirm no score fields, results table, or standings
      are shown.
- [ ] Select **Round Played** once and confirm the next round opens directly.
- [ ] Refresh and use browser Back; confirm the played round and current round
      remain correct.
- [ ] Confirm **Skip round** remains distinct from **Round Played**.
- [ ] Confirm adaptive add, remove, reorder, and substitute actions still work.
- [ ] In staff mode, confirm official publishing is unavailable.
- [ ] Open a view-only public link and confirm organizer controls are absent.

## PRR-14 — Final-round completion

- [ ] Finish the final scored round through Round Results and Full Standings.
- [ ] Select **Finish session** from Standings and confirm a clear Session complete
      state with final standings preserved.
- [ ] Finish the final unscored round with **Round Played** and confirm a clear
      Session complete state without competitive standings.
- [ ] Confirm refresh and Back preserve the completed session.
'''
write("docs/round_robin_manual_acceptance_checklist.md", CHECKLIST)
