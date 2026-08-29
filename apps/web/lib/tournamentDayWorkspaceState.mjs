const DAY_ACTION_CONFIRMATIONS = Object.freeze({
  activate_day: "ACTIVATE DAY",
  activate_draw: "ACTIVATE DRAW",
  pause_draw: "PAUSE DRAW",
  resume_draw: "RESUME DRAW",
  auto_fill_courts: "AUTO FILL COURTS",
  assign_next_court: "ASSIGN NEXT OPEN COURT",
  assign_game_to_court: "ASSIGN GAME TO COURT",
  reserve_game_for_court: "WAIT FOR SELECTED COURT",
  requeue_game: "RETURN GAME TO QUEUE",
  move_game_to_court: "MOVE GAME TO COURT",
  score_and_release: "SAVE SCORE AND RELEASE COURT",
  correct_completed_score: "CORRECT COMPLETED SCORE",
  record_non_played_result: "RECORD NON-PLAYED RESULT",
  generate_playoffs: "GENERATE PLAYOFFS",
  close_day: "CLOSE TOURNAMENT DAY"
});

const DAY_NOT_STARTED_STATES = new Set(["", "DRAFT", "NOT_STARTED", "INACTIVE"]);

export function dayActionConfirmation(action) {
  const confirmation = DAY_ACTION_CONFIRMATIONS[String(action || "")];
  if (!confirmation) throw new Error("Unsupported tournament day action.");
  return confirmation;
}

export function dayRunHasStarted(state) {
  return !DAY_NOT_STARTED_STATES.has(String(state || "").trim().toUpperCase());
}

export function dayRunAcceptsLiveCommands(state) {
  return String(state || "").trim().toUpperCase() === "ACTIVE";
}

export function visibleServerQueue(queue, drawId) {
  const rows = Array.isArray(queue) ? queue : [];
  if (!drawId || drawId === "all") return rows;
  return rows.filter((row) => String(row?.draw_id || "") === String(drawId));
}

export function readyActiveDrawQueue(queue, draws) {
  const activeDrawIds = new Set(
    (Array.isArray(draws) ? draws : [])
      .filter((draw) => String(draw?.activation_state || "").trim().toUpperCase() === "ACTIVE")
      .map((draw) => String(draw?.id || ""))
      .filter(Boolean)
  );
  return (Array.isArray(queue) ? queue : []).filter((row) => (
    activeDrawIds.has(String(row?.draw_id || ""))
    && String(row?.state || "").trim().toUpperCase() === "WAITING"
    && (!Array.isArray(row?.blockers) || row.blockers.length === 0)
  ));
}

export function resetFocusForDay(current, dayId) {
  return {
    dayId: String(dayId || ""),
    drawId: "",
    courtId: "",
    gameId: "",
    panel: current?.panel === "draws" ? "draws" : current?.panel === "queue" ? "queue" : "board"
  };
}

export function workspaceScopeKey(accessToken, tournamentId, dayId) {
  return [accessToken, tournamentId, dayId].map((value) => String(value || "")).join("\u0000");
}

export function retainedDayCommandStorageKey(clubId, tournamentId, dayId) {
  return `jupr_tournament_day_ops_pending_v1:${clubId}:${tournamentId}:${dayId}`;
}

export function advanceCountSelection(allowedCounts, defaultCount, currentSelection) {
  const allowed = new Set(
    (Array.isArray(allowedCounts) ? allowedCounts : [])
      .map((value) => Number(value))
      .filter((value) => Number.isInteger(value))
  );
  const current = Number(String(currentSelection ?? "").trim());
  if (String(currentSelection ?? "").trim() && allowed.has(current)) return String(current);
  const configured = Number(defaultCount);
  if (defaultCount != null && allowed.has(configured)) return String(configured);
  return "";
}

export function validateDayScoreDraft(scoreA, scoreB, scoring = null, unusualScoreAcknowledged = false) {
  const textA = String(scoreA ?? "").trim();
  const textB = String(scoreB ?? "").trim();
  if (!textA || !textB) {
    return { ok: false, message: "Enter both scores before review." };
  }
  const a = Number(textA);
  const b = Number(textB);
  if (!Number.isInteger(a) || !Number.isInteger(b) || a < 0 || b < 0) {
    return { ok: false, message: "Scores must be non-negative whole numbers." };
  }
  if (a === b) {
    return { ok: false, message: "Tournament games cannot be saved with a tied score." };
  }
  const legacyScoringSnapshot = scoring == null;
  const format = String(legacyScoringSnapshot ? "GAME_TO_11" : scoring?.format || "").trim().toUpperCase();
  const winner = Math.max(a, b);
  const loser = Math.min(a, b);
  const impossibleReasons = [];
  const unusualReasons = [];
  if (!legacyScoringSnapshot && !["GAME_TO_11", "GAME_TO_15", "GAME_TO_21", "BEST_2_OF_3"].includes(format)) {
    impossibleReasons.push("Configured scoring format is unavailable.");
  } else if (format === "BEST_2_OF_3") {
    if (!((winner === 2 && loser === 0) || (winner === 2 && loser === 1))) {
      impossibleReasons.push("BEST_2_OF_3 stores games won; the final must be 2–0 or 2–1.");
    }
  } else {
    const target = Number(scoring?.target ?? ({ GAME_TO_11: 11, GAME_TO_15: 15, GAME_TO_21: 21 })[format]);
    if (![11, 15, 21].includes(target)) {
      impossibleReasons.push("Configured scoring target is unavailable.");
    } else if (winner < target) {
      impossibleReasons.push(`The winner must reach at least ${target} points.`);
    } else if (winner - loser < 2) {
      impossibleReasons.push("This format requires a two-point winning margin.");
    } else {
      if (winner > target && winner - loser !== 2) {
        unusualReasons.push(`The winning score is above ${target} without a two-point deuce finish.`);
      }
      if (winner > target + 20) {
        unusualReasons.push(`The winning score is more than 20 points above the target of ${target}.`);
      }
    }
  }
  if (impossibleReasons.length) {
    return { ok: false, message: `Impossible tournament score: ${impossibleReasons.join(" ")}`, impossible: true, reasons: impossibleReasons };
  }
  const unusual = unusualReasons.length > 0;
  return {
    ok: true,
    scoreA: a,
    scoreB: b,
    unusual,
    reasons: unusualReasons,
    acknowledgementRequired: unusual && !unusualScoreAcknowledged,
    scoringFormat: format
  };
}

export function validateDayCorrectionDraft(scoreA, scoreB, currentScoreA, currentScoreB, scoring = null, unusualScoreAcknowledged = false) {
  const result = validateDayScoreDraft(scoreA, scoreB, scoring, unusualScoreAcknowledged);
  if (!result.ok) return result;
  if (result.scoreA === Number(currentScoreA) && result.scoreB === Number(currentScoreB)) {
    return { ok: false, message: "Enter a changed final score before review." };
  }
  return result;
}

export function validateNonPlayedOutcomeDraft(resultType, winnerTeamId, resultNote) {
  const type = String(resultType || "").trim().toUpperCase();
  if (!["FORFEIT", "NO_SHOW", "RETIREMENT"].includes(type)) {
    return { ok: false, message: "Choose forfeit, no-show, or retirement." };
  }
  const winner = String(winnerTeamId || "").trim();
  if (!winner) return { ok: false, message: "Choose the winning team." };
  const note = String(resultNote || "").trim();
  if (!note) return { ok: false, message: "Add an operator note explaining the outcome." };
  if (note.length > 500) return { ok: false, message: "The operator note is limited to 500 characters." };
  return { ok: true, resultType: type, winnerTeamId: winner, resultNote: note };
}
