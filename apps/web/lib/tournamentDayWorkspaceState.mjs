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

export function oldestReadyQueue(queue) {
  const rows = Array.isArray(queue) ? queue : [];
  return [...rows].sort((left, right) => {
    const leftPosition = Number(left?.position);
    const rightPosition = Number(right?.position);
    const safeLeftPosition = Number.isFinite(leftPosition) && leftPosition > 0
      ? leftPosition
      : Number.MAX_SAFE_INTEGER;
    const safeRightPosition = Number.isFinite(rightPosition) && rightPosition > 0
      ? rightPosition
      : Number.MAX_SAFE_INTEGER;
    if (safeLeftPosition !== safeRightPosition) {
      return safeLeftPosition - safeRightPosition;
    }
    const leftReadyAt = Date.parse(String(left?.eligible_since || ""));
    const rightReadyAt = Date.parse(String(right?.eligible_since || ""));
    const safeLeftReadyAt = Number.isFinite(leftReadyAt) ? leftReadyAt : Number.MAX_SAFE_INTEGER;
    const safeRightReadyAt = Number.isFinite(rightReadyAt) ? rightReadyAt : Number.MAX_SAFE_INTEGER;
    if (safeLeftReadyAt !== safeRightReadyAt) return safeLeftReadyAt - safeRightReadyAt;
    return String(left?.game_id || "").localeCompare(String(right?.game_id || ""));
  });
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

export function tournamentDayMedalMatchKind(game) {
  if (String(game?.stage || "").trim().toUpperCase() !== "PLAYOFF") return null;
  const round = String(game?.playoff_round || game?.round_label || "")
    .trim()
    .toUpperCase()
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ");
  if (["BRONZE", "BRONZE MEDAL", "THIRD PLACE", "3RD PLACE"].includes(round)) {
    return "bronze";
  }
  if (["FINAL", "GOLD", "GOLD MEDAL", "CHAMPIONSHIP"].includes(round)) {
    return "gold";
  }
  return null;
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

function drawIsReadyForPlayoffReview(draw) {
  const progressionStatus = String(draw?.progression_status || draw?.status || "")
    .trim()
    .toUpperCase();
  if (["PLAYOFFS_GENERATED", "PLAYOFF", "COMPLETE", "CLOSED"].includes(progressionStatus)) return false;
  if (progressionStatus === "READY_FOR_PLAYOFF_REVIEW") return true;
  return Boolean(draw?.round_robin_complete && draw?.readiness?.generate_playoffs?.ready);
}

export function readyPlayoffReviewDraws(snapshot) {
  const draws = Array.isArray(snapshot?.draws) ? snapshot.draws : [];
  const alerts = Array.isArray(snapshot?.progression_alerts) ? snapshot.progression_alerts : [];
  const readyAlertDrawIds = new Set(
    alerts
      .filter((alert) => Boolean(alert?.ready))
      .map((alert) => String(alert?.draw_id || ""))
      .filter(Boolean)
  );
  return draws.filter((draw) => (
    readyAlertDrawIds.has(String(draw?.id || "")) || drawIsReadyForPlayoffReview(draw)
  ));
}

export function newlyReadyPlayoffNotice(previous, current) {
  if (!previous || !current) return null;
  const previousIds = new Set(readyPlayoffReviewDraws(previous).map((draw) => String(draw?.id || "")));
  const newlyReady = readyPlayoffReviewDraws(current).filter((draw) => !previousIds.has(String(draw?.id || "")));
  if (!newlyReady.length) return null;
  if (newlyReady.length === 1) {
    return `Round robin complete — ${String(newlyReady[0]?.name || "this draw")} is ready for playoff review.`;
  }
  return `${newlyReady.length} round robins are complete and ready for playoff review: ${newlyReady
    .map((draw) => String(draw?.name || "Unnamed draw"))
    .join(", ")}.`;
}

const CLOSEOUT_RECOVERY_CODES = new Set(["OPERATION_UNSETTLED"]);
const CLOSEOUT_MATCH_CODES = new Set([
  "COURT_ASSIGNMENTS_ACTIVE",
  "PLAYER_CLAIMS_ACTIVE",
  "GAMES_UNFINISHED"
]);
const CLOSEOUT_PROGRESSION_CODES = new Set([
  "NO_ACTIVATED_DRAWS",
  "SCHEDULED_DRAWS_NOT_ACTIVATED",
  "PLAYOFFS_REQUIRED",
  "PLAYOFFS_INCOMPLETE"
]);
const CLOSEOUT_PODIUM_CODES = new Set([
  "PODIUM_INCOMPLETE",
  "PODIUM_RESULT_MISMATCH",
  "PODIUM_REVIEW_REQUIRED",
  "AWARDS_INCOMPLETE"
]);
const ACTIVE_OPERATION_STATUSES = new Set(["intent", "mutated", "recovery_required"]);

function closeoutBlockerCode(blocker) {
  return typeof blocker === "string" ? "" : String(blocker?.code || "").trim().toUpperCase();
}

function readinessBlockerCodes(readiness) {
  return (Array.isArray(readiness?.blockers) ? readiness.blockers : [])
    .map(closeoutBlockerCode)
    .filter(Boolean);
}

/**
 * Returns the next operator-facing closeout step once a live day has gone idle.
 * Active draw labels are intentionally not treated as unfinished work: closing
 * the day retires those durable day-draw activations after progression, podium,
 * awards, and recovery evidence have passed the server's close readiness gate.
 */
export function tournamentDayCloseoutGuidance(snapshot) {
  if (!snapshot || typeof snapshot !== "object") return null;
  const runState = String(snapshot?.day_run?.state || "").trim().toUpperCase();
  if (runState === "CLOSED") {
    return {
      phase: "closed",
      nextStep: "done",
      playComplete: true,
      progressionComplete: true,
      podiumComplete: true,
      readyToClose: false,
      blockerCodes: [],
      progressionDrawIds: [],
      podiumDrawIds: []
    };
  }
  if (!["ACTIVE", "PAUSED"].includes(runState)) return null;

  const summary = snapshot?.summary || {};
  const courts = Array.isArray(snapshot?.courts) ? snapshot.courts : [];
  const draws = Array.isArray(snapshot?.draws) ? snapshot.draws : [];
  const completedGames = Number(summary.completed_games || 0);
  const courtCount = Number(summary.courts || courts.length || 0);
  const availableCourtCount = Number(summary.available_courts || 0);
  const queueArraysPresent = ["eligible_queue", "reserved_queue", "held_games", "blocked_games"]
    .every((key) => Array.isArray(snapshot?.[key]));
  const assignableQueueIsClear = queueArraysPresent
    && Number(summary.eligible_games || 0) === 0
    && Number(summary.reserved_games || 0) === 0
    && snapshot.eligible_queue.length === 0
    && snapshot.reserved_queue.length === 0;
  const courtsAreClear = courtCount > 0
    && availableCourtCount === courtCount
    && courts.every((court) => (
      String(court?.state || "").trim().toUpperCase() === "AVAILABLE"
      && !court?.current_assignment
      && !court?.next_assignment
    ));
  const drawAssignmentsAreClear = draws.every((draw) => Number(draw?.active_games || 0) === 0);
  if (completedGames < 1 || !assignableQueueIsClear || !courtsAreClear || !drawAssignmentsAreClear) return null;

  const closeReadiness = snapshot?.readiness?.close_day || {};
  const closeBlockerRows = (Array.isArray(closeReadiness?.blockers) ? closeReadiness.blockers : [])
    .map((blocker) => ({
      code: closeoutBlockerCode(blocker),
      drawId: typeof blocker === "string" ? "" : String(blocker?.draw_id || "")
    }));
  const closeCodes = closeBlockerRows.map((row) => row.code).filter(Boolean);
  const drawCodeRows = draws.map((draw) => ({
    drawId: String(draw?.id || ""),
    codes: readinessBlockerCodes(draw?.readiness?.closeout)
  }));
  const blockerCodes = [...new Set([
    ...closeCodes,
    ...drawCodeRows.flatMap((row) => row.codes)
  ])];
  const hasCode = (codes) => blockerCodes.some((code) => codes.has(code));
  const unsettledOperation = (Array.isArray(snapshot?.operations) ? snapshot.operations : [])
    .some((operation) => ACTIVE_OPERATION_STATUSES.has(String(operation?.status || "").trim().toLowerCase()));
  const unresolvedQueueExceptions = ["held_games", "blocked_games"]
    .some((key) => Array.isArray(snapshot?.[key]) && snapshot[key].length > 0)
    || Number(summary.held_games || 0) > 0
    || draws.some((draw) => (
      Number(draw?.queued_games || 0) > 0
      || Number(draw?.held_games || 0) > 0
    ));
  const allDrawGamesFinalized = draws.length > 0
    && draws.every((draw) => (
      Number(draw?.total_games || 0) > 0
      && Number(draw?.finalized_games || 0) >= Number(draw?.total_games || 0)
    ));
  const playComplete = allDrawGamesFinalized
    && !unresolvedQueueExceptions
    && !hasCode(CLOSEOUT_MATCH_CODES);
  const progressionDrawIds = [...new Set([...drawCodeRows, ...closeBlockerRows]
    .filter((row) => CLOSEOUT_PROGRESSION_CODES.has(row.code || "")
      || row.codes?.some((code) => CLOSEOUT_PROGRESSION_CODES.has(code)))
    .map((row) => row.drawId)
    .filter(Boolean))];
  const podiumDrawIds = [...new Set([...drawCodeRows, ...closeBlockerRows]
    .filter((row) => CLOSEOUT_PODIUM_CODES.has(row.code || "")
      || row.codes?.some((code) => CLOSEOUT_PODIUM_CODES.has(code)))
    .map((row) => row.drawId)
    .filter(Boolean))];
  const progressionComplete = playComplete && !hasCode(CLOSEOUT_PROGRESSION_CODES);
  const activatedDraws = draws.filter((draw) => (
    !["", "INACTIVE", "REMOVED"].includes(String(draw?.activation_state || draw?.state || "").trim().toUpperCase())
  ));
  const activatedDrawCloseoutReady = activatedDraws.length > 0
    && activatedDraws.every((draw) => draw?.readiness?.closeout?.ready === true);
  const podiumComplete = progressionComplete
    && !hasCode(CLOSEOUT_PODIUM_CODES)
    && (closeReadiness.ready === true || activatedDrawCloseoutReady);
  const readyToClose = closeReadiness.ready === true && !unsettledOperation;

  let nextStep = "review";
  if (unsettledOperation || hasCode(CLOSEOUT_RECOVERY_CODES)) nextStep = "recovery";
  else if (!playComplete || hasCode(CLOSEOUT_MATCH_CODES)) nextStep = "matches";
  else if (hasCode(CLOSEOUT_PROGRESSION_CODES)) nextStep = "draws";
  else if (hasCode(CLOSEOUT_PODIUM_CODES)) nextStep = "podium";
  else if (readyToClose) nextStep = "close";

  return {
    phase: "closeout",
    nextStep,
    playComplete,
    progressionComplete,
    podiumComplete,
    readyToClose,
    blockerCodes,
    progressionDrawIds,
    podiumDrawIds
  };
}

function roundCode(value) {
  const normalized = String(typeof value === "string" ? value : value?.code || value?.round || "")
    .trim()
    .toUpperCase();
  if (["QF", "QUARTERFINAL", "QUARTERFINALS", "PLAY_IN", "PLAY-IN"].includes(normalized)) return "QF";
  if (["SF", "SEMIFINAL", "SEMIFINALS", "SEMI_FINAL", "SEMI-FINAL"].includes(normalized)) return "SF";
  if (["FINAL", "GOLD", "GOLD_MEDAL", "CHAMPIONSHIP"].includes(normalized)) return "FINAL";
  if (["BRONZE", "BRONZE_MEDAL", "THIRD_PLACE"].includes(normalized)) return "BRONZE";
  return normalized;
}

export function playoffTemplateRoundCodes(template) {
  const rounds = Array.isArray(template?.rounds) ? template.rounds : [];
  const applicableRounds = Array.isArray(template?.applicable_rounds) ? template.applicable_rounds : [];
  const games = Array.isArray(template?.games) ? template.games : [];
  return [...new Set([
    ...applicableRounds.map(roundCode),
    ...rounds.map(roundCode),
    ...games.map((game) => roundCode(game?.round || game?.playoff_round))
  ].filter(Boolean))];
}

export function validatePlayoffReviewConfiguration(review, configuration) {
  const templates = Array.isArray(review?.templates) ? review.templates : [];
  const templateCode = String(configuration?.template_code || "");
  const template = templates.find((candidate) => String(candidate?.code || "") === templateCode);
  if (!template) return { ok: false, message: "Choose an available playoff structure." };

  const advanceCount = Number(template?.advance_count);
  if (!Number.isInteger(advanceCount) || advanceCount < 2) {
    return { ok: false, message: "The selected playoff structure has no valid qualifier count." };
  }
  const seedTeamIds = Array.isArray(configuration?.seed_team_ids)
    ? configuration.seed_team_ids.map((value) => String(value || "").trim())
    : [];
  if (seedTeamIds.length !== advanceCount || seedTeamIds.some((teamId) => !teamId)) {
    return { ok: false, message: `Choose exactly ${advanceCount} seeded playoff teams.` };
  }
  if (new Set(seedTeamIds).size !== seedTeamIds.length) {
    return { ok: false, message: "Each playoff seed must use a different team." };
  }
  const eligibleTeamIds = new Set(
    (Array.isArray(review?.eligible_team_ids) ? review.eligible_team_ids : [])
      .map((value) => String(value || "").trim())
      .filter(Boolean)
  );
  if (!eligibleTeamIds.size || seedTeamIds.some((teamId) => !eligibleTeamIds.has(teamId))) {
    return { ok: false, message: "Every seeded team must still be eligible in the authoritative playoff review." };
  }

  const formats = new Set(
    (Array.isArray(review?.scoring_formats) ? review.scoring_formats : [])
      .map((format) => String(format?.code || "").trim())
      .filter(Boolean)
  );
  const roundCodes = playoffTemplateRoundCodes(template);
  if (!roundCodes.length) {
    return { ok: false, message: "The selected playoff structure has no reviewable rounds." };
  }
  const roundScoring = configuration?.round_scoring && typeof configuration.round_scoring === "object"
    ? configuration.round_scoring
    : {};
  const extraRound = Object.keys(roundScoring).find((code) => !roundCodes.includes(String(code)));
  if (extraRound) {
    return { ok: false, message: `Remove scoring for the unavailable ${String(extraRound).replaceAll("_", " ").toLowerCase()} round.` };
  }
  const invalidRound = roundCodes.find((code) => !formats.has(String(roundScoring[code] || "")));
  if (invalidRound) {
    return { ok: false, message: `Choose an available scoring format for ${invalidRound.replaceAll("_", " ").toLowerCase()}.` };
  }
  return { ok: true, template, advanceCount, seedTeamIds, roundScoring, roundCodes };
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
    impossibleReasons.push("Enter the individual Game 1, Game 2, and, when needed, Game 3 scores for BEST_2_OF_3.");
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

function bestOfThreeIndividualScoring(scoring) {
  return {
    format: String(scoring?.individual_game_format || "GAME_TO_11").trim().toUpperCase(),
    target: Number(scoring?.individual_game_target ?? 11),
    win_by_two: scoring?.individual_game_win_by_two ?? true
  };
}

function bestOfThreeDraftRows(gameScores) {
  const rows = Array.isArray(gameScores) ? gameScores : [];
  const byNumber = new Map();
  for (const row of rows) {
    const gameNumber = Number(row?.game_number);
    if (![1, 2, 3].includes(gameNumber)) {
      return { ok: false, message: "Best-of-three game numbers must be 1, 2, or 3." };
    }
    if (byNumber.has(gameNumber)) {
      return { ok: false, message: `Game ${gameNumber} appears more than once.` };
    }
    byNumber.set(gameNumber, row);
  }
  return { ok: true, byNumber };
}

function validateBestOfThreeGame(row, gameNumber, scoring, unusualScoreAcknowledged) {
  const scoreA = String(row?.score_a ?? "").trim();
  const scoreB = String(row?.score_b ?? "").trim();
  if (!scoreA || !scoreB) {
    return { ok: false, message: `Enter both scores for Game ${gameNumber}.` };
  }
  const validation = validateDayScoreDraft(
    scoreA,
    scoreB,
    bestOfThreeIndividualScoring(scoring),
    unusualScoreAcknowledged
  );
  if (!validation.ok) {
    return { ...validation, message: `Game ${gameNumber}: ${validation.message}` };
  }
  return {
    ...validation,
    gameScore: {
      game_number: gameNumber,
      score_a: validation.scoreA,
      score_b: validation.scoreB
    }
  };
}

export function validateBestOfThreeGameScores(
  gameScores,
  scoring = null,
  unusualScoreAcknowledged = false
) {
  const draft = bestOfThreeDraftRows(gameScores);
  if (!draft.ok) return draft;

  const validatedGames = [];
  const unusualReasons = [];
  for (const gameNumber of [1, 2]) {
    const validation = validateBestOfThreeGame(
      draft.byNumber.get(gameNumber),
      gameNumber,
      scoring,
      unusualScoreAcknowledged
    );
    if (!validation.ok) return validation;
    validatedGames.push(validation.gameScore);
    unusualReasons.push(...validation.reasons.map((reason) => `Game ${gameNumber}: ${reason}`));
  }

  const firstWinner = validatedGames[0].score_a > validatedGames[0].score_b ? "A" : "B";
  const secondWinner = validatedGames[1].score_a > validatedGames[1].score_b ? "A" : "B";
  const thirdRow = draft.byNumber.get(3);
  const thirdHasScore = Boolean(
    String(thirdRow?.score_a ?? "").trim() || String(thirdRow?.score_b ?? "").trim()
  );
  if (firstWinner === secondWinner) {
    if (thirdHasScore) {
      return {
        ok: false,
        message: "Game 3 must stay empty because the series was won in the first two games."
      };
    }
  } else {
    if (!thirdHasScore) {
      return { ok: false, message: "Enter Game 3 because the series is tied 1–1." };
    }
    const validation = validateBestOfThreeGame(
      thirdRow,
      3,
      scoring,
      unusualScoreAcknowledged
    );
    if (!validation.ok) return validation;
    validatedGames.push(validation.gameScore);
    unusualReasons.push(...validation.reasons.map((reason) => `Game 3: ${reason}`));
  }

  const scoreA = validatedGames.filter((game) => game.score_a > game.score_b).length;
  const scoreB = validatedGames.length - scoreA;
  const unusual = unusualReasons.length > 0;
  return {
    ok: true,
    scoreA,
    scoreB,
    gameScores: validatedGames,
    unusual,
    reasons: unusualReasons,
    acknowledgementRequired: unusual && !unusualScoreAcknowledged,
    scoringFormat: "BEST_2_OF_3"
  };
}

export function validateBestOfThreeRetirementGameScores(
  gameScores,
  scoring = null,
  unusualScoreAcknowledged = false
) {
  const draft = bestOfThreeDraftRows(gameScores);
  if (!draft.ok) return draft;

  const hasInput = (gameNumber) => {
    const row = draft.byNumber.get(gameNumber);
    return Boolean(
      String(row?.score_a ?? "").trim() || String(row?.score_b ?? "").trim()
    );
  };
  const hasCompleteInput = (gameNumber) => {
    const row = draft.byNumber.get(gameNumber);
    return Boolean(
      String(row?.score_a ?? "").trim() && String(row?.score_b ?? "").trim()
    );
  };

  if (hasInput(3)) {
    return {
      ok: false,
      message: "A completed Game 3 finishes the series. Record it as a played score instead of a retirement."
    };
  }
  if (hasInput(1) && !hasCompleteInput(1)) {
    return { ok: false, message: "Enter both scores for completed Game 1." };
  }
  if (hasInput(2) && !hasCompleteInput(2)) {
    return { ok: false, message: "Enter both scores for completed Game 2." };
  }
  if (hasInput(2) && !hasCompleteInput(1)) {
    return { ok: false, message: "Enter completed Game 1 before Game 2." };
  }
  if (!hasInput(1)) {
    return {
      ok: true,
      gameScores: [],
      unusual: false,
      reasons: [],
      acknowledgementRequired: false,
      scoringFormat: "BEST_2_OF_3"
    };
  }

  const validatedGames = [];
  const unusualReasons = [];
  const first = validateBestOfThreeGame(
    draft.byNumber.get(1),
    1,
    scoring,
    unusualScoreAcknowledged
  );
  if (!first.ok) return first;
  validatedGames.push(first.gameScore);
  unusualReasons.push(...first.reasons.map((reason) => `Game 1: ${reason}`));

  if (hasInput(2)) {
    const second = validateBestOfThreeGame(
      draft.byNumber.get(2),
      2,
      scoring,
      unusualScoreAcknowledged
    );
    if (!second.ok) return second;
    const firstWinner = first.gameScore.score_a > first.gameScore.score_b ? "A" : "B";
    const secondWinner = second.gameScore.score_a > second.gameScore.score_b ? "A" : "B";
    if (firstWinner === secondWinner) {
      return {
        ok: false,
        message: "One team already won Games 1 and 2. Record the completed 2–0 series as a played score instead of a retirement."
      };
    }
    validatedGames.push(second.gameScore);
    unusualReasons.push(...second.reasons.map((reason) => `Game 2: ${reason}`));
  }

  const unusual = unusualReasons.length > 0;
  return {
    ok: true,
    gameScores: validatedGames,
    unusual,
    reasons: unusualReasons,
    acknowledgementRequired: unusual && !unusualScoreAcknowledged,
    scoringFormat: "BEST_2_OF_3"
  };
}

export function validateBestOfThreeCorrectionDraft(
  gameScores,
  currentGameScores,
  scoring = null,
  unusualScoreAcknowledged = false
) {
  const result = validateBestOfThreeGameScores(
    gameScores,
    scoring,
    unusualScoreAcknowledged
  );
  if (!result.ok) return result;
  const current = (Array.isArray(currentGameScores) ? currentGameScores : [])
    .map((game) => ({
      game_number: Number(game?.game_number),
      score_a: Number(game?.score_a),
      score_b: Number(game?.score_b)
    }))
    .filter((game) => [1, 2, 3].includes(game.game_number))
    .sort((left, right) => left.game_number - right.game_number);
  if (JSON.stringify(result.gameScores) === JSON.stringify(current)) {
    return { ok: false, message: "Enter a changed individual game score before review." };
  }
  return result;
}

export function validateDayCorrectionDraft(scoreA, scoreB, currentScoreA, currentScoreB, scoring = null, unusualScoreAcknowledged = false) {
  const result = validateDayScoreDraft(scoreA, scoreB, scoring, unusualScoreAcknowledged);
  if (!result.ok) return result;
  if (result.scoreA === Number(currentScoreA) && result.scoreB === Number(currentScoreB)) {
    return { ok: false, message: "Enter a changed final score before review." };
  }
  return result;
}

export function validateNonPlayedOutcomeDraft(resultType, nonPlayingTeamId, resultNote) {
  const type = String(resultType || "").trim().toUpperCase();
  if (!["FORFEIT", "NO_SHOW", "RETIREMENT"].includes(type)) {
    return { ok: false, message: "Choose forfeit, no-show, or retirement." };
  }
  const nonPlayingTeam = String(nonPlayingTeamId || "").trim();
  if (!nonPlayingTeam) {
    return { ok: false, message: "Choose the team responsible for this non-play result." };
  }
  const note = String(resultNote || "").trim();
  if (note.length > 500) return { ok: false, message: "The operator note is limited to 500 characters." };
  return { ok: true, resultType: type, nonPlayingTeamId: nonPlayingTeam, resultNote: note };
}
