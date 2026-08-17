const DAY_ACTION_CONFIRMATIONS = Object.freeze({
  activate_day: "ACTIVATE DAY",
  activate_draw: "ACTIVATE DRAW",
  pause_draw: "PAUSE DRAW",
  resume_draw: "RESUME DRAW",
  auto_fill_courts: "AUTO FILL COURTS",
  score_and_release: "SAVE SCORE AND RELEASE COURT",
  correct_completed_score: "CORRECT COMPLETED SCORE",
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

export function validateDayScoreDraft(scoreA, scoreB) {
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
  return { ok: true, scoreA: a, scoreB: b };
}

export function validateDayCorrectionDraft(scoreA, scoreB, currentScoreA, currentScoreB) {
  const result = validateDayScoreDraft(scoreA, scoreB);
  if (!result.ok) return result;
  if (result.scoreA === Number(currentScoreA) && result.scoreB === Number(currentScoreB)) {
    return { ok: false, message: "Enter a changed final score before review." };
  }
  return result;
}
