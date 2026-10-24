export type ScoreMatch = {
  id: string;
  sideA?: string[];
  sideB?: string[];
  teamA?: string[];
  teamB?: string[];
  scoreA?: number | null;
  scoreB?: number | null;
  status?: string;
};

export type PendingScoreSave = { context: string; path: string; body: string };

export class GeneratorScoreValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "GeneratorScoreValidationError";
  }
}

export function storedScoreSave(context: string): PendingScoreSave | null {
  try {
    const value = JSON.parse(sessionStorage.getItem(`generator-score-save:${context}`) || "null");
    if (value?.context !== context || typeof value.path !== "string" || typeof value.body !== "string") return null;
    const body = JSON.parse(value.body);
    return typeof body.idempotency_key === "string" && Array.isArray(body.scores) ? value : null;
  } catch { return null; }
}

export function rememberScoreSave(context: string, request: PendingScoreSave | null, expected?: PendingScoreSave): void {
  try {
    if (request) sessionStorage.setItem(`generator-score-save:${context}`, JSON.stringify(request));
    else {
      const current = storedScoreSave(context);
      // An older component must never erase a newer component's recovery record.
      if (!expected || (current?.path === expected.path && current?.body === expected.body)) {
        sessionStorage.removeItem(`generator-score-save:${context}`);
      }
    }
  } catch { /* Browsers may disable storage; the mounted runner still retains its request. */ }
}

export function submittedScoreRows(matches: ScoreMatch[], scores: Record<string, string>, requireAll: boolean, requiredWinnerIds: string[] = []) {
  const requiredWinners = new Set(requiredWinnerIds);
  return matches.map(match => {
    const left = (scores[`${match.id}:a`] ?? "").trim();
    const right = (scores[`${match.id}:b`] ?? "").trim();
    if (Boolean(left) !== Boolean(right)) throw new GeneratorScoreValidationError("Enter both scores for each played game, or leave both blank for a game that was not played.");
    if (!left && requireAll) throw new GeneratorScoreValidationError("Moneyball requires scores for every game in the rotation before submitting.");
    if (!left && requiredWinners.has(match.id)) throw new GeneratorScoreValidationError("Enter a winning score for every semifinal before continuing to the final.");
    if (!left) return { match_id: match.id, score_a: null, score_b: null };
    const scoreA = Number(left), scoreB = Number(right);
    if (![scoreA, scoreB].every(score => Number.isInteger(score) && score >= 0 && score <= 99)) throw new GeneratorScoreValidationError("Scores must be whole numbers from 0 to 99.");
    if (scoreA === scoreB) throw new GeneratorScoreValidationError("Played games need a winner. Check the tied scores before submitting.");
    return { match_id: match.id, score_a: scoreA, score_b: scoreB };
  });
}

function lineup(match: ScoreMatch): string {
  return JSON.stringify([match.sideA || match.teamA || [], match.sideB || match.teamB || []]);
}

export function refreshedScoreDraft(previous: ScoreMatch[], next: ScoreMatch[], current: Record<string, string>, preserve: boolean): Record<string, string> {
  const oldMatches = new Map(previous.map(match => [match.id, match]));
  const result: Record<string, string> = {};
  for (const match of next) {
    const old = oldMatches.get(match.id);
    for (const side of ["a", "b"] as const) {
      const key = `${match.id}:${side}`;
      const saved = side === "a" ? match.scoreA : match.scoreB;
      const oldSaved = side === "a" ? old?.scoreA : old?.scoreB;
      const dirty = current[key] !== undefined && current[key] !== (oldSaved == null ? "" : String(oldSaved));
      result[key] = preserve && old && lineup(old) === lineup(match) && dirty
        ? current[key] : saved == null ? "" : String(saved);
    }
  }
  return result;
}
