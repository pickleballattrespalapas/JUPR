import { apiError, type InterclubMeet, type RegistrationSeason } from "./interclubRegistration";

export type CompetitionPhase = "regular" | "final" | "qualifier";
export type CompetitionFormat = "gender" | "mixed" | "mlp";
export type CompetitionScheduleMode = "simultaneous" | "staggered";
export type GameStatus = "pending" | "completed" | "retired" | "forfeit" | "double_forfeit" | "unplayed" | "not_needed";
export type CompetitionPlayer = { entry_id: string; name: string; gender?: string; rating?: number; eligibility_rating?: number; starting_rating?: number; division?: string; rating_locked?: boolean };
export type CompetitionTeam = { id: string; club_id: string; division: string; revision: number; name: string; roster: CompetitionPlayer[] };
export type CompetitionGame = { id: string; status: GameStatus; a: number | null; b: number | null; winner: "a" | "b" | null; players_a: string[]; players_b: string[]; played_at: string | null; injury_reason?: string | null };
export type CompetitionPairing = { id: string; kind: "women" | "men" | "mixed_a" | "mixed_b"; court?: number | null; eligibility_deadline?: string | null; players_a: string[]; players_b: string[]; games: CompetitionGame[] };
export type CompetitionEncounter = { id: string; division: string; club_a: string; club_b: string; rotation: number; pairings: CompetitionPairing[]; tiebreak: { status: "pending" | "completed"; a: number | null; b: number | null; order_a: string[]; order_b: string[] } | null };
export type CompetitionDocument = { schema_version: 1; meet_id: string; phase: CompetitionPhase; format: CompetitionFormat; schedule_mode?: CompetitionScheduleMode; weather: "normal" | "delay" | "rescheduled" | "finalized_partial"; encounters: CompetitionEncounter[] };

export function scheduleRoundLabel(document: CompetitionDocument): string {
  return document.schedule_mode === "staggered" ? "Wave" : "Rotation";
}

export const regularCourtBlockInstructions = "Each court assignment covers Games 1-3 against the same opponents. Stay on that court and finish all three games, even at 2-0, before leaving.";
export const championshipGameInstructions = "Play women’s doubles, men’s doubles, then Mixed A. At 3–0, the matchup is decided and Mixed B is not played. Otherwise, play Mixed B. At 2–2, play the rotating singles tiebreak.";

export function scheduledEncounters(document: CompetitionDocument): CompetitionEncounter[] {
  return [...document.encounters].sort((a, b) => a.rotation - b.rotation ||
    Math.min(...a.pairings.map(p => p.court ?? 101)) - Math.min(...b.pairings.map(p => p.court ?? 101)) ||
    a.division.localeCompare(b.division, undefined, { numeric: true }) || a.id.localeCompare(b.id));
}
export type CompetitionBatch = { approved_revision?: number | null; meet_id: string; phase: CompetitionPhase; revision: number; state: "draft" | "submitted" | "approved"; document: CompetitionDocument; roster_sources: { team_id: string; revision: number }[]; ratings_status: "not_requested" | "pending" | "failed" | "completed"; ratings_error?: string | null; updated_at?: string };
export type StandingRow = { club_id: string; division?: string; points: number; pairings_won: number; games_won: number; point_differential: number; meets_played?: number; regular_points?: number; championship_points?: number; qualified?: boolean; position?: number; tied?: boolean };
export type StandingsGroup = { division: string; standings: StandingRow[]; qualifying_playoff_required?: boolean; tied_clubs?: string[] };
export type Qualification = { qualifiers: string[]; playoff_required: string[]; eligible: string[]; status: "ready" | "playoff_required" | "insufficient_entries" };
export type CompetitionMeet = InterclubMeet & { competition_phase?: CompetitionPhase; schedule_editable?: boolean; schedule_locked_reason?: string | null; schedule_deadline_editable?: boolean; courts_editable?: boolean };
export type CompetitionContext = { season: RegistrationSeason; clubs: { id: string; name: string }[]; meets: CompetitionMeet[]; batches: CompetitionBatch[]; is_organizer: boolean; standings: { divisions: Record<string, StandingRow[]>; qualification: Record<string, Qualification> }; club_cup: { standings: StandingRow[]; champions: string[]; status: "provisional" | "complete" }; qualifying?: Record<string, Qualification> };
export type MeetCompetition = { meet: CompetitionMeet; batch: CompetitionBatch | null; teams: CompetitionTeam[]; can_manage: boolean; is_organizer: boolean; lineups_hidden?: boolean; eligible_players?: Record<string, CompetitionPlayer[]>; display_players?: CompetitionPlayer[] };

export const pairingLabels: Record<CompetitionPairing["kind"], string> = { women: "Women’s doubles", men: "Men’s doubles", mixed_a: "Mixed doubles A", mixed_b: "Mixed doubles B" };
export const phaseLabels: Record<CompetitionPhase, string> = { regular: "Regular season", final: "Championship final", qualifier: "Qualifying playoff" };
export const gameStatusLabels: Record<GameStatus, string> = { pending: "Not entered", completed: "Completed game", retired: "Injury retirement", forfeit: "Unplayed forfeit", double_forfeit: "Both clubs unable to field this game", unplayed: "Weather: not played", not_needed: "Not needed — matchup decided 3–0" };

export function regularSeasonComplete(context: CompetitionContext): boolean {
  // Other clubs receive only their own meets, so cannot infer season completion.
  if (!context.is_organizer) return false;
  const meets = context.meets.filter(meet => !meet.competition_phase || meet.competition_phase === "regular");
  return meets.length > 0 && meets.every(meet => context.batches.some(batch => batch.meet_id === meet.id && batch.phase === "regular" && batch.state === "approved"));
}

export function championshipQualifications(context: CompetitionContext): (Qualification & { division: string })[] {
  return [...context.season.details.divisions].sort((a, b) => a.localeCompare(b, undefined, { numeric: true })).map(division => ({
    division, ...(context.standings?.qualification?.[division] || context.qualifying?.[division] || { qualifiers: [], playoff_required: [], eligible: [], status: "insufficient_entries" as const }),
  }));
}

export function competitionPath(api: string, clubId: string, seasonId: string): string {
  return `${api}/admin/clubs/${encodeURIComponent(clubId)}/interclub/competition/${encodeURIComponent(seasonId)}`;
}
export async function competitionRequest<T>(url: string, token: string, signal: AbortSignal, method?: string, body?: unknown): Promise<T> {
  const response = await fetch(url, { method, cache: "no-store", signal, headers: { Authorization: `Bearer ${token}`, ...(body !== undefined ? { "Content-Type": "application/json" } : {}) }, ...(body !== undefined ? { body: JSON.stringify(body) } : {}) });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw Object.assign(new Error(apiError(data, "Unable to load or save meet operations. Please try again.")), { status: response.status });
  return data as T;
}
export function competitionPlayers(detail: MeetCompetition): Map<string, CompetitionPlayer> {
  return new Map([...(detail.display_players || []), ...detail.teams.flatMap(team => team.roster), ...Object.values(detail.eligible_players || {}).flat()].map(player => [player.entry_id, player]));
}
export function playerNames(ids: string[], players: Map<string, CompetitionPlayer>): string {
  return ids.length ? ids.map(id => players.get(id)?.name || "Player unavailable").join(" / ") : "Pairing not fielded";
}
export function isFinalScore(a: number | null, b: number | null, target = 11): boolean {
  if (a === null || b === null || !Number.isInteger(a) || !Number.isInteger(b) || a < 0 || b < 0) return false;
  const high = Math.max(a, b), low = Math.min(a, b);
  return high >= target && high - low >= 2 && (high === target || high - low === 2);
}
export function automaticGameStatus(game: CompetitionGame, enteredAt = new Date().toISOString()): CompetitionGame {
  const status = game.status === "pending" || game.status === "completed" ? isFinalScore(game.a, game.b) ? "completed" : "pending" : game.status;
  const played_at = !game.played_at && (status === "completed" || status === "retired") ? enteredAt : game.played_at;
  return status === game.status && played_at === game.played_at ? game : { ...game, status, played_at, ...(status !== game.status ? { winner: null } : {}) };
}
export function championshipClincher(encounter: CompetitionEncounter): "a" | "b" | null {
  const winners = (["women", "men", "mixed_a"] as const).map(kind => {
    const pairing = encounter.pairings.find(pairing => pairing.kind === kind);
    const game = pairing?.games.length === 1 ? pairing.games[0] : null;
    if (!game || !gameHasOutcome(game)) return null;
    return game.status === "completed" ? game.a! > game.b! ? "a" : "b" : ["retired", "forfeit"].includes(game.status) ? game.winner : null;
  });
  return winners[0] && winners.every(winner => winner === winners[0]) ? winners[0] : null;
}
export function automaticDraftScores(document: CompetitionDocument, enteredAt = new Date().toISOString()): CompetitionDocument {
  return { ...document, encounters: document.encounters.map(encounter => {
    const scored = { ...encounter, pairings: encounter.pairings.map(pairing => ({ ...pairing, games: pairing.games.map(game => automaticGameStatus(game, enteredAt)) })) };
    const clincher = document.phase !== "regular" ? championshipClincher(scored) : null;
    return { ...scored, pairings: scored.pairings.map(pairing => ({ ...pairing, games: pairing.games.map(game => {
      if (document.phase === "regular" || pairing.kind !== "mixed_b") return game;
      if (clincher && ["pending", "double_forfeit", "unplayed", "not_needed"].includes(game.status) && game.a === null && game.b === null) return { ...game, status: "not_needed" as const, winner: null, played_at: null };
      if (!clincher && game.status === "not_needed") return automaticGameStatus({ ...game, status: "pending" }, enteredAt);
      return game;
    }) })),
    tiebreak: encounter.tiebreak && !(clincher && encounter.tiebreak.a === null && encounter.tiebreak.b === null)
      ? { ...encounter.tiebreak, status: isFinalScore(encounter.tiebreak.a, encounter.tiebreak.b, 21) ? "completed" as const : "pending" as const } : null };
  }) };
}
export function gameHasOutcome(game: CompetitionGame): boolean {
  if (game.status === "completed") return isFinalScore(game.a, game.b);
  if (game.status === "retired") return game.a !== null && game.b !== null && !!game.winner;
  if (game.status === "forfeit") return !!game.winner;
  return game.status === "double_forfeit" || game.status === "unplayed" || game.status === "not_needed";
}
export function gameCount(document: CompetitionDocument): { entered: number; total: number } {
  const games = document.encounters.flatMap(encounter => encounter.pairings.flatMap(pairing => pairing.games)).filter(game => game.status !== "not_needed");
  return { entered: games.filter(gameHasOutcome).length, total: games.length };
}
export function toLocalInput(iso: string | null): string {
  if (!iso) return "";
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? "" : new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
}
export function fromLocalInput(value: string): string | null {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date.toISOString();
}
export function singlesCourt(division: string): "Skinny" | "Full-court" {
  const level = Number.parseFloat(division);
  return Number.isFinite(level) && level < 4 ? "Skinny" : "Full-court";
}
export function activeSinglesPlayers(encounter: CompetitionEncounter, side: "a" | "b"): string[] {
  return Array.from(new Set(encounter.pairings.filter(pairing => pairing.kind.startsWith("mixed")).flatMap(pairing => {
    const last = pairing.games[pairing.games.length - 1];
    return last?.[`players_${side}`]?.length ? last[`players_${side}`] : pairing[`players_${side}`];
  })));
}
export function matchesSkillLevel(player: CompetitionPlayer, division: string): boolean {
  const rating = player.eligibility_rating ?? player.rating ?? player.starting_rating;
  if (rating == null || !Number.isFinite(rating) || rating <= 0) return false;
  const normalized = division.toLowerCase();
  if (normalized === "open" || normalized === "4.5/open") return true;
  return /^[2-6]\.[05]$/.test(normalized) && rating < Number.parseFloat(normalized) + .5;
}
