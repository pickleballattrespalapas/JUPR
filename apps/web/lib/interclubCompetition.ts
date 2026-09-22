import { apiError, type InterclubMeet, type RegistrationSeason } from "./interclubRegistration";

export type CompetitionPhase = "regular" | "final" | "qualifier";
export type CompetitionFormat = "gender" | "mixed" | "mlp";
export type GameStatus = "pending" | "completed" | "retired" | "forfeit" | "double_forfeit" | "unplayed";
export type CompetitionPlayer = { entry_id: string; name: string; gender?: string; rating?: number; eligibility_rating?: number; starting_rating?: number; division?: string; rating_locked?: boolean };
export type CompetitionTeam = { id: string; club_id: string; division: string; revision: number; name: string; roster: CompetitionPlayer[] };
export type CompetitionGame = { id: string; status: GameStatus; a: number | null; b: number | null; winner: "a" | "b" | null; players_a: string[]; players_b: string[]; played_at: string | null; injury_reason?: string | null };
export type CompetitionPairing = { id: string; kind: "women" | "men" | "mixed_a" | "mixed_b"; court?: number | null; eligibility_deadline?: string | null; players_a: string[]; players_b: string[]; games: CompetitionGame[] };
export type CompetitionEncounter = { id: string; division: string; club_a: string; club_b: string; rotation: number; pairings: CompetitionPairing[]; tiebreak: { status: "pending" | "completed"; a: number | null; b: number | null; order_a: string[]; order_b: string[] } | null };
export type CompetitionDocument = { schema_version: 1; meet_id: string; phase: CompetitionPhase; format: CompetitionFormat; weather: "normal" | "delay" | "rescheduled" | "finalized_partial"; encounters: CompetitionEncounter[] };
export type CompetitionBatch = { meet_id: string; phase: CompetitionPhase; revision: number; state: "draft" | "submitted" | "approved"; document: CompetitionDocument; roster_sources: { team_id: string; revision: number }[]; ratings_status: "not_requested" | "pending" | "failed" | "completed"; ratings_error?: string | null; updated_at?: string };
export type StandingRow = { club_id: string; division?: string; points: number; pairings_won: number; games_won: number; point_differential: number; meets_played?: number; regular_points?: number; championship_points?: number; qualified?: boolean; position?: number; tied?: boolean };
export type StandingsGroup = { division: string; standings: StandingRow[]; qualifying_playoff_required?: boolean; tied_clubs?: string[] };
export type Qualification = { qualifiers: string[]; playoff_required: string[]; eligible: string[]; status: "ready" | "playoff_required" | "insufficient_entries" };
export type CompetitionMeet = InterclubMeet & { competition_phase?: CompetitionPhase; schedule_editable?: boolean; schedule_locked_reason?: string | null; schedule_deadline_editable?: boolean; courts_editable?: boolean };
export type CompetitionContext = { season: RegistrationSeason; clubs: { id: string; name: string }[]; meets: CompetitionMeet[]; batches: CompetitionBatch[]; is_organizer: boolean; standings: { divisions: Record<string, StandingRow[]>; qualification: Record<string, Qualification> }; club_cup: { standings: StandingRow[]; champions: string[]; status: "provisional" | "complete" }; qualifying?: Record<string, Qualification> };
export type MeetCompetition = { meet: CompetitionMeet; batch: CompetitionBatch | null; teams: CompetitionTeam[]; can_manage: boolean; is_organizer: boolean; lineups_hidden?: boolean; eligible_players?: Record<string, CompetitionPlayer[]>; display_players?: CompetitionPlayer[] };

export const pairingLabels: Record<CompetitionPairing["kind"], string> = { women: "Women’s doubles", men: "Men’s doubles", mixed_a: "Mixed doubles A", mixed_b: "Mixed doubles B" };
export const phaseLabels: Record<CompetitionPhase, string> = { regular: "Regular season", final: "Championship final", qualifier: "Qualifying playoff" };
export const gameStatusLabels: Record<GameStatus, string> = { pending: "Not entered", completed: "Completed game", retired: "Injury retirement", forfeit: "Unplayed forfeit", double_forfeit: "Both clubs unable to field this game", unplayed: "Weather: not played" };

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
export function gameCount(document: CompetitionDocument): { entered: number; total: number } {
  const games = document.encounters.flatMap(encounter => encounter.pairings.flatMap(pairing => pairing.games));
  return { entered: games.filter(game => game.status !== "pending").length, total: games.length };
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
