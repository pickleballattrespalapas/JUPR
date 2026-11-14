import type { CompetitionDocument, CompetitionPlayer, CompetitionTeam, MeetCompetition, StandingRow, Qualification } from "./interclubCompetition";
import type { InterclubMeet } from "./interclubRegistration";

export type ManagedPlayer = Omit<CompetitionPlayer, "starting_rating"> & { entry_id: string; name: string; gender: "female" | "male"; starting_rating: number | null };
export type ManagedRoster = { division: string; roster: ManagedPlayer[] };
export type ManagedClub = { id: string; name: string; captain: string; contact: string; rosters: ManagedRoster[] };
export type ManagedDetails = { name: string; start_date: string | null; end_date: string | null; timezone: string; divisions: string[]; clubs: ManagedClub[]; rules: string };
export type ManagedSeason = { id: string; organizer_club_id: string; revision: number; details: ManagedDetails; published: boolean; source_season_id?: string | null; updated_at: string };
export type ManagedTeam = ManagedRoster & { club_id: string; reserves?: ManagedPlayer[] };
export type ManagedPlan = { host_club_id: string; club_ids: string[]; starts_at: string; duration_minutes: number; courts: number; phase: "regular" | "final" | "qualifier"; format: "gender" | "mixed" | "mlp"; teams: ManagedTeam[] };
export type ManagedMeet = { id: string; season_id: string; revision: number; details: ManagedPlan; state: "planning" | "draft" | "submitted" | "approved"; document: CompetitionDocument | null; approved_document?: CompetitionDocument | null; host_expires_at?: string | null; updated_at: string };
export type ManagedContext = { season: ManagedSeason; meets: ManagedMeet[]; standings: { divisions: Record<string, StandingRow[]>; qualification: Record<string, Qualification> }; club_cup: { standings: StandingRow[]; champions: string[]; status: "provisional" | "complete" }; host_token?: string };
export type HostContext = { season: { id: string; details: Pick<ManagedDetails, "name" | "timezone" | "rules" | "divisions"> }; clubs: { id: string; name: string }[]; meet: ManagedMeet };

export const managedState: Record<ManagedMeet["state"], string> = { planning: "Choose lineups", draft: "Ready for scores", submitted: "Awaiting your review", approved: "Official results" };
export function printMeet(meet: ManagedMeet): InterclubMeet {
  return { ...meet.details, id: meet.id, season_id: meet.season_id, revision: meet.revision, plan_index: 0,
    roster_deadline: meet.details.starts_at, roster_open: false, deadline_editable: false, competition_phase: meet.details.phase };
}
export function scoreDetail(meet: ManagedMeet): MeetCompetition {
  const player = (p: ManagedPlayer): CompetitionPlayer => ({ ...p, starting_rating: p.starting_rating ?? undefined });
  const teams: CompetitionTeam[] = meet.details.teams.map((team, index) => ({ ...team, roster: team.roster.map(player), id: `${meet.id}:${index}`, name: team.club_id, revision: 1 }));
  const eligible_players: Record<string, CompetitionPlayer[]> = {};
  for (const team of meet.details.teams) eligible_players[team.club_id] = [...(eligible_players[team.club_id] || []), ...[...team.roster, ...(team.reserves || [])].map(p => ({ ...player(p), division: team.division }))];
  return { meet: printMeet(meet), teams, eligible_players, can_manage: meet.state === "draft", is_organizer: false,
    batch: meet.document ? { meet_id: meet.id, phase: meet.details.phase, revision: meet.revision, state: meet.state === "planning" ? "draft" : meet.state, document: meet.document, roster_sources: [], ratings_status: "not_requested" } : null };
}
export function rosterText(players: ManagedPlayer[], gender: "female" | "male"): string {
  return players.filter(p => p.gender === gender).map(p => p.name + (p.starting_rating == null ? "" : ` | ${p.starting_rating}`)).join("\n");
}
export function readRoster(women: string, men: string, club: ManagedClub): ManagedPlayer[] {
  const existing = new Map(club.rosters.flatMap(r => r.roster).map(p => [p.name.trim().toLocaleLowerCase(), p]));
  const names = new Set<string>();
  return ([[women, "female"], [men, "male"]] as const).flatMap(([text, gender]) => text.split("\n").filter(line => line.trim()).map(line => {
    const pieces = line.split("|");
    const name = pieces[0].trim(), raw = pieces[1]?.trim(), rating = raw ? Number(raw) : null;
    if (!name || pieces.length > 2 || (rating !== null && (!Number.isFinite(rating) || rating <= 0 || rating > 8))) throw new Error("Use one name per line, optionally followed by | rating.");
    const key = name.toLocaleLowerCase();
    if (names.has(key)) throw new Error(`${name} appears twice. Give players with the same name a distinguishing initial.`);
    names.add(key);
    return { entry_id: existing.get(key)?.entry_id || crypto.randomUUID(), name, gender, starting_rating: rating };
  }));
}
export async function managedRequest<T>(url: string, accessToken: string | null, method = "GET", body?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { method, cache: "no-store", signal,
    headers: { ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}), ...(body === undefined ? {} : { "Content-Type": "application/json" }) },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail;
    const message = typeof detail === "string" ? detail : Array.isArray(detail) ? detail.map((d: { msg: string }) => d.msg.replace(/^Value error, /, "")).join(" ") : "Could not confirm the request. Reload to check the saved version.";
    throw Object.assign(new Error(message), { status: response.status });
  }
  return data;
}
