export type DivisionRule = { min_rating: number | null; max_rating: number | null; women_required: number | null };
export type Participation = { season_id: string; club_id: string; status: string; revision: number };
export type RegistrationSeason = {
  id: string; organizer_club_id: string; source_revision: number;
  details: { name: string; start_date: string; end_date: string; timezone: string; divisions: string[]; club_ids: string[];
    meets: { host_club_id: string; club_ids: string[]; starts_at: string; duration_minutes: number; courts: number }[] };
  rules: Record<string, DivisionRule>; participation?: Participation | null; registration?: SeasonRegistrationWindow;
};
export type RosterPlayer = { entry_id?: string; player_id?: string; name: string; starting_rating: number; eligibility_rating?: number; rating_locked?: boolean; rating_deadline?: string; gender?: string };
export type RosterVersion = { revision: number; name: string; roster: RosterPlayer[]; issues: { code: string; message: string }[];
  status: string; late_change: boolean; submitted_at: string; decision_reason: string | null };
export type InterclubTeam = RosterVersion & { id: string; season_id: string; meet_id: string | null; club_id: string; division: string; withdrawn: boolean };
export type InterclubMeet = { id: string; season_id: string; plan_index: number; host_club_id: string; club_ids: string[];
  starts_at: string; duration_minutes: number; courts: number; roster_deadline: string; revision: number; roster_open: boolean; deadline_editable: boolean; competition_phase?: "regular" | "final" | "qualifier" };
export type MeetRegistrationDetail = { meet: InterclubMeet; teams: InterclubTeam[]; next_team_offset: number | null };
export type RegistrationDetail = { meets: InterclubMeet[]; meet_schedule?: Pick<InterclubMeet, "id" | "starts_at" | "host_club_id" | "club_ids">[]; first_meet_at?: string | null; season: RegistrationSeason; is_organizer: boolean; own_participation: Participation | null;
  participations: Participation[]; clubs: { id: string; name: string; slug: string }[]; teams: InterclubTeam[]; next_team_offset: number | null };
export const rosterStatus: Record<string, string> = { eligible: "Eligible", needs_exception: "Needs organizer exception", exception_approved: "Exception approved", exception_denied: "Exception declined", withdrawn: "Withdrawn" };
export type LineupPlayerChoice = { id: string; name: string; starting_rating: number | null; eligibility_rating?: number | null; gender?: string; rating_locked?: boolean; rating_deadline?: string };

export function lineupGender(gender?: string): "female" | "male" | "unknown" {
  const value = (gender || "").trim().toLowerCase();
  if (["female", "f", "woman", "women"].includes(value)) return "female";
  if (["male", "m", "man", "men"].includes(value)) return "male";
  return "unknown";
}

// Mirrors the league's exclusive rating ceiling; playing up has no lower floor.
export function lineupPlayerIssue(player: LineupPlayerChoice, division: string): string | null {
  const rating = player.eligibility_rating ?? player.starting_rating;
  if (rating == null || !Number.isFinite(rating) || rating <= 0) return "Needs a valid league rating";
  const level = division.trim().toLowerCase();
  if (/^[2-6]\.[05]$/.test(level)) {
    const ceiling = Number(level) + 0.5;
    if (rating >= ceiling) return `Rating must be below ${ceiling.toFixed(1)} for ${division}`;
  } else if (!["open", "4.5/open"].includes(level)) return "Division eligibility needs checking";
  if (lineupGender(player.gender) === "unknown") return "Gender needs review in the season player pool";
  return null;
}

export function defaultLineupName(clubName: string, division: string, teams: InterclubTeam[]): string {
  const base = `${clubName} ${division}`.slice(0, 74);
  const names = new Set(teams.filter(team => team.division === division).map(team => team.name.toLowerCase()));
  let name = base, suffix = 2;
  while (names.has(name.toLowerCase())) name = `${base} ${suffix++}`;
  return name;
}
export function composition(rule: DivisionRule): string {
  return rule.women_required == null ? "Any four players" : `${rule.women_required} women, ${4 - rule.women_required} men`;
}
export function apiError(data: { detail?: unknown }, fallback: string): string {
  if (typeof data.detail === "string") return data.detail;
  if (Array.isArray(data.detail)) return data.detail.map(item => String(item.msg || "").replace(/^Value error, /, "")).join(" ") || fallback;
  return fallback;
}
import type { SeasonRegistrationWindow } from "./interclubRegistrationWindow";
