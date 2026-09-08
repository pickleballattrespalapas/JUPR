export type DivisionRule = { min_rating: number | null; max_rating: number | null; women_required: number | null };
export type Participation = { season_id: string; club_id: string; status: string; revision: number };
export type RegistrationSeason = {
  id: string; organizer_club_id: string; source_revision: number; roster_deadline: string;
  details: { name: string; start_date: string; end_date: string; timezone: string; divisions: string[]; club_ids: string[];
    meets: { host_club_id: string; club_ids: string[]; starts_at: string; duration_minutes: number; courts: number }[] };
  rules: Record<string, DivisionRule>; participation?: Participation | null;
};
export type RosterPlayer = { entry_id?: string; player_id?: string; name: string; starting_rating: number; gender?: string };
export type RosterVersion = { revision: number; name: string; roster: RosterPlayer[]; issues: { code: string; message: string }[];
  status: string; late_change: boolean; submitted_at: string; decision_reason: string | null };
export type InterclubTeam = RosterVersion & { id: string; season_id: string; club_id: string; division: string; withdrawn: boolean };
export type RegistrationDetail = { season: RegistrationSeason; is_organizer: boolean; own_participation: Participation | null;
  participations: Participation[]; clubs: { id: string; name: string; slug: string }[]; teams: InterclubTeam[]; next_team_offset: number | null };
export const rosterStatus: Record<string, string> = { eligible: "Eligible", needs_exception: "Needs organizer exception", exception_approved: "Exception approved", exception_denied: "Exception declined", withdrawn: "Withdrawn" };
export function composition(rule: DivisionRule): string {
  return rule.women_required == null ? "Any four players" : `${rule.women_required} women, ${4 - rule.women_required} men`;
}
export function apiError(data: { detail?: unknown }, fallback: string): string {
  if (typeof data.detail === "string") return data.detail;
  if (Array.isArray(data.detail)) return data.detail.map(item => String(item.msg || "").replace(/^Value error, /, "")).join(" ") || fallback;
  return fallback;
}
