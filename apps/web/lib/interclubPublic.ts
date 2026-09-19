export type GameScore = { a: number; b: number };
export type Encounter = {
  id: string;
  meet_id: string;
  division: string;
  club_a: string;
  club_b: string;
  games: GameScore[];
};
export type PublicMeet = {
  id: string;
  starts_at: string;
  host_club_id: string | null;
  club_ids: string[];
  duration_minutes: number;
  courts: number;
};
export type PublicLeagueDocument = {
  name: string;
  start_date: string;
  end_date: string;
  timezone: string;
  divisions: string[];
  clubs: { id: string; name: string }[];
  meets: PublicMeet[];
  results: Encounter[];
  scoring_version?: 1;
  competition_results?: CompetitionResult[];
};
export type CompetitionResult = {
  id: string; meet_id: string; phase: "regular" | "final" | "qualifier";
  weather: string; division: string; club_a: string; club_b: string;
  pairings: { kind: string; games: { status: string; a: number | null; b: number | null; winner: "a" | "b" | null }[] }[];
  tiebreak: { status: string; a: number | null; b: number | null } | null;
};
export type StandingRow = {
  club_id: string;
  name: string;
  played: number;
  wins: number;
  losses: number;
  games_won: number;
  games_lost: number;
  point_difference: number;
  points?: number; pairings_won?: number; point_differential?: number;
  meets_played?: number; position?: number; tied?: boolean;
};
export type ClubCup = {
  standings: { club_id: string; name?: string; position?: number; points: number; regular_points: number; championship_points: number; tied?: boolean }[];
  champions: string[]; status: string;
};
export type PublicLeague = {
  id?: string;
  document: PublicLeagueDocument;
  standings: { division: string; rows: StandingRow[] }[];
  published_at?: string;
  club_cup?: ClubCup;
  qualification?: Record<string, { qualifiers: string[]; playoff_required: string[] | boolean; eligible: string[]; status: string }>;
};
export function meetTime(value: string, zone: string) {
  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: zone,
  }).format(new Date(value));
}
