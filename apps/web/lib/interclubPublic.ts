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
};
export type PublicLeague = {
  id?: string;
  document: PublicLeagueDocument;
  standings: { division: string; rows: StandingRow[] }[];
  published_at?: string;
};
export function meetTime(value: string, zone: string) {
  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: zone,
  }).format(new Date(value));
}
