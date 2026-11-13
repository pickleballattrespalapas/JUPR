import type { PlayGeneratorStanding } from "@/components/PlayGeneratorStandingsTable";

export type RoundRobinSummary = {
  session_key: string;
  title: string;
  created_at: string | null;
  completed_at: string | null;
  play_format: string;
  participant_count: number;
  kind: "round_robin" | "moneyball";
  rating_mode: string;
  scoring_mode: string;
};
export type RoundRobinHistory = {
  sessions: RoundRobinSummary[];
  has_more: boolean;
  offset: number;
  limit: number;
};
export type RoundRobinResult = RoundRobinSummary & {
  standings: PlayGeneratorStanding[];
  standings_sort: "wins" | "points" | "differential" | "moneyball_expectations";
  rounds: Array<{
    number: number;
    status: string;
    matches: Array<{
      court: number | null;
      side_a: string[];
      side_b: string[];
      score_a: number | null;
      score_b: number | null;
      status: string;
    }>;
  }>;
};

export function roundRobinDate(value: string | null): string {
  if (!value) return "Date unavailable";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? "Date unavailable" : new Intl.DateTimeFormat("en-US", {
    month: "short", day: "numeric", year: "numeric", timeZone: "America/Mazatlan",
  }).format(parsed);
}

export function roundRobinFormat(value: string): string {
  return value === "singles" ? "Singles" : value === "doubles_singles" ? "Doubles + Singles" : "Doubles";
}
