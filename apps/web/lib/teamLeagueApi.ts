import type { LeagueAwardProgress } from "./api";

export type TeamLeagueSettings = {
  league_name: string;
  status: string;
  registration_open: boolean;
  registration_configured_open?: boolean;
  online_team_registration_supported?: boolean;
  team_size: 2 | 3 | 4;
  team_category: "open" | "mens" | "womens" | "mixed";
  max_alternates?: number;
  substitute_pool_enabled?: boolean;
  mixed_required_men?: number;
  mixed_required_women?: number;
  allow_substitutes: boolean;
  playoff_format: string;
  playoff_team_count?: number | null;
  start_date?: string | null;
  start_time?: string | null;
  timezone?: string | null;
  venue?: string | null;
  registration_closes_at?: string | null;
  schedule_version?: number;
  standings_version?: number;
};

export type TeamLeagueTeam = {
  id: string;
  team_name: string;
  players: Array<{
    player_id: number;
    player_name: string;
    role: "captain" | "primary" | "alternate";
  }>;
  team_size?: 2 | 3 | 4;
  roster_complete?: boolean;
};

export type TeamLeagueFixture = {
  id: string;
  phase: "regular" | "playoff";
  round_number: number;
  week_number?: number | null;
  scheduled_at?: string | null;
  team_a_id?: string | null;
  team_b_id?: string | null;
  status: string;
  team_a_score?: number | null;
  team_b_score?: number | null;
  winner_team_id?: string | null;
};

export type PublicTeamLeagueDetail = {
  ok: boolean;
  league: TeamLeagueSettings;
  teams: TeamLeagueTeam[];
  fixtures: TeamLeagueFixture[];
  standings: Array<Record<string, unknown>>;
  award_progress: LeagueAwardProgress;
  registration: {
    open: boolean;
    payment_mode: "offline";
    signup_types: string[];
    partner_confirmation_required: boolean;
    online_team_registration_supported: boolean;
    unavailable_reason?: string | null;
  };
  registration_players: Array<{
    player_id: number;
    player_name: string;
    rating_jupr?: number | null;
    gender?: string | null;
  }>;
};

type ApiResult<T> = { data: T | null; error: string | null };

export function teamLeagueApiBaseUrl(): string | null {
  return (
    process.env.JUPR_API_BASE_URL ||
    process.env.NEXT_PUBLIC_JUPR_API_BASE_URL ||
    null
  );
}

async function fetchJson<T>(path: string): Promise<ApiResult<T>> {
  const base = teamLeagueApiBaseUrl();
  if (!base) return { data: null, error: "Team leagues are unavailable." };
  try {
    const response = await fetch(`${base.replace(/\/$/, "")}${path}`, {
      next: { revalidate: 30 }
    });
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      return {
        data: null,
        error: String(body?.detail || `API error (${response.status})`)
      };
    }
    return { data: body as T, error: null };
  } catch (error) {
    return {
      data: null,
      error: error instanceof Error ? error.message : "Unable to reach the API."
    };
  }
}

export async function getPublicTeamLeagues(
  clubSlug: string
): Promise<
  ApiResult<{ ok: boolean; leagues: TeamLeagueSettings[]; league_count: number }>
> {
  return fetchJson(`/clubs/${encodeURIComponent(clubSlug)}/team-leagues`);
}

export async function getPublicTeamLeague(
  clubSlug: string,
  leagueName: string
): Promise<ApiResult<PublicTeamLeagueDetail>> {
  return fetchJson(
    `/clubs/${encodeURIComponent(clubSlug)}/team-leagues/${encodeURIComponent(leagueName)}`
  );
}
