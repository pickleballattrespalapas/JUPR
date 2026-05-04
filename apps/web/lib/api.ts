export type ClubSummary = {
  id: string;
  slug: string;
  name: string;
  tagline?: string | null;
  support_email?: string | null;
  public_base_url?: string | null;
  logo_url?: string | null;
  primary_color?: string | null;
  is_active?: boolean | null;
};

export type LeaderboardEntry = {
  rank?: number | null;
  rank_position?: number | null;
  club_id?: string;
  league_name?: string | null;
  player_id?: string | number;
  player_name: string;
  rating?: number | null;
  rating_jupr?: number | null;
  wins?: number | null;
  losses?: number | null;
  matches_played?: number | null;
  is_active?: boolean | null;
  updated_at?: string | null;
};

export type LeaderboardResponse = {
  club: {
    id: string;
    slug: string;
    name: string;
  };
  leaderboard: LeaderboardEntry[];
};

type ApiResult<T> = {
  data: T | null;
  error: string | null;
};

function baseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

async function fetchJson<T>(path: string): Promise<ApiResult<T>> {
  const apiBase = baseUrl();
  if (!apiBase) {
    return { data: null, error: "Missing JUPR API base URL environment variable." };
  }

  const url = `${apiBase.replace(/\/$/, "")}${path}`;

  try {
    const response = await fetch(url, { next: { revalidate: 60 } });
    if (!response.ok) {
      return { data: null, error: `API error (${response.status}).` };
    }

    const data = (await response.json()) as T;
    return { data, error: null };
  } catch (error) {
    return {
      data: null,
      error: `Unable to reach API: ${error instanceof Error ? error.message : "Unknown error"}`
    };
  }
}

export async function getClub(clubSlug: string): Promise<ApiResult<ClubSummary>> {
  return fetchJson<ClubSummary>(`/clubs/${clubSlug}`);
}

export async function getClubLeaderboard(clubSlug: string): Promise<ApiResult<LeaderboardResponse>> {
  return fetchJson<LeaderboardResponse>(`/clubs/${clubSlug}/leaderboards`);
}
