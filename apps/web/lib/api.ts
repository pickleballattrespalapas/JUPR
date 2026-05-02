export type ClubSummary = {
  slug: string;
  name: string;
  location?: string | null;
  description?: string | null;
};

export type LeaderboardEntry = {
  rank: number;
  player_name: string;
  rating?: number | null;
  matches_played?: number | null;
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

export async function getClubLeaderboard(clubSlug: string): Promise<ApiResult<LeaderboardEntry[]>> {
  return fetchJson<LeaderboardEntry[]>(`/clubs/${clubSlug}/leaderboards/public`);
}
