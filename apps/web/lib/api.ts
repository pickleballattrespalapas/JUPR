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

export type PublicLiveMatch = {
  id: string;
  round_number: number;
  court_number?: number | null;
  mini_round_number?: number | null;
  slot?: number | null;
  label: string;
  team_a: string[];
  team_b: string[];
  score_a?: number | null;
  score_b?: number | null;
  is_scored?: boolean;
  winner?: string | null;
};

export type PublicLiveRound = {
  number: number;
  matches: PublicLiveMatch[];
  courts?: Array<{
    court_number: number;
    size?: number | null;
    matches: PublicLiveMatch[];
  }>;
};

export type PublicLiveSessionSummary = {
  session_key: string;
  title: string;
  status: string;
  event_type?: string | null;
  current_round?: number | null;
  has_event?: boolean;
  created_at?: string | null;
  updated_at?: string | null;
  last_seen_at?: string | null;
  expires_at?: string | null;
};

export type PublicLiveSessionDetail = PublicLiveSessionSummary & {
  rounds: PublicLiveRound[];
  standings: Array<Record<string, string | number | boolean | null>>;
  bracket?: {
    champion?: string | null;
    rows: Array<Record<string, string | number | boolean | null>>;
  } | null;
  court_standings?: Array<Record<string, unknown>>;
};

export type LiveSessionsResponse = {
  club: {
    id: string;
    slug: string;
    name: string;
  };
  sessions: PublicLiveSessionSummary[];
};

export type LiveSessionDetailResponse = {
  club: {
    id: string;
    slug: string;
    name: string;
  };
  session: PublicLiveSessionDetail;
};

type ApiResult<T> = {
  data: T | null;
  error: string | null;
};

function baseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

async function apiErrorMessage(response: Response): Promise<string> {
  const fallback = `API error (${response.status}).`;
  let bodyText = "";
  try {
    bodyText = await response.text();
  } catch {
    return fallback;
  }

  if (!bodyText) {
    return fallback;
  }

  try {
    const payload = JSON.parse(bodyText) as { detail?: unknown; message?: unknown; error?: unknown };
    const detail = payload.detail ?? payload.message ?? payload.error;
    if (Array.isArray(detail)) {
      return `${fallback} ${detail.map((item) => JSON.stringify(item)).join("; ")}`;
    }
    if (detail) {
      return `${fallback} ${String(detail)}`;
    }
  } catch {
    // Fall through to a short text excerpt below.
  }

  return `${fallback} ${bodyText.slice(0, 240)}`;
}

async function fetchJson<T>(path: string, options: { noStore?: boolean } = {}): Promise<ApiResult<T>> {
  const apiBase = baseUrl();
  if (!apiBase) {
    return { data: null, error: "Missing JUPR API base URL environment variable." };
  }

  const url = `${apiBase.replace(/\/$/, "")}${path}`;

  try {
    const response = await fetch(
      url,
      options.noStore ? { cache: "no-store" } : { next: { revalidate: 60 } }
    );
    if (!response.ok) {
      return { data: null, error: await apiErrorMessage(response) };
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

export async function getClubLiveSessions(clubSlug: string): Promise<ApiResult<LiveSessionsResponse>> {
  return fetchJson<LiveSessionsResponse>(`/clubs/${clubSlug}/live-sessions`, { noStore: true });
}

export async function getClubLiveSession(
  clubSlug: string,
  sessionKey: string
): Promise<ApiResult<LiveSessionDetailResponse>> {
  return fetchJson<LiveSessionDetailResponse>(`/clubs/${clubSlug}/live-sessions/${sessionKey}`, { noStore: true });
}
