export type PublicLadderPlayer = {
  player_id: string | number;
  player_name: string;
  rank?: number | null;
  rating_jupr?: number | null;
  status: string;
  status_short?: string | null;
  detail?: string | null;
  until?: string | null;
  challenge_id?: string | number | null;
  eligibility: {
    authority: "python";
    can_initiate: boolean;
    can_receive: boolean;
    hint: string;
    eligible_opponents: Array<{
      player_id: string | number;
      player_name: string;
      rank?: number | null;
      status: string;
      status_short?: string | null;
      rank_gap?: number | null;
    }>;
  };
};

export type PublicLadderTier = {
  tier_id: string;
  label: string;
  range?: string | null;
  players: PublicLadderPlayer[];
};

export type PublicLadderChallengeSide = {
  player_id?: string | number | null;
  player_name: string;
};

export type PublicLadderChallenge = {
  id?: string | number | null;
  tier_id?: string | null;
  status: string;
  bucket: string;
  challenger: PublicLadderChallengeSide;
  defender: PublicLadderChallengeSide;
  winner?: PublicLadderChallengeSide | null;
  created_at?: string | null;
  accept_by?: string | null;
  play_by?: string | null;
  completed_at?: string | null;
};

export type PublicLadderChallengeSection = {
  name: string;
  challenges: PublicLadderChallenge[];
};

export type PublicChallengeLadderResponse = {
  club: { id: string; slug: string; name: string };
  settings: {
    challenge_range: number;
    accept_window_hours: number;
    play_window_days: number;
    cooldown_hours: number;
    protected_hours: number;
    pass_hold_hours: number;
  };
  summary: {
    tier_count: number;
    active_player_count: number;
    populated_tier_count: number;
    active_challenge_count: number;
    eligible_pair_count: number;
  };
  tiers: PublicLadderTier[];
  challenge_sections: PublicLadderChallengeSection[];
  quick_rules: string[];
  rulebook: Array<{
    title: string;
    rules: Array<{ title: string; body: string }>;
  }>;
  status_legend: Array<{
    status: string;
    short: string;
    can_initiate: boolean;
    can_receive: boolean;
    meaning: string;
  }>;
  eligibility_authority: "python";
};

type ApiResult<T> = { data: T | null; error: string | null };

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
  if (!bodyText) return fallback;
  try {
    const payload = JSON.parse(bodyText) as { detail?: unknown; message?: unknown; error?: unknown };
    const detail = payload.detail ?? payload.message ?? payload.error;
    if (Array.isArray(detail)) return `${fallback} ${detail.map((item) => JSON.stringify(item)).join("; ")}`;
    if (detail) return `${fallback} ${String(detail)}`;
  } catch {
    // Fall through to a short text excerpt below.
  }
  return `${fallback} ${bodyText.slice(0, 240)}`;
}

async function fetchJson<T>(path: string): Promise<ApiResult<T>> {
  const apiBase = baseUrl();
  if (!apiBase) return { data: null, error: "Missing JUPR API base URL environment variable." };
  const url = `${apiBase.replace(/\/$/, "")}${path}`;
  try {
    const response = await fetch(url, { next: { revalidate: 60 } });
    if (!response.ok) return { data: null, error: await apiErrorMessage(response) };
    return { data: (await response.json()) as T, error: null };
  } catch (error) {
    return { data: null, error: `Unable to reach API: ${error instanceof Error ? error.message : "Unknown error"}` };
  }
}

export async function getClubChallengeLadder(clubSlug: string): Promise<ApiResult<PublicChallengeLadderResponse>> {
  return fetchJson<PublicChallengeLadderResponse>(`/clubs/${clubSlug}/challenge-ladder`);
}
