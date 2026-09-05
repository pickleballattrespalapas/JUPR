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
  rank_at_create?: number | null;
  current_rank?: number | null;
  current_rating_jupr?: number | null;
};

export type PublicLadderResultPlayer = {
  player_id?: string | number | null;
  player_name: string;
};

export type PublicLadderResultRatingChange = PublicLadderResultPlayer & {
  before_jupr?: number | null;
  after_jupr?: number | null;
  delta_jupr?: number | null;
};

export type PublicLadderResultRankChange = PublicLadderResultPlayer & {
  before: number;
  after: number;
  delta: number;
};

export type PublicLadderResultDetails = {
  version: 1;
  completeness: "full" | "partial";
  rank_change?: {
    swapped: boolean;
    challenger: PublicLadderResultRankChange;
    defender: PublicLadderResultRankChange;
  } | null;
  matches: Array<{
    slot: "a" | "b";
    match_id?: string | number | null;
    date?: string | null;
    score_challenger_team: number;
    score_defender_team: number;
    games?: Array<{ game: number; challenger: number; defender: number }>;
    challenger_partner?: PublicLadderResultPlayer | null;
    defender_partner?: PublicLadderResultPlayer | null;
    rating_changes: PublicLadderResultRatingChange[];
  }>;
  notice?: string | null;
  warnings?: string[];
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
  result_details?: PublicLadderResultDetails | null;
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

const PUBLIC_LOAD_ERROR = "We couldn't load the challenge ladder. Please try again.";

async function apiErrorMessage(_response: Response): Promise<string> {
  return PUBLIC_LOAD_ERROR;
}

async function fetchJson<T>(path: string): Promise<ApiResult<T>> {
  const apiBase = baseUrl();
  if (!apiBase) return { data: null, error: PUBLIC_LOAD_ERROR };
  const url = `${apiBase.replace(/\/$/, "")}${path}`;
  try {
    // Completed challenges and eligibility can change immediately after an
    // operator action. Do not let a prior public page render mask that update.
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) return { data: null, error: await apiErrorMessage(response) };
    return { data: (await response.json()) as T, error: null };
  } catch {
    return { data: null, error: PUBLIC_LOAD_ERROR };
  }
}

export async function getClubChallengeLadder(clubSlug: string): Promise<ApiResult<PublicChallengeLadderResponse>> {
  return fetchJson<PublicChallengeLadderResponse>(`/clubs/${clubSlug}/challenge-ladder`);
}
