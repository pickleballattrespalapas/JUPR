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
  starting_rating?: number | null;
  starting_rating_jupr?: number | null;
  rating_gain_jupr?: number | null;
  gap_jupr?: number | null;
  wins?: number | null;
  losses?: number | null;
  matches_played?: number | null;
  win_pct?: number | null;
  is_active?: boolean | null;
  qualified?: boolean | null;
  min_games?: number | null;
  badges?: LeaderboardBadge[];
  badge_count?: number;
  updated_at?: string | null;
};

export type LeaderboardBadge = {
  badge_id: string;
  name: string;
  prestige?: number | null;
  category?: string | null;
  icon_key?: string | null;
  rarity?: string | null;
  earned_at?: string | null;
};

export type LeaderboardScope = {
  name: string;
  label: string;
  min_games: number;
};

export type LeaderboardResponse = {
  club: { id: string; slug: string; name: string };
  scopes: LeaderboardScope[];
  selected_scope: string;
  scope: LeaderboardScope;
  filters: { status: "active" | "inactive" | "all"; search: string; sort: string };
  summary: {
    ranked_players: number;
    active_players: number;
    inactive_players: number;
    leaderboard_scopes: number;
    filtered_players: number;
  };
  leaderboard: LeaderboardEntry[];
  snapshot?: LeaderboardEntry | null;
  highlights: {
    highest_rating: LeaderboardEntry[];
    most_improved: LeaderboardEntry[];
    best_win_pct: LeaderboardEntry[];
    most_wins: LeaderboardEntry[];
  };
  pagination: { total: number; offset: number; limit: number; has_more: boolean };
};

export type PublicPlayer = {
  id: string | number;
  name: string;
  display_name?: string;
  rating?: number | null;
  rating_jupr?: number | null;
  starting_rating?: number | null;
  starting_rating_jupr?: number | null;
  wins?: number | null;
  losses?: number | null;
  matches_played?: number | null;
  singles_rating?: number | null;
  singles_rating_jupr?: number | null;
  singles_wins?: number | null;
  singles_losses?: number | null;
  singles_matches_played?: number | null;
  singles_last_game_at?: string | null;
  is_active?: boolean | null;
  last_game_at?: string | null;
};

export type PublicLeagueRating = {
  id?: string | number | null;
  league_name?: string | null;
  rating?: number | null;
  rating_jupr?: number | null;
  starting_rating?: number | null;
  starting_rating_jupr?: number | null;
  rating_gain_jupr?: number | null;
  wins?: number | null;
  losses?: number | null;
  matches_played?: number | null;
  is_active?: boolean | null;
};

export type PublicMatchPlayer = { id: string | number; name: string };

export type PublicRatingSnapshotEntry = {
  player_id?: string | number | null;
  start_rating?: number | null;
  end_rating?: number | null;
};

export type PublicMatch = {
  id?: string | number | null;
  club_id?: string | null;
  date?: string | null;
  league?: string | null;
  week_tag?: string | null;
  match_type?: string | null;
  match_format?: string | null;
  match_format_label?: string | null;
  rating_scope?: string | null;
  context_type?: string | null;
  team_1: PublicMatchPlayer[];
  team_2: PublicMatchPlayer[];
  score_t1?: number | null;
  score_t2?: number | null;
  winner?: string | null;
  elo_delta?: number | null;
  player_result?: "win" | "loss" | null;
  player_rating_before_jupr?: number | null;
  player_rating_after_jupr?: number | null;
  player_rating_delta_jupr?: number | null;
  rating_snapshot?: {
    team_1: PublicRatingSnapshotEntry[];
    team_2: PublicRatingSnapshotEntry[];
  };
};

export type PlayersResponse = {
  club: { id: string; slug: string; name: string };
  players: PublicPlayer[];
  filters?: { search: string; status: "active" | "inactive" | "all"; sort: string };
  summary?: { public_players: number; active_players: number; inactive_players: number; filtered_players: number };
  pagination?: { total: number; limit: number; offset: number; has_more: boolean };
};

export type PublicRatingHistoryPoint = {
  match_number: number;
  match_id?: string | number | null;
  date?: string | null;
  league?: string | null;
  match_type?: string | null;
  match_format: "doubles" | "singles";
  match_format_label: string;
  result?: "win" | "loss" | null;
  rating_before_jupr?: number | null;
  rating_after_jupr?: number | null;
  rating_delta_jupr?: number | null;
};

export type PublicRatingBreakdown = {
  format: "doubles" | "singles";
  label: string;
  matches: number;
  wins: number;
  losses: number;
  win_pct?: number | null;
  rating_delta_jupr?: number | null;
};

export type PublicBadgeAward = {
  badge_id: string;
  name: string;
  category: string;
  prestige: number;
  rarity?: string | null;
  icon_key?: string | null;
  description?: string | null;
  requirements?: string | null;
  count: number;
  last_earned_at?: string | null;
};

export type PublicTrophy = {
  badge_id: string;
  title: string;
  placement?: number | null;
  context_type?: string | null;
  context_label?: string | null;
  earned_at?: string | null;
};

export type PublicRelationship = {
  player_id: string | number;
  player_name: string;
  matches: number;
  wins: number;
  losses: number;
  win_pct?: number | null;
  balance?: number | null;
};

export type PublicSocialProjection = {
  available: boolean;
  identity: { linked: boolean; label: string };
  summary?: { events: number; matches: number; wins: number; losses: number; score_diff: number; last_appearance?: string | null } | null;
  skill_breakdown: Array<{ label: string; events: number; matches: number; wins: number; losses: number; score_diff: number }>;
  recent_events: Array<{ date?: string | null; name: string; event_type: string; skill_labels: string[]; matches: number; wins: number; losses: number; score_diff: number }>;
};

export type PlayerProfileResponse = {
  club: { id: string; slug: string; name: string };
  player: PublicPlayer;
  identity: { display_name: string; public_name_policy: "public_display_name"; verification_status: "enabled" | "pending" | "available" };
  verified_updates: { status: "enabled" | "pending" | "available"; can_request: boolean };
  rating_summary: {
    current_rating_jupr?: number | null;
    current_singles_rating_jupr?: number | null;
    starting_rating_jupr?: number | null;
    highest_rating_jupr?: number | null;
    lowest_rating_jupr?: number | null;
    last_10_record: string;
    last_10_delta_jupr?: number | null;
    current_streak?: string | null;
  };
  rating_breakdowns: PublicRatingBreakdown[];
  rating_history: PublicRatingHistoryPoint[];
  league_ratings: PublicLeagueRating[];
  awards: { badge_count: number; badge_award_count: number; prestige_total: number; badges: PublicBadgeAward[]; trophies: PublicTrophy[] };
  relationships: { best_partner?: PublicRelationship | null; rival?: PublicRelationship | null; partners: PublicRelationship[]; rivals: PublicRelationship[] };
  social: PublicSocialProjection;
  recent_matches: PublicMatch[];
  match_history: PublicMatch[];
  history: { total_matches: number; recent_limit: number; history_limit: number; has_more: boolean };
};

export type MatchesResponse = {
  club: { id: string; slug: string; name: string };
  matches: PublicMatch[];
};

export type MatchDetailResponse = {
  club: { id: string; slug: string; name: string };
  match: PublicMatch;
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
  courts?: Array<{ court_number: number; size?: number | null; matches: PublicLiveMatch[] }>;
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
  bracket?: { champion?: string | null; rows: Array<Record<string, string | number | boolean | null>> } | null;
  court_standings?: Array<Record<string, unknown>>;
};

export type LiveSessionsResponse = {
  club: { id: string; slug: string; name: string };
  sessions: PublicLiveSessionSummary[];
};

export type LiveSessionDetailResponse = {
  club: { id: string; slug: string; name: string };
  session: PublicLiveSessionDetail;
};

export type MatchExplorerContextResponse = {
  club: { id: string; slug: string; name: string };
  contexts: string[];
};

export type MatchExplorerPlayer = {
  id: string | number;
  name: string;
  overall_rating: number;
  overall_jupr?: number | null;
  context_rating: number;
  context_jupr?: number | null;
};

export type MatchExplorerPreview = {
  context: { name: string; k_factor: number };
  teams: {
    you: { average_rating: number; average_jupr?: number | null; players: MatchExplorerPlayer[] };
    opponents: { average_rating: number; average_jupr?: number | null; players: MatchExplorerPlayer[] };
  };
  expected: { you: number; opponents: number; label: string };
  score: { you: number; opponents: number };
  rating_delta: {
    you_team_elo: number;
    opponent_team_elo: number;
    you_team_jupr?: number | null;
    opponent_team_jupr?: number | null;
  };
};

export type MatchExplorerPreviewResponse = {
  club: { id: string; slug: string; name: string };
  preview: MatchExplorerPreview;
};

export type LeagueResultsLeague = {
  name: string;
  min_games?: number | null;
  k_factor?: number | null;
  start_week?: number | null;
  end_week?: number | null;
  num_weeks?: number | null;
};

export type LeagueResultsStanding = {
  rank?: number | null;
  player_id: string | number;
  player_name: string;
  rating?: number | null;
  rating_jupr?: number | null;
  wins?: number | null;
  losses?: number | null;
  matches_played?: number | null;
  win_pct?: number | null;
  rating_delta_jupr?: number | null;
};

export type LeagueResultsStatRow = {
  week_num?: number | null;
  player_id: string | number;
  player_name: string;
  games?: number | null;
  wins?: number | null;
  losses?: number | null;
  win_pct?: number | null;
  rating_jupr?: number | null;
  rating_delta_jupr?: number | null;
  rank?: number | null;
  prev_rank?: number | null;
  rank_delta?: number | null;
};

export type LeagueResultsHighlights = {
  scope?: "week" | "season" | string | null;
  week_num?: number | null;
  min_games?: number | null;
  biggest_climbers: LeagueResultsStatRow[];
  best_win_pct: LeagueResultsStatRow[];
  most_active: LeagueResultsStatRow[];
};

export type LeagueResultsPlayerOption = {
  player_id: string | number;
  player_name: string;
};

export type LeagueResultsPlayerSummary = LeagueResultsStatRow & {
  rating_jupr?: number | null;
  rank?: number | null;
};

export type LeagueResultsRecentMatch = {
  match_id: string | number;
  date?: string | null;
  week_num?: number | null;
  week_label?: string | null;
  partner?: LeagueResultsPlayerOption | null;
  opponents: LeagueResultsPlayerOption[];
  result: "W" | "L" | "D" | string;
  score_for: number;
  score_against: number;
  rating_delta_jupr?: number | null;
};

export type LeagueResultsResponse = {
  club: { id: string; slug: string; name: string };
  leagues: LeagueResultsLeague[];
  selected_league?: string | null;
  league?: LeagueResultsLeague | null;
  standings: LeagueResultsStanding[];
  weeks: Array<{ week_num: number; week_label: string; has_results?: boolean | null }>;
  selected_week?: number | null;
  weekly_results: LeagueResultsStatRow[];
  cumulative: LeagueResultsStatRow[];
  players: LeagueResultsPlayerOption[];
  selected_player_id?: string | number | null;
  player_summary?: LeagueResultsPlayerSummary | null;
  player_weekly: LeagueResultsStatRow[];
  recent_matches: LeagueResultsRecentMatch[];
  weekly_highlights: LeagueResultsHighlights;
  season_highlights: LeagueResultsHighlights;
  highlights: LeagueResultsHighlights;
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
    // Fall through to short text excerpt below.
  }
  return `${fallback} ${bodyText.slice(0, 240)}`;
}

async function fetchJson<T>(path: string, options: { noStore?: boolean } = {}): Promise<ApiResult<T>> {
  const apiBase = baseUrl();
  if (!apiBase) return { data: null, error: "Missing JUPR API base URL environment variable." };
  const url = `${apiBase.replace(/\/$/, "")}${path}`;
  try {
    const response = await fetch(url, options.noStore ? { cache: "no-store" } : { next: { revalidate: 60 } });
    if (!response.ok) return { data: null, error: await apiErrorMessage(response) };
    return { data: (await response.json()) as T, error: null };
  } catch (error) {
    return { data: null, error: `Unable to reach API: ${error instanceof Error ? error.message : "Unknown error"}` };
  }
}

export async function getClub(clubSlug: string): Promise<ApiResult<ClubSummary>> {
  return fetchJson<ClubSummary>(`/clubs/${clubSlug}`);
}

export type LeaderboardRequest = {
  leagueName?: string | null;
  status?: "active" | "inactive" | "all";
  search?: string | null;
  sort?: "rank" | "rating" | "matches" | "win_pct" | "gain" | "name";
  playerId?: string | number | null;
  limit?: number;
  offset?: number;
};

export async function getClubLeaderboard(
  clubSlug: string,
  options: LeaderboardRequest = {}
): Promise<ApiResult<LeaderboardResponse>> {
  const params = new URLSearchParams();
  if (options.leagueName) params.set("league_name", String(options.leagueName));
  if (options.status) params.set("status", options.status);
  if (options.search) params.set("q", String(options.search));
  if (options.sort) params.set("sort", options.sort);
  if (options.playerId != null && String(options.playerId).trim()) params.set("player_id", String(options.playerId));
  if (options.limit != null) params.set("limit", String(options.limit));
  if (options.offset != null) params.set("offset", String(options.offset));
  const query = params.toString();
  return fetchJson<LeaderboardResponse>(`/clubs/${clubSlug}/leaderboards${query ? `?${query}` : ""}`);
}

export async function getClubPlayers(
  clubSlug: string,
  filters: { q?: string | null; status?: "active" | "inactive" | "all" | null; sort?: string | null; limit?: number | null; offset?: number | null } = {}
): Promise<ApiResult<PlayersResponse>> {
  const params = new URLSearchParams();
  if (filters.q) params.set("q", filters.q);
  if (filters.status) params.set("status", filters.status);
  if (filters.sort) params.set("sort", filters.sort);
  if (filters.limit != null) params.set("limit", String(filters.limit));
  if (filters.offset != null) params.set("offset", String(filters.offset));
  const query = params.toString();
  return fetchJson<PlayersResponse>(`/clubs/${clubSlug}/players${query ? `?${query}` : ""}`);
}

export async function getClubPlayerProfile(
  clubSlug: string,
  playerId: string,
  limits: { recent?: number; history?: number } = {}
): Promise<ApiResult<PlayerProfileResponse>> {
  const params = new URLSearchParams();
  if (limits.recent != null) params.set("recent_limit", String(limits.recent));
  if (limits.history != null) params.set("history_limit", String(limits.history));
  const query = params.toString();
  return fetchJson<PlayerProfileResponse>(`/clubs/${clubSlug}/players/${playerId}${query ? `?${query}` : ""}`);
}

export async function getClubMatches(clubSlug: string): Promise<ApiResult<MatchesResponse>> {
  return fetchJson<MatchesResponse>(`/clubs/${clubSlug}/matches`);
}

export async function getClubMatch(clubSlug: string, matchId: string): Promise<ApiResult<MatchDetailResponse>> {
  return fetchJson<MatchDetailResponse>(`/clubs/${clubSlug}/matches/${matchId}`);
}

export async function getClubPlayerMatches(clubSlug: string, playerId: string): Promise<ApiResult<MatchesResponse>> {
  return fetchJson<MatchesResponse>(`/clubs/${clubSlug}/players/${playerId}/matches`);
}

export async function getClubLiveSessions(clubSlug: string): Promise<ApiResult<LiveSessionsResponse>> {
  return fetchJson<LiveSessionsResponse>(`/clubs/${clubSlug}/live-sessions`, { noStore: true });
}

export async function getClubLiveSession(clubSlug: string, sessionKey: string): Promise<ApiResult<LiveSessionDetailResponse>> {
  return fetchJson<LiveSessionDetailResponse>(`/clubs/${clubSlug}/live-sessions/${sessionKey}`, { noStore: true });
}

export async function getClubMatchExplorerContext(clubSlug: string): Promise<ApiResult<MatchExplorerContextResponse>> {
  return fetchJson<MatchExplorerContextResponse>(`/clubs/${clubSlug}/match-explorer`);
}

export async function getClubLeagueResults(
  clubSlug: string,
  leagueName?: string | null,
  week?: number | null,
  player?: string | number | null,
  weeklyMinGames?: number | null
): Promise<ApiResult<LeagueResultsResponse>> {
  const params = new URLSearchParams();
  if (leagueName) params.set("league_name", leagueName);
  if (week) params.set("week", String(week));
  if (player) params.set("player", String(player));
  if (weeklyMinGames) params.set("weekly_min_games", String(weeklyMinGames));
  const query = params.toString();
  return fetchJson<LeagueResultsResponse>(`/clubs/${clubSlug}/league-results${query ? `?${query}` : ""}`);
}
