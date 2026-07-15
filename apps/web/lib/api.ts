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
  club: { id: string; slug: string; name: string };
  leaderboard: LeaderboardEntry[];
};

export type PublicPlayer = {
  id: string | number;
  club_id?: string;
  name: string;
  rating?: number | null;
  wins?: number | null;
  losses?: number | null;
  matches_played?: number | null;
  singles_rating?: number | null;
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
  starting_rating?: number | null;
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
  rating_scope?: string | null;
  context_type?: string | null;
  context_id?: string | null;
  team_1: PublicMatchPlayer[];
  team_2: PublicMatchPlayer[];
  score_t1?: number | null;
  score_t2?: number | null;
  winner?: string | null;
  elo_delta?: number | null;
  rating_snapshot?: {
    team_1: PublicRatingSnapshotEntry[];
    team_2: PublicRatingSnapshotEntry[];
  };
};

export type PlayersResponse = {
  club: { id: string; slug: string; name: string };
  players: PublicPlayer[];
};

export type PlayerProfileResponse = {
  club: { id: string; slug: string; name: string };
  player: PublicPlayer;
  league_ratings: PublicLeagueRating[];
  recent_matches: PublicMatch[];
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
};
