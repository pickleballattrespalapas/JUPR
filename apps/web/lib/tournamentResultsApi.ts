export type PublicTournamentResultsChoice = {
  id: string;
  name: string;
  status: string;
  start_date?: string | null;
  end_date?: string | null;
  settings: {
    registration_slug?: string | null;
    registration_status?: string | null;
  };
};

export type PublicTournamentResultsIndex = {
  club: { id: string; slug: string; name: string };
  view: "current" | "past";
  tournaments: PublicTournamentResultsChoice[];
};

export type PublicTournamentSeriesGameScore = {
  game_number: number;
  score_a: number;
  score_b: number;
};

export type PublicTournamentGameResult = {
  public_game_key: string;
  stage: string;
  round_number?: number | null;
  slot_number?: number | null;
  playoff_game_code?: string | null;
  playoff_round?: string | null;
  team_a_name: string;
  team_b_name: string;
  score_a?: number | null;
  score_b?: number | null;
  winner_name?: string | null;
  outcome_label?: string | null;
  state: "PENDING" | "READY" | "FINAL";
  finalized_at?: string | null;
  game_scores?: PublicTournamentSeriesGameScore[];
};

export type PublicTournamentTiebreakExplanationStep = {
  criterion: string;
  outcome: string;
  detail: string;
};

export type PublicTournamentTiebreakExplanation = {
  title: string;
  summary: string;
  steps: PublicTournamentTiebreakExplanationStep[];
};

export type PublicTournamentDrawResult = {
  public_draw_key: string;
  name: string;
  state: "SCHEDULED" | "READY" | "LIVE" | "COMPLETE";
  round_robin_complete?: boolean;
  ranking_policy?: {
    description?: string | null;
    criteria?: string[];
    retired_teams_eligible?: boolean;
  } | null;
  event_family_label: string;
  division_name: string;
  event_type?: string | null;
  scheduled_days: Array<{
    label: string;
    event_date?: string | null;
  }>;
  teams: Array<{
    public_team_key: string;
    team_number?: number | null;
    seed?: number | null;
    name: string;
    competition_status?: string | null;
  }>;
  standings: Array<{
    public_team_key: string;
    rank?: number | null;
    team_name: string;
    wins?: number | null;
    losses?: number | null;
    points_for?: number | null;
    points_against?: number | null;
    differential?: number | null;
    competition_status?: string | null;
    retired?: boolean;
  }>;
  tiebreak_explanations?: PublicTournamentTiebreakExplanation[];
  scores: PublicTournamentGameResult[];
  bracket: PublicTournamentGameResult[];
  podium: Array<{
    placement?: number | null;
    medal?: string | null;
    team_name: string;
  }>;
};

export type PublicTournamentResults = {
  club: { id: string; slug: string; name: string };
  tournament: PublicTournamentResultsChoice;
  draws: PublicTournamentDrawResult[];
};

export type TournamentResultsApiResult<T> = {
  data: T | null;
  error: string | null;
  status?: number | null;
};

function apiBaseUrl(): string | null {
  return (
    process.env.JUPR_API_BASE_URL ||
    process.env.NEXT_PUBLIC_JUPR_API_BASE_URL ||
    null
  );
}

async function fetchTournamentJson<T>(
  path: string,
  init?: RequestInit
): Promise<TournamentResultsApiResult<T>> {
  const apiBase = apiBaseUrl();
  if (!apiBase) {
    return { data: null, error: "Tournament results are unavailable." };
  }
  try {
    const response = await fetch(`${apiBase.replace(/\/$/, "")}${path}`, init);
    if (!response.ok) {
      return {
        data: null,
        error:
          response.status === 404
            ? "Tournament results were not found."
            : `Tournament results API error (${response.status}).`,
        status: response.status
      };
    }
    return {
      data: (await response.json()) as T,
      error: null,
      status: response.status
    };
  } catch (error) {
    return {
      data: null,
      error:
        error instanceof Error ? error.message : "Unable to reach the API.",
      status: null
    };
  }
}

export async function getPublicTournamentResultsIndex(
  clubSlug: string,
  view: "current" | "past" = "current"
): Promise<TournamentResultsApiResult<PublicTournamentResultsIndex>> {
  const query = new URLSearchParams({ view });
  return fetchTournamentJson<PublicTournamentResultsIndex>(
    `/clubs/${clubSlug}/tournaments?${query.toString()}`,
    { next: { revalidate: 60 } }
  );
}

export async function getPublicTournamentResults(
  clubSlug: string,
  tournamentId: string
): Promise<TournamentResultsApiResult<PublicTournamentResults>> {
  const query = new URLSearchParams({ tournament_id: tournamentId });
  return fetchTournamentJson<PublicTournamentResults>(
    `/clubs/${clubSlug}/tournament-results?${query.toString()}`,
    { cache: "no-store" }
  );
}
