type DrawStatusSource = {
  status?: string | null;
};

type DrawLifecycleStatusSource = {
  counts: {
    games?: number;
    finalized_games?: number;
    open_games?: number;
    published_games?: number;
    duplicate_publications?: number;
    duplicate_official_links?: number;
    mismatched_official_matches?: number;
    official_matches_without_publication_evidence?: number;
  };
  states: Record<string, unknown>;
};

export function isInactiveTournamentDraw(draw: DrawStatusSource): boolean;
export function drawOperationalStatus(
  draw: DrawStatusSource,
  lifecycleDraw?: DrawLifecycleStatusSource
): string;
