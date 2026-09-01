export type TournamentDayWorkspacePanelFocus = "board" | "queue" | "draws" | "corrections";

export type TournamentDayCommandAction =
  | "activate_day"
  | "activate_draw"
  | "pause_draw"
  | "resume_draw"
  | "auto_fill_courts"
  | "assign_next_court"
  | "assign_game_to_court"
  | "reserve_game_for_court"
  | "requeue_game"
  | "move_game_to_court"
  | "score_and_release"
  | "correct_completed_score"
  | "record_non_played_result"
  | "generate_playoffs"
  | "close_day";

export type TournamentDayWorkspaceFocus = {
  dayId: string;
  drawId: string;
  courtId: string;
  gameId: string;
  panel: TournamentDayWorkspacePanelFocus;
};

export type ServerQueueRow = {
  draw_id?: string | null;
  state?: string | null;
  blockers?: readonly unknown[] | null;
  [key: string]: unknown;
};

export type DayDrawRow = {
  id?: string | null;
  name?: string | null;
  activation_state?: string | null;
  round_robin_complete?: boolean;
  progression_status?: string | null;
  playoff_review_fingerprint?: string | null;
  status?: string | null;
  readiness?: { generate_playoffs?: { ready?: boolean } };
  [key: string]: unknown;
};

export type PlayoffProgressionSnapshot<T extends DayDrawRow = DayDrawRow> = {
  draws?: readonly T[] | null;
  progression_alerts?: readonly {
    draw_id?: string | null;
    ready?: boolean;
  }[] | null;
};

export type PlayoffReviewTemplate = {
  code?: string | null;
  advance_count?: number | null;
  applicable_rounds?: readonly string[] | null;
  rounds?: readonly (string | { code?: string | null; round?: string | null })[] | null;
  games?: readonly { round?: string | null; playoff_round?: string | null }[] | null;
  default_seed_team_ids?: readonly string[] | null;
  default_round_scoring?: Readonly<Record<string, string>> | null;
};

export type PlayoffReviewContract = {
  eligible_team_ids?: readonly string[] | null;
  templates?: readonly PlayoffReviewTemplate[] | null;
  scoring_formats?: readonly { code?: string | null }[] | null;
};

export type PlayoffReviewConfiguration = {
  template_code?: string | null;
  seed_team_ids?: readonly string[] | null;
  round_scoring?: Readonly<Record<string, string>> | null;
};

export type TournamentDayMedalMatchKind = "gold" | "bronze";

export type TournamentDayMedalGame = {
  stage?: string | null;
  round_label?: string | null;
  playoff_round?: string | null;
};

export function dayActionConfirmation(action: TournamentDayCommandAction): string;
export function dayRunHasStarted(state: string | null | undefined): boolean;
export function dayRunAcceptsLiveCommands(state: string | null | undefined): boolean;
export function visibleServerQueue<T extends ServerQueueRow>(queue: readonly T[], drawId: string): T[];
export function oldestReadyQueue<T extends ServerQueueRow>(queue: readonly T[]): T[];
export function readyActiveDrawQueue<T extends ServerQueueRow>(queue: readonly T[], draws: readonly DayDrawRow[]): T[];
export function tournamentDayMedalMatchKind(
  game: TournamentDayMedalGame | null | undefined
): TournamentDayMedalMatchKind | null;
export function resetFocusForDay(
  current: Partial<TournamentDayWorkspaceFocus>,
  dayId: string
): TournamentDayWorkspaceFocus;
export function workspaceScopeKey(accessToken: string, tournamentId: string, dayId: string): string;
export function retainedDayCommandStorageKey(clubId: string, tournamentId: string, dayId: string): string;
export function advanceCountSelection(
  allowedCounts: readonly number[],
  defaultCount: number | null | undefined,
  currentSelection: string | number | null | undefined
): string;
export function readyPlayoffReviewDraws<T extends DayDrawRow>(
  snapshot: PlayoffProgressionSnapshot<T> | null | undefined
): T[];
export function newlyReadyPlayoffNotice(
  previous: PlayoffProgressionSnapshot | null | undefined,
  current: PlayoffProgressionSnapshot | null | undefined
): string | null;
export function playoffTemplateRoundCodes(template: PlayoffReviewTemplate | null | undefined): string[];
export function validatePlayoffReviewConfiguration(
  review: PlayoffReviewContract | null | undefined,
  configuration: PlayoffReviewConfiguration | null | undefined
):
  | { ok: false; message: string }
  | {
      ok: true;
      template: PlayoffReviewTemplate;
      advanceCount: number;
      seedTeamIds: string[];
      roundScoring: Readonly<Record<string, string>>;
      roundCodes: string[];
    };
export function validateDayScoreDraft(
  scoreA: string | number,
  scoreB: string | number,
  scoring?: Record<string, unknown> | null,
  unusualScoreAcknowledged?: boolean
):
  | { ok: true; scoreA: number; scoreB: number; unusual: boolean; reasons: string[]; acknowledgementRequired: boolean; scoringFormat: string }
  | { ok: false; message: string; impossible?: boolean; reasons?: string[] };
export type BestOfThreeGameScoreDraft = {
  game_number: 1 | 2 | 3;
  score_a: string | number;
  score_b: string | number;
};
export type BestOfThreeGameScore = {
  game_number: 1 | 2 | 3;
  score_a: number;
  score_b: number;
};
export function validateBestOfThreeGameScores(
  gameScores: readonly BestOfThreeGameScoreDraft[],
  scoring?: Record<string, unknown> | null,
  unusualScoreAcknowledged?: boolean
):
  | {
      ok: true;
      scoreA: number;
      scoreB: number;
      gameScores: BestOfThreeGameScore[];
      unusual: boolean;
      reasons: string[];
      acknowledgementRequired: boolean;
      scoringFormat: "BEST_2_OF_3";
    }
  | { ok: false; message: string; impossible?: boolean; reasons?: string[] };
export function validateBestOfThreeRetirementGameScores(
  gameScores: readonly BestOfThreeGameScoreDraft[],
  scoring?: Record<string, unknown> | null,
  unusualScoreAcknowledged?: boolean
):
  | {
      ok: true;
      gameScores: BestOfThreeGameScore[];
      unusual: boolean;
      reasons: string[];
      acknowledgementRequired: boolean;
      scoringFormat: "BEST_2_OF_3";
    }
  | { ok: false; message: string; impossible?: boolean; reasons?: string[] };
export function validateBestOfThreeCorrectionDraft(
  gameScores: readonly BestOfThreeGameScoreDraft[],
  currentGameScores: readonly BestOfThreeGameScore[],
  scoring?: Record<string, unknown> | null,
  unusualScoreAcknowledged?: boolean
): ReturnType<typeof validateBestOfThreeGameScores>;
export function validateDayCorrectionDraft(
  scoreA: string | number,
  scoreB: string | number,
  currentScoreA: number | null | undefined,
  currentScoreB: number | null | undefined,
  scoring?: Record<string, unknown> | null,
  unusualScoreAcknowledged?: boolean
): ReturnType<typeof validateDayScoreDraft>;
export function validateNonPlayedOutcomeDraft(
  resultType: unknown,
  nonPlayingTeamId: unknown,
  resultNote: unknown
):
  | { ok: true; resultType: "FORFEIT" | "NO_SHOW" | "RETIREMENT"; nonPlayingTeamId: string; resultNote: string }
  | { ok: false; message: string };
