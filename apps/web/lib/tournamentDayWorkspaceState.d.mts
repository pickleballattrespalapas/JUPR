export type TournamentDayWorkspacePanelFocus = "board" | "queue" | "draws" | "corrections";

export type TournamentDayCommandAction =
  | "activate_day"
  | "activate_draw"
  | "pause_draw"
  | "resume_draw"
  | "auto_fill_courts"
  | "assign_next_court"
  | "assign_game_to_court"
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
  activation_state?: string | null;
  [key: string]: unknown;
};

export function dayActionConfirmation(action: TournamentDayCommandAction): string;
export function dayRunHasStarted(state: string | null | undefined): boolean;
export function dayRunAcceptsLiveCommands(state: string | null | undefined): boolean;
export function visibleServerQueue<T extends ServerQueueRow>(queue: readonly T[], drawId: string): T[];
export function readyActiveDrawQueue<T extends ServerQueueRow>(queue: readonly T[], draws: readonly DayDrawRow[]): T[];
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
export function validateDayScoreDraft(
  scoreA: string | number,
  scoreB: string | number,
  scoring?: Record<string, unknown> | null,
  unusualScoreAcknowledged?: boolean
):
  | { ok: true; scoreA: number; scoreB: number; unusual: boolean; reasons: string[]; acknowledgementRequired: boolean; scoringFormat: string }
  | { ok: false; message: string; impossible?: boolean; reasons?: string[] };
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
  winnerTeamId: unknown,
  resultNote: unknown
):
  | { ok: true; resultType: "FORFEIT" | "NO_SHOW" | "RETIREMENT"; winnerTeamId: string; resultNote: string }
  | { ok: false; message: string };
