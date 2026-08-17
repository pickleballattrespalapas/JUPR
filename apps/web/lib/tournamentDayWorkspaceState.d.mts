export type TournamentDayWorkspacePanelFocus = "board" | "queue" | "draws" | "corrections";

export type TournamentDayCommandAction =
  | "activate_day"
  | "activate_draw"
  | "pause_draw"
  | "resume_draw"
  | "auto_fill_courts"
  | "score_and_release"
  | "correct_completed_score"
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
  [key: string]: unknown;
};

export function dayActionConfirmation(action: TournamentDayCommandAction): string;
export function dayRunHasStarted(state: string | null | undefined): boolean;
export function dayRunAcceptsLiveCommands(state: string | null | undefined): boolean;
export function visibleServerQueue<T extends ServerQueueRow>(queue: readonly T[], drawId: string): T[];
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
  scoreB: string | number
):
  | { ok: true; scoreA: number; scoreB: number }
  | { ok: false; message: string };
export function validateDayCorrectionDraft(
  scoreA: string | number,
  scoreB: string | number,
  currentScoreA: number | null | undefined,
  currentScoreB: number | null | undefined
):
  | { ok: true; scoreA: number; scoreB: number }
  | { ok: false; message: string };
