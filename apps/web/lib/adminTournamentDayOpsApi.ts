export type AdminTournamentDayCommandAction =
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

export type AdminTournamentDayBlocker = {
  code: string;
  message: string;
  title?: string | null;
  detail?: string | null;
  scope?: string | null;
  entity_id?: string | null;
};

export type AdminTournamentDayBlockerValue = AdminTournamentDayBlocker | string;

export type AdminTournamentDayReadiness = {
  ready: boolean;
  confirmation?: string | null;
  blockers: AdminTournamentDayBlockerValue[];
};

export type AdminTournamentDayOption = {
  id: string;
  label: string;
  event_date?: string | null;
  sort_order?: number | null;
  court_count?: number | null;
  court_labels?: string[];
  available_court_ids?: string[];
};

export type AdminTournamentDaySide = {
  team_id?: string | null;
  name: string;
  participant_names: string[];
  competition_status?: "ACTIVE" | "RETIRED";
};

export type AdminTournamentDayGame = {
  id: string;
  draw_id: string;
  draw_name: string;
  state: string;
  stage: string;
  round_label: string;
  slot_label?: string | null;
  playoff_game_code?: string | null;
  playoff_round?: string | null;
  team_a: AdminTournamentDaySide;
  team_b: AdminTournamentDaySide;
  score_a?: number | null;
  score_b?: number | null;
  scoring: {
    format?: "GAME_TO_11" | "GAME_TO_15" | "GAME_TO_21" | "BEST_2_OF_3" | null;
    target?: number | null;
    win_by_two?: boolean | null;
    best_of_three_score_semantics?: string | null;
    blocker?: string | null;
  };
  result_type?: "PLAYED" | "FORFEIT" | "NO_SHOW" | "RETIREMENT";
  result_note?: string | null;
  result_recorded_by?: string | null;
  score_review?: Record<string, unknown>;
  winner_name?: string | null;
  finalized_at?: string | null;
  updated_at?: string | null;
  version: string;
  queue_entry_version?: string;
  court_id?: string | null;
  blockers: AdminTournamentDayBlockerValue[];
  correction_readiness: AdminTournamentDayReadiness;
};

export type AdminTournamentDayCourtAssignment = {
  id: string;
  game_id: string;
  state: string;
  version: string;
  assigned_at?: string | null;
  started_at?: string | null;
  reserved_at?: string | null;
};

export type AdminTournamentDayCourt = {
  id: string;
  label: string;
  position: number;
  state: string;
  version: string;
  current_assignment?: AdminTournamentDayCourtAssignment | null;
  next_assignment?: AdminTournamentDayCourtAssignment | null;
};

export type AdminTournamentDayQueueEntry = {
  game_id: string;
  draw_id: string;
  position: number;
  priority?: number | null;
  state: string;
  version: string;
  court_id?: string | null;
  reserved_court_id?: string | null;
  immediate_fill_candidate?: boolean;
  eligible_since?: string | null;
  reason?: string | null;
  note?: string | null;
  reserved_at?: string | null;
  blockers: AdminTournamentDayBlockerValue[];
};

export type AdminTournamentDayHeldEntry = {
  game_id: string;
  draw_id: string;
  state: string;
  reason?: string | null;
  note?: string | null;
  held_at?: string | null;
  version: string;
  blockers: AdminTournamentDayBlockerValue[];
};

export type AdminTournamentDayDraw = {
  id: string;
  name: string;
  state: string;
  activation_state: string;
  version: string;
  stage?: string | null;
  total_games: number;
  finalized_games: number;
  queued_games: number;
  active_games: number;
  held_games: number;
  readiness: {
    activate: AdminTournamentDayReadiness;
    pause: AdminTournamentDayReadiness;
    resume: AdminTournamentDayReadiness;
    assignments: AdminTournamentDayReadiness;
    generate_playoffs: AdminTournamentDayReadiness & {
      allowed_advance_counts: number[];
      default_advance_count: number | null;
    };
    podium: AdminTournamentDayReadiness;
    closeout?: AdminTournamentDayReadiness;
  };
  progression?: {
    allowed_advance_counts: number[];
    default_advance_count: number | null;
    podium_href?: string | null;
    review_href?: string | null;
  };
};

export type AdminTournamentDayOperation = {
  operation_key: string;
  client_idempotency_key: string;
  action: string;
  status: string;
  entity_label?: string | null;
  updated_at?: string | null;
  error_text?: string | null;
  retryable?: boolean;
};

export type AdminTournamentDayRun = {
  id: string;
  registration_day_id: string;
  state: string;
  version: string;
  updated_at?: string | null;
};

export type AdminTournamentDayWorkspaceSnapshot = {
  ok: boolean;
  mode: string;
  scope: {
    club_id: string;
    tournament_id: string;
    registration_day_id: string;
  };
  tournament: { id: string; name: string; status?: string | null };
  day_scope: {
    selected_day_id: string;
    selected_day: AdminTournamentDayOption;
    available_days: AdminTournamentDayOption[];
  };
  day_run: AdminTournamentDayRun;
  state_fingerprint: string;
  queue_version: string;
  generated_at?: string | null;
  summary: {
    courts: number;
    available_courts: number;
    active_draws: number;
    eligible_games: number;
    reserved_games: number;
    held_games: number;
    completed_games: number;
  };
  draws: AdminTournamentDayDraw[];
  activated_draws?: AdminTournamentDayDraw[];
  courts: AdminTournamentDayCourt[];
  games: AdminTournamentDayGame[];
  eligible_queue: AdminTournamentDayQueueEntry[];
  reserved_queue: AdminTournamentDayQueueEntry[];
  held_games: AdminTournamentDayHeldEntry[];
  blocked_games: AdminTournamentDayHeldEntry[];
  operations: AdminTournamentDayOperation[];
  readiness: {
    activate_day: AdminTournamentDayReadiness;
    auto_fill_courts: AdminTournamentDayReadiness;
    close_day: AdminTournamentDayReadiness;
    correct_completed_score: AdminTournamentDayReadiness;
  };
  runtime?: { writes_enabled?: boolean; warnings?: string[] };
  warnings?: string[];
};

export type AdminTournamentDayCommandExpected = {
  day_run_version: string;
  state_fingerprint: string;
  draw_version?: string;
  game_version?: string;
  court_version?: string;
  target_court_version?: string;
  queue_version?: string;
  queue_entry_version?: string;
};

export type AdminTournamentDayCommandPayload = {
  draw_id?: string;
  advance_count?: number;
  game_id?: string;
  court_id?: string;
  score_a?: number;
  score_b?: number;
  unusual_score_acknowledgement?: boolean;
  result_type?: "FORFEIT" | "NO_SHOW" | "RETIREMENT";
  non_playing_team_id?: string;
  result_note?: string;
};

export type AdminTournamentDayCommandRequest = {
  action: AdminTournamentDayCommandAction;
  client_idempotency_key: string;
  confirmation_text: string;
  expected: AdminTournamentDayCommandExpected;
  payload: AdminTournamentDayCommandPayload;
};

export type AdminTournamentDayCommandResponse = {
  command: {
    action: AdminTournamentDayCommandAction | "reconcile";
    confirmation_text?: string | null;
    idempotent_replay?: boolean;
  };
  operation: AdminTournamentDayOperation;
  snapshot: AdminTournamentDayWorkspaceSnapshot;
};

type ApiOptions = {
  apiBase: string;
  clubId: string;
  tournamentId: string;
  dayId: string;
  accessToken: string;
  signal?: AbortSignal;
};

export class AdminTournamentDayOpsApiError extends Error {
  status: number;
  detail: unknown;

  constructor(message: string, status: number, detail: unknown) {
    super(message);
    this.name = "AdminTournamentDayOpsApiError";
    this.status = status;
    this.detail = detail;
  }
}

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function dayBase(options: ApiOptions): string {
  return apiUrl(
    options.apiBase,
    `/admin/clubs/${encodeURIComponent(options.clubId)}/tournament-live/tournaments/${encodeURIComponent(options.tournamentId)}/days/${encodeURIComponent(options.dayId)}`
  );
}

async function requestJson<T>(
  url: string,
  accessToken: string,
  options: RequestInit
): Promise<T> {
  const headers = new Headers(options.headers);
  headers.set("Authorization", `Bearer ${accessToken}`);
  if (options.body) headers.set("Content-Type", "application/json");
  const response = await fetch(url, { ...options, headers, cache: "no-store" });
  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const objectPayload = payload && typeof payload === "object" ? payload as { detail?: unknown; message?: unknown } : null;
    const detail = objectPayload?.detail ?? objectPayload?.message;
    throw new AdminTournamentDayOpsApiError(
      typeof detail === "string" ? detail : `Tournament day operations API error (${response.status}).`,
      response.status,
      payload
    );
  }
  return payload as T;
}

function dayOption(row: unknown): AdminTournamentDayOption | null {
  if (!row || typeof row !== "object") return null;
  const value = row as { [key: string]: unknown };
  const id = String(value.id || "").trim();
  if (!id) return null;
  const labels = Array.isArray(value.court_labels)
    ? value.court_labels.map((label) => String(label || ""))
    : [];
  const availableCourtIds = Array.isArray(value.available_court_ids)
    ? value.available_court_ids.map((courtId) => String(courtId || "")).filter(Boolean)
    : [];
  return {
    id,
    label: String(value.label || value.name || value.event_date || "Tournament day"),
    event_date: value.event_date == null ? null : String(value.event_date),
    sort_order: Number.isFinite(Number(value.sort_order)) ? Number(value.sort_order) : null,
    court_count: Number.isFinite(Number(value.court_count)) ? Number(value.court_count) : null,
    court_labels: labels,
    available_court_ids: availableCourtIds
  };
}

export async function fetchAdminTournamentDayOptions(options: Omit<ApiOptions, "dayId">): Promise<AdminTournamentDayOption[]> {
  const payload = await requestJson<{ registration_days?: unknown[] }>(
    apiUrl(
      options.apiBase,
      `/admin/clubs/${encodeURIComponent(options.clubId)}/tournament-live/tournaments/${encodeURIComponent(options.tournamentId)}/snapshot`
    ),
    options.accessToken,
    { method: "GET", signal: options.signal }
  );
  return (payload.registration_days || [])
    .map(dayOption)
    .filter((row): row is AdminTournamentDayOption => Boolean(row))
    .sort((left, right) => (left.sort_order ?? 0) - (right.sort_order ?? 0));
}

export async function fetchAdminTournamentDayWorkspace(options: ApiOptions): Promise<AdminTournamentDayWorkspaceSnapshot> {
  return requestJson<AdminTournamentDayWorkspaceSnapshot>(
    `${dayBase(options)}/snapshot`,
    options.accessToken,
    { method: "GET", signal: options.signal }
  );
}

export async function executeAdminTournamentDayCommand(
  options: ApiOptions & { request: AdminTournamentDayCommandRequest }
): Promise<AdminTournamentDayCommandResponse> {
  return requestJson<AdminTournamentDayCommandResponse>(
    `${dayBase(options)}/commands`,
    options.accessToken,
    { method: "POST", body: JSON.stringify(options.request), signal: options.signal }
  );
}

export async function reconcileAdminTournamentDayOperation(
  options: ApiOptions & { operationKey: string; confirmationText: string }
): Promise<AdminTournamentDayCommandResponse> {
  return requestJson<AdminTournamentDayCommandResponse>(
    `${dayBase(options)}/operations/${encodeURIComponent(options.operationKey)}/reconcile`,
    options.accessToken,
    {
      method: "POST",
      body: JSON.stringify({ confirmation_text: options.confirmationText }),
      signal: options.signal
    }
  );
}
