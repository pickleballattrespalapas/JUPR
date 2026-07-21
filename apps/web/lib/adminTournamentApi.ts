export type AdminTournamentStatusResponse = {
  enabled: boolean;
  status: string;
  tournaments_endpoint?: string | null;
  tournament_detail_endpoint?: string | null;
  registration_update_endpoint?: string | null;
  selection_update_endpoint?: string | null;
  bulk_registration_update_endpoint?: string | null;
  registration_export_endpoint?: string | null;
  broadcast_preview_endpoint?: string | null;
  tournament_update_endpoint?: string | null;
  import_handoff_endpoint?: string | null;
  mutation_runtime?: {
    environment: string;
    staging_only: boolean;
    service_role_ready: boolean;
    surface_flags: Record<string, { name: string; enabled: boolean }>;
  };
  operations_runtime?: {
    environment: string;
    operations_mutations_enabled: boolean;
    official_publish_enabled: boolean;
    email_handoff_enabled: boolean;
    auto_player_updates_enabled: boolean;
    email_mode: string;
    staging_only: boolean;
  };
  streamlit_fallback_url?: string;
  tournament_count?: number | null;
  warnings: string[];
};

export type AdminTournament = {
  id: string;
  name: string;
  status: string;
  start_date?: string | null;
  end_date?: string | null;
  registration_slug?: string | null;
  registration_status?: string | null;
  registration_count?: number | null;
  selection_count?: number | null;
  created_at?: string | null;
  updated_at?: string | null;
};

export type AdminTournamentRegistration = {
  id: string;
  player_id?: string | number | null;
  display_name: string;
  email?: string | null;
  phone?: string | null;
  registration_status?: string | null;
  payment_status?: string | null;
  notes?: string | null;
  wants_partner_board_contact?: boolean | null;
  selection_count?: number | null;
  created_at?: string | null;
  updated_at?: string | null;
};

export type AdminTournamentSelection = {
  id: string;
  registration_id: string;
  registration_day_id?: string | null;
  event_option_id?: string | null;
  event_label?: string | null;
  partner_mode?: string | null;
  partner_name?: string | null;
  partner_email?: string | null;
  partner_phone?: string | null;
  partner_note?: string | null;
  show_on_partner_board?: boolean | null;
  created_at?: string | null;
  updated_at?: string | null;
};

export type AdminTournamentDraw = {
  id: string;
  tournament_id?: string | null;
  registration_day_id?: string | null;
  event_option_id?: string | null;
  name: string;
  status?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
};

export type AdminTournamentOpsPlayer = {
  id: number;
  name: string;
  active?: boolean | null;
};

export type AdminTournamentOpsTeam = {
  id?: string | null;
  tournament_id?: string | null;
  draw_id?: string | null;
  registration_day_id?: string | null;
  event_option_id?: string | null;
  team_number?: number | null;
  player1_id?: number | null;
  player2_id?: number | null;
  seed?: number | null;
  source?: string | null;
  notes?: string | null;
  updated_at?: string | null;
};

export type AdminTournamentListResponse = {
  ok: boolean;
  mode?: string;
  tournaments: AdminTournament[];
  count: number;
  warnings?: string[];
};

export type AdminTournamentDetailResponse = {
  ok: boolean;
  mode?: string;
  tournament: AdminTournament;
  settings?: Record<string, unknown>;
  days: Array<Record<string, unknown>>;
  event_options: Array<Record<string, unknown>>;
  registrations: AdminTournamentRegistration[];
  selections: AdminTournamentSelection[];
  summary: {
    registrations: number;
    selections: number;
    by_registration_status: Record<string, number>;
    by_payment_status: Record<string, number>;
  };
  state_fingerprint?: string;
  warnings?: string[];
};

export type AdminTournamentOpsSnapshotResponse = {
  ok: boolean;
  mode?: string;
  tournament: AdminTournament;
  draw_id?: string | null;
  summary: {
    draws: number;
    teams: number;
    games: number;
    podium: number;
    completed_games?: number;
  };
  draws: AdminTournamentDraw[];
  teams: AdminTournamentOpsTeam[];
  games: Array<Record<string, unknown>>;
  podium: Array<Record<string, unknown>>;
  registration_days?: Array<Record<string, unknown>>;
  event_options?: Array<Record<string, unknown>>;
  players?: AdminTournamentOpsPlayer[];
  state_fingerprint?: string | null;
  state_ready?: boolean;
  operation_runtime?: {
    environment: string;
    operations_mutations_enabled: boolean;
    official_publish_enabled: boolean;
    email_handoff_enabled: boolean;
    auto_player_updates_enabled: boolean;
    email_mode: string;
    staging_only: boolean;
  };
  streamlit_fallback_url?: string;
  warnings?: string[];
};

export type AdminTournamentLiveStatusResponse = {
  enabled: boolean;
  status: "staging_write_ready" | "read_only_fallback" | string;
  authority: "python_fastapi";
  product_boundary: "draw_scoped_tournament_runner_not_jupr_live";
  club_id: string;
  environment: string;
  staging_only: true;
  writes_enabled: boolean;
  service_role_ready: boolean;
  operation_store_ready: boolean;
  audit_store_ready: boolean;
  write_flag: { name: string; enabled: boolean };
  snapshot_endpoint?: string;
  command_endpoint?: string;
  reconcile_endpoint?: string;
  streamlit_fallback_url: string;
  warnings: string[];
};

export type AdminTournamentLiveReadiness = {
  ready: boolean;
  confirmation: string;
  blockers: string[];
};

export type AdminTournamentLiveOperation = {
  operation_key: string;
  request_fingerprint: string;
  client_idempotency_key: string;
  action: string;
  command?: string | null;
  status: "intent" | "mutated" | "completed" | "failed" | "recovery_required" | string;
  expected_state: string;
  attempt_count: number;
  error_text?: string | null;
  result_mode?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  completion_audited_at?: string | null;
  audit_evidence: {
    actions: string[];
    intent_present: boolean;
    completion_present: boolean;
    failure_present: boolean;
    latest_at?: string | null;
  };
};

export type AdminTournamentLiveSnapshotResponse = AdminTournamentOpsSnapshotResponse & {
  mode: "tournament_live_draw_selector" | "tournament_live_draw_snapshot";
  scope: "tournament_selector" | "draw";
  authority: "python_fastapi";
  product_boundary: "draw_scoped_tournament_runner_not_jupr_live";
  state_fingerprint?: string | null;
  runtime: AdminTournamentLiveStatusResponse;
  progression?: {
    phase: string;
    open_games: number;
    completed_games: number;
    published_games: number;
    expected_awards: number;
    verified_awards: number;
  };
  publication_evidence?: {
    available: boolean;
    published_game_ids: string[];
    match_count: number;
    duplicate_game_ids: string[];
    complete: boolean;
  };
  award_evidence?: {
    available: boolean;
    expected: Array<{ player_id: number; badge_id: string; context_id: string }>;
    verified_count: number;
    active_row_count: number;
    unexpected_count: number;
  };
  readiness: Record<string, AdminTournamentLiveReadiness>;
  active_operation?: AdminTournamentLiveOperation | null;
  operations: AdminTournamentLiveOperation[];
};

export type AdminTournamentWriteResponse = {
  ok: boolean;
  mode?: string;
  action?: string;
  tournament?: AdminTournament | null;
  tournament_id?: string;
  usage_summary?: Record<string, number>;
  draw?: AdminTournamentDraw | null;
  registration?: AdminTournamentRegistration | null;
  registrations?: AdminTournamentRegistration[];
  selection?: AdminTournamentSelection | null;
  teams?: AdminTournamentOpsTeam[];
  games?: Array<Record<string, unknown>>;
  game?: Record<string, unknown> | null;
  game_count?: number;
  match_count?: number;
  singles_match_count?: number;
  doubles_match_count?: number;
  tournament_game_ids?: string[];
  playoff_winner_bonus_elo?: number;
  bonus_match_count?: number;
  bonus_tournament_game_ids?: string[];
  process_result?: Record<string, unknown>;
  advance_count?: number;
  standings?: Array<Record<string, unknown>>;
  podium?: Array<Record<string, unknown>>;
  podium_source?: string;
  candidate_count?: number;
  awarded_count?: number;
  import_mode?: string;
  updated_count?: number;
  team_count?: number;
  podium_count?: number;
  created_player_count?: number;
  review_fingerprint?: string;
  auto_player_updates?: Record<string, unknown>;
  registration_ids?: string[];
  skipped?: string[];
  operation_key?: string;
  request_fingerprint?: string;
  client_idempotency_key?: string;
  idempotent_replay?: boolean;
  reconciled?: boolean;
  recovery_disposition?: string;
  recovery_evidence?: Record<string, unknown>;
  warnings?: string[];
};

export type AdminTournamentResultsImportPreviewResponse = {
  ok: boolean;
  mode: "tournament_results_import_preview";
  dry_run: true;
  write_count: 0;
  tournament_id: string;
  draw_id: string;
  import_mode: string;
  review_fingerprint: string;
  players: Array<Record<string, unknown>>;
  player_options: Array<{ id: string | number; name: string; dupr_id?: string | null }>;
  suggestions: Record<string, Record<string, unknown>>;
  mapping_decisions: Record<string, { action?: string; player_id?: string | number | null }>;
  matches: Array<Record<string, unknown>>;
  match_reviews: Record<string, { include?: boolean; stage?: string }>;
  teams: Array<Record<string, unknown>>;
  podium_candidates: string[];
  podium_refs: Record<string, string | null>;
  summary: { imported_players: number; teams: number; matches: number; create_players: number };
  errors: string[];
  warnings: string[];
};

export type AdminTournamentBroadcastRecipient = {
  name: string;
  email: string;
  registration_status: string;
  payment_status: string;
};

export type AdminTournamentBroadcastPreviewResponse = {
  ok: boolean;
  mode: "tournament_broadcast_preview";
  dry_run: true;
  send_available: false;
  recipient_count: number;
  recipients: AdminTournamentBroadcastRecipient[];
  recipient_csv: string;
  preview: {
    to_name: string;
    to_email: string;
    subject: string;
    text: string;
    html: string;
  };
  warnings?: string[];
};

type ApiResult<T> = { data: T | null; error: string | null };

function baseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

export function getAdminTournamentApiBaseUrl(): string | null {
  return baseUrl();
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

async function fetchJson<T>(path: string): Promise<ApiResult<T>> {
  const apiBase = baseUrl();
  if (!apiBase) return { data: null, error: "Missing JUPR API base URL environment variable." };
  const url = `${apiBase.replace(/\/$/, "")}${path}`;
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) return { data: null, error: await apiErrorMessage(response) };
    return { data: (await response.json()) as T, error: null };
  } catch (error) {
    return { data: null, error: `Unable to reach API: ${error instanceof Error ? error.message : "Unknown error"}` };
  }
}

export async function getAdminTournamentStatus(clubId = "tres_palapas"): Promise<ApiResult<AdminTournamentStatusResponse>> {
  return fetchJson<AdminTournamentStatusResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/status`);
}

export async function getAdminTournamentLiveStatus(clubId = "tres_palapas"): Promise<ApiResult<AdminTournamentLiveStatusResponse>> {
  return fetchJson<AdminTournamentLiveStatusResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournament-live/status`);
}
