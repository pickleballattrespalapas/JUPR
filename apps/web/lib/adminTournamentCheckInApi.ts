export type TournamentCheckInBlocker = {
  code: string;
  status: "BLOCKED" | "NEEDS_REVIEW" | string;
  title: string;
  detail: string;
};

export type TournamentCheckInEvent = {
  selection_id: string;
  event_option_id: string;
  event_label: string;
  team_state: "CONFIRMED_LINK" | "UNRESOLVED" | "NOT_REQUIRED" | string;
  partner_name?: string | null;
  entered_partner_name?: string | null;
  blockers: TournamentCheckInBlocker[];
};

export type TournamentCheckInRegistrant = {
  registration_id: string;
  registration_day_id: string;
  registration_status: string;
  attendance_status: "EXPECTED" | "CHECKED_IN" | "ABSENT" | string;
  registration_updated_at?: string | null;
  original_registrant: { player_id?: number | null; name: string };
  attendee: {
    player_id?: number | null;
    name: string;
    is_approved_substitute: boolean;
  };
  substitution: {
    allowed: boolean;
    event_policy_allows: boolean;
    blocker: TournamentCheckInBlocker & {
      code:
        | "SUBSTITUTE_POLICY_UNAVAILABLE"
        | "SUBSTITUTE_POLICY_NOT_ALLOWED"
        | "SUBSTITUTE_ASSIGNMENT_ATOMICITY_UNAVAILABLE"
        | string;
    };
  };
  check_in: {
    registration_day_id: string;
    attendance_status: "EXPECTED" | "CHECKED_IN" | "ABSENT" | string;
    checked_in: boolean;
    notes?: string | null;
    updated_at?: string | null;
    updated_by?: string | null;
    identity_current: boolean;
    requires_reconfirmation: boolean;
  };
  waiver: {
    verified: boolean;
    subject: "attending_player";
    subject_name: string;
  };
  payment: {
    status: string;
    source: "offline_payment_tracking" | "offline_registration_record" | string;
    ready: boolean;
  };
  events: TournamentCheckInEvent[];
  blockers: TournamentCheckInBlocker[];
};

export type TournamentCheckInSnapshot = {
  ok: true;
  mode: "tournament_registration_check_in";
  authority: "python_fastapi_supabase";
  tournament: {
    id: string;
    name: string;
    status: string;
    start_date?: string | null;
    end_date?: string | null;
  };
  day_scope: {
    selected_day_id: string;
    selected_day: TournamentCheckInDay;
    available_days: TournamentCheckInDay[];
  };
  summary: {
    expected: number;
    checked_in: number;
    absent: number;
    not_checked_in: number;
    unresolved: number;
  };
  registrants: TournamentCheckInRegistrant[];
  inactive_registrants: Array<{
    registration_id: string;
    name: string;
    registration_status: string;
  }>;
  unresolved_participants: Array<{
    kind: string;
    registration_id: string;
    registration_name: string;
    selection_id: string;
    event_label: string;
    entered_partner_name?: string | null;
    title: string;
    detail: string;
  }>;
  player_options: Array<{ id: number; name: string }>;
  readiness: {
    schedule: {
      status: string;
      timezone?: string | null;
      active_day_count: number;
      blockers: TournamentCheckInBlocker[];
      days: Array<{
        id: string;
        label: string;
        event_date?: string | null;
        court_count?: number | null;
        court_labels: string[];
        court_open_time?: string | null;
        court_close_time?: string | null;
      }>;
    };
    draws: {
      status: string;
      active_division_count: number;
      draw_count: number;
      blockers: TournamentCheckInBlocker[];
    };
    staffing: {
      status: "NEEDS_REVIEW" | string;
      source: string;
      blockers: TournamentCheckInBlocker[];
    };
  };
  completed_items: Array<{
    code: string;
    title: string;
    status: string;
    detail: string;
  }>;
  blockers: TournamentCheckInBlocker[];
  runtime?: Record<string, unknown>;
};

export type TournamentCheckInDay = {
  id: string;
  label: string;
  event_date?: string | null;
  sort_order?: number | null;
};

export type TournamentCheckInUpdate = {
  operation_key: string;
  expected_updated_at: string | null;
  attendance_status: "EXPECTED" | "CHECKED_IN" | "ABSENT";
  waiver_verified: boolean;
  approved_substitute_player_id: number | null;
  notes: string | null;
};

export type TournamentCheckInUpdateResponse = {
  ok: true;
  mode: "tournament_registration_check_in_update";
  check_in: {
    registration_id: string;
    registration_day_id: string;
    attendance_status: "EXPECTED" | "CHECKED_IN" | "ABSENT";
    checked_in: boolean;
    waiver_verified: boolean;
    approved_substitute_player_id?: number | null;
    approved_substitute_name?: string | null;
    notes?: string | null;
    updated_by?: string | null;
    updated_at: string;
  };
  attendee_identity_changed: boolean;
  attendance_reset: boolean;
  idempotent_replay: boolean;
  message: string;
};

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

async function requestJson<T>(
  url: string,
  accessToken: string,
  options: RequestInit = {}
): Promise<T> {
  const headers = new Headers(options.headers);
  headers.set("Authorization", `Bearer ${accessToken}`);
  if (options.body) headers.set("Content-Type", "application/json");
  const response = await fetch(url, { ...options, headers, cache: "no-store" });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(String(payload?.detail || `Tournament check-in API error (${response.status})`));
  }
  return payload as T;
}

export async function fetchAdminTournamentCheckIn(options: {
  apiBase: string;
  clubId: string;
  tournamentId: string;
  accessToken: string;
  dayId?: string;
  signal?: AbortSignal;
}): Promise<TournamentCheckInSnapshot> {
  const searchParams = new URLSearchParams();
  if (options.dayId) searchParams.set("day_id", options.dayId);
  const query = searchParams.toString();
  return requestJson<TournamentCheckInSnapshot>(
    apiUrl(
      options.apiBase,
      `/admin/clubs/${encodeURIComponent(options.clubId)}/tournament-live/tournaments/${encodeURIComponent(options.tournamentId)}/check-in${query ? `?${query}` : ""}`
    ),
    options.accessToken,
    { signal: options.signal }
  );
}

export async function updateAdminTournamentCheckIn(options: {
  apiBase: string;
  clubId: string;
  tournamentId: string;
  registrationId: string;
  dayId: string;
  accessToken: string;
  input: TournamentCheckInUpdate;
  signal?: AbortSignal;
}): Promise<TournamentCheckInUpdateResponse> {
  return requestJson<TournamentCheckInUpdateResponse>(
    apiUrl(
      options.apiBase,
      `/admin/clubs/${encodeURIComponent(options.clubId)}/tournament-live/tournaments/${encodeURIComponent(options.tournamentId)}/check-in/${encodeURIComponent(options.registrationId)}?day_id=${encodeURIComponent(options.dayId)}`
    ),
    options.accessToken,
    {
      method: "PUT",
      body: JSON.stringify(options.input),
      signal: options.signal
    }
  );
}
