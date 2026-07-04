export type PublicTournamentSummary = {
  id: string;
  name: string;
  status?: string | null;
  start_date?: string | null;
  end_date?: string | null;
  event_tags?: Record<string, unknown> | null;
};

export type PublicTournamentSettings = {
  registration_slug?: string | null;
  registration_status?: string | null;
  registration_open_at?: string | null;
  registration_close_at?: string | null;
  waitlist_enabled?: boolean | null;
  partner_board_enabled?: boolean | null;
  rules_markdown?: string | null;
  refund_policy_markdown?: string | null;
  sponsor_markdown?: string | null;
};

export type PublicTournamentChoice = {
  tournament: PublicTournamentSummary;
  settings: PublicTournamentSettings;
};

export type PublicRegistrationDay = {
  id: string;
  label: string;
  event_date?: string | null;
  sort_order?: number | null;
  enabled?: boolean | null;
};

export type PublicRegistrationEvent = {
  id: string;
  registration_day_id: string;
  label: string;
  event_family_label: string;
  division_name: string;
  event_type?: string | null;
  gender_restriction?: string | null;
  skill_label?: string | null;
  age_label?: string | null;
  skill_mode?: string | null;
  age_mode?: string | null;
  event_format?: string | null;
  scoring?: string | null;
  capacity_teams?: number | null;
  price_usd?: number | null;
  partner_required?: boolean | null;
  partner_board_enabled?: boolean | null;
  waitlist_enabled?: boolean | null;
  status?: string | null;
  visibility?: string | null;
  selectable?: boolean | null;
};

export type TournamentRegistrationResponse = {
  club: { id: string; slug: string; name: string };
  available: boolean;
  setup_error?: string | null;
  tournaments: PublicTournamentChoice[];
  tournament?: PublicTournamentSummary | null;
  settings?: PublicTournamentSettings | null;
  registration_open: boolean;
  registration_closed_reason?: string | null;
  days: PublicRegistrationDay[];
  events: PublicRegistrationEvent[];
  roster_summary?: {
    total_registrations?: number | null;
    total_players?: number | null;
    players_needing_partners?: number | null;
    waitlist?: number | null;
  } | null;
};

export type PublicTournamentRosterMember = {
  registration_id?: string | null;
  selection_id?: string | null;
  player_id?: string | number | null;
  display_name: string;
  skill?: string | number | null;
  age?: number | null;
  age_bracket?: string | null;
  dupr_id?: string | null;
};

export type PublicTournamentRosterEntry = {
  event_day_id?: string | null;
  event_day_label?: string | null;
  event_family: string;
  division: string;
  event_label: string;
  status?: string | null;
  entry_type?: string | null;
  partner_request_id?: string | null;
  partner_link_id?: string | null;
  source_registration_ids?: string[];
  source_selection_ids?: string[];
  source_player_ids?: Array<string | number>;
  members: PublicTournamentRosterMember[];
};

export type PublicTournamentNeedsPartnerEntry = {
  player_name?: string | null;
  selection_id?: string | null;
  registration_id?: string | null;
  player_id?: string | number | null;
  event_option_id?: string | null;
  event_day_label?: string | null;
  event_family?: string | null;
  division?: string | null;
  event_label?: string | null;
  skill?: string | number | null;
  age?: number | null;
  age_bracket?: string | null;
  note?: string | null;
};

export type PublicTournamentRosterState = {
  registrations_by_event: PublicTournamentRosterEntry[];
  confirmed_teams: PublicTournamentRosterEntry[];
  pending_partner_requests: PublicTournamentRosterEntry[];
  unresolved_partner_entries: PublicTournamentRosterEntry[];
  players_needing_partners: PublicTournamentNeedsPartnerEntry[];
  summary: {
    total_registrations?: number | null;
    total_players?: number | null;
    players_needing_partners?: number | null;
    waitlist?: number | null;
  };
};

export type TournamentRosterResponse = {
  club: { id: string; slug: string; name: string };
  available: boolean;
  setup_error?: string | null;
  tournaments: PublicTournamentChoice[];
  tournament?: PublicTournamentSummary | null;
  settings?: PublicTournamentSettings | null;
  days: PublicRegistrationDay[];
  events: PublicRegistrationEvent[];
  roster?: PublicTournamentRosterState | null;
  summary?: PublicTournamentRosterState["summary"] | null;
  empty_reason?: string | null;
};

export type PublicRegistrationSelectionPayload = {
  event_option_id: string;
  registration_day_id?: string | null;
  partner_mode: "NONE" | "HAS_PARTNER" | "NEEDS_PARTNER";
  partner_name?: string | null;
  partner_email?: string | null;
  partner_phone?: string | null;
  partner_dupr_id?: string | null;
  partner_skill?: number | null;
  partner_age?: number | null;
  partner_note?: string | null;
  show_on_partner_board?: boolean | null;
};

export type PublicRegistrationSubmitPayload = {
  tournament_id?: string | null;
  registration_slug?: string | null;
  first_name?: string | null;
  last_name?: string | null;
  display_name?: string | null;
  email: string;
  phone?: string | null;
  player_id?: string | number | null;
  dupr_id?: string | null;
  doubles_skill?: number | null;
  singles_skill?: number | null;
  age?: number | null;
  gender?: string | null;
  notes?: string | null;
  wants_partner_board_contact?: boolean | null;
  terms_accepted: boolean;
  website?: string | null;
  selections: PublicRegistrationSelectionPayload[];
};

export type PublicRegistrationEditRegistration = {
  id: string;
  first_name?: string | null;
  last_name?: string | null;
  display_name: string;
  email: string;
  phone?: string | null;
  dupr_id?: string | null;
  doubles_skill?: number | null;
  singles_skill?: number | null;
  age?: number | null;
  gender?: string | null;
  notes?: string | null;
  wants_partner_board_contact?: boolean | null;
  status?: string | null;
  payment_status?: string | null;
  submitted_at?: string | null;
};

export type PublicRegistrationEditSelection = PublicRegistrationSelectionPayload & { id: string };

export type PublicRegistrationEditResponse = TournamentRegistrationResponse & {
  edit_mode: boolean;
  edit_token_valid: boolean;
  edit_token_expires_at?: string | null;
  registration: PublicRegistrationEditRegistration;
  selections: PublicRegistrationEditSelection[];
  total_price_usd: number;
};

export type PublicRegistrationEditSubmitPayload = Omit<PublicRegistrationSubmitPayload, "email"> & {
  edit_token: string;
  email?: string | null;
};

export type PublicRegistrationSubmitResponse = {
  club: { id: string; slug: string; name: string };
  ok: boolean;
  tournament: PublicTournamentSummary;
  settings?: PublicTournamentSettings | null;
  registration_id: string;
  submitted_at?: string | null;
  selection_count?: number | null;
};

export type PublicRegistrationConfirmationResponse = {
  club: { id: string; slug: string; name: string };
  tournament: PublicTournamentSummary;
  settings?: PublicTournamentSettings | null;
  registration: {
    id: string;
    display_name: string;
    email: string;
    status?: string | null;
    payment_status?: string | null;
    submitted_at?: string | null;
  };
  selections: Array<{
    selection_id: string;
    event_label: string;
    event_family_label: string;
    day_label: string;
    partner_mode?: string | null;
    partner_name?: string | null;
    show_on_partner_board?: boolean | null;
  }>;
  total_price_usd: number;
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

async function fetchJson<T>(path: string, init?: RequestInit): Promise<ApiResult<T>> {
  const apiBase = baseUrl();
  if (!apiBase) return { data: null, error: "Missing JUPR API base URL environment variable." };
  const url = `${apiBase.replace(/\/$/, "")}${path}`;
  try {
    const response = await fetch(url, init ?? { next: { revalidate: 60 } });
    if (!response.ok) return { data: null, error: await apiErrorMessage(response) };
    return { data: (await response.json()) as T, error: null };
  } catch (error) {
    return { data: null, error: `Unable to reach API: ${error instanceof Error ? error.message : "Unknown error"}` };
  }
}

export async function getClubTournamentRegistration(
  clubSlug: string,
  params?: { tournamentId?: string | null; registrationSlug?: string | null }
): Promise<ApiResult<TournamentRegistrationResponse>> {
  const query = new URLSearchParams();
  if (params?.tournamentId) query.set("tournament_id", params.tournamentId);
  if (params?.registrationSlug) query.set("registration_slug", params.registrationSlug);
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return fetchJson<TournamentRegistrationResponse>(`/clubs/${clubSlug}/tournament-registration${suffix}`);
}

export async function getClubTournamentRegistrationEdit(
  clubSlug: string,
  params: { editToken: string; tournamentId?: string | null; registrationSlug?: string | null }
): Promise<ApiResult<PublicRegistrationEditResponse>> {
  const query = new URLSearchParams({ edit_token: params.editToken });
  if (params?.tournamentId) query.set("tournament_id", params.tournamentId);
  if (params?.registrationSlug) query.set("registration_slug", params.registrationSlug);
  return fetchJson<PublicRegistrationEditResponse>(`/clubs/${clubSlug}/tournament-registration/edit?${query.toString()}`);
}

export async function getClubTournamentRoster(
  clubSlug: string,
  params?: { tournamentId?: string | null; registrationSlug?: string | null }
): Promise<ApiResult<TournamentRosterResponse>> {
  const query = new URLSearchParams();
  if (params?.tournamentId) query.set("tournament_id", params.tournamentId);
  if (params?.registrationSlug) query.set("registration_slug", params.registrationSlug);
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return fetchJson<TournamentRosterResponse>(`/clubs/${clubSlug}/tournament-roster${suffix}`);
}

export async function submitClubTournamentRegistration(
  clubSlug: string,
  payload: PublicRegistrationSubmitPayload
): Promise<ApiResult<PublicRegistrationSubmitResponse>> {
  return fetchJson<PublicRegistrationSubmitResponse>(`/clubs/${clubSlug}/tournament-registration`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export async function submitClubTournamentRegistrationEdit(
  clubSlug: string,
  payload: PublicRegistrationEditSubmitPayload
): Promise<ApiResult<PublicRegistrationSubmitResponse>> {
  return fetchJson<PublicRegistrationSubmitResponse>(`/clubs/${clubSlug}/tournament-registration/edit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export async function getClubTournamentRegistrationConfirmation(
  clubSlug: string,
  registrationId: string,
  params?: { tournamentId?: string | null; registrationSlug?: string | null }
): Promise<ApiResult<PublicRegistrationConfirmationResponse>> {
  const query = new URLSearchParams();
  if (params?.tournamentId) query.set("tournament_id", params.tournamentId);
  if (params?.registrationSlug) query.set("registration_slug", params.registrationSlug);
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return fetchJson<PublicRegistrationConfirmationResponse>(
    `/clubs/${clubSlug}/tournament-registration/confirmations/${encodeURIComponent(registrationId)}${suffix}`
  );
}
