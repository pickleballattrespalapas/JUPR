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
  weather_policy_markdown?: string | null;
  sponsor_markdown?: string | null;
  location_name?: string | null;
  venue_address?: string | null;
  venue_directions?: string | null;
  timezone?: string | null;
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
  scheduled_day_ids?: string[] | null;
  label: string;
  event_family_label: string;
  division_name: string;
  event_type?: string | null;
  gender_restriction?: string | null;
  skill_label?: string | null;
  age_label?: string | null;
  skill_mode?: string | null;
  age_mode?: string | null;
  age_rules?: Record<string, unknown> | string | null;
  event_format?: string | null;
  scoring?: string | null;
  capacity_teams?: number | null;
  price_usd?: number | null;
  partner_required?: boolean | null;
  partner_board_enabled?: boolean | null;
  waitlist_enabled?: boolean | null;
  eligibility_mode?: string | null;
  skill_min_rating?: number | null;
  skill_max_rating?: number | null;
  combined_rating_cap?: number | null;
  rating_source_policy?: string | null;
  rating_review_timing?: string | null;
  competition_format?: string | null;
  team_roster_size?: number | null;
  team_gender_rule?: string | null;
  team_tiebreak_mode?: string | null;
  team_playoff_format?: string | null;
  team_allow_substitutes?: boolean | null;
  status?: string | null;
  visibility?: string | null;
  selectable?: boolean | null;
};

export type PublicRegistrationPlayer = {
  id: string;
  display_name: string;
  dupr_id?: string | null;
  doubles_skill?: number | null;
  singles_skill?: number | null;
};

export type PublicRegistrationProfileResolutionPayload = {
  tournament_id?: string | null;
  registration_slug?: string | null;
  first_name: string;
  last_name: string;
  email: string;
  age: number;
  gender: string;
  website?: string | null;
};

export type PublicRegistrationProfileResolutionResponse = {
  club: { id: string; slug: string; name: string };
  ok: boolean;
  status: "ready" | "existing_registration" | "closed";
  can_start_new: boolean;
  registration_open: boolean;
  registration_closed_reason?: string | null;
  masked_email: string;
  profile_match_kind: "email_exact" | "name_exact" | "none";
  profile_candidates: PublicRegistrationPlayer[];
  profile_policy: {
    linkage: "staff_review_required";
    public_submission_links_player: false;
  };
  message: string;
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
  players: PublicRegistrationPlayer[];
  roster_summary?: {
    total_registrations?: number | null;
    total_players?: number | null;
    players_needing_partners?: number | null;
    waitlist?: number | null;
  } | null;
  commerce?: TournamentCommerceCatalog | null;
};

export type PublicTournamentRosterMember = {
  display_name: string;
  skill?: string | number | null;
  age_bracket?: string | null;
};

export type PublicTournamentRosterEntry = {
  public_entry_key?: string | null;
  event_day_label?: string | null;
  event_family: string;
  division: string;
  event_label: string;
  status?: string | null;
  entry_type?: string | null;
  members: PublicTournamentRosterMember[];
};

export type PublicTournamentNeedsPartnerEntry = {
  player_name?: string | null;
  board_entry_key?: string | null;
  event_day_label?: string | null;
  event_family?: string | null;
  division?: string | null;
  event_label?: string | null;
  skill?: string | number | null;
  age_bracket?: string | null;
  note?: string | null;
};

export type PublicTournamentRosterState = {
  registrations_by_event: PublicTournamentRosterEntry[];
  confirmed_teams: PublicTournamentRosterEntry[];
  pending_partner_requests: PublicTournamentRosterEntry[];
  unresolved_partner_entries: PublicTournamentRosterEntry[];
  players_needing_partners: PublicTournamentNeedsPartnerEntry[];
  partner_board_entries: PublicTournamentNeedsPartnerEntry[];
  summary: {
    total_registrations?: number | null;
    total_players?: number | null;
    players_needing_partners?: number | null;
    partner_board_entries?: number | null;
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
  partner_gender?: string | null;
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
  commerce?: {
    item_selections: TournamentCommerceSelection[];
    expected_quote_fingerprint: string;
    idempotency_key: string;
    expected_order_updated_at?: string | null;
  } | null;
};

export type PublicRegistrationEditRegistration = {
  id: string;
  first_name?: string | null;
  last_name?: string | null;
  display_name: string;
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
  status?: string | null;
  payment_status?: string | null;
  submitted_at?: string | null;
  updated_at: string;
};

export type PublicRegistrationEditSelection = PublicRegistrationSelectionPayload & {
  id: string;
  updated_at: string;
};

export type PublicRegistrationEditResponse = TournamentRegistrationResponse & {
  edit_mode: boolean;
  edit_token_valid: boolean;
  edit_token_expires_at?: string | null;
  registration: PublicRegistrationEditRegistration;
  selections: PublicRegistrationEditSelection[];
  total_price_usd: number;
  commerce_order?: TournamentCommerceOrder | null;
};

export type PublicRegistrationEditSubmitPayload = Omit<PublicRegistrationSubmitPayload, "email" | "selections"> & {
  edit_token: string;
  email?: string | null;
  expected_updated_at: string;
  expected_selection_versions: Array<{ id: string; updated_at: string }>;
  selections: Array<PublicRegistrationSelectionPayload & { id?: string | null }>;
};

export type PublicRegistrationEditLinkRequestPayload = {
  tournament_id?: string | null;
  registration_slug?: string | null;
  email: string;
  website?: string | null;
  idempotency_key: string;
};

export type PublicRegistrationEditLinkRequestResponse = {
  club: { id: string; slug: string; name: string };
  ok: boolean;
  mode?: string | null;
  accepted?: boolean | null;
  message?: string | null;
  email_status?: string | null;
  provider_message_id?: string | null;
};

export type PublicRegistrationSubmitResponse = {
  club: { id: string; slug: string; name: string };
  ok: boolean;
  tournament: PublicTournamentSummary;
  settings?: PublicTournamentSettings | null;
  registration_id: string;
  submitted_at?: string | null;
  selection_count?: number | null;
  commerce_order?: Record<string, unknown> | null;
  updated_at?: string | null;
  confirmation_delivery?: {
    status: "sent" | "staging_redirect" | "dry_run" | "failed" | "unknown";
    delivered: boolean;
  } | null;
  confirmation_available?: boolean | null;
  confirmation_token?: string | null;
  email_delivery?: {
    status?: "dry_run" | "staging_redirect" | "sent" | "failed" | string | null;
    message?: string | null;
  } | null;
};

export type PublicRegistrationConfirmationResponse = {
  club: { id: string; slug: string; name: string };
  tournament: PublicTournamentSummary;
  settings?: PublicTournamentSettings | null;
  registration: {
    display_name: string;
    status?: string | null;
    payment_status?: string | null;
    submitted_at?: string | null;
  };
  selections: Array<{
    event_label: string;
    event_family_label: string;
    day_label: string;
    event_date?: string | null;
    scheduled_days?: Array<{
      label: string;
      event_date?: string | null;
    }> | null;
    skill_label?: string | null;
    age_label?: string | null;
    price_usd?: number | null;
    partner_mode?: string | null;
    partner_name?: string | null;
    show_on_partner_board?: boolean | null;
  }>;
  total_price_usd: number;
  commerce_order?: TournamentCommerceOrder | null;
  payment_note: string;
  confirmation_expires_at?: string | null;
  notification_sender?: {
    from_name?: string | null;
    from_email?: string | null;
  } | null;
};

export type ApiResult<T> = {
  data: T | null;
  error: string | null;
  status?: number | null;
  current_quote?: TournamentCommerceQuote | null;
};

function baseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

async function apiErrorDetails(
  response: Response
): Promise<{ message: string; currentQuote: TournamentCommerceQuote | null }> {
  const fallback = `API error (${response.status}).`;
  let bodyText = "";
  try {
    bodyText = await response.text();
  } catch {
    return { message: fallback, currentQuote: null };
  }
  if (!bodyText) return { message: fallback, currentQuote: null };
  try {
    const payload = JSON.parse(bodyText) as {
      detail?: unknown;
      message?: unknown;
      error?: unknown;
    };
    const detail = payload.detail ?? payload.message ?? payload.error;
    if (Array.isArray(detail)) {
      return {
        message: `${fallback} ${detail
          .map((item) => JSON.stringify(item))
          .join("; ")}`,
        currentQuote: null
      };
    }
    if (detail && typeof detail === "object") {
      const record = detail as Record<string, unknown>;
      return {
        message: String(record.message || fallback),
        currentQuote:
          record.current_quote &&
          typeof record.current_quote === "object"
            ? (record.current_quote as TournamentCommerceQuote)
            : null
      };
    }
    if (detail) {
      return {
        message: `${fallback} ${String(detail)}`,
        currentQuote: null
      };
    }
  } catch {
    // Fall through to short text excerpt below.
  }
  return {
    message: `${fallback} ${bodyText.slice(0, 240)}`,
    currentQuote: null
  };
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<ApiResult<T>> {
  const apiBase = baseUrl();
  if (!apiBase) return { data: null, error: "Missing JUPR API base URL environment variable.", status: null };
  const url = `${apiBase.replace(/\/$/, "")}${path}`;
  try {
    const response = await fetch(url, init ?? { next: { revalidate: 60 } });
    if (!response.ok) {
      const details = await apiErrorDetails(response);
      return {
        data: null,
        error: details.message,
        status: response.status,
        current_quote: details.currentQuote
      };
    }
    return { data: (await response.json()) as T, error: null, status: response.status };
  } catch (error) {
    return { data: null, error: `Unable to reach API: ${error instanceof Error ? error.message : "Unknown error"}`, status: null };
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

export async function resolveClubTournamentRegistrationProfile(
  clubSlug: string,
  payload: PublicRegistrationProfileResolutionPayload
): Promise<ApiResult<PublicRegistrationProfileResolutionResponse>> {
  return fetchJson<PublicRegistrationProfileResolutionResponse>(
    `/clubs/${clubSlug}/tournament-registration/profile-resolution`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }
  );
}

export async function getClubTournamentRegistrationEdit(
  clubSlug: string,
  params: { editToken: string; tournamentId?: string | null; registrationSlug?: string | null }
): Promise<ApiResult<PublicRegistrationEditResponse>> {
  const query = new URLSearchParams({ edit_token: params.editToken });
  if (params?.tournamentId) query.set("tournament_id", params.tournamentId);
  if (params?.registrationSlug) query.set("registration_slug", params.registrationSlug);
  return fetchJson<PublicRegistrationEditResponse>(`/clubs/${clubSlug}/tournament-registration/edit?${query.toString()}`, {
    cache: "no-store"
  });
}

export async function requestClubTournamentRegistrationEditLink(
  clubSlug: string,
  payload: PublicRegistrationEditLinkRequestPayload
): Promise<ApiResult<PublicRegistrationEditLinkRequestResponse>> {
  return fetchJson<PublicRegistrationEditLinkRequestResponse>(`/clubs/${clubSlug}/tournament-registration/edit-link/request`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
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
  confirmationToken: string
): Promise<ApiResult<PublicRegistrationConfirmationResponse>> {
  const query = new URLSearchParams({ confirmation_token: confirmationToken });
  return fetchJson<PublicRegistrationConfirmationResponse>(
    `/clubs/${clubSlug}/tournament-registration/confirmation?${query.toString()}`
  );
}
import type {
  TournamentCommerceCatalog,
  TournamentCommerceOrder,
  TournamentCommerceQuote,
  TournamentCommerceSelection
} from "@/lib/tournamentCommerceApi";
