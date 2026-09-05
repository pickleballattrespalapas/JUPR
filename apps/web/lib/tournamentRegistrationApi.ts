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
  player_entry_key?: string | null;
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
  const fallback = publicRegistrationErrorMessage(response.status);
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
    const detailMessage =
      typeof detail === "string"
        ? detail
        : detail &&
            typeof detail === "object" &&
            !Array.isArray(detail) &&
            typeof (detail as Record<string, unknown>).message === "string"
          ? String((detail as Record<string, unknown>).message)
          : "";
    if (detail && typeof detail === "object") {
      const record = detail as Record<string, unknown>;
      return {
        message: publicRegistrationErrorMessage(
          response.status,
          detailMessage
        ),
        currentQuote:
          record.current_quote &&
          typeof record.current_quote === "object"
            ? (record.current_quote as TournamentCommerceQuote)
            : null
      };
    }
    return {
      message: publicRegistrationErrorMessage(response.status, detailMessage),
      currentQuote: null
    };
  } catch {
    // The response body is intentionally not exposed to the public page.
  }
  return { message: fallback, currentQuote: null };
}

function safeRegistrationValidation(detail: string): string | null {
  const message = detail.trim();
  if (!message || message.length > 400 || /[<>\u0000-\u001f]/.test(message)) return null;
  if (/\b(?:database|supabase|credential|runtime|stack|traceback|exception|rpc|sql|uuid|fingerprint|idempotency|atomic)\b|JUPR_|status=|enabled=|\bexpected_\w+\b|\b[a-f\d]{8}-(?:[a-f\d]{4}-){3}[a-f\d]{12}\b/i.test(message)) {
    return null;
  }
  if (!/\b(?:registration|division|event|partner|email|name|age|gender|rating|skill|profile|policy|policies|select|choose|required|open|closed|available|limit|number)\b/i.test(message)) {
    return null;
  }
  return message;
}

function publicRegistrationErrorMessage(
  status?: number | null,
  detail = ""
): string {
  const normalized = detail.trim().toLowerCase();
  if (/already registered|registration already exists|duplicate registration/.test(normalized)) {
    return "You’re already registered with this email. Request an edit link to make changes.";
  }
  if (normalized.includes("edit link")) {
    if (normalized.includes("expired")) {
      return "This edit link has expired. Request a new one to continue.";
    }
    if (/invalid|different (?:club|tournament|registration|email)/.test(normalized)) {
      return "This edit link isn’t valid for this registration. Request a new one to continue.";
    }
  }
  if (/imported into a draw|already (?:part of|in) (?:a |the )?draw/.test(normalized)) {
    return "This registration is already in the tournament draw. Contact the organizer to make changes.";
  }
  if (/active partner relationship|relationship.*locked/.test(normalized)) {
    return "This event has an active partner connection. Contact the organizer before changing it.";
  }
  if (/changed after it was loaded|changed since|stale|event entry changed/.test(normalized)) {
    return "This registration changed since you opened it. Refresh the page and try again.";
  }
  if (/pricing changed|quote changed|total changed|extras.*changed/.test(normalized)) {
    return "The total changed. Review the updated price and try again.";
  }
  if (/selected division.*no longer available|division.*not open for public registration|selected event.*no longer open|division.*enabled registration day/.test(normalized)) {
    return "A division you selected is no longer open. Review your events and try again.";
  }
  if (/registration (?:is )?not configured/.test(normalized)) {
    return "Registration isn’t available for this tournament yet.";
  }
  if (/each event selection must be an object|registration selection.*invalid|selected event.*doesn’t match/.test(normalized)) {
    return "Review the events you selected and try again.";
  }
  if (/same (?:registration )?selection.*more than once|same division.*more than once/.test(normalized)) {
    return "Choose each division only once.";
  }
  if (normalized.includes("invalid partner status")) {
    return "Choose whether you already have a partner or need one.";
  }
  if (/linked jupr player profile.*cannot be changed/.test(normalized)) {
    return "The player profile linked to this registration can’t be changed from an edit link. Contact the organizer.";
  }
  if (normalized.includes("contact consent is required")) {
    return "Please agree to share your contact information before joining Players Needing Partners.";
  }
  if (/finite whole number.*between 1 and 120/.test(normalized)) {
    return "Enter an age from 1 to 120.";
  }
  if (/finite number.*between 1 and 7/.test(normalized)) {
    return "Enter a rating from 1 to 7.";
  }
  if (normalized.includes("selected jupr player profile")) {
    return normalized.includes("not active")
      ? "The player profile you selected isn’t active in this club."
      : "The player profile you selected wasn’t found in this club.";
  }
  if (/not open|registration (?:is )?closed/.test(normalized)) {
    return safeRegistrationValidation(detail) || "Registration is closed for this tournament.";
  }
  if (status === 401) {
    return "This registration link is no longer valid. Request a new link and try again.";
  }
  if (status === 403) {
    return "This change isn’t available from this link. Contact the tournament organizer.";
  }
  if (status === 404) {
    return "We couldn’t find that tournament registration.";
  }
  if (status === 429) {
    return "Too many attempts were made. Wait a moment and try again.";
  }
  if (status === 400 || status === 422) {
    return (
      safeRegistrationValidation(detail) ||
      "Check your registration information and try again."
    );
  }
  if (status === 409) {
    return "This registration changed since you opened it. Refresh the page and try again.";
  }
  return "Tournament registration is temporarily unavailable. Please try again.";
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<ApiResult<T>> {
  const apiBase = baseUrl();
  if (!apiBase) {
    return {
      data: null,
      error: publicRegistrationErrorMessage(null),
      status: null
    };
  }
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
  } catch {
    return {
      data: null,
      error: publicRegistrationErrorMessage(null),
      status: null
    };
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
