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
  event_format?: string | null;
  scoring?: string | null;
  capacity_teams?: number | null;
  price_usd?: number | null;
  partner_required?: boolean | null;
  partner_board_enabled?: boolean | null;
  waitlist_enabled?: boolean | null;
  eligibility_mode?: string | null;
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

export type TournamentCommerceExtraOption = {
  id: string;
  label: string;
  sku?: string | null;
  price_delta_cents?: number | null;
  inventory_total?: number | null;
  inventory_committed?: number | null;
  inventory_reserved?: number | null;
  inventory_available?: number | null;
  status?: string | null;
};

export type TournamentCommerceExtra = {
  id: string;
  name: string;
  description?: string | null;
  category?: string | null;
  base_price_cents?: number | null;
  inventory_total?: number | null;
  inventory_committed?: number | null;
  inventory_reserved?: number | null;
  inventory_available?: number | null;
  max_per_registration?: number | null;
  requires_fulfillment?: boolean | null;
  fulfillment_instructions?: string | null;
  status?: string | null;
  available_from?: string | null;
  available_through?: string | null;
  options: TournamentCommerceExtraOption[];
};

export type TournamentCommerceBundleComponent = {
  event_option_id?: string | null;
  extra_option_id?: string | null;
  quantity: number;
  label?: string | null;
};

export type TournamentCommerceBundle = {
  id: string;
  name: string;
  description?: string | null;
  price_cents?: number | null;
  max_per_registration?: number | null;
  status?: string | null;
  available_from?: string | null;
  available_through?: string | null;
  components: TournamentCommerceBundleComponent[];
};

export type TournamentCommerceGiveaway = {
  id: string;
  extra_option_id: string;
  trigger_type: string;
  first_n?: number | null;
  available_from?: string | null;
  available_through?: string | null;
  status?: string | null;
};

export type TournamentCommerceCatalog = {
  catalog_fingerprint: string;
  extras: TournamentCommerceExtra[];
  bundles: TournamentCommerceBundle[];
  giveaways: TournamentCommerceGiveaway[];
};

export type TournamentCommerceSelection = {
  item_type: "extra" | "bundle";
  item_id: string;
  option_id?: string | null;
  quantity: number;
};

export type TournamentCommerceQuoteLine = {
  line_key: string;
  item_type: string;
  item_id: string;
  option_id?: string | null;
  label: string;
  quantity: number;
  unit_price_cents: number;
  line_total_cents: number;
  giveaway_applied?: boolean | null;
};

export type TournamentCommerceQuote = {
  tournament_id: string;
  currency: string;
  subtotal_cents: number;
  discount_cents: number;
  total_cents: number;
  lines: TournamentCommerceQuoteLine[];
  expected_quote_fingerprint: string;
};

export type TournamentCommerceOrderLine = {
  id: string;
  line_key: string;
  item_type: string;
  item_id: string;
  option_id?: string | null;
  label: string;
  quantity: number;
  unit_price_cents: number;
  line_total_cents: number;
  fulfillment_status?: string | null;
};

export type TournamentCommerceOrder = {
  id: string;
  tournament_id: string;
  registration_id: string;
  payment_status?: string | null;
  fulfillment_status?: string | null;
  total_cents?: number | null;
  currency?: string | null;
  updated_at?: string | null;
  lines: TournamentCommerceOrderLine[];
};

function apiBase(): string | null {
  return (
    process.env.JUPR_API_BASE_URL ||
    process.env.NEXT_PUBLIC_JUPR_API_BASE_URL ||
    null
  );
}

function apiUrl(base: string, path: string): string {
  return `${base.replace(/\/$/, "")}${path}`;
}

async function getJson<T>(path: string): Promise<{ data: T | null; error: string | null }> {
  const base = apiBase();
  if (!base) return { data: null, error: "Missing JUPR API base URL." };
  try {
    const response = await fetch(apiUrl(base, path), { cache: "no-store" });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      return {
        data: null,
        error: String(payload?.detail || `API error (${response.status}).`)
      };
    }
    return { data: payload as T, error: null };
  } catch (error) {
    return {
      data: null,
      error: error instanceof Error ? error.message : "Unable to reach API."
    };
  }
}

export async function getClubTournamentRegistration(
  clubSlug: string,
  options: { registrationSlug?: string | null; tournamentId?: string | null } = {}
): Promise<{ data: TournamentRegistrationResponse | null; error: string | null }> {
  const query = new URLSearchParams();
  if (options.registrationSlug) query.set("tournament", options.registrationSlug);
  if (options.tournamentId) query.set("tournament_id", options.tournamentId);
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return getJson<TournamentRegistrationResponse>(
    `/clubs/${encodeURIComponent(clubSlug)}/tournament-registration${suffix}`
  );
}

export async function getClubTournamentRoster(
  clubSlug: string,
  options: { registrationSlug?: string | null; tournamentId?: string | null } = {}
): Promise<{ data: TournamentRosterResponse | null; error: string | null }> {
  const query = new URLSearchParams();
  if (options.registrationSlug) query.set("tournament", options.registrationSlug);
  if (options.tournamentId) query.set("tournament_id", options.tournamentId);
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return getJson<TournamentRosterResponse>(
    `/clubs/${encodeURIComponent(clubSlug)}/tournament-roster${suffix}`
  );
}

export async function resolveTournamentRegistrationProfile(
  clubSlug: string,
  payload: PublicRegistrationProfileResolutionPayload
): Promise<PublicRegistrationProfileResolutionResponse> {
  const base = apiBase();
  if (!base) throw new Error("Missing JUPR API base URL.");
  const response = await fetch(
    apiUrl(
      base,
      `/clubs/${encodeURIComponent(clubSlug)}/tournament-registration/resolve-profile`
    ),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }
  );
  const result = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(String(result?.detail || `API error (${response.status})`));
  }
  return result as PublicRegistrationProfileResolutionResponse;
}

export async function quoteTournamentRegistration(
  clubSlug: string,
  payload: {
    tournament_id?: string | null;
    registration_slug?: string | null;
    event_option_ids: string[];
    commerce: { item_selections: TournamentCommerceSelection[] };
  }
): Promise<TournamentCommerceQuote> {
  const base = apiBase();
  if (!base) throw new Error("Missing JUPR API base URL.");
  const response = await fetch(
    apiUrl(
      base,
      `/clubs/${encodeURIComponent(clubSlug)}/tournament-registration/quote`
    ),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }
  );
  const result = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(String(result?.detail || `API error (${response.status})`));
  }
  return result as TournamentCommerceQuote;
}

export async function submitTournamentRegistration(
  clubSlug: string,
  payload: PublicRegistrationSubmitPayload
): Promise<Record<string, unknown>> {
  const base = apiBase();
  if (!base) throw new Error("Missing JUPR API base URL.");
  const response = await fetch(
    apiUrl(
      base,
      `/clubs/${encodeURIComponent(clubSlug)}/tournament-registration`
    ),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }
  );
  const result = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(String(result?.detail || `API error (${response.status})`));
  }
  return result as Record<string, unknown>;
}

export async function fetchTournamentRegistrationEdit(
  clubSlug: string,
  token: string
): Promise<PublicRegistrationEditResponse> {
  const base = apiBase();
  if (!base) throw new Error("Missing JUPR API base URL.");
  const response = await fetch(
    apiUrl(
      base,
      `/clubs/${encodeURIComponent(clubSlug)}/tournament-registration/edit?token=${encodeURIComponent(token)}`
    ),
    { cache: "no-store" }
  );
  const result = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(String(result?.detail || `API error (${response.status})`));
  }
  return result as PublicRegistrationEditResponse;
}

export async function updateTournamentRegistration(
  clubSlug: string,
  payload: {
    token: string;
    expected_registration_updated_at: string;
    registration: Record<string, unknown>;
    selections: PublicRegistrationSelectionPayload[];
    commerce?: {
      item_selections: TournamentCommerceSelection[];
      expected_quote_fingerprint: string;
      idempotency_key: string;
      expected_order_updated_at?: string | null;
    } | null;
  }
): Promise<Record<string, unknown>> {
  const base = apiBase();
  if (!base) throw new Error("Missing JUPR API base URL.");
  const response = await fetch(
    apiUrl(
      base,
      `/clubs/${encodeURIComponent(clubSlug)}/tournament-registration/edit`
    ),
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }
  );
  const result = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(String(result?.detail || `API error (${response.status})`));
  }
  return result as Record<string, unknown>;
}

export async function requestTournamentRegistrationEditLink(
  clubSlug: string,
  payload: {
    tournament_id?: string | null;
    registration_slug?: string | null;
    email: string;
    website?: string | null;
  }
): Promise<{ ok: boolean; message: string }> {
  const base = apiBase();
  if (!base) throw new Error("Missing JUPR API base URL.");
  const response = await fetch(
    apiUrl(
      base,
      `/clubs/${encodeURIComponent(clubSlug)}/tournament-registration/edit-link`
    ),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }
  );
  const result = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(String(result?.detail || `API error (${response.status})`));
  }
  return result as { ok: boolean; message: string };
}
