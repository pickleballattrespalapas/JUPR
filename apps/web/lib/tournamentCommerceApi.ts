export type TournamentCommerceItemKind =
  | "MERCHANDISE"
  | "LODGING"
  | "MEAL"
  | "DRINK_PACK"
  | "OTHER";

export type TournamentCommerceCatalogStatus = "DRAFT" | "ACTIVE" | "ARCHIVED";

export type TournamentCommerceItem = {
  id: string;
  name: string;
  description: string;
  kind: TournamentCommerceItemKind;
  status: TournamentCommerceCatalogStatus;
  base_price_minor: number;
  inventory_limit?: number | null;
  remaining?: number | null;
  max_per_registration: number;
  available_from?: string | null;
  available_until?: string | null;
  requires_fulfillment: boolean;
  fulfillment_instructions?: string | null;
  sort_order: number;
};

export type TournamentCommerceVariant = {
  id: string;
  item_id: string;
  name: string;
  sku?: string | null;
  status: TournamentCommerceCatalogStatus;
  price_delta_minor: number;
  price_minor: number;
  inventory_limit?: number | null;
  remaining?: number | null;
  sort_order: number;
};

export type TournamentCommerceBundleComponent = {
  id?: string;
  bundle_id?: string;
  resource_key?: string;
  component_type: "EVENT_OPTION" | "EVENT_CHOICE" | "ITEM_VARIANT";
  event_option_id?: string | null;
  item_id?: string | null;
  variant_id?: string | null;
  label?: string;
  option_label?: string | null;
  sku?: string | null;
  quantity?: number;
  quantity_per_bundle?: number;
  unit_price_minor?: number;
  total_quantity?: number;
  requires_fulfillment?: boolean;
  fulfillment_instructions?: string | null;
};

export type TournamentCommerceBundle = {
  id: string;
  name: string;
  description: string;
  status: TournamentCommerceCatalogStatus;
  price_minor: number;
  regular_minor?: number;
  savings_minor?: number;
  max_per_registration: number;
  available_from?: string | null;
  available_until?: string | null;
  sort_order: number;
  components?: TournamentCommerceBundleComponent[];
};

export type TournamentCommercePromotion = {
  id: string;
  name: string;
  promotion_type:
    | "DATE_WINDOW_FREE"
    | "FIRST_N_REGISTRANTS"
    | "FIRST_N_CLAIMS";
  target_type: "ITEM" | "ITEM_VARIANT" | "BUNDLE";
  item_id?: string | null;
  variant_id?: string | null;
  bundle_id?: string | null;
  starts_at?: string | null;
  ends_at?: string | null;
  giveaway_limit?: number | null;
  per_registration_limit: number;
  priority?: number;
  status: TournamentCommerceCatalogStatus;
};

export type TournamentCommerceCatalog = {
  available: boolean;
  tournament_id: string;
  reason?: string | null;
  currency: "USD";
  offline_payment: true;
  catalog_revision?: number;
  catalog_fingerprint?: string;
  items: TournamentCommerceItem[];
  variants: TournamentCommerceVariant[];
  bundles: TournamentCommerceBundle[];
  bundle_components: TournamentCommerceBundleComponent[];
  promotions: TournamentCommercePromotion[];
  current_order?: {
    status?: string | null;
    payment_status?: string | null;
    updated_at?: string | null;
    quote_fingerprint?: string | null;
  } | null;
};

export type TournamentCommerceSelection = {
  variant_id: string;
  quantity: number;
};

export type TournamentCommerceQuoteLine = {
  line_key: string;
  line_type: "EVENT" | "ITEM" | "BUNDLE";
  event_option_id?: string | null;
  item_id?: string | null;
  variant_id?: string | null;
  bundle_id?: string | null;
  promotion_id?: string | null;
  label: string;
  option_label?: string | null;
  quantity: number;
  list_unit_minor: number;
  final_unit_minor: number;
  list_total_minor: number;
  final_total_minor: number;
  savings_minor: number;
  component_snapshot?: TournamentCommerceBundleComponent[];
};

export type TournamentCommerceQuote = {
  quote_version: number;
  review_binding_version: number;
  currency: "USD";
  quote_fingerprint: string;
  request_fingerprint: string;
  catalog_fingerprint: string;
  request: {
    event_option_ids: string[];
    item_selections: TournamentCommerceSelection[];
  };
  lines: TournamentCommerceQuoteLine[];
  applied_bundles: Array<Record<string, unknown>>;
  applied_promotions: Array<Record<string, unknown>>;
  list_subtotal_minor: number;
  discount_minor: number;
  total_minor: number;
  offline_payment: true;
  payment_status: "UNPAID";
};

export type TournamentCommerceOrder = {
  id?: string;
  status?: string;
  payment_status?: string;
  currency?: "USD";
  current_revision?: number;
  list_subtotal_minor?: number;
  discount_minor?: number;
  total_minor?: number;
  updated_at?: string | null;
  quote?: TournamentCommerceQuote | null;
  offline_payment?: boolean;
};

export type AdminTournamentCommerceDetail = {
  ok: boolean;
  tournament: {
    id: string;
    name: string;
    status?: string;
    start_date?: string | null;
    end_date?: string | null;
  };
  catalog: Omit<TournamentCommerceCatalog, "available" | "tournament_id" | "offline_payment"> & {
    currency: "USD";
    event_options: Array<Record<string, unknown>>;
  };
  orders: Array<Record<string, unknown>>;
  fulfillment: Array<Record<string, unknown>>;
  operations: Array<Record<string, unknown>>;
  audit: Array<Record<string, unknown>>;
  offline_payment_only: true;
};

type ApiResult<T> = {
  data: T | null;
  error: string | null;
  status?: number | null;
};

function baseUrl(): string | null {
  return (
    process.env.JUPR_API_BASE_URL ||
    process.env.NEXT_PUBLIC_JUPR_API_BASE_URL ||
    null
  );
}

async function adminErrorMessage(response: Response): Promise<string> {
  const fallback = `API error (${response.status}).`;
  try {
    const payload = (await response.json()) as {
      detail?: unknown;
      message?: unknown;
    };
    const detail = payload.detail ?? payload.message;
    if (
      detail &&
      typeof detail === "object" &&
      "message" in (detail as Record<string, unknown>)
    ) {
      return String((detail as Record<string, unknown>).message);
    }
    return detail ? String(detail) : fallback;
  } catch {
    return fallback;
  }
}

async function fetchAdminJson<T>(
  path: string,
  init?: RequestInit
): Promise<ApiResult<T>> {
  const apiBase = baseUrl();
  if (!apiBase) {
    return {
      data: null,
      error: "Missing JUPR API base URL environment variable.",
      status: null
    };
  }
  try {
    const response = await fetch(`${apiBase.replace(/\/$/, "")}${path}`, init);
    if (!response.ok) {
      return {
        data: null,
        error: await adminErrorMessage(response),
        status: response.status
      };
    }
    return {
      data: (await response.json()) as T,
      error: null,
      status: response.status
    };
  } catch (error) {
    return {
      data: null,
      error: `Unable to reach API: ${
        error instanceof Error ? error.message : "Unknown error"
      }`,
      status: null
    };
  }
}

function safeCommerceValidation(detail: string): string | null {
  const message = detail.trim();
  if (!message || message.length > 300 || /[<>\u0000-\u001f]/.test(message)) return null;
  if (/\b(?:uuid|database|supabase|credential|runtime|stack|traceback|exception|rpc|sql|fingerprint)\b|JUPR_|\b\w+_id\b|status=|enabled=/i.test(message)) {
    return null;
  }
  if (!/\b(?:quantity|extra|select|selected|event|item|available|registration|cart|bundle|offer)\b/i.test(message)) {
    return null;
  }
  return message;
}

function publicCommerceError(status: number, detail: string): string {
  const normalized = detail.toLowerCase();
  if (status === 400 || status === 422) {
    if (normalized.includes("quantity exceeds")) {
      return "Choose a smaller quantity for that extra.";
    }
    return (
      safeCommerceValidation(detail) ||
      "Check your extras and quantities, then try again."
    );
  }
  if (status === 403) {
    if (normalized.includes("extras are unavailable for this registration")) {
      return "Tournament extras aren’t available for this registration.";
    }
    return "Tournament extras aren’t available right now.";
  }
  if (status === 404) return "Tournament extras aren’t available right now.";
  if (status === 409) {
    if (/pricing|quote|total/.test(normalized)) {
      return "The total changed. Review the updated price and try again.";
    }
    return (
      safeCommerceValidation(detail) ||
      "The available extras changed. Review your choices and try again."
    );
  }
  if (status === 429) return "Too many attempts were made. Wait a moment and try again.";
  return "We couldn’t update your total right now. Please try again.";
}

async function fetchPublicJson<T>(
  path: string,
  init?: RequestInit
): Promise<ApiResult<T>> {
  const apiBase = baseUrl();
  if (!apiBase) {
    return {
      data: null,
      error: "We couldn’t update your total right now. Please try again.",
      status: null
    };
  }
  try {
    const response = await fetch(`${apiBase.replace(/\/$/, "")}${path}`, init);
    if (!response.ok) {
      return {
        data: null,
        error: publicCommerceError(
          response.status,
          await adminErrorMessage(response)
        ),
        status: response.status
      };
    }
    return {
      data: (await response.json()) as T,
      error: null,
      status: response.status
    };
  } catch {
    return {
      data: null,
      error: "We couldn’t update your total right now. Please try again.",
      status: null
    };
  }
}

export function formatCommerceMoney(minor: number | null | undefined): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD"
  }).format(Number(minor || 0) / 100);
}

export async function quoteTournamentCommerce(
  clubSlug: string,
  payload: {
    tournament_id: string;
    registration_id?: string | null;
    event_option_ids: string[];
    item_selections: TournamentCommerceSelection[];
  }
): Promise<ApiResult<{ ok: true; quote: TournamentCommerceQuote }>> {
  return fetchPublicJson(
    `/clubs/${encodeURIComponent(clubSlug)}/tournament-commerce/quote`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }
  );
}

function adminHeaders(accessToken: string): HeadersInit {
  return {
    "Content-Type": "application/json",
    Authorization: `Bearer ${accessToken}`
  };
}

export async function getAdminTournamentCommerceStatus(
  clubId: string,
  accessToken: string
): Promise<ApiResult<Record<string, unknown>>> {
  return fetchAdminJson(
    `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/commerce/status`,
    { headers: adminHeaders(accessToken), cache: "no-store" }
  );
}

export async function listAdminTournamentCommerceTournaments(
  clubId: string,
  accessToken: string
): Promise<
  ApiResult<{
    tournaments: Array<{ id: string; name: string; status?: string }>;
  }>
> {
  return fetchAdminJson(
    `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/commerce/tournaments`,
    { headers: adminHeaders(accessToken), cache: "no-store" }
  );
}

export async function getAdminTournamentCommerceDetail(
  clubId: string,
  tournamentId: string,
  accessToken: string
): Promise<ApiResult<AdminTournamentCommerceDetail>> {
  return fetchAdminJson(
    `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/commerce/tournaments/${encodeURIComponent(tournamentId)}`,
    { headers: adminHeaders(accessToken), cache: "no-store" }
  );
}

export async function getAdminTournamentCommerceOperation(
  clubId: string,
  tournamentId: string,
  operationId: string,
  accessToken: string
): Promise<ApiResult<Record<string, unknown>>> {
  return fetchAdminJson(
    `/admin/clubs/${encodeURIComponent(
      clubId
    )}/tournaments/commerce/tournaments/${encodeURIComponent(
      tournamentId
    )}/operations/${encodeURIComponent(operationId)}`,
    { headers: adminHeaders(accessToken), cache: "no-store" }
  );
}

export async function mutateAdminTournamentCommerce<T>(
  path: string,
  method: "POST" | "PUT" | "PATCH",
  payload: Record<string, unknown>,
  accessToken: string
): Promise<ApiResult<T>> {
  return fetchAdminJson(path, {
    method,
    headers: adminHeaders(accessToken),
    body: JSON.stringify(payload)
  });
}

export function adminTournamentCommerceExportUrl(
  clubId: string,
  tournamentId: string
): string | null {
  const apiBase = baseUrl();
  if (!apiBase) return null;
  return `${apiBase.replace(/\/$/, "")}/admin/clubs/${encodeURIComponent(
    clubId
  )}/tournaments/commerce/tournaments/${encodeURIComponent(
    tournamentId
  )}/fulfillment/export`;
}
