export function isNextAdminScoreEntryEnabled(): boolean {
  return ["1", "true", "yes", "on"].includes(
    String(process.env.NEXT_PUBLIC_JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY || "")
      .trim()
      .toLowerCase()
  );
}

export type AdminScoreEntryStatus = {
  enabled: boolean;
  ready: boolean;
  status: "ready" | "fallback_required";
  service_role_configured: boolean;
  submit_endpoint?: string | null;
  max_matches?: number;
  fallback?: {
    match_uploader_route?: string;
    match_log_route?: string;
    streamlit_url?: string;
  };
};

const SCORE_ENTRY_READINESS_ERROR =
  "We couldn’t check score entry right now. Use another staff option below.";

function optionalString(value: unknown): string | null | undefined {
  return typeof value === "string" ? value : value === null ? null : undefined;
}

function normalizeStatus(payload: unknown): AdminScoreEntryStatus | null {
  if (!payload || typeof payload !== "object") return null;
  const value = payload as Record<string, unknown>;
  const rawFallback =
    value.fallback && typeof value.fallback === "object"
      ? (value.fallback as Record<string, unknown>)
      : null;

  return {
    enabled: value.enabled === true,
    ready: value.ready === true,
    status: value.ready === true ? "ready" : "fallback_required",
    service_role_configured: value.service_role_configured === true,
    submit_endpoint: optionalString(value.submit_endpoint),
    max_matches:
      typeof value.max_matches === "number" ? value.max_matches : undefined,
    fallback: rawFallback
      ? {
          match_uploader_route:
            optionalString(rawFallback.match_uploader_route) ?? undefined,
          match_log_route:
            optionalString(rawFallback.match_log_route) ?? undefined,
          streamlit_url: optionalString(rawFallback.streamlit_url) ?? undefined
        }
      : undefined
  };
}

export async function getAdminScoreEntryStatus(
  apiBase: string | null,
  clubId: string
): Promise<{ data: AdminScoreEntryStatus | null; error: string | null }> {
  if (!apiBase) return { data: null, error: SCORE_ENTRY_READINESS_ERROR };
  try {
    const response = await fetch(
      `${apiBase.replace(/\/$/, "")}/admin/clubs/${encodeURIComponent(clubId)}/score-entry/status`,
      { cache: "no-store" }
    );
    if (!response.ok) return { data: null, error: SCORE_ENTRY_READINESS_ERROR };
    const data = normalizeStatus(await response.json());
    return data
      ? { data, error: null }
      : { data: null, error: SCORE_ENTRY_READINESS_ERROR };
  } catch {
    return { data: null, error: SCORE_ENTRY_READINESS_ERROR };
  }
}
