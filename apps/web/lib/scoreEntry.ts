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
  status: string;
  service_role_configured: boolean;
  submit_endpoint?: string | null;
  max_matches?: number;
  fallback?: {
    match_uploader_route?: string;
    match_log_route?: string;
    streamlit_url?: string;
  };
  warnings?: string[];
};

export async function getAdminScoreEntryStatus(
  apiBase: string | null,
  clubId: string
): Promise<{ data: AdminScoreEntryStatus | null; error: string | null }> {
  if (!apiBase) return { data: null, error: "API base URL is not configured." };
  try {
    const response = await fetch(
      `${apiBase.replace(/\/$/, "")}/admin/clubs/${encodeURIComponent(clubId)}/score-entry/status`,
      { cache: "no-store" }
    );
    if (!response.ok) return { data: null, error: `Score-entry readiness check failed (${response.status}).` };
    return { data: (await response.json()) as AdminScoreEntryStatus, error: null };
  } catch (error) {
    return { data: null, error: `Unable to reach score-entry readiness check: ${error instanceof Error ? error.message : "Unknown error"}` };
  }
}
