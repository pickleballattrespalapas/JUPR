export type VerifiedUpdatePlayer = {
  id: number;
  name: string;
  is_active?: boolean | null;
  already_requested?: boolean | null;
  request_status?: string | null;
};

export type VerifiedUpdateOptionsResponse = {
  ok: boolean;
  players: VerifiedUpdatePlayer[];
  count: number;
  club?: { id: string; slug: string; name: string };
};

export type VerifiedUpdateRequestResponse = {
  ok: boolean;
  mode?: string;
  request_status?: string | null;
  player?: VerifiedUpdatePlayer | null;
  subscription_id?: string | null;
};

export type ApiResult<T> = { data: T | null; error: string | null };

export function verifiedUpdatesApiBaseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

async function apiError(response: Response): Promise<string> {
  const fallback = `API error (${response.status}).`;
  const text = await response.text().catch(() => "");
  if (!text) return fallback;
  try {
    const payload = JSON.parse(text) as { detail?: unknown; message?: unknown; error?: unknown };
    return `${fallback} ${String(payload.detail ?? payload.message ?? payload.error ?? text)}`;
  } catch {
    return `${fallback} ${text.slice(0, 240)}`;
  }
}

export async function loadVerifiedUpdatePlayers(clubSlug = "tres-palapas"): Promise<ApiResult<VerifiedUpdateOptionsResponse>> {
  const apiBase = verifiedUpdatesApiBaseUrl();
  if (!apiBase) return { data: null, error: "Missing JUPR API base URL." };
  try {
    const response = await fetch(`${apiBase.replace(/\/$/, "")}/clubs/${encodeURIComponent(clubSlug)}/verified-updates/options`, { next: { revalidate: 60 } });
    if (!response.ok) return { data: null, error: await apiError(response) };
    return { data: (await response.json()) as VerifiedUpdateOptionsResponse, error: null };
  } catch (error) {
    return { data: null, error: error instanceof Error ? error.message : "Unable to reach API." };
  }
}
