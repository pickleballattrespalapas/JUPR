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
  deduplicated?: boolean | null;
};

export type VerifiedUpdateStatusResponse = {
  ok: boolean;
  mode?: string;
  player?: VerifiedUpdatePlayer | null;
  club?: { id: string; slug: string; name: string };
};

export type ApiResult<T> = { data: T | null; error: string | null };

export function verifiedUpdatesApiBaseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

const VERIFIED_UPDATES_LOAD_ERROR = "We couldn’t load player updates. Please try again later.";

export async function loadVerifiedUpdatePlayers(clubSlug: string): Promise<ApiResult<VerifiedUpdateOptionsResponse>> {
  const apiBase = verifiedUpdatesApiBaseUrl();
  if (!apiBase) return { data: null, error: VERIFIED_UPDATES_LOAD_ERROR };
  try {
    const response = await fetch(`${apiBase.replace(/\/$/, "")}/clubs/${encodeURIComponent(clubSlug)}/verified-updates/options`, { next: { revalidate: 60 } });
    if (!response.ok) return { data: null, error: VERIFIED_UPDATES_LOAD_ERROR };
    return { data: (await response.json()) as VerifiedUpdateOptionsResponse, error: null };
  } catch {
    return { data: null, error: VERIFIED_UPDATES_LOAD_ERROR };
  }
}
