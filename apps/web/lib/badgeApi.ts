export type PublicBadgeEarner = {
  player_id: string | number;
  player_name: string;
  earned_at?: string | null;
};

export type PublicBadge = {
  badge_id: string;
  name: string;
  category?: string | null;
  prestige?: number | null;
  rarity?: string | null;
  icon_key?: string | null;
  scope?: string | null;
  state?: string | null;
  description?: string | null;
  requirements?: string | null;
  earners_count?: number | null;
  recent_earners?: PublicBadgeEarner[];
};

export type BadgeCodexSection = {
  name: string;
  badges: PublicBadge[];
};

export type BadgeCodexResponse = {
  club: { id: string; slug: string; name: string };
  summary: {
    badge_count: number;
    earned_badge_count: number;
    unclaimed_badge_count: number;
    total_unique_earners_by_badge: number;
  };
  sections: BadgeCodexSection[];
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
    // Fall through to a short text excerpt below.
  }
  return `${fallback} ${bodyText.slice(0, 240)}`;
}

async function fetchJson<T>(path: string): Promise<ApiResult<T>> {
  const apiBase = baseUrl();
  if (!apiBase) return { data: null, error: "Missing JUPR API base URL environment variable." };
  const url = `${apiBase.replace(/\/$/, "")}${path}`;
  try {
    const response = await fetch(url, { next: { revalidate: 60 } });
    if (!response.ok) return { data: null, error: await apiErrorMessage(response) };
    return { data: (await response.json()) as T, error: null };
  } catch (error) {
    return { data: null, error: `Unable to reach API: ${error instanceof Error ? error.message : "Unknown error"}` };
  }
}

export async function getClubBadgeCodex(clubSlug: string): Promise<ApiResult<BadgeCodexResponse>> {
  return fetchJson<BadgeCodexResponse>(`/clubs/${clubSlug}/badges`);
}
