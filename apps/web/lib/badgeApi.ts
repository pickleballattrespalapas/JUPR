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
  lifecycle_state?: string | null;
  badge_status: string;
  badge_award_timing: string;
  badge_scope?: string | null;
  catalog_bucket: string;
  availability: string;
  description?: string | null;
  requirements?: string | null;
  earners_count?: number | null;
  recent_earners?: PublicBadgeEarner[];
};

export type BadgeCodexSection = {
  name: string;
  badges: PublicBadge[];
};

export type BadgeCatalogBucket = {
  name: string;
  description: string;
  badge_count: number;
  sections: BadgeCodexSection[];
};

export type BadgeTrophyRoomEntry = {
  player_id: string | number;
  player_name: string;
  unique_badge_count: number;
  award_count: number;
  prestige_total: number;
  latest_earned_at?: string | null;
  latest_badges: Array<{
    badge_id: string;
    badge_name: string;
    earned_at?: string | null;
  }>;
};

export type BadgeCodexResponse = {
  seasons?: Array<{ id: string; name: string; start_date: string; end_date: string; timezone: string }>;
  club: { id: string; slug: string; name: string };
  summary: {
    badge_count: number;
    earned_badge_count: number;
    unclaimed_badge_count: number;
    total_unique_earners_by_badge: number;
    unique_earner_count: number;
    complete_definition_count: number;
  };
  sections: BadgeCodexSection[];
  catalog_buckets: BadgeCatalogBucket[];
  filters: {
    categories: string[];
    scopes: string[];
    statuses: string[];
    award_timings: string[];
  };
  trophy_room: BadgeTrophyRoomEntry[];
};

export type BadgeEarnersResponse = {
  club: { id: string; slug: string; name: string };
  badge_id: string;
  badge: PublicBadge;
  earners: PublicBadgeEarner[];
  total: number;
  offset: number;
  limit: number;
  has_more: boolean;
};

type ApiResult<T> = { data: T | null; error: string | null };

const PUBLIC_BADGE_LOAD_ERROR = "We couldn’t load badge information right now. Please try again.";

export function publicBadgeRarityLabel(value?: string | null): string {
  const labels: Record<string, string> = {
    common: "Common",
    rare: "Rare",
    epic: "Epic",
    legendary: "Legendary"
  };
  return labels[String(value || "").trim().toLowerCase()] || "Special";
}

function baseUrl(): string | null {
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}

function apiErrorMessage(response: Response): string {
  console.error(`[badgeApi] Request failed with status ${response.status}.`);
  return PUBLIC_BADGE_LOAD_ERROR;
}

async function fetchJson<T>(path: string): Promise<ApiResult<T>> {
  const apiBase = baseUrl();
  if (!apiBase) {
    console.error("[badgeApi] JUPR API base URL is not configured.");
    return { data: null, error: PUBLIC_BADGE_LOAD_ERROR };
  }
  const url = `${apiBase.replace(/\/$/, "")}${path}`;
  try {
    const response = await fetch(url, { next: { revalidate: 60 } });
    if (!response.ok) return { data: null, error: apiErrorMessage(response) };
    return { data: (await response.json()) as T, error: null };
  } catch (error) {
    console.error("[badgeApi] Request failed.", error);
    return { data: null, error: PUBLIC_BADGE_LOAD_ERROR };
  }
}

export async function getClubBadgeCodex(clubSlug: string): Promise<ApiResult<BadgeCodexResponse>> {
  return fetchJson<BadgeCodexResponse>(`/clubs/${clubSlug}/badges`);
}

export async function getClubBadgeEarners(
  clubSlug: string,
  badgeId: string,
  { offset = 0, limit = 25 }: { offset?: number; limit?: number } = {}
): Promise<ApiResult<BadgeEarnersResponse>> {
  const params = new URLSearchParams({
    offset: String(Math.max(0, Math.round(offset))),
    limit: String(Math.max(1, Math.min(100, Math.round(limit))))
  });
  return fetchJson<BadgeEarnersResponse>(
    `/clubs/${encodeURIComponent(clubSlug)}/badges/${encodeURIComponent(badgeId)}/earners?${params.toString()}`
  );
}
