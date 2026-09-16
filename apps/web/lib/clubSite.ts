export const DISPLAY_LABELS = {
  ratings: "Current ratings",
  records: "Win–loss records",
  match_counts: "Matches played",
  win_percentage: "Win percentage",
  rating_changes: "Rating gains and gaps",
  singles: "Singles statistics",
  last_played: "Last played date",
  player_status: "Active/inactive status",
  qualification: "Leaderboard qualification",
  badges: "Badges and prestige",
  profile_ratings: "Profile: rating history",
  profile_positions: "Profile: league positions",
  profile_trophies: "Profile: trophy case",
  profile_social: "Profile: social results and relationships",
  profile_matches: "Profile: match history",
  result_scores: "Results: game scores",
  result_summary: "Results: summary statistics",
  registration_description: "Registration: event description",
  registration_schedule: "Registration: event schedule",
} as const;
export type DisplayKey = keyof typeof DISPLAY_LABELS;
export function accentTextColor(hex: string): string {
  const rgb = [1, 3, 5].map((at) => parseInt(hex.slice(at, at + 2), 16) / 255);
  const linear = rgb.map((c) => c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  return linear[0] * 0.2126 + linear[1] * 0.7152 + linear[2] * 0.0722 > 0.179 ? "#000000" : "#ffffff";
}
export type DisplaySettings = Partial<Record<DisplayKey, boolean>>;
export type SiteBlock = {
  id: string;
  kind: "text" | "image" | "button" | "divider" | "links";
  heading: string;
  text: string;
  url: string;
  alt: string;
  span: 3 | 4 | 6 | 8 | 12;
  align: "left" | "center" | "right";
  tone: "plain" | "soft" | "accent";
  padding: "small" | "medium" | "large";
};
export type SitePage = {
  slug: string;
  title: string;
  in_navigation: boolean;
  blocks: SiteBlock[];
};
export type SiteDocument = {
  schema_version: 1;
  name: string;
  description: string;
  location: string;
  visitor_info: string;
  logo_url: string;
  accent: string;
  visibility: "listed" | "unlisted";
  display: DisplaySettings;
  pages: SitePage[];
};
export type PublicSite = {
  club_id: string;
  slug: string;
  document: SiteDocument;
  published_at: string;
};
export type AdminSite = {
  club_id: string;
  slug: string;
  club_active: boolean;
  revision: number;
  draft: SiteDocument;
  published: SiteDocument | null;
  published_at: string | null;
};
export type DirectoryClub = {
  slug: string;
  name: string;
  description: string;
  location: string;
  logo_url: string;
};
export const LAST_CLUB_COOKIE = "pcs_last_public_club";
export function validClubSlug(slug: string) {
  return /^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug) && slug.length <= 100;
}
export function visible(display: DisplaySettings | undefined, key: DisplayKey) {
  return display?.[key] !== false;
}
export const clubPageHref = (slug: string, page: string) =>
  `/clubs/${encodeURIComponent(slug)}${page === "home" ? "" : `/pages/${encodeURIComponent(page)}`}`;
export const CLUB_LINKS = [
  ["Players", "players"],
  ["Leaderboards", "leaderboards"],
  ["Leagues", "leagues"],
  ["Tournaments", "tournaments"],
  ["Match history", "matches"],
  ["Play", "play"],
  ["Match Explorer", "match-explorer"],
  ["Weekly recap", "weekly-recap"],
  ["Badges & trophies", "badge-codex"],
  ["Interclub leagues", "interclub"],
] as const;

export function safeSiteUrl(url: string): string | undefined {
  if (!url || /[\u0000-\u0020\\]/.test(url)) return undefined;
  if (url.startsWith("/") && !url.startsWith("//")) return url;
  try {
    const u = new URL(url);
    return u.protocol === "https:" && !u.username && !u.password
      ? url
      : undefined;
  } catch {
    return undefined;
  }
}
export function safeImageUrl(url: string): string | undefined {
  return /^data:image\/(png|jpeg|webp);base64,[A-Za-z0-9+/=]+$/.test(url)
    ? url
    : safeSiteUrl(url);
}
