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
  page_visibility?: Partial<Record<ClubPageKey, PageVisibility>>;
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

export type ClubPageKey = (typeof CLUB_LINKS)[number][1];
export type PageVisibility = "public" | "private";
export type PageNavigationSettings = Pick<SiteDocument, "page_visibility"> & {
  pages: Pick<SitePage, "slug" | "in_navigation">[];
};
export const PAGE_SCOPE_LABELS: Record<ClubPageKey, string> = {
  players: "Player directory and individual profiles",
  leaderboards: "Club leaderboards",
  leagues: "Leagues, team leagues, results and challenge ladder",
  tournaments: "Tournament pages, registration, rosters and results",
  matches: "Match history and individual match details",
  play: "Play tools, generators and live sessions",
  "match-explorer": "Match Explorer",
  "weekly-recap": "Weekly recap",
  "badge-codex": "Badges and trophies catalogue",
  interclub: "This club’s interclub league list",
};
export function publicClubLinks(doc: PageNavigationSettings) {
  return CLUB_LINKS.filter(([, key]) => doc.page_visibility?.[key] !== "private");
}
// All routes in a section inherit its discovery setting. A shared deep link
// still opens normally, including a player profile or tournament registration.
export function clubPageSection(href: string, slug: string): string | null {
  let parts: string[];
  try {
    parts = new URL(href, "https://pcs.invalid").pathname
      .split("/").filter(Boolean).map(decodeURIComponent);
  } catch {
    return null;
  }
  if (parts[0] !== "clubs" || parts[1] !== slug) return null;
  const section = parts[2] || "home";
  if (section === "pages") return `pages/${parts[3] || ""}`;
  if (section === "league-results" || section === "challenge-ladder" || section.startsWith("team-league")) {
    return "leagues";
  }
  if (section.startsWith("tournament-")) return "tournaments";
  if (["live", "round-robin-generator", "ladder-generator", "team-match-generator", "play-generators"].includes(section)) {
    return "play";
  }
  if (section === "verified-updates") return "players";
  return section;
}
export function publicClubPage(doc: PageNavigationSettings, slug: string, href: string) {
  const section = clubPageSection(href, slug);
  if (section?.startsWith("pages/")) {
    return doc.pages.find((p) => p.slug === section.slice(6))?.in_navigation !== false;
  }
  return doc.page_visibility?.[section as ClubPageKey] !== "private";
}
export function canLinkClubPage(doc: PageNavigationSettings, slug: string, href: string, currentPath = "") {
  return publicClubPage(doc, slug, href) ||
    (!!currentPath && clubPageSection(href, slug) === clubPageSection(currentPath, slug));
}

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
