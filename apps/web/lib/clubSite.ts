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
export const LEADERBOARD_CARD_LABELS = {
  highest_rating: "Highest rating",
  most_improved: "Most improved",
  best_win_pct: "Best win %",
  most_wins: "Most wins",
  most_matches: "Most games played",
  hot_hand: "Current win streak",
  point_differential: "Best point differential",
  average_margin: "Best average margin",
  longest_win_streak: "Longest win streak",
  close_game_record: "Best close-game record",
  biggest_upset: "Biggest upset",
  most_upsets: "Most upset wins",
  opponent_strength: "Strongest opposition",
  over_performance: "Wins above expectation",
  best_partnership: "Best partnership",
  partner_variety: "Most partners",
  playing_days: "Most playing days",
} as const;
export type LeaderboardCard = keyof typeof LEADERBOARD_CARD_LABELS;
export const LEADERBOARD_CARD_GROUPS = ["Ratings & improvement", "Results", "Streaks", "Competition", "Participation & partnerships"] as const;
export const LEADERBOARD_CARD_DETAILS: Record<LeaderboardCard, {
  group: (typeof LEADERBOARD_CARD_GROUPS)[number];
  description: string;
  sample: string;
  fields: DisplayKey[];
}> = {
  highest_rating: { group: "Ratings & improvement", description: "Highest current rating, including when viewing a past season.", sample: "games", fields: ["ratings"] },
  most_improved: { group: "Ratings & improvement", description: "Rating gained from the start of the selected period.", sample: "games", fields: ["rating_changes"] },
  best_win_pct: { group: "Results", description: "Percentage of games won in the selected period.", sample: "games", fields: ["win_percentage"] },
  most_wins: { group: "Results", description: "Total games won in the selected period.", sample: "games", fields: ["records"] },
  most_matches: { group: "Participation & partnerships", description: "Total games played in the selected period.", sample: "games", fields: ["match_counts"] },
  hot_hand: { group: "Streaks", description: "Consecutive wins ending with the player’s latest game in this period.", sample: "games", fields: ["records"] },
  point_differential: { group: "Results", description: "Team points scored minus points conceded across this period.", sample: "games", fields: ["records"] },
  average_margin: { group: "Results", description: "Average team point difference per game, including wins and losses.", sample: "games", fields: ["records"] },
  longest_win_streak: { group: "Streaks", description: "Longest run of consecutive wins within this period.", sample: "games", fields: ["records"] },
  close_game_record: { group: "Results", description: "Win percentage in games decided by two points or fewer.", sample: "close games", fields: ["win_percentage"] },
  biggest_upset: { group: "Competition", description: "Teams ranked by the largest rating gap they overcame in a win against opponents averaging at least 0.25 JUPR higher before the game. Each pair appears once.", sample: "upset wins together", fields: ["ratings", "records"] },
  most_upsets: { group: "Competition", description: "Wins against teams averaging at least 0.25 JUPR higher before the game.", sample: "upset wins", fields: ["ratings", "records"] },
  opponent_strength: { group: "Competition", description: "Average opposing team rating, using ratings before each game.", sample: "games", fields: ["ratings"] },
  over_performance: { group: "Competition", description: "Actual wins minus expected wins based on both teams’ pre-game ratings.", sample: "games", fields: ["ratings", "records"] },
  best_partnership: { group: "Participation & partnerships", description: "Each player’s highest win percentage with a qualifying partner in this period.", sample: "games with that partner", fields: ["win_percentage"] },
  partner_variety: { group: "Participation & partnerships", description: "Number of different doubles partners in this period.", sample: "games", fields: ["match_counts"] },
  playing_days: { group: "Participation & partnerships", description: "Different days with recorded games, using the selected period’s timezone.", sample: "playing days", fields: ["match_counts"] },
};
export type LeaderboardCardOptions = { minimum: number; depth: number };
export type LeaderboardSeason = {
  id: string;
  name: string;
  start_date: string;
  end_date: string | null;
  timezone: string;
};
export type LeaderboardSettings = {
  cards: LeaderboardCard[];
  show_summary: boolean;
  seasons: LeaderboardSeason[];
  default_season_id: string | null;
  min_games: number;
  timezone?: string;
  card_options?: Partial<Record<LeaderboardCard, LeaderboardCardOptions>>;
};
export const DEFAULT_LEADERBOARD_CARDS: LeaderboardCard[] = [
  "highest_rating", "most_improved", "best_win_pct", "most_wins",
];
export function leaderboardSettings(settings?: Partial<LeaderboardSettings>): LeaderboardSettings {
  return {
    cards: settings?.cards ?? [...DEFAULT_LEADERBOARD_CARDS],
    show_summary: settings?.show_summary ?? true,
    seasons: settings?.seasons ?? [],
    default_season_id: settings?.default_season_id ?? null,
    min_games: settings?.min_games ?? 0,
    timezone: settings?.timezone ?? "America/Mazatlan",
    card_options: settings?.card_options ?? {},
  };
}
export function leaderboardSettingsError(settings?: LeaderboardSettings): string | null {
  if (!settings) return null;
  if (!Number.isInteger(settings.min_games) || settings.min_games < 0 || settings.min_games > 10000) {
    return "Use a whole number from 0 to 10,000 for the minimum games.";
  }
  if (settings.timezone !== undefined) {
    if (!settings.timezone.trim()) return "Choose a timezone for All time statistics.";
    try { new Intl.DateTimeFormat("en", { timeZone: settings.timezone }).format(); }
    catch { return "For All time statistics, use a timezone such as America/Mazatlan or UTC."; }
  }
  for (const [key, options] of Object.entries(settings.card_options ?? {})) {
    const label = LEADERBOARD_CARD_LABELS[key as LeaderboardCard];
    if (!label) return "Choose a supported leaderboard statistic.";
    if (!options || !Number.isInteger(options.minimum) || options.minimum < 0 || options.minimum > 10000) {
      return `${label}: use a minimum from 0 to 10,000.`;
    }
    if (!Number.isInteger(options.depth) || options.depth < 1 || options.depth > 10) {
      return `${label}: show between 1 and 10 ${key === "biggest_upset" ? "teams" : "players"}.`;
    }
  }
  for (const [index, season] of settings.seasons.entries()) {
    const label = season.name.trim() || `Season ${index + 1}`;
    if (!season.name.trim()) return `${label}: add a season name.`;
    if (!season.start_date) return `${label}: choose a start date.`;
    if (season.end_date && season.end_date < season.start_date) return `${label}: the end date must be on or after the start date.`;
    if (!season.timezone.trim()) return `${label}: choose a timezone.`;
    try { new Intl.DateTimeFormat("en", { timeZone: season.timezone }).format(); }
    catch { return `${label}: use a timezone such as America/Mazatlan or UTC.`; }
  }
  return null;
}
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
  leaderboard?: LeaderboardSettings;
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
