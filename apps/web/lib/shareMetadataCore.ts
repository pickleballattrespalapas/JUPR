/** Public-only sharing information. No cookies, credentials, or private APIs. */
export type PublicRecord = Record<string, unknown>;
export type ReadPublic = (path: string) => Promise<PublicRecord | null>;
export type SharingClub = {
  name: string; slug: string; description?: string; location?: string;
  image?: string; accent?: string; discoverable?: boolean; available?: boolean;
  page?: { title: string; description: string; image?: string };
};
export type PageShare = {
  title: string; description: string; siteName: string; path: string;
  query: string; image?: string; accent?: string; noindex: boolean;
};

export const SHARE_QUERY_KEYS = [
  "tournament", "tournament_id", "registration_slug", "league", "league_name",
  "week", "tab", "player", "player_id", "pid", "season", "season_id",
  "view", "division_id", "date", "week_start"
] as const;
export function publicShareQuery(query: URLSearchParams): URLSearchParams {
  const result = new URLSearchParams();
  for (const key of SHARE_QUERY_KEYS) {
    const value = query.get(key);
    if (value && value.length <= 200 && !/[\u0000-\u001f]/.test(value)) result.set(key, value);
  }
  return result;
}
export function privateShareLink(path: string, query: URLSearchParams): boolean {
  return /\/(admin|auth|account|settings|confirmation|edit|respond|accept-invitation)(\/|$)/i.test(path)
    || Array.from(query.keys()).some((key) => /token|email|password|secret|authorization|^edit$/i.test(key));
}
export function record(value: unknown): PublicRecord {
  return value && typeof value === "object" && !Array.isArray(value) ? value as PublicRecord : {};
}
function rows(value: unknown): PublicRecord[] { return Array.isArray(value) ? value.map(record) : []; }
export function plainText(value: unknown, limit = 230): string {
  if (typeof value !== "string") return "";
  const text = value.replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, " ")
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, " ").replace(/<[^>]*>/g, " ")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, " ").replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/&amp;/g, "&").replace(/&nbsp;/g, " ").replace(/&quot;/g, '"')
    .replace(/&#39;|&apos;/g, "'").replace(/(^|\s)[#*_`>]+/g, "$1")
    .replace(/[\u0000-\u001f]+/g, " ").replace(/\s+/g, " ").trim();
  return text.length > limit ? `${text.slice(0, limit - 1).trimEnd()}…` : text;
}
export function publicImage(value: unknown, origin: string): string | undefined {
  if (typeof value !== "string" || !value.trim()) return undefined;
  try {
    const url = new URL(value, origin);
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) return undefined;
    if (/^(localhost|127\.|0\.|10\.|192\.168\.|169\.254\.|\[)/i.test(url.hostname)) return undefined;
    if (/^172\.(1[6-9]|2\d|3[01])\./.test(url.hostname) || !url.hostname.includes('.')) return undefined;
    // Signed/private images and access tokens must never become public metadata.
    if (privateShareLink(url.pathname, url.searchParams) || /\/object\/sign\//.test(url.pathname)) return undefined;
    return url.href;
  } catch { return undefined; }
}
function pretty(value: string): string {
  if (/^[0-9a-f-]{32,}$/i.test(value) || /^\d+$/.test(value)) return "Details";
  return plainText(value.replace(/[-_]+/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()), 100);
}
function date(value: unknown): Date | null {
  const text = typeof value === "string" ? value.slice(0, 10) : "";
  if (!/^\d{4}-\d{2}-\d{2}$/.test(text)) return null;
  const parsed = new Date(`${text}T12:00:00Z`);
  return Number.isNaN(parsed.getTime()) || parsed.toISOString().slice(0, 10) !== text ? null : parsed;
}
export function dateRange(start: unknown, end: unknown): string {
  const first = date(start), last = date(end);
  const format = (d: Date) => new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" }).format(d);
  if (!first) return "";
  if (!last || last <= first) return format(first);
  if (first.getUTCFullYear() === last.getUTCFullYear() && first.getUTCMonth() === last.getUTCMonth()) {
    return `${new Intl.DateTimeFormat("en-US", { month: "short", timeZone: "UTC" }).format(first)} ${first.getUTCDate()}–${last.getUTCDate()}, ${first.getUTCFullYear()}`;
  }
  return `${format(first)} – ${format(last)}`;
}
const PAGES: Record<string, [string, string]> = {
  tournaments: ["Tournaments", "Explore pickleball tournaments at {club}. Find event dates, registration, player rosters, partners, and results."],
  leagues: ["Leagues", "Explore pickleball leagues at {club}. Follow league schedules, standings, weekly results, and player achievements."],
  "league-results": ["League Results", "See league standings, weekly results, and player achievements at {club}."],
  "team-leagues": ["Team Leagues", "Follow team leagues, schedules, and results at {club}."],
  "team-league": ["Team Leagues", "Follow team leagues, schedules, and results at {club}."],
  leaderboards: ["Leaderboards", "Follow the club leaderboard and player achievements at {club}."],
  players: ["Players", "Meet the players at {club} and explore their public player profiles."],
  matches: ["Match History", "Explore recorded matches and results at {club}."],
  play: ["Play", "Find pickleball play tools, round robins, ladders, and live sessions at {club}."],
  live: ["Live Play", "Follow live pickleball sessions, court assignments, and results at {club}."],
  "live-sessions": ["Live Sessions", "Follow pickleball sessions, court assignments, and results at {club}."],
  "round-robin-generator": ["Round Robins", "Build and follow pickleball round robins at {club}. View matchups, rounds, and results."],
  "play-generators": ["Play Tools", "Create and follow round robins, ladders, and team matches at {club}."],
  "ladder-generator": ["Ladders", "Build and follow pickleball ladder sessions at {club}. View courts, rounds, and results."],
  "team-match-generator": ["Team Matches", "Build and follow team pickleball matches at {club}."],
  "challenge-ladder": ["Challenge Ladder", "Follow the challenge ladder, standings, and challenges at {club}."],
  "weekly-recap": ["Weekly Recap", "Catch up on pickleball events, results, and highlights at {club}."],
  "badge-codex": ["Badges & Trophies", "Explore pickleball achievements, badges, and trophies at {club}."],
  "match-explorer": ["Match Explorer", "Explore pickleball matchups at {club}."],
  interclub: ["Interclub Leagues", "Follow interclub pickleball leagues involving {club}, including meet schedules, standings, and results."],
  "tournament-registration": ["Tournament Registration", "Explore tournament registration at {club}."],
  "tournament-roster": ["Tournament Rosters", "See who is playing in tournaments at {club}."],
  "tournament-results": ["Tournament Results", "Follow tournament draws and results at {club}."],
};
const TABS: Record<string, string> = {
  register: "Registration", registration: "Registration", roster: "Player Roster",
  partners: "Partner Board", "partner-board": "Partner Board", "players-needing-partners": "Partner Board",
  "needs-partner": "Partner Board", results: "Results", live: "Live Results", "live-results": "Live Results",
  draws: "Draws", standings: "Standings", "weekly-history": "Weekly History",
  weekly: "Weekly Results", players: "Player Summaries", schedule: "Schedule", rules: "Rules",
  "edit-registration": "Edit Registration"
};
function eventImage(event: PublicRecord, settings: PublicRecord): unknown {
  const tags = record(event.event_tags);
  return event.share_image_url || event.image_url || event.poster_url || event.banner_url
    || settings.share_image_url || settings.image_url || settings.poster_url || settings.banner_url
    || tags.share_image_url || tags.image_url || tags.poster_url || tags.banner_url;
}

export async function resolveClubShare(input: {
  path: string; query: URLSearchParams; origin: string; privateLink?: boolean;
}, club: SharingClub, read: ReadPublic): Promise<PageShare> {
  const query = publicShareQuery(input.query);
  let parts: string[];
  try { parts = input.path.split('/').filter(Boolean).map(decodeURIComponent); }
  catch { parts = []; }
  const section = parts[2] || "home";
  const page = PAGES[section] || [pretty(section), `Explore ${pretty(section).toLowerCase()} at {club}.`];
  const name = plainText(club.name, 110) || "Pickleball Club";
  const base: PageShare = {
    title: `${page[0]} at ${name}`, description: page[1].replace(/\{club\}/g, name),
    siteName: name, path: input.path, query: query.toString(),
    image: publicImage(club.image, input.origin), accent: club.accent,
    noindex: club.discoverable === false
  };
  if (club.available === false) return { ...base, title: "Club unavailable", description: "This club page is currently unavailable.", image: undefined, query: "", noindex: true };
  if (input.privateLink || privateShareLink(input.path, input.query)) {
    return { ...base, title: `Private page | ${name}`, description: `A private link for ${name}.`, path: `/clubs/${encodeURIComponent(club.slug)}`, query: "", noindex: true };
  }
  if (club.page) return { ...base, title: `${plainText(club.page.title, 140)} | ${name}`, description: plainText(club.page.description) || `Explore ${club.page.title} at ${name}.`, image: publicImage(club.page.image, input.origin) || base.image };
  if (section === "home") return { ...base, title: name, description: plainText(club.description) || `Explore pickleball at ${name}. Find leagues, tournaments, players, and club results.` };
  const subpage = parts[3] || query.get("tab") || "";
  const detail = (title: string, description: string, image?: unknown): PageShare => ({
    ...base, title: `${plainText(title, 170)} | ${name}`, description: plainText(description),
    image: publicImage(image, input.origin) || base.image
  });
  const missing = (kind: string): PageShare => ({ ...detail(`${kind} unavailable`, `This ${kind.toLowerCase()} could not be found at ${name}.`), noindex: true });
  const prefix = `/clubs/${encodeURIComponent(club.slug)}`;
  const scopedRead: ReadPublic = async (path) => {
    const data = await read(path);
    const owner = record(data?.club);
    return owner.slug && owner.slug !== club.slug ? null : data;
  };

  if (section === "tournaments" || section.startsWith("tournament-")) {
    const slug = query.get("tournament") || query.get("registration_slug");
    const pathId = section === "tournaments" && parts[3] && !TABS[parts[3]] && !["past", "edit", "confirmation"].includes(parts[3]) ? parts[3] : null;
    const id = query.get("tournament_id") || pathId;
    if (!slug && !id) {
      if (subpage === "past") return { ...base, title: `Past Tournaments at ${name}`, description: `Explore past pickleball tournaments and results at ${name}.` };
      return base; // An index must NEVER advertise the API's default tournament.
    }
    const selectors = new URLSearchParams();
    if (slug) selectors.set("registration_slug", slug);
    if (id) selectors.set("tournament_id", id);
    const data = await scopedRead(`${prefix}/tournament-registration?${selectors}`);
    const event = record(data?.tournament), settings = record(data?.settings);
    if (!event.id || !event.name || (id && event.id !== id) || (slug && settings.registration_slug !== slug)) return missing("Tournament");
    const sponsors = await scopedRead(`${prefix}/tournaments/${encodeURIComponent(String(event.id))}/sponsors`);
    const presenting = sponsors?.tournament_id === event.id ? rows(sponsors.sponsors).filter((s) => s.tier === "presenting").map((s) => plainText(s.name, 90)).filter(Boolean).join(" & ") : "";
    const tab = TABS[pathId ? parts[4] : subpage] || (section === "tournament-registration" ? "Registration" : section === "tournament-roster" ? "Player Roster" : section === "tournament-results" ? "Results" : "");
    const title = `${event.name}${tab ? ` — ${tab}` : ""}`;
    const facts = [dateRange(event.start_date, event.end_date), plainText(settings.location_name || club.location, 100)].filter(Boolean).join(" · ");
    const action = tab === "Player Roster" ? "See who's playing." : tab === "Partner Board" ? "Find a tournament partner." : /Results|Draws/.test(tab) ? "Follow tournament draws and results." : data?.registration_open === true ? "Registration open. View events, partners, and tournament details." : "View tournament events, registration information, and results.";
    return detail(title, [facts, presenting ? `Presented by ${presenting}.` : "", action].filter(Boolean).join(" "), eventImage(event, settings));
  }
  const selectedLeague = section === "leagues" ? parts[3] : ["league-results", "leaderboards"].includes(section) ? query.get("league_name") || query.get("league") : null;
  if (selectedLeague && selectedLeague !== "OVERALL" && selectedLeague !== "past") {
    const data = await scopedRead(`${prefix}/league-results?${new URLSearchParams({ league_name: selectedLeague })}`);
    if (data?.selected_league !== selectedLeague) return missing("League");
    const league = record(data.league);
    const tab = TABS[parts[4] || query.get("tab") || ""] || (section === "leaderboards" ? "Leaderboard" : section === "league-results" ? "Results" : "");
    return detail(`${plainText(league.label || league.name) || selectedLeague}${tab ? ` — ${tab}` : ""}`, `${dateRange(league.start_date, league.end_date)} Follow ${plainText(league.name) || selectedLeague} at ${name}: schedules, standings, and weekly results.`, eventImage(league, {}));
  }
  if (section === "leagues" && subpage === "past") return { ...base, title: `Past Leagues at ${name}`, description: `Explore completed pickleball leagues and results at ${name}.` };
  if (section === "players" && parts[3]) {
    const data = await scopedRead(`${prefix}/players/${encodeURIComponent(parts[3])}?recent_limit=1&history_limit=1`);
    const player = record(data?.player), identity = record(data?.identity);
    if (String(player.id ?? "") !== parts[3]) return missing("Player");
    const playerName = plainText(identity.display_name || player.display_name || player.name, 110);
    return detail(`${playerName}${parts[4] === "matches" ? " — Match History" : " — Player Profile"}`, `Explore ${playerName}'s public pickleball profile at ${name}.`);
  }
  const sessionIndex = parts.indexOf("sessions", 3);
  const sessionKey = sessionIndex >= 0 ? parts[sessionIndex + 1] : ["live", "live-sessions"].includes(section) ? parts[3] : null;
  if (sessionKey) {
    const generator = /generator/.test(section);
    const data = await scopedRead(`${prefix}/${generator ? 'play-generators/sessions' : 'live-sessions'}/${encodeURIComponent(sessionKey)}`);
    const session = record(data?.session);
    if (session.session_key !== sessionKey) return missing("Session");
    const event = record(session.event);
    const title = plainText(session.title || session.name || event.name, 130) || page[0];
    const roundIndex = parts.indexOf("rounds");
    const round = roundIndex >= 0 && /^\d+$/.test(parts[roundIndex + 1] || "") ? `Round ${Number(parts[roundIndex + 1])}` : "";
    const suffix = round || (parts.includes("standings") ? "Standings" : "");
    return detail(`${title}${suffix ? ` — ${suffix}` : ""}`, `Follow ${title} at ${name}. View ${round ? `${round.toLowerCase()}, ` : ""}matchups, court assignments, and results.`);
  }
  if (section === "matches" && parts[3]) {
    const data = await scopedRead(`${prefix}/matches/${encodeURIComponent(parts[3])}`);
    const match = record(data?.match);
    if (String(match.id ?? "") !== parts[3]) return missing("Match");
    const team = (value: unknown) => rows(value).map((p) => plainText(p.name, 70)).filter(Boolean).join(" & ");
    return detail(`${team(match.team_1)} vs ${team(match.team_2)}`, `View this recorded pickleball match at ${name}.`);
  }
  return base;
}
