import type { MetadataRoute } from "next";

const routes = [
  "/",
  "/site-map",
  "/clubs/tres-palapas",
  "/clubs/tres-palapas/live",
  "/clubs/tres-palapas/leaderboards",
  "/clubs/tres-palapas/match-explorer",
  "/clubs/tres-palapas/league-results",
  "/clubs/tres-palapas/badge-codex",
  "/clubs/tres-palapas/challenge-ladder",
  "/clubs/tres-palapas/weekly-recap",
  "/clubs/tres-palapas/tournament-registration",
  "/clubs/tres-palapas/tournament-roster",
  "/clubs/tres-palapas/players",
  "/clubs/tres-palapas/matches",
  "/how-ratings-work",
  "/faq",
  "/privacy",
  "/terms",
  "/support",
  "/data-corrections"
];

function baseUrl(): string {
  return (process.env.NEXT_PUBLIC_JUPR_WEB_BASE_URL || process.env.JUPR_WEB_BASE_URL || "https://juprleagues.com").replace(/\/$/, "");
}

export default function sitemap(): MetadataRoute.Sitemap {
  const origin = baseUrl();
  const now = new Date();
  return routes.map((path) => ({
    url: `${origin}${path}`,
    lastModified: now,
    changeFrequency: "weekly",
    priority: path === "/" ? 1 : 0.7
  }));
}
