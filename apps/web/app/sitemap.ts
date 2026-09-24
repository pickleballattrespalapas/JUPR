import type { MetadataRoute } from "next";
import { getPublicSite } from "@/lib/clubSiteServer";
import { publicClubPage, clubPageHref } from "@/lib/clubSite";

const publicRoutes = [
  "/",
  "/site-map",
  "/clubs/tres-palapas",
  "/clubs/tres-palapas/leagues",
  "/clubs/tres-palapas/tournaments",
  "/clubs/tres-palapas/live",
  "/clubs/tres-palapas/leaderboards",
  "/clubs/tres-palapas/match-explorer",
  "/clubs/tres-palapas/league-results",
  "/clubs/tres-palapas/team-leagues",
  "/clubs/tres-palapas/badge-codex",
  "/clubs/tres-palapas/challenge-ladder",
  "/clubs/tres-palapas/weekly-recap",
  "/clubs/tres-palapas/tournament-registration",
  "/clubs/tres-palapas/tournament-roster",
  "/clubs/tres-palapas/tournament-partner-board",
  "/clubs/tres-palapas/tournament-team-results",
  "/clubs/tres-palapas/players",
  "/clubs/tres-palapas/matches",
  "/how-ratings-work",
  "/faq",
  "/privacy",
  "/terms",
  "/support",
  "/contact",
  "/data-corrections",
  "/profile-privacy",
  "/verified-updates",
  "/email-preferences"
];

function baseUrl(): string {
  return (
    process.env.NEXT_PUBLIC_JUPR_WEB_BASE_URL ||
    process.env.JUPR_WEB_BASE_URL ||
    "https://pickleballclubsandwich.com"
  ).replace(/\/$/, "");
}

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const site = await getPublicSite("tres-palapas").catch(() => null);
  const canIndex = (path: string) => !path.startsWith("/clubs/tres-palapas") || (!!site && site.document.visibility === "listed" && !!publicClubPage(site.document, site.slug, path));
  const custom = site?.document.pages.filter(p => p.slug !== "home" && p.in_navigation).map(p => clubPageHref(site.slug, p.slug)) || [];
  const origin = baseUrl();
  const now = new Date();
  return [...publicRoutes, ...custom].filter(canIndex).map((path) => ({
    url: `${origin}${path}`,
    lastModified: now,
    changeFrequency: "weekly",
    priority: path === "/" ? 1 : 0.7
  }));
}
