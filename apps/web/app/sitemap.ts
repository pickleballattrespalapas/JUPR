import type { MetadataRoute } from "next";

const publicRoutes = [
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
  "/clubs/tres-palapas/tournament-partner-board",
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

const staffRoutes = [
  "/admin",
  "/admin/guide",
  "/admin/theme-qa",
  "/admin/login",
  "/admin/reset-password",
  "/admin/match-log",
  "/admin/replay-history",
  "/admin/match-uploader",
  "/admin/players",
  "/admin/player-updates",
  "/admin/player-updates/verified-requests",
  "/admin/support-requests",
  "/admin/league-manager",
  "/admin/league-manager/live",
  "/admin/league-manager/awards",
  "/admin/league-manager/print",
  "/admin/top-players-printable",
  "/admin/tournaments",
  "/admin/tournaments/bulk",
  "/admin/tournaments/ops",
  "/admin/tournaments/status",
  "/admin/tournaments/delete-draft",
  "/admin/tournament-live",
  "/admin/weekly-recap",
  "/admin/badges",
  "/admin/moneyball",
  "/admin/jupr-live",
  "/admin/challenge-ladder",
  "/admin/match-canonical-audit",
  "/admin/tools"
];

const routes = [...publicRoutes, ...staffRoutes];

function baseUrl(): string {
  return (process.env.NEXT_PUBLIC_JUPR_WEB_BASE_URL || process.env.JUPR_WEB_BASE_URL || "https://pickleballclubsandwich.com").replace(/\/$/, "");
}

export default function sitemap(): MetadataRoute.Sitemap {
  const origin = baseUrl();
  const now = new Date();
  return routes.map((path) => ({
    url: `${origin}${path}`,
    lastModified: now,
    changeFrequency: path.startsWith("/admin") ? "monthly" : "weekly",
    priority: path === "/" ? 1 : path.startsWith("/admin") ? 0.3 : 0.7
  }));
}
