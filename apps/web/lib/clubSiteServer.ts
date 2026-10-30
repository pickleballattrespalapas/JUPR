import { cache } from "react";
import type { DirectoryClub, PublicSite, PublicSiteShell } from "./clubSite";

export async function publicSiteFetch<T>(path: string): Promise<T | null> {
  const base =
    process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL;
  if (!base) throw new Error("Public websites are temporarily unavailable.");
  const response = await fetch(`${base.replace(/\/$/, "")}${path}`, {
    cache: "no-store",
    signal: AbortSignal.timeout(10_000),
  });
  if (response.status === 404) return null;
  if (!response.ok)
    throw new Error("Public websites are temporarily unavailable.");
  return response.json();
}
export const getPublicSite = cache((slug: string) =>
  publicSiteFetch<PublicSite>(`/public/clubs/${encodeURIComponent(slug)}/site`),
);
// This is a request-local dedupe, not a stale publication/permission cache.
// Full custom page content and inline image bytes are not layout dependencies.
export const getPublicSiteShell = cache(async (slug: string) => {
  const site = await publicSiteFetch<PublicSiteShell>(`/public/clubs/${encodeURIComponent(slug)}/site/shell`);
  if (site?.document.logo_url.startsWith("/public/")) {
    const base = process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || "";
    return { ...site, document: { ...site.document, logo_url: `${base.replace(/\/$/, "")}${site.document.logo_url}` } };
  }
  return site;
});
export type Directory = {
  clubs: DirectoryClub[];
  total: number;
  offset: number;
  limit: number;
};
export const getClubDirectory = (q = "", offset = 0) =>
  publicSiteFetch<Directory>(
    `/public/clubs?q=${encodeURIComponent(q)}&offset=${offset}&limit=100`,
  );
