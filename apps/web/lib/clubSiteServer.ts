import { cache } from "react";
import type { DirectoryClub, PublicSite } from "./clubSite";

export async function publicSiteFetch<T>(path: string): Promise<T | null> {
  const base =
    process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL;
  if (!base) throw new Error("Public websites are temporarily unavailable.");
  const response = await fetch(`${base.replace(/\/$/, "")}${path}`, {
    cache: "no-store",
  });
  if (response.status === 404) return null;
  if (!response.ok)
    throw new Error("Public websites are temporarily unavailable.");
  return response.json();
}
export const getPublicSite = cache((slug: string) =>
  publicSiteFetch<PublicSite>(`/public/clubs/${encodeURIComponent(slug)}/site`),
);
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
