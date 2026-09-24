import { cache } from "react";
import type { PublicSite } from "./clubSite";

export async function publicSiteFetch<T>(path: string): Promise<T | null> {
  const base =
    process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL;
  if (!base) throw new Error("Public websites are temporarily unavailable.");
  const response = await fetch(`${base.replace(/\/$/, "")}${path}`, {
    cache: "no-store",
    signal: AbortSignal.timeout(10000),
  });
  if (response.status === 404) return null;
  if (!response.ok)
    throw new Error("Public websites are temporarily unavailable.");
  return response.json();
}
export const getPublicSite = cache((slug: string) =>
  publicSiteFetch<PublicSite>(`/public/clubs/${encodeURIComponent(slug)}/site`),
);
