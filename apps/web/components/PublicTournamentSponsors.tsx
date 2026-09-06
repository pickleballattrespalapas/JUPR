import "server-only";
import { cache } from "react";
import TournamentSponsorDisplay from "./TournamentSponsorDisplay";
import type { TournamentSponsor } from "@/lib/tournamentSponsors";

const loadSponsors = cache(async (clubSlug: string, tournamentId: string): Promise<TournamentSponsor[]> => {
  const base = process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL;
  if (!base) return [];
  try {
    const response = await fetch(`${base.replace(/\/$/, "")}/clubs/${encodeURIComponent(clubSlug)}/tournaments/${encodeURIComponent(tournamentId)}/sponsors`, { cache: "no-store" });
    if (!response.ok) return [];
    const data = await response.json();
    return data.tournament_id === tournamentId && Array.isArray(data.sponsors) ? data.sponsors : [];
  } catch { return []; }
});

export default async function PublicTournamentSponsors({ clubSlug, tournamentId, placement, title, headingLevel }: { title?: string; headingLevel?: "h1" | "h2"; clubSlug: string; tournamentId?: string | null; placement: "header" | "footer" }) {
  return <TournamentSponsorDisplay sponsors={tournamentId ? await loadSponsors(clubSlug, tournamentId) : []} placement={placement} title={title} headingLevel={headingLevel} />;
}
