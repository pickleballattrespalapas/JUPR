"use client";
import { useEffect, useState } from "react";
import TournamentSponsorDisplay from "./TournamentSponsorDisplay";
import type { TournamentSponsor } from "@/lib/tournamentSponsors";

// Token-based invitation pages resolve their tournament in the browser.
export default function PublicTournamentSponsorsClient({ clubSlug, tournamentId, placement, title, headingLevel }: { title?: string; headingLevel?: "h1" | "h2"; clubSlug: string; tournamentId?: string; placement: "header" | "footer" }) {
  const [loaded, setLoaded] = useState<{ id: string; sponsors: TournamentSponsor[] } | null>(null);
  useEffect(() => {
    const base = process.env.NEXT_PUBLIC_JUPR_API_BASE_URL;
    if (!base || !tournamentId) return;
    const controller = new AbortController();
    void fetch(`${base.replace(/\/$/, "")}/clubs/${encodeURIComponent(clubSlug)}/tournaments/${encodeURIComponent(tournamentId)}/sponsors`, { signal: controller.signal, cache: "no-store", referrerPolicy: "no-referrer" })
      .then(response => response.ok ? response.json() : null)
      .then(data => { if (!controller.signal.aborted && data?.tournament_id === tournamentId && Array.isArray(data.sponsors)) setLoaded({ id: `${clubSlug}:${tournamentId}`, sponsors: data.sponsors }); })
      .catch(() => {});
    return () => controller.abort();
  }, [clubSlug, tournamentId]);
  return <TournamentSponsorDisplay sponsors={loaded?.id === `${clubSlug}:${tournamentId}` ? loaded.sponsors : []} placement={placement} title={title} headingLevel={headingLevel} />;
}
