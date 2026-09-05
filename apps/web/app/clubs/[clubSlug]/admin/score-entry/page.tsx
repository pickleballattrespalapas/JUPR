import Link from "next/link";
import { getClubPlayers } from "@/lib/api";
import { getAdminScoreEntryStatus, isNextAdminScoreEntryEnabled } from "@/lib/scoreEntry";
import ScoreEntryForm from "./ScoreEntryForm";

type ScoreEntryPageProps = {
  params: { clubSlug: string };
};

function apiBase(): string | null {
  return process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || process.env.JUPR_API_BASE_URL || null;
}

export default async function ScoreEntryPage({ params }: ScoreEntryPageProps) {
  const { clubSlug } = params;
  if (!isNextAdminScoreEntryEnabled()) {
    return (
      <section>
        <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
          Staff score entry
        </p>
        <h1 style={{ marginTop: 0 }}>Score entry isn’t available here</h1>
        <p style={{ color: "#475569" }}>
          Use one of the staff options below to enter scores.
        </p>
        <p><Link href="/admin/match-uploader">Open Match Uploader</Link> · <a href={process.env.JUPR_STREAMLIT_FALLBACK_URL || "https://juprtrespalapas.streamlit.app"}>Open backup score entry</a></p>
        <p><Link href="/admin">Return to staff home</Link></p>
      </section>
    );
  }

  const { data, error } = await getClubPlayers(clubSlug);
  const club = data?.club;
  const players = data?.players ?? [];
  const clubId = club?.id ?? clubSlug;
  const apiOrigin = apiBase();
  const readiness = await getAdminScoreEntryStatus(apiOrigin, String(clubId));

  if (!readiness.data?.ready) {
    const fallback = readiness.data?.fallback;
    return (
      <section>
        <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Staff score entry</p>
        <h1 style={{ marginTop: 0 }}>Score entry isn’t ready here</h1>
        <p style={{ color: "#475569" }}>Use one of the staff options below to enter scores.</p>
        {readiness.error ? <p style={{ color: "#b91c1c" }}>{readiness.error}</p> : null}
        <p><Link href={fallback?.match_uploader_route || "/admin/match-uploader"}>Open Match Uploader</Link> · <a href={fallback?.streamlit_url || process.env.JUPR_STREAMLIT_FALLBACK_URL || "https://juprtrespalapas.streamlit.app"}>Open backup score entry</a></p>
        <p><Link href="/admin">Return to staff home</Link></p>
      </section>
    );
  }

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Staff score entry
      </p>
      <h1 style={{ marginTop: 0 }}>{club?.name ?? clubSlug} score entry</h1>
      <p style={{ color: "#475569" }}>
        Enter one match at a time. Saved scores update ratings, leaderboards, and player profiles.
      </p>
      {error ? <p style={{ color: "#b91c1c" }}>Player lookup is unavailable. {error}</p> : null}
      {!error && players.length === 0 ? <p>No players are available for score entry yet.</p> : null}
      {players.length > 0 ? <ScoreEntryForm apiBase={apiOrigin} clubId={clubId} clubSlug={clubSlug} players={players} /> : null}
      <p style={{ marginTop: "1rem" }}>
        <Link href={`/clubs/${clubSlug}/matches`}>View match history</Link>
      </p>
    </section>
  );
}
