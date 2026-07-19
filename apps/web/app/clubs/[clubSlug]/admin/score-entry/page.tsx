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
          Admin score entry
        </p>
        <h1 style={{ marginTop: 0 }}>Score entry is disabled</h1>
        <p style={{ color: "#475569" }}>
          This guarded workflow is available only in explicitly enabled staff environments.
        </p>
        <p><Link href="/admin/match-uploader">Use guarded Match Uploader</Link> · <a href={process.env.JUPR_STREAMLIT_FALLBACK_URL || "https://juprtrespalapas.streamlit.app"}>Open Streamlit fallback</a></p>
        <p><Link href="/admin">Return to the operations cockpit</Link></p>
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
        <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Admin score entry</p>
        <h1 style={{ marginTop: 0 }}>Score entry is in fallback mode</h1>
        <p style={{ color: "#475569" }}>The browser flag is on, but FastAPI has not confirmed both its write flag and server-only Supabase service role. The write form remains hidden.</p>
        {readiness.error ? <p style={{ color: "#b91c1c" }}>{readiness.error}</p> : null}
        {readiness.data?.warnings?.length ? <ul style={{ color: "#92400e" }}>{readiness.data.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
        <p><Link href={fallback?.match_uploader_route || "/admin/match-uploader"}>Use guarded Match Uploader</Link> · <a href={fallback?.streamlit_url || process.env.JUPR_STREAMLIT_FALLBACK_URL || "https://juprtrespalapas.streamlit.app"}>Open Streamlit fallback</a></p>
        <p><Link href="/admin">Return to the operations cockpit</Link></p>
      </section>
    );
  }

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Admin score entry MVP
      </p>
      <h1 style={{ marginTop: 0 }}>{club?.name ?? clubSlug} score entry</h1>
      <p style={{ color: "#475569" }}>
        This is the first Next/Vercel score-entry surface for the core JUPR loop: save a score, update ratings, refresh leaderboards, and reflect the result on player profiles.
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
