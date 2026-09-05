import Link from "next/link";
import { loadVerifiedUpdatePlayers, verifiedUpdatesApiBaseUrl } from "@/lib/verifiedUpdatesApi";
import VerifiedUpdatesRequestForm from "./VerifiedUpdatesRequestForm";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

export default async function VerifiedUpdatesPageContent({ clubSlug, initialPlayerId }: { clubSlug: string; initialPlayerId?: string | null }) {
  const { data, error } = await loadVerifiedUpdatePlayers(clubSlug);
  const clubName = data?.club?.name || clubSlug;

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        {clubName} · Verified player updates
      </p>
      <h1 style={{ marginTop: 0 }}>Get updates for a player</h1>
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        Get an email when club staff publish new league, round-robin, or tournament results for this player.
      </p>
      {error ? <article style={{ ...cardStyle, background: "#fff7ed", color: "#9a3412" }}>We couldn&apos;t load player profiles right now. Please try again later.</article> : null}
      {data?.players?.length ? (
        <VerifiedUpdatesRequestForm apiBase={verifiedUpdatesApiBaseUrl()} clubSlug={clubSlug} players={data.players} initialPlayerId={initialPlayerId} />
      ) : !error ? (
        <article style={cardStyle}>No player profiles are available yet.</article>
      ) : null}
      <p style={{ marginTop: "1rem" }}><Link href={`/clubs/${encodeURIComponent(clubSlug)}/players`}>Back to player search</Link> · <Link href="/email-preferences">Email preferences</Link></p>
    </section>
  );
}
