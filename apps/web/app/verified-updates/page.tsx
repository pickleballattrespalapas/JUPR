import Link from "next/link";
import { loadVerifiedUpdatePlayers, verifiedUpdatesApiBaseUrl } from "@/lib/verifiedUpdatesApi";
import VerifiedUpdatesRequestForm from "./VerifiedUpdatesRequestForm";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

export default async function VerifiedUpdatesPage({ searchParams }: { searchParams?: { player_id?: string; pid?: string } }) {
  const clubSlug = "tres-palapas";
  const { data, error } = await loadVerifiedUpdatePlayers(clubSlug);
  const initialPlayerId = searchParams?.player_id || searchParams?.pid || null;

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Verified Updates Request
      </p>
      <h1 style={{ marginTop: 0 }}>Subscribe to verified player updates</h1>
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        Request admin-approved update emails for a player profile. These are sent after completed batch uploads such as league sessions, round robins, or tournament publishing.
      </p>
      {error ? <article style={{ ...cardStyle, background: "#fff7ed", color: "#9a3412" }}>Unable to load player options. {error}</article> : null}
      {data?.players?.length ? (
        <VerifiedUpdatesRequestForm apiBase={verifiedUpdatesApiBaseUrl()} clubSlug={clubSlug} players={data.players} initialPlayerId={initialPlayerId} />
      ) : !error ? (
        <article style={cardStyle}>No player options are available yet.</article>
      ) : null}
      <p style={{ marginTop: "1rem" }}><Link href="/clubs/tres-palapas/players">Back to player search</Link> · <Link href="/email-preferences">Email preferences</Link></p>
    </section>
  );
}
