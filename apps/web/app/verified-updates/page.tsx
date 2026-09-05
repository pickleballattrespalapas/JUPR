import Link from "next/link";
import VerifiedUpdatesPageContent from "./VerifiedUpdatesPageContent";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

export default async function VerifiedUpdatesPage({ searchParams }: { searchParams?: { club?: string; club_slug?: string; player_id?: string; pid?: string } }) {
  const clubSlug = String(searchParams?.club || searchParams?.club_slug || "").trim();
  const initialPlayerId = searchParams?.player_id || searchParams?.pid || null;

  if (clubSlug) {
    return <VerifiedUpdatesPageContent clubSlug={clubSlug} initialPlayerId={initialPlayerId} />;
  }

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Verified player updates
      </p>
      <h1 style={{ marginTop: 0 }}>Choose a club profile first</h1>
      <article style={cardStyle}>
        <p style={{ marginTop: 0 }}>Open a player profile and choose <strong>Request verified updates</strong>. We&apos;ll open the form for the right club and player.</p>
        <p style={{ marginBottom: 0 }}><Link href="/site-map">Find your club</Link> · <Link href="/email-preferences">Email preferences</Link></p>
      </article>
    </section>
  );
}
