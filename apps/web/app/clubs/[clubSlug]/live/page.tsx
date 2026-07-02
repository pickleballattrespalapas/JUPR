import Link from "next/link";

type LivePageProps = {
  params: { clubSlug: string };
};

export default function ClubLivePage({ params }: LivePageProps) {
  const { clubSlug } = params;

  return (
    <section style={{ maxWidth: "760px" }}>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        JUPR Live
      </p>
      <h1 style={{ marginTop: 0 }}>Public live scoreboards are coming here</h1>
      <p style={{ color: "#334155" }}>
        The Streamlit admin now has durable live-session recovery. The next PR wires those sessions into public website scoreboards for active events.
      </p>
      <p>
        For now, continue to the <Link href={`/clubs/${clubSlug}/leaderboards`}>public leaderboards</Link>.
      </p>
    </section>
  );
}
