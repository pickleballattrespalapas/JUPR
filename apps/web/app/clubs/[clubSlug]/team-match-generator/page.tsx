import Link from "next/link";
import TeamMatchGenerator from "./TeamMatchGenerator";

type Props = { params: { clubSlug: string } };

export default function TeamMatchGeneratorPage({ params }: Props) {
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Public play tools
      </p>
      <h1 style={{ marginTop: 0 }}>Team Match Generator</h1>
      <p style={{ color: "#334155", maxWidth: "900px" }}>
        Build four-player teams with two women and two men, then open a dedicated live session page that guides the event through one team matchup and one score submission at a time.
      </p>
      <p style={{ color: "#475569", maxWidth: "900px" }}>
        Women&apos;s Doubles and Men&apos;s Doubles are played first. Each team then chooses its mixed pairings by player name. A 2–2 regulation tie automatically advances to a DreamBreaker. This public tool is unrated and does not publish official matches.
      </p>
      <p>
        <Link href={`/clubs/${params.clubSlug}/play`}>← Back to Play tools</Link>
      </p>
      <TeamMatchGenerator clubId={params.clubSlug} />
    </section>
  );
}
