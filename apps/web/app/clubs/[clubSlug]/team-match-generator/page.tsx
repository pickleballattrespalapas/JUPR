import Link from "next/link";
import TeamMatchGenerator from "./TeamMatchGenerator";

type Props = { params: { clubSlug: string } };

export default function TeamMatchGeneratorPage({ params }: Props) {
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Play tools
      </p>
      <h1 style={{ marginTop: 0 }}>Team Match Generator</h1>
      <p style={{ color: "#334155", maxWidth: "900px" }}>
        Set up four-player teams, then record each matchup one game at a time.
      </p>
      <p style={{ color: "#475569", maxWidth: "900px" }}>
        Play Women&apos;s Doubles and Men&apos;s Doubles first, then choose each team&apos;s two mixed pairs. A 2–2 tie goes to a DreamBreaker.
      </p>
      <p style={{ color: "#475569", maxWidth: "900px" }}>
        Scores entered here won&apos;t affect player ratings.
      </p>
      <p>
        <Link href={`/clubs/${params.clubSlug}/play`}>← Back to Play tools</Link>
      </p>
      <TeamMatchGenerator clubId={params.clubSlug} />
    </section>
  );
}
