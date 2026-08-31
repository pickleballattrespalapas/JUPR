import Link from "next/link";

type Props = { params: { clubSlug: string } };

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white",
  color: "#0f172a",
  textDecoration: "none"
};

export default function PublicPlayHub({ params }: Props) {
  const base = `/clubs/${params.clubSlug}`;
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Play tools
      </p>
      <h1 style={{ marginTop: 0 }}>Create and run play</h1>
      <p style={{ color: "#334155", maxWidth: "860px" }}>
        Build a printable schedule, share a view-only link, and run one round at a time. Public sessions are unrated; official match publishing remains a staff action.
      </p>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1rem" }}>
        <Link href={`${base}/round-robin-generator`} style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Round-Robin Generator</h2>
          <p style={{ color: "#475569" }}>
            Singles, doubles, or a balanced mix of both. Preview every planned round and bye, export the schedule, then run one round at a time.
          </p>
          <strong>Open Round-Robin Generator →</strong>
        </Link>
        <Link href={`${base}/ladder-generator`} style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Ladder Generator</h2>
          <p style={{ color: "#475569" }}>
            Preview Round 1, record results, and generate later court assignments adaptively from play.
          </p>
          <strong>Open Ladder Generator →</strong>
        </Link>
        <Link href={`${base}/team-match-generator`} style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Team Match Generator</h2>
          <p style={{ color: "#475569" }}>
            Build four-player teams, schedule every team matchup, and run women&apos;s, men&apos;s, and mixed doubles games with live team standings.
          </p>
          <strong>Open Team Match Generator →</strong>
        </Link>
      </div>
    </section>
  );
}
