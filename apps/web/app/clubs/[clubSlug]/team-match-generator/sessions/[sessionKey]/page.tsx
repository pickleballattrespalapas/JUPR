import Link from "next/link";
import TeamMatchSessionRunner from "./TeamMatchSessionRunner";

type Props = {
  params: {
    clubSlug: string;
    sessionKey: string;
  };
};

export default function TeamMatchSessionPage({ params }: Props) {
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Team Match Generator
      </p>
      <p style={{ marginTop: 0 }}>
        <Link href={`/clubs/${params.clubSlug}/team-match-generator`}>← Back to Team Match setup</Link>
      </p>
      <TeamMatchSessionRunner clubId={params.clubSlug} sessionKey={params.sessionKey} />
    </section>
  );
}
