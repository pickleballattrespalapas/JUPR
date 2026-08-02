import Link from "next/link";
import { getClubLiveSession } from "@/lib/api";
import LiveSessionRunner from "./LiveSessionRunner";

type LiveSessionPageProps = {
  params: { clubSlug: string; sessionKey: string };
};

function apiBase(): string {
  return "/api";
}

export default async function ClubLiveSessionPage({ params }: LiveSessionPageProps) {
  const { clubSlug, sessionKey } = params;
  const { data, error } = await getClubLiveSession(clubSlug, sessionKey);
  const clubName = data?.club?.name ?? clubSlug;
  const session = data?.session;

  if (error || !session) {
    return (
      <section style={{ maxWidth: "760px" }}>
        <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
          Play Generators
        </p>
        <h1>Play session unavailable</h1>
        <p style={{ color: "#b91c1c" }}>We could not load this live session. {error}</p>
        <p><Link href={`/clubs/${clubSlug}/live`}>Back to play sessions</Link></p>
      </section>
    );
  }

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        {clubName} · Play Generators
      </p>
      <LiveSessionRunner
        apiBase={apiBase()}
        clubSlug={clubSlug}
        initialSession={session}
      />
    </section>
  );
}
