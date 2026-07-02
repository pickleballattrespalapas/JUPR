import Link from "next/link";
import { getClubLiveSession } from "@/lib/api";
import LiveSessionRunner from "./LiveSessionRunner";

type LiveSessionPageProps = {
  params: { clubSlug: string; sessionKey: string };
  searchParams?: { edit?: string | string[] };
};

function apiBase(): string | null {
  return process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || process.env.JUPR_API_BASE_URL || null;
}

function searchParamText(value: string | string[] | undefined): string {
  if (Array.isArray(value)) return value[0] || "";
  return value || "";
}

export default async function ClubLiveSessionPage({ params, searchParams }: LiveSessionPageProps) {
  const { clubSlug, sessionKey } = params;
  const { data, error } = await getClubLiveSession(clubSlug, sessionKey);
  const clubName = data?.club?.name ?? clubSlug;
  const session = data?.session;

  if (error || !session) {
    return (
      <section style={{ maxWidth: "760px" }}>
        <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
          JUPR Live
        </p>
        <h1>Live session unavailable</h1>
        <p style={{ color: "#b91c1c" }}>We could not load this live session. {error}</p>
        <p><Link href={`/clubs/${clubSlug}/live`}>Back to live sessions</Link></p>
      </section>
    );
  }

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        {clubName} · JUPR Live
      </p>
      <LiveSessionRunner
        apiBase={apiBase()}
        clubSlug={clubSlug}
        initialSession={session}
        editToken={searchParamText(searchParams?.edit)}
      />
    </section>
  );
}
