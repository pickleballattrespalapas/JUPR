import Link from "next/link";
import { getClubLiveSessions } from "@/lib/api";

type LivePageProps = {
  params: { clubSlug: string };
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

function formatTimestamp(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("en-US", { dateStyle: "medium", timeStyle: "short" });
}

function eventTypeLabel(value?: string | null): string {
  const normalized = String(value || "").replace(/_/g, " ").trim();
  return normalized ? normalized.replace(/\b\w/g, (char) => char.toUpperCase()) : "JUPR Live";
}

export default async function ClubLivePage({ params }: LivePageProps) {
  const { clubSlug } = params;
  const { data, error } = await getClubLiveSessions(clubSlug);
  const clubName = data?.club?.name ?? clubSlug;
  const sessions = data?.sessions ?? [];

  return (
    <section>
      <div style={{ marginBottom: "1.25rem" }}>
        <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
          JUPR Live
        </p>
        <h1 style={{ margin: "0 0 0.5rem", fontSize: "2.2rem", lineHeight: 1.1 }}>{clubName} live sessions</h1>
        <p style={{ color: "#334155", marginTop: 0 }}>
          Follow active public JUPR Live scoreboards from the website. Admin scoring remains in Streamlit while the new web scoring workflow is built.
        </p>
      </div>

      {error ? (
        <p style={{ color: "#b91c1c" }}>Live sessions are temporarily unavailable. {error}</p>
      ) : null}

      {!error && sessions.length === 0 ? (
        <div style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>No live sessions right now</h2>
          <p style={{ color: "#475569" }}>
            When an organizer starts a durable JUPR Live session, it will appear here for public viewing.
          </p>
          <Link href={`/clubs/${clubSlug}/leaderboards`}>View leaderboards instead</Link>
        </div>
      ) : null}

      {sessions.length > 0 ? (
        <div style={{ display: "grid", gap: "1rem" }}>
          {sessions.map((session) => (
            <article key={session.session_key} style={cardStyle}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", flexWrap: "wrap" }}>
                <div>
                  <h2 style={{ margin: "0 0 0.25rem", fontSize: "1.15rem" }}>
                    <Link href={`/clubs/${clubSlug}/live/${session.session_key}`}>{session.title}</Link>
                  </h2>
                  <p style={{ margin: 0, color: "#475569" }}>
                    {eventTypeLabel(session.event_type)}
                    {session.current_round ? ` · Round ${session.current_round}` : ""}
                  </p>
                </div>
                <span style={{ alignSelf: "flex-start", border: "1px solid #bfdbfe", borderRadius: "999px", padding: "0.2rem 0.6rem", color: "#1d4ed8", background: "#eff6ff", fontSize: "0.85rem", fontWeight: 700 }}>
                  {session.status}
                </span>
              </div>
              <p style={{ marginBottom: 0, color: "#64748b", fontSize: "0.9rem" }}>
                Last updated {formatTimestamp(session.updated_at ?? session.last_seen_at)}
              </p>
            </article>
          ))}
        </div>
      ) : null}
    </section>
  );
}
