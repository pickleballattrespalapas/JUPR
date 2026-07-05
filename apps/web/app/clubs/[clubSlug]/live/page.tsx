import Link from "next/link";
import { getClubLiveSessions, getClubPlayers } from "@/lib/api";
import PublicLiveCreator from "./PublicLiveCreator";

type LivePageProps = {
  params: { clubSlug: string };
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

function apiBase(): string {
  return "/api";
}

function formatTimestamp(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("en-US", { dateStyle: "medium", timeStyle: "short" });
}

function eventTypeLabel(value?: string | null): string {
  const normalized = String(value || "").replace(/_/g, " ").trim();
  return normalized ? normalized.replace(/\b\w/g, (char) => char.toUpperCase()) : "Live Event";
}

export default async function ClubLivePage({ params }: LivePageProps) {
  const { clubSlug } = params;
  const [liveResult, playersResult] = await Promise.all([
    getClubLiveSessions(clubSlug),
    getClubPlayers(clubSlug)
  ]);
  const { data, error } = liveResult;
  const clubName = data?.club?.name ?? playersResult.data?.club?.name ?? clubSlug;
  const sessions = data?.sessions ?? [];
  const players = playersResult.data?.players ?? [];

  return (
    <section>
      <div style={{ marginBottom: "1.25rem" }}>
        <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
          Live Events
        </p>
        <h1 style={{ margin: "0 0 0.5rem", fontSize: "2.2rem", lineHeight: 1.1 }}>{clubName} live sessions</h1>
        <p style={{ color: "#334155", marginTop: 0, maxWidth: "900px" }}>
          Start a public JUPR Live quick session, enter browser-only scores, and share the live scoreboard. Official rated workflows remain separate in JUPR Live Admin.
        </p>
      </div>

      <PublicLiveCreator apiBase={apiBase()} clubSlug={clubSlug} players={players} />

      {error ? (
        <p style={{ color: "#b91c1c" }}>Live sessions are temporarily unavailable. {error}</p>
      ) : null}
      {playersResult.error ? <p style={{ color: "#b45309" }}>Current-player picker unavailable. {playersResult.error}</p> : null}

      {!error && sessions.length === 0 ? (
        <div style={cardStyle}>
          <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>No shared live sessions right now</h2>
          <p style={{ color: "#475569" }}>
            Create a public live event above, or open a shared session link from an organizer.
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
