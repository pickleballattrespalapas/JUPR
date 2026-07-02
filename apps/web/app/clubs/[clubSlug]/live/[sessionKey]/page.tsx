import Link from "next/link";
import { getClubLiveSession, type PublicLiveMatch } from "@/lib/api";

type LiveSessionPageProps = {
  params: { clubSlug: string; sessionKey: string };
};

const thStyle = { textAlign: "left" as const, borderBottom: "1px solid #cbd5e1", padding: "0.5rem", whiteSpace: "nowrap" as const };
const tdStyle = { borderBottom: "1px solid #e2e8f0", padding: "0.5rem", whiteSpace: "nowrap" as const };
const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

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

function teamLabel(names: string[]): string {
  return names.filter(Boolean).join(" / ") || "TBD";
}

function scoreLabel(match: PublicLiveMatch): string {
  const scoreA = match.score_a ?? null;
  const scoreB = match.score_b ?? null;
  if (scoreA == null && scoreB == null) return "—";
  return `${scoreA ?? 0}–${scoreB ?? 0}`;
}

function MatchCard({ match }: { match: PublicLiveMatch }) {
  return (
    <article style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.8rem", background: "#f8fafc" }}>
      <p style={{ margin: "0 0 0.35rem", color: "#64748b", fontSize: "0.85rem" }}>{match.label}</p>
      <div style={{ display: "grid", gridTemplateColumns: "1fr auto 1fr", alignItems: "center", gap: "0.75rem" }}>
        <strong>{teamLabel(match.team_a)}</strong>
        <span style={{ fontWeight: 800, fontSize: "1.1rem" }}>{scoreLabel(match)}</span>
        <strong style={{ textAlign: "right" }}>{teamLabel(match.team_b)}</strong>
      </div>
      {match.winner ? <p style={{ margin: "0.4rem 0 0", color: "#166534", fontSize: "0.9rem" }}>Winner: {match.winner}</p> : null}
    </article>
  );
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
      <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", flexWrap: "wrap", alignItems: "flex-start", marginBottom: "1rem" }}>
        <div>
          <h1 style={{ margin: "0 0 0.35rem", fontSize: "2.2rem", lineHeight: 1.1 }}>{session.title}</h1>
          <p style={{ margin: 0, color: "#475569" }}>
            {eventTypeLabel(session.event_type)}
            {session.current_round ? ` · Current round ${session.current_round}` : ""}
          </p>
          <p style={{ margin: "0.35rem 0 0", color: "#64748b", fontSize: "0.9rem" }}>
            Last updated {formatTimestamp(session.updated_at ?? session.last_seen_at)}
          </p>
        </div>
        <span style={{ border: "1px solid #bfdbfe", borderRadius: "999px", padding: "0.25rem 0.75rem", color: "#1d4ed8", background: "#eff6ff", fontSize: "0.85rem", fontWeight: 800 }}>
          {session.status}
        </span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1.5fr) minmax(280px, 1fr)", gap: "1rem", alignItems: "start" }}>
        <div style={{ display: "grid", gap: "1rem" }}>
          {session.rounds.length === 0 ? (
            <div style={cardStyle}>
              <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>No public match state yet</h2>
              <p style={{ color: "#475569" }}>This session exists, but the organizer has not created the event schedule yet.</p>
            </div>
          ) : null}

          {session.rounds.map((round) => (
            <section key={round.number} style={cardStyle}>
              <h2 style={{ marginTop: 0, fontSize: "1.2rem" }}>Round {round.number}</h2>
              {round.courts && round.courts.length > 0 ? (
                <div style={{ display: "grid", gap: "0.75rem" }}>
                  {round.courts.map((court) => (
                    <div key={court.court_number}>
                      <h3 style={{ fontSize: "1rem", marginBottom: "0.5rem" }}>Court {court.court_number}</h3>
                      <div style={{ display: "grid", gap: "0.5rem" }}>
                        {court.matches.map((match) => <MatchCard key={match.id} match={match} />)}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ display: "grid", gap: "0.5rem" }}>
                  {round.matches.map((match) => <MatchCard key={match.id} match={match} />)}
                </div>
              )}
            </section>
          ))}
        </div>

        <aside style={{ display: "grid", gap: "1rem" }}>
          <section style={cardStyle}>
            <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Standings</h2>
            {session.standings.length === 0 ? <p style={{ color: "#475569" }}>No standings yet.</p> : null}
            {session.standings.length > 0 ? (
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem" }}>
                  <thead>
                    <tr>
                      <th style={thStyle}>Rank</th>
                      <th style={thStyle}>Player</th>
                      <th style={thStyle}>W/L</th>
                      <th style={thStyle}>Diff</th>
                    </tr>
                  </thead>
                  <tbody>
                    {session.standings.map((row, index) => (
                      <tr key={`${row.participantId ?? row.name ?? index}`}>
                        <td style={tdStyle}>{String(row.rank ?? index + 1)}</td>
                        <td style={tdStyle}>{String(row.name ?? "—")}</td>
                        <td style={tdStyle}>{String(row.wins ?? 0)}/{String(row.losses ?? 0)}</td>
                        <td style={tdStyle}>{String(row.differential ?? "—")}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
          </section>

          {session.bracket ? (
            <section style={cardStyle}>
              <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Bracket</h2>
              <p style={{ color: "#475569" }}>Champion: {session.bracket.champion || "Pending"}</p>
            </section>
          ) : null}
        </aside>
      </div>

      <p style={{ marginTop: "1rem" }}><Link href={`/clubs/${clubSlug}/live`}>Back to live sessions</Link></p>
    </section>
  );
}
