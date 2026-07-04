import Link from "next/link";
import { getClubTournamentRoster } from "@/lib/tournamentRegistrationApi";
import type { PublicTournamentRosterEntry, PublicTournamentRosterMember } from "@/lib/tournamentRegistrationApi";

type TournamentRosterPageProps = {
  params: { clubSlug: string };
  searchParams?: { tournament?: string; tournament_id?: string };
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

function dateLabel(value?: string | null): string | null {
  if (!value) return null;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value).slice(0, 10);
  return date.toISOString().slice(0, 10);
}

function queryFor(tournamentId?: string | null, registrationSlug?: string | null): string {
  const query = new URLSearchParams();
  if (registrationSlug) query.set("tournament", registrationSlug);
  else if (tournamentId) query.set("tournament_id", tournamentId);
  const text = query.toString();
  return text ? `?${text}` : "";
}

function memberLabel(member: PublicTournamentRosterMember): string {
  const extras = [member.skill ? `Skill ${member.skill}` : null, member.age_bracket || null].filter(Boolean).join(" · ");
  return extras ? `${member.display_name} (${extras})` : member.display_name;
}

function groupKey(entry: PublicTournamentRosterEntry): string {
  return [entry.event_day_label || "Day", entry.event_family || "Event", entry.division || "Division"].join(" · ");
}

export default async function TournamentRosterPage({ params, searchParams }: TournamentRosterPageProps) {
  const { clubSlug } = params;
  const { data, error } = await getClubTournamentRoster(clubSlug, {
    registrationSlug: searchParams?.tournament ?? null,
    tournamentId: searchParams?.tournament_id ?? null
  });

  const tournament = data?.tournament;
  const settings = data?.settings;
  const roster = data?.roster;
  const entries = roster?.registrations_by_event ?? [];
  const grouped = entries.reduce<Record<string, PublicTournamentRosterEntry[]>>((acc, entry) => {
    const key = groupKey(entry);
    acc[key] = [...(acc[key] || []), entry];
    return acc;
  }, {});
  const query = queryFor(tournament?.id, settings?.registration_slug);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Tournament Roster
      </p>
      <h1 style={{ marginTop: 0 }}>{tournament?.name ?? "Tournament roster"}</h1>
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        Public roster view for tournament registration. This is read-only and hides private contact fields; staff draw seeding and operations remain in admin tools.
      </p>

      {error ? <p style={{ color: "#b91c1c" }}>Tournament roster is temporarily unavailable. {error}</p> : null}
      {data?.setup_error ? <p style={{ color: "#b91c1c" }}>{data.setup_error}</p> : null}
      {!error && data && !tournament ? <p>{data.empty_reason || "No tournament roster is currently published."}</p> : null}

      {data?.tournaments?.length ? (
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1rem" }}>
          {data.tournaments.map((choice) => {
            const slug = choice.settings.registration_slug;
            const active = choice.tournament.id === tournament?.id;
            const href = slug ? `/clubs/${clubSlug}/tournament-roster?tournament=${encodeURIComponent(slug)}` : `/clubs/${clubSlug}/tournament-roster?tournament_id=${encodeURIComponent(choice.tournament.id)}`;
            return (
              <Link key={choice.tournament.id} href={href} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: active ? "#dbeafe" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                {choice.tournament.name}
              </Link>
            );
          })}
        </div>
      ) : null}

      {tournament ? (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
            <article style={cardStyle}><strong>Start</strong><br />{dateLabel(tournament.start_date) ?? "TBD"}</article>
            <article style={cardStyle}><strong>Registrations</strong><br />{data?.summary?.total_registrations ?? 0}</article>
            <article style={cardStyle}><strong>Players</strong><br />{data?.summary?.total_players ?? 0}</article>
            <article style={cardStyle}><strong>Needs partner</strong><br />{data?.summary?.players_needing_partners ?? 0}</article>
          </div>

          <p style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
            <Link href={`/clubs/${clubSlug}/tournament-registration${query}`}>Open registration</Link>
            <Link href={`/clubs/${clubSlug}/tournament-partner-board${query}`}>Partner board</Link>
          </p>
        </>
      ) : null}

      {roster?.players_needing_partners?.length ? (
        <article style={{ ...cardStyle, marginBottom: "1rem", background: "#fffbeb", borderColor: "#fde68a" }}>
          <h2 style={{ marginTop: 0 }}>Players looking for partners</h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
            {roster.players_needing_partners.map((entry) => (
              <div key={`${entry.selection_id}-${entry.player_name}`} style={{ border: "1px solid #fde68a", borderRadius: "12px", padding: "0.75rem", background: "white" }}>
                <strong>{entry.player_name}</strong>
                <div style={{ color: "#92400e" }}>{entry.event_family} · {entry.division}</div>
                {entry.note ? <p style={{ marginBottom: 0, color: "#475569" }}>{entry.note}</p> : null}
              </div>
            ))}
          </div>
        </article>
      ) : null}

      {entries.length ? (
        <div style={{ display: "grid", gap: "1rem" }}>
          {Object.entries(grouped).map(([label, rows]) => (
            <article key={label} style={cardStyle}>
              <h2 style={{ marginTop: 0 }}>{label}</h2>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead>
                    <tr>
                      <th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Entry</th>
                      <th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Players</th>
                      <th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((entry, index) => (
                      <tr key={`${entry.event_label}-${index}-${entry.source_selection_ids?.join("-")}`}>
                        <td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{entry.entry_type || "Registration"}</td>
                        <td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{entry.members.map(memberLabel).join(" / ")}</td>
                        <td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{entry.status || "Confirmed"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </article>
          ))}
        </div>
      ) : tournament ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>No roster entries yet</h2>
          <p style={{ color: "#475569" }}>Registrations will appear here after players submit tournament entries.</p>
        </article>
      ) : null}
    </section>
  );
}
