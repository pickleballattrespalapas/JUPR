import Link from "next/link";
import { getClubTournamentRoster } from "@/lib/tournamentRegistrationApi";

type TournamentPartnerBoardPageProps = {
  params: { clubSlug: string };
  searchParams?: { tournament?: string; tournament_id?: string };
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

function queryFor(tournamentId?: string | null, registrationSlug?: string | null): string {
  const query = new URLSearchParams();
  if (registrationSlug) query.set("tournament", registrationSlug);
  else if (tournamentId) query.set("tournament_id", tournamentId);
  const text = query.toString();
  return text ? `?${text}` : "";
}

export default async function TournamentPartnerBoardPage({ params, searchParams }: TournamentPartnerBoardPageProps) {
  const { clubSlug } = params;
  const { data, error } = await getClubTournamentRoster(clubSlug, {
    registrationSlug: searchParams?.tournament ?? null,
    tournamentId: searchParams?.tournament_id ?? null
  });

  const tournament = data?.tournament;
  const settings = data?.settings;
  const partnerEntries = data?.roster?.players_needing_partners ?? [];
  const query = queryFor(tournament?.id, settings?.registration_slug);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Tournament Partner Board
      </p>
      <h1 style={{ marginTop: 0 }}>{tournament?.name ?? "Partner board"}</h1>
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        Public read-only partner board for players who registered as needing a partner. Contact details and request/accept actions are intentionally not exposed in this slice.
      </p>

      {error ? <p style={{ color: "#b91c1c" }}>Partner board is temporarily unavailable. {error}</p> : null}
      {data?.setup_error ? <p style={{ color: "#b91c1c" }}>{data.setup_error}</p> : null}
      {!error && data && !tournament ? <p>{data.empty_reason || "No tournament partner board is currently published."}</p> : null}

      {data?.tournaments?.length ? (
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1rem" }}>
          {data.tournaments.map((choice) => {
            const slug = choice.settings.registration_slug;
            const active = choice.tournament.id === tournament?.id;
            const href = slug ? `/clubs/${clubSlug}/tournament-partner-board?tournament=${encodeURIComponent(slug)}` : `/clubs/${clubSlug}/tournament-partner-board?tournament_id=${encodeURIComponent(choice.tournament.id)}`;
            return (
              <Link key={choice.tournament.id} href={href} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: active ? "#dbeafe" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                {choice.tournament.name}
              </Link>
            );
          })}
        </div>
      ) : null}

      {tournament ? (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
          <article style={cardStyle}><strong>Players looking</strong><br />{partnerEntries.length}</article>
          <article style={cardStyle}><strong>Registrations</strong><br />{data?.summary?.total_registrations ?? 0}</article>
          <article style={cardStyle}><strong>Roster players</strong><br />{data?.summary?.total_players ?? 0}</article>
          <article style={cardStyle}><strong>Board enabled</strong><br />{settings?.partner_board_enabled ? "Yes" : "No"}</article>
        </div>
      ) : null}

      {tournament ? (
        <p style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
          <Link href={`/clubs/${clubSlug}/tournament-registration${query}`}>Open registration</Link>
          <Link href={`/clubs/${clubSlug}/tournament-roster${query}`}>Open roster</Link>
        </p>
      ) : null}

      {tournament && !settings?.partner_board_enabled ? (
        <article style={{ ...cardStyle, marginBottom: "1rem", background: "#f8fafc" }}>
          <h2 style={{ marginTop: 0 }}>Partner board is disabled</h2>
          <p style={{ color: "#475569" }}>The public partner board is not enabled for this tournament.</p>
        </article>
      ) : null}

      {partnerEntries.length ? (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "1rem" }}>
          {partnerEntries.map((entry) => (
            <article key={`${entry.selection_id}-${entry.player_name}`} style={cardStyle}>
              <h2 style={{ marginTop: 0, fontSize: "1.05rem" }}>{entry.player_name}</h2>
              <p style={{ color: "#475569" }}>{entry.event_family} · {entry.division}</p>
              <dl style={{ display: "grid", gap: "0.35rem", margin: 0 }}>
                {entry.skill ? <><dt style={{ fontWeight: 800 }}>Skill</dt><dd style={{ margin: 0 }}>{entry.skill}</dd></> : null}
                {entry.age_bracket ? <><dt style={{ fontWeight: 800 }}>Age bracket</dt><dd style={{ margin: 0 }}>{entry.age_bracket}</dd></> : null}
                {entry.note ? <><dt style={{ fontWeight: 800 }}>Note</dt><dd style={{ margin: 0 }}>{entry.note}</dd></> : null}
              </dl>
            </article>
          ))}
        </div>
      ) : tournament && settings?.partner_board_enabled ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>No open partner requests</h2>
          <p style={{ color: "#475569" }}>There are no public partner-board entries for this tournament yet.</p>
        </article>
      ) : null}
    </section>
  );
}
