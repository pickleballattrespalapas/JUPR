import Link from "next/link";
import { getClubTournamentRegistration } from "@/lib/tournamentRegistrationApi";
import EditLinkRequestForm from "./EditLinkRequestForm";
import TournamentRegistrationForm from "./TournamentRegistrationForm";

type TournamentRegistrationPageProps = {
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

function markdownish(text?: string | null) {
  if (!text) return null;
  return text.split("\n").filter(Boolean).map((line) => <p key={line} style={{ margin: "0 0 0.5rem", color: "#475569" }}>{line.replace(/^#+\s*/, "")}</p>);
}

export default async function TournamentRegistrationPage({ params, searchParams }: TournamentRegistrationPageProps) {
  const { clubSlug } = params;
  const { data, error } = await getClubTournamentRegistration(clubSlug, {
    registrationSlug: searchParams?.tournament ?? null,
    tournamentId: searchParams?.tournament_id ?? null
  });

  const tournament = data?.tournament;
  const settings = data?.settings;
  const selectableCount = data?.events?.filter((event) => event.selectable).length ?? 0;

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Tournament Registration
      </p>
      <h1 style={{ marginTop: 0 }}>{tournament?.name ?? "Tournament registration"}</h1>
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        Public tournament registration intake for published JUPR events. Submissions create registration records only; draw seeding, score entry, ratings, and tournament operations remain staff-managed.
      </p>

      {error ? <p style={{ color: "#b91c1c" }}>Tournament registration is temporarily unavailable. {error}</p> : null}
      {data?.setup_error ? <p style={{ color: "#b91c1c" }}>{data.setup_error}</p> : null}
      {!error && data && !tournament ? <p>No tournament registration is currently published.</p> : null}

      {data?.tournaments?.length ? (
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1rem" }}>
          {data.tournaments.map((choice) => {
            const slug = choice.settings.registration_slug;
            const active = choice.tournament.id === tournament?.id;
            const href = slug ? `/clubs/${clubSlug}/tournament-registration?tournament=${encodeURIComponent(slug)}` : `/clubs/${clubSlug}/tournament-registration?tournament_id=${encodeURIComponent(choice.tournament.id)}`;
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
          <article style={cardStyle}><strong>Start</strong><br />{dateLabel(tournament.start_date) ?? "TBD"}</article>
          <article style={cardStyle}><strong>Status</strong><br />{settings?.registration_status ?? "draft"}</article>
          <article style={cardStyle}><strong>Open divisions</strong><br />{selectableCount}</article>
          <article style={cardStyle}><strong>Registrations</strong><br />{data?.roster_summary?.total_registrations ?? 0}</article>
        </div>
      ) : null}

      {tournament ? (
        <article style={{ ...cardStyle, marginBottom: "1rem", background: "#f8fafc" }}>
          <h2 style={{ marginTop: 0 }}>Already registered?</h2>
          <p style={{ color: "#475569" }}>Enter the email address used for your registration and we will send a secure edit link if a matching registration exists.</p>
          <EditLinkRequestForm clubSlug={clubSlug} tournamentId={tournament.id} registrationSlug={settings?.registration_slug ?? null} />
        </article>
      ) : null}

      {tournament && !data?.registration_open ? (
        <article style={{ ...cardStyle, marginBottom: "1rem" }}>
          <h2 style={{ marginTop: 0 }}>Registration is closed</h2>
          <p style={{ color: "#475569" }}>{data?.registration_closed_reason || "Registration is not currently open."}</p>
          <Link href="/support">Contact support</Link>
        </article>
      ) : null}

      {tournament && settings?.rules_markdown ? (
        <article style={{ ...cardStyle, marginBottom: "1rem" }}>
          <h2 style={{ marginTop: 0 }}>Rules</h2>
          {markdownish(settings.rules_markdown)}
        </article>
      ) : null}

      {tournament && data?.registration_open ? (
        <TournamentRegistrationForm
          clubSlug={clubSlug}
          tournamentId={tournament.id}
          registrationSlug={settings?.registration_slug ?? null}
          days={data.days ?? []}
          events={data.events ?? []}
        />
      ) : null}
    </section>
  );
}
