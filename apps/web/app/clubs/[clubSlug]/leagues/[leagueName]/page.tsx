import Link from "next/link";
import PublicLeagueNav, {
  publicLeagueResultsHref
} from "@/components/PublicLeagueNav";
import { LeagueAwardRaceGrid } from "@/components/LeagueAwardRace";
import { getClubLeagueResults } from "@/lib/api";

type Props = {
  params: { clubSlug: string; leagueName: string };
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white",
  minWidth: 0
};

function decodeLeagueName(value: string): string {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

export default async function PublicLeagueHomePage({ params }: Props) {
  const leagueName = decodeLeagueName(params.leagueName);
  const { data, error } = await getClubLeagueResults(
    params.clubSlug,
    leagueName
  );
  const found = data?.selected_league === leagueName;
  const standings = found ? data?.standings || [] : [];
  const weeks = found ? data?.weeks || [] : [];
  const recentWeek = weeks.length ? weeks[weeks.length - 1] : null;

  if (error || !data || !found) {
    return (
      <section>
        <p
          style={{
            margin: "0 0 0.5rem",
            color: "#2563eb",
            fontWeight: 800,
            textTransform: "uppercase",
            letterSpacing: "0.08em",
            fontSize: "0.78rem"
          }}
        >
          League Home
        </p>
        <h1 style={{ marginTop: 0 }}>{leagueName}</h1>
        <article
          style={{
            ...cardStyle,
            borderColor: "#fecaca",
            background: "#fef2f2"
          }}
        >
          <h2 style={{ marginTop: 0 }}>League unavailable</h2>
          <p style={{ color: "#7f1d1d" }}>
            {error || "This league is not currently available as an active public league."}
          </p>
          <Link href={`/clubs/${params.clubSlug}/leagues`}>
            Return to all leagues
          </Link>
        </article>
      </section>
    );
  }

  const modules = [
    {
      title: "Awards race & player roster",
      description:
        "Award placement and qualification first, with an unranked player roster below.",
      href: publicLeagueResultsHref(
        params.clubSlug,
        leagueName,
        "overall"
      )
    },
    {
      title: "Weekly History",
      description:
        "Week-by-week results, activity, rating movement, and weekly highlights.",
      href: publicLeagueResultsHref(
        params.clubSlug,
        leagueName,
        "weekly"
      )
    },
    {
      title: "Player Summaries",
      description:
        "Open a player’s league record, weekly trend, and recent matches.",
      href: publicLeagueResultsHref(
        params.clubSlug,
        leagueName,
        "player"
      )
    }
  ];

  return (
    <section>
      <p
        style={{
          margin: "0 0 0.5rem",
          color: "#2563eb",
          fontWeight: 800,
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          fontSize: "0.78rem"
        }}
      >
        League Home
      </p>
      <h1 style={{ marginTop: 0 }}>{leagueName}</h1>
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        Awards-race placement, an unranked player roster, weekly history, player summaries, and public league results
        for this league.
      </p>

      <PublicLeagueNav
        clubSlug={params.clubSlug}
        leagueName={leagueName}
        active="home"
      />

      <article
        style={{
          ...cardStyle,
          marginBottom: "1rem",
          background: "#eff6ff",
          borderColor: "#bfdbfe"
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            gap: "1rem",
            flexWrap: "wrap",
            alignItems: "flex-start"
          }}
        >
          <div>
            <h2 style={{ marginTop: 0 }}>{leagueName}</h2>
            <p style={{ marginBottom: 0, color: "#475569" }}>
              Active public league
            </p>
          </div>
          <span
            style={{
              border: "1px solid #86efac",
              borderRadius: "999px",
              padding: "0.25rem 0.6rem",
              background: "#dcfce7",
              color: "#166534",
              fontWeight: 800
            }}
          >
            Active
          </span>
        </div>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
            gap: "0.75rem",
            marginTop: "1rem"
          }}
        >
          <div>
            <strong>Players with a league rating</strong>
            <br />
            {standings.filter((row) => row.rating_jupr != null).length}
          </div>
          <div>
            <strong>Players shown</strong>
            <br />
            {standings.length}
          </div>
          <div>
            <strong>Weeks with results</strong>
            <br />
            {weeks.filter((week) => week.has_results !== false).length}
          </div>
          <div>
            <strong>Minimum games</strong>
            <br />
            {data.league?.min_games ?? 0}
          </div>
          <div>
            <strong>Latest week</strong>
            <br />
            {recentWeek?.week_label || "No results yet"}
          </div>
        </div>
      </article>

      <section style={{ marginBottom: "1.25rem" }}>
        <h2>League pages</h2>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
            gap: "0.75rem"
          }}
        >
          {modules.map((module) => (
            <Link
              key={module.title}
              href={module.href}
              style={{
                ...cardStyle,
                color: "#0f172a",
                textDecoration: "none"
              }}
            >
              <h3 style={{ marginTop: 0 }}>{module.title}</h3>
              <p style={{ color: "#475569" }}>{module.description}</p>
              <strong>Open {module.title} →</strong>
            </Link>
          ))}
        </div>
      </section>

      <section>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            gap: "1rem",
            alignItems: "center",
            flexWrap: "wrap"
          }}
        >
          <div>
            <h2 style={{ marginBottom: "0.25rem" }}>Awards race</h2>
            <p style={{ marginTop: 0, color: "#64748b" }}>
              Top five qualified players for every award, with the full eligible field available on the standings page.
            </p>
          </div>
          <Link
            href={publicLeagueResultsHref(
              params.clubSlug,
              leagueName,
              "overall"
            )}
            style={{ fontWeight: 800 }}
          >
            View awards race and player roster
          </Link>
        </div>

        {data.award_progress.awards.length ? <LeagueAwardRaceGrid progress={data.award_progress} clubSlug={params.clubSlug} /> : <article style={{ ...cardStyle, background: "#f8fafc", marginBottom: "1rem" }}>No player has met the current award qualification criteria yet.</article>}
      </section>
    </section>
  );
}
