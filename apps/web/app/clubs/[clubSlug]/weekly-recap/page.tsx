import Link from "next/link";
import { getClubWeeklyRecaps, getWeeklyRecapPdfUrl } from "@/lib/weeklyRecapApi";
import type { WeeklyRecapJson, WeeklyRecapSpotlight, WeeklyRecapTournament } from "@/lib/weeklyRecapApi";

type WeeklyRecapPageProps = {
  params: { clubSlug: string };
  searchParams?: { week?: string; print?: string };
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

const accentByKey: Record<string, string> = {
  TOP_PERFORMER_WEEK: "Top Performer",
  BIGGEST_JUMP_WEEK: "Biggest Jump",
  COMMUNITY_STANDOUT_WEEK: "Community Standout",
  GIANT_SLAYER_WEEK: "Giant Slayer",
  GRIND_WEEK: "Grind",
  SOCIAL_GRIND_WEEK: "Social Grind",
  PERFECT_RUN: "Perfect Run"
};

function dateRangeLabel(recap?: WeeklyRecapJson | null): string {
  const start = recap?.week_start || recap?.start_date;
  const end = recap?.week_end || recap?.end_date;
  if (!start && !end) return "";
  return [start, end].filter(Boolean).join(" to ");
}

function dateLabel(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value).slice(0, 10);
  return date.toISOString().slice(0, 10);
}

function NumberStrip({ recap }: { recap: WeeklyRecapJson }) {
  const cards = recap.numbers_cards?.length
    ? recap.numbers_cards
    : [
        { key: "matches", label: "Matches", value: recap.numbers?.matches ?? 0 },
        { key: "players", label: "Players", value: recap.numbers?.players ?? 0 },
        { key: "leagues", label: "Leagues", value: recap.numbers?.leagues ?? 0 },
        { key: "round_robins", label: "Pop-Ups", value: recap.numbers?.round_robins ?? 0 }
      ];
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
      {cards.map((card) => (
        <article key={`${card.key ?? card.label}`} style={{ ...cardStyle, textAlign: "center" }}>
          <div style={{ fontSize: "1.6rem", fontWeight: 800 }}>{card.value ?? 0}</div>
          <div style={{ color: "#64748b", fontSize: "0.78rem", textTransform: "uppercase", letterSpacing: "0.06em" }}>{card.label}</div>
        </article>
      ))}
    </div>
  );
}

function SpotlightCard({ item }: { item: WeeklyRecapSpotlight }) {
  return (
    <article style={cardStyle}>
      <p style={{ margin: "0 0 0.3rem", color: "#2563eb", fontSize: "0.75rem", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.06em" }}>
        {accentByKey[String(item.key)] ?? item.label}
      </p>
      <h3 style={{ margin: "0 0 0.5rem" }}>{item.label}</h3>
      {item.description ? <p style={{ color: "#475569" }}>{item.description}</p> : null}
      {item.players?.length ? (
        <ul style={{ margin: 0, paddingLeft: "1.25rem" }}>
          {item.players.map((player) => <li key={`${item.label}-${player}`}>{player}</li>)}
        </ul>
      ) : <p style={{ color: "#64748b" }}>No spotlight players listed.</p>}
    </article>
  );
}

function AroundCard({ title, rows }: { title: string; rows: string[] }) {
  return (
    <article style={cardStyle}>
      <h3 style={{ marginTop: 0 }}>{title}</h3>
      {rows.length ? (
        <ul style={{ marginBottom: 0, paddingLeft: "1.25rem" }}>
          {rows.map((row) => <li key={`${title}-${row}`}>{row}</li>)}
        </ul>
      ) : <p style={{ color: "#64748b" }}>No highlights listed.</p>}
    </article>
  );
}

function Podium({ tournament }: { tournament: WeeklyRecapTournament }) {
  const podium = new Map((tournament.podium || []).map((row) => [Number(row.placement), row.display_name]));
  return (
    <article style={{ ...cardStyle, background: "#fff7ed" }}>
      <h3 style={{ marginTop: 0 }}>{tournament.tournament_name}</h3>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0.75rem", textAlign: "center" }}>
        <div><div style={{ color: "#64748b", fontSize: "0.8rem" }}>2nd</div><strong>🥈 {podium.get(2) || "—"}</strong></div>
        <div><div style={{ color: "#64748b", fontSize: "0.8rem" }}>1st</div><strong>🥇 {podium.get(1) || "—"}</strong></div>
        <div><div style={{ color: "#64748b", fontSize: "0.8rem" }}>3rd</div><strong>🥉 {podium.get(3) || "—"}</strong></div>
      </div>
    </article>
  );
}

function RecapBody({ recap }: { recap: WeeklyRecapJson }) {
  const spotlight = (recap.spotlight || []).filter((item) => item.include !== false).sort((a, b) => Number(a.order ?? 999) - Number(b.order ?? 999));
  const aroundCards: Array<{ title: string; rows: string[] }> = [];
  for (const item of recap.around_club?.leagues || []) aroundCards.push({ title: item.league_name, rows: (item.highlights || []).map((row) => row.display).filter(Boolean) });
  for (const item of recap.around_club?.round_robins || []) aroundCards.push({ title: item.event_name, rows: (item.highlights || []).map((row) => row.display).filter(Boolean) });
  for (const item of recap.around_club?.community_events || []) {
    const title = item.skill_level ? (item.event_type_label || "Community Event") : `${item.event_name} (${item.event_type_label || "Community Event"})`;
    aroundCards.push({ title, rows: (item.highlights || []).map((row) => row.display).filter(Boolean) });
  }

  return (
    <>
      <NumberStrip recap={recap} />
      <h2>Spotlight Reel</h2>
      {spotlight.length ? (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1rem" }}>
          {spotlight.map((item) => <SpotlightCard key={`${item.key ?? item.label}-${item.order ?? 0}`} item={item} />)}
        </div>
      ) : <p style={{ color: "#64748b" }}>No spotlight reel published for this week.</p>}

      <h2>Around the Club</h2>
      {aroundCards.length ? (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1rem" }}>
          {aroundCards.map((item) => <AroundCard key={item.title} title={item.title} rows={item.rows} />)}
        </div>
      ) : <p style={{ color: "#64748b" }}>No around-the-club highlights published for this week.</p>}

      {recap.tournaments?.length ? (
        <>
          <h2>Tournaments</h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1rem" }}>
            {recap.tournaments.map((tournament) => <Podium key={tournament.tournament_name} tournament={tournament} />)}
          </div>
        </>
      ) : null}

      {recap.looking_ahead?.length ? (
        <>
          <h2>Looking Ahead</h2>
          <article style={cardStyle}>
            <ul style={{ margin: 0, paddingLeft: "1.25rem" }}>
              {recap.looking_ahead.map((item) => <li key={item}>{item}</li>)}
            </ul>
          </article>
        </>
      ) : null}
    </>
  );
}

export default async function WeeklyRecapPage({ params, searchParams }: WeeklyRecapPageProps) {
  const { clubSlug } = params;
  const selectedWeek = searchParams?.week ? decodeURIComponent(searchParams.week) : null;
  const printMode = ["1", "true", "yes"].includes(String(searchParams?.print ?? "").toLowerCase());
  const { data, error } = await getClubWeeklyRecaps(clubSlug, selectedWeek);
  const selected = data?.selected_recap ?? null;
  const pdfUrl = selected?.week_start ? getWeeklyRecapPdfUrl(clubSlug, selected.week_start) : null;

  return (
    <section>
      {!printMode ? (
        <>
          <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
            Weekly Recap
          </p>
          <h1 style={{ marginTop: 0 }}>{data?.club.name ?? clubSlug} weekly recap</h1>
          <p style={{ color: "#334155", maxWidth: "820px" }}>
            Published club-wide highlights, weekly numbers, spotlight reels, tournament podiums, and looking-ahead notes.
          </p>
        </>
      ) : null}

      {error ? <p style={{ color: "#b91c1c" }}>Weekly Recap is temporarily unavailable. {error}</p> : null}
      {!error && !data?.recaps?.length ? <p>No published weekly recaps yet.</p> : null}

      {!printMode && data?.recaps?.length ? (
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1rem" }}>
          {data.recaps.map((recap) => {
            const active = recap.week_start === selected?.week_start;
            return (
              <Link key={recap.week_start} href={`/clubs/${clubSlug}/weekly-recap?week=${encodeURIComponent(recap.week_start)}`} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: active ? "#dbeafe" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                {dateLabel(recap.week_start)}
              </Link>
            );
          })}
        </div>
      ) : null}

      {selected ? (
        <>
          {!printMode ? (
            <div style={{ ...cardStyle, marginBottom: "1rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", flexWrap: "wrap" }}>
                <div>
                  <strong>{dateRangeLabel(selected.recap)}</strong>
                  {selected.summary.headline ? <p style={{ margin: "0.35rem 0 0", color: "#475569" }}>{selected.summary.headline}</p> : null}
                </div>
                <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                  {pdfUrl ? <a href={pdfUrl}>Download PDF</a> : null}
                  <a href={`/clubs/${clubSlug}/weekly-recap?week=${encodeURIComponent(selected.week_start)}&print=1`}>Print-friendly view</a>
                </div>
              </div>
            </div>
          ) : <h1>{data?.club.name ?? "Club"} Weekly Recap — {dateRangeLabel(selected.recap)}</h1>}
          <RecapBody recap={selected.recap} />
        </>
      ) : null}
    </section>
  );
}
