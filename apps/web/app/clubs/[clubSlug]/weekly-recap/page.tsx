import Link from "next/link";
import { getClubWeeklyRecaps, getWeeklyRecapPdfUrl } from "@/lib/weeklyRecapApi";
import type { WeeklyRecapJson, WeeklyRecapSpotlight, WeeklyRecapTournament } from "@/lib/weeklyRecapApi";
import PrintButton from "./PrintButton";

type WeeklyRecapPageProps = {
  params: { clubSlug: string };
  searchParams?: Record<string, string | string[] | undefined>;
};

type RecapSection = "all" | "spotlight" | "around" | "tournaments" | "ahead";

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

const sectionLabels: Record<RecapSection, string> = {
  all: "Full recap",
  spotlight: "Spotlight Reel",
  around: "Around the Club",
  tournaments: "Tournaments",
  ahead: "Looking Ahead"
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

const spotlightStyleByKey: Record<string, { background: string; borderLeft: string }> = {
  TOP_PERFORMER_WEEK: { background: "linear-gradient(135deg, #fff3e6, #ffe0c2)", borderLeft: "6px solid #ff7a00" },
  BIGGEST_JUMP_WEEK: { background: "linear-gradient(135deg, #e6faf7, #c8f3ee)", borderLeft: "6px solid #00b3a4" },
  COMMUNITY_STANDOUT_WEEK: { background: "linear-gradient(135deg, #e8f7ff, #d6eeff)", borderLeft: "6px solid #1f78b4" },
  GIANT_SLAYER_WEEK: { background: "linear-gradient(135deg, #f3e8ff, #e3d4ff)", borderLeft: "6px solid #7e57c2" },
  GRIND_WEEK: { background: "linear-gradient(135deg, #f1f1f1, #e2e2e2)", borderLeft: "6px solid #444" },
  SOCIAL_GRIND_WEEK: { background: "linear-gradient(135deg, #ecfff1, #d8fbe4)", borderLeft: "6px solid #2e8b57" },
  PERFECT_RUN: { background: "linear-gradient(135deg, #fff0f5, #ffd6e8)", borderLeft: "6px solid #d63384" }
};

function firstParam(searchParams: WeeklyRecapPageProps["searchParams"], key: string): string | null {
  const value = searchParams?.[key];
  if (Array.isArray(value)) return value[0] ?? null;
  return value ?? null;
}

function decodeParam(value: string | null): string | null {
  if (!value) return null;
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function normalizeSection(value: string | null): RecapSection {
  if (value === "spotlight" || value === "around" || value === "tournaments" || value === "ahead") return value;
  return "all";
}

function pageHref({ clubSlug, week, section, print, page }: { clubSlug: string; week?: string | null; section?: RecapSection | null; print?: boolean; page?: number | null }): string {
  const params = new URLSearchParams();
  if (week) params.set("week", week);
  if (section && section !== "all") params.set("section", section);
  if (print) params.set("print", "1");
  if (page && page > 1) params.set("page", String(page));
  const query = params.toString();
  return `/clubs/${clubSlug}/weekly-recap${query ? `?${query}` : ""}`;
}

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

function showSection(active: RecapSection, section: RecapSection): boolean {
  return active === "all" || active === section;
}

function NumberStrip({ recap }: { recap: WeeklyRecapJson }) {
  const cards = recap.numbers_cards?.length
    ? recap.numbers_cards
    : [
        { key: "matches", label: "Matches", value: recap.numbers?.matches ?? 0 },
        { key: "players", label: "Players", value: recap.numbers?.players ?? 0 },
        { key: "leagues", label: "Leagues", value: recap.numbers?.leagues ?? 0 },
        { key: "round_robins", label: "Pop-Ups", value: recap.numbers?.round_robins ?? 0 },
        { key: "community_events", label: "Community Events", value: recap.numbers?.community_events ?? recap.numbers?.social_round_robins ?? 0 },
        { key: "new_faces", label: "New Faces", value: recap.numbers?.new_faces ?? 0 }
      ];
  return (
    <div data-testid="weekly-recap-number-cards" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
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
  const accentStyle = spotlightStyleByKey[String(item.key)] ?? spotlightStyleByKey.TOP_PERFORMER_WEEK;
  return (
    <article style={{ ...cardStyle, ...accentStyle, boxShadow: "0 8px 22px rgba(15, 23, 42, 0.08)", breakInside: "avoid" }}>
      <p style={{ margin: "0 0 0.3rem", color: "#2563eb", fontSize: "0.75rem", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.06em" }}>
        {accentByKey[String(item.key)] ?? item.label}
      </p>
      <h3 style={{ margin: "0 0 0.5rem" }}>{item.label}</h3>
      {item.description ? <p style={{ color: "#475569" }}>{item.description}</p> : null}
      {item.players?.length ? (
        <ul style={{ margin: 0, paddingLeft: "1.25rem" }}>
          {item.players.map((player) => <li key={`${item.label}-${player}`}>{player}</li>)}
        </ul>
      ) : <p style={{ color: "#64748b" }}>No players were selected for this spotlight.</p>}
    </article>
  );
}

function AroundCard({ title, rows, tone }: { title: string; rows: string[]; tone: "league" | "round_robin" | "community" }) {
  const toneStyle = tone === "community"
    ? { background: "linear-gradient(135deg, #eafbff, #d9f4ff)", borderLeft: "6px solid #0077b6" }
    : tone === "round_robin"
      ? { background: "linear-gradient(135deg, #e8fff4, #d2f7e7)", borderLeft: "6px solid #1b9e77" }
      : { background: "linear-gradient(135deg, #eaf4ff, #d4e8ff)", borderLeft: "6px solid #1976d2" };
  return (
    <article style={{ ...cardStyle, ...toneStyle, boxShadow: "0 8px 22px rgba(15, 23, 42, 0.08)", breakInside: "avoid" }}>
      <h3 style={{ marginTop: 0 }}>{title}</h3>
      {rows.length ? (
        <ul style={{ marginBottom: 0, paddingLeft: "1.25rem" }}>
          {rows.map((row) => <li key={`${title}-${row}`}>{row}</li>)}
        </ul>
      ) : <p style={{ color: "#64748b" }}>No highlights this week.</p>}
    </article>
  );
}

function Podium({ tournament }: { tournament: WeeklyRecapTournament }) {
  const podium = new Map((tournament.podium || []).map((row) => [Number(row.placement), row.display_name]));
  return (
    <article style={{ ...cardStyle, background: "linear-gradient(135deg, #ffe7a3, #ffd166)", border: "2px solid #e0a100", boxShadow: "0 10px 26px rgba(15, 23, 42, 0.1)", breakInside: "avoid" }}>
      <h3 style={{ marginTop: 0 }}>{tournament.tournament_name}</h3>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0.75rem", textAlign: "center" }}>
        <div><div style={{ color: "#64748b", fontSize: "0.8rem" }}>2nd</div><strong>🥈 {podium.get(2) || "—"}</strong></div>
        <div><div style={{ color: "#64748b", fontSize: "0.8rem" }}>1st</div><strong>🥇 {podium.get(1) || "—"}</strong></div>
        <div><div style={{ color: "#64748b", fontSize: "0.8rem" }}>3rd</div><strong>🥉 {podium.get(3) || "—"}</strong></div>
      </div>
    </article>
  );
}

function RecapBody({ recap, activeSection }: { recap: WeeklyRecapJson; activeSection: RecapSection }) {
  const spotlight = (recap.spotlight || []).filter((item) => item.include !== false).sort((a, b) => Number(a.order ?? 999) - Number(b.order ?? 999));
  const aroundCards: Array<{ title: string; rows: string[]; tone: "league" | "round_robin" | "community" }> = [];
  for (const item of recap.around_club?.leagues || []) aroundCards.push({ title: item.league_name, rows: (item.highlights || []).map((row) => row.display).filter(Boolean), tone: "league" });
  for (const item of recap.around_club?.round_robins || []) aroundCards.push({ title: item.event_name, rows: (item.highlights || []).map((row) => row.display).filter(Boolean), tone: "round_robin" });
  for (const item of recap.around_club?.community_events || []) {
    const title = item.skill_level ? (item.event_type_label || "Community Event") : `${item.event_name} (${item.event_type_label || "Community Event"})`;
    aroundCards.push({ title, rows: (item.highlights || []).map((row) => row.display).filter(Boolean), tone: "community" });
  }
  const visibleAroundCards = aroundCards.filter((item) => item.rows.length > 0);

  return (
    <>
      <NumberStrip recap={recap} />

      {showSection(activeSection, "spotlight") ? (
        <section id="spotlight" data-testid="weekly-recap-section-content-spotlight" style={{ pageBreakInside: "avoid" }}>
          <h2>Spotlight Reel</h2>
          {spotlight.length ? (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1rem" }}>
              {spotlight.map((item) => <SpotlightCard key={`${item.key ?? item.label}-${item.order ?? 0}`} item={item} />)}
            </div>
          ) : <p style={{ color: "#64748b" }}>No player spotlights this week.</p>}
        </section>
      ) : null}

      {showSection(activeSection, "around") ? (
        <section id="around" data-testid="weekly-recap-section-content-around" style={{ pageBreakInside: "avoid" }}>
          <h2>Around the Club</h2>
          {visibleAroundCards.length ? (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1rem" }}>
              {visibleAroundCards.map((item) => <AroundCard key={`${item.tone}-${item.title}`} title={item.title} rows={item.rows} tone={item.tone} />)}
            </div>
          ) : <p style={{ color: "#64748b" }}>No club highlights this week.</p>}
        </section>
      ) : null}

      {showSection(activeSection, "tournaments") ? (
        <section id="tournaments" data-testid="weekly-recap-section-content-tournaments" style={{ pageBreakInside: "avoid" }}>
          <h2>Tournaments</h2>
          {recap.tournaments?.length ? (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1rem" }}>
              {recap.tournaments.map((tournament) => <Podium key={tournament.tournament_name} tournament={tournament} />)}
            </div>
          ) : <p style={{ color: "#64748b" }}>No tournament finishes this week.</p>}
        </section>
      ) : null}

      {showSection(activeSection, "ahead") && recap.looking_ahead?.length ? (
        <section id="ahead" data-testid="weekly-recap-section-content-ahead" style={{ pageBreakInside: "avoid" }}>
          <h2>Looking Ahead</h2>
          <article style={cardStyle}>
            <ul style={{ margin: 0, paddingLeft: "1.25rem" }}>
              {recap.looking_ahead.map((item) => <li key={item}>{item}</li>)}
            </ul>
          </article>
        </section>
      ) : null}
    </>
  );
}

export default async function WeeklyRecapPage({ params, searchParams }: WeeklyRecapPageProps) {
  const { clubSlug } = params;
  const selectedWeek = decodeParam(firstParam(searchParams, "week"));
  const activeSection = normalizeSection(firstParam(searchParams, "section"));
  const printMode = ["1", "true", "yes"].includes(String(firstParam(searchParams, "print") ?? "").toLowerCase());
  const requestedPage = Number(firstParam(searchParams, "page") || "1");
  const activePage = Number.isFinite(requestedPage) ? Math.max(1, Math.round(requestedPage)) : 1;
  const { data, error } = await getClubWeeklyRecaps(clubSlug, selectedWeek, activePage);
  const selected = data?.selected_recap ?? null;
  const pagination = data?.pagination;
  const pdfUrl = selected?.week_start ? getWeeklyRecapPdfUrl(clubSlug, selected.week_start) : null;

  return (
    <section data-testid="weekly-recap-page" data-print-mode={printMode ? "true" : "false"}>
      <style>{`
        @page { size: letter; margin: 0.45in; }
        @media print {
          .no-print { display: none !important; }
          body { background: white !important; }
          a { color: inherit !important; text-decoration: none !important; }
          section { color: #0f172a; }
          article { box-shadow: none !important; }
          .weekly-recap-document { max-width: none !important; }
          .weekly-recap-document section, .weekly-recap-document article { break-inside: avoid; }
        }
      `}</style>

      {!printMode ? (
        <>
          <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
            Weekly Recap
          </p>
          <h1 style={{ marginTop: 0 }}>{data?.club.name ?? clubSlug} weekly recap</h1>
          <p style={{ color: "#334155", maxWidth: "820px" }}>
            Catch up on this week&apos;s matches, standout players, tournament finishes, and what&apos;s coming next.
          </p>
        </>
      ) : null}

      {error ? <p role="alert" style={{ color: "#b91c1c" }}>Weekly Recap is unavailable right now. Please try again shortly.</p> : null}
      {!error && !data?.recaps?.length ? (
        pagination?.has_previous ? (
          <p>There are no recaps on this page. <Link href={pageHref({ clubSlug, section: activeSection })}>View the newest recaps</Link>.</p>
        ) : <p>No weekly recaps have been posted yet.</p>
      ) : null}

      {!printMode && data?.recaps?.length ? (
        <div className="no-print" style={{ display: "grid", gap: "0.75rem", marginBottom: "1rem" }}>
          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
            {data.recaps.map((recap) => {
              const active = recap.week_start === selected?.week_start;
              return (
                <Link data-testid="weekly-recap-week-link" key={recap.week_start} href={pageHref({ clubSlug, week: recap.week_start, section: activeSection, page: activePage })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: active ? "#dbeafe" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                  {dateLabel(recap.week_start)}
                </Link>
              );
            })}
          </div>
          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", alignItems: "center" }}>
            {(Object.keys(sectionLabels) as RecapSection[]).map((section) => {
              const active = section === activeSection;
              return (
                <Link data-testid={`weekly-recap-section-${section}`} key={section} href={pageHref({ clubSlug, week: selected?.week_start, section, page: activePage })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: active ? "#dcfce7" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                  {sectionLabels[section]}
                </Link>
              );
            })}
            <Link data-testid="weekly-recap-print-link" href={pageHref({ clubSlug, week: selected?.week_start, section: activeSection, print: true, page: activePage })} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: printMode ? "#fef9c3" : "white", color: "#0f172a", textDecoration: "none", fontWeight: 800 }}>
              Print view
            </Link>
            <PrintButton />
          </div>
          {pagination && (pagination.has_previous || pagination.has_next) ? (
            <nav aria-label="Weekly recap pages" style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
              {pagination.has_previous ? (
                <Link data-testid="weekly-recap-previous-page" href={pageHref({ clubSlug, section: activeSection, page: pagination.page - 1 })}>Newer recaps</Link>
              ) : null}
              <span style={{ color: "#64748b" }}>Page {pagination.page}</span>
              {pagination.has_next ? (
                <Link data-testid="weekly-recap-next-page" href={pageHref({ clubSlug, section: activeSection, page: pagination.page + 1 })}>Older recaps</Link>
              ) : null}
            </nav>
          ) : null}
        </div>
      ) : null}

      {selected ? (
        <div className="weekly-recap-document" data-testid="weekly-recap-document">
          {!printMode ? (
            <div style={{ ...cardStyle, marginBottom: "1rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", flexWrap: "wrap" }}>
                <div>
                  <strong>{dateRangeLabel(selected.recap)}</strong>
                  {selected.summary.headline ? <p style={{ margin: "0.35rem 0 0", color: "#475569" }}>{selected.summary.headline}</p> : null}
                </div>
                <div className="no-print" style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                  {pdfUrl ? <a data-testid="weekly-recap-pdf-link" href={pdfUrl}>Download PDF</a> : null}
                  <a href={pageHref({ clubSlug, week: selected.week_start, section: activeSection, print: true, page: activePage })}>Print-friendly view</a>
                </div>
              </div>
            </div>
          ) : (
            <header style={{ marginBottom: "1rem", borderBottom: "1px solid #e2e8f0", paddingBottom: "1rem" }}>
              <p style={{ margin: 0, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Weekly Recap</p>
              <h1>{data?.club.name ?? "Club"} — {dateRangeLabel(selected.recap)}</h1>
              {selected.summary.headline ? <p style={{ color: "#475569" }}>{selected.summary.headline}</p> : null}
              <div className="no-print"><PrintButton /></div>
            </header>
          )}
          <RecapBody recap={selected.recap} activeSection={activeSection} />
        </div>
      ) : null}
    </section>
  );
}
