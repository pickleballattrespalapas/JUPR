import type { WeeklyReport } from "@/lib/weeklyRecapApi";
import { emailUrl } from "@/lib/weeklyReportEmail";

const card = { background: "#fff", border: "1px solid #e6e8e1", borderRadius: 12, padding: "1.2rem", breakInside: "avoid" as const };
export function WeeklyReportIntro({ report }: { report: WeeklyReport }) {
  return <div style={{ color: "#263e3a", marginBottom: "1rem" }}>{report.opening ? <p style={{ whiteSpace: "pre-line", fontSize: "1.1rem", lineHeight: 1.7 }}>{report.opening}</p> : null}</div>;
}
export function WeeklyReportEvents({ report, clubSlug }: { report: WeeklyReport; clubSlug: string }) {
  return report.event_results?.length ? <section data-testid="weekly-report-events"><h2>Event results</h2><div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(min(100%,280px),1fr))", gap: "1rem" }}>{report.event_results.map((event, i) => <article key={`${event.id || event.title}-${i}`} style={card}><p style={{ color: "#17665c", margin: 0, fontSize: 13, fontWeight: 700 }}>{event.kind}{event.date ? ` · ${event.date}` : ""}</p><h3 style={{ color: "#123e37" }}>{event.title}</h3>{event.summary ? <p>{event.summary}</p> : null}{event.details?.length ? <ul style={{ paddingLeft: "1.2rem", lineHeight: 1.7 }}>{event.details.map((line, idx) => <li key={idx}>{line}</li>)}</ul> : null}{event.href ? <a href={`/clubs/${encodeURIComponent(clubSlug)}/${event.href}`} style={{ color: "#17665c", fontWeight: 700 }}>View results</a> : null}</article>)}</div></section> : null;
}
export function WeeklyReportProgress({ report, clubSlug }: { report: WeeklyReport; clubSlug: string }) {
  return <>
    {report.event_notes ? <section><h2>Around the club</h2><p style={{ ...card, whiteSpace: "pre-line" }}>{report.event_notes}</p></section> : null}
    {report.season_races?.length ? <section data-testid="weekly-report-season-races"><h2>Season awards race</h2><p style={{ color: "#536c64" }}>Standings through the end of this report. Changes compare with the start of the period.</p><div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(min(100%,280px),1fr))", gap: "1rem" }}>{report.season_races.map((race, i) => <article style={{ ...card, background: "#f8f3e9" }} key={i}><h3>{race.title}</h3><ul style={{ paddingLeft: "1.2rem", lineHeight: 1.7 }}>{race.lines.map((line, idx) => <li key={idx}>{line}</li>)}</ul>{race.href ? <a href={`/clubs/${encodeURIComponent(clubSlug)}/${race.href}`} style={{ color: "#17665c" }}>View season standings</a> : null}</article>)}</div></section> : null}
    {report.achievements?.length ? <section><h2>New achievements</h2><ul style={{ ...card, paddingLeft: "2.5rem", lineHeight: 1.8 }}>{report.achievements.map((line, i) => <li key={i}>{line}</li>)}</ul></section> : null}
    {report.new_faces?.length ? <section><h2>Welcome to the courts</h2><p style={card}>{report.new_faces.join(", ")}</p></section> : null}
  </>;
}
export function WeeklyReportUpcoming({ report }: { report: WeeklyReport }) {
  return report.upcoming?.length ? <section><h2>Coming up</h2><ul style={{ ...card, paddingLeft: "2.5rem", lineHeight: 1.8 }}>{report.upcoming.map((item, i) => <li key={i}>{emailUrl(item.url) ? <a href={emailUrl(item.url)} style={{ color: "#17665c" }}>{item.text}</a> : item.text}</li>)}</ul></section> : null;
}
