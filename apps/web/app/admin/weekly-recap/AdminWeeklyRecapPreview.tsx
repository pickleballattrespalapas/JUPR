"use client";

import type { AdminWeeklyRecapRow } from "@/lib/adminWeeklyRecapApi";
import type { WeeklyRecapJson, WeeklyRecapSpotlight, WeeklyRecapTournament } from "@/lib/weeklyRecapApi";

type Props = { recap: AdminWeeklyRecapRow; printMode?: boolean };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white", breakInside: "avoid" as const };

function NumberStrip({ recap }: { recap: WeeklyRecapJson }) {
  const cards = recap.numbers_cards?.length ? recap.numbers_cards : [
    { key: "matches", label: "Matches", value: recap.numbers?.matches ?? 0 },
    { key: "players", label: "Players", value: recap.numbers?.players ?? 0 },
    { key: "leagues", label: "Leagues", value: recap.numbers?.leagues ?? 0 },
    { key: "round_robins", label: "Pop-Ups", value: recap.numbers?.round_robins ?? 0 }
  ];
  return <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: "0.75rem" }}>{cards.map((card) => <article key={String(card.key || card.label)} style={{ ...cardStyle, textAlign: "center" }}><div style={{ fontSize: "1.6rem", fontWeight: 800 }}>{card.value ?? 0}</div><div style={{ color: "#64748b", textTransform: "uppercase", fontSize: "0.78rem" }}>{card.label}</div></article>)}</div>;
}

function Spotlight({ item }: { item: WeeklyRecapSpotlight }) {
  return <article style={cardStyle}><p style={{ color: "#2563eb", textTransform: "uppercase", fontWeight: 800, fontSize: "0.75rem", marginTop: 0 }}>{item.label}</p><h3>{item.label}</h3>{item.description ? <p>{item.description}</p> : null}{item.players?.length ? <ul>{item.players.map((player) => <li key={`${item.label}-${player}`}>{player}</li>)}</ul> : <p>No spotlight players listed.</p>}</article>;
}

function Podium({ tournament }: { tournament: WeeklyRecapTournament }) {
  const podium = new Map((tournament.podium || []).map((row) => [Number(row.placement), row.display_name]));
  return <article style={{ ...cardStyle, background: "#fff7ed" }}><h3>{tournament.tournament_name}</h3><div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0.75rem", textAlign: "center" }}><div>🥈 {podium.get(2) || "—"}</div><div>🥇 {podium.get(1) || "—"}</div><div>🥉 {podium.get(3) || "—"}</div></div></article>;
}

export default function AdminWeeklyRecapPreview({ recap: row, printMode = false }: Props) {
  const savedFinal = row.final_json || {};
  const recap = (Object.keys(savedFinal).length ? savedFinal : (row.generated_json || {})) as WeeklyRecapJson;
  const spotlight = (recap.spotlight || []).filter((item) => item.include !== false).sort((a, b) => Number(a.order ?? 999) - Number(b.order ?? 999));
  const around: Array<{ title: string; rows: string[] }> = [];
  for (const item of recap.around_club?.leagues || []) around.push({ title: item.league_name, rows: (item.highlights || []).map((highlight) => highlight.display).filter(Boolean) });
  for (const item of recap.around_club?.round_robins || []) around.push({ title: item.event_name, rows: (item.highlights || []).map((highlight) => highlight.display).filter(Boolean) });
  for (const item of recap.around_club?.community_events || []) around.push({ title: item.event_name || item.event_type_label || "Community event", rows: (item.highlights || []).map((highlight) => highlight.display).filter(Boolean) });

  return (
    <section className="admin-recap-preview" data-testid="admin-weekly-recap-preview" style={{ display: "grid", gap: "1rem" }}>
      <style>{`
        @media print {
          body * { visibility: hidden !important; }
          .admin-recap-preview, .admin-recap-preview * { visibility: visible !important; }
          .admin-recap-preview { position: absolute; inset: 0; padding: 0.35in; color: #0f172a; }
          .admin-recap-no-print { display: none !important; }
          .admin-recap-preview article { box-shadow: none !important; break-inside: avoid; }
        }
      `}</style>
      <header style={{ borderBottom: "2px solid #0f172a", paddingBottom: "0.75rem" }}>
        <p style={{ color: row.status === "published" ? "#166534" : "#92400e", textTransform: "uppercase", fontWeight: 900, letterSpacing: "0.08em" }}>{row.status === "published" ? "Published recap preview" : "Unpublished draft — operator preview only"}</p>
        <h1 style={{ marginBottom: "0.25rem" }}>Weekly Recap · {row.week_start} to {row.week_end}</h1>
        <p style={{ color: "#475569" }}>Version {row.row_version || 1} · Loaded {String(row.updated_at || "unknown update time")}</p>
        {!printMode ? <button type="button" className="admin-recap-no-print" onClick={() => window.print()} style={{ padding: "0.55rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "white", fontWeight: 800 }}>Print / save unpublished preview</button> : null}
      </header>
      <NumberStrip recap={recap} />
      <section><h2>Spotlight Reel</h2>{spotlight.length ? <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: "1rem" }}>{spotlight.map((item) => <Spotlight key={`${item.key}-${item.order}`} item={item} />)}</div> : <p>No spotlight reel in this draft.</p>}</section>
      <section><h2>Around the Club</h2>{around.length ? <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: "1rem" }}>{around.map((item) => <article key={item.title} style={cardStyle}><h3>{item.title}</h3>{item.rows.length ? <ul>{item.rows.map((value) => <li key={`${item.title}-${value}`}>{value}</li>)}</ul> : <p>No highlights listed.</p>}</article>)}</div> : <p>No around-the-club highlights in this draft.</p>}</section>
      {recap.tournaments?.length ? <section><h2>Tournaments</h2><div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: "1rem" }}>{recap.tournaments.map((item) => <Podium key={item.tournament_name} tournament={item} />)}</div></section> : null}
      <section><h2>Looking Ahead</h2>{recap.looking_ahead?.length ? <article style={cardStyle}><ul>{recap.looking_ahead.map((item) => <li key={item}>{item}</li>)}</ul></article> : <p>No looking-ahead notes in this draft.</p>}</section>
    </section>
  );
}
