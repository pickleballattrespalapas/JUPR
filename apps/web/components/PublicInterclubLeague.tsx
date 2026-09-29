"use client";

import { useId, useState } from "react";
import resultsStyles from "./InterclubResults.module.css";
import type { PublicLeague } from "@/lib/interclubPublic";
import { meetTime } from "@/lib/interclubPublic";
import styles from "./ClubWebsite.module.css";
import { CompetitionStandings, CompetitionResults } from "./PublicInterclubCompetition";
export default function PublicInterclubLeague({
  league,
}: {
  league: PublicLeague;
}) {
  const [view, setView] = useState("standings"), [resultClub, setResultClub] = useState("");
  const panelId = useId();
  const tabs = [{ id: "standings", label: "Overall standings" }, { id: "results", label: "Results" }, { id: "schedule", label: "Schedule" }];
  const doc = league.document,
    names = Object.fromEntries(doc.clubs.map((c) => [c.id, c.name]));
  return (
    <section>
      <p className={styles.eyebrow}>Interclub league</p>
      <h1>{doc.name}</h1>
      <p>
        {doc.start_date} – {doc.end_date} · {doc.clubs.length} participating
        clubs
      </p>
      <div className={resultsStyles.tabs} role="tablist" aria-label="League sections">
        {tabs.map((tab, index) => <button key={tab.id} id={`${panelId}-${tab.id}-tab`} type="button" role="tab" aria-selected={view === tab.id} aria-controls={`${panelId}-${tab.id}`} tabIndex={view === tab.id ? 0 : -1} className={resultsStyles.tab} onClick={() => setView(tab.id)} onKeyDown={event => {
          const next = event.key === "ArrowRight" ? (index + 1) % tabs.length : event.key === "ArrowLeft" ? (index + tabs.length - 1) % tabs.length : event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : -1;
          if (next >= 0) { event.preventDefault(); setView(tabs[next].id); document.getElementById(`${panelId}-${tabs[next].id}-tab`)?.focus(); }
        }}>{tab.label}</button>)}
      </div>
      <section id={`${panelId}-standings`} role="tabpanel" aria-labelledby={`${panelId}-standings-tab`} hidden={view !== "standings"}>
        {doc.scoring_version === 1 ? <CompetitionStandings league={league} onClubSelect={club => { setResultClub(club); setView("results"); }} /> : <>
        <p>
          Ranked by encounter wins, then game difference and point difference.
          Clubs with equal totals are tied.
        </p>
        {!doc.results.length && (
          <p className={styles.notice}>
            No results have been published yet. All clubs start at zero.
          </p>
        )}
        <p className={styles.notice}>Historical results recorded with the earlier scoring format.</p>
        {league.standings.map((group) => (
          <section key={group.division}>
            <h3>{group.division} division</h3>
            <div style={{ overflowX: "auto" }}>
              <table className={styles.table}>
                <thead>
                  <tr>
                    {[
                      "Club",
                      "Played",
                      "Won",
                      "Lost",
                      "Games won",
                      "Games lost",
                      "Point difference",
                    ].map((h) => (
                      <th key={h} scope="col">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {group.rows.map((row) => (
                    <tr key={row.club_id}>
                      <th scope="row">{row.name}</th>
                      <td>{row.played}</td>
                      <td>{row.wins}</td>
                      <td>{row.losses}</td>
                      <td>{row.games_won}</td>
                      <td>{row.games_lost}</td>
                      <td>
                        {(row.point_difference || 0) > 0 ? "+" : ""}
                        {row.point_difference}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        ))}
        </>}
      </section>
      <section id={`${panelId}-schedule`} role="tabpanel" aria-labelledby={`${panelId}-schedule-tab`} hidden={view !== "schedule"}>
        <h2>Meet schedule</h2>
        <p>
          All times use {doc.timezone}. Clubs choose players separately for each
          meet.
        </p>
        <div className={styles.grid}>
          {doc.meets.map((meet, i) => (
            <article key={meet.id} className={styles.card}>
              <p className={styles.eyebrow}>Meet {i + 1}</p>
              <h3>{meetTime(meet.starts_at, doc.timezone)}</h3>
              <p>
                <strong>Host:</strong>{" "}
                {names[meet.host_club_id || ""] || "To be confirmed"}
              </p>
              <p>
                {meet.club_ids
                  .map((id) => names[id])
                  .filter(Boolean)
                  .join(" · ")}
              </p>
              <small>
                {meet.courts} courts · {meet.duration_minutes} minutes
              </small>
            </article>
          ))}
        </div>
        {!doc.meets.length && <p>Meet dates will be announced here.</p>}
      </section>
      <section id={`${panelId}-results`} role="tabpanel" aria-labelledby={`${panelId}-results-tab`} hidden={view !== "results"}>
        <h2>Results</h2>
        {doc.scoring_version === 1 ? <CompetitionResults key={resultClub} results={doc.competition_results || []} names={names} meets={doc.meets} timezone={doc.timezone} players={doc.players} initialClub={resultClub} /> : !doc.results.length ? (
          <p>Results will appear after the organizer publishes them.</p>
        ) : (
          <div className={styles.grid}>
            {doc.results.map((row) => {
              const meet = doc.meets.find((m) => m.id === row.meet_id);
              const wins = row.games.filter((g) => g.a > g.b).length;
              return (
                <article className={styles.card} key={row.id}>
                  <p className={styles.eyebrow}>
                    {row.division} ·{" "}
                    {meet
                      ? meetTime(meet.starts_at, doc.timezone)
                      : "Meet result"}
                  </p>
                  <h3>
                    {names[row.club_a]} vs {names[row.club_b]}
                  </h3>
                  <p>
                    <strong>
                      {names[wins >= 2 ? row.club_a : row.club_b]}
                    </strong>{" "}
                    won {Math.max(wins, 3 - wins)}–{Math.min(wins, 3 - wins)}{" "}
                    games.
                  </p>
                  <p>
                    {row.games.map((g, i) => (
                      <span key={i}>
                        {i ? " · " : ""}Game {i + 1}: {g.a}–{g.b}
                      </span>
                    ))}
                  </p>
                </article>
              );
            })}
          </div>
        )}
      </section>
    </section>
  );
}
