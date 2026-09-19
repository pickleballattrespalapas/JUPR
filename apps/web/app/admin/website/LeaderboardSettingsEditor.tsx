"use client";

import {
  LEADERBOARD_CARD_LABELS, leaderboardSettings, leaderboardSettingsError,
  type LeaderboardCard, type LeaderboardSeason, type LeaderboardSettings, type SiteDocument,
} from "@/lib/clubSite";
import styles from "@/components/ClubWebsite.module.css";

export default function LeaderboardSettingsEditor({ clubId, document: doc, onChange }: {
  clubId: string; document: SiteDocument; onChange: (patch: Partial<SiteDocument>) => void;
}) {
  const settings = leaderboardSettings(doc.leaderboard);
  const validation = leaderboardSettingsError(doc.leaderboard);
  function change(patch: Partial<LeaderboardSettings>) {
    onChange({ leaderboard: { ...settings, ...patch } });
  }
  function moveCard(index: number, direction: number) {
    const cards = [...settings.cards];
    [cards[index], cards[index + direction]] = [cards[index + direction], cards[index]];
    change({ cards });
  }
  function updateSeason(id: string, patch: Partial<LeaderboardSeason>) {
    change({ seasons: settings.seasons.map(season => season.id === id ? { ...season, ...patch } : season) });
  }
  function addSeason() {
    const timezone = clubId === "tres_palapas" ? "America/Mazatlan" : Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
    change({ seasons: [...settings.seasons, {
      id: crypto.randomUUID(), name: "", start_date: "", end_date: null, timezone,
    }] });
  }
  return <section aria-labelledby="overall-leaderboard-settings">
    <h2 id="overall-leaderboard-settings">Overall leaderboard</h2>
    <p>Choose the cards and date ranges for your club’s Overall leaderboard. Save the draft and publish the website to apply your choices.</p>
    {validation ? <p role="alert" className={styles.error}>{validation}</p> : null}
    <section className={styles.card}>
      <h3 style={{ marginTop: 0 }}>Featured cards</h3>
      <p>Cards appear in this order. Remove any you do not want to show, or add another below.</p>
      {settings.cards.length ? <ol style={{ paddingLeft: "1.5rem" }}>
        {settings.cards.map((key, index) => <li key={key} style={{ padding: ".5rem 0" }}>
          <div className={styles.actions}>
            <strong style={{ flex: "1 1 180px" }}>{LEADERBOARD_CARD_LABELS[key]}</strong>
            <button type="button" className={styles.button} disabled={index === 0} aria-label={`Move ${LEADERBOARD_CARD_LABELS[key]} up`} onClick={() => moveCard(index, -1)}>Move up</button>
            <button type="button" className={styles.button} disabled={index === settings.cards.length - 1} aria-label={`Move ${LEADERBOARD_CARD_LABELS[key]} down`} onClick={() => moveCard(index, 1)}>Move down</button>
            <button type="button" className={styles.button} aria-label={`Remove ${LEADERBOARD_CARD_LABELS[key]} card`} onClick={() => change({ cards: settings.cards.filter(card => card !== key) })}>Remove</button>
          </div>
        </li>)}
      </ol> : <p>No featured cards will be shown.</p>}
      <div className={styles.actions}>
        {(Object.keys(LEADERBOARD_CARD_LABELS) as LeaderboardCard[]).filter(key => !settings.cards.includes(key)).map(key =>
          <button type="button" key={key} className={styles.button} onClick={() => change({ cards: [...settings.cards, key] })}>Add {LEADERBOARD_CARD_LABELS[key]}</button>,
        )}
      </div>
      <p><small>Cards also follow your choices under Stats &amp; information. For example, hiding ratings hides the Highest rating card.</small></p>
      <label><input type="checkbox" checked={settings.show_summary} onChange={event => change({ show_summary: event.target.checked })} /> Show summary counts above the cards</label>
      <div className={styles.form} style={{ maxWidth: 340, marginTop: "1rem" }}>
        <label>Minimum games for performance cards
          <input type="number" min={0} max={10000} step={1} value={settings.min_games} onChange={event => change({ min_games: Number(event.target.value) })} />
        </label>
      </div>
      <p><small>Applies to improvement, win percentage, wins and games played in the selected period. Highest rating remains open to everyone. Players need at least one game to appear on performance cards.</small></p>
    </section>
    <section className={styles.card} style={{ marginTop: "1rem" }}>
      <h3 style={{ marginTop: 0 }}>Seasons &amp; date ranges</h3>
      <p>Each season starts wins and games at zero. Most improved measures the rating change from the season’s starting point. Player ratings and match history are kept.</p>
      <p>Visitors can select any published season or All time. With an end date, that period’s results stop at the end of that day. Leave it blank to keep counting.</p>
      <div className={styles.form} style={{ maxWidth: 420 }}>
        <label>Show this period by default
          <select value={settings.default_season_id ?? "all"} onChange={event => change({ default_season_id: event.target.value === "all" ? null : event.target.value })}>
            <option value="all">All time</option>
            {settings.seasons.map((season, index) => <option key={season.id} value={season.id}>{season.name || `Season ${index + 1}`}</option>)}
          </select>
        </label>
      </div>
      {!settings.seasons.length ? <p>No seasons added. The leaderboard currently uses all-time statistics.</p> : null}
      {settings.seasons.map((season, index) => <fieldset key={season.id} style={{ border: "1px solid #cbd5e1", borderRadius: 12, margin: "1rem 0", padding: "1rem" }}>
        <legend style={{ fontWeight: 750 }}>{season.name || `Season ${index + 1}`}</legend>
        <div className={`${styles.form} ${styles.grid}`}>
          <label>Season name<input aria-label={`Season ${index + 1} name`} required maxLength={100} value={season.name} placeholder="2026–27 season" onChange={event => updateSeason(season.id, { name: event.target.value })} /></label>
          <label>Start date<input aria-label={`Season ${index + 1} start date`} type="date" required value={season.start_date} onChange={event => updateSeason(season.id, { start_date: event.target.value })} /></label>
          <label>End date (optional)<input aria-label={`Season ${index + 1} end date`} type="date" min={season.start_date} value={season.end_date ?? ""} onChange={event => updateSeason(season.id, { end_date: event.target.value || null })} /></label>
          <label>Timezone<input aria-label={`Season ${index + 1} timezone`} required list="leaderboard-timezones" value={season.timezone} onChange={event => updateSeason(season.id, { timezone: event.target.value })} /></label>
        </div>
        <button type="button" className={styles.button} aria-label={`Remove ${season.name || `season ${index + 1}`} from leaderboard`} onClick={() => change({
          seasons: settings.seasons.filter(item => item.id !== season.id),
          default_season_id: settings.default_season_id === season.id ? null : settings.default_season_id,
        })}>Remove from leaderboard</button>
      </fieldset>)}
      <datalist id="leaderboard-timezones"><option value="America/Mazatlan" /><option value="America/Los_Angeles" /><option value="America/New_York" /><option value="UTC" /></datalist>
      <button type="button" className={styles.button} disabled={settings.seasons.length >= 40} onClick={addSeason}>Add season or date range</button>
      <p><small>These dates control the Overall leaderboard. Badge seasons are managed separately under Badges &amp; Seasons.</small></p>
    </section>
  </section>;
}
