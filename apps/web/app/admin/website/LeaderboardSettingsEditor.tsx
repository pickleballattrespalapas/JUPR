"use client";

import {
  LEADERBOARD_CARD_LABELS, LEADERBOARD_CARD_DETAILS, LEADERBOARD_CARD_GROUPS,
  leaderboardSettings, leaderboardSettingsError,
  type LeaderboardCard, type LeaderboardCardOptions, type LeaderboardSeason, type LeaderboardSettings, type SiteDocument,
} from "@/lib/clubSite";
import styles from "@/components/ClubWebsite.module.css";

export default function LeaderboardSettingsEditor({ clubId, document: doc, onChange }: {
  clubId: string; document: Pick<SiteDocument, "leaderboard">; onChange: (patch: Partial<SiteDocument>) => void;
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
  function toggleCard(key: LeaderboardCard, checked: boolean) {
    change({ cards: checked ? [...settings.cards, key] : settings.cards.filter(card => card !== key) });
  }
  function updateCard(key: LeaderboardCard, patch: Partial<LeaderboardCardOptions>) {
    change({ card_options: {
      ...settings.card_options,
      [key]: { minimum: 0, depth: 5, ...settings.card_options?.[key], ...patch },
    } });
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
    <p>Choose the cards and date ranges for your club’s Overall leaderboard. Save the draft and publish the settings to apply your choices.</p>
    {validation ? <p role="alert" className={styles.error}>{validation}</p> : null}
    <section className={styles.card}>
      <h3 style={{ marginTop: 0 }}>Featured cards</h3>
      <p>Select the statistics you want from the choices below, just like league awards. Then set their order and how many players each card shows. Biggest upset shows teams, with both teammates together.</p>
      {LEADERBOARD_CARD_GROUPS.map(group => <fieldset key={group} style={{ border: "1px solid #cbd5e1", borderRadius: 12, padding: "1rem", margin: "1rem 0" }}>
        <legend style={{ fontWeight: 750 }}>{group}</legend>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 230px), 1fr))", gap: "1rem" }}>
          {(Object.keys(LEADERBOARD_CARD_LABELS) as LeaderboardCard[]).filter(key => LEADERBOARD_CARD_DETAILS[key].group === group).map(key => <label key={key} style={{ display: "block", padding: ".75rem", borderRadius: 10, background: settings.cards.includes(key) ? "#eff6ff" : "#f8fafc", cursor: "pointer" }}>
            <span style={{ display: "flex", gap: ".5rem", alignItems: "baseline" }}><input type="checkbox" aria-label={`Show ${LEADERBOARD_CARD_LABELS[key]} card`} checked={settings.cards.includes(key)} onChange={event => toggleCard(key, event.target.checked)} /><strong>{LEADERBOARD_CARD_LABELS[key]}</strong></span>
            <span style={{ display: "block", marginTop: ".4rem", color: "#475569", fontSize: ".9rem" }}>{LEADERBOARD_CARD_DETAILS[key].description}</span>
          </label>)}
        </div>
      </fieldset>)}
      <h4>Selected cards &amp; order</h4>
      <p>Card minimums apply alongside the overall minimum games below. For close-game and partnership cards, use a minimum that gives a meaningful record. Hidden cards keep their settings.</p>
      {settings.cards.length ? <ol style={{ paddingLeft: "1.5rem" }}>
        {settings.cards.map((key, index) => <li key={key} style={{ padding: "1rem 0", borderBottom: "1px solid #e2e8f0" }}>
          <div className={styles.actions}>
            <strong style={{ flex: "1 1 180px" }}>{LEADERBOARD_CARD_LABELS[key]}</strong>
            <button type="button" className={styles.button} disabled={index === 0} aria-label={`Move ${LEADERBOARD_CARD_LABELS[key]} up`} onClick={() => moveCard(index, -1)}>Move up</button>
            <button type="button" className={styles.button} disabled={index === settings.cards.length - 1} aria-label={`Move ${LEADERBOARD_CARD_LABELS[key]} down`} onClick={() => moveCard(index, 1)}>Move down</button>
            <button type="button" className={styles.button} aria-label={`Remove ${LEADERBOARD_CARD_LABELS[key]} card`} onClick={() => toggleCard(key, false)}>Remove</button>
          </div>
          <div className={styles.form} style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 200px), 1fr))", gap: ".75rem", marginTop: ".75rem", maxWidth: 620 }}>
            <label>{key === "biggest_upset" ? "Teams" : "Players"} to show<input type="number" aria-label={`${LEADERBOARD_CARD_LABELS[key]} ${key === "biggest_upset" ? "teams" : "players"} to show`} min={1} max={10} step={1} value={settings.card_options?.[key]?.depth ?? 5} onChange={event => updateCard(key, { depth: Number(event.target.value) })} /></label>
            <label>Minimum {LEADERBOARD_CARD_DETAILS[key].sample}<input type="number" aria-label={`${LEADERBOARD_CARD_LABELS[key]} minimum ${LEADERBOARD_CARD_DETAILS[key].sample}`} min={0} max={10000} step={1} value={settings.card_options?.[key]?.minimum ?? 0} onChange={event => updateCard(key, { minimum: Number(event.target.value) })} /></label>
          </div>
        </li>)}
      </ol> : <p>No featured cards will be shown.</p>}
      <label><input type="checkbox" aria-label="Show summary counts above the cards" checked={settings.show_summary} onChange={event => change({ show_summary: event.target.checked })} /> Show summary counts above the cards</label>
      <div className={styles.form} style={{ maxWidth: 340, marginTop: "1rem" }}>
        <label>Minimum games for performance cards
          <input type="number" aria-label="Minimum games for performance cards" min={0} max={10000} step={1} value={settings.min_games} onChange={event => change({ min_games: Number(event.target.value) })} />
        </label>
      </div>
      <p><small>Applies to every performance card in the selected period. Players need at least one game. For Biggest upset, the minimum counts games the two teammates played together. Highest rating is exempt from this overall minimum; its own card minimum still applies.</small></p>
      <div className={styles.form} style={{ maxWidth: 340 }}>
        <label>Timezone for All time statistics<input aria-label="Timezone for All time statistics" list="leaderboard-timezones" value={settings.timezone} onChange={event => change({ timezone: event.target.value })} /></label>
      </div>
      <p><small>Used to count playing days in All time. Each season uses its own timezone below.</small></p>
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
