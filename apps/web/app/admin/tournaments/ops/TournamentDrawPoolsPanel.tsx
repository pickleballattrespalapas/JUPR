"use client";

import { useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import type { ActionCompletion } from "@/components/interaction";
import type { TournamentDrawPoolConfiguration } from "@/lib/adminTournamentApi";
import { inspectPoolConfiguration, moveTeamToPool, seededPoolConfiguration, type PoolEditorTeam } from "@/lib/tournamentDrawPools.mjs";

const field = { padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit", width: "100%" };
const button = { padding: "0.55rem 0.8rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit", background: "white" };

type Props = {
  teams: PoolEditorTeam[];
  savedConfiguration: TournamentDrawPoolConfiguration | null;
  hasGames: boolean;
  disabled: boolean;
  busy: boolean;
  onDirtyChange: (dirty: boolean) => void;
  onSave: (configuration: TournamentDrawPoolConfiguration | null, confirmation: string) => Promise<ActionCompletion>;
};

export default function TournamentDrawPoolsPanel({ teams, savedConfiguration, hasGames, disabled, busy, onDirtyChange, onSave }: Props) {
  const [configuration, setConfiguration] = useState<TournamentDrawPoolConfiguration | null>(() => savedConfiguration ? structuredClone(savedConfiguration) : null);
  const review = inspectPoolConfiguration(configuration, teams);
  const dirty = JSON.stringify(configuration) !== JSON.stringify(savedConfiguration);
  const locked = disabled || busy || hasGames;
  const canSplit = teams.length >= 4 && teams.length <= 16;
  function change(next: TournamentDrawPoolConfiguration | null) {
    setConfiguration(next);
    onDirtyChange(JSON.stringify(next) !== JSON.stringify(savedConfiguration));
  }
  return <article style={{ border: "1px solid #bfdbfe", borderRadius: "14px", padding: "1rem", background: "#eff6ff" }} aria-label="Draw pools">
    <h2 style={{ marginTop: 0 }}>Split into pools</h2>
    <p>Assign teams to round-robin pools inside this draw. Each pool keeps its own standings; qualifiers join one playoff bracket and one division podium.</p>
    {hasGames ? <p role="status"><strong>Pool assignments are locked because games exist.</strong> Review the saved split below. Qualifiers and seeds can still be overridden in the playoff review after pool play.</p> : <p>Save your teams first, then review the pools before generating games. A three-team pool plays two matches per team, with a rotating bye.</p>}
    {!configuration ? <>
      <p>{savedConfiguration ? "The split will be removed when you save. All teams will play in one round-robin." : "This draw currently uses one round-robin."}</p>
      <button type="button" style={button} disabled={locked || !canSplit} onClick={() => change(seededPoolConfiguration(teams, 2))}>Split into two pools</button>
      {!canSplit ? <p>Save 4–16 teams to enable pool play.</p> : null}
    </> : <>
      <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap", alignItems: "end" }}>
        <label>Number of pools<select aria-label="Number of pools" style={field} value={configuration.pools.length} disabled={locked} onChange={event => change(seededPoolConfiguration(teams, Number(event.target.value)))}>
          {Array.from({ length: Math.max(0, Math.min(8, Math.floor(teams.length / 2)) - 1) }, (_, index) => index + 2).map(count => <option key={count} value={count}>{count}</option>)}
        </select></label>
        <button type="button" style={button} disabled={locked} onClick={() => change(seededPoolConfiguration(teams, configuration.pools.length))}>Distribute by seed</button>
        <button type="button" style={button} disabled={locked} onClick={() => change(null)}>Remove pool split</button>
      </div>
      <p style={{ color: "#475569" }}>Distribution follows the saved team seed order in a snake pattern. Move any team below to adjust it.</p>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "0.75rem" }}>
        {configuration.pools.map((pool, index) => <fieldset key={pool.id} style={{ border: "1px solid #cbd5e1", borderRadius: "10px", minWidth: 0 }}>
          <legend>{pool.name || `Pool ${index + 1}`}</legend>
          <label>Pool name<input aria-label={`Pool ${pool.id} name`} style={field} maxLength={80} value={pool.name} disabled={locked} onChange={event => change({ ...configuration, pools: configuration.pools.map(row => row.id === pool.id ? { ...row, name: event.target.value } : row) })} /></label>
          <p>{pool.team_ids.length} teams · {review.poolMatches[index]?.matches || 0} matches · {Math.max(0, pool.team_ids.length - 1)} per team</p>
          <label>Teams advancing<select aria-label={`${pool.name} teams advancing`} style={field} value={pool.advance_count} disabled={locked} onChange={event => change({ ...configuration, pools: configuration.pools.map(row => row.id === pool.id ? { ...row, advance_count: Number(event.target.value) } : row) })}>
            {Array.from({ length: pool.team_ids.length + 1 }, (_, count) => <option key={count} value={count}>{count}</option>)}
          </select></label>
        </fieldset>)}
      </div>
      <div style={{ overflowX: "auto", marginTop: "1rem" }}><table style={{ width: "100%", textAlign: "left" }}>
        <caption style={{ textAlign: "left", fontWeight: 700, marginBottom: "0.5rem" }}>Team assignments</caption>
        <thead><tr><th>Team</th><th>Pool</th></tr></thead>
        <tbody>{teams.map(team => <tr key={team.id}><th scope="row" style={{ padding: "0.5rem" }}>{team.label || `Team ${team.team_number}`}</th><td style={{ padding: "0.5rem" }}>
          <select aria-label={`Pool for ${team.label || `Team ${team.team_number}`}`} style={field} value={configuration.pools.find(pool => pool.team_ids.includes(team.id))?.id || ""} disabled={locked} onChange={event => change(moveTeamToPool(configuration, team.id, event.target.value))}>
            <option value="" disabled>Choose a pool…</option>{configuration.pools.map(pool => <option key={pool.id} value={pool.id}>{pool.name}</option>)}
          </select>
        </td></tr>)}</tbody>
      </table></div>
      <p><strong>{review.totalMatches} pool matches · {review.totalQualifiers} playoff teams</strong></p>
      {review.uneven ? <p>Uneven pools: teams play different numbers of matches. Qualifiers are selected within each pool; raw wins are not compared across pools.</p> : null}
      <p>Standings use wins, tied-team head-to-head, point differential, points scored, then original team number. Review ties and confirm or override the advancing teams before generating playoffs.</p>
      {configuration.pools.length === 2 && configuration.pools.every(pool => pool.advance_count === 2) ? <p>Recommended semifinals: {configuration.pools[0].name} first vs {configuration.pools[1].name} second, and {configuration.pools[1].name} first vs {configuration.pools[0].name} second.</p> : null}
    </>}
    {review.errors.length ? <ul role="alert">{review.errors.map(error => <li key={error}>{error}</li>)}</ul> : null}
    {dirty ? <p role="status"><strong>Unsaved pool changes.</strong> Save or reset before generating games.</p> : null}
    <div style={{ display: "flex", gap: "0.65rem", flexWrap: "wrap", marginTop: "1rem" }}>
      <ConfirmAction triggerLabel="Save pools" title="Save this pool setup?" description={configuration ? `${review.totalMatches} pool matches will be generated later. ${review.totalQualifiers} teams will be recommended for the shared playoff, subject to your final review.` : "This removes the pool split. The draw will use one round-robin when games are generated."} confirmLabel="Yes, save pools" confirmationText="SAVE POOLS" disabled={locked || !dirty || !review.ok} busy={busy} onConfirm={confirmation => onSave(configuration, confirmation)} />
      <button type="button" style={button} disabled={busy || !dirty} onClick={() => change(savedConfiguration ? structuredClone(savedConfiguration) : null)}>Reset to saved pools</button>
    </div>
  </article>;
}
