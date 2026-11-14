"use client";
import { useState } from "react";
import { meetLocalTime, meetUtcTime } from "@/lib/interclubSetup";
import type { ManagedDetails, ManagedMeet, ManagedPlan, ManagedTeam } from "@/lib/managedInterclub";
import styles from "../setup.module.css";

export default function MeetPlanner({ details, meet, disabled, onSave, onCancel }: { details: ManagedDetails; meet?: ManagedMeet; disabled: boolean;
  onSave: (id: string, revision: number, details: ManagedPlan) => Promise<void>; onCancel: () => void }) {
  const [id] = useState(meet?.id || crypto.randomUUID());
  const [plan, setPlan] = useState<ManagedPlan>(meet?.details || { host_club_id: details.clubs[0]?.id || "", club_ids: details.clubs.slice(0, 4).map(c => c.id), starts_at: "", courts: 4, duration_minutes: 180, phase: "regular", format: "gender", teams: [] });
  const [time, setTime] = useState(meetLocalTime(meet?.details.starts_at || null, details.timezone)), [error, setError] = useState("");
  function team(clubId: string, division: string): ManagedTeam | undefined { return plan.teams.find(t => t.club_id === clubId && t.division === division); }
  function toggleTeam(clubId: string, division: string, include: boolean) {
    const pool = details.clubs.find(c => c.id === clubId)?.rosters.find(r => r.division === division)?.roster || [];
    const roster = [...pool.filter(p => p.gender === "female").slice(0, 2), ...pool.filter(p => p.gender === "male").slice(0, 2)];
    setPlan({ ...plan, teams: [...plan.teams.filter(t => !(t.club_id === clubId && t.division === division)), ...(include ? [{ club_id: clubId, division, roster, reserves: pool.filter(p => !roster.some(r => r.entry_id === p.entry_id)) }] : [])] });
  }
  return <form className={styles.panel} onSubmit={async e => { e.preventDefault(); setError(""); try {
    const starts_at = meetUtcTime(time, details.timezone); if (!starts_at) throw new Error("Choose a meet date and time.");
    await onSave(id, meet?.revision || 0, { ...plan, starts_at });
  } catch (reason) { setError((reason as Error).message); } }}>
    <h2>{meet ? "Edit meet & lineups" : "Schedule a meet"}</h2><fieldset disabled={disabled} className={styles.form}><div className={styles.grid}>
      <label className={styles.field}>Date & time ({details.timezone})<input required type="datetime-local" value={time} onChange={e => setTime(e.target.value)} /></label>
      <label className={styles.field}>Host<select aria-label="Host" value={plan.host_club_id} onChange={e => setPlan({ ...plan, host_club_id: e.target.value })}>
        {details.clubs.filter(c => plan.club_ids.includes(c.id)).map(c => <option key={c.id} value={c.id}>{c.name}</option>)}</select></label>
      <label className={styles.field}>Courts<input required type="number" min={1} max={100} value={plan.courts} onChange={e => setPlan({ ...plan, courts: Number(e.target.value) })} /></label>
      <label className={styles.field}>Duration (minutes)<input required type="number" min={30} max={720} value={plan.duration_minutes} onChange={e => setPlan({ ...plan, duration_minutes: Number(e.target.value) })} /></label>
      <label className={styles.field}>Competition<select value={plan.phase} onChange={e => { const phase = e.target.value as ManagedPlan["phase"]; setPlan({ ...plan, phase, format: phase === "regular" ? "gender" : "mlp" }); }}>
        <option value="regular">Regular meet</option><option value="final">Championship final</option><option value="qualifier">Qualifying playoff</option></select></label>
      {plan.phase === "regular" && <label className={styles.field}>Format<select value={plan.format} onChange={e => setPlan({ ...plan, format: e.target.value as ManagedPlan["format"] })}>
        <option value="gender">Women’s & men’s doubles</option><option value="mixed">Mixed doubles</option></select></label>}
    </div><p>Participating clubs (two to four)</p><div className={styles.choices}>{details.clubs.map(c => <label className={styles.choice} key={c.id}>
      <input type="checkbox" checked={plan.club_ids.includes(c.id)} onChange={e => { const club_ids = e.target.checked ? [...plan.club_ids, c.id] : plan.club_ids.filter(id => id !== c.id);
        setPlan({ ...plan, club_ids, host_club_id: club_ids.includes(plan.host_club_id) ? plan.host_club_id : club_ids[0] || "", teams: plan.teams.filter(t => club_ids.includes(t.club_id)) }); }} />{c.name}</label>)}</div>
    <h3>Lineups</h3><p>Choose the divisions playing, then the players for each club. You can save the date now and finish lineups later.</p>
    {details.clubs.filter(c => plan.club_ids.includes(c.id)).map(club => <section key={club.id} className={styles.card}><h4>{club.name}</h4>
      {details.divisions.map(division => { const selected = team(club.id, division), pool = club.rosters.find(r => r.division === division)?.roster || [];
        return <div key={division}><label><input type="checkbox" checked={!!selected} onChange={e => toggleTeam(club.id, division, e.target.checked)} /> {division}</label>
          {selected && <><p>{selected.roster.length} players selected · two women and two men per full team</p>
            {!pool.length && <p>Add this division’s players in Teams & Players, then return here.</p>}
            <div className={styles.choices}>{pool.map(player => <label key={player.entry_id} className={styles.choice}><input type="checkbox" checked={selected.roster.some(p => p.entry_id === player.entry_id)}
              onChange={e => setPlan({ ...plan, teams: plan.teams.map(t => {
                if (t !== selected) return t;
                const roster = e.target.checked ? [...t.roster, player] : t.roster.filter(p => p.entry_id !== player.entry_id);
                return { ...t, roster, reserves: pool.filter(p => !roster.some(r => r.entry_id === p.entry_id)) };
              }) })} />
              {player.name} · {player.gender === "female" ? "W" : "M"}{player.starting_rating == null ? "" : ` · ${player.starting_rating}`}</label>)}</div></>}
        </div>;
      })}</section>)}
    {error && <p role="alert" className={styles.error}>{error}</p>}
    <div className={styles.toolbar}><button className={styles.primary}>Save meet & lineups</button><button type="button" onClick={onCancel}>Cancel</button></div>
    </fieldset>
  </form>;
}
