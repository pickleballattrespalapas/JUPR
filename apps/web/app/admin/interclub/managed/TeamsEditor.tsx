"use client";
import { useState } from "react";
import { readRoster, rosterText, type ManagedClub, type ManagedDetails } from "@/lib/managedInterclub";
import styles from "../setup.module.css";

export default function TeamsEditor({ details, disabled, onSave }: { details: ManagedDetails; disabled: boolean; onSave: (details: ManagedDetails) => Promise<void> }) {
  const [name, setName] = useState("");
  return <section><h2>Teams & players</h2><p>Add each club’s name and captain, then paste its players. Captains can send you the names directly.</p>
    <form className={styles.toolbar} onSubmit={async e => { e.preventDefault(); try { await onSave({ ...details, clubs: [...details.clubs,
      { id: crypto.randomUUID(), name: name.trim(), captain: "", contact: "", rosters: [] }] }); setName(""); } catch { /* Parent displays the request error. */ } }}>
      <label className={styles.field}>Participating club<input required maxLength={120} value={name} onChange={e => setName(e.target.value)} placeholder="e.g. La Ribera" /></label>
      <button disabled={disabled || !name.trim()}>Add club</button></form>
    {details.clubs.map(club => <ClubEditor key={`${club.id}:${JSON.stringify(club)}`} club={club} divisions={details.divisions} disabled={disabled}
      onSave={updated => onSave({ ...details, clubs: details.clubs.map(c => c.id === club.id ? updated : c) })} />)}
  </section>;
}

function ClubEditor({ club, divisions, disabled, onSave }: { club: ManagedClub; divisions: string[]; disabled: boolean; onSave: (club: ManagedClub) => Promise<void> }) {
  const [draft, setDraft] = useState(club), [division, setDivision] = useState(divisions[0]);
  const [women, setWomen] = useState(rosterText(club.rosters.find(r => r.division === division)?.roster || [], "female"));
  const [men, setMen] = useState(rosterText(club.rosters.find(r => r.division === division)?.roster || [], "male"));
  const [error, setError] = useState("");
  function select(value: string) {
    try {
      const roster = readRoster(women, men, draft);
      const next = { ...draft, rosters: [...draft.rosters.filter(r => r.division !== division), { division, roster }] };
      setDraft(next); setDivision(value); setWomen(rosterText(next.rosters.find(r => r.division === value)?.roster || [], "female"));
      setMen(rosterText(next.rosters.find(r => r.division === value)?.roster || [], "male")); setError("");
    } catch (reason) { setError((reason as Error).message); }
  }
  return <details className={styles.panel}><summary><strong>{club.name}</strong> · {new Set(club.rosters.flatMap(r => r.roster.map(p => p.entry_id))).size} players</summary>
    <form onSubmit={async e => { e.preventDefault(); try { const roster = readRoster(women, men, draft); setError("");
      await onSave({ ...draft, rosters: [...draft.rosters.filter(r => r.division !== division), { division, roster }] });
    } catch (reason) { setError((reason as Error).message); } }}>
      <fieldset disabled={disabled} className={styles.form}><div className={styles.grid}>
        <label className={styles.field}>Club name<input required maxLength={120} value={draft.name} onChange={e => setDraft({ ...draft, name: e.target.value })} /></label>
        <label className={styles.field}>Captain<input maxLength={120} value={draft.captain} onChange={e => setDraft({ ...draft, captain: e.target.value })} /></label>
        <label className={styles.field}>Captain’s email or phone<input maxLength={200} value={draft.contact} onChange={e => setDraft({ ...draft, contact: e.target.value })} /><small>Visible only to the organizer.</small></label>
        <label className={styles.field}>Division<select value={division} onChange={e => select(e.target.value)}>{divisions.map(d => <option key={d}>{d}</option>)}</select></label>
        <label className={styles.field}>Women<textarea rows={6} value={women} onChange={e => setWomen(e.target.value)} placeholder={"One player per line\nOptional: Name | 3.75"} /></label>
        <label className={styles.field}>Men<textarea rows={6} value={men} onChange={e => setMen(e.target.value)} placeholder={"One player per line\nOptional: Name | 3.75"} /></label>
      </div><p>Save the player pool here. Choose the two women and two men playing each meet in Schedule.</p>
      {error && <p role="alert">{error}</p>}<button className={styles.primary}>Save club & players</button></fieldset>
    </form>
  </details>;
}
