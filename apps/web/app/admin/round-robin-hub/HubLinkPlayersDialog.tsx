"use client";

import { useState } from "react";
import { InteractionDialog } from "@/components/interaction";
import { matchesPlayerSearch } from "@/lib/playerSearch";
import type { HubGroup } from "@/lib/adminRoundRobinHubApi";
import styles from "./RoundRobinHubPanel.module.css";

export type PlayerReference = { session_key: string; participant_id: string };
const referenceKey = (ref: PlayerReference) => JSON.stringify([ref.session_key, ref.participant_id]);
export function committedIdentityCount(groups: HubGroup[], refs: PlayerReference[]): number {
  return new Set(refs.flatMap(ref => {
    const player = groups.find(group => group.session_key === ref.session_key)?.players?.find(player => player.participant_id === ref.participant_id);
    return player && (player.on_court || player.reserved || player.current_court || player.reserved_court) ? [player.identity] : [];
  })).size;
}
type Props = {
  groups: HubGroup[]; busy: boolean; disabled: boolean; pending: boolean; error: string;
  initialSelection?: PlayerReference[];
  onClose: () => void; onRetry: () => void; onSubmit: (participants: PlayerReference[]) => void;
};
export default function HubLinkPlayersDialog({ groups, busy, disabled, pending, error, initialSelection = [], onClose, onRetry, onSubmit }: Props) {
  const [selection, setSelection] = useState(initialSelection);
  const [search, setSearch] = useState("");
  const [confirmed, setConfirmed] = useState(false);
  const entries = groups.flatMap(group => (group.players || []).map(player => ({ group, player, ref: { session_key: group.session_key, participant_id: player.participant_id } })));
  const selected = entries.filter(entry => selection.some(ref => referenceKey(ref) === referenceKey(entry.ref)));
  const profiles = new Set(selected.map(entry => entry.player.profile_id).filter(id => id != null));
  const busyIdentities = committedIdentityCount(groups, selection);
  const reason = selected.length !== selection.length ? "A selected roster entry is no longer available. Review your selection." : profiles.size > 1 ? "These entries link to different club profiles and must stay separate." : busyIdentities > 1 ? "Finish, release, or resolve affected assignments before combining these players." : "";
  const frozen = busy || disabled || pending;
  const blocked = frozen || selected.length < 2 || !confirmed || Boolean(reason);
  const close = () => { if (!busy && !pending) onClose(); };
  return <InteractionDialog open phase={busy ? "working" : "ready"} title="Link name variants" description={<p>Select the entries that belong to the same person, including different spellings.</p>} dirty={selection.length > 0} onRequestClose={close}
    actions={<><button type="button" className={styles.button} disabled={busy || pending} onClick={close}>Cancel</button><button type="submit" form="hub-link-players-form" className={styles.primary} disabled={blocked}>Link selected entries</button></>}>
    <form id="hub-link-players-form" onSubmit={event => { event.preventDefault(); if (!blocked) onSubmit(selection); }}>
      <label className={styles.field}>Find roster entry<input aria-label="Find roster entry" type="search" value={search} disabled={frozen} onChange={event => setSearch(event.target.value)} /></label>
      <div className={styles.identityOptions}>{entries.filter(entry => matchesPlayerSearch(`${entry.player.name} ${entry.group.title}`, search)).map(entry => <label key={referenceKey(entry.ref)} className={styles.checkbox}>
        <input type="checkbox" aria-label={`Link ${entry.player.name} from ${entry.group.title} (${entry.player.participant_id})`} checked={selection.some(ref => referenceKey(ref) === referenceKey(entry.ref))} disabled={frozen} onChange={event => { setSelection(current => event.target.checked ? [...current, entry.ref] : current.filter(ref => referenceKey(ref) !== referenceKey(entry.ref))); setConfirmed(false); }} />{entry.player.name} · {entry.group.title}{entry.player.membership_status === "departed" ? " · Departed" : ""}
      </label>)}</div>
      <p className={styles.small}>This shares one identity across groups and consolidates duplicate active entries within each group. Saved results and historical roster entries are preserved. An existing committed lineup stays intact.</p>
      <label className={styles.checkbox}><input type="checkbox" aria-label="Confirm selected entries are the same person" checked={confirmed} disabled={frozen} onChange={event => setConfirmed(event.target.checked)} />I checked the selected entries and confirm they are the same person.</label>
      {reason ? <p role="alert">{reason}</p> : null}{error ? <p role="alert">{error}</p> : null}
      {pending ? <button type="button" className={styles.button} disabled={busy} onClick={onRetry}>Retry saved action</button> : null}
    </form>
  </InteractionDialog>;
}
