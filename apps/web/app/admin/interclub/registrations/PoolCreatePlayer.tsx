"use client";

import { useEffect, useRef, useState } from "react";
import type { NewPoolPlayerInput } from "@/lib/interclubPlayerPool";
import { emptyNewPlayer, newPlayerDetails, PoolNewPlayerFields } from "./PoolNewPlayerFields";
import styles from "./playerPool.module.css";

export function PoolCreatePlayer({ disabled, onCreate, onClose }: {
  disabled: boolean; onCreate: (request: NewPoolPlayerInput) => Promise<void>; onClose: () => void;
}) {
  const [draft, setDraft] = useState(emptyNewPlayer), [notes, setNotes] = useState("");
  const heading = useRef<HTMLHeadingElement | null>(null);
  useEffect(() => { heading.current?.focus(); }, []);
  const player = newPlayerDetails(draft);
  return <section className={styles.bulkPanel} aria-label="Create a new club player">
    <div className={styles.row}><h4 ref={heading} tabIndex={-1}>Create new player</h4><button disabled={disabled} onClick={onClose}>Close new player</button></div>
    <p>Create their club player record and add them to this season’s pool together. For a player already in the club directory, use Add players.</p>
    <p className={styles.muted}>No email is sent. An optional email is saved as signup contact information; it does not give permission to email the player.</p>
    <form onSubmit={event => { event.preventDefault(); if (!disabled && player) void onCreate({ new_player: player, reason: notes.trim() }); }}><fieldset disabled={disabled}>
      <PoolNewPlayerFields draft={draft} onChange={setDraft} />
      <label>Notes (optional)<textarea maxLength={500} rows={3} value={notes} onChange={event => setNotes(event.target.value)} /></label>
      <button className={styles.primary} type="submit" disabled={disabled || !player}>Create player and add to pool</button>
    </fieldset></form>
  </section>;
}
