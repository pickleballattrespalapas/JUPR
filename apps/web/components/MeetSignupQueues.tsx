import type { ReactNode } from "react";
import { MeetSignupBoard, MeetSignupEntry, signupPlacement } from "@/lib/interclubMeetSignup";
import styles from "./meetSignupQueues.module.css";

export type SignupSlotGender = "female" | "male";
export default function MeetSignupQueues({ board, actions, onAdd, addingDisabled, picker }: {
  board: MeetSignupBoard; actions?: (entry: MeetSignupEntry) => ReactNode;
  onAdd?: (division: string, gender: SignupSlotGender, trigger: HTMLButtonElement) => void;
  addingDisabled?: boolean; picker?: (division: string, gender: SignupSlotGender) => ReactNode;
}) {
  const active = board.entries.filter(entry => entry.status === "active");
  return <div className={styles.queues}>{board.season.divisions.map(division => {
    const entries = active.filter(entry => entry.division === division);
    const review = entries.filter(entry => entry.placement === "review");
    const row = (entry: MeetSignupEntry) => <li key={entry.id} data-placement={entry.placement}>
      <strong>{entry.name}</strong> <span>{entry.rating == null ? "Rating needed" : Number(entry.rating).toFixed(2)}</span>
      <small>{signupPlacement(entry)}</small>{entry.priority === "play_up" && <small>{entry.reason}</small>}{actions?.(entry)}
    </li>;
    return <section key={division} className={styles.division} aria-label={`${division} signup queue`}>
      <h3>{division} division</h3><div className={styles.columns}>
        {([["female", "Women"], ["male", "Men"]] as const).map(([gender, label]) => {
          const confirmed = entries.filter(entry => entry.gender === gender && entry.placement === "confirmed");
          const waiting = entries.filter(entry => entry.gender === gender && entry.placement === "waitlist").sort((a, b) => (a.queue_position || 0) - (b.queue_position || 0));
          return <section key={gender} aria-label={`${division} ${label.toLowerCase()}`}>
            <h4>{label} <span>{confirmed.length} / 2 spots</span></h4>
            {confirmed.length > 0 && <ul aria-label={`${label} with reserved spots`}>{confirmed.map(row)}</ul>}
            {confirmed.length < 2 && (onAdd ? <button type="button" className={styles.openSpot} disabled={addingDisabled} aria-label={`Add player to ${division} ${label.toLowerCase()}`} onClick={event => onAdd(division, gender, event.currentTarget)}>
              <span>{2 - confirmed.length} {confirmed.length === 1 ? "spot" : "spots"} available</span><strong>+ Add player</strong>
            </button> : <p className={styles.vacancy}>{2 - confirmed.length} {confirmed.length === 1 ? "spot" : "spots"} available</p>)}
            {picker?.(division, gender)}
            <h5>Substitute pool</h5>{waiting.length ? <ol>{waiting.map(row)}</ol> : <p>No substitutes yet.</p>}
            {onAdd && <button type="button" disabled={addingDisabled} aria-label={`Add substitute to ${division} ${label.toLowerCase()}`} onClick={event => onAdd(division, gender, event.currentTarget)}>+ Add substitute</button>}
          </section>;
        })}
      </div>{review.length > 0 && <section><h4>Waiting for club review</h4><ul>{review.map(entry => <li key={entry.id}><strong>{entry.name}</strong><small>{entry.reason}</small>{actions?.(entry)}</li>)}</ul></section>}
    </section>;
  })}</div>;
}
