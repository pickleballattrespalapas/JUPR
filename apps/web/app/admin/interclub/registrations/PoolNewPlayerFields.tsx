"use client";

import styles from "./playerPool.module.css";
import type { NewClubPlayer } from "@/lib/interclubPlayerPool";

export type NewPlayerDraft = { name: string; rating: string; gender: string; email: string };
export const emptyNewPlayer = (): NewPlayerDraft => ({ name: "", rating: "", gender: "", email: "" });
export function newPlayerDetails(draft: NewPlayerDraft): NewClubPlayer | null {
  const name = draft.name.trim(), rating = Number(draft.rating), email = draft.email.trim();
  if (!name || name.length > 120 || !draft.rating.trim() || !Number.isFinite(rating) || rating < 1 || rating > 7 ||
      !["", "male", "female"].includes(draft.gender) || email.length > 254 || email && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) return null;
  return { name, starting_jupr: rating, ...(draft.gender ? { gender: draft.gender as "male" | "female" } : {}), ...(email ? { email } : {}) };
}

export function PoolNewPlayerFields({ draft, onChange }: { draft: NewPlayerDraft; onChange: (draft: NewPlayerDraft) => void }) {
  return <div className={styles.grid}>
    <label>Player name<input aria-label="Player name" required maxLength={120} autoComplete="name" value={draft.name} onChange={event => onChange({ ...draft, name: event.target.value })} /></label>
    <label>Starting JUPR<input aria-label="Starting JUPR" type="number" required min={1} max={7} step="0.01" value={draft.rating} onChange={event => onChange({ ...draft, rating: event.target.value })} placeholder="Enter a reviewed rating" /><span className={styles.muted}>Use a reviewed starting rating from 1.0 to 7.0. No rating is assumed.</span></label>
    <label>Gender (optional)<select aria-label="Gender (optional)" value={draft.gender} onChange={event => onChange({ ...draft, gender: event.target.value })}><option value="">Choose gender</option><option value="female">Women</option><option value="male">Men</option></select><span className={styles.muted}>Confirm gender before selecting a women’s or men’s team place.</span></label>
    <label>Email (optional)<input aria-label="Email (optional)" type="email" maxLength={254} autoComplete="email" value={draft.email} onChange={event => onChange({ ...draft, email: event.target.value })} /></label>
  </div>;
}
