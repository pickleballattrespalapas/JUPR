"use client";
import { useEffect, useRef, useState } from "react";
import MeetSignupGender from "@/components/MeetSignupGender";
import type { SignupSlotGender } from "@/components/MeetSignupQueues";
import { MeetSignupBoard, needsGenderReview, SignupGender, signupGender } from "@/lib/interclubMeetSignup";
import { LineupPlayerChoice, lineupGender, lineupPlayerIssue, MeetRegistrationDetail } from "@/lib/interclubRegistration";
import styles from "./playerPool.module.css";

type Props = {
  root: string; accessToken: string; board: MeetSignupBoard; division: string; gender: SignupSlotGender;
  disabled: boolean; mutationError: string; blocked: boolean; onReload: () => void;
  onAdd: (player: LineupPlayerChoice, gender: SignupGender) => void; onClose: () => void;
};
const rating = (player: LineupPlayerChoice) => Number(player.eligibility_rating ?? player.starting_rating);

export default function MeetSignupPlayerPicker({ root, accessToken, board, division, gender, disabled, mutationError, blocked, onReload, onAdd, onClose }: Props) {
  const [players, setPlayers] = useState<LineupPlayerChoice[]>([]), [assigned, setAssigned] = useState<string[]>([]);
  const [query, setQuery] = useState(""), [loading, setLoading] = useState(true), [error, setError] = useState(""), [retry, setRetry] = useState(0);
  const [selected, setSelected] = useState<LineupPlayerChoice | null>(null), [declared, setDeclared] = useState<SignupGender | "">("");
  const search = useRef<HTMLInputElement>(null), token = useRef(accessToken); token.current = accessToken;
  useEffect(() => { search.current?.focus(); }, []);
  useEffect(() => {
    const controller = new AbortController(); setLoading(true); setError(""); setPlayers([]); setAssigned([]);
    async function read<T,>(url: string): Promise<T> {
      const response = await fetch(url, { signal: controller.signal, headers: { Authorization: `Bearer ${token.current}` }, cache: "no-store" });
      if (!response.ok) throw new Error("Could not load eligible players. Try again.");
      return response.json();
    }
    async function loadPlayers() {
      const rows: LineupPlayerChoice[] = []; let offset: number | null = 0;
      do { const page: { players: LineupPlayerChoice[]; next_offset: number | null } = await read(`${root}/players?offset=${offset}`); rows.push(...page.players); offset = page.next_offset; } while (offset != null && !controller.signal.aborted);
      return rows;
    }
    async function loadAssigned() {
      const ids: string[] = []; let offset: number | null = 0;
      do { const page: MeetRegistrationDetail = await read(`${root}?team_offset=${offset}`);
        for (const team of page.teams) if (team.club_id === board.club.id && !team.withdrawn) for (const player of team.roster) if (player.player_id) ids.push(String(player.player_id));
        offset = page.next_team_offset;
      } while (offset != null && !controller.signal.aborted);
      return ids;
    }
    Promise.all([loadPlayers(), loadAssigned()]).then(([choices, ids]) => { if (!controller.signal.aborted) { setPlayers(choices); setAssigned(ids); } })
      .catch(cause => { if (!controller.signal.aborted) setError(cause.message || "Could not load players. Try again."); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [root, board.club.id, retry]);
  const excluded = new Set([...assigned, ...board.entries.filter(entry => entry.status === "active").map(entry => String(entry.player_id))]);
  const eligible = players.filter(player => !excluded.has(String(player.id)) && !lineupPlayerIssue({ ...player, gender }, division)
    && [gender, "unknown"].includes(lineupGender(player.gender)) && player.name.toLowerCase().includes(query.trim().toLowerCase()))
    .sort((a, b) => rating(b) - rating(a) || a.name.localeCompare(b.name) || a.id.localeCompare(b.id));
  const label = `${division} ${gender === "female" ? "women" : "men"}`;
  const playUp = selected && rating(selected) < (/open/i.test(division) ? 4.5 : Number(division));
  return <section className={styles.slotPicker} aria-label={`Choose player for ${label}`} onKeyDown={event => { if (event.key === "Escape" && !disabled) { event.preventDefault(); onClose(); } }}>
    <h5>Add a player · {label}</h5>
    <p className={styles.muted}>Eligible players, highest league rating first. Playing up joins the waitlist.</p>
    <label>Find a player<input ref={search} value={query} disabled={disabled} onChange={event => { setQuery(event.target.value); setSelected(null); setDeclared(""); }} /></label>
    {loading && <p role="status">Loading eligible players…</p>}
    {error && <p role="alert">{error} <button type="button" onClick={() => setRetry(value => value + 1)}>Retry player lookup</button></p>}
    {!loading && !error && !selected && <>{eligible.length ? <ul className={styles.slotPlayerList} aria-label="Eligible players by rating">{eligible.map(player => <li key={player.id}>
      <button type="button" disabled={disabled} aria-label={`Choose ${player.name} for ${label}`} onClick={() => { setSelected(player); setDeclared(signupGender(player.gender)); }}>
        <strong>{player.name}</strong><span>League rating {rating(player).toFixed(2)}{rating(player) < (/open/i.test(division) ? 4.5 : Number(division)) ? " · Play-up waitlist" : ""}{lineupGender(player.gender) === "unknown" ? " · Confirm gender" : ""}</span>
      </button>
    </li>)}</ul> : <p>No eligible players match. Players already registered or assigned to this meet are omitted.</p>}</>}
    {selected && <form aria-label="Confirm player registration" onSubmit={event => { event.preventDefault(); if (declared && !disabled) onAdd(selected, declared); }}>
      <p><strong>{selected.name}</strong> · League rating {rating(selected).toFixed(2)}</p>
      <MeetSignupGender value={declared} onChange={setDeclared} disabled={disabled} />
      <button type="submit" disabled={disabled || !declared}>{needsGenderReview(declared) ? "Submit for admin review" : playUp ? "Add to play-up waitlist" : "Register player"}</button>
      <button type="button" disabled={disabled} onClick={() => { setSelected(null); setDeclared(""); search.current?.focus(); }}>Choose someone else</button>
    </form>}
    {mutationError && <p role="alert">{mutationError}</p>}{blocked && <button type="button" onClick={onReload}>Reload signup section</button>}
    <button type="button" disabled={disabled && !blocked} onClick={onClose}>Cancel</button>
  </section>;
}
