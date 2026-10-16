"use client";

import SearchablePlayerSelect from "@/components/SearchablePlayerSelect";
import { useEffect, useRef, useState } from "react";
import { parsePoolPlayerList, poolGender, poolRating, sortedPoolDivisions, type BulkPoolMember, type BulkPoolPreview, type BulkPoolResult, type PoolPlayerChoice, type PoolPlayerChoices, type SeasonPool } from "@/lib/interclubPlayerPool";
import { RequestStatus } from "./PoolPanelCommon";
import { usePoolResource } from "./usePoolResource";
import styles from "./playerPool.module.css";
import { registrationCanAccept, registrationWindowMessage, type SeasonRegistrationWindow } from "@/lib/interclubRegistrationWindow";
import { useRegistrationWindow } from "@/lib/useRegistrationWindow";

type Props = { registration?: SeasonRegistrationWindow; root: string; accessToken: string; divisions: string[]; members: SeasonPool["members"]; onAdded: (result: BulkPoolResult) => void; onClose: () => void };
const statusLabel = { matched: "Linked club player", new: "New signup · needs player link", ambiguous: "Choose matching player", duplicate: "Already in pool · skipped" };

export function PoolBulkAdd({ registration, root, accessToken, divisions, members, onAdded, onClose }: Props) {
  const [query, setQuery] = useState(""), [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<PoolPlayerChoice[]>([]), [pasted, setPasted] = useState("");
  const [selectedDivisions, setSelectedDivisions] = useState<string[]>([]);
  const [preview, setPreview] = useState<BulkPoolPreview | null>(null), [draft, setDraft] = useState<BulkPoolMember[]>([]), [error, setError] = useState("");
  const { canRegister } = useRegistrationWindow(registration);
  const heading = useRef<HTMLHeadingElement | null>(null);
  const resource = usePoolResource<PoolPlayerChoices>(`${root}/pool/players?q=${encodeURIComponent(query)}&offset=${offset}`, accessToken);
  useEffect(() => { heading.current?.focus(); }, []);
  const existing = new Map(members.filter(member => member.player_id).map(member => [String(member.player_id), member]));
  const parsed = parsePoolPlayerList(pasted), requestedCount = selected.length + parsed.members.length;
  function edit() { setPreview(null); setError(""); }
  async function review(entries?: BulkPoolMember[]) {
    if (!registrationCanAccept(registration)) { setPreview(null); setError(registrationWindowMessage(registration)); return; }
    if (!entries && parsed.errors.length) { setError(parsed.errors.join(" ")); return; }
    const next = entries || [
      ...selected.map(player => ({ player_id: String(player.id), name: player.name, divisions: selectedDivisions.length ? selectedDivisions : player.eligible_divisions })),
      ...parsed.members.map(member => ({ ...member, ...(selectedDivisions.length ? { divisions: selectedDivisions } : {}) })),
    ];
    if (!next.length || next.length > 200) { setError("Choose or paste between 1 and 200 players at a time."); return; }
    setError(""); setPreview(null); setDraft(next);
    const result = await resource.perform<BulkPoolPreview>(json => json(`${root}/pool/bulk-preview`, "POST", { members: next }));
    if (result) setPreview(result);
  }
  async function add() {
    if (!registrationCanAccept(registration)) { setPreview(null); setError(registrationWindowMessage(registration)); return; }
    if (!preview || preview.ambiguous_count || !preview.ready_count) return;
    const result = await resource.perform<BulkPoolResult>(json => json(`${root}/pool/bulk-add`, "POST", { members: draft }));
    if (result) onAdded(result);
  }
  return <section className={styles.bulkPanel} aria-label="Add players to season pool">
    <div className={styles.row}><h4 ref={heading} tabIndex={-1}>Add players</h4><button disabled={resource.busy} onClick={onClose}>Close add players</button></div>
    <p>Add players who have already said yes in person or by email. An email address is optional.</p>
    <p className={styles.muted}>After season registration closes, you can place linked, approved players directly into a meet lineup after confirming they can play. Email invitations are optional.</p>
    <RequestStatus {...resource} />{error && <p role="alert" className={styles.error}>{error}</p>}
    {resource.blocked && <button onClick={() => { setPreview(null); resource.reload(); }}>Reload add players</button>}
    {!canRegister && <p className={styles.notice}>{registrationWindowMessage(registration)}</p>}
    <fieldset disabled={!canRegister || resource.busy || resource.blocked}>
      <div className={styles.grid}>
        <div><label>Find club players<input type="search" value={query} maxLength={80} onChange={event => { setQuery(event.target.value); setOffset(0); }} placeholder="Search by name" /></label>
          <div className={styles.candidates}>
            {resource.data?.players.map(player => {
              const existingMember = existing.get(String(player.id)), alreadyAdded = !!existingMember, checked = selected.some(item => String(item.id) === String(player.id));
              return <label key={player.id} className={styles.choice}><input type="checkbox" checked={checked || alreadyAdded} disabled={alreadyAdded} onChange={() => { edit(); setSelected(old => checked ? old.filter(item => String(item.id) !== String(player.id)) : [...old, player]); }} />
                <span><strong>{player.name}</strong><span className={styles.playerMeta}>Club {poolRating(player.rating)}{player.league_rating != null ? ` · League ${poolRating(player.league_rating)}` : ""} · {poolGender(player.gender)}{existingMember?.status === "withdrawn" ? " · Withdrawn; restore in player pool" : alreadyAdded ? " · Already in pool" : ""}</span></span></label>;
            })}
            {resource.data?.players.length === 0 && <p>No matching club players. You can paste their names to add signups below.</p>}
          </div>
          <div className={styles.toolbar}>{offset > 0 && <button disabled={resource.loading} onClick={() => setOffset(0)}>First club players</button>}{resource.data?.next_offset != null && <button disabled={resource.loading} onClick={() => setOffset(resource.data!.next_offset!)}>More club players</button>}</div>
          <p className={styles.muted}>{selected.length} selected{selected.length > 0 && <> · <button className={styles.textButton} onClick={() => { edit(); setSelected([]); }}>Clear selection</button></>}</p>
          {selected.length > 0 && <p className={styles.selectedNames}>{selected.map(player => player.name).join(", ")}</p>}
        </div>
        <div><label>Paste names and optional emails<textarea value={pasted} onChange={event => { edit(); setPasted(event.target.value); }} placeholder={"Alex Garcia\nPat Jones, pat@example.com\nSam Lee <sam@example.com>"} rows={8} /></label><p className={styles.muted}>One player per line. Names are checked against your club’s players before adding. Up to 200 players per batch.</p></div>
      </div>
      <details><summary>Division preferences for this group (optional)</summary><p className={styles.muted}>Leave blank to use each linked player’s eligible divisions. Players without a profile can choose preferences later.</p><div className={styles.toolbar}>{sortedPoolDivisions(divisions).map(division => <label key={division} className={styles.choice}><input type="checkbox" checked={selectedDivisions.includes(division)} onChange={() => { edit(); setSelectedDivisions(old => old.includes(division) ? old.filter(item => item !== division) : [...old, division]); }} />{division}</label>)}</div></details>
      <button disabled={!canRegister || resource.disabled || !requestedCount} onClick={() => void review()}>{resource.busy ? "Checking players…" : `Preview ${requestedCount || ""} player${requestedCount === 1 ? "" : "s"}`}</button>
    </fieldset>
    {preview && <div className={styles.bulkPreview}>
      <h4>Check players before adding</h4><p>{preview.ready_count} ready to add · {preview.duplicate_count} already in pool · {preview.ambiguous_count} need a match</p>
      <div className={styles.tableScroll}><table className={styles.poolTable}><caption>Players to add</caption><thead><tr><th scope="col">Player</th><th scope="col">Profile match</th><th scope="col">Club rating</th><th scope="col">Review</th></tr></thead><tbody>
        {preview.rows.map(row => <tr key={row.index}><th scope="row">{row.name}<span className={styles.playerMeta}>{row.email || "No email"}</span></th><td>{statusLabel[row.status]}{row.status === "ambiguous" && <label className={styles.matchChoice}>Club profile for {row.name}<SearchablePlayerSelect aria-label={["Club profile for",String(row.name)].join(" ")} value="" disabled={!canRegister || resource.disabled} onValueChange={playerValue => { if (playerValue) void review(draft.map((member, index) => index === row.index ? { ...member, player_id: playerValue === "__new__" ? null : playerValue } : member)); }}><option value="">Choose a player…</option>{row.candidates.map(candidate => <option key={candidate.id} value={String(candidate.id)}>{candidate.name} · {poolRating(candidate.rating)} · {poolGender(candidate.gender)}</option>)}<option value="__new__">None of these — add as a new signup</option></SearchablePlayerSelect></label>}</td><td>{poolRating(row.rating)}{row.league_rating != null && <span className={styles.playerMeta}>League {poolRating(row.league_rating)}</span>}</td><td><button disabled={!canRegister || resource.disabled || draft.length === 1} onClick={() => void review(draft.filter((_, index) => index !== row.index))} aria-label={`Remove ${row.name} from this batch`}>Remove</button></td></tr>)}
      </tbody></table></div>
      {preview.rows.some(row => row.status === "new") && <p className={styles.notice}>New names are saved as signups. Link them to a club player before selecting them for a lineup.</p>}
      <p className={styles.muted}>Adding players does not send email. Late signups still need the league organizer’s approval.</p>
      <button className={styles.primary} disabled={!canRegister || resource.disabled || !!preview.ambiguous_count || !preview.ready_count} onClick={() => void add()}>{resource.busy ? "Adding players…" : `Add ${preview.ready_count} player${preview.ready_count === 1 ? "" : "s"} to pool`}</button>
    </div>}
  </section>;
}
