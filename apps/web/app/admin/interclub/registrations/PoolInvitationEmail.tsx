"use client";
import { useEffect, useState } from "react";
import type { InvitationAudience, InvitationBatch, InvitationDelivery, InvitationPreview } from "@/lib/interclubPlayerPool";
import { usePoolResource } from "./usePoolResource";
import { RequestStatus, ShareLink } from "./PoolPanelCommon";
import styles from "./playerPool.module.css";

export function InvitationEmail({ root, accessToken, kind, meetId, onPrepared }: { root: string; accessToken: string; kind: "season" | "meet"; meetId?: string; onPrepared?: () => void }) {
  const query = `kind=${kind}${meetId ? `&meet_id=${encodeURIComponent(meetId)}` : ""}`;
  const resource = usePoolResource<InvitationAudience>(`${root}/audience?${query}`, accessToken);
  const [selected, setSelected] = useState<string[]>([]), [subject, setSubject] = useState(""), [message, setMessage] = useState("");
  const [preview, setPreview] = useState<InvitationPreview | null>(null), [batch, setBatch] = useState<InvitationBatch | null>(null), [search, setSearch] = useState("");
  const recoveryKey = `pcs_interclub_invitation:${root}:${kind}:${meetId || "season"}`;
  const [operationKey, setOperationKey] = useState(() => { try { return typeof window !== "undefined" ? window.sessionStorage.getItem(recoveryKey) || "" : ""; } catch { return ""; } });
  useEffect(() => { if (resource.data) { setSubject(resource.data.defaults.subject); setMessage(resource.data.defaults.message); setSelected([]); setPreview(null); } }, [resource.data]);
  function edit(change: () => void) { change(); setPreview(null); setBatch(null); }
  const body = { kind, ...(meetId ? { meet_id: meetId } : {}), recipient_ids: selected, subject, message };
  async function showPreview() {
    const next = await resource.perform<InvitationPreview>(json => json(`${root}/preview`, "POST", body)); if (next) setPreview(next);
  }
  function remember(key: string) { setOperationKey(key); try { if (key) window.sessionStorage.setItem(recoveryKey, key); else window.sessionStorage.removeItem(recoveryKey); } catch { /* Keep the in-memory key if session storage is unavailable. */ } }
  async function deliver(existing?: InvitationBatch) {
    if (!existing && !preview) return;
    const key = existing?.operation_key || crypto.randomUUID();
    const next = await resource.perform<InvitationBatch>(async json => {
      remember(key);
      const queued = existing || await json<InvitationBatch>(root, "POST", { ...body, operation_key: key, preview_fingerprint: preview!.preview_fingerprint });
      setBatch(queued);
      const rows: InvitationDelivery[] = [];
      for (const recipient of queued.recipients) {
        if (["pending", "queued"].includes(recipient.status)) {
          const result = await json<InvitationDelivery>(`${root}/${encodeURIComponent(key)}/recipients/${recipient.index}/send`, "POST", {});
          rows.push({ ...recipient, ...result });
        } else rows.push(recipient);
      }
      return { ...queued, recipients: rows, pending_count: rows.filter(row => ["pending", "queued"].includes(row.status)).length };
    });
    if (next) { setBatch(next); setPreview(null); if (!next.pending_count) remember(""); onPrepared?.(); }
  }
  async function recover() {
    const next = await resource.perform<InvitationBatch>(json => json(`${root}/${encodeURIComponent(operationKey)}`), status => { if (status === 404) { remember(""); setPreview(null); setBatch(null); } });
    if (next) { setBatch(next); setPreview(null); if (!next.pending_count) remember(""); }
  }
  const dryRun = resource.data?.delivery_mode === "dry_run";
  return <section className={styles.card} aria-label={`${kind === "season" ? "Season" : "Meet"} invitation email`}>
    <h4>{kind === "season" ? "Invite existing club players" : "Invite season pool players"}</h4><RequestStatus {...resource} />
    {operationKey && !resource.busy && <div className={styles.notice}><p>An invitation batch is saved. Check it before preparing another batch.</p><button disabled={resource.disabled} onClick={() => void recover()}>Check saved invitations</button></div>}
    {resource.data && <>
      <p className={styles.muted}>{kind === "season" ? "Only players with an email recorded at your club can receive an invitation. You can also share the season signup link." : "Choose active players from this club’s season pool. Existing replies will be preserved if you invite a player again."}</p>
      <fieldset disabled={resource.disabled || !!operationKey}>
        <label>Find recipients<input type="search" value={search} onChange={event => setSearch(event.target.value)} /></label>
        <div className={styles.toolbar}><button type="button" onClick={() => edit(() => setSelected(resource.data!.candidates.filter(row => row.available).slice(0, 200).map(row => row.id)))}>Select eligible players (up to 200)</button><button type="button" onClick={() => edit(() => setSelected([]))}>Clear selection</button><span>{selected.length} selected</span></div>
        <div className={styles.candidates}>{resource.data.candidates.filter(row => `${row.name} ${row.email}`.toLowerCase().includes(search.toLowerCase())).map(row => <label className={styles.choice} key={row.id}>
          <input type="checkbox" checked={selected.includes(row.id)} disabled={!row.available || (!selected.includes(row.id) && selected.length >= 200)} onChange={() => edit(() => setSelected(old => old.includes(row.id) ? old.filter(id => id !== row.id) : [...old, row.id]))} /><span><strong>{row.name}</strong><br />{row.email || "No email recorded"}{!row.available && <><br />{row.unavailable_reason || "Not available for invitation"}</>}</span>
        </label>)}</div>
        {!resource.data.candidates.length && <p>{kind === "season" ? "No player contacts are available. Share the season signup link with your players." : "No active season signups yet. Share your season signup link first."}</p>}
        <label>Email subject<input maxLength={160} value={subject} onChange={event => edit(() => setSubject(event.target.value))} /></label><label style={{ marginTop: ".8rem" }}>Message<textarea maxLength={4000} value={message} onChange={event => edit(() => setMessage(event.target.value))} /></label>
        <div className={styles.toolbar}><button type="button" disabled={!selected.length || !subject.trim() || !message.trim()} onClick={() => void showPreview()}>Preview invitations</button></div>
      </fieldset>
      {preview && <div className={styles.card}><h4>Review {preview.recipient_count} invitation{preview.recipient_count === 1 ? "" : "s"}</h4><p>{preview.recipients.map(row => `${row.name} (${row.email})`).join(", ")}</p><p><strong>{preview.preview.subject}</strong></p><div className={styles.preview}>{preview.preview.text}</div>
        <p>{preview.delivery_mode === "dry_run" ? "No email will be sent. Test invitations will be prepared so you can copy their links." : "Send these invitations to the selected players."}</p>{!preview.send_available && <p role="status">{preview.send_unavailable_reason || "Invitation sending is unavailable right now."}</p>}<button className={styles.primary} disabled={resource.disabled || !preview.send_available || !!operationKey} onClick={() => void deliver()}>{resource.busy ? "Preparing invitations…" : preview.delivery_mode === "dry_run" ? "Prepare test invitations" : "Send invitations"}</button>
      </div>}
      {batch && <div role="status" className={styles.notice}><strong>{batch.delivery_mode === "dry_run" ? "Test invitation results — no emails sent" : "Invitation results"}</strong><ul>{batch.recipients.map(row => <li key={row.index}>{row.name || row.email}: {row.detail || row.status}{row.links?.map((link, index) => <ShareLink key={index} label={`${link.name || row.name || "Player"} invitation link`} url={link.url} />)}</li>)}</ul>
        {!!batch.pending_count && <button disabled={resource.disabled} onClick={() => void deliver(batch)}>Continue pending invitations</button>}{batch.delivery_mode === "dry_run" && !batch.pending_count && <p>{kind === "season" ? "Use the season signup link above to test a player signup." : "The private reply links are listed with each player’s response below."}</p>}
      </div>}
      {dryRun && !preview && !batch && <p className={styles.muted}>Preview first, then prepare test invitations. No email is sent in staging.</p>}
    </>}
    {resource.error && <button disabled={resource.busy || resource.loading} onClick={resource.reload}>Reload invitation options</button>}
  </section>;
}
