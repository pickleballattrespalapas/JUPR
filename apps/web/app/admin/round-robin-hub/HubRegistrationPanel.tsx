"use client";

import { useState, useSyncExternalStore } from "react";
import { QRCodeSVG } from "qrcode.react";
import type { HubCheckin, HubCommandAction, HubGroup, HubRegistration } from "@/lib/adminRoundRobinHubApi";
import HubAddPlayerDialog from "./HubAddPlayerDialog";
import styles from "./RoundRobinHubPanel.module.css";

type Props = {
  apiBase: string; clubId: string; hubId: string; groups: HubGroup[]; registration?: HubRegistration;
  disabled: boolean; busy: boolean; pending: boolean; error: string;
  onCommand: (action: HubCommandAction, payload: Record<string, unknown>) => Promise<boolean>;
  onRetry: () => void;
};

const subscribeOrigin = () => () => {};
const browserOrigin = () => window.location.origin;
const serverOrigin = () => "";

export default function HubRegistrationPanel({ apiBase, clubId, hubId, groups, registration, disabled, busy, pending, error, onCommand, onRetry }: Props) {
  const origin = useSyncExternalStore(subscribeOrigin, browserOrigin, serverOrigin);
  const url = origin ? `${origin}/round-robin-hub/${encodeURIComponent(hubId)}/check-in` : "";
  const [copyMessage, setCopyMessage] = useState("");
  const [filter, setFilter] = useState("all");
  const [adding, setAdding] = useState<HubCheckin | null>(null);
  const [matches, setMatches] = useState<Record<string, string>>({});
  if (!registration) return <p>Registration is unavailable. Refresh the hub to try again.</p>;
  const rows = registration.checkins.filter(row => filter === "all" || row.status === filter);
  const review = (checkin: HubCheckin, disposition: string, extra: Record<string, unknown> = {}) =>
    onCommand("review_checkin", { checkin_id: checkin.id, disposition, ...extra });

  return <div className={styles.root}>
    <section className={styles.card}>
      <div className={styles.heading}><div><h2>Public check-in</h2><p className={styles.muted}>Players enter their name and choose a round robin. Round robins created in this hub add arrivals to the roster and prepare matches automatically.</p></div>
        <span className={`${styles.badge} ${registration.open ? styles.good : ""}`}>{registration.open ? "Check-in open" : "Check-in closed"}</span></div>
      <div className={styles.actions}>
        <button type="button" className={styles.primary} disabled={disabled || (!registration.open && !groups.length)} onClick={() => void onCommand("set_registration", { open: !registration.open })}>{registration.open ? "Close check-in" : "Open check-in"}</button>
        {url ? <><a className={styles.button} href={url} target="_blank" rel="noreferrer">Open public page</a>
          <button type="button" className={styles.button} onClick={async () => { try { await navigator.clipboard.writeText(url); setCopyMessage("Check-in link copied."); } catch { setCopyMessage("Select and copy the link below."); } }}>Copy check-in link</button></> : null}
      </div>
      {!groups.length ? <p>Add your round robins before opening check-in.</p> : null}
      {url ? <div style={{ display: "flex", alignItems: "center", gap: "1.5rem", flexWrap: "wrap", marginTop: "1.25rem" }}>
        <div style={{ background: "white", padding: ".75rem" }}><QRCodeSVG value={url} size={152} marginSize={2} title="Scan to check in for this event" /></div>
        <div style={{ flex: "1 1 16rem", minWidth: 0 }}><strong>Scan to check in</strong><p className={styles.small}>Share this link or display the QR code at the entrance. You can also keep the public page open on a check-in tablet.</p>
          <input aria-label="Public check-in link" value={url} readOnly onFocus={event => event.target.select()} style={{ boxSizing: "border-box", width: "100%", padding: ".7rem", fontSize: ".9rem" }} />
          {copyMessage ? <p role="status">{copyMessage}</p> : null}</div>
      </div> : null}
      <p className={styles.small}>Only active round robins are available. Finished or submitted round robins stop accepting check-ins. Your staff controls and player list stay private.</p>
    </section>
    <section className={styles.card}>
      <div className={styles.heading}><div><h2>Arrivals · {registration.pending_count} to review</h2><p className={styles.small}>Updates automatically every 15 seconds. Automatic arrivals appear as added. Review matching names, full rosters, and check-ins for existing fixed-lineup sessions here.</p></div>
        <label className={styles.field}>Show check-ins<select value={filter} onChange={event => setFilter(event.target.value)}><option value="pending">To review</option><option value="accepted">Added / matched</option><option value="dismissed">Dismissed</option><option value="all">All check-ins</option></select></label></div>
      {!rows.length ? <p className={styles.empty}>{filter === "pending" ? "No check-ins waiting for review." : "No check-ins in this view."}</p> : null}
      {rows.map(checkin => {
        const group = groups.find(item => item.session_key === checkin.session_key);
        const players = (group?.players || []).filter(player => player.active || player.can_remove);
        const sameName = players.filter(player => player.name.trim().toLocaleLowerCase() === checkin.name.trim().toLocaleLowerCase());
        const selected = matches[checkin.id] ?? (sameName.length === 1 ? sameName[0].participant_id : "");
        const validMatch = players.some(player => player.participant_id === selected);
        const available = group && group.status !== "completed" && !group.results_locked && !group.finish_requested;
        return <article key={checkin.id} style={{ borderTop: "1px solid #e2e8f0", padding: "1rem 0" }}>
          <div className={styles.heading}><div><strong>{checkin.name}</strong><div className={styles.small}>{group?.title || "Round robin removed from event"} · {new Date(checkin.created_at).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}</div></div>
            <span className={styles.badge}>{checkin.status === "accepted" ? "Added / matched" : checkin.status === "dismissed" ? "Dismissed" : "To review"}</span></div>
          {checkin.review_reason ? <p className={styles.small}>{checkin.review_reason}</p> : null}
          {checkin.status === "pending" ? <>
            {available && players.length ? <div className={styles.inlineForm} style={{ marginTop: ".75rem" }}>
              <label className={styles.field}>Already on the roster?<select aria-label={`Match ${checkin.name} to roster`} value={selected} disabled={disabled} onChange={event => setMatches(current => ({ ...current, [checkin.id]: event.target.value }))}>
                <option value="">Choose an existing player</option>{players.map(player => <option key={player.participant_id} value={player.participant_id}>{player.display_name || player.name}</option>)}
              </select></label><button type="button" className={styles.button} disabled={disabled || !validMatch} onClick={() => void review(checkin, "existing", { participant_id: selected })}>Confirm on roster</button>
            </div> : null}
            <div className={styles.actions} style={{ marginTop: ".75rem" }}><button type="button" className={styles.primary} disabled={disabled || !available || !group?.can_add_player} onClick={() => setAdding(checkin)}>Add new arrival</button>
              <button type="button" className={styles.button} disabled={disabled} onClick={() => void review(checkin, "dismiss")}>Dismiss</button></div>
            {!available ? <p className={styles.small}>This round robin is no longer accepting players.</p> : !group.can_add_player ? <p className={styles.small}>{group.add_player_unavailable_reason}</p> : null}
          </> : checkin.status === "dismissed" ? <button type="button" className={styles.button} disabled={disabled} onClick={() => void review(checkin, "restore")}>Return to review</button> : null}
        </article>;
      })}
    </section>
    {adding && registration.checkins.some(item => item.id === adding.id && item.status === "pending") ? <HubAddPlayerDialog key={adding.id} apiBase={apiBase} clubId={clubId} groups={groups.filter(group => group.session_key === adding.session_key)} initialName={adding.name} initialSessionKey={adding.session_key}
      disabled={disabled} busy={busy} pending={pending} error={error} onClose={() => setAdding(null)} onRetry={onRetry}
      onSubmit={(_sessionKey, name, playerId) => { void review(adding, "add", { name, ...(playerId == null ? {} : { player_id: playerId }) }).then(ok => { if (ok) setAdding(null); }); }} /> : null}
  </div>;
}
