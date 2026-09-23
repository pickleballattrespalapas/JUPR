"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { clearAdminSession } from "@/lib/adminAuthClient";
import { clearAdminNotifications, getAdminNotifications, updateAdminNotificationPreferences, updateAdminNotificationState, type AdminNotifications, type NotificationResult, type NotificationState } from "@/lib/adminNotificationsApi";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import styles from "./AdminNotificationCenter.module.css";

type Props = { accessToken: string; clubId: string; compact?: boolean };
type Filter = "inbox" | "flagged" | "cleared";

function NotificationCenter({ accessToken, clubId, compact = false }: Props) {
  const [data, setData] = useState<AdminNotifications | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [filter, setFilter] = useState<Filter>("inbox");
  const [category, setCategory] = useState("all");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [preferences, setPreferences] = useState<Record<string, boolean>>({});
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const selectAll = useRef<HTMLInputElement>(null);
  const inFlight = useRef(false);
  const mounted = useRef(true);
  const request = useLatestRequestGuard(`${accessToken}\u0000${clubId}`);

  async function run(operation: () => Promise<NotificationResult>, success = "", read = false) {
    if (inFlight.current) return false;
    inFlight.current = true;
    const generation = request.begin();
    setBusy(true);
    setError(null);
    if (!read) setNotice("");
    try {
      const result = await operation();
      if (!mounted.current || !request.isCurrent(generation)) return false;
      if (result.status === 401 || (read && result.status === 403)) {
        setData(null);
        clearAdminSession();
        return false;
      }
      if (!result.data) {
        if (read) setData(null);
        setError(result.error || "Notifications couldn’t be updated. Please try again.");
        return false;
      }
      setData(result.data);
      if (success) setNotice(success);
      return true;
    } catch {
      if (mounted.current && request.isCurrent(generation)) {
        if (read) setData(null);
        setError("Notifications couldn’t be updated. Please try again.");
      }
      return false;
    } finally {
      inFlight.current = false;
      if (mounted.current && request.isCurrent(generation)) setBusy(false);
    }
  }
  const refresh = () => run(() => getAdminNotifications(accessToken, clubId), "", true);
  useAuthenticatedAutoLoad(accessToken, refresh, clubId);
  const refreshRef = useRef(refresh);
  refreshRef.current = refresh;
  useEffect(() => {
    mounted.current = true;
    const onFocus = () => { if (document.visibilityState === "visible") void refreshRef.current(); };
    window.addEventListener("focus", onFocus);
    document.addEventListener("visibilitychange", onFocus);
    const timer = window.setInterval(onFocus, 60_000);
    return () => {
      mounted.current = false;
      window.removeEventListener("focus", onFocus);
      document.removeEventListener("visibilitychange", onFocus);
      window.clearInterval(timer);
    };
  }, []);

  function openSettings() {
    if (!data) return;
    setPreferences(Object.fromEntries(data.categories.map(item => [item.key, item.enabled])));
    setSettingsOpen(true);
  }
  async function savePreferences() {
    if (await run(() => updateAdminNotificationPreferences(accessToken, clubId, preferences), "Notification preferences saved.")) setSettingsOpen(false);
  }
  const changeState = (key: string, state: NotificationState) => run(
    () => updateAdminNotificationState(accessToken, clubId, key, state),
    state === "cleared" ? "Notice cleared." : state === "flagged" ? "Notice flagged to keep it visible." : "Notice returned to your inbox."
  );
  const items = data?.items ?? [];
  const counts = { inbox: items.filter(item => item.state !== "cleared").length, flagged: items.filter(item => item.state === "flagged").length, cleared: items.filter(item => item.state === "cleared").length };
  const filtered = items.filter(item => (filter === "inbox" ? item.state !== "cleared" : item.state === filter) && (category === "all" || item.category === category))
    .sort((a, b) => Number(b.state === "flagged") - Number(a.state === "flagged") || (b.occurred_at ?? "").localeCompare(a.occurred_at ?? "") || a.key.localeCompare(b.key));
  const displayed = compact ? filtered.slice(0, 6) : filtered;
  const selectable = displayed.filter(item => item.state !== "cleared");
  const selectedKeys = selectable.filter(item => selected.has(item.key)).map(item => item.key);
  const allSelected = selectable.length > 0 && selectedKeys.length === selectable.length;
  const selectableKey = JSON.stringify(selectable.map(item => item.key));
  useEffect(() => {
    // A filter change or refreshed feed must never leave hidden selections.
    const visible = new Set<string>(JSON.parse(selectableKey));
    setSelected(current => {
      const next = new Set([...current].filter(key => visible.has(key)));
      return next.size === current.size ? current : next;
    });
  }, [selectableKey]);
  useEffect(() => {
    if (selectAll.current) selectAll.current.indeterminate = selectedKeys.length > 0 && !allSelected;
  }, [selectedKeys.length, allSelected]);
  function toggleSelected(key: string, checked: boolean) {
    setSelected(current => {
      const next = new Set(current);
      if (checked) next.add(key); else next.delete(key);
      return next;
    });
  }
  async function clearSelected() {
    if (!selectedKeys.length) return;
    const keys = [...selectedKeys];
    if (await run(() => clearAdminNotifications(accessToken, clubId, keys), `${keys.length} ${keys.length === 1 ? "notice" : "notices"} cleared.`)) setSelected(new Set());
  }
  const unavailable = data?.categories.filter(item => item.enabled && item.status === "unavailable") ?? [];
  const queues = data?.categories.filter(item => item.kind === "action") ?? [];
  const enabled = data?.categories.filter(item => item.enabled).length ?? 0;
  useEffect(() => {
    if (data && category !== "all" && !data.categories.some(item => item.key === category && item.enabled)) setCategory("all");
  }, [data, category]);

  return <section className={styles.center} aria-labelledby="notification-heading" aria-busy={busy}>
    <div className={styles.heading}>
      <div><p className={styles.eyebrow}>Your club workspace</p><h2 id="notification-heading">Notifications</h2></div>
      <div className={styles.controls}>
        <button type="button" disabled={busy || !data} onClick={openSettings}>Notification settings</button>
        <button type="button" disabled={busy} onClick={() => void refresh()}>{busy ? "Updating…" : "Refresh"}</button>
      </div>
    </div>
    <p className={styles.explanation}>Choose what you follow. Flag a notice to keep it visible, or clear it when you’re done.</p>
    <p className={styles.summary} role="status" aria-live="polite">{!data ? busy ? "Checking notifications…" : "Notifications unavailable" : counts.inbox
      ? `${counts.inbox}${data.truncated || unavailable.length ? "+" : ""} ${counts.inbox === 1 && !data.truncated && !unavailable.length ? "notice" : "notices"} in your inbox`
      : unavailable.length ? "Some notifications couldn’t be checked" : !data.categories.length ? "No notification categories are available for your role" : !enabled ? "Notifications are turned off" : "No new notices in your selected categories"}</p>
    {error ? <p role="alert" className={styles.warning}>{error}</p> : null}
    {notice ? <p role="status" className={styles.feedback}>{notice}</p> : null}

    {settingsOpen && data ? <section className={styles.settings} aria-labelledby="notification-settings-heading">
      <h3 id="notification-settings-heading">What would you like to follow?</h3>
      <p>These in-app preferences apply to you in this club. Other staff keep their own settings.</p>
      {(["action", "activity"] as const).map(kind => <fieldset key={kind} disabled={busy}>
        <legend>{kind === "action" ? "Pending work" : "Club activity"}</legend>
        {data.categories.filter(item => item.kind === kind).map(item => <label key={item.key} className={styles.preference}>
          <input type="checkbox" checked={preferences[item.key] ?? item.enabled} onChange={event => setPreferences(current => ({ ...current, [item.key]: event.target.checked }))} />
          <span><strong>{item.label}</strong><span>{item.description}</span></span>
        </label>)}
      </fieldset>)}
      <p className={styles.note}>Turning off a category hides its notices and keeps your saved flags. Pending tasks remain in their review queues.</p>
      <div className={styles.controls}><button type="button" className={styles.primary} disabled={busy} onClick={() => void savePreferences()}>Save preferences</button><button type="button" disabled={busy} onClick={() => setSettingsOpen(false)}>Cancel</button></div>
    </section> : null}

    {data ? <>
      <div className={styles.filters}>
        <div className={styles.tabs} role="group" aria-label="Notification view">
          {(["inbox", "flagged", "cleared"] as const).map(view => <button key={view} type="button" disabled={busy} aria-pressed={filter === view} onClick={() => { setSelected(new Set()); setFilter(view); }}>{view === "inbox" ? "Inbox" : view === "flagged" ? "Flagged" : "Cleared"} <span>{counts[view]}</span></button>)}
        </div>
        {!compact ? <label className={styles.category}>Category <select disabled={busy} value={category} onChange={event => { setSelected(new Set()); setCategory(event.target.value); }}><option value="all">All categories</option>{data.categories.filter(item => item.enabled).map(item => <option key={item.key} value={item.key}>{item.label}</option>)}</select></label> : null}
      </div>
      {selectable.length ? <div className={styles.bulkControls} role="group" aria-label="Bulk notification actions">
        <label className={styles.selectAll}><input ref={selectAll} type="checkbox" disabled={busy} checked={allSelected} aria-label="Select all shown notifications" onChange={event => setSelected(event.target.checked ? new Set(selectable.map(item => item.key)) : new Set())} />Select all shown</label>
        <span className={styles.selectionCount} aria-live="polite">{selectedKeys.length} selected</span>
        <button type="button" className={styles.primary} disabled={busy || !selectedKeys.length} onClick={() => void clearSelected()}>Clear selected{selectedKeys.length ? ` (${selectedKeys.length})` : ""}</button>
      </div> : null}
      {displayed.length ? <ul className={styles.items}>{displayed.map(item => <li key={item.key} className={`${styles.item} ${item.state === "flagged" ? styles.flagged : ""}`}>
        {item.state !== "cleared" ? <label className={styles.itemSelect}><input type="checkbox" disabled={busy} checked={selected.has(item.key)} aria-label={`Select ${item.title}`} onChange={event => toggleSelected(item.key, event.target.checked)} /></label> : null}
        <div className={styles.itemBody}>
          <div className={styles.itemMeta}><span>{item.state === "flagged" ? "⚑ Flagged · " : ""}{item.kind === "action" ? "Pending work" : "Club activity"}</span>{item.occurred_at && Number.isFinite(Date.parse(item.occurred_at)) ? <time dateTime={item.occurred_at}>{new Date(item.occurred_at).toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" })}</time> : null}</div>
          <Link href={item.href} className={styles.itemLink}>{item.title}<span aria-hidden="true"> →</span></Link>
          {item.description ? <p>{item.description}</p> : null}
        </div>
        <div className={styles.itemActions}>
          {item.kind === "action" ? <Link className={styles.reviewLink} href={item.href} aria-label={`Review ${item.title}`}>Review →</Link> : null}
          {item.state === "cleared" ? <button type="button" disabled={busy} onClick={() => void changeState(item.key, "new")} aria-label={`Restore ${item.title}`}>Restore</button> : <>
            <button type="button" disabled={busy} aria-pressed={item.state === "flagged"} onClick={() => void changeState(item.key, item.state === "flagged" ? "new" : "flagged")} aria-label={`${item.state === "flagged" ? "Unflag" : "Flag"} ${item.title}`}>{item.state === "flagged" ? "Unflag" : "Flag"}</button>
            <button type="button" disabled={busy} onClick={() => void changeState(item.key, "cleared")} aria-label={`Clear ${item.title}`}>Clear</button>
          </>}
        </div>
      </li>)}</ul> : <p className={styles.empty}>{filter === "flagged" ? "No flagged notices. Use Flag on any notice you want to keep." : filter === "cleared" ? "No cleared notices in the available history." : "There are no notices to show here."}</p>}
      {compact ? <Link className={styles.allLink} href="/admin/notifications">{filtered.length > displayed.length ? `View all ${filtered.length} notices` : "Open notification center"} →</Link> : null}
      {unavailable.length ? <div role="alert" className={styles.warning}><p>We couldn’t check these categories. Their notices may be missing.</p><ul>{unavailable.map(item => <li key={item.key}><Link href={item.href}>{item.label}</Link></li>)}</ul></div> : null}
      {data.truncated ? <p className={styles.warning}>This view has reached its history limit. Open the relevant program or review queue to see every item.</p> : null}
      <p className={styles.note}>Clearing only removes your notice. It does not approve, delete, or finish the underlying task. Activity covers the last {data.history_days} days; flagged activity stays until you clear it.</p>
      {queues.length ? <details className={styles.queues}><summary>All review queues · Pending work stays here when you clear a notice</summary><ul>{queues.map(item => <li key={item.key}><Link href={item.href}>{item.label}</Link><span>{item.status === "unavailable" ? "Unavailable" : `${item.total_count} pending`}</span></li>)}</ul></details> : null}
      <p className={styles.updated}>Checked at {new Date(data.checked_at).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })} · Refreshes automatically while this page is open.</p>
    </> : null}
  </section>;
}

export default function AdminNotificationCenter(props: Props) {
  // A new identity/workspace starts with an empty view before any effect runs.
  return <NotificationCenter key={`${props.accessToken}\u0000${props.clubId}`} {...props} />;
}
