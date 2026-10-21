"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import styles from "./checkin.module.css";

type Registration = {
  event: { id: string; title: string; registration_open: boolean; finished: boolean };
  groups: { session_key: string; title: string }[];
};

export default function PublicHubCheckin({ apiBase, hubId }: { apiBase: string | null; hubId: string }) {
  const [data, setData] = useState<Registration | null>(null);
  const [name, setName] = useState("");
  const [sessionKey, setSessionKey] = useState("");
  const [error, setError] = useState("");
  const [loadError, setLoadError] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [sent, setSent] = useState<{ name: string; group: string } | null>(null);
  const lock = useRef(false);
  const input = useRef<HTMLInputElement>(null);
  const base = `${apiBase?.replace(/\/$/, "")}/public/round-robin-hubs/${encodeURIComponent(hubId)}`;

  const refresh = useCallback(async (signal?: AbortSignal) => {
    if (!apiBase) { setLoadError("Check-in is temporarily unavailable. Please ask the event organizer."); setLoading(false); return; }
    try {
      const response = await fetch(`${base}/registration`, { cache: "no-store", signal });
      const value = await response.json();
      if (!response.ok) throw new Error(typeof value.detail === "string" ? value.detail : "Unable to load this event.");
      if (!signal?.aborted) { setData(value); setLoadError(""); }
    } catch (cause) {
      if (!signal?.aborted) setLoadError(cause instanceof Error ? cause.message : "Unable to load this event. Please retry.");
    } finally { if (!signal?.aborted) setLoading(false); }
  }, [apiBase, base]);

  useEffect(() => {
    const controller = new AbortController();
    void refresh(controller.signal);
    const reload = () => { if (document.visibilityState === "visible" && !lock.current) void refresh(controller.signal); };
    const timer = window.setInterval(reload, 15_000);
    window.addEventListener("focus", reload);
    return () => { controller.abort(); window.clearInterval(timer); window.removeEventListener("focus", reload); };
  }, [refresh]);

  const group = data?.groups.find(item => item.session_key === sessionKey);
  const accepting = Boolean(data?.event.registration_open && data.groups.length && !loadError);
  async function submit() {
    if (lock.current || !accepting || !group || name.trim().length < 2) return;
    lock.current = true; setBusy(true); setError("");
    try {
      const response = await fetch(`${base}/checkins`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name.trim(), session_key: sessionKey }) });
      const value = await response.json();
      if (!response.ok) throw new Error(typeof value.detail === "string" ? value.detail : "Check-in was not confirmed. Please retry.");
      if (value.ok !== true) throw new Error("Check-in was not confirmed. Please retry.");
      setSent({ name: value.name, group: group.title });
      setName(""); setSessionKey("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Check-in was not confirmed. Please retry; duplicate submissions are handled safely.");
      void refresh();
    } finally { lock.current = false; setBusy(false); }
  }

  return <section className={styles.page}>
    <p className={styles.eyebrow}>Round Robin Hub</p>
    <h1>{data?.event.title || "Event check-in"}</h1>
    <div className={styles.card}>
      {sent ? <div role="status">
        <div className={styles.tick} aria-hidden="true">✓</div>
        <h2>You’re checked in, {sent.name}.</h2>
        <p><strong>{sent.group}</strong></p>
        <p>Your name has been sent to the event organizer. They’ll confirm your place in the playing roster.</p>
        <button type="button" onClick={() => { setSent(null); setError(""); void refresh(); requestAnimationFrame(() => input.current?.focus()); }}>Check in another player</button>
      </div> : <>
        <h2>Check in for tonight’s play</h2>
        <p>Enter your full name and choose the round robin you’re joining. No account needed.</p>
        {loading ? <p role="status">Loading event…</p> : null}
        {loadError ? <div role="alert"><p>{loadError}</p><button type="button" className={styles.secondary} onClick={() => void refresh()}>Retry loading event</button></div> : null}
        {data && !data.event.registration_open ? <p className={styles.notice} role="status">{data.event.finished ? "This event has finished." : "Check-in is currently closed. Please ask the event organizer."}</p> : null}
        {data?.event.registration_open && !data.groups.length ? <p className={styles.notice}>No round robins are accepting check-ins right now. Please ask the event organizer.</p> : null}
        {data?.event.registration_open && data.groups.length ? <form onSubmit={event => { event.preventDefault(); void submit(); }}>
          <label htmlFor="checkin-name">Full name</label>
          <input ref={input} id="checkin-name" autoComplete="off" autoCapitalize="words" required minLength={2} maxLength={120}
            value={name} disabled={busy} placeholder="First and last name" onChange={event => setName(event.target.value)} />
          <label htmlFor="checkin-round-robin">Round robin</label>
          <select id="checkin-round-robin" required value={sessionKey} disabled={busy} onChange={event => setSessionKey(event.target.value)}>
            <option value="">Choose your round robin</option>
            {sessionKey && !group ? <option value={sessionKey} disabled>This round robin is no longer available</option> : null}
            {data.groups.map(item => <option key={item.session_key} value={item.session_key}>{item.title}</option>)}
          </select>
          {sessionKey && !group ? <p role="alert">Choose an available round robin.</p> : null}
          {error ? <p className={styles.error} role="alert">{error}</p> : null}
          <button type="submit" disabled={busy || !accepting || !group || name.trim().length < 2}>{busy ? "Sending check-in…" : "Check in"}</button>
          <p className={styles.note}>Your check-in is visible to event staff. Checking in again for the same round robin won’t add your name twice.</p>
        </form> : null}
      </>}
    </div>
  </section>;
}
