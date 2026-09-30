"use client";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { useAdminWorkspace } from "@/lib/useAdminWorkspace";
import { useAdminSession } from "@/lib/useAdminSession";
import { getAdminApiBaseUrl } from "@/lib/adminAuthClient";
import { readBrowserWorkspace } from "@/lib/adminWorkspace";
import { apiError } from "@/lib/interclubRegistration";
import type { AdminEventHistory, EventKind, EventSeason } from "@/lib/eventSeasons";
import EventHistoryView from "@/components/EventHistoryView";
import styles from "@/components/EventHistory.module.css";
import buttons from "@/components/ClubWebsite.module.css";

export default function EventSeasonManager({ kind, sourceId }: { kind: EventKind; sourceId: string }) {
  const { clubId } = useAdminWorkspace();
  const { session, accessToken, loading } = useAdminSession();
  const allowed = session?.capabilities?.assignments.some(row => row.club_id === clubId && ["administrator", "super_admin", "club_owner"].includes(row.role));
  if (loading) return <p>Checking access…</p>;
  if (!accessToken || !allowed) return <p>Club administrator access is required.</p>;
  return <Manager key={`${clubId}:${kind}:${sourceId}:${session?.user?.id}`} clubId={clubId} kind={kind} sourceId={sourceId} accessToken={accessToken} />;
}

function Manager({ clubId, kind, sourceId, accessToken }: { clubId: string; kind: EventKind; sourceId: string; accessToken: string }) {
  const [data, setData] = useState<AdminEventHistory | null>(null), [error, setError] = useState("");
  const [refresh, setRefresh] = useState(0), [busy, setBusy] = useState(false), [blocked, setBlocked] = useState(false);
  const [mode, setMode] = useState<"start" | "link" | "">(""), [review, setReview] = useState(false);
  const [seriesName, setSeriesName] = useState(""), [currentLabel, setCurrentLabel] = useState("");
  const [label, setLabel] = useState(""), [name, setName] = useState(""), [start, setStart] = useState(""), [end, setEnd] = useState("");
  const [past, setPast] = useState(""), [before, setBefore] = useState(sourceId), [pastData, setPastData] = useState<EventSeason | null>(null);
  const [created, setCreated] = useState<{ name: string; admin_href: string } | null>(null), [message, setMessage] = useState("");
  const token = useRef(accessToken), pending = useRef<AbortController | null>(null), lock = useRef(false), requestId = useRef("");
  token.current = accessToken;
  const root = `${getAdminApiBaseUrl()}/admin/clubs/${encodeURIComponent(clubId)}/event-seasons`;
  const query = new URLSearchParams({ kind, event: sourceId }).toString();
  useEffect(() => {
    const controller = new AbortController();
    setData(null); setError(""); setBlocked(false); setMode(""); setReview(false);
    void fetch(`${root}?${query}`, { headers: { Authorization: `Bearer ${token.current}` }, signal: controller.signal, cache: "no-store" })
      .then(async response => { const body = await response.json(); if (!response.ok) throw new Error(apiError(body, "Could not load event history.")); if (!controller.signal.aborted) { setData(body); setSeriesName(body.series_name); setCurrentLabel(body.current_label); } })
      .catch(reason => { if (!controller.signal.aborted) setError(reason.message); });
    return () => controller.abort();
  }, [root, query, refresh]);
  useEffect(() => {
    setPastData(null);
    if (!past) return;
    const controller = new AbortController();
    void fetch(`${root}/source?${new URLSearchParams({ kind, event: past })}`, { headers: { Authorization: `Bearer ${token.current}` }, signal: controller.signal, cache: "no-store" })
      .then(async response => { const body = await response.json(); if (!response.ok) throw new Error(apiError(body, "Could not load the past season.")); if (!controller.signal.aborted) setPastData(body); })
      .catch(reason => { if (!controller.signal.aborted) setError(reason.message); });
    return () => controller.abort();
  }, [root, kind, past]);
  useEffect(() => () => pending.current?.abort(), []);
  function openForm(action: "start" | "link") {
    if (!data) return;
    setMode(action); setReview(false); setError(""); setPast(""); setBefore(sourceId); requestId.current = "";
    const year = Number((data.current.end_date || data.current.start_date || "").slice(0, 4));
    const nextLabel = action === "start" ? year ? String(year + 1) : `Season ${data.seasons.length + 1}` : "";
    setLabel(nextLabel); setName(`${data.series_name} — ${nextLabel}`.slice(0, 120)); setStart(""); setEnd("");
  }
  async function save() {
    if (!data || !review || !mode || blocked || lock.current) return;
    if (readBrowserWorkspace()?.clubId !== clubId) { setError("The selected club changed. Reopen this event."); setBlocked(true); return; }
    lock.current = true; setBusy(true); setError("");
    const controller = new AbortController(); pending.current = controller;
    try {
      const body = { request_id: requestId.current, fingerprint: data.current.fingerprint, series_name: seriesName, current_label: currentLabel, label,
        ...(mode === "start" ? { name, start_date: start, end_date: end } : { past_source_id: past, past_fingerprint: pastData?.fingerprint, before_source_id: before }) };
      const response = await fetch(`${root}/${mode}?${query}`, { method: "POST", signal: controller.signal,
        headers: { Authorization: `Bearer ${token.current}`, "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const result = await response.json();
      if (!response.ok) throw new Error(apiError(result, "Could not confirm the change. Reload History before retrying."));
      if (!controller.signal.aborted) { if (mode === "start") setCreated(result); else setMessage("Past season linked. Its saved results and awards are now in this event’s history."); setRefresh(value => value + 1); }
    } catch (reason) { if (!controller.signal.aborted) { setError(reason instanceof Error ? reason.message : "Could not confirm the change."); setBlocked(true); } }
    finally { if (!controller.signal.aborted) { lock.current = false; setBusy(false); } }
  }
  const edition = kind === "tournament" ? "edition" : "season";
  return <section className={styles.history}>
    {data && <p><Link href={data.current.admin_href}>← {data.current.name}</Link></p>}
    <h1>History & seasons</h1>
    {error && <p className={styles.error} role="alert">{error}</p>}
    {message && <p className={styles.notice} role="status">{message}</p>}
    {created && <div className={styles.notice} role="status"><strong>{created.name} is ready to set up.</strong><p><Link href={created.admin_href}>Continue new {edition} setup →</Link></p></div>}
    {(!data || blocked) && <button className={buttons.button} disabled={busy} onClick={() => setRefresh(value => value + 1)}>Reload History</button>}
    {!data && !error && <p>Loading history…</p>}
    {data && <>
      <div className={styles.actions}>
        {data.can_start && <button className={buttons.primary} disabled={busy || blocked} onClick={() => openForm("start")}>Start new {edition}</button>}
        {!!data.past_candidates.length && <button className={buttons.button} disabled={busy || blocked} onClick={() => openForm("link")}>Link a past {edition}</button>}
      </div>
      {data.reason && <p className={styles.muted}>{data.reason}</p>}
      {mode && <form className={`${styles.season} ${styles.form}`} onSubmit={event => { event.preventDefault(); requestId.current = crypto.randomUUID(); setReview(true); }}>
        <h2>{mode === "start" ? `New ${edition}` : `Link a past ${edition}`}</h2>
        {!data.series_id && <>
          <label>Recurring event name<input required maxLength={180} value={seriesName} disabled={review || busy} onChange={event => setSeriesName(event.target.value)} placeholder="Southern BCS" /></label>
          <label>This {edition}’s label<input required maxLength={120} value={currentLabel} disabled={review || busy} onChange={event => setCurrentLabel(event.target.value)} placeholder="2026 or Fall 2026" /></label>
        </>}
        {mode === "link" && <>
          <label>Past event<select required value={past} disabled={review || busy} onChange={event => setPast(event.target.value)}><option value="">Choose a completed event</option>{data.past_candidates.map(item => <option key={item.source_id} value={item.source_id}>{item.name}</option>)}</select></label>
          {pastData && <p className={styles.muted}>{pastData.name} · {pastData.start_date} {pastData.end_date ? `– ${pastData.end_date}` : ""} · {pastData.status}</p>}
          {pastData && !pastData.complete && <p className={styles.error}>Finish and publish that event before linking it as a past season.</p>}
          <label>Place before<select required value={before} disabled={review || busy} onChange={event => setBefore(event.target.value)}>{data.seasons.map(item => <option key={item.source_id} value={item.source_id}>{item.label} — {item.name}</option>)}</select></label>
        </>}
        <label>{mode === "start" ? `New ${edition} label` : `Past ${edition} label`}<input required maxLength={120} value={label} disabled={review || busy} onChange={event => setLabel(event.target.value)} placeholder="2027 or Spring 2027" /></label>
        {mode === "start" && <>
          <label>New event name<input required maxLength={120} value={name} disabled={review || busy} onChange={event => setName(event.target.value)} /></label>
          <div className={styles.dates}><label>Start date<input required type="date" value={start} disabled={review || busy} onChange={event => setStart(event.target.value)} /></label><label>End date<input required type="date" min={start || undefined} value={end} disabled={review || busy} onChange={event => setEnd(event.target.value)} /></label></div>
          <p className={styles.notice}>{kind === "interclub" ? "Carries forward the participating clubs, divisions and eligibility rules. Meet dates and player signups start fresh." : kind === "league" ? "Carries forward the format, rules, court settings and awards configuration. Rosters and match results start fresh." : "Carries forward the divisions, formats, venue and policies. Tournament days move to the new dates; entries, payments, sponsors, draws and scores start fresh."} The new {edition} opens as a draft for you to review.</p>
        </>}
        {review && <p className={styles.notice}>Review the details above.{mode === "link" ? " Confirm that these events are editions of the same recurring event, in the chosen order." : " Your previous season’s results and trophies stay in History."}</p>}
        <div className={styles.actions}>
          {review ? <><button className={buttons.primary} type="button" disabled={busy || blocked} onClick={() => void save()}>{busy ? "Saving…" : mode === "start" ? `Create ${edition} draft` : `Link past ${edition}`}</button><button className={buttons.button} type="button" disabled={busy} onClick={() => setReview(false)}>Edit details</button></> : <button className={buttons.primary} type="submit" disabled={busy || blocked || (mode === "link" && !pastData?.complete)}>Review {mode === "start" ? `new ${edition}` : "history link"}</button>}
          <button className={buttons.button} type="button" disabled={busy} onClick={() => { setMode(""); setReview(false); }}>Cancel</button>
        </div>
      </form>}
      <EventHistoryView history={data} admin />
    </>}
  </section>;
}
