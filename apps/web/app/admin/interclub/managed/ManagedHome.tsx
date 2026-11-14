"use client";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { getAdminPlayerEditorApiBaseUrl } from "@/lib/adminPlayerEditorApi";
import { managedRequest, type ManagedContext, type ManagedDetails, type ManagedSeason } from "@/lib/managedInterclub";
import type { RegistrationSeason } from "@/lib/interclubRegistration";
import ManagedSeasonWorkspace from "./ManagedSeasonWorkspace";
import styles from "../setup.module.css";

export default function ManagedHome({ clubId, accessToken, initialId }: { clubId: string; accessToken: string; initialId: string }) {
  const api = getAdminPlayerEditorApiBaseUrl();
  const root = `${api}/admin/clubs/${encodeURIComponent(clubId)}/interclub/managed`;
  const [seasons, setSeasons] = useState<ManagedSeason[]>([]), [legacy, setLegacy] = useState<RegistrationSeason[]>([]);
  const [active, setActive] = useState<ManagedContext | null>(null), [busy, setBusy] = useState(false), [ready, setReady] = useState(false);
  const [error, setError] = useState(""), [name, setName] = useState("Southern BCS Interclub"), [reload, setReload] = useState(0);
  const token = useRef(accessToken);
  useEffect(() => { token.current = accessToken; }, [accessToken]);
  const pending = useRef(false);
  const openedFromLink = useRef(false);
  useEffect(() => {
    const controller = new AbortController(); setReady(false); setError("");
    if (!api) { setError("Interclub setup is temporarily unavailable."); return; }
    Promise.all([
      managedRequest<{ seasons: ManagedSeason[] }>(root, token.current, "GET", undefined, controller.signal),
      managedRequest<{ seasons: RegistrationSeason[] }>(root.replace(/\/managed$/, "/registrations"), token.current, "GET", undefined, controller.signal).catch(() => ({ seasons: [] })),
    ]).then(async ([managed, old]) => {
      if (controller.signal.aborted) return;
      setSeasons(managed.seasons); setLegacy(old.seasons.filter(s => s.organizer_club_id === clubId)); setReady(true);
      if (initialId && !openedFromLink.current && managed.seasons.some(s => s.id === initialId)) {
        openedFromLink.current = true;
        const result = await managedRequest<ManagedContext>(`${root}/${initialId}`, token.current, "GET", undefined, controller.signal);
        if (!controller.signal.aborted) setActive(result);
      }
    }).catch(reason => { if (!controller.signal.aborted) setError(reason.message); });
    return () => controller.abort();
  }, [api, root, clubId, reload, initialId]);
  async function open(id: string, convert = false) {
    if (pending.current) return;
    pending.current = true; setBusy(true); setError("");
    try { setActive(await managedRequest<ManagedContext>(`${root}/${id}${convert ? "/import" : ""}`, token.current, convert ? "POST" : "GET")); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Could not open this season."); }
    finally { pending.current = false; setBusy(false); }
  }
  async function create(event: React.FormEvent) {
    event.preventDefault(); if (pending.current) return;
    pending.current = true; setBusy(true); setError("");
    const details: ManagedDetails = { name: name.trim(), start_date: null, end_date: null, timezone: "America/Mazatlan", divisions: ["3.5", "4.0"], rules: "",
      clubs: [{ id: crypto.randomUUID(), name: clubId === "tres_palapas" ? "Tres Palapas" : "Home club", captain: "", contact: "", rosters: [] }] };
    try { setActive(await managedRequest<ManagedContext>(`${root}/${crypto.randomUUID()}`, token.current, "PUT", { expected_revision: 0, details })); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Could not create the season."); }
    finally { pending.current = false; setBusy(false); }
  }
  if (active) return <ManagedSeasonWorkspace key={active.season.id} root={`${root}/${active.season.id}`} accessToken={accessToken} initial={active}
    onClose={() => { setActive(null); setReload(n => n + 1); }} />;
  return <section className={styles.page}>
    <header className={styles.header}><div><p className={styles.eyebrow}>{clubId === "tres_palapas" ? "Tres Palapas" : "Your club"} · League organizer</p><h1>Interclub leagues</h1>
      <p>Manage the teams and schedule here. Send each host a private meet link to print score sheets or enter scores live.</p></div></header>
    {error && <div role="alert" className={styles.error}>{error} <button onClick={() => setReload(n => n + 1)}>Reload seasons</button></div>}
    {!ready && !error && <p>Loading your leagues…</p>}
    <form onSubmit={create} className={styles.panel}><h2>Create a season</h2><p>Participating clubs need only a name and roster. No club accounts or invitations.</p>
      <div className={styles.toolbar}><label className={styles.field}>Season name<input required maxLength={120} value={name} onChange={e => setName(e.target.value)} /></label>
        <button className={styles.primary} disabled={busy || !ready || !name.trim()}>{busy ? "Opening…" : "Create season"}</button></div></form>
    <div className={styles.cards}>{seasons.map(season => <article key={season.id} className={styles.card}><h2>{season.details.name}</h2>
      <p>{season.details.clubs.length} clubs · {season.published ? "Public page published" : "Not published"}</p><button disabled={busy} onClick={() => void open(season.id)}>Open league</button></article>)}</div>
    {legacy.filter(s => !seasons.some(row => row.id === s.id)).length > 0 && <section className={styles.panel}><h2>Continue an existing season</h2>
      <p>Bring its clubs, lineups, meet dates and saved results into this simpler workspace. The original records remain available.</p>
      {legacy.filter(s => !seasons.some(row => row.id === s.id)).map(season => <div key={season.id} className={styles.toolbar}><strong>{season.details.name}</strong>
        <button disabled={busy} onClick={() => void open(season.id, true)}>Use simplified setup</button></div>)}</section>}
    <p><Link href="/admin/interclub/legacy">Previous club-account workspaces</Link></p>
  </section>;
}
