"use client";
import { useEffect, useRef, useState } from "react";
import { getAdminPlayerEditorApiBaseUrl } from "@/lib/adminPlayerEditorApi";
import { managedRequest, type HostContext } from "@/lib/managedInterclub";
import ManagedScoreboard from "@/components/interclub/ManagedScoreboard";
import styles from "@/app/admin/interclub/setup.module.css";

export default function HostMeet() {
  const api = getAdminPlayerEditorApiBaseUrl();
  const [data, setData] = useState<HostContext | null>(null), [error, setError] = useState(""), [reload, setReload] = useState(0);
  const token = useRef("");
  useEffect(() => {
    const controller = new AbortController(); setData(null); setError("");
    // Fragments never reach server access logs or link-preview fetchers.
    token.current = new URLSearchParams(window.location.hash.slice(1)).get("token") || "";
    if (!token.current) { setError("Open the complete private meet link sent by your league organizer."); return; }
    managedRequest<HostContext>(`${api}/public/interclub-host/read`, null, "POST", { token: token.current }, controller.signal)
      .then(saved => { if (!controller.signal.aborted) setData(saved); })
      .catch(reason => { if (!controller.signal.aborted) setError(reason.message); });
    const changed = () => setReload(n => n + 1); window.addEventListener("hashchange", changed);
    return () => { controller.abort(); window.removeEventListener("hashchange", changed); };
  }, [api, reload]);
  if (error) return <section className={styles.page}><h1>Interclub meet</h1><p role="alert">{error}</p><button onClick={() => setReload(n => n + 1)}>Try again</button></section>;
  if (!data) return <p>Opening your meet…</p>;
  const { meet, season } = data, clubName = (id: string) => data.clubs.find(c => c.id === id)?.name || id;
  return <main className={styles.page}>
    <p className={styles.eyebrow}>Meet host</p><h1>{season.details.name}</h1>
    <p><strong>{clubName(meet.details.host_club_id)}</strong> · {new Date(meet.details.starts_at).toLocaleString(undefined, { timeZone: season.details.timezone, dateStyle: "full", timeStyle: "short" })}</p>
    <p>Print and record scores by hand, enter results as games finish, or use both. Scores save automatically while you are connected.</p>
    {season.details.rules && <details className={styles.panel}><summary>League information & rules</summary><p style={{ whiteSpace: "pre-wrap" }}>{season.details.rules}</p></details>}
    <ManagedScoreboard key={meet.id} meet={meet} seasonName={season.details.name} timezone={season.details.timezone} clubName={clubName}
      onWrite={async (action, revision, document) => {
        const saved = await managedRequest<HostContext>(`${api}/public/interclub-host/write`, null, "POST", {
          token: token.current, action, expected_revision: revision, ...(document ? { document } : {}),
        }); setData(saved); return saved.meet;
      }} />
  </main>;
}
