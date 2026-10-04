"use client";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { useAdminWorkspace } from "@/lib/useAdminWorkspace";
import { useAdminSession } from "@/lib/useAdminSession";
import { getAdminApiBaseUrl } from "@/lib/adminAuthClient";
import { readBrowserWorkspace } from "@/lib/adminWorkspace";
import { apiError } from "@/lib/interclubRegistration";
import type { SeasonAwardPreview } from "@/lib/interclubAwards";
import InterclubFinalResults from "@/components/InterclubFinalResults";
import styles from "@/components/InterclubHonors.module.css";

export default function SeasonAwards({ seasonId }: { seasonId: string }) {
  const { clubId } = useAdminWorkspace();
  const { session, accessToken, loading } = useAdminSession();
  const allowed = session?.capabilities?.assignments.some(row => row.club_id === clubId && ["administrator", "super_admin", "club_owner"].includes(row.role));
  if (loading) return <p>Checking access…</p>;
  if (!accessToken || !allowed) return <p>Season organizer access is required.</p>;
  if (!seasonId) return <Link href="/admin/interclub">Choose a season first.</Link>;
  return <AwardReview key={`${clubId}:${seasonId}:${session?.user?.id}`} clubId={clubId} seasonId={seasonId} accessToken={accessToken} />;
}

function AwardReview({ clubId, seasonId, accessToken }: { clubId: string; seasonId: string; accessToken: string }) {
  const [data, setData] = useState<SeasonAwardPreview | null>(null), [error, setError] = useState(""), [message, setMessage] = useState("");
  const [reviewed, setReviewed] = useState(false), [busy, setBusy] = useState(false), [blocked, setBlocked] = useState(false), [refresh, setRefresh] = useState(0);
  const token = useRef(accessToken), lock = useRef(false), pending = useRef<AbortController | null>(null);
  token.current = accessToken;
  const endpoint = `${getAdminApiBaseUrl()}/admin/clubs/${encodeURIComponent(clubId)}/interclub/${encodeURIComponent(seasonId)}/awards`;
  useEffect(() => {
    const controller = new AbortController();
    setData(null); setReviewed(false); setError(""); setBlocked(false);
    void fetch(endpoint, { headers: { Authorization: `Bearer ${token.current}` }, cache: "no-store", signal: controller.signal })
      .then(async response => { const body = await response.json(); if (!response.ok) throw new Error(apiError(body, "Could not load season awards.")); if (!controller.signal.aborted) setData(body); })
      .catch(reason => { if (!controller.signal.aborted) setError(reason.message); });
    return () => controller.abort();
  }, [endpoint, refresh]);
  useEffect(() => () => pending.current?.abort(), []);

  async function award() {
    if (!data?.ready || !reviewed || data.current || blocked || lock.current) return;
    if (readBrowserWorkspace()?.clubId !== clubId) { setError("The selected club changed. Reopen this workspace."); setBlocked(true); return; }
    const controller = new AbortController(); pending.current = controller; lock.current = true; setBusy(true); setError("");
    try {
      const response = await fetch(endpoint, { method: "POST", signal: controller.signal,
        headers: { Authorization: `Bearer ${token.current}`, "Content-Type": "application/json" },
        body: JSON.stringify({ revision: data.revision, publication_revision: data.publication_revision, preview_fingerprint: data.preview_fingerprint }) });
      const body = await response.json();
      if (!response.ok) throw new Error(apiError(body, "Could not confirm the awards. Reload the saved awards before trying again."));
      if (!controller.signal.aborted) { setMessage("Final results published and season trophies awarded."); setRefresh(value => value + 1); }
    } catch (reason) {
      if (!controller.signal.aborted) { setError(reason instanceof Error ? reason.message : "Could not confirm the awards."); setBlocked(true); }
    } finally { if (!controller.signal.aborted) { lock.current = false; setBusy(false); } }
  }

  return <section className={styles.section}>
    <div className={styles.actions}><Link href={`/admin/interclub/competition?season=${encodeURIComponent(seasonId)}`}>← Championship results</Link><Link href={`/admin/event-history?${new URLSearchParams({ kind: "interclub", event: seasonId })}`}>History & seasons →</Link></div>
    <p><Link href={`/admin/interclub/season?season=${encodeURIComponent(seasonId)}`}>← Season guide</Link> · Step 5 of 6: Publish results & awards</p>
    <div><h1>Season awards</h1><p className={styles.muted}>Review the final results and recipients, then publish the season honors to player profiles and club trophy cases.</p></div>
    {error && <p className={styles.error} role="alert">{error}</p>}
    {message && <p className={styles.notice} role="status">{message}</p>}
    {(!data || blocked) && <div className={styles.actions}><button disabled={busy} onClick={() => setRefresh(value => value + 1)}>Reload awards</button></div>}
    {!data && !error && <p>Loading season awards…</p>}
    {data && <>
      {data.problems.length > 0 && <div className={styles.notice}><h2>Finish the season first</h2><ul>{data.problems.map(problem => <li key={problem}>{problem}</li>)}</ul></div>}
      {data.current && <div className={styles.notice}><h2>Season finished</h2><p>Season trophies awarded. These honors match the current final results.</p><p>Next, create a linked season draft. This season’s results and trophies stay in History.</p><Link href={`/admin/interclub/season?season=${encodeURIComponent(seasonId)}`}>Continue to the next season →</Link></div>}
      <p className={styles.muted}>Season Participant recognizes anyone who played in a meet. Division Champion recognizes every player who played in that division for the winning club during the season. League Champion recognizes every season participant for the Club Cup winner.</p>
      <p className={styles.muted}>Win Matchup: win at least 2 of 3 games. Sweep Matchup: play and win all 3. Undefeated Day: win every game played in that meet. These badges appear with published results and count each player’s actual games, including substitutions. A sweep earns both matchup badges. Forfeits and unplayed games do not count as appearances.</p>
      <InterclubFinalResults league={data.preview} awards={[...data.awards, ...(data.achievements || [])]} />
      {data.ready && <section aria-label="Award season trophies" className={styles.card}>
        <h2>{data.current ? "Season honors published" : data.revision ? "Update season trophies" : "Award season trophies"}</h2>
        <p>{data.awards.filter(row => row.recipient_type === "player").length} player trophies · {data.awards.filter(row => row.recipient_type === "club").length} club trophies</p>
        {!data.current && <label className={styles.review}><input type="checkbox" checked={reviewed} disabled={busy || blocked} onChange={event => setReviewed(event.target.checked)} />I have reviewed the final results and trophy recipients.</label>}
        <div className={styles.actions}>{!data.current && <button className={styles.primary} disabled={!reviewed || busy || blocked} onClick={() => void award()}>{busy ? "Awarding trophies…" : data.revision ? "Publish final results and update trophies" : "Publish final results and award trophies"}</button>}
          {data.current && <Link href={`/interclub/${seasonId}/final-results`}>View final season results →</Link>}
        </div>
      </section>}
    </>}
  </section>;
}
