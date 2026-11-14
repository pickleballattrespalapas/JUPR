"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import CourtSchedule from "@/app/admin/interclub/competition/CourtSchedule";
import PrintPacket from "@/app/admin/interclub/competition/PrintPacket";
import ScoreEditor from "@/app/admin/interclub/competition/ScoreEditor";
import { automaticDraftScores, competitionPlayers, gameCount, type CompetitionDocument } from "@/lib/interclubCompetition";
import { printMeet, scoreDetail, type ManagedMeet } from "@/lib/managedInterclub";
import styles from "@/app/admin/interclub/competition/competition.module.css";

type Props = { meet: ManagedMeet; seasonName: string; timezone: string; clubName: (id: string) => string;
  onDirtyChange?: (dirty: boolean) => void;
  onWrite: (action: "scores" | "submit", revision: number, document?: CompetitionDocument) => Promise<ManagedMeet> };

export default function ManagedScoreboard({ meet: initial, seasonName, timezone, clubName, onWrite, onDirtyChange }: Props) {
  const [meet, setMeet] = useState(initial);
  const [draft, setDraft] = useState<CompetitionDocument | null>(initial.document);
  const [saving, setSaving] = useState(false), [paused, setPaused] = useState(false), [error, setError] = useState("");
  const [locked, setLocked] = useState(false);
  const [status, setStatus] = useState("All scores saved.");
  const current = useRef({ meet, draft });
  const pending = useRef(false), alive = useRef(true), writer = useRef(onWrite);
  useLayoutEffect(() => { current.current = { meet, draft }; writer.current = onWrite; }, [meet, draft, onWrite]);
  const dirty = JSON.stringify(draft) !== JSON.stringify(meet.document);
  useEffect(() => { onDirtyChange?.(dirty || saving); }, [dirty, saving, onDirtyChange]);
  useEffect(() => () => onDirtyChange?.(false), [onDirtyChange]);
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  useEffect(() => {
    if (initial.revision === current.current.meet.revision) return;
    if (JSON.stringify(current.current.draft) !== JSON.stringify(current.current.meet.document) && !pending.current) {
      setPaused(true); setLocked(true); setError("The saved meet changed in another tab. Keep a copy of your unsaved scores, then reload to review the saved version."); return;
    }
    if (!pending.current) { setMeet(initial); setDraft(initial.document); setPaused(false); setLocked(false); setError(""); }
  }, [initial]);
  useEffect(() => {
    if (!dirty) return;
    const warn = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ""; };
    window.addEventListener("beforeunload", warn); return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  const write = useCallback(async (action: "scores" | "submit") => {
    if (pending.current || locked || !current.current.draft || current.current.meet.state !== "draft") return;
    const snapshot = current.current.draft, revision = current.current.meet.revision;
    pending.current = true; setSaving(true); setError(""); setStatus(action === "scores" ? "Saving scores…" : "Submitting meet…");
    try {
      const saved = await writer.current(action, revision, action === "scores" ? snapshot : undefined);
      if (!alive.current) return;
      setMeet(saved);
      // A save response must not overwrite typing that happened during the request.
      setDraft(latest => JSON.stringify(latest) === JSON.stringify(snapshot) ? saved.document : latest);
      setPaused(false); setStatus(action === "scores" ? "All scores saved." : "Meet submitted. The organizer will review the results.");
    } catch (reason) {
      if (alive.current) { setPaused(true); setLocked([403, 404, 409].includes((reason as { status?: number }).status || 0)); setError(reason instanceof Error ? reason.message : "Scores were not confirmed saved. Keep this page open and try again."); setStatus("Save not confirmed."); }
    } finally { pending.current = false; if (alive.current) setSaving(false); }
  }, [locked]);
  useEffect(() => {
    if (!dirty || saving || paused || meet.state !== "draft") return;
    const timer = window.setTimeout(() => void write("scores"), 900);
    return () => window.clearTimeout(timer);
  }, [draft, dirty, saving, paused, meet.state, write]);

  if (!draft) return <p>The organizer is preparing this meet’s schedule.</p>;
  const detail = scoreDetail(meet), players = competitionPlayers(detail), count = gameCount(draft);
  const canSubmit = !saving && !dirty && !paused && count.total > 0 && count.entered === count.total && meet.state === "draft";
  return <section className={styles.page}>
    <div className={styles.toolbar}>
      <button type="button" onClick={() => window.print()}>Print schedule & score sheets</button>
      {meet.state === "draft" && <><button type="button" disabled={saving || !dirty || locked} onClick={() => void write("scores")}>Save scores now</button>
        <button type="button" disabled={!canSubmit} onClick={() => void write("submit")}>Submit completed meet</button></>}
    </div>
    <p role="status" aria-live="polite">{saving ? status : dirty ? "Unsaved scores" : status} · {count.entered} / {count.total} games recorded</p>
    {error && <div className={styles.error} role="alert"><p>{error}</p><button type="button" onClick={() => {
      const blob = new Blob([JSON.stringify(draft, null, 2)], { type: "application/json" }); const url = URL.createObjectURL(blob);
      const link = document.createElement("a"); link.href = url; link.download = "interclub-unsaved-scores.json"; link.click(); URL.revokeObjectURL(url);
    }}>Download unsaved scores</button><p>Reloading this page discards any unsaved changes.</p></div>}
    {meet.state === "submitted" && <p className={styles.notice}>Submitted for organizer review. Scores are locked until the organizer reopens the meet.</p>}
    {meet.state === "approved" && <p className={styles.notice}>These results are official. Contact the organizer for corrections.</p>}
    <CourtSchedule document={draft} clubName={clubName} />
    <ScoreEditor document={draft} detail={detail} players={players} clubName={clubName} disabled={meet.state !== "draft" || locked}
      allowStartingLineupChanges={false} onChange={next => { setDraft(automaticDraftScores(next)); setPaused(false); }} />
    <PrintPacket document={draft} meet={printMeet(meet)} seasonName={seasonName} timezone={timezone} revision={meet.revision} players={players} clubName={clubName} showRosterDeadline={false} />
  </section>;
}
