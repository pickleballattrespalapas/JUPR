"use client";

import Link from "@/components/PublicClubLink";
import { useEffect, useState } from "react";
import { readGeneratorEditToken } from "@/lib/generatorBrowserStorage";

type Session = {
  session_key: string;
  title: string;
  status: string;
  generator_kind: string;
  current_round_number?: number | null;
};

export default function GeneratorResumeSessions({ apiBase, clubId, generatorKind }: {
  apiBase: string | null;
  clubId: string;
  generatorKind: "round_robin" | "ladder";
}) {
  const [result, setResult] = useState<{ context: string; sessions: Session[]; failed: boolean } | null>(null);
  const [reload, setReload] = useState(0);
  const context = `${apiBase}:${clubId}:${generatorKind}`;

  useEffect(() => {
    if (!apiBase) return;
    const controller = new AbortController();
    async function load() {
      try {
        const response = await fetch(`${apiBase!.replace(/\/$/, "")}/clubs/${encodeURIComponent(clubId)}/play-generators/sessions?generator_kind=${generatorKind}&limit=100`, {
          cache: "no-store", signal: controller.signal
        });
        if (!response.ok) throw new Error("Unable to load saved sessions");
        const payload = await response.json();
        if (!Array.isArray(payload?.sessions)) throw new Error("Invalid session list");
        const sessions = (payload.sessions as Session[]).filter(session =>
          session.generator_kind === generatorKind && readGeneratorEditToken(clubId, session.session_key)
        ).sort((a, b) => Number(b.status === "active") - Number(a.status === "active"));
        if (!controller.signal.aborted) setResult({ context, sessions, failed: false });
      } catch {
        if (!controller.signal.aborted) setResult({ context, sessions: [], failed: true });
      }
    }
    void load();
    return () => controller.abort();
  }, [apiBase, clubId, generatorKind, context, reload]);

  if (result?.context !== context || (!result.failed && !result.sessions.length)) return null;
  const slug = generatorKind === "round_robin" ? "round-robin-generator" : "ladder-generator";
  return <section aria-label="Saved sessions" style={{ border: "1px solid #cbd5e1", borderRadius: "14px", padding: "1rem", background: "#f0fdf4" }}>
    <h2 style={{ marginTop: 0 }}>Your saved sessions</h2>
    {result.failed ? <p role="status">We couldn’t load your saved sessions. Try again before starting a replacement.</p> : <>
      <p>Resume a session you organized in this browser. Saved rounds and scores are kept with the session.</p>
      <ul style={{ paddingLeft: "1.25rem" }}>
        {result.sessions.map(session => <li key={session.session_key} style={{ margin: "0.75rem 0" }}>
          <strong>{session.title}</strong>{" · "}Round {session.current_round_number || 1}{" · "}
          {session.status === "active" ? "In progress" : "Complete"}{" · "}
          <Link href={`/clubs/${encodeURIComponent(clubId)}/${slug}/sessions/${encodeURIComponent(session.session_key)}/rounds/${session.current_round_number || 1}`}>
            {session.status === "active" ? "Resume session" : "Open session"}
          </Link>
        </li>)}
      </ul>
    </>}
    <button type="button" onClick={() => setReload(value => value + 1)}>Refresh saved sessions</button>
  </section>;
}
