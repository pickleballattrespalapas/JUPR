"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import type { PublicLiveMatch, PublicLiveSessionDetail } from "@/lib/api";

type LiveSessionRunnerProps = {
  apiBase: string | null;
  clubSlug: string;
  initialSession: PublicLiveSessionDetail;
  editToken: string;
};

const thStyle = { textAlign: "left" as const, borderBottom: "1px solid #cbd5e1", padding: "0.5rem", whiteSpace: "nowrap" as const };
const tdStyle = { borderBottom: "1px solid #e2e8f0", padding: "0.5rem", whiteSpace: "nowrap" as const };
const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function formatTimestamp(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("en-US", { dateStyle: "medium", timeStyle: "short" });
}

function eventTypeLabel(value?: string | null): string {
  const normalized = String(value || "").replace(/_/g, " ").trim();
  return normalized ? normalized.replace(/\b\w/g, (char) => char.toUpperCase()) : "JUPR Live";
}

function teamLabel(names: string[]): string {
  return names.filter(Boolean).join(" / ") || "TBD";
}

function scoreLabel(match: PublicLiveMatch): string {
  const scoreA = match.score_a ?? null;
  const scoreB = match.score_b ?? null;
  if (scoreA == null && scoreB == null) return "—";
  return `${scoreA ?? 0}–${scoreB ?? 0}`;
}

function scoreInputKey(matchId: string, side: "a" | "b"): string {
  return `${matchId}:${side}`;
}

export default function LiveSessionRunner({ apiBase, clubSlug, initialSession, editToken }: LiveSessionRunnerProps) {
  const [session, setSession] = useState(initialSession);
  const [scoreValues, setScoreValues] = useState<Record<string, string>>(() => {
    const initial: Record<string, string> = {};
    for (const round of initialSession.rounds || []) {
      for (const match of round.matches || []) {
        initial[scoreInputKey(match.id, "a")] = match.score_a == null ? "" : String(match.score_a);
        initial[scoreInputKey(match.id, "b")] = match.score_b == null ? "" : String(match.score_b);
      }
      for (const court of round.courts || []) {
        for (const match of court.matches || []) {
          initial[scoreInputKey(match.id, "a")] = match.score_a == null ? "" : String(match.score_a);
          initial[scoreInputKey(match.id, "b")] = match.score_b == null ? "" : String(match.score_b);
        }
      }
    }
    return initial;
  });
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const canEdit = Boolean(editToken);

  const allMatches = useMemo(() => {
    const seen = new Set<string>();
    const matches: PublicLiveMatch[] = [];
    for (const round of session.rounds || []) {
      const roundMatches = round.courts?.length
        ? round.courts.flatMap((court) => court.matches || [])
        : round.matches || [];
      for (const match of roundMatches) {
        if (seen.has(match.id)) continue;
        seen.add(match.id);
        matches.push(match);
      }
    }
    return matches;
  }, [session]);

  async function saveScores() {
    if (!apiBase) {
      setMessage("The public API base URL is not configured for this deployment.");
      return;
    }
    if (!editToken) {
      setMessage("This link is view-only. Use the original edit link to enter scores.");
      return;
    }
    setSaving(true);
    setMessage(null);
    try {
      const scores = allMatches.map((match) => {
        const aRaw = scoreValues[scoreInputKey(match.id, "a")] ?? "";
        const bRaw = scoreValues[scoreInputKey(match.id, "b")] ?? "";
        return {
          match_id: match.id,
          score_a: aRaw === "" ? null : Number(aRaw),
          score_b: bRaw === "" ? null : Number(bRaw)
        };
      });
      const response = await fetch(apiUrl(apiBase, `/clubs/${clubSlug}/live-sessions/${session.session_key}/scores`), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ edit_token: editToken, scores })
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error(String(payload?.detail || `API error (${response.status})`));
      }
      setSession(payload.session);
      setMessage("Scores saved.");
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Unable to save scores.");
    } finally {
      setSaving(false);
    }
  }

  function renderMatch(match: PublicLiveMatch) {
    return (
      <article key={match.id} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.8rem", background: "#f8fafc" }}>
        <p style={{ margin: "0 0 0.35rem", color: "#64748b", fontSize: "0.85rem" }}>{match.label}</p>
        <div style={{ display: "grid", gridTemplateColumns: canEdit ? "1fr 4.2rem 4.2rem 1fr" : "1fr auto 1fr", alignItems: "center", gap: "0.75rem" }}>
          <strong>{teamLabel(match.team_a)}</strong>
          {canEdit ? (
            <>
              <input
                aria-label={`${match.label} team A score`}
                type="number"
                min={0}
                max={99}
                value={scoreValues[scoreInputKey(match.id, "a")] ?? ""}
                onChange={(event) => setScoreValues((current) => ({ ...current, [scoreInputKey(match.id, "a")]: event.target.value }))}
                style={{ width: "100%", padding: "0.45rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" }}
              />
              <input
                aria-label={`${match.label} team B score`}
                type="number"
                min={0}
                max={99}
                value={scoreValues[scoreInputKey(match.id, "b")] ?? ""}
                onChange={(event) => setScoreValues((current) => ({ ...current, [scoreInputKey(match.id, "b")]: event.target.value }))}
                style={{ width: "100%", padding: "0.45rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" }}
              />
            </>
          ) : (
            <span style={{ fontWeight: 800, fontSize: "1.1rem" }}>{scoreLabel(match)}</span>
          )}
          <strong style={{ textAlign: "right" }}>{teamLabel(match.team_b)}</strong>
        </div>
        {match.winner ? <p style={{ margin: "0.4rem 0 0", color: "#166534", fontSize: "0.9rem" }}>Winner: {match.winner}</p> : null}
      </article>
    );
  }

  return (
    <>
      <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", flexWrap: "wrap", alignItems: "flex-start", marginBottom: "1rem" }}>
        <div>
          <h1 style={{ margin: "0 0 0.35rem", fontSize: "2.2rem", lineHeight: 1.1 }}>{session.title}</h1>
          <p style={{ margin: 0, color: "#475569" }}>
            {eventTypeLabel(session.event_type)}
            {session.current_round ? ` · Current round ${session.current_round}` : ""}
          </p>
          <p style={{ margin: "0.35rem 0 0", color: "#64748b", fontSize: "0.9rem" }}>
            Last updated {formatTimestamp(session.updated_at ?? session.last_seen_at)}
          </p>
          {!canEdit ? <p style={{ color: "#64748b" }}>View-only scoreboard. The edit link is only shown to the person who created this event.</p> : null}
        </div>
        <span style={{ border: "1px solid #bfdbfe", borderRadius: "999px", padding: "0.25rem 0.75rem", color: "#1d4ed8", background: "#eff6ff", fontSize: "0.85rem", fontWeight: 800 }}>
          {session.status}
        </span>
      </div>

      {canEdit ? (
        <div style={{ ...cardStyle, marginBottom: "1rem", background: "#eff6ff", borderColor: "#bfdbfe" }}>
          <strong>Score entry enabled.</strong> Save as you go, then share the URL without the <code>?edit=</code> token for view-only access.
          <div style={{ marginTop: "0.75rem" }}>
            <button
              type="button"
              onClick={saveScores}
              disabled={saving}
              style={{ border: 0, borderRadius: "999px", padding: "0.65rem 1rem", background: "#2563eb", color: "white", fontWeight: 800, cursor: saving ? "default" : "pointer" }}
            >
              {saving ? "Saving…" : "Save scores"}
            </button>
            {message ? <span style={{ marginLeft: "0.75rem", color: message === "Scores saved." ? "#166534" : "#b91c1c" }}>{message}</span> : null}
          </div>
        </div>
      ) : null}

      <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1.5fr) minmax(280px, 1fr)", gap: "1rem", alignItems: "start" }}>
        <div style={{ display: "grid", gap: "1rem" }}>
          {session.rounds.length === 0 ? (
            <div style={cardStyle}>
              <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>No public match state yet</h2>
              <p style={{ color: "#475569" }}>This session exists, but the event schedule is not ready yet.</p>
            </div>
          ) : null}

          {session.rounds.map((round) => (
            <section key={round.number} style={cardStyle}>
              <h2 style={{ marginTop: 0, fontSize: "1.2rem" }}>Round {round.number}</h2>
              {round.courts && round.courts.length > 0 ? (
                <div style={{ display: "grid", gap: "0.75rem" }}>
                  {round.courts.map((court) => (
                    <div key={court.court_number}>
                      <h3 style={{ fontSize: "1rem", marginBottom: "0.5rem" }}>Court {court.court_number}</h3>
                      <div style={{ display: "grid", gap: "0.5rem" }}>
                        {court.matches.map((match) => renderMatch(match))}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ display: "grid", gap: "0.5rem" }}>
                  {round.matches.map((match) => renderMatch(match))}
                </div>
              )}
            </section>
          ))}
        </div>

        <aside style={{ display: "grid", gap: "1rem" }}>
          <section style={cardStyle}>
            <h2 style={{ marginTop: 0, fontSize: "1.1rem" }}>Standings</h2>
            {session.standings.length === 0 ? <p style={{ color: "#475569" }}>No standings yet.</p> : null}
            {session.standings.length > 0 ? (
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem" }}>
                  <thead>
                    <tr>
                      <th style={thStyle}>Rank</th>
                      <th style={thStyle}>Player</th>
                      <th style={thStyle}>W/L</th>
                      <th style={thStyle}>Diff</th>
                    </tr>
                  </thead>
                  <tbody>
                    {session.standings.map((row, index) => (
                      <tr key={`${row.participantId ?? row.name ?? index}`}>
                        <td style={tdStyle}>{String(row.rank ?? index + 1)}</td>
                        <td style={tdStyle}>{String(row.name ?? "—")}</td>
                        <td style={tdStyle}>{String(row.wins ?? 0)}/{String(row.losses ?? 0)}</td>
                        <td style={tdStyle}>{String(row.differential ?? "—")}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
          </section>
        </aside>
      </div>

      <p style={{ marginTop: "1rem" }}><Link href={`/clubs/${clubSlug}/live`}>Back to live sessions</Link></p>
    </>
  );
}
