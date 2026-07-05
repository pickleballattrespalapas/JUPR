"use client";

import Link from "next/link";
import { useState } from "react";
import type { PublicPlayer } from "@/lib/api";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type ScoreEntryFormProps = {
  apiBase: string | null;
  clubId: string;
  clubSlug?: string;
  players: PublicPlayer[];
};

type ScoreFeedbackPlayer = {
  id: number;
  name: string;
  rating_before?: number | null;
  rating_after?: number | null;
  rating_delta?: number | null;
  matches_played_before?: number | null;
  matches_played_after?: number | null;
};

type ScoreFeedback = {
  ratings_updated?: boolean;
  affected_players?: ScoreFeedbackPlayer[];
  latest_match_id?: string | number | null;
};

function todayIsoDate(): string {
  return new Date().toISOString().slice(0, 10);
}

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function ratingLabel(value?: number | null): string {
  return value == null ? "—" : Math.round(Number(value)).toString();
}

function deltaLabel(value?: number | null): string {
  if (value == null) return "—";
  const rounded = Math.round(Number(value));
  return `${rounded >= 0 ? "+" : ""}${rounded}`;
}

export default function ScoreEntryForm({ apiBase, clubId, clubSlug = "tres-palapas", players }: ScoreEntryFormProps) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [league, setLeague] = useState("Open");
  const [date, setDate] = useState(todayIsoDate());
  const [t1p1, setT1p1] = useState("");
  const [t1p2, setT1p2] = useState("");
  const [t2p1, setT2p1] = useState("");
  const [t2p2, setT2p2] = useState("");
  const [scoreT1, setScoreT1] = useState("11");
  const [scoreT2, setScoreT2] = useState("0");
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<ScoreFeedback | null>(null);

  async function submitMatch() {
    if (!apiBase) {
      setMessage("API base URL is not configured.");
      return;
    }
    if (!accessToken) {
      setMessage("Sign in at /admin/login before saving rated matches.");
      return;
    }
    if (!t1p1 || !t1p2 || !t2p1 || !t2p2) {
      setMessage("Select four players.");
      return;
    }
    const uniquePlayers = new Set([t1p1, t1p2, t2p1, t2p2]);
    if (uniquePlayers.size !== 4) {
      setMessage("Select four different players.");
      return;
    }
    setSaving(true);
    setMessage(null);
    setFeedback(null);
    try {
      const response = await fetch(apiUrl(apiBase, `/admin/clubs/${clubId}/matches/batch`), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`
        },
        body: JSON.stringify({
          source: "next_score_entry_mvp",
          matches: [
            {
              date,
              league,
              match_type: "Web Score Entry",
              rating_scope: "overall",
              t1_p1: Number(t1p1),
              t1_p2: Number(t1p2),
              t2_p1: Number(t2p1),
              t2_p2: Number(t2p2),
              score_t1: Number(scoreT1),
              score_t2: Number(scoreT2)
            }
          ]
        })
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error(String(payload?.detail || `API error (${response.status})`));
      }
      setFeedback(payload?.feedback ?? null);
      setMessage(`Match saved. Inserted ${payload?.result?.inserted ?? 0} match${payload?.result?.inserted === 1 ? "" : "es"}.`);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Unable to save match.");
    } finally {
      setSaving(false);
    }
  }

  const playerOptions = players.map((player) => (
    <option key={String(player.id)} value={String(player.id)}>{player.name}</option>
  ));

  const selectStyle = { padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit", width: "100%" };
  const labelStyle = { display: "grid", gap: "0.25rem", fontWeight: 700 };

  return (
    <section style={{ border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" }}>
      <h2 style={{ marginTop: 0 }}>Enter one rated match</h2>
      <p style={{ color: "#475569" }}>
        This MVP uses the existing FastAPI match submission service. It requires a signed-in Supabase admin session with score-entry permission and the backend feature flag enabled.
      </p>
      <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: accessToken ? "#f0fdf4" : "#fffbeb", marginBottom: "1rem" }}>
        <strong>{accessToken ? `Admin session: ${adminSessionLabel(session)}` : "Admin session required"}</strong>
        <p style={{ margin: "0.35rem 0 0", color: accessToken ? "#166534" : "#92400e" }}>
          {accessToken ? "Ready to send authorized score-entry requests." : sessionLoading ? "Checking admin session…" : "Sign in before saving rated matches."}
        </p>
        {sessionMessage ? <p style={{ color: "#b91c1c", marginBottom: 0 }}>{sessionMessage}</p> : null}
        {!accessToken && !sessionLoading ? <p style={{ marginBottom: 0 }}><Link href="/admin/login">Open admin login</Link></p> : null}
      </div>
      <div style={{ display: "grid", gap: "0.75rem" }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem" }}>
          <label style={labelStyle}>Date<input value={date} onChange={(event) => setDate(event.target.value)} type="date" style={selectStyle} /></label>
          <label style={labelStyle}>League<input value={league} onChange={(event) => setLeague(event.target.value)} style={selectStyle} /></label>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem" }}>
          <label style={labelStyle}>Team 1 Player 1<select value={t1p1} onChange={(event) => setT1p1(event.target.value)} style={selectStyle}><option value="">Select</option>{playerOptions}</select></label>
          <label style={labelStyle}>Team 1 Player 2<select value={t1p2} onChange={(event) => setT1p2(event.target.value)} style={selectStyle}><option value="">Select</option>{playerOptions}</select></label>
          <label style={labelStyle}>Team 2 Player 1<select value={t2p1} onChange={(event) => setT2p1(event.target.value)} style={selectStyle}><option value="">Select</option>{playerOptions}</select></label>
          <label style={labelStyle}>Team 2 Player 2<select value={t2p2} onChange={(event) => setT2p2(event.target.value)} style={selectStyle}><option value="">Select</option>{playerOptions}</select></label>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: "0.75rem" }}>
          <label style={labelStyle}>Team 1 Score<input value={scoreT1} onChange={(event) => setScoreT1(event.target.value)} type="number" min={0} max={99} style={selectStyle} /></label>
          <label style={labelStyle}>Team 2 Score<input value={scoreT2} onChange={(event) => setScoreT2(event.target.value)} type="number" min={0} max={99} style={selectStyle} /></label>
        </div>
        <div>
          <button type="button" onClick={submitMatch} disabled={saving || !accessToken} style={{ border: 0, borderRadius: "999px", padding: "0.65rem 1rem", background: !accessToken ? "#94a3b8" : "#2563eb", color: "white", fontWeight: 800 }}>
            {saving ? "Saving…" : "Save rated match"}
          </button>
        </div>
        {message ? <p style={{ color: message.startsWith("Match saved") ? "#166534" : "#b91c1c" }}>{message}</p> : null}
      </div>

      {feedback ? (
        <div style={{ borderTop: "1px solid #e2e8f0", marginTop: "1rem", paddingTop: "1rem" }}>
          <h3 style={{ marginTop: 0 }}>Rating update</h3>
          <p style={{ color: "#475569" }}>{feedback.ratings_updated ? "Ratings changed for this match." : "The match saved; no rating movement was detected."}</p>
          {feedback.latest_match_id ? <p><Link href={`/clubs/${clubSlug}/matches/${feedback.latest_match_id}`}>Open match detail</Link></p> : null}
          {feedback.affected_players?.length ? (
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead><tr><th style={{ textAlign: "left", borderBottom: "1px solid #cbd5e1", padding: "0.5rem" }}>Player</th><th style={{ textAlign: "left", borderBottom: "1px solid #cbd5e1", padding: "0.5rem" }}>Before</th><th style={{ textAlign: "left", borderBottom: "1px solid #cbd5e1", padding: "0.5rem" }}>After</th><th style={{ textAlign: "left", borderBottom: "1px solid #cbd5e1", padding: "0.5rem" }}>Change</th></tr></thead>
                <tbody>
                  {feedback.affected_players.map((player) => (
                    <tr key={player.id}>
                      <td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.5rem" }}><Link href={`/clubs/${clubSlug}/players/${player.id}`}>{player.name}</Link></td>
                      <td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.5rem" }}>{ratingLabel(player.rating_before)}</td>
                      <td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.5rem" }}>{ratingLabel(player.rating_after)}</td>
                      <td style={{ borderBottom: "1px solid #e2e8f0", padding: "0.5rem" }}>{deltaLabel(player.rating_delta)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
