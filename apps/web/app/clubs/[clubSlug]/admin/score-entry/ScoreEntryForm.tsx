"use client";

import { useState } from "react";
import type { PublicPlayer } from "@/lib/api";

type ScoreEntryFormProps = {
  apiBase: string | null;
  clubId: string;
  players: PublicPlayer[];
};

function todayIsoDate(): string {
  return new Date().toISOString().slice(0, 10);
}

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

export default function ScoreEntryForm({ apiBase, clubId, players }: ScoreEntryFormProps) {
  const [token, setToken] = useState("");
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

  async function submitMatch() {
    if (!apiBase) {
      setMessage("API base URL is not configured.");
      return;
    }
    if (!token.trim()) {
      setMessage("Paste a Supabase admin access token first.");
      return;
    }
    if (!t1p1 || !t1p2 || !t2p1 || !t2p2) {
      setMessage("Select four players.");
      return;
    }
    setSaving(true);
    setMessage(null);
    try {
      const response = await fetch(apiUrl(apiBase, `/admin/clubs/${clubId}/matches/batch`), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token.trim()}`
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
      setMessage(`Match saved. ${JSON.stringify(payload?.result ?? {})}`);
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
        This MVP uses the existing FastAPI match submission service. It requires a Supabase JWT with score-entry permission and the backend feature flag enabled.
      </p>
      <div style={{ display: "grid", gap: "0.75rem" }}>
        <label style={labelStyle}>Supabase access token<input value={token} onChange={(event) => setToken(event.target.value)} type="password" style={selectStyle} /></label>
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
          <button type="button" onClick={submitMatch} disabled={saving} style={{ border: 0, borderRadius: "999px", padding: "0.65rem 1rem", background: "#2563eb", color: "white", fontWeight: 800 }}>
            {saving ? "Saving…" : "Save rated match"}
          </button>
        </div>
        {message ? <p style={{ color: message.startsWith("Match saved") ? "#166534" : "#b91c1c" }}>{message}</p> : null}
      </div>
    </section>
  );
}
