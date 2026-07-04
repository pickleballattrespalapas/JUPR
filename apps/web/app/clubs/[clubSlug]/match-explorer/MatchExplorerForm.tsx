"use client";

import { useMemo, useState } from "react";
import type { MatchExplorerPreview, MatchExplorerPreviewResponse, PublicPlayer } from "@/lib/api";

type MatchExplorerFormProps = {
  apiBase: string | null;
  clubSlug: string;
  players: PublicPlayer[];
  contexts: string[];
};

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function ratingLabel(value?: number | null): string {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return (Number(value) / 400).toFixed(3);
}

function percentLabel(value?: number | null): string {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${Math.round(Number(value) * 100)}%`;
}

function deltaLabel(value?: number | null): string {
  if (value == null || Number.isNaN(Number(value))) return "—";
  const delta = Number(value) / 400;
  return `${delta >= 0 ? "+" : ""}${delta.toFixed(4)}`;
}

function playerName(players: PublicPlayer[], id: string): string {
  return players.find((player) => String(player.id) === String(id))?.name ?? "";
}

function TeamSummary({ title, team }: { title: string; team: MatchExplorerPreview["teams"]["you"] }) {
  return (
    <article style={{ border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" }}>
      <h3 style={{ marginTop: 0 }}>{title}</h3>
      <p style={{ margin: "0 0 0.5rem", color: "#475569" }}>
        Average rating: <strong>{ratingLabel(team.average_rating)}</strong>
      </p>
      <ul style={{ marginBottom: 0 }}>
        {team.players.map((player) => (
          <li key={String(player.id)}>
            {player.name}: {ratingLabel(player.context_rating)}
            {Math.round(player.context_rating) !== Math.round(player.overall_rating) ? ` context / ${ratingLabel(player.overall_rating)} overall` : ""}
          </li>
        ))}
      </ul>
    </article>
  );
}

export default function MatchExplorerForm({ apiBase, clubSlug, players, contexts }: MatchExplorerFormProps) {
  const activePlayers = useMemo(() => players.filter((player) => player.is_active !== false), [players]);
  const initial = activePlayers.slice(0, 4).map((player) => String(player.id));
  const [context, setContext] = useState(contexts[0] ?? "OVERALL");
  const [me, setMe] = useState(initial[0] ?? "");
  const [partner, setPartner] = useState(initial[1] ?? "");
  const [opp1, setOpp1] = useState(initial[2] ?? "");
  const [opp2, setOpp2] = useState(initial[3] ?? "");
  const [scoreYou, setScoreYou] = useState("11");
  const [scoreOpp, setScoreOpp] = useState("9");
  const [preview, setPreview] = useState<MatchExplorerPreview | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const options = activePlayers.map((player) => (
    <option key={String(player.id)} value={String(player.id)}>{player.name}</option>
  ));

  async function runPreview() {
    if (!apiBase) {
      setMessage("API base URL is not configured.");
      return;
    }
    const selected = [me, partner, opp1, opp2];
    if (selected.some((value) => !value)) {
      setMessage("Select four players.");
      return;
    }
    if (new Set(selected).size !== 4) {
      setMessage("Select four different players.");
      return;
    }

    setLoading(true);
    setMessage(null);
    setPreview(null);
    try {
      const params = new URLSearchParams({
        me,
        partner,
        opp1,
        opp2,
        context,
        score_you: String(Number(scoreYou || 0)),
        score_opp: String(Number(scoreOpp || 0))
      });
      const response = await fetch(apiUrl(apiBase, `/clubs/${clubSlug}/match-explorer/preview?${params.toString()}`), {
        method: "GET"
      });
      const payload = (await response.json().catch(() => null)) as MatchExplorerPreviewResponse | { detail?: string } | null;
      if (!response.ok) {
        throw new Error(String((payload as { detail?: string } | null)?.detail || `API error (${response.status})`));
      }
      setPreview((payload as MatchExplorerPreviewResponse).preview);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to build Match Explorer preview.");
    } finally {
      setLoading(false);
    }
  }

  const selectStyle = { padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit", width: "100%" };
  const labelStyle = { display: "grid", gap: "0.25rem", fontWeight: 700 };

  return (
    <section style={{ border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" }}>
      <h2 style={{ marginTop: 0 }}>Preview a doubles matchup</h2>
      <div style={{ display: "grid", gap: "0.85rem" }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem" }}>
          <label style={labelStyle}>Rating context<select value={context} onChange={(event) => setContext(event.target.value)} style={selectStyle}>{contexts.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
          <label style={labelStyle}>Your points<input value={scoreYou} onChange={(event) => setScoreYou(event.target.value)} type="number" min={0} max={99} style={selectStyle} /></label>
          <label style={labelStyle}>Opponent points<input value={scoreOpp} onChange={(event) => setScoreOpp(event.target.value)} type="number" min={0} max={99} style={selectStyle} /></label>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem" }}>
          <label style={labelStyle}>I am<select value={me} onChange={(event) => setMe(event.target.value)} style={selectStyle}><option value="">Select</option>{options}</select></label>
          <label style={labelStyle}>My partner<select value={partner} onChange={(event) => setPartner(event.target.value)} style={selectStyle}><option value="">Select</option>{options}</select></label>
          <label style={labelStyle}>Opponent 1<select value={opp1} onChange={(event) => setOpp1(event.target.value)} style={selectStyle}><option value="">Select</option>{options}</select></label>
          <label style={labelStyle}>Opponent 2<select value={opp2} onChange={(event) => setOpp2(event.target.value)} style={selectStyle}><option value="">Select</option>{options}</select></label>
        </div>

        <div>
          <button type="button" onClick={runPreview} disabled={loading} style={{ border: 0, borderRadius: "999px", padding: "0.65rem 1rem", background: "#2563eb", color: "white", fontWeight: 800 }}>
            {loading ? "Previewing…" : "Preview matchup"}
          </button>
        </div>

        <p style={{ color: "#475569", margin: 0 }}>
          Current selection: {playerName(activePlayers, me) || "—"} / {playerName(activePlayers, partner) || "—"} vs {playerName(activePlayers, opp1) || "—"} / {playerName(activePlayers, opp2) || "—"}
        </p>
        {message ? <p style={{ color: "#b91c1c" }}>{message}</p> : null}
      </div>

      {preview ? (
        <div style={{ borderTop: "1px solid #e2e8f0", marginTop: "1rem", paddingTop: "1rem" }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
            <article style={{ border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "#f8fafc" }}>
              <h3 style={{ marginTop: 0 }}>Expected win rate</h3>
              <p style={{ fontSize: "2rem", fontWeight: 800, margin: 0 }}>{percentLabel(preview.expected.you)}</p>
              <p style={{ margin: 0, color: "#475569" }}>{preview.expected.label} • {preview.context.name}</p>
            </article>
            <article style={{ border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "#f8fafc" }}>
              <h3 style={{ marginTop: 0 }}>Your team delta</h3>
              <p style={{ fontSize: "2rem", fontWeight: 800, margin: 0 }}>{deltaLabel(preview.rating_delta.you_team_elo)}</p>
              <p style={{ margin: 0, color: "#475569" }}>Preview only — nothing is saved.</p>
            </article>
            <article style={{ border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "#f8fafc" }}>
              <h3 style={{ marginTop: 0 }}>Opponents delta</h3>
              <p style={{ fontSize: "2rem", fontWeight: 800, margin: 0 }}>{deltaLabel(preview.rating_delta.opponent_team_elo)}</p>
              <p style={{ margin: 0, color: "#475569" }}>K-factor {preview.context.k_factor}</p>
            </article>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1rem" }}>
            <TeamSummary title="Your team" team={preview.teams.you} />
            <TeamSummary title="Opponents" team={preview.teams.opponents} />
          </div>
        </div>
      ) : null}
    </section>
  );
}
