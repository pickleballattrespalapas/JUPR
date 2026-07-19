"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { MatchExplorerPreview, MatchExplorerPreviewResponse, PublicPlayer } from "@/lib/api";

type InitialMatchExplorerSelection = {
  context?: string | null;
  me?: string | null;
  partner?: string | null;
  opp1?: string | null;
  opp2?: string | null;
  scoreYou?: string | null;
  scoreOpp?: string | null;
};

type MatchExplorerFormProps = {
  apiBase: string | null;
  clubSlug: string;
  players: PublicPlayer[];
  contexts: string[];
  initialSelection?: InitialMatchExplorerSelection;
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

function cleanScore(value: string | null | undefined, fallback: string): string {
  if (value == null || String(value).trim() === "") return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return String(Math.min(99, Math.max(0, Math.round(parsed))));
}

function initialPlayerId(playerIds: Set<string>, value: string | null | undefined, fallback: string): string {
  const cleaned = String(value ?? "").trim();
  return cleaned && playerIds.has(cleaned) ? cleaned : fallback;
}

function beatExpectationLabel(preview: MatchExplorerPreview): { value: string; caption: string } {
  const share = preview.score.you_share;
  const beatPp = preview.score.beat_expectation_pp;
  if (share == null || beatPp == null) {
    return { value: "—", caption: "No movement on ties / empty scores." };
  }
  return {
    value: `${beatPp >= 0 ? "+" : ""}${beatPp.toFixed(0)} pp`,
    caption: `Your share ${(share * 100).toFixed(1)}% vs expected ${(preview.expected.you * 100).toFixed(1)}%`
  };
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

function SummaryCards({ preview }: { preview: MatchExplorerPreview }) {
  const beat = beatExpectationLabel(preview);

  return (
    <div data-testid="match-explorer-summary" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
      <article style={{ border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "#f8fafc" }}>
        <h3 style={{ marginTop: 0 }}>Expected win rate</h3>
        <p style={{ fontSize: "2rem", fontWeight: 800, margin: 0 }}>{percentLabel(preview.expected.you)}</p>
        <p style={{ margin: 0, color: "#475569" }}>
          {preview.expected.label} • expected {preview.expected.score_to_11.label} • {preview.context.name}
        </p>
      </article>
      <article style={{ border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "#f8fafc" }}>
        <h3 style={{ marginTop: 0 }}>Beat expectation</h3>
        <p style={{ fontSize: "2rem", fontWeight: 800, margin: 0 }}>{beat.value}</p>
        <p style={{ margin: 0, color: "#475569" }}>{beat.caption}</p>
      </article>
      <article style={{ border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "#f8fafc" }}>
        <h3 style={{ marginTop: 0 }}>Your / partner delta</h3>
        <p style={{ fontSize: "2rem", fontWeight: 800, margin: 0 }}>{deltaLabel(preview.rating_delta.you_team_elo)}</p>
        <p style={{ margin: 0, color: "#475569" }}>Same team-based update.</p>
      </article>
      <article style={{ border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "#f8fafc" }}>
        <h3 style={{ marginTop: 0 }}>Opponents delta</h3>
        <p style={{ fontSize: "2rem", fontWeight: 800, margin: 0 }}>{deltaLabel(preview.rating_delta.opponent_team_elo)}</p>
        <p style={{ margin: 0, color: "#475569" }}>K-factor {preview.context.k_factor}</p>
      </article>
    </div>
  );
}

function PlayerImpactTable({ preview }: { preview: MatchExplorerPreview }) {
  return (
    <div data-testid="match-explorer-player-impact" style={{ overflowX: "auto", border: "1px solid #e2e8f0", borderRadius: "14px", marginTop: "1rem" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "620px" }}>
        <thead>
          <tr style={{ background: "#f8fafc", textAlign: "left" }}>
            <th style={{ padding: "0.75rem", borderBottom: "1px solid #e2e8f0" }}>Role</th>
            <th style={{ padding: "0.75rem", borderBottom: "1px solid #e2e8f0" }}>Player</th>
            <th style={{ padding: "0.75rem", borderBottom: "1px solid #e2e8f0" }}>Current JUPR</th>
            <th style={{ padding: "0.75rem", borderBottom: "1px solid #e2e8f0" }}>Projected JUPR</th>
            <th style={{ padding: "0.75rem", borderBottom: "1px solid #e2e8f0" }}>Δ JUPR</th>
          </tr>
        </thead>
        <tbody>
          {preview.player_impacts.map((row) => (
            <tr key={`${row.role}-${String(row.player.id)}`}>
              <td style={{ padding: "0.75rem", borderTop: "1px solid #f1f5f9", fontWeight: 700 }}>{row.role}</td>
              <td style={{ padding: "0.75rem", borderTop: "1px solid #f1f5f9" }}>{row.player.name}</td>
              <td style={{ padding: "0.75rem", borderTop: "1px solid #f1f5f9" }}>{ratingLabel(row.current_rating)}</td>
              <td style={{ padding: "0.75rem", borderTop: "1px solid #f1f5f9" }}>{ratingLabel(row.projected_rating)}</td>
              <td style={{ padding: "0.75rem", borderTop: "1px solid #f1f5f9" }}>{deltaLabel(row.delta_elo)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RatingImpactChart({ preview }: { preview: MatchExplorerPreview }) {
  const width = 720;
  const height = 320;
  const margin = { top: 18, right: 24, bottom: 58, left: 62 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const curve = preview.impact_chart.points;
  const selectedMarker = preview.impact_chart.selected_marker;
  const selectedShare = selectedMarker?.score_share ?? null;
  const selectedDelta = selectedMarker?.you_team_jupr ?? null;
  const values = curve.map((point) => point.you_team_jupr).concat([0, selectedDelta ?? 0]);
  let minDelta = Math.min(...values);
  let maxDelta = Math.max(...values);
  const pad = Math.max((maxDelta - minDelta) * 0.12, 0.01);
  minDelta -= pad;
  maxDelta += pad;

  const x = (share: number) => margin.left + share * innerWidth;
  const y = (delta: number) => margin.top + (maxDelta - delta) * (innerHeight / (maxDelta - minDelta));
  const path = curve.map((point, index) => `${index === 0 ? "M" : "L"}${x(point.score_share).toFixed(1)} ${y(point.you_team_jupr).toFixed(1)}`).join(" ");
  const xTicks = preview.impact_chart.score_ticks;
  const yTicks = Array.from(new Set([minDelta, 0, maxDelta].map((value) => Number(value.toFixed(4))))).sort((a, b) => a - b);
  const selectedX = selectedShare == null ? null : x(selectedShare);
  const selectedY = selectedShare == null || selectedDelta == null ? null : y(selectedDelta);
  const expectedX = x(preview.impact_chart.expected_marker.score_share);

  return (
    <article data-testid="match-explorer-impact-chart" style={{ border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "#ffffff", marginTop: "1rem" }}>
      <h3 style={{ marginTop: 0 }}>Rating Impact Predictor</h3>
      <p style={{ color: "#475569", marginTop: "-0.35rem" }}>
        The curve mirrors the Streamlit predictor: x-axis is score share translated to an 11-point score, and y-axis is projected JUPR change for your team.
      </p>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby="mx-impact-title" style={{ width: "100%", height: "auto", display: "block" }}>
        <title id="mx-impact-title">Projected JUPR change by score share</title>
        <line x1={margin.left} y1={y(0)} x2={width - margin.right} y2={y(0)} stroke="#cbd5e1" strokeDasharray="4 4" />
        {yTicks.map((tick) => (
          <g key={`y-${tick}`}>
            <line x1={margin.left - 6} y1={y(tick)} x2={width - margin.right} y2={y(tick)} stroke="#f1f5f9" />
            <text x={margin.left - 10} y={y(tick) + 4} textAnchor="end" fontSize="11" fill="#475569">{`${tick >= 0 ? "+" : ""}${tick.toFixed(4)}`}</text>
          </g>
        ))}
        {xTicks.map((tick) => (
          <g key={`x-${tick.score_share}`}>
            <line x1={x(tick.score_share)} y1={height - margin.bottom} x2={x(tick.score_share)} y2={height - margin.bottom + 6} stroke="#94a3b8" />
            <text x={x(tick.score_share)} y={height - margin.bottom + 22} textAnchor="middle" fontSize="11" fill="#475569">{tick.score_to_11.label}</text>
          </g>
        ))}
        <line x1={expectedX} y1={margin.top} x2={expectedX} y2={height - margin.bottom} stroke="#64748b" strokeDasharray="6 4" />
        <text x={expectedX + 6} y={margin.top + 12} fontSize="11" fill="#475569">Expected {preview.impact_chart.expected_marker.score_to_11.label}</text>
        <path d={path} fill="none" stroke="#2563eb" strokeWidth="3" strokeLinecap="round" />
        {selectedX != null && selectedY != null ? (
          <g>
            <circle cx={selectedX} cy={selectedY} r="6" fill="#0f172a" />
            <text x={selectedX + 8} y={selectedY - 8} fontSize="11" fill="#0f172a">Actual {preview.score.you}–{preview.score.opponents}</text>
          </g>
        ) : null}
        <line x1={margin.left} y1={height - margin.bottom} x2={width - margin.right} y2={height - margin.bottom} stroke="#94a3b8" />
        <line x1={margin.left} y1={margin.top} x2={margin.left} y2={height - margin.bottom} stroke="#94a3b8" />
        <text x={margin.left + innerWidth / 2} y={height - 14} textAnchor="middle" fontSize="12" fill="#334155">Score share, shown as equivalent score to 11</text>
        <text x="16" y={margin.top + innerHeight / 2} textAnchor="middle" fontSize="12" fill="#334155" transform={`rotate(-90 16 ${margin.top + innerHeight / 2})`}>Projected Δ JUPR</text>
      </svg>
    </article>
  );
}

export default function MatchExplorerForm({ apiBase, clubSlug, players, contexts, initialSelection }: MatchExplorerFormProps) {
  const activePlayers = useMemo(() => players.filter((player) => player.is_active !== false), [players]);
  const activePlayerIds = useMemo(() => new Set(activePlayers.map((player) => String(player.id))), [activePlayers]);
  const initial = useMemo(() => activePlayers.slice(0, 4).map((player) => String(player.id)), [activePlayers]);
  const initialContextCandidate = String(initialSelection?.context ?? "").trim();
  const initialContext = contexts.includes(initialContextCandidate) ? initialContextCandidate : contexts[0] ?? "OVERALL";
  const initialMe = initialPlayerId(activePlayerIds, initialSelection?.me, initial[0] ?? "");
  const initialPartner = initialPlayerId(activePlayerIds, initialSelection?.partner, initial[1] ?? "");
  const initialOpp1 = initialPlayerId(activePlayerIds, initialSelection?.opp1, initial[2] ?? "");
  const initialOpp2 = initialPlayerId(activePlayerIds, initialSelection?.opp2, initial[3] ?? "");

  const [context, setContext] = useState(initialContext);
  const [me, setMe] = useState(initialMe);
  const [partner, setPartner] = useState(initialPartner);
  const [opp1, setOpp1] = useState(initialOpp1);
  const [opp2, setOpp2] = useState(initialOpp2);
  const [scoreYou, setScoreYou] = useState(cleanScore(initialSelection?.scoreYou, "11"));
  const [scoreOpp, setScoreOpp] = useState(cleanScore(initialSelection?.scoreOpp, "9"));
  const [preview, setPreview] = useState<MatchExplorerPreview | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [infoMessage, setInfoMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const requestSequence = useRef(0);

  const selectedPlayers = useMemo(() => [me, partner, opp1, opp2], [me, partner, opp1, opp2]);
  const hasValidSelection = selectedPlayers.every((value) => Boolean(value)) && new Set(selectedPlayers).size === 4;

  const options = activePlayers.map((player) => (
    <option key={String(player.id)} value={String(player.id)}>{player.name}</option>
  ));

  const syncUrlParams = useCallback(() => {
    if (typeof window === "undefined") return null;
    const url = new URL(window.location.href);
    url.searchParams.set("ctx", context);
    url.searchParams.set("me", me);
    url.searchParams.set("partner", partner);
    url.searchParams.set("opp1", opp1);
    url.searchParams.set("opp2", opp2);
    url.searchParams.set("sy", cleanScore(scoreYou, "11"));
    url.searchParams.set("so", cleanScore(scoreOpp, "9"));
    url.searchParams.delete("context");
    url.searchParams.delete("league");
    url.searchParams.delete("score_you");
    url.searchParams.delete("score_opp");
    window.history.replaceState(null, "", `${url.pathname}?${url.searchParams.toString()}${url.hash}`);
    return url.toString();
  }, [context, me, partner, opp1, opp2, scoreOpp, scoreYou]);

  const runPreview = useCallback(async (optionsOverride: { syncUrl?: boolean; signal?: AbortSignal } = {}) => {
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

    const requestId = requestSequence.current + 1;
    requestSequence.current = requestId;
    setLoading(true);
    setMessage(null);
    setInfoMessage(null);
    setPreview(null);
    try {
      if (optionsOverride.syncUrl !== false) syncUrlParams();
      const params = new URLSearchParams({
        me,
        partner,
        opp1,
        opp2,
        context,
        score_you: cleanScore(scoreYou, "11"),
        score_opp: cleanScore(scoreOpp, "9")
      });
      const response = await fetch(apiUrl(apiBase, `/clubs/${clubSlug}/match-explorer/preview?${params.toString()}`), {
        method: "GET",
        signal: optionsOverride.signal
      });
      const payload = (await response.json().catch(() => null)) as MatchExplorerPreviewResponse | { detail?: string } | null;
      if (!response.ok) {
        throw new Error(String((payload as { detail?: string } | null)?.detail || `API error (${response.status})`));
      }
      if (requestId === requestSequence.current) {
        setPreview((payload as MatchExplorerPreviewResponse).preview);
      }
    } catch (error) {
      if (optionsOverride.signal?.aborted || (error instanceof DOMException && error.name === "AbortError")) return;
      if (requestId === requestSequence.current) {
        setMessage(error instanceof Error ? error.message : "Unable to build Match Explorer preview.");
      }
    } finally {
      if (requestId === requestSequence.current) setLoading(false);
    }
  }, [apiBase, clubSlug, context, me, opp1, opp2, partner, scoreOpp, scoreYou, syncUrlParams]);

  useEffect(() => {
    if (!selectedPlayers.every((value) => Boolean(value))) {
      setPreview(null);
      setMessage("Select four players.");
      return;
    }
    if (!hasValidSelection) {
      setPreview(null);
      setMessage("Select four different players.");
      return;
    }

    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      void runPreview({ syncUrl: false, signal: controller.signal });
    }, 350);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [hasValidSelection, runPreview, selectedPlayers]);

  async function copyShareLink() {
    if (!hasValidSelection) {
      setMessage("Select four different players before copying a share link.");
      return;
    }
    const nextUrl = syncUrlParams();
    if (!nextUrl) return;
    try {
      await navigator.clipboard.writeText(nextUrl);
      setInfoMessage("Share link copied.");
      setMessage(null);
    } catch {
      setInfoMessage("Share link updated in the address bar.");
      setMessage(null);
    }
  }

  const selectStyle = { padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit", width: "100%" };
  const labelStyle = { display: "grid", gap: "0.25rem", fontWeight: 700 };

  return (
    <section data-testid="match-explorer-form" style={{ border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" }}>
      <h2 style={{ marginTop: 0 }}>Preview a doubles matchup</h2>
      <p style={{ color: "#475569", marginTop: "-0.25rem" }}>
        Selections are shareable through the URL. Use the same `ctx`, `me`, `partner`, `opp1`, `opp2`, `sy`, and `so` deep-link fields as the Streamlit page.
      </p>
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

        <div style={{ display: "flex", flexWrap: "wrap", gap: "0.6rem" }}>
          <button type="button" onClick={() => void runPreview()} disabled={loading || !hasValidSelection} style={{ border: 0, borderRadius: "999px", padding: "0.65rem 1rem", background: "#2563eb", color: "white", fontWeight: 800 }}>
            {loading ? "Previewing…" : "Preview matchup"}
          </button>
          <button type="button" onClick={() => void copyShareLink()} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.65rem 1rem", background: "white", color: "#0f172a", fontWeight: 800 }}>
            Copy share link
          </button>
        </div>

        <p style={{ color: "#475569", margin: 0 }}>
          Current selection: {playerName(activePlayers, me) || "—"} / {playerName(activePlayers, partner) || "—"} vs {playerName(activePlayers, opp1) || "—"} / {playerName(activePlayers, opp2) || "—"}
        </p>
        <div aria-live="polite">
          {message ? <p data-testid="match-explorer-validation" style={{ color: "#b91c1c" }}>{message}</p> : null}
          {infoMessage ? <p style={{ color: "#047857", fontWeight: 700 }}>{infoMessage}</p> : null}
        </div>
      </div>

      {preview ? (
        <div style={{ borderTop: "1px solid #e2e8f0", marginTop: "1rem", paddingTop: "1rem" }}>
          <SummaryCards preview={preview} />

          {preview.context.name !== "OVERALL" ? (
            <p style={{ border: "1px solid #bfdbfe", borderRadius: "12px", padding: "0.75rem", background: "#eff6ff", color: "#1e3a8a" }}>
              Graph and projected movement are computed using league ratings only. Overall ratings in the team cards remain visible for reference.
            </p>
          ) : null}

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1rem" }}>
            <TeamSummary title="Your team" team={preview.teams.you} />
            <TeamSummary title="Opponents" team={preview.teams.opponents} />
          </div>

          <PlayerImpactTable preview={preview} />
          <RatingImpactChart preview={preview} />
        </div>
      ) : null}
    </section>
  );
}
