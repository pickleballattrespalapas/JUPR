"use client";

import { useMemo, useState } from "react";
import type { PublicPlayer } from "@/lib/api";
import type { AdminMatchUploaderStatusResponse, AdminMatchUploaderWriteResult } from "@/lib/adminMatchUploaderApi";

type MatchUploaderFormProps = {
  apiBase: string | null;
  clubId: string;
  players: PublicPlayer[];
  status: AdminMatchUploaderStatusResponse;
};

type MatchRow = {
  row_id: string;
  date: string;
  league: string;
  week_tag: string;
  match_type: string;
  rating_scope: "" | "overall_only" | "unrated";
  t1_p1: string;
  t1_p2: string;
  t2_p1: string;
  t2_p2: string;
  score_t1: string;
  score_t2: string;
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.5rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };

function todayIsoDate(): string {
  return new Date().toISOString().slice(0, 10);
}

function newRow(defaults?: Partial<MatchRow>): MatchRow {
  const base: MatchRow = {
    row_id: `${Date.now()}-${Math.random()}`,
    date: todayIsoDate(),
    league: "Open",
    week_tag: "Week 1",
    match_type: "Live Match",
    rating_scope: "",
    t1_p1: "",
    t1_p2: "",
    t2_p1: "",
    t2_p2: "",
    score_t1: "0",
    score_t2: "0"
  };
  return { ...base, ...(defaults || {}) };
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

function isFilled(row: MatchRow): boolean {
  return Boolean(row.t1_p1 || row.t1_p2 || row.t2_p1 || row.t2_p2 || Number(row.score_t1 || 0) + Number(row.score_t2 || 0) > 0);
}

function validateRow(row: MatchRow, index: number): string | null {
  if (!isFilled(row)) return null;
  const pids = [row.t1_p1, row.t1_p2, row.t2_p1, row.t2_p2].filter(Boolean);
  if (pids.length !== 4) return `Row ${index + 1}: select four players.`;
  if (new Set(pids).size !== 4) return `Row ${index + 1}: select four different players.`;
  const score1 = Number(row.score_t1 || 0);
  const score2 = Number(row.score_t2 || 0);
  if (!Number.isFinite(score1) || !Number.isFinite(score2) || score1 < 0 || score2 < 0) return `Row ${index + 1}: scores must be non-negative numbers.`;
  if (score1 + score2 <= 0) return `Row ${index + 1}: enter a non-zero score.`;
  if (!row.league.trim()) return `Row ${index + 1}: league is required.`;
  return null;
}

export default function MatchUploaderForm({ apiBase, clubId, players, status }: MatchUploaderFormProps) {
  const [token, setToken] = useState("");
  const [context, setContext] = useState<"league" | "popup">("league");
  const [defaultLeague, setDefaultLeague] = useState(status.league_options[0] || "Open");
  const [defaultWeekTag, setDefaultWeekTag] = useState(status.week_tag_options[0] || "Week 1");
  const [defaultDate, setDefaultDate] = useState(todayIsoDate());
  const [rows, setRows] = useState<MatchRow[]>(() => Array.from({ length: 5 }, () => newRow()));
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [result, setResult] = useState<AdminMatchUploaderWriteResult | null>(null);

  const playerOptions = useMemo(() => players.map((player) => <option key={String(player.id)} value={String(player.id)}>{player.name}</option>), [players]);
  const validRows = rows.filter(isFilled);

  function patchRow(rowId: string, patch: Partial<MatchRow>) {
    setRows((current) => current.map((row) => row.row_id === rowId ? { ...row, ...patch } : row));
  }

  function addRows(count: number) {
    const defaults = {
      date: defaultDate,
      league: context === "popup" ? "POPUP" : defaultLeague,
      week_tag: defaultWeekTag,
      match_type: context === "popup" ? "PopUp" : "Live Match"
    };
    setRows((current) => [...current, ...Array.from({ length: count }, () => newRow(defaults))]);
  }

  function applyDefaultsToEmptyRows() {
    const defaults = {
      date: defaultDate,
      league: context === "popup" ? "POPUP" : defaultLeague,
      week_tag: defaultWeekTag,
      match_type: context === "popup" ? "PopUp" : "Live Match"
    };
    setRows((current) => current.map((row) => isFilled(row) ? row : { ...row, ...defaults }));
  }

  async function submitBatch() {
    setMessage(null);
    setResult(null);
    if (!apiBase) {
      setMessage("API base URL is not configured.");
      return;
    }
    if (!token.trim()) {
      setMessage("Paste a Supabase admin access token first.");
      return;
    }
    if (!status.enabled) {
      setMessage("Next Match Uploader is disabled on the API.");
      return;
    }
    const errors = rows.map(validateRow).filter(Boolean) as string[];
    if (errors.length) {
      setMessage(errors[0]);
      return;
    }
    const matches = validRows.map((row) => ({
      date: row.date,
      league: context === "popup" ? "POPUP" : row.league,
      week_tag: row.week_tag,
      match_type: context === "popup" ? "PopUp" : row.match_type,
      rating_scope: row.rating_scope || undefined,
      is_popup: context === "popup",
      context_type: context === "popup" ? "event" : null,
      t1_p1: Number(row.t1_p1),
      t1_p2: Number(row.t1_p2),
      t2_p1: Number(row.t2_p1),
      t2_p2: Number(row.t2_p2),
      score_t1: Number(row.score_t1),
      score_t2: Number(row.score_t2)
    }));
    if (!matches.length) {
      setMessage("Enter at least one complete match row.");
      return;
    }
    setSaving(true);
    try {
      const response = await fetch(apiUrl(apiBase, `/admin/clubs/${encodeURIComponent(clubId)}/match-uploader/batch`), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token.trim()}`
        },
        body: JSON.stringify({ source: "next_match_uploader_manual_batch", matches })
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
      setResult(payload as AdminMatchUploaderWriteResult);
      setMessage(`Submitted ${payload?.submitted_count ?? matches.length} row(s); inserted ${payload?.result?.inserted ?? 0} rated match(es).`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to submit batch.");
    } finally {
      setSaving(false);
    }
  }

  if (!status.enabled) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Next Match Uploader is disabled</h2>
        <p style={{ color: "#475569" }}>{status.warnings?.[0] || "Enable the Match Uploader pilot flag on FastAPI."}</p>
      </article>
    );
  }

  return (
    <section style={{ display: "grid", gap: "1rem" }}>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Manual / batch score entry</h2>
        <p style={{ color: "#475569" }}>
          This replaces the Streamlit manual batch path for the closed-club pilot. It submits only through FastAPI, uses Supabase JWT role authorization, and calls the existing Python match-processing service.
        </p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem" }}>
          <label><strong>Supabase access token</strong><br /><input value={token} onChange={(event) => setToken(event.target.value)} type="password" style={inputStyle} /></label>
          <label><strong>Context</strong><br />
            <select value={context} onChange={(event) => setContext(event.target.value as "league" | "popup")} style={inputStyle}>
              <option value="league">Official League</option>
              <option value="popup">Pop-Up / Social</option>
            </select>
          </label>
          <label><strong>Default date</strong><br /><input value={defaultDate} onChange={(event) => setDefaultDate(event.target.value)} type="date" style={inputStyle} /></label>
          <label><strong>Default league</strong><br />
            <select value={defaultLeague} onChange={(event) => setDefaultLeague(event.target.value)} disabled={context === "popup"} style={inputStyle}>
              {status.league_options.map((league) => <option key={league}>{league}</option>)}
            </select>
          </label>
          <label><strong>Default week/session</strong><br />
            <select value={defaultWeekTag} onChange={(event) => setDefaultWeekTag(event.target.value)} style={inputStyle}>
              {status.week_tag_options.map((week) => <option key={week}>{week}</option>)}
            </select>
          </label>
        </div>
        <p style={{ marginBottom: 0 }}>
          <button type="button" onClick={applyDefaultsToEmptyRows} style={{ ...buttonStyle, background: "white", color: "#0f172a" }}>Apply defaults to empty rows</button>{" "}
          <button type="button" onClick={() => addRows(5)} disabled={rows.length >= status.max_batch_rows} style={{ ...buttonStyle, background: "white", color: "#0f172a" }}>Add 5 rows</button>
        </p>
      </article>

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Rows</h2>
        <div style={{ display: "grid", gap: "0.75rem" }}>
          {rows.map((row, index) => (
            <div key={row.row_id} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: isFilled(row) ? "#f8fafc" : "white" }}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: "0.5rem", alignItems: "center", marginBottom: "0.5rem" }}>
                <strong>Match {index + 1}</strong>
                <button type="button" onClick={() => setRows((current) => current.filter((item) => item.row_id !== row.row_id))} disabled={rows.length <= 1}>Remove</button>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: "0.5rem" }}>
                <input type="date" value={row.date} onChange={(event) => patchRow(row.row_id, { date: event.target.value })} style={inputStyle} aria-label={`Date row ${index + 1}`} />
                <input value={context === "popup" ? "POPUP" : row.league} onChange={(event) => patchRow(row.row_id, { league: event.target.value })} disabled={context === "popup"} placeholder="League" style={inputStyle} />
                <input value={row.week_tag} onChange={(event) => patchRow(row.row_id, { week_tag: event.target.value })} placeholder="Week" style={inputStyle} />
                <input value={context === "popup" ? "PopUp" : row.match_type} onChange={(event) => patchRow(row.row_id, { match_type: event.target.value })} disabled={context === "popup"} placeholder="Match type" style={inputStyle} />
                <select value={row.rating_scope} onChange={(event) => patchRow(row.row_id, { rating_scope: event.target.value as MatchRow["rating_scope"] })} style={inputStyle}>
                  <option value="">Overall + league</option>
                  <option value="overall_only">Overall only</option>
                  <option value="unrated">Unrated / record only</option>
                </select>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "0.5rem", marginTop: "0.5rem" }}>
                <select value={row.t1_p1} onChange={(event) => patchRow(row.row_id, { t1_p1: event.target.value })} style={inputStyle}><option value="">T1 P1</option>{playerOptions}</select>
                <select value={row.t1_p2} onChange={(event) => patchRow(row.row_id, { t1_p2: event.target.value })} style={inputStyle}><option value="">T1 P2</option>{playerOptions}</select>
                <input value={row.score_t1} onChange={(event) => patchRow(row.row_id, { score_t1: event.target.value })} type="number" min={0} max={99} style={inputStyle} aria-label={`Team 1 score row ${index + 1}`} />
                <input value={row.score_t2} onChange={(event) => patchRow(row.row_id, { score_t2: event.target.value })} type="number" min={0} max={99} style={inputStyle} aria-label={`Team 2 score row ${index + 1}`} />
                <select value={row.t2_p1} onChange={(event) => patchRow(row.row_id, { t2_p1: event.target.value })} style={inputStyle}><option value="">T2 P1</option>{playerOptions}</select>
                <select value={row.t2_p2} onChange={(event) => patchRow(row.row_id, { t2_p2: event.target.value })} style={inputStyle}><option value="">T2 P2</option>{playerOptions}</select>
              </div>
            </div>
          ))}
        </div>
        <p><strong>Ready rows:</strong> {validRows.length} / {rows.length}</p>
        <button type="button" onClick={submitBatch} disabled={saving || !validRows.length} style={buttonStyle}>{saving ? "Submitting…" : "Submit batch"}</button>
        {message ? <p style={{ color: result?.ok ? "#166534" : "#b91c1c" }}>{message}</p> : null}
      </article>

      {result ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Submission result</h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "0.75rem" }}>
            <div><strong>Inserted</strong><br />{result.result?.inserted ?? 0}</div>
            <div><strong>Skipped incomplete</strong><br />{result.result?.skipped_incomplete ?? 0}</div>
            <div><strong>Skipped empty</strong><br />{result.result?.skipped_empty ?? 0}</div>
            <div><strong>Skipped unrated</strong><br />{result.result?.skipped_unrated ?? 0}</div>
          </div>
          {result.feedback?.affected_players?.length ? (
            <div style={{ overflowX: "auto", marginTop: "1rem" }}>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead><tr><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Player</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Before</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>After</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Change</th></tr></thead>
                <tbody>
                  {result.feedback.affected_players.map((player) => (
                    <tr key={player.id}>
                      <td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{player.name}</td>
                      <td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{ratingLabel(player.rating_before)}</td>
                      <td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{ratingLabel(player.rating_after)}</td>
                      <td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{deltaLabel(player.rating_delta)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
          {result.warnings?.length ? <ul style={{ color: "#92400e" }}>{result.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
        </article>
      ) : null}
    </section>
  );
}
