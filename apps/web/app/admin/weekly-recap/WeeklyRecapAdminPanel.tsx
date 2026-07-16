"use client";

import { useMemo, useState } from "react";
import type {
  AdminWeeklyRecapCandidate,
  AdminWeeklyRecapDetailResponse,
  AdminWeeklyRecapListResponse,
  AdminWeeklyRecapRow,
  AdminWeeklyRecapStatusResponse,
  AdminWeeklyRecapWriteResponse
} from "@/lib/adminWeeklyRecapApi";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type Props = { apiBase: string | null; clubId: string; status: AdminWeeklyRecapStatusResponse };
type SpotlightEdit = { include: boolean; order: string; description: string; players: string[] };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };
const dangerButtonStyle = { ...buttonStyle, background: "#991b1b", borderColor: "#991b1b" };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function todayIso(): string {
  return new Date().toISOString().slice(0, 10);
}

function firstDayOfWeekIso(): string {
  const d = new Date();
  const day = d.getDay();
  const diff = (day + 6) % 7;
  d.setDate(d.getDate() - diff);
  return d.toISOString().slice(0, 10);
}

function shortValue(value: unknown): string {
  if (value == null || value === "") return "—";
  if (typeof value === "object") return JSON.stringify(value).slice(0, 220);
  return String(value);
}

function numbersFromRecap(recap: AdminWeeklyRecapRow | null): Record<string, unknown> {
  const finalJson = recap?.final_json || {};
  const numbers = (finalJson as Record<string, unknown>).numbers;
  return numbers && typeof numbers === "object" ? numbers as Record<string, unknown> : {};
}

function finalSpotlight(recap: AdminWeeklyRecapRow | null): Array<Record<string, unknown>> {
  const finalJson = recap?.final_json || {};
  const spotlight = (finalJson as Record<string, unknown>).spotlight;
  return Array.isArray(spotlight) ? spotlight.filter((item) => item && typeof item === "object") as Array<Record<string, unknown>> : [];
}

function normalizeCandidateKeys(candidates: Record<string, AdminWeeklyRecapCandidate[]>): string[] {
  return Object.keys(candidates || {}).filter((key) => (candidates[key] || []).length > 0);
}

function buildSpotlightEdits(recap: AdminWeeklyRecapRow | null, candidates: Record<string, AdminWeeklyRecapCandidate[]>): Record<string, SpotlightEdit> {
  const generated = Array.isArray(recap?.generated_json?.spotlight) ? recap?.generated_json?.spotlight as Array<Record<string, unknown>> : [];
  const currentOverrides = (recap?.edits_json?.spotlight_overrides && typeof recap.edits_json.spotlight_overrides === "object")
    ? recap.edits_json.spotlight_overrides as Record<string, Record<string, unknown>>
    : {};
  const result: Record<string, SpotlightEdit> = {};
  normalizeCandidateKeys(candidates).forEach((key, idx) => {
    const generatedRow = generated.find((item) => String(item.key || "") === key) || {};
    const override = currentOverrides[key] || {};
    const fallbackCandidateIds = (candidates[key] || []).slice(0, 1).map((candidate) => candidate.candidate_id);
    const rawPlayers = Array.isArray(override.players) ? override.players : Array.isArray(generatedRow.candidate_ids) ? generatedRow.candidate_ids : fallbackCandidateIds;
    result[key] = {
      include: override.include == null ? Boolean(generatedRow.include ?? true) : Boolean(override.include),
      order: String(override.order || generatedRow.order || idx + 1),
      description: String(override.description || generatedRow.description || ""),
      players: rawPlayers.map(String).slice(0, 3)
    };
  });
  return result;
}

function buildEditsPayload(lookingAhead: string[], spotlightEdits: Record<string, SpotlightEdit>) {
  const spotlight_overrides: Record<string, { include: boolean; order: number; description: string; players: string[] }> = {};
  Object.entries(spotlightEdits).forEach(([key, value]) => {
    spotlight_overrides[key] = {
      include: Boolean(value.include),
      order: Number(value.order) || 999,
      description: value.description || "",
      players: (value.players || []).filter(Boolean).slice(0, 3)
    };
  });
  return {
    looking_ahead: lookingAhead.map((item) => item.trim()).filter(Boolean),
    spotlight_overrides
  };
}

function messageColor(message: string | null): string {
  const text = (message || "").toLowerCase();
  if (text.includes("unable") || text.includes("error") || text.includes("type") || text.includes("missing")) return "#b91c1c";
  return "#166534";
}

export default function WeeklyRecapAdminPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [weekStart, setWeekStart] = useState(firstDayOfWeekIso());
  const [weekEnd, setWeekEnd] = useState(todayIso());
  const [recaps, setRecaps] = useState<AdminWeeklyRecapRow[]>([]);
  const [selectedWeekStart, setSelectedWeekStart] = useState("");
  const [selectedRecap, setSelectedRecap] = useState<AdminWeeklyRecapRow | null>(null);
  const [candidates, setCandidates] = useState<Record<string, AdminWeeklyRecapCandidate[]>>({});
  const [lookingAhead, setLookingAhead] = useState<string[]>(["", "", ""]);
  const [spotlightEdits, setSpotlightEdits] = useState<Record<string, SpotlightEdit>>({});
  const [generateConfirm, setGenerateConfirm] = useState("");
  const [saveConfirm, setSaveConfirm] = useState("");
  const [publishConfirm, setPublishConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const candidateKeys = useMemo(() => normalizeCandidateKeys(candidates), [candidates]);
  const recapNumbers = numbersFromRecap(selectedRecap);
  const spotlightPreview = finalSpotlight(selectedRecap);

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using Weekly Recap Admin.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
  }

  function applyDetail(payload: AdminWeeklyRecapDetailResponse) {
    setSelectedRecap(payload.recap);
    setSelectedWeekStart(payload.recap.week_start);
    setWeekStart(payload.recap.week_start);
    setWeekEnd(payload.recap.week_end);
    const nextCandidates = payload.candidates || {};
    setCandidates(nextCandidates);
    const edits = payload.recap.edits_json || {};
    const looking = Array.isArray(edits.looking_ahead)
      ? edits.looking_ahead.map(String)
      : Array.isArray(payload.recap.generated_json?.looking_ahead)
        ? (payload.recap.generated_json?.looking_ahead as unknown[]).map(String)
        : [];
    setLookingAhead([looking[0] || "", looking[1] || "", looking[2] || ""]);
    setSpotlightEdits(buildSpotlightEdits(payload.recap, nextCandidates));
  }

  async function loadRecaps() {
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminWeeklyRecapListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/weekly-recap/recaps?limit=100`);
      setRecaps(payload.recaps || []);
      if (!selectedWeekStart && payload.recaps?.length) setSelectedWeekStart(payload.recaps[0].week_start);
      setMessage(`Loaded ${payload.count ?? payload.recaps?.length ?? 0} recap(s).`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to load weekly recaps.");
    } finally {
      setBusy(false);
    }
  }

  async function loadSelectedRecap() {
    const target = selectedWeekStart || weekStart;
    if (!target) {
      setMessage("Select a recap first.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminWeeklyRecapDetailResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/weekly-recap/recaps/${encodeURIComponent(target)}?include_candidates=true`);
      applyDetail(payload);
      setMessage("Weekly recap loaded.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to load weekly recap.");
    } finally {
      setBusy(false);
    }
  }

  async function generateDraft() {
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminWeeklyRecapWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/weekly-recap/generate`, {
        method: "POST",
        body: JSON.stringify({ week_start: weekStart, week_end: weekEnd, confirmation_text: generateConfirm, source: "next_weekly_recap_generate" })
      });
      applyDetail(payload);
      setGenerateConfirm("");
      await loadRecaps();
      setMessage("Draft generated from current match, social, and tournament data.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to generate weekly recap draft.");
    } finally {
      setBusy(false);
    }
  }

  async function saveDraft() {
    if (!selectedRecap) {
      setMessage("Load or generate a recap before saving edits.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminWeeklyRecapWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/weekly-recap/recaps/${encodeURIComponent(selectedRecap.week_start)}`, {
        method: "PATCH",
        body: JSON.stringify({ edits_json: buildEditsPayload(lookingAhead, spotlightEdits), confirmation_text: saveConfirm, source: "next_weekly_recap_save" })
      });
      applyDetail(payload);
      setSaveConfirm("");
      setMessage("Draft edits saved.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to save weekly recap draft.");
    } finally {
      setBusy(false);
    }
  }

  async function publishAction(action: "publish" | "unpublish") {
    if (!selectedRecap) {
      setMessage("Load or generate a recap before publishing.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminWeeklyRecapWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/weekly-recap/recaps/${encodeURIComponent(selectedRecap.week_start)}/publish`, {
        method: "POST",
        body: JSON.stringify({ action, edits_json: buildEditsPayload(lookingAhead, spotlightEdits), confirmation_text: publishConfirm, source: action === "publish" ? "next_weekly_recap_publish" : "next_weekly_recap_unpublish" })
      });
      applyDetail(payload);
      setPublishConfirm("");
      await loadRecaps();
      setMessage(action === "publish" ? "Weekly recap published." : "Weekly recap unpublished and returned to draft.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : `Unable to ${action} weekly recap.`);
    } finally {
      setBusy(false);
    }
  }

  function updateSpotlight(key: string, patch: Partial<SpotlightEdit>) {
    setSpotlightEdits((current) => ({ ...current, [key]: { ...(current[key] || { include: true, order: "999", description: "", players: [] }), ...patch } }));
  }

  function updateSpotlightPlayer(key: string, index: number, value: string) {
    const current = spotlightEdits[key] || { include: true, order: "999", description: "", players: [] };
    const nextPlayers = [...(current.players || [])];
    nextPlayers[index] = value;
    updateSpotlight(key, { players: nextPlayers });
  }

  if (!status.enabled) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Weekly Recap Admin is disabled</h2>
        <p style={{ color: "#475569" }}>Enable <code>JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP</code> on FastAPI before using this workflow.</p>
        {status.warnings?.map((warning) => <p key={warning} style={{ color: "#92400e" }}>{warning}</p>)}
      </article>
    );
  }

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Admin session</h2>
        <p style={{ color: "#475569" }}>{adminSessionLabel(session)}</p>
        {sessionLoading ? <p>Checking session…</p> : null}
        {sessionMessage ? <p style={{ color: "#64748b" }}>{sessionMessage}</p> : null}
      </article>

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>1. Generate or load a recap</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label>Start date<br /><input type="date" value={weekStart} onChange={(event) => setWeekStart(event.target.value)} style={inputStyle} /></label>
          <label>End date<br /><input type="date" value={weekEnd} onChange={(event) => setWeekEnd(event.target.value)} style={inputStyle} /></label>
          <label>Generate confirmation<br /><input value={generateConfirm} onChange={(event) => setGenerateConfirm(event.target.value)} placeholder="GENERATE RECAP" style={inputStyle} /></label>
          <button type="button" onClick={generateDraft} disabled={busy || !accessToken} style={buttonStyle}>{busy ? "Working…" : "Generate draft"}</button>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "minmax(240px, 1fr) auto auto", gap: "0.75rem", marginTop: "0.75rem", alignItems: "end" }}>
          <label>Existing recaps<br />
            <select value={selectedWeekStart} onChange={(event) => setSelectedWeekStart(event.target.value)} style={inputStyle}>
              <option value="">Select recap…</option>
              {recaps.map((row) => <option key={row.week_start} value={row.week_start}>{row.week_start} → {row.week_end} · {row.status}</option>)}
            </select>
          </label>
          <button type="button" onClick={loadRecaps} disabled={busy || !accessToken} style={ghostButtonStyle}>Load recaps</button>
          <button type="button" onClick={loadSelectedRecap} disabled={busy || !selectedWeekStart} style={ghostButtonStyle}>Open selected</button>
        </div>
        {message ? <p style={{ color: messageColor(message) }}>{message}</p> : null}
      </article>

      {selectedRecap ? (
        <>
          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>2. Edit draft</h2>
            <p style={{ color: "#475569" }}><strong>Recap:</strong> {selectedRecap.week_start} → {selectedRecap.week_end} · <strong>Status:</strong> {selectedRecap.status}</p>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
              {[0, 1, 2].map((idx) => (
                <label key={idx}>Looking ahead #{idx + 1}<br />
                  <input value={lookingAhead[idx] || ""} onChange={(event) => setLookingAhead((current) => current.map((item, itemIdx) => itemIdx === idx ? event.target.value : item))} style={inputStyle} />
                </label>
              ))}
            </div>
          </article>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Spotlight reel</h2>
            <p style={{ color: "#475569" }}>Choose up to three candidates per spotlight category. Leave a slot blank to omit it.</p>
            {candidateKeys.length ? candidateKeys.map((key, idx) => {
              const edit = spotlightEdits[key] || { include: true, order: String(idx + 1), description: "", players: [] };
              const options = candidates[key] || [];
              const label = options[0]?.label || key;
              return (
                <section key={key} style={{ borderTop: idx ? "1px solid #e2e8f0" : undefined, paddingTop: idx ? "0.75rem" : 0, marginTop: idx ? "0.75rem" : 0 }}>
                  <div style={{ display: "grid", gridTemplateColumns: "120px 90px 1fr", gap: "0.75rem", alignItems: "end" }}>
                    <label><input type="checkbox" checked={edit.include} onChange={(event) => updateSpotlight(key, { include: event.target.checked })} /> Include<br /><strong>{label}</strong></label>
                    <label>Order<br /><input value={edit.order} onChange={(event) => updateSpotlight(key, { order: event.target.value })} style={inputStyle} /></label>
                    <label>Description<br /><input value={edit.description} onChange={(event) => updateSpotlight(key, { description: event.target.value })} style={inputStyle} /></label>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem", marginTop: "0.75rem" }}>
                    {[0, 1, 2].map((slot) => (
                      <label key={slot}>Candidate {slot + 1}<br />
                        <select value={edit.players?.[slot] || ""} onChange={(event) => updateSpotlightPlayer(key, slot, event.target.value)} style={inputStyle}>
                          <option value="">None</option>
                          {options.map((candidate) => <option key={candidate.candidate_id} value={candidate.candidate_id}>{candidate.display}</option>)}
                        </select>
                      </label>
                    ))}
                  </div>
                </section>
              );
            }) : <p style={{ color: "#92400e" }}>No spotlight candidates are available for this date range.</p>}
            <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: "0.75rem", marginTop: "1rem", alignItems: "end" }}>
              <label>Save confirmation<br /><input value={saveConfirm} onChange={(event) => setSaveConfirm(event.target.value)} placeholder="SAVE RECAP" style={inputStyle} /></label>
              <button type="button" onClick={saveDraft} disabled={busy} style={buttonStyle}>Save draft edits</button>
            </div>
          </article>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>3. Preview</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.75rem" }}>
              {Object.entries(recapNumbers).slice(0, 8).map(([key, value]) => <div key={key} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem" }}><strong>{key.replace(/_/g, " ")}</strong><br />{shortValue(value)}</div>)}
            </div>
            <h3>Spotlight preview</h3>
            {spotlightPreview.length ? <ul>{spotlightPreview.map((item, idx) => <li key={`${item.key}-${idx}`}><strong>{shortValue(item.label || item.key)}</strong>: {Array.isArray(item.players) ? item.players.join(", ") : "—"}<br /><span style={{ color: "#475569" }}>{shortValue(item.description)}</span></li>)}</ul> : <p style={{ color: "#64748b" }}>Save draft edits to refresh the final spotlight preview.</p>}
          </article>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>4. Publish control</h2>
            <p style={{ color: "#475569" }}>Publishing makes this recap visible on the public Weekly Recap page. Unpublishing returns it to draft.</p>
            <label>Publish / unpublish confirmation<br /><input value={publishConfirm} onChange={(event) => setPublishConfirm(event.target.value)} placeholder="PUBLISH RECAP or UNPUBLISH RECAP" style={inputStyle} /></label>
            <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap", marginTop: "0.75rem" }}>
              <button type="button" onClick={() => publishAction("publish")} disabled={busy} style={buttonStyle}>Publish recap</button>
              <button type="button" onClick={() => publishAction("unpublish")} disabled={busy} style={dangerButtonStyle}>Unpublish recap</button>
            </div>
          </article>
        </>
      ) : null}
    </div>
  );
}
