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
import { ConfirmAction } from "@/components/ConfirmAction";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";
import AdminWeeklyRecapPreview from "./AdminWeeklyRecapPreview";
import { clubTodayIso, clubWeekStartIso } from "@/lib/clubDate";

type Props = { apiBase: string | null; clubId: string; status: AdminWeeklyRecapStatusResponse; initialWeekStart?: string; printMode?: boolean };
type SpotlightEdit = { include: boolean; order: string; description: string; players: string[] };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
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

export default function WeeklyRecapAdminPanel({ apiBase, clubId, status, initialWeekStart = "", printMode = false }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [weekStart, setWeekStart] = useState(() => clubWeekStartIso());
  const [weekEnd, setWeekEnd] = useState(() => clubTodayIso());
  const [recaps, setRecaps] = useState<AdminWeeklyRecapRow[]>([]);
  const [selectedWeekStart, setSelectedWeekStart] = useState(initialWeekStart);
  const [selectedRecap, setSelectedRecap] = useState<AdminWeeklyRecapRow | null>(null);
  const [candidates, setCandidates] = useState<Record<string, AdminWeeklyRecapCandidate[]>>({});
  const [lookingAhead, setLookingAhead] = useState<string[]>(["", "", ""]);
  const [spotlightEdits, setSpotlightEdits] = useState<Record<string, SpotlightEdit>>({});
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [messageSeverity, setMessageSeverity] = useState<"success" | "error" | null>(null);
  const listRequest = useLatestRequestGuard(accessToken, clearProtectedRecapState);
  const recapRequest = useLatestRequestGuard(accessToken);
  const writeRequest = useLatestRequestGuard(accessToken);

  const candidateKeys = useMemo(() => normalizeCandidateKeys(candidates), [candidates]);
  const mutationControlsDisabled = busy || !status.mutations_enabled;
  const recapNumbers = numbersFromRecap(selectedRecap);
  const spotlightPreview = finalSpotlight(selectedRecap);
  const targetExistingRecap = recaps.find((recap) => recap.week_start === weekStart) || (selectedRecap?.week_start === weekStart ? selectedRecap : null);

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using Weekly Recap Admin.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = String(payload?.detail || `API error (${response.status})`);
      if (response.status === 409) throw new Error(`${detail} Reload the recap before trying again.`);
      throw new Error(detail);
    }
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

  function clearProtectedRecapState() {
    recapRequest.invalidate();
    writeRequest.invalidate();
    setBusy(false); setMessage(null); setMessageSeverity(null);
    setRecaps([]); setSelectedWeekStart(""); setSelectedRecap(null); setCandidates({}); setSpotlightEdits({});
  }

  async function loadRecaps() {
    const generation = listRequest.begin();
    setBusy(true);
    setMessage(null);
    setMessageSeverity(null);
    try {
      const payload = await requestJson<AdminWeeklyRecapListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/weekly-recap/recaps?limit=100`);
      if (!listRequest.isCurrent(generation)) return;
      const nextRecaps = payload.recaps || [];
      setRecaps(nextRecaps);
      if (selectedWeekStart && !nextRecaps.some((recap) => recap.week_start === selectedWeekStart)) {
        recapRequest.invalidate();
        setSelectedWeekStart("");
        setSelectedRecap(null);
        setCandidates({});
      }
      setMessage(nextRecaps.length ? `Loaded ${payload.count ?? nextRecaps.length} recap(s).` : "No saved recaps are available.");
      setMessageSeverity("success");
    } catch (error) {
      if (listRequest.isCurrent(generation)) {
        setMessage(error instanceof Error ? error.message : "Unable to load weekly recaps.");
        setMessageSeverity("error");
      }
    } finally {
      if (listRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function loadSelectedRecap(explicitWeekStart?: string) {
    const generation = recapRequest.begin();
    const target = explicitWeekStart || selectedWeekStart || weekStart;
    if (!target) {
      setMessage("Select a recap first.");
      setMessageSeverity("error");
      return;
    }
    setBusy(true);
    setMessage(null);
    setMessageSeverity(null);
    setSelectedRecap(null);
    setCandidates({});
    try {
      const payload = await requestJson<AdminWeeklyRecapDetailResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/weekly-recap/recaps/${encodeURIComponent(target)}?include_candidates=true`);
      if (!recapRequest.isCurrent(generation)) return;
      applyDetail(payload);
      setMessage("Weekly recap loaded.");
      setMessageSeverity("success");
    } catch (error) {
      if (recapRequest.isCurrent(generation)) {
        setMessage(error instanceof Error ? error.message : "Unable to load weekly recap.");
        setMessageSeverity("error");
      }
    } finally {
      if (recapRequest.isCurrent(generation)) setBusy(false);
    }
  }

  function selectRecap(target: string) {
    setSelectedWeekStart(target);
    setSelectedRecap(null);
    setCandidates({});
    if (target) void loadSelectedRecap(target);
    else recapRequest.invalidate();
  }

  async function loadInitialRecapWorkspace() {
    await loadRecaps();
    if (initialWeekStart) await loadSelectedRecap(initialWeekStart);
  }

  async function generateDraft(confirmationText: string) {
    const generation = writeRequest.begin();
    setBusy(true);
    setMessage(null);
    setMessageSeverity(null);
    try {
      const payload = await requestJson<AdminWeeklyRecapWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/weekly-recap/generate`, {
        method: "POST",
        body: JSON.stringify({ week_start: weekStart, week_end: weekEnd, confirmation_text: confirmationText, expected_row_version: selectedRecap?.week_start === weekStart ? selectedRecap.row_version : null, source: "next_weekly_recap_generate" })
      });
      if (!writeRequest.isCurrent(generation)) return;
      applyDetail(payload);
      await loadRecaps();
      if (!writeRequest.isCurrent(generation)) return;
      setMessage("Draft generated from current match, social, and tournament data.");
      setMessageSeverity("success");
    } catch (error) {
      if (writeRequest.isCurrent(generation)) {
        setMessage(error instanceof Error ? error.message : "Unable to generate weekly recap draft.");
        setMessageSeverity("error");
      }
    } finally {
      if (writeRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function saveDraft(confirmationText: string) {
    if (!selectedRecap) {
      setMessage("Load or generate a recap before saving edits.");
      setMessageSeverity("error");
      return;
    }
    const generation = writeRequest.begin();
    setBusy(true);
    setMessage(null);
    setMessageSeverity(null);
    try {
      const payload = await requestJson<AdminWeeklyRecapWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/weekly-recap/recaps/${encodeURIComponent(selectedRecap.week_start)}`, {
        method: "PATCH",
        body: JSON.stringify({ edits_json: buildEditsPayload(lookingAhead, spotlightEdits), confirmation_text: confirmationText, expected_row_version: selectedRecap.row_version, source: "next_weekly_recap_save" })
      });
      if (!writeRequest.isCurrent(generation)) return;
      applyDetail(payload);
      setMessage("Draft edits saved.");
      setMessageSeverity("success");
    } catch (error) {
      if (writeRequest.isCurrent(generation)) {
        setMessage(error instanceof Error ? error.message : "Unable to save weekly recap draft.");
        setMessageSeverity("error");
      }
    } finally {
      if (writeRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function publishAction(action: "publish" | "unpublish", confirmationText: string) {
    if (!selectedRecap) {
      setMessage("Load or generate a recap before publishing.");
      setMessageSeverity("error");
      return;
    }
    const generation = writeRequest.begin();
    setBusy(true);
    setMessage(null);
    setMessageSeverity(null);
    try {
      const payload = await requestJson<AdminWeeklyRecapWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/weekly-recap/recaps/${encodeURIComponent(selectedRecap.week_start)}/publish`, {
        method: "POST",
        body: JSON.stringify({ action, edits_json: buildEditsPayload(lookingAhead, spotlightEdits), confirmation_text: confirmationText, expected_row_version: selectedRecap.row_version, source: action === "publish" ? "next_weekly_recap_publish" : "next_weekly_recap_unpublish" })
      });
      if (!writeRequest.isCurrent(generation)) return;
      applyDetail(payload);
      await loadRecaps();
      if (!writeRequest.isCurrent(generation)) return;
      setMessage(action === "publish" ? "Weekly recap published." : "Weekly recap unpublished and returned to draft.");
      setMessageSeverity("success");
    } catch (error) {
      if (writeRequest.isCurrent(generation)) {
        setMessage(error instanceof Error ? error.message : `Unable to ${action} weekly recap.`);
        setMessageSeverity("error");
      }
    } finally {
      if (writeRequest.isCurrent(generation)) setBusy(false);
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

  useAuthenticatedAutoLoad(status.enabled ? accessToken : "", loadInitialRecapWorkspace, initialWeekStart);

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
      {!status.mutations_enabled ? <p role="status" style={{ color: "#92400e" }}><strong>Read-only:</strong> saved recaps and unpublished previews remain available, while generate, save, publish, and unpublish actions stay disabled until the isolated communications write wave is active.</p> : null}
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
          <ConfirmAction
            triggerLabel={targetExistingRecap ? "Regenerate draft" : "Generate draft"}
            title={targetExistingRecap ? "Regenerate this weekly recap draft?" : "Generate this weekly recap draft?"}
            description={targetExistingRecap ? `This rebuilds the recap for ${weekStart} through ${weekEnd} from current staging data and discards its saved looking-ahead and spotlight edits.` : `This generates a new recap for ${weekStart} through ${weekEnd} from current staging data.`}
            confirmLabel={targetExistingRecap ? "Yes, regenerate draft" : "Yes, generate draft"}
            confirmationText="GENERATE RECAP"
            tone={targetExistingRecap ? "danger" : "default"}
            disabled={mutationControlsDisabled || !accessToken || targetExistingRecap?.status === "published"}
            busy={busy}
            onConfirm={generateDraft}
          />
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "minmax(240px, 1fr) auto auto", gap: "0.75rem", marginTop: "0.75rem", alignItems: "end" }}>
          <label>Existing recaps<br />
            <select value={selectedWeekStart} onChange={(event) => selectRecap(event.target.value)} disabled={busy} style={inputStyle}>
              <option value="">Select recap…</option>
              {recaps.map((row) => <option key={row.week_start} value={row.week_start}>{row.week_start} → {row.week_end} · {row.status}</option>)}
            </select>
          </label>
          <button type="button" onClick={loadRecaps} disabled={busy || !accessToken} style={ghostButtonStyle}>{busy ? "Refreshing…" : "Refresh recaps"}</button>
          <button type="button" onClick={() => loadSelectedRecap()} disabled={busy || !selectedWeekStart} style={ghostButtonStyle}>Retry selected recap</button>
        </div>
        {message ? <p role={messageSeverity === "error" ? "alert" : "status"} style={{ color: messageSeverity === "error" ? "#b91c1c" : "#166534" }}>{message}</p> : null}
      </article>

      {selectedRecap ? (
        <>
          <article style={cardStyle} className="admin-recap-no-print">
            <h2 style={{ marginTop: 0 }}>2. Edit draft</h2>
            <p style={{ color: "#475569" }}><strong>Recap:</strong> {selectedRecap.week_start} → {selectedRecap.week_end} · <strong>Status:</strong> {selectedRecap.status}</p>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
              {[0, 1, 2].map((idx) => (
                <label key={idx}>Looking ahead #{idx + 1}<br />
                  <input value={lookingAhead[idx] || ""} onChange={(event) => setLookingAhead((current) => current.map((item, itemIdx) => itemIdx === idx ? event.target.value : item))} disabled={mutationControlsDisabled || selectedRecap.status === "published"} style={inputStyle} />
                </label>
              ))}
            </div>
          </article>

          <article style={cardStyle} className="admin-recap-no-print">
            <h2 style={{ marginTop: 0 }}>Spotlight reel</h2>
            <p style={{ color: "#475569" }}>Choose up to three candidates per spotlight category. Leave a slot blank to omit it.</p>
            {candidateKeys.length ? candidateKeys.map((key, idx) => {
              const edit = spotlightEdits[key] || { include: true, order: String(idx + 1), description: "", players: [] };
              const options = candidates[key] || [];
              const label = options[0]?.label || key;
              return (
                <section key={key} style={{ borderTop: idx ? "1px solid #e2e8f0" : undefined, paddingTop: idx ? "0.75rem" : 0, marginTop: idx ? "0.75rem" : 0 }}>
                  <div style={{ display: "grid", gridTemplateColumns: "120px 90px 1fr", gap: "0.75rem", alignItems: "end" }}>
                    <label><input type="checkbox" checked={edit.include} onChange={(event) => updateSpotlight(key, { include: event.target.checked })} disabled={mutationControlsDisabled || selectedRecap.status === "published"} /> Include<br /><strong>{label}</strong></label>
                    <label>Order<br /><input value={edit.order} onChange={(event) => updateSpotlight(key, { order: event.target.value })} disabled={mutationControlsDisabled || selectedRecap.status === "published"} style={inputStyle} /></label>
                    <label>Description<br /><input value={edit.description} onChange={(event) => updateSpotlight(key, { description: event.target.value })} disabled={mutationControlsDisabled || selectedRecap.status === "published"} style={inputStyle} /></label>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem", marginTop: "0.75rem" }}>
                    {[0, 1, 2].map((slot) => (
                      <label key={slot}>Candidate {slot + 1}<br />
                        <select value={edit.players?.[slot] || ""} onChange={(event) => updateSpotlightPlayer(key, slot, event.target.value)} disabled={mutationControlsDisabled || selectedRecap.status === "published"} style={inputStyle}>
                          <option value="">None</option>
                          {options.map((candidate) => <option key={candidate.candidate_id} value={candidate.candidate_id}>{candidate.display}</option>)}
                        </select>
                      </label>
                    ))}
                  </div>
                </section>
              );
            }) : <p style={{ color: "#92400e" }}>No spotlight candidates are available for this date range.</p>}
            <div style={{ marginTop: "1rem" }}>
              <ConfirmAction
                triggerLabel="Save draft edits"
                title="Save these weekly recap edits?"
                description="This saves the current looking-ahead and spotlight edits to the selected draft."
                confirmLabel="Yes, save edits"
                confirmationText="SAVE RECAP"
                disabled={mutationControlsDisabled || selectedRecap.status === "published"}
                busy={busy}
                onConfirm={saveDraft}
              />
            </div>
          </article>

          <article style={cardStyle} className="admin-recap-no-print">
            <h2 style={{ marginTop: 0 }}>3. Preview summary</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.75rem" }}>
              {Object.entries(recapNumbers).slice(0, 8).map(([key, value]) => <div key={key} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem" }}><strong>{key.replace(/_/g, " ")}</strong><br />{shortValue(value)}</div>)}
            </div>
            <h3>Spotlight preview</h3>
            {spotlightPreview.length ? <ul>{spotlightPreview.map((item, idx) => <li key={`${item.key}-${idx}`}><strong>{shortValue(item.label || item.key)}</strong>: {Array.isArray(item.players) ? item.players.join(", ") : "—"}<br /><span style={{ color: "#475569" }}>{shortValue(item.description)}</span></li>)}</ul> : <p style={{ color: "#64748b" }}>Save draft edits to refresh the final spotlight preview.</p>}
          </article>

          <article style={cardStyle} className="admin-recap-no-print">
            <h2 style={{ marginTop: 0 }}>4. Publish control</h2>
            <p style={{ color: "#475569" }}>Publishing makes this recap visible on the public Weekly Recap page. Unpublishing returns it to draft.</p>
            <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap", marginTop: "0.75rem" }}>
              <ConfirmAction triggerLabel="Publish recap" title="Publish this weekly recap?" description="This saves the current looking-ahead and spotlight edits into the final recap, then makes that result visible on the public Weekly Recap page." confirmLabel="Yes, publish recap" confirmationText="PUBLISH RECAP" disabled={mutationControlsDisabled || selectedRecap.status === "published"} busy={busy} onConfirm={(confirmationText) => publishAction("publish", confirmationText)} />
              <ConfirmAction triggerLabel="Unpublish recap" title="Unpublish this weekly recap?" description="This removes the recap from the public page and returns it to draft status." confirmLabel="Yes, unpublish recap" confirmationText="UNPUBLISH RECAP" tone="danger" disabled={mutationControlsDisabled || selectedRecap.status !== "published"} busy={busy} onConfirm={(confirmationText) => publishAction("unpublish", confirmationText)} />
            </div>
          </article>

          <article style={cardStyle}>
            <h2 className="admin-recap-no-print" style={{ marginTop: 0 }}>5. Full unpublished preview and print proof</h2>
            <p className="admin-recap-no-print" style={{ color: "#475569" }}>This renders the complete saved <code>final_json</code> even while status is draft. It never exposes an unpublished recap through the public API.</p>
            <AdminWeeklyRecapPreview recap={selectedRecap} printMode={printMode} />
          </article>
        </>
      ) : null}
    </div>
  );
}
