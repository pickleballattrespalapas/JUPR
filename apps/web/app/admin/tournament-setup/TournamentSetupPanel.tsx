"use client";

import { useState } from "react";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type StatusResponse = { enabled: boolean; status: string; tournament_count?: number | null; warnings?: string[]; confirmation_text?: Record<string, string> };
type TournamentRow = { id: string; name?: string; status?: string; start_date?: string | null; end_date?: string | null; registration_status?: string | null; registration_slug?: string | null; day_count?: number; event_option_count?: number; registration_count?: number };
type SetupTemplate = { key: string; label: string; description?: string; days: Array<Record<string, unknown>>; event_families: Array<Record<string, unknown>>; event_options: Array<Record<string, unknown>> };
type DetailResponse = { ok: boolean; tournament: Record<string, unknown>; settings: Record<string, unknown>; days: Array<Record<string, unknown>>; event_options: Array<Record<string, unknown>>; builder_draft?: Record<string, unknown> | null; publish_impact?: Record<string, unknown> | null; publish_impact_warning?: string | null; registration_count?: number; state_fingerprint: string; templates?: SetupTemplate[] };
type WriteResponse = { ok: boolean; mode?: string; settings?: Record<string, unknown>; builder_draft?: Record<string, unknown>; publish_result?: Record<string, unknown>; publish_impact?: Record<string, unknown>; days?: Array<Record<string, unknown>>; event_options?: Array<Record<string, unknown>>; warnings?: string[]; operation_key?: string; request_fingerprint?: string; idempotent_replay?: boolean; reconciled?: boolean };
type ImpactResponse = { ok: boolean; mode: string; dry_run: true; write_count: 0; state_fingerprint: string; impact_fingerprint: string; publish_impact: Record<string, unknown> };

type Props = { apiBase: string | null; clubId: string; status: StatusResponse | null };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string { return `${apiBase.replace(/\/$/, "")}${path}`; }
function safeString(value: unknown): string { return value == null ? "" : String(value); }
function pretty(value: unknown): string { return JSON.stringify(value ?? [], null, 2); }
function parseArrayJson(raw: string, label: string): Array<Record<string, unknown>> {
  const parsed = JSON.parse(raw || "[]") as unknown;
  if (!Array.isArray(parsed)) throw new Error(`${label} must be a JSON array.`);
  return parsed.map((row) => (row && typeof row === "object" ? row as Record<string, unknown> : {}));
}
function draftSignature(days: Array<Record<string, unknown>>, events: Array<Record<string, unknown>>): string { return JSON.stringify({ days, events }); }

export default function TournamentSetupPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [tournaments, setTournaments] = useState<TournamentRow[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [detail, setDetail] = useState<DetailResponse | null>(null);
  const [settings, setSettings] = useState<Record<string, unknown>>({});
  const [daysJson, setDaysJson] = useState("[]");
  const [eventsJson, setEventsJson] = useState("[]");
  const [familiesJson, setFamiliesJson] = useState("[]");
  const [settingsConfirm, setSettingsConfirm] = useState("");
  const [draftConfirm, setDraftConfirm] = useState("");
  const [publishConfirm, setPublishConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<WriteResponse | null>(null);
  const [impactReview, setImpactReview] = useState<ImpactResponse | null>(null);
  const [reviewedDraftSignature, setReviewedDraftSignature] = useState("");

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("Missing JUPR API base URL.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using Tournament Setup.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status}).`));
    return payload as T;
  }

  function hydrateFromDetail(payload: DetailResponse) {
    const draft = payload.builder_draft || {};
    const days = Array.isArray(draft.days) ? draft.days : payload.days || [];
    const events = Array.isArray(draft.event_options) ? draft.event_options : (Array.isArray(draft.divisions) ? draft.divisions : payload.event_options || []);
    const families = Array.isArray(draft.event_families) ? draft.event_families : [];
    setDetail(payload); setSettings(payload.settings || {}); setDaysJson(pretty(days)); setEventsJson(pretty(events)); setFamiliesJson(pretty(families)); setImpactReview(null); setReviewedDraftSignature("");
  }

  async function loadTournaments() {
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<{ ok: boolean; tournaments: TournamentRow[]; count: number }>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/setup/tournaments?include_archived=true`);
      setTournaments(payload.tournaments || []);
      if (!selectedId && payload.tournaments?.length) setSelectedId(payload.tournaments[0].id);
      setMessage(`Loaded ${payload.count || 0} tournament setup row(s).`);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to load tournaments."); }
    finally { setBusy(false); }
  }

  async function loadDetail(id = selectedId) {
    if (!id) { setMessage("Choose a tournament first."); return; }
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<DetailResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/setup/tournaments/${encodeURIComponent(id)}`);
      hydrateFromDetail(payload); setMessage("Loaded tournament setup detail.");
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to load tournament setup detail."); }
    finally { setBusy(false); }
  }

  async function saveSettings() {
    if (settingsConfirm.trim().toUpperCase() !== "SAVE SETUP") { setMessage("Type SAVE SETUP to save registration settings."); return; }
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<WriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/setup/tournaments/${encodeURIComponent(selectedId)}/settings`, { method: "PATCH", body: JSON.stringify({ ...settings, expected_state_fingerprint: detail?.state_fingerprint, confirmation_text: settingsConfirm }) });
      setLastResult(payload); setSettingsConfirm(""); await loadDetail(selectedId); setMessage(payload.idempotent_replay ? "Settings response reconciled from the durable operation." : "Tournament setup settings saved.");
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to save settings."); }
    finally { setBusy(false); }
  }

  async function saveDraft() {
    if (draftConfirm.trim().toUpperCase() !== "SAVE SETUP DRAFT") { setMessage("Type SAVE SETUP DRAFT to save the builder draft."); return; }
    setBusy(true); setMessage(null);
    try {
      const days = parseArrayJson(daysJson, "Days");
      const events = parseArrayJson(eventsJson, "Event options");
      const eventFamilies = parseArrayJson(familiesJson, "Event families");
      const payload = await requestJson<WriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/setup/tournaments/${encodeURIComponent(selectedId)}/draft`, { method: "PUT", body: JSON.stringify({ days, event_families: eventFamilies, event_options: events, expected_state_fingerprint: detail?.state_fingerprint, confirmation_text: draftConfirm }) });
      setLastResult(payload); setDraftConfirm(""); await loadDetail(selectedId); setMessage(payload.idempotent_replay ? "Draft response reconciled from the durable operation." : "Tournament setup draft saved.");
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to save draft."); }
    finally { setBusy(false); }
  }

  async function publishSetup() {
    if (publishConfirm.trim().toUpperCase() !== "PUBLISH SETUP") { setMessage("Type PUBLISH SETUP to publish registration days and event options."); return; }
    setBusy(true); setMessage(null);
    try {
      const days = parseArrayJson(daysJson, "Days");
      const events = parseArrayJson(eventsJson, "Event options");
      if (!impactReview || reviewedDraftSignature !== draftSignature(days, events)) throw new Error("Review publish impact for the current draft before publishing.");
      const payload = await requestJson<WriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/setup/tournaments/${encodeURIComponent(selectedId)}/publish`, { method: "POST", body: JSON.stringify({ days, event_options: events, expected_state_fingerprint: detail?.state_fingerprint, reviewed_impact_fingerprint: impactReview.impact_fingerprint, confirmation_text: publishConfirm }) });
      setLastResult(payload); setPublishConfirm(""); setMessage(payload.idempotent_replay ? "Publish response reconciled without republishing." : "Tournament setup published.");
      await loadDetail(selectedId);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to publish setup."); }
    finally { setBusy(false); }
  }

  function seedFromPublished() {
    if (!detail) return;
    setDaysJson(pretty(detail.days || [])); setEventsJson(pretty(detail.event_options || [])); setFamiliesJson("[]"); setImpactReview(null); setReviewedDraftSignature("");
    setMessage("Seeded draft from published registration configuration.");
  }

  function seedStandardEvents() {
    const template = detail?.templates?.find((row) => row.key === "standard_doubles_singles");
    if (!template) { setMessage("The Python setup template is unavailable. Reload before continuing."); return; }
    setDaysJson(pretty(template.days)); setEventsJson(pretty(template.event_options)); setFamiliesJson(pretty(template.event_families)); setImpactReview(null); setReviewedDraftSignature("");
    setMessage(`Applied Python template: ${template.label}.`);
  }

  async function reviewImpact() {
    if (!detail) return;
    setBusy(true); setMessage(null);
    try {
      const days = parseArrayJson(daysJson, "Days");
      const events = parseArrayJson(eventsJson, "Event options");
      const payload = await requestJson<ImpactResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/setup/tournaments/${encodeURIComponent(selectedId)}/impact`, { method: "POST", body: JSON.stringify({ days, event_options: events, expected_state_fingerprint: detail.state_fingerprint }) });
      setImpactReview(payload); setReviewedDraftSignature(draftSignature(days, events)); setMessage("Publish impact reviewed by FastAPI. No rows were written.");
    } catch (error) { setImpactReview(null); setReviewedDraftSignature(""); setMessage(error instanceof Error ? error.message : "Unable to review publish impact."); }
    finally { setBusy(false); }
  }

  if (!status?.enabled) return <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Tournament Setup is disabled</h2><p>{status?.warnings?.[0] || "Enable JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS on FastAPI."}</p></article>;
  const impact = (impactReview?.publish_impact || detail?.publish_impact) as Record<string, unknown> | null | undefined;
  const summary = (impact?.summary || {}) as Record<string, unknown>;
  return <div style={{ display: "grid", gap: "1rem" }}>
    <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Admin session</h2><p style={{ color: "#475569" }}>{adminSessionLabel(session)}</p>{sessionLoading ? <p>Checking session…</p> : null}{sessionMessage ? <p>{sessionMessage}</p> : null}</article>
    <article style={cardStyle}>
      <h2 style={{ marginTop: 0 }}>1. Select tournament</h2>
      <div style={{ display: "grid", gridTemplateColumns: "1fr auto auto", gap: "0.75rem", alignItems: "end" }}>
        <label>Tournament<br /><select value={selectedId} onChange={(event) => setSelectedId(event.target.value)} style={inputStyle}>{tournaments.map((row) => <option key={row.id} value={row.id}>{row.name || row.id} · {row.status || "status"} · {row.registration_status || "registration"}</option>)}</select></label>
        <button type="button" onClick={loadTournaments} disabled={busy || !accessToken} style={ghostButtonStyle}>Load list</button>
        <button type="button" onClick={() => loadDetail()} disabled={busy || !selectedId} style={buttonStyle}>Load setup</button>
      </div>
      {message ? <p role="status" style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("type") || message.toLowerCase().includes("blocked") || message.toLowerCase().includes("reload") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
    </article>

    {detail ? <>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>2. Registration settings</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
          <label>Slug<br /><input value={safeString(settings.registration_slug)} onChange={(e) => setSettings((s) => ({ ...s, registration_slug: e.target.value }))} style={inputStyle} /></label>
          <label>Status<br /><select value={safeString(settings.registration_status || "draft")} onChange={(e) => setSettings((s) => ({ ...s, registration_status: e.target.value }))} style={inputStyle}><option value="draft">draft</option><option value="open">open</option><option value="closed">closed</option></select></label>
          <label>Open at<br /><input value={safeString(settings.registration_open_at)} onChange={(e) => setSettings((s) => ({ ...s, registration_open_at: e.target.value }))} placeholder="ISO date/time" style={inputStyle} /></label>
          <label>Close at<br /><input value={safeString(settings.registration_close_at)} onChange={(e) => setSettings((s) => ({ ...s, registration_close_at: e.target.value }))} placeholder="ISO date/time" style={inputStyle} /></label>
          <label><input type="checkbox" checked={Boolean(settings.waitlist_enabled)} onChange={(e) => setSettings((s) => ({ ...s, waitlist_enabled: e.target.checked }))} /> Waitlist enabled</label>
          <label><input type="checkbox" checked={Boolean(settings.partner_board_enabled)} onChange={(e) => setSettings((s) => ({ ...s, partner_board_enabled: e.target.checked }))} /> Partner Board enabled</label>
        </div>
        <label>Rules markdown<br /><textarea value={safeString(settings.rules_markdown)} onChange={(e) => setSettings((s) => ({ ...s, rules_markdown: e.target.value }))} rows={4} style={inputStyle} /></label>
        <label>Refund / operations policy markdown<br /><textarea value={safeString(settings.refund_policy_markdown)} onChange={(e) => setSettings((s) => ({ ...s, refund_policy_markdown: e.target.value }))} rows={3} style={inputStyle} /></label>
        <label>Confirmation<br /><input value={settingsConfirm} onChange={(e) => setSettingsConfirm(e.target.value)} placeholder="SAVE SETUP" style={inputStyle} /></label>
        <p><button type="button" onClick={saveSettings} disabled={busy} style={buttonStyle}>Save settings</button></p>
      </article>

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>3. Builder draft</h2>
        <p style={{ color: "#475569" }}>Use the guided buttons to seed the setup, then edit days/divisions before saving or publishing. The publish step uses the guarded Python diff that preserves populated rows and blocks destructive changes.</p>
        <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}><button type="button" onClick={seedFromPublished} style={ghostButtonStyle}>Seed from published config</button><button type="button" onClick={seedStandardEvents} style={ghostButtonStyle}>Generate standard divisions</button></p>
        <label>Days JSON<br /><textarea value={daysJson} onChange={(e) => { setDaysJson(e.target.value); setImpactReview(null); setReviewedDraftSignature(""); }} rows={8} style={{ ...inputStyle, fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" }} /></label>
        <label>Event families JSON<br /><textarea value={familiesJson} onChange={(e) => { setFamiliesJson(e.target.value); setImpactReview(null); setReviewedDraftSignature(""); }} rows={5} style={{ ...inputStyle, fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" }} /></label>
        <label>Event options / divisions JSON<br /><textarea value={eventsJson} onChange={(e) => { setEventsJson(e.target.value); setImpactReview(null); setReviewedDraftSignature(""); }} rows={14} style={{ ...inputStyle, fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" }} /></label>
        <label>Draft confirmation<br /><input value={draftConfirm} onChange={(e) => setDraftConfirm(e.target.value)} placeholder="SAVE SETUP DRAFT" style={inputStyle} /></label>
        <p><button type="button" onClick={saveDraft} disabled={busy} style={buttonStyle}>Save draft</button></p>
      </article>

      <article style={{ ...cardStyle, background: "#fff7ed", borderColor: "#fed7aa" }}>
        <h2 style={{ marginTop: 0 }}>4. Publish setup</h2>
        <p><button type="button" onClick={reviewImpact} disabled={busy} style={ghostButtonStyle}>Review publish impact (dry run)</button></p>
        <p style={{ color: impactReview ? "#166534" : "#92400e" }}>{impactReview ? `Reviewed ${impactReview.impact_fingerprint.slice(0, 16)}…; ${impactReview.write_count} writes.` : "Current draft has not been reviewed by FastAPI."}</p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: "0.75rem" }}>
          {Object.entries(summary).map(([key, value]) => <article key={key} style={cardStyle}><strong>{key.replace(/_/g, " ")}</strong><br />{String(value)}</article>)}
        </div>
        {detail.publish_impact_warning ? <p style={{ color: "#b91c1c" }}>{detail.publish_impact_warning}</p> : null}
        {impact?.blocked && Array.isArray(impact.blocked) && impact.blocked.length ? <pre style={{ whiteSpace: "pre-wrap", color: "#b91c1c" }}>{JSON.stringify(impact.blocked, null, 2)}</pre> : null}
        {impact?.warnings && Array.isArray(impact.warnings) && impact.warnings.length ? <pre style={{ whiteSpace: "pre-wrap", color: "#92400e" }}>{JSON.stringify(impact.warnings, null, 2)}</pre> : null}
        <label>Publish confirmation<br /><input value={publishConfirm} onChange={(e) => setPublishConfirm(e.target.value)} placeholder="PUBLISH SETUP" style={inputStyle} /></label>
        <p><button type="button" onClick={publishSetup} disabled={busy || !impactReview || publishConfirm.trim().toUpperCase() !== "PUBLISH SETUP"} style={buttonStyle}>Publish setup</button></p>
      </article>
    </> : null}
    {lastResult ? <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Last result</h2><pre style={{ whiteSpace: "pre-wrap", background: "#0f172a", color: "white", padding: "1rem", borderRadius: "12px", overflowX: "auto" }}>{JSON.stringify(lastResult, null, 2)}</pre></article> : null}
  </div>;
}
