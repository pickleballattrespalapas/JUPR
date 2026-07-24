"use client";

import { useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";
import { TournamentSetupBuilder } from "./TournamentSetupBuilder";
import {
  configurationPayload,
  draftSignature,
  publishConfigurationPayload,
  validateSetupConfiguration,
  wrapBuilderRows,
  type SetupConfiguration
} from "./tournamentSetupBuilder";

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
function readableImpactItem(value: unknown): string {
  if (value == null) return "No detail supplied.";
  if (typeof value !== "object") return String(value);
  const record = value as Record<string, unknown>;
  for (const key of ["message", "detail", "reason", "warning", "name"]) {
    if (record[key] != null && String(record[key]).trim()) return String(record[key]);
  }
  return "See advanced impact diagnostics for the complete API detail.";
}

const emptyConfiguration: SetupConfiguration = { days: [], eventFamilies: [], eventOptions: [] };

export default function TournamentSetupPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [tournaments, setTournaments] = useState<TournamentRow[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [loadedDetailId, setLoadedDetailId] = useState("");
  const [detailLoadingId, setDetailLoadingId] = useState("");
  const [detail, setDetail] = useState<DetailResponse | null>(null);
  const [settings, setSettings] = useState<Record<string, unknown>>({});
  const [configuration, setConfiguration] = useState<SetupConfiguration>(emptyConfiguration);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<WriteResponse | null>(null);
  const [impactReview, setImpactReview] = useState<ImpactResponse | null>(null);
  const [reviewedDraftSignature, setReviewedDraftSignature] = useState("");

  function clearDetailState() {
    setLoadedDetailId("");
    setDetailLoadingId("");
    setDetail(null);
    setSettings({});
    setConfiguration(emptyConfiguration);
    setLastResult(null);
    setImpactReview(null);
    setReviewedDraftSignature("");
  }

  function resetWorkspace() {
    setTournaments([]);
    setSelectedId("");
    clearDetailState();
    setBusy(false);
    setMessage(null);
  }

  const listRequest = useLatestRequestGuard(accessToken, resetWorkspace);
  const detailRequest = useLatestRequestGuard(accessToken);
  const operationRequest = useLatestRequestGuard(accessToken);

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
    setDetail(payload);
    setSettings(payload.settings || {});
    setConfiguration({
      days: wrapBuilderRows(days, "day"),
      eventFamilies: wrapBuilderRows(families, "family"),
      eventOptions: wrapBuilderRows(events, "division")
    });
    setImpactReview(null);
    setReviewedDraftSignature("");
  }

  async function loadTournaments() {
    const generation = listRequest.begin();
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<{ ok: boolean; tournaments: TournamentRow[]; count: number }>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/setup/tournaments?include_archived=true`);
      if (!listRequest.isCurrent(generation)) return;
      const rows = payload.tournaments || [];
      const nextId = rows.some((row) => row.id === selectedId) ? selectedId : rows[0]?.id || "";
      setTournaments(rows);
      setSelectedId(nextId);
      if (nextId) {
        const preserveCurrentEdits = nextId === selectedId && nextId === loadedDetailId && Boolean(detail);
        if (preserveCurrentEdits) {
          setMessage("Tournament list refreshed. Unsaved setup edits were preserved.");
        } else {
          if (nextId !== loadedDetailId) clearDetailState();
          await loadDetail(nextId);
        }
      } else {
        detailRequest.invalidate();
        clearDetailState();
        setMessage("No tournaments are available for setup.");
      }
    } catch (error) {
      if (listRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to load tournaments.");
    } finally {
      if (listRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function loadDetail(id = selectedId): Promise<boolean> {
    if (!id) {
      detailRequest.invalidate();
      clearDetailState();
      setMessage("Choose a tournament first.");
      return false;
    }
    const generation = detailRequest.begin();
    clearDetailState();
    setBusy(true); setDetailLoadingId(id);
    setMessage("Loading the selected tournament setup…");
    try {
      const payload = await requestJson<DetailResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/setup/tournaments/${encodeURIComponent(id)}`);
      if (!detailRequest.isCurrent(generation)) return false;
      hydrateFromDetail(payload); setLoadedDetailId(id); setMessage("Loaded the selected tournament setup."); return true;
    } catch (error) {
      if (detailRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to load tournament setup detail.");
      return false;
    } finally {
      if (detailRequest.isCurrent(generation)) {
        setDetailLoadingId("");
        setBusy(false);
      }
    }
  }

  function selectTournament(id: string) {
    detailRequest.invalidate();
    operationRequest.invalidate();
    clearDetailState();
    setSelectedId(id);
    if (id) void loadDetail(id);
  }

  useAuthenticatedAutoLoad(status?.enabled ? accessToken : "", loadTournaments);

  async function saveSettings(confirmationText: string) {
    if (!detail || loadedDetailId !== selectedId) { setMessage("Reload the selected tournament before saving settings."); return; }
    const generation = operationRequest.begin();
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<WriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/setup/tournaments/${encodeURIComponent(selectedId)}/settings`, { method: "PATCH", body: JSON.stringify({ ...settings, expected_state_fingerprint: detail?.state_fingerprint, confirmation_text: confirmationText }) });
      if (!operationRequest.isCurrent(generation)) return;
      const reloaded = await loadDetail(selectedId);
      if (operationRequest.isCurrent(generation)) {
        setLastResult(payload);
        if (reloaded) setMessage(payload.idempotent_replay ? "Settings response reconciled from the durable operation." : "Tournament setup settings saved.");
      }
    } catch (error) {
      if (operationRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to save settings.");
    } finally {
      if (operationRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function saveDraft(confirmationText: string) {
    if (!detail || loadedDetailId !== selectedId) { setMessage("Reload the selected tournament before saving its draft."); return; }
    const generation = operationRequest.begin();
    setBusy(true); setMessage(null);
    try {
      const draft = configurationPayload(configuration);
      const payload = await requestJson<WriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/setup/tournaments/${encodeURIComponent(selectedId)}/draft`, { method: "PUT", body: JSON.stringify({ ...draft, expected_state_fingerprint: detail?.state_fingerprint, confirmation_text: confirmationText }) });
      if (!operationRequest.isCurrent(generation)) return;
      const reloaded = await loadDetail(selectedId);
      if (operationRequest.isCurrent(generation)) {
        setLastResult(payload);
        if (reloaded) setMessage(payload.idempotent_replay ? "Draft response reconciled from the durable operation." : "Tournament setup draft saved.");
      }
    } catch (error) {
      if (operationRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to save draft.");
    } finally {
      if (operationRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function publishSetup(confirmationText: string) {
    if (!detail || loadedDetailId !== selectedId) { setMessage("Reload the selected tournament before publishing its setup."); return; }
    const generation = operationRequest.begin();
    setBusy(true); setMessage(null);
    try {
      const draft = publishConfigurationPayload(configuration);
      if (!impactReview || reviewedDraftSignature !== draftSignature(configuration)) throw new Error("Review publish impact for the current draft before publishing.");
      const payload = await requestJson<WriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/setup/tournaments/${encodeURIComponent(selectedId)}/publish`, { method: "POST", body: JSON.stringify({ days: draft.days, event_options: draft.event_options, expected_state_fingerprint: detail?.state_fingerprint, reviewed_impact_fingerprint: impactReview.impact_fingerprint, confirmation_text: confirmationText }) });
      if (!operationRequest.isCurrent(generation)) return;
      const reloaded = await loadDetail(selectedId);
      if (operationRequest.isCurrent(generation)) {
        setLastResult(payload);
        if (reloaded) setMessage(payload.idempotent_replay ? "Publish response reconciled without republishing." : "Tournament setup published.");
      }
    } catch (error) {
      if (operationRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to publish setup.");
    } finally {
      if (operationRequest.isCurrent(generation)) setBusy(false);
    }
  }

  function seedFromPublished() {
    if (!detail) return;
    setConfiguration({
      days: wrapBuilderRows(detail.days || [], "day"),
      eventFamilies: [],
      eventOptions: wrapBuilderRows(detail.event_options || [], "division")
    });
    setImpactReview(null); setReviewedDraftSignature("");
    setMessage("Seeded draft from published registration configuration.");
  }

  function seedStandardEvents() {
    const template = detail?.templates?.find((row) => row.key === "standard_doubles_singles");
    if (!template) { setMessage("The Python setup template is unavailable. Reload before continuing."); return; }
    setConfiguration({
      days: wrapBuilderRows(template.days, "day"),
      eventFamilies: wrapBuilderRows(template.event_families, "family"),
      eventOptions: wrapBuilderRows(template.event_options, "division")
    });
    setImpactReview(null); setReviewedDraftSignature("");
    setMessage(`Applied Python template: ${template.label}.`);
  }

  async function reviewImpact() {
    if (!detail || loadedDetailId !== selectedId) { setMessage("Reload the selected tournament before reviewing publish impact."); return; }
    const generation = operationRequest.begin();
    setBusy(true); setMessage(null);
    try {
      const draft = publishConfigurationPayload(configuration);
      const payload = await requestJson<ImpactResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/setup/tournaments/${encodeURIComponent(selectedId)}/impact`, { method: "POST", body: JSON.stringify({ days: draft.days, event_options: draft.event_options, expected_state_fingerprint: detail.state_fingerprint }) });
      if (!operationRequest.isCurrent(generation)) return;
      setImpactReview(payload); setReviewedDraftSignature(draftSignature(configuration)); setMessage("Publish impact reviewed by FastAPI. No rows were written.");
    } catch (error) {
      if (operationRequest.isCurrent(generation)) {
        setImpactReview(null);
        setReviewedDraftSignature("");
        setMessage(error instanceof Error ? error.message : "Unable to review publish impact.");
      }
    } finally {
      if (operationRequest.isCurrent(generation)) setBusy(false);
    }
  }

  if (!status?.enabled) return <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Tournament Setup is disabled</h2><p>{status?.warnings?.[0] || "Enable JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS on FastAPI."}</p></article>;
  const impact = (impactReview?.publish_impact || detail?.publish_impact) as Record<string, unknown> | null | undefined;
  const summary = (impact?.summary || {}) as Record<string, unknown>;
  const settingsConfirmation = status.confirmation_text?.settings || "SAVE SETUP";
  const draftConfirmation = status.confirmation_text?.draft || "SAVE SETUP DRAFT";
  const publishConfirmation = status.confirmation_text?.publish || "PUBLISH SETUP";
  const detailIsCurrent = Boolean(detail && loadedDetailId === selectedId);
  const loadedTournament = tournaments.find((row) => row.id === loadedDetailId);
  const builderIssues = validateSetupConfiguration(configuration);
  const builderReady = detailIsCurrent && builderIssues.length === 0;
  const blockedImpactItems = Array.isArray(impact?.blocked) ? impact.blocked : [];
  const impactWarnings = Array.isArray(impact?.warnings) ? impact.warnings : [];
  return <div style={{ display: "grid", gap: "1rem" }}>
    <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Admin session</h2><p style={{ color: "#475569" }}>{adminSessionLabel(session)}</p>{sessionLoading ? <p>Checking session…</p> : null}{sessionMessage ? <p>{sessionMessage}</p> : null}</article>
    <article style={cardStyle} aria-busy={Boolean(detailLoadingId)}>
      <h2 style={{ marginTop: 0 }}>1. Select tournament</h2>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
        <label>Tournament<br /><select value={selectedId} onChange={(event) => selectTournament(event.target.value)} disabled={busy || !accessToken} aria-busy={busy && !tournaments.length} style={inputStyle}><option value="" disabled>{busy && !tournaments.length ? "Loading tournaments…" : "Choose a tournament"}</option>{tournaments.map((row) => <option key={row.id} value={row.id}>{row.name || row.id} · {row.status || "status"} · {row.registration_status || "registration"}</option>)}</select></label>
        <button type="button" onClick={loadTournaments} disabled={busy || !accessToken} style={ghostButtonStyle}>Refresh list</button>
        <button type="button" onClick={() => loadDetail()} disabled={busy || !selectedId} style={buttonStyle}>{detailLoadingId ? "Loading setup…" : "Reload setup"}</button>
      </div>
      {detailIsCurrent ? <p style={{ color: "#475569" }}>{`Loaded setup: ${loadedTournament?.name || loadedDetailId}`}</p> : null}
      {message ? <p role="status" style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("type") || message.toLowerCase().includes("blocked") || message.toLowerCase().includes("reload") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
    </article>

    {detail && detailIsCurrent ? <>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>2. Registration settings</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
          <label>Slug<br /><input value={safeString(settings.registration_slug)} onChange={(e) => setSettings((s) => ({ ...s, registration_slug: e.target.value }))} disabled={!detailIsCurrent || busy} style={inputStyle} /></label>
          <label>Status<br /><select value={safeString(settings.registration_status || "draft")} onChange={(e) => setSettings((s) => ({ ...s, registration_status: e.target.value }))} disabled={!detailIsCurrent || busy} style={inputStyle}><option value="draft">draft</option><option value="open">open</option><option value="closed">closed</option></select></label>
          <label>Open at<br /><input value={safeString(settings.registration_open_at)} onChange={(e) => setSettings((s) => ({ ...s, registration_open_at: e.target.value }))} disabled={!detailIsCurrent || busy} placeholder="ISO date/time" style={inputStyle} /></label>
          <label>Close at<br /><input value={safeString(settings.registration_close_at)} onChange={(e) => setSettings((s) => ({ ...s, registration_close_at: e.target.value }))} disabled={!detailIsCurrent || busy} placeholder="ISO date/time" style={inputStyle} /></label>
          <label><input type="checkbox" checked={Boolean(settings.waitlist_enabled)} onChange={(e) => setSettings((s) => ({ ...s, waitlist_enabled: e.target.checked }))} disabled={!detailIsCurrent || busy} /> Waitlist enabled</label>
          <label><input type="checkbox" checked={Boolean(settings.partner_board_enabled)} onChange={(e) => setSettings((s) => ({ ...s, partner_board_enabled: e.target.checked }))} disabled={!detailIsCurrent || busy} /> Partner Board enabled</label>
        </div>
        <label>Rules markdown<br /><textarea value={safeString(settings.rules_markdown)} onChange={(e) => setSettings((s) => ({ ...s, rules_markdown: e.target.value }))} disabled={!detailIsCurrent || busy} rows={4} style={inputStyle} /></label>
        <label>Refund / operations policy markdown<br /><textarea value={safeString(settings.refund_policy_markdown)} onChange={(e) => setSettings((s) => ({ ...s, refund_policy_markdown: e.target.value }))} disabled={!detailIsCurrent || busy} rows={3} style={inputStyle} /></label>
        <p><ConfirmAction triggerLabel="Save settings" title="Save tournament registration settings?" description="This updates the selected tournament's registration status, dates, rules, and public registration options." confirmLabel="Yes, save settings" confirmationText={settingsConfirmation} disabled={!detailIsCurrent} busy={busy} onConfirm={saveSettings} /></p>
      </article>

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>3. Builder draft</h2>
        <p style={{ color: "#475569" }}>Build the schedule with guided day, event, and division controls. The publish step uses the guarded Python diff that preserves populated rows and blocks destructive changes.</p>
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1rem" }}>
          <ConfirmAction
            triggerLabel="Seed from published config"
            title="Replace this local draft with the published setup?"
            description="Unsaved builder edits will be replaced. This does not write to the server."
            confirmLabel="Yes, replace local draft"
            confirmationText=""
            disabled={!detailIsCurrent}
            busy={busy}
            onConfirm={seedFromPublished}
          />
          <ConfirmAction
            triggerLabel="Generate standard divisions"
            title="Replace this local draft with standard divisions?"
            description="Unsaved builder edits will be replaced with the standard doubles and singles template. This does not write to the server."
            confirmLabel="Yes, use standard divisions"
            confirmationText=""
            disabled={!detailIsCurrent}
            busy={busy}
            onConfirm={seedStandardEvents}
          />
        </div>
        <TournamentSetupBuilder
          configuration={configuration}
          issues={builderIssues}
          disabled={!detailIsCurrent || busy}
          onChange={(nextConfiguration) => {
            setConfiguration(nextConfiguration);
            setImpactReview(null);
            setReviewedDraftSignature("");
          }}
          onNotice={setMessage}
        />
        <p><ConfirmAction triggerLabel="Save draft" title="Save this tournament setup draft?" description="This stores the reviewed days, event defaults, and divisions as the builder draft without publishing them." confirmLabel="Yes, save draft" confirmationText={draftConfirmation} disabled={!builderReady} busy={busy} onConfirm={saveDraft} /></p>
        {!builderReady && detailIsCurrent ? <p style={{ color: "#92400e" }}>Resolve the builder validation messages before saving.</p> : null}
      </article>

      <article style={{ ...cardStyle, background: "#fff7ed", borderColor: "#fed7aa" }}>
        <h2 style={{ marginTop: 0 }}>4. Publish setup</h2>
        <p><button type="button" onClick={reviewImpact} disabled={busy || !builderReady} style={ghostButtonStyle}>Review publish impact (dry run)</button></p>
        <p style={{ color: impactReview ? "#166534" : "#92400e" }}>{impactReview ? `Impact review complete; ${impactReview.write_count} writes were performed.` : "Current draft has not been reviewed by FastAPI."}</p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: "0.75rem" }}>
          {Object.entries(summary).map(([key, value]) => <article key={key} style={cardStyle}><strong>{key.replace(/_/g, " ")}</strong><br />{String(value)}</article>)}
        </div>
        {detail.publish_impact_warning ? <p style={{ color: "#b91c1c" }}>{detail.publish_impact_warning}</p> : null}
        {blockedImpactItems.length ? <div role="alert" style={{ color: "#b91c1c" }}><strong>Blocked changes</strong><ul>{blockedImpactItems.map((item, index) => <li key={index}>{readableImpactItem(item)}</li>)}</ul></div> : null}
        {impactWarnings.length ? <div style={{ color: "#92400e" }}><strong>Review notes</strong><ul>{impactWarnings.map((item, index) => <li key={index}>{readableImpactItem(item)}</li>)}</ul></div> : null}
        {impactReview || blockedImpactItems.length || impactWarnings.length ? <details><summary style={{ cursor: "pointer", fontWeight: 800 }}>Advanced impact diagnostics</summary><pre style={{ whiteSpace: "pre-wrap", overflowX: "auto" }}>{JSON.stringify({ impact_fingerprint: impactReview?.impact_fingerprint, blocked: blockedImpactItems, warnings: impactWarnings }, null, 2)}</pre></details> : null}
        <p><ConfirmAction triggerLabel="Publish setup" title="Publish this tournament setup?" description="This applies the reviewed registration days and event options to the live tournament setup. Review the impact summary before continuing." confirmLabel="Yes, publish setup" confirmationText={publishConfirmation} tone="danger" disabled={!impactReview || !builderReady} busy={busy} onConfirm={publishSetup} /></p>
      </article>
    </> : null}
    {lastResult ? <details style={cardStyle}><summary style={{ cursor: "pointer", fontWeight: 800 }}>Advanced: last API result</summary><pre style={{ whiteSpace: "pre-wrap", background: "#0f172a", color: "white", padding: "1rem", borderRadius: "12px", overflowX: "auto" }}>{JSON.stringify(lastResult, null, 2)}</pre></details> : null}
  </div>;
}
