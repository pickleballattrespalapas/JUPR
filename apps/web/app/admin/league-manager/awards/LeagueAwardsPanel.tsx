"use client";

import { useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import type { AdminLeagueManagerListResponse, AdminLeagueManagerStatusResponse } from "@/lib/adminLeagueManagerApi";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type Props = { apiBase: string | null; clubId: string; status: AdminLeagueManagerStatusResponse };
type AwardRow = {
  category_key: string;
  category_label: string;
  player_id: number;
  player_name?: string;
  metric_display?: string;
  rank?: number;
  min_games?: number;
};
type EligiblePlayer = { player_id: number; player_name: string };
type WizardPreview = { awards?: AwardRow[]; fingerprint?: string; generated_at?: string; award_count?: number };
type MintState = {
  status?: string;
  attempt_count?: number;
  expected_count?: number;
  verified_count?: number;
  verified_at?: string;
  last_error?: string | null;
};
type AwardsWizard = {
  version?: number;
  status: string;
  revision?: number;
  frozen_at?: string | null;
  frozen_by?: string | null;
  preview?: WizardPreview | null;
  final_awards?: AwardRow[];
  override_notes?: Record<string, string>;
  mint?: MintState;
  archive?: { status?: string; archived_at?: string; archived_by?: string };
};
type AwardsResponse = {
  ok: boolean;
  mode?: string;
  league_name: string;
  league?: Record<string, unknown>;
  awards: AwardRow[];
  award_count: number;
  eligible_players?: EligiblePlayer[];
  wizard: AwardsWizard;
  writes_enabled?: boolean;
  service_role_ready?: boolean;
  badge_definitions_ready?: boolean;
  badge_definition_count?: number;
  badge_definition_required_count?: number;
  missing_badge_ids?: string[];
  badge_seed_migration?: string;
  badge_definition_readiness_error?: string;
  badge_expected_count?: number;
  badge_verified_count?: number;
  idempotent_replay?: boolean;
  warnings?: string[];
};
type OverrideDraft = { playerId: number; reason: string };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800, cursor: "pointer" };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function shortValue(value: unknown): string {
  if (value == null || value === "") return "—";
  if (typeof value === "object") return JSON.stringify(value).slice(0, 160);
  return String(value);
}

function awardKey(award: AwardRow): string {
  return `${award.category_key}:${award.rank || 1}`;
}

function newOperationKey(action: string): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") return `${action}:${crypto.randomUUID()}`;
  return `${action}:${Date.now()}:${Math.random().toString(16).slice(2)}`;
}

export default function LeagueAwardsPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [leagues, setLeagues] = useState<string[]>([]);
  const [leagueName, setLeagueName] = useState("");
  const [state, setState] = useState<AwardsResponse | null>(null);
  const [overrideDrafts, setOverrideDrafts] = useState<Record<string, OverrideDraft>>({});
  const [operationKeys, setOperationKeys] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const listRequest = useLatestRequestGuard(accessToken, clearProtectedAwardsState);
  const wizardRequest = useLatestRequestGuard(accessToken);
  const actionRequest = useLatestRequestGuard(accessToken);

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using League Awards.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
  }

  function leagueAwardsPath(suffix = "", selectedLeague = leagueName): string {
    return `/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(selectedLeague)}/awards${suffix}`;
  }

  function hydrate(payload: AwardsResponse): void {
    setState(payload);
    const previewAwards = payload.wizard?.preview?.awards || payload.awards || [];
    const finalByKey = new Map((payload.wizard?.final_awards || []).map((award) => [awardKey(award), award]));
    const notes = payload.wizard?.override_notes || {};
    const drafts: Record<string, OverrideDraft> = {};
    for (const award of previewAwards) {
      const key = awardKey(award);
      const finalAward = finalByKey.get(key);
      drafts[key] = { playerId: finalAward?.player_id ?? award.player_id, reason: notes[key] || "" };
    }
    setOverrideDrafts(drafts);
  }

  function keyFor(action: string): string {
    const existing = operationKeys[action];
    if (existing) return existing;
    const created = newOperationKey(action);
    setOperationKeys((current) => ({ ...current, [action]: created }));
    return created;
  }

  function clearKey(action: string): void {
    setOperationKeys((current) => {
      const next = { ...current };
      delete next[action];
      return next;
    });
  }

  function clearProtectedAwardsState() {
    wizardRequest.invalidate();
    actionRequest.invalidate();
    setBusy(false); setMessage(null); setLeagues([]); setLeagueName(""); setState(null);
    setOverrideDrafts({}); setOperationKeys({});
  }

  async function loadLeagues() {
    const selectedBeforeRefresh = leagueName;
    const generation = listRequest.begin();
    wizardRequest.invalidate();
    actionRequest.invalidate();
    setBusy(true);
    setMessage(null);
    setState(null);
    setOverrideDrafts({});
    try {
      const payload = await requestJson<AdminLeagueManagerListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues`);
      if (!listRequest.isCurrent(generation)) return;
      const names = (payload.leagues || []).map((league) => league.league_name).filter(Boolean);
      setLeagues(names);
      if (selectedBeforeRefresh && names.includes(selectedBeforeRefresh)) {
        await recoverWizard(selectedBeforeRefresh);
      } else if (selectedBeforeRefresh) {
        setLeagueName("");
        setOperationKeys({});
      }
      setMessage(names.length ? `Loaded ${names.length} league(s). Select one to open its saved awards state.` : "No leagues are available for awards yet.");
    } catch (error) {
      if (listRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to load leagues.");
    } finally {
      if (listRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function recoverWizard(selectedLeague = leagueName) {
    const generation = wizardRequest.begin();
    if (!selectedLeague) {
      setMessage("Select a league before recovering awards state.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AwardsResponse>(leagueAwardsPath("", selectedLeague));
      if (!wizardRequest.isCurrent(generation)) return;
      hydrate(payload);
      setMessage(`Recovered revision ${payload.wizard.revision || 0}; current step is ${payload.wizard.status.replace(/_/g, " ")}.`);
    } catch (error) {
      if (wizardRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to recover the awards workflow.");
    } finally {
      if (wizardRequest.isCurrent(generation)) setBusy(false);
    }
  }

  function selectLeague(selectedLeague: string) {
    actionRequest.invalidate();
    setLeagueName(selectedLeague);
    setState(null);
    setOverrideDrafts({});
    setOperationKeys({});
    setMessage(null);
    if (selectedLeague) void recoverWizard(selectedLeague);
    else wizardRequest.invalidate();
  }

  async function runAction(action: "freeze" | "preview" | "mint" | "archive", confirmationText = "") {
    if (!leagueName) {
      setMessage("Select a league first.");
      return;
    }
    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    const idempotencyKey = keyFor(action);
    try {
      const payload = await requestJson<AwardsResponse>(leagueAwardsPath(`/${action}`), {
        method: "POST",
        body: JSON.stringify({
          idempotency_key: idempotencyKey,
          confirmation_text: confirmationText,
          source: `next_league_manager_awards_${action}`
        })
      });
      if (!actionRequest.isCurrent(generation)) return;
      hydrate(payload);
      clearKey(action);
      setMessage(
        action === "mint"
          ? `Mint verified ${payload.badge_verified_count || 0} of ${payload.badge_expected_count || 0} expected badge row(s).`
          : `${action[0].toUpperCase()}${action.slice(1)} saved at workflow revision ${payload.wizard.revision || 0}.`
      );
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(`${error instanceof Error ? error.message : `Unable to ${action} awards.`} The same operation key is retained for a safe retry; use Recover saved state first.`);
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function saveOverrides() {
    const fingerprint = state?.wizard?.preview?.fingerprint;
    const awards = state?.wizard?.preview?.awards || [];
    if (!fingerprint) {
      setMessage("Recover and persist an award preview before confirming overrides.");
      return;
    }
    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    const action = "overrides";
    const idempotencyKey = keyFor(action);
    try {
      const payload = await requestJson<AwardsResponse>(leagueAwardsPath("/overrides"), {
        method: "POST",
        body: JSON.stringify({
          idempotency_key: idempotencyKey,
          preview_fingerprint: fingerprint,
          overrides: awards.map((award) => {
            const draft = overrideDrafts[awardKey(award)] || { playerId: award.player_id, reason: "" };
            return { category_key: award.category_key, rank: award.rank || 1, player_id: draft.playerId, reason: draft.reason };
          }),
          source: "next_league_manager_awards_overrides"
        })
      });
      if (!actionRequest.isCurrent(generation)) return;
      hydrate(payload);
      clearKey(action);
      setMessage(`Confirmed ${payload.award_count} final award row(s). Changed winners include a persisted reason.`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(`${error instanceof Error ? error.message : "Unable to save award overrides."} The operation key is retained for retry.`);
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  useAuthenticatedAutoLoad(status.enabled ? accessToken : "", loadLeagues);

  const wizard = state?.wizard;
  const workflowStatus = wizard?.status || "not_started";
  const previewAwards = wizard?.preview?.awards || [];
  const finalAwards = wizard?.final_awards || [];
  const displayAwards = finalAwards.length ? finalAwards : previewAwards;
  const writeReady = Boolean(state?.writes_enabled && state?.service_role_ready && status.awards_write_enabled !== false);
  const mintReady = Boolean(writeReady && state?.badge_definitions_ready === true);
  const messageIsError = Boolean(message && /unable|error|disabled|required|stale|failed|not |could not|before/i.test(message));

  if (!status.enabled) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>League Manager is disabled</h2>
        <p style={{ color: "#475569" }}>Enable the guarded League Manager flag before using awards. Streamlit remains the fallback.</p>
      </article>
    );
  }

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Admin session and recovery</h2>
        <p style={{ color: "#475569" }}>{adminSessionLabel(session)}</p>
        {sessionLoading ? <p>Checking session…</p> : null}
        {sessionMessage ? <p style={{ color: "#64748b" }}>{sessionMessage}</p> : null}
        <p style={{ color: "#475569" }}>Every step is saved in league award metadata. A refresh, timeout, or failed mint can be recovered without guessing what completed.</p>
      </article>

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>1. Select and recover league</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label>League<br /><select value={leagueName} onChange={(event) => selectLeague(event.target.value)} disabled={busy || !accessToken} style={inputStyle}><option value="">Select a league</option>{leagues.map((name) => <option key={name} value={name}>{name}</option>)}</select></label>
          <button type="button" onClick={loadLeagues} disabled={busy || !accessToken} style={buttonStyle}>{busy ? "Refreshing…" : "Refresh leagues"}</button>
          <button type="button" onClick={() => void recoverWizard()} disabled={busy || !leagueName} style={ghostButtonStyle}>Retry saved state</button>
        </div>
        {!busy && !leagues.length ? <p style={{ color: "#64748b" }}>No leagues are available.</p> : null}
        {message ? <p role="status" style={{ color: messageIsError ? "#b91c1c" : "#166534" }}>{message}</p> : null}
        {wizard ? <p style={{ color: "#475569" }}>Saved step: <strong>{workflowStatus.replace(/_/g, " ")}</strong> · Revision <strong>{wizard.revision || 0}</strong> · League status <strong>{shortValue(state?.league?.status)}</strong></p> : null}
        {state && !writeReady ? <p style={{ color: "#92400e" }}>Writes are closed. FastAPI must have <code>JUPR_ENABLE_NEXT_ADMIN_LEAGUE_AWARDS_WRITE=1</code> and a server-only service-role key. Use Streamlit fallback until the gate is ready.</p> : null}
        {state && state.badge_definitions_ready !== true ? (
          <p role="alert" style={{ color: "#b91c1c" }}>
            Badge minting is blocked: found {state.badge_definition_count || 0} of {state.badge_definition_required_count || 4} required definitions.
            {state.missing_badge_ids?.length ? <> Missing <code>{state.missing_badge_ids.join(", ")}</code>.</> : null}
            {state.badge_seed_migration ? <> Apply the reviewed deployment equivalent of <code>{state.badge_seed_migration}</code> before the staging write smoke.</> : null}
          </p>
        ) : null}
      </article>

      {wizard && workflowStatus === "not_started" ? (
        <article style={{ ...cardStyle, borderColor: "#fed7aa" }}>
          <h2 style={{ marginTop: 0 }}>2. Freeze league</h2>
          <p style={{ color: "#7c2d12" }}>Freezing marks the league ended and locks the award snapshot workflow. Match corrections must happen before this step or through Match Log and Replay History.</p>
          <ConfirmAction
            triggerLabel={busy ? "Working…" : "Freeze and save"}
            title="Freeze this league for awards?"
            description="This marks the league ended and freezes the awards workflow snapshot. Match corrections must use Match Log and Replay History afterward."
            confirmLabel="Yes, freeze league"
            confirmationText="FREEZE LEAGUE AWARDS"
            tone="danger"
            disabled={!writeReady}
            busy={busy}
            onConfirm={(confirmationText) => runAction("freeze", confirmationText)}
          />
        </article>
      ) : null}

      {wizard && ["frozen", "previewed", "overrides_confirmed"].includes(workflowStatus) && Number(wizard.mint?.attempt_count || 0) === 0 ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>3. Persist award preview</h2>
          <p style={{ color: "#475569" }}>FastAPI recomputes the Python-authoritative top performers and stores the exact rows and fingerprint used by later steps.</p>
          <button type="button" onClick={() => runAction("preview")} disabled={busy || !writeReady} style={buttonStyle}>{wizard.preview ? "Recompute and replace preview" : "Compute and save preview"}</button>
        </article>
      ) : null}

      {wizard?.preview ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>4. Review and document overrides</h2>
          <p style={{ color: "#475569" }}>Preview fingerprint <code>{wizard.preview.fingerprint?.slice(0, 16)}…</code>. Changing a winner requires a reason of at least eight characters; both are persisted and audit-attributed.</p>
          {displayAwards.length ? (
            <div style={{ display: "grid", gap: "0.75rem" }}>
              {previewAwards.map((award) => {
                const key = awardKey(award);
                const draft = overrideDrafts[key] || { playerId: award.player_id, reason: "" };
                const changed = Number(draft.playerId) !== Number(award.player_id);
                return (
                  <fieldset key={key} style={{ border: "1px solid #e2e8f0", borderRadius: "10px", padding: "0.75rem" }}>
                    <legend style={{ fontWeight: 800 }}>{award.category_label || award.category_key} #{award.rank || 1}</legend>
                    <p style={{ color: "#475569" }}>Computed: {award.player_name || `Player ${award.player_id}`} · {award.metric_display || "—"} · Min {award.min_games ?? "—"} games</p>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "0.75rem" }}>
                      <label>Winner<br /><select value={draft.playerId} onChange={(event) => setOverrideDrafts((current) => ({ ...current, [key]: { ...draft, playerId: Number(event.target.value) } }))} disabled={Number(wizard.mint?.attempt_count || 0) > 0} style={inputStyle}>{(state?.eligible_players || []).map((player) => <option key={player.player_id} value={player.player_id}>{player.player_name} (#{player.player_id})</option>)}</select></label>
                      <label>Override reason {changed ? "(required)" : "(not needed)"}<br /><input value={draft.reason} onChange={(event) => setOverrideDrafts((current) => ({ ...current, [key]: { ...draft, reason: event.target.value } }))} disabled={!changed || Number(wizard.mint?.attempt_count || 0) > 0} style={inputStyle} /></label>
                    </div>
                  </fieldset>
                );
              })}
              <button type="button" onClick={saveOverrides} disabled={busy || !writeReady || Number(wizard.mint?.attempt_count || 0) > 0} style={buttonStyle}>{workflowStatus === "overrides_confirmed" ? "Save revised confirmations" : "Confirm winners and reasons"}</button>
            </div>
          ) : <p style={{ color: "#92400e" }}>No qualifying awards were found. Confirm the empty preview before minting zero rows.</p>}
          {!previewAwards.length ? <button type="button" onClick={saveOverrides} disabled={busy || !writeReady} style={buttonStyle}>Confirm empty preview</button> : null}
        </article>
      ) : null}

      {wizard && ["overrides_confirmed", "minting", "mint_failed"].includes(workflowStatus) ? (
        <article style={{ ...cardStyle, borderColor: workflowStatus === "mint_failed" ? "#fecaca" : "#bfdbfe" }}>
          <h2 style={{ marginTop: 0 }}>5. Mint and verify badges</h2>
          <p style={{ color: "#334155" }}>A mint is successful only after FastAPI reads back every expected <code>player_badges</code> row. Partial or unavailable writes remain <strong>mint failed</strong> and can be retried with the retained idempotency key.</p>
          {wizard.mint?.last_error ? <p style={{ color: "#b91c1c" }}>Last verified failure: {wizard.mint.last_error}</p> : null}
          <p style={{ color: "#475569" }}>Attempts: {wizard.mint?.attempt_count || 0} · Expected: {wizard.mint?.expected_count ?? "—"} · Verified: {wizard.mint?.verified_count ?? "—"}</p>
          <ConfirmAction
            triggerLabel={workflowStatus === "mint_failed" || workflowStatus === "minting" ? "Retry mint and verification" : "Mint and verify"}
            title="Mint and verify these league awards?"
            description={`FastAPI will mint the reviewed award set and verify every expected badge row. Expected rows: ${wizard.mint?.expected_count ?? previewAwards.length}.`}
            confirmLabel="Yes, mint and verify"
            confirmationText="MINT AWARDS"
            disabled={!mintReady}
            busy={busy}
            onConfirm={(confirmationText) => runAction("mint", confirmationText)}
          />
        </article>
      ) : null}

      {wizard && ["minted", "archived"].includes(workflowStatus) ? (
        <article style={{ ...cardStyle, borderColor: "#bbf7d0" }}>
          <h2 style={{ marginTop: 0 }}>6. Archive</h2>
          <p style={{ color: "#166534" }}>Mint result: <strong>{wizard.mint?.status}</strong> · Verified {wizard.mint?.verified_count || 0} of {wizard.mint?.expected_count || 0} expected row(s).</p>
          {workflowStatus === "archived" ? <p><strong>Archived.</strong> This workflow is read-only and remains recoverable for audit review.</p> : (
            <ConfirmAction
              triggerLabel="Archive completed league"
              title="Archive this completed league?"
              description="This closes the awards workflow as read-only while retaining its saved state and audit history."
              confirmLabel="Yes, archive league"
              confirmationText="ARCHIVE LEAGUE"
              tone="danger"
              disabled={!writeReady}
              busy={busy}
              onConfirm={(confirmationText) => runAction("archive", confirmationText)}
            />
          )}
        </article>
      ) : null}

      {state?.warnings?.length ? <article style={{ ...cardStyle, background: "#fffbeb" }}><strong>Warnings</strong><ul>{state.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul></article> : null}
    </div>
  );
}
