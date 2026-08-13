"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import { InteractionDialog, StaticActionFeedback, actionSuccess, actionUncertain, type ActionCompletion } from "@/components/interaction";
import type { AdminPlayerEditorDetailResponse, AdminPlayerEditorLeagueRating, AdminPlayerEditorListResponse, AdminPlayerEditorOperationResponse, AdminPlayerEditorPlayer, AdminPlayerEditorStatusResponse, AdminPlayerEditorWriteResponse, AdminPlayerMergePreview, AdminPlayerSocialIdentity, AdminPlayerSocialIdentityListResponse } from "@/lib/adminPlayerEditorApi";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type Props = { apiBase: string | null; clubId: string; status: AdminPlayerEditorStatusResponse };
type PlayerWriteRecovery = {
  operationKey: string;
  scope: string;
  workflow: "create" | "update" | "league-rating";
  entityId?: string;
  message: string;
  status: string;
};
type StoredPlayerWriteRecovery = PlayerWriteRecovery & { version: 1 };
const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.5rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };
const leagueRatingConfirmText = "SAVE LEAGUE RATING"; const socialConfirmText = "LINK SOCIAL"; const mergeConfirmText = "MERGE"; const compensateConfirmText = "COMPENSATE MERGE"; const replayEvidenceConfirmText = "CONFIRM REPLAY RECOVERY";
function apiUrl(apiBase: string, path: string): string { return `${apiBase.replace(/\/$/, "")}${path}`; }
function juprLabel(value?: number | null): string { return value == null ? "—" : Number(value).toFixed(2); }
function socialLabel(row: AdminPlayerSocialIdentity): string { return `${row.display_name || "Unknown"}${row.linked_player_name ? ` → ${row.linked_player_name}` : " [unlinked]"}`; }
function playerOptionLabel(player: AdminPlayerEditorPlayer): string { return `${player.name} #${player.id}${player.active === false ? " [inactive]" : ""}`; }
function sumRecord(values?: Record<string, number>): number { if (values?.total != null) return Number(values.total || 0); return Object.values(values || {}).reduce((sum, value) => sum + Number(value || 0), 0); }
function isErrorMessage(message: string): boolean {
  return /\b(unable|required|sign in|error|missing|invalid|unknown|not found|blocked|unavailable|failed|must|cannot)\b/i.test(message);
}

class PlayerEditorRequestError extends Error {
  readonly status: number;
  readonly uncertain: boolean;
  readonly operationKey: string | null;

  constructor(message: string, status: number, uncertain = false, operationKey: string | null = null) {
    super(message);
    this.name = "PlayerEditorRequestError";
    this.status = status;
    this.uncertain = uncertain;
    this.operationKey = operationKey;
  }
}

function playerEditorWriteIsUncertain(error: unknown): boolean {
  return !(error instanceof PlayerEditorRequestError) || error.uncertain;
}

export default function PlayerEditorPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [players, setPlayers] = useState<AdminPlayerEditorPlayer[]>([]);
  const [detail, setDetail] = useState<AdminPlayerEditorDetailResponse | null>(null);
  const [selectedId, setSelectedId] = useState("");
  const [newName, setNewName] = useState(""); const [newStartingJupr, setNewStartingJupr] = useState("3.5");
  const [editName, setEditName] = useState(""); const [editRating, setEditRating] = useState("3.5"); const [editStartingRating, setEditStartingRating] = useState("3.5"); const [editActive, setEditActive] = useState(true);
  const [selectedLeagueRatingId, setSelectedLeagueRatingId] = useState(""); const [editLeagueRating, setEditLeagueRating] = useState("3.5"); const [editLeagueStartingRating, setEditLeagueStartingRating] = useState("3.5"); const [editLeagueActive, setEditLeagueActive] = useState(true);
  const [socialPeople, setSocialPeople] = useState<AdminPlayerSocialIdentity[]>([]); const [socialPlayers, setSocialPlayers] = useState<Array<{ id: number; name: string; active?: boolean | null }>>([]); const [selectedSocialId, setSelectedSocialId] = useState(""); const [socialLinkedPlayerId, setSocialLinkedPlayerId] = useState(""); const [socialDisplayName, setSocialDisplayName] = useState("");
  const [mergeSourceId, setMergeSourceId] = useState(""); const [mergeTargetId, setMergeTargetId] = useState(""); const [mergePreview, setMergePreview] = useState<AdminPlayerMergePreview | null>(null);
  const [mergeOperationId, setMergeOperationId] = useState(""); const [mergeAttempted, setMergeAttempted] = useState(false); const [mergeRecovery, setMergeRecovery] = useState<AdminPlayerEditorWriteResponse | null>(null); const [replayJobId, setReplayJobId] = useState("");
  const [saving, setSaving] = useState(false); const [message, setMessage] = useState<string | null>(null); const [resultDialog, setResultDialog] = useState<string | null>(null);
  const [writeRecovery, setWriteRecovery] = useState<PlayerWriteRecovery | null>(null);
  const [checkingWriteRecovery, setCheckingWriteRecovery] = useState(false);
  const operationKeysRef = useRef(new Map<string, string>());
  const workspaceRequest = useLatestRequestGuard(accessToken, clearProtectedPlayerState);
  const playersRequest = useLatestRequestGuard(accessToken);
  const detailRequest = useLatestRequestGuard(accessToken);
  const socialRequest = useLatestRequestGuard(accessToken);
  const actionRequest = useLatestRequestGuard(accessToken);
  const writeRecoveryStorageKey = `jupr-player-editor-write-recovery:${clubId}`;

  useEffect(() => {
    setWriteRecovery(null);
    try {
      const raw = globalThis.sessionStorage?.getItem(writeRecoveryStorageKey);
      if (!raw) return;
      const stored = JSON.parse(raw) as Partial<StoredPlayerWriteRecovery>;
      if (
        stored.version === 1
        && typeof stored.operationKey === "string"
        && typeof stored.scope === "string"
        && ["create", "update", "league-rating"].includes(String(stored.workflow))
        && typeof stored.message === "string"
        && typeof stored.status === "string"
      ) {
        setWriteRecovery(stored as StoredPlayerWriteRecovery);
      }
    } catch {
      // The in-memory write guard remains available if session storage is blocked.
    }
  }, [writeRecoveryStorageKey]);

  function retainWriteRecovery(recovery: PlayerWriteRecovery) {
    setWriteRecovery(recovery);
    try {
      globalThis.sessionStorage?.setItem(
        writeRecoveryStorageKey,
        JSON.stringify({ version: 1, ...recovery } satisfies StoredPlayerWriteRecovery),
      );
    } catch {
      // The in-memory state still blocks additional writes in this page session.
    }
  }

  function clearWriteRecovery() {
    setWriteRecovery(null);
    try {
      globalThis.sessionStorage?.removeItem(writeRecoveryStorageKey);
    } catch {
      // A conclusive server status remains authoritative if cleanup is blocked.
    }
  }

  function mutationOperationKey(scope: string, request: Record<string, unknown>): string {
    const fingerprint = `${scope}:${JSON.stringify(request)}`;
    const existing = operationKeysRef.current.get(fingerprint);
    if (existing) return existing;
    const idempotencyKey = `player-editor:${globalThis.crypto.randomUUID()}`;
    operationKeysRef.current.set(fingerprint, idempotencyKey);
    return idempotencyKey;
  }

  function clearMutationOperationKey(scope: string, request: Record<string, unknown>) {
    operationKeysRef.current.delete(`${scope}:${JSON.stringify(request)}`);
  }

  function forgetMutationOperationKey(operationKey: string) {
    for (const [fingerprint, retainedKey] of operationKeysRef.current.entries()) {
      if (retainedKey === operationKey) operationKeysRef.current.delete(fingerprint);
    }
  }

  function requireReady(): boolean { if (!apiBase) { setMessage("API base URL is not configured."); return false; } if (!accessToken) { setMessage("Sign in at /admin/login before using the Player Editor."); return false; } if (!status.enabled) { setMessage("Next Player Editor is disabled on the API."); return false; } return true; }
  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> { if (!apiBase) throw new Error("API base URL is not configured."); if (!accessToken) throw new Error("Sign in at /admin/login before using the Player Editor."); const headers = new Headers(options?.headers); if (options?.body !== undefined) headers.set("Content-Type", "application/json"); headers.set("Authorization", `Bearer ${accessToken}`); const response = await fetch(apiUrl(apiBase, path), { ...options, headers }); const payload = await response.json().catch(() => null); if (!response.ok) { const detail = payload?.detail; const detailRecord = detail && typeof detail === "object" ? detail as Record<string, unknown> : null; const errorMessage = typeof detail === "string" ? detail : String(detailRecord?.message || `API error (${response.status})`); const explicitlyFailed = detailRecord?.kind === "failed" && detailRecord?.recovery_required !== true; const uncertainStatus = response.status >= 500 || [408, 425, 429].includes(response.status); throw new PlayerEditorRequestError(errorMessage, response.status, detailRecord?.kind === "uncertain" || detailRecord?.recovery_required === true || (!explicitlyFailed && uncertainStatus), typeof detailRecord?.operation_key === "string" ? detailRecord.operation_key : null); } return payload as T; }
  function seedEditForm(player: AdminPlayerEditorPlayer) { setEditName(player.name || ""); setEditRating(String(player.rating_jupr ?? 3.5)); setEditStartingRating(String(player.starting_jupr ?? player.rating_jupr ?? 3.5)); setEditActive(player.active !== false); }
  function seedLeagueRatingForm(row: AdminPlayerEditorLeagueRating | null) { setSelectedLeagueRatingId(row ? String(row.id) : ""); setEditLeagueRating(String(row?.rating_jupr ?? 3.5)); setEditLeagueStartingRating(String(row?.starting_jupr ?? row?.rating_jupr ?? 3.5)); setEditLeagueActive(row?.is_active !== false); }
  function seedSocialForm(row: AdminPlayerSocialIdentity | null) { setSelectedSocialId(row?.id || ""); setSocialDisplayName(row?.display_name || ""); setSocialLinkedPlayerId(row?.linked_player_id == null ? "" : String(row.linked_player_id)); }
  function clearPlayerDetail() { setDetail(null); setEditName(""); setEditRating("3.5"); setEditStartingRating("3.5"); setEditActive(true); seedLeagueRatingForm(null); }
  function clearProtectedPlayerState() {
    playersRequest.invalidate();
    detailRequest.invalidate();
    socialRequest.invalidate();
    setSaving(false); setMessage(null); setResultDialog(null);
    setPlayers([]); setSelectedId(""); clearPlayerDetail();
    setSocialPeople([]); setSocialPlayers([]); seedSocialForm(null);
    setMergeSourceId(""); setMergeTargetId(""); setMergePreview(null); setMergeRecovery(null); setMergeOperationId(""); setMergeAttempted(false); setReplayJobId("");
  }

  async function loadPlayers() { const generation = playersRequest.begin(); setMessage(null); if (!requireReady()) return null; setSaving(true); try { const payload = await requestJson<AdminPlayerEditorListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/players/editor/players`); if (!playersRequest.isCurrent(generation)) return null; setPlayers(payload.players || []); setMessage(`Loaded ${payload.count ?? payload.players?.length ?? 0} player(s).`); return payload; } catch (error) { if (playersRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to load players."); return null; } finally { if (playersRequest.isCurrent(generation)) setSaving(false); } }
  async function loadDetail(playerId: string) { const generation = detailRequest.begin(); setSelectedId(playerId); setDetail(null); setMessage(null); if (!playerId || !requireReady()) return; setSaving(true); try { const payload = await requestJson<AdminPlayerEditorDetailResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/players/editor/players/${encodeURIComponent(playerId)}`); if (!detailRequest.isCurrent(generation)) return; setDetail(payload); seedEditForm(payload.player); seedLeagueRatingForm(payload.league_ratings?.[0] || null); } catch (error) { if (detailRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to load player detail."); } finally { if (detailRequest.isCurrent(generation)) setSaving(false); } }
  async function loadSocialIdentities(completionMessage?: string) { const generation = socialRequest.begin(); setMessage(null); if (!requireReady()) return; setSaving(true); try { const payload = await requestJson<AdminPlayerSocialIdentityListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/players/editor/social-identities`); if (!socialRequest.isCurrent(generation)) return; setSocialPeople(payload.people || []); setSocialPlayers(payload.players || []); seedSocialForm((payload.people || [])[0] || null); setMessage(completionMessage || `Loaded ${payload.summary?.people ?? payload.people?.length ?? 0} social identit${(payload.people?.length ?? 0) === 1 ? "y" : "ies"}.`); } catch (error) { if (socialRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to load social identities."); } finally { if (socialRequest.isCurrent(generation)) setSaving(false); } }
  async function loadWorkspace() {
    const selectedPlayerBeforeRefresh = selectedId;
    const selectedSocialBeforeRefresh = selectedSocialId;
    const generation = workspaceRequest.begin();
    setMessage(null);
    if (!requireReady()) return;
    setSaving(true);
    try {
      const [playerPayload, socialPayload] = await Promise.all([
        requestJson<AdminPlayerEditorListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/players/editor/players`),
        requestJson<AdminPlayerSocialIdentityListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/players/editor/social-identities`)
      ]);
      if (!workspaceRequest.isCurrent(generation)) return;
      const nextPlayers = playerPayload.players || [];
      const nextSocialPeople = socialPayload.people || [];
      setPlayers(nextPlayers);
      setSocialPeople(nextSocialPeople);
      setSocialPlayers(socialPayload.players || []);
      if (selectedPlayerBeforeRefresh && !nextPlayers.some((row) => String(row.id) === selectedPlayerBeforeRefresh)) {
        detailRequest.invalidate();
        setSelectedId("");
        clearPlayerDetail();
      }
      if (!selectedSocialBeforeRefresh || !nextSocialPeople.some((row) => row.id === selectedSocialBeforeRefresh)) {
        seedSocialForm(nextSocialPeople[0] || null);
      }
      setMessage(`Loaded ${playerPayload.count ?? nextPlayers.length} player(s) and ${socialPayload.summary?.people ?? nextSocialPeople.length} social identit${nextSocialPeople.length === 1 ? "y" : "ies"}.`);
    } catch (error) {
      if (workspaceRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to load Player Editor options.");
    } finally {
      if (workspaceRequest.isCurrent(generation)) setSaving(false);
    }
  }

  function playerOperationPath(operationKey: string): string {
    const template = status.operation_endpoint
      || "/admin/clubs/{club_id}/players/editor/operations/{operation_key}";
    return template
      .replace("{club_id}", encodeURIComponent(clubId))
      .replace("{operation_key}", encodeURIComponent(operationKey));
  }

  async function applyRecoveredPlayerWrite(
    recovered: AdminPlayerEditorWriteResponse,
    recovery: PlayerWriteRecovery,
  ) {
    if (recovered.player) {
      const recoveredPlayer = recovered.player;
      setPlayers((current) => [
        ...current.filter((player) => player.id !== recoveredPlayer.id),
        recoveredPlayer,
      ].sort((left, right) => left.name.localeCompare(right.name)));
      setSelectedId(String(recoveredPlayer.id));
      await loadDetail(String(recoveredPlayer.id));
    } else if (recovery.entityId) {
      await loadDetail(recovery.entityId);
    } else {
      await loadPlayers();
    }
  }

  async function reconcilePlayerWrite(
    retainedRecovery: PlayerWriteRecovery | null = writeRecovery,
  ): Promise<ActionCompletion> {
    const recovery = retainedRecovery;
    if (!recovery) throw new Error("No Player Editor operation is waiting for recovery.");
    if (!requireReady()) throw new Error("Player Editor is not ready to inspect the retained operation.");
    setCheckingWriteRecovery(true);
    setMessage(null);
    try {
      const path = playerOperationPath(recovery.operationKey);
      const operation = await requestJson<AdminPlayerEditorOperationResponse>(path);
      let recovered = operation.result || null;
      let operationStatus = String(operation.status || "unknown");
      let recoveryRequired = operation.recovery_required === true;

      if (operationStatus !== "completed" && operationStatus !== "failed") {
        recovered = await requestJson<AdminPlayerEditorWriteResponse>(`${path}/reconcile`, {
          method: "POST",
          body: JSON.stringify({
            confirmation_text: "RECONCILE PLAYER OPERATION",
            source: "next_player_editor_operation_reconcile",
          }),
        });
        operationStatus = recovered.ok === false
          ? String(recovered.status || "failed")
          : "completed";
        recoveryRequired = recovered.recovery_required === true;
      }

      if (operationStatus === "completed" && recovered?.ok !== false) {
        await applyRecoveredPlayerWrite(recovered || { ok: true }, recovery);
        forgetMutationOperationKey(recovery.operationKey);
        clearWriteRecovery();
        const successMessage = `Exact Player Editor operation ${recovery.operationKey} completed and its authoritative result was applied.`;
        setMessage(successMessage);
        return actionSuccess("Player operation reconciled", successMessage);
      }

      if (operationStatus === "failed" && !recoveryRequired) {
        forgetMutationOperationKey(recovery.operationKey);
        clearWriteRecovery();
        const failedMessage = `Exact Player Editor operation ${recovery.operationKey} is proven failed. Review current data before starting a new request.`;
        setMessage(failedMessage);
        return actionSuccess("Player operation checked", failedMessage);
      }

      const pendingMessage = operation.error || recovery.message;
      const pending = { ...recovery, status: operationStatus, message: pendingMessage };
      retainWriteRecovery(pending);
      return actionUncertain(
        "Player operation still needs verification",
        `Operation ${recovery.operationKey} is ${operationStatus.replace(/_/g, " ")}. New Player Editor writes remain blocked.`,
        recovery.operationKey,
        "Check and reconcile exact operation",
        () => reconcilePlayerWrite(pending),
      );
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unable to reconcile the Player Editor operation.";
      retainWriteRecovery({ ...recovery, status: "recovery_required", message: errorMessage });
      setMessage(`${errorMessage} Operation ${recovery.operationKey} remains retained; do not submit another write.`);
      return actionUncertain(
        "Player operation still needs verification",
        `${errorMessage} The exact operation reference remains retained.`,
        recovery.operationKey,
        "Check and reconcile exact operation",
        () => reconcilePlayerWrite({ ...recovery, status: "recovery_required", message: errorMessage }),
      );
    } finally {
      setCheckingWriteRecovery(false);
    }
  }

  async function createPlayer() {
    setMessage(null);
    if (!requireReady()) return;
    if (writeRecovery) {
      setMessage(`Resolve exact operation ${writeRecovery.operationKey} before creating another player.`);
      return;
    }
    const name = newName.trim();
    const starting = Number(newStartingJupr);
    if (!name || !Number.isFinite(starting) || starting < 1 || starting > 7) { setMessage("Enter a name and a Starting JUPR between 1.0 and 7.0."); return; }
    const generation = actionRequest.begin();
    const request = { name, starting_jupr: starting, source: "next_player_editor_create" };
    const operationScope = `create:${clubId}`;
    const idempotencyKey = mutationOperationKey(operationScope, request);
    setSaving(true);
    try {
      const payload = await requestJson<AdminPlayerEditorWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/players/editor/players`, { method: "POST", body: JSON.stringify({ ...request, idempotency_key: idempotencyKey }) });
      clearMutationOperationKey(operationScope, request);
      if (!actionRequest.isCurrent(generation)) return;
      if (payload.player) {
        setPlayers((current) => [...current.filter((player) => player.id !== payload.player?.id), payload.player as AdminPlayerEditorPlayer].sort((left, right) => left.name.localeCompare(right.name)));
        setSelectedId(String(payload.player.id));
        await loadDetail(String(payload.player.id));
        if (!actionRequest.isCurrent(generation)) return;
      }
      setNewName(""); setNewStartingJupr("3.5"); setMessage("Player created or confirmed.");
    } catch (error) {
      const uncertain = playerEditorWriteIsUncertain(error);
      if (uncertain) {
        const operationKey = error instanceof PlayerEditorRequestError && error.operationKey ? error.operationKey : idempotencyKey;
        retainWriteRecovery({ operationKey, scope: operationScope, workflow: "create", message: error instanceof Error ? error.message : "Unable to confirm player creation.", status: "uncertain" });
      } else {
        clearMutationOperationKey(operationScope, request);
      }
      if (actionRequest.isCurrent(generation)) setMessage(`${error instanceof Error ? error.message : "Unable to create player."}${uncertain ? ` The exact create operation is retained as ${error instanceof PlayerEditorRequestError && error.operationKey ? error.operationKey : idempotencyKey}; check it before making another write.` : ""}`);
    } finally {
      if (actionRequest.isCurrent(generation)) setSaving(false);
    }
  }

  async function savePlayer() {
    setMessage(null);
    if (!selectedId || !requireReady()) return;
    if (writeRecovery) {
      setMessage(`Resolve exact operation ${writeRecovery.operationKey} before saving another player change.`);
      return;
    }
    const rating = Number(editRating);
    const starting = Number(editStartingRating);
    if (!editName.trim() || !Number.isFinite(rating) || !Number.isFinite(starting) || rating < 1 || rating > 7 || starting < 1 || starting > 7) { setMessage("Name, Overall JUPR, and Starting JUPR are required. Ratings must be between 1.0 and 7.0."); return; }
    if (!detail?.player.state_fingerprint) { setMessage("Reload this player before saving so the current server version can be verified."); return; }
    const generation = actionRequest.begin();
    const requestedPlayerId = selectedId;
    const request = { name: editName.trim(), rating_jupr: rating, starting_jupr: starting, active: editActive, expected_state_fingerprint: detail.player.state_fingerprint, source: "next_player_editor_update" };
    const operationScope = `player:${clubId}:${requestedPlayerId}`;
    const idempotencyKey = mutationOperationKey(operationScope, request);
    setSaving(true);
    try {
      const payload = await requestJson<AdminPlayerEditorWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/players/editor/players/${encodeURIComponent(requestedPlayerId)}`, { method: "PATCH", body: JSON.stringify({ ...request, idempotency_key: idempotencyKey }) });
      clearMutationOperationKey(operationScope, request);
      if (!actionRequest.isCurrent(generation)) return;
      if (payload.player) {
        setPlayers((current) => current.map((player) => player.id === payload.player?.id ? payload.player as AdminPlayerEditorPlayer : player).sort((left, right) => left.name.localeCompare(right.name)));
        await loadDetail(String(payload.player.id));
        if (!actionRequest.isCurrent(generation)) return;
      }
      setMessage(null); setResultDialog(`Player saved: ${payload.player?.name || editName.trim()}. Use Match Log and Replay History if downstream rating repair is needed.`);
    } catch (error) {
      const uncertain = playerEditorWriteIsUncertain(error);
      if (uncertain) {
        const operationKey = error instanceof PlayerEditorRequestError && error.operationKey ? error.operationKey : idempotencyKey;
        retainWriteRecovery({ operationKey, scope: operationScope, workflow: "update", entityId: requestedPlayerId, message: error instanceof Error ? error.message : "Unable to confirm the player update.", status: "uncertain" });
      } else {
        clearMutationOperationKey(operationScope, request);
      }
      if (actionRequest.isCurrent(generation)) setMessage(`${error instanceof Error ? error.message : "Unable to save player."}${uncertain ? ` The exact player operation is retained as ${error instanceof PlayerEditorRequestError && error.operationKey ? error.operationKey : idempotencyKey}; check it before making another write.` : ""}`);
    } finally {
      if (actionRequest.isCurrent(generation)) setSaving(false);
    }
  }

  async function saveLeagueRating(confirmationText: string): Promise<ActionCompletion> {
    setMessage(null);
    if (!selectedId || !selectedLeagueRatingId || !requireReady()) throw new Error("Select a player and league rating before saving.");
    if (writeRecovery) throw new Error(`Resolve exact operation ${writeRecovery.operationKey} before saving another league rating.`);
    const rating = Number(editLeagueRating);
    const starting = Number(editLeagueStartingRating);
    if (!Number.isFinite(rating) || !Number.isFinite(starting) || rating < 1 || rating > 7 || starting < 1 || starting > 7) { const text = "League JUPR and league Starting JUPR must be between 1.0 and 7.0."; setMessage(text); throw new Error(text); }
    const selectedLeagueRating = detail?.league_ratings.find((row) => String(row.id) === selectedLeagueRatingId);
    if (!selectedLeagueRating?.state_fingerprint) { const text = "Reload this league rating before saving so the current server version can be verified."; setMessage(text); throw new Error(text); }
    const generation = actionRequest.begin();
    const playerId = selectedId;
    const leagueRatingId = selectedLeagueRatingId;
    const request = { rating_jupr: rating, starting_jupr: starting, is_active: editLeagueActive, expected_state_fingerprint: selectedLeagueRating.state_fingerprint, confirmation_text: confirmationText, source: "next_player_editor_league_rating" };
    const operationScope = `league-rating:${clubId}:${playerId}:${leagueRatingId}`;
    const idempotencyKey = mutationOperationKey(operationScope, request);
    setSaving(true);
    try {
      const payload = await requestJson<AdminPlayerEditorWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/players/editor/players/${encodeURIComponent(playerId)}/league-ratings/${encodeURIComponent(leagueRatingId)}`, { method: "PATCH", body: JSON.stringify({ ...request, idempotency_key: idempotencyKey }) });
      clearMutationOperationKey(operationScope, request);
      const completion = actionSuccess("League rating saved", "The targeted league-rating correction was saved and audit-flagged for review.");
      if (!actionRequest.isCurrent(generation)) return completion;
      if (payload.league_ratings && detail) {
        setDetail({ ...detail, league_ratings: payload.league_ratings });
        const updated = payload.league_ratings.find((row) => String(row.id) === leagueRatingId) || payload.league_ratings[0] || null;
        seedLeagueRatingForm(updated);
      }
      setMessage("League rating saved and audit-flagged for review. Run Replay History if you need to rebuild derived history.");
      return completion;
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to save league rating.");
      if (playerEditorWriteIsUncertain(error)) {
        const operationKey = error instanceof PlayerEditorRequestError && error.operationKey ? error.operationKey : idempotencyKey;
        retainWriteRecovery({
          operationKey,
          scope: operationScope,
          workflow: "league-rating",
          entityId: playerId,
          message: error instanceof Error ? error.message : "Unable to confirm the league-rating update.",
          status: "uncertain",
        });
        return actionUncertain(
          "League rating outcome needs checking",
          "The request was sent, but its durable result could not be confirmed. Check and reconcile the retained operation before making another rating change.",
          operationKey,
          "Check and reconcile exact operation",
          () => reconcilePlayerWrite({
            operationKey,
            scope: operationScope,
            workflow: "league-rating",
            entityId: playerId,
            message: error instanceof Error ? error.message : "Unable to confirm the league-rating update.",
            status: "uncertain",
          }),
        );
      }
      clearMutationOperationKey(operationScope, request);
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setSaving(false);
    }
  }

  async function saveSocialIdentity(confirmationText: string) {
    setMessage(null);
    if (!selectedSocialId || !requireReady()) throw new Error("Select a social identity before saving.");
    const generation = actionRequest.begin();
    const socialId = selectedSocialId;
    setSaving(true);
    try {
      const payload = await requestJson<AdminPlayerEditorWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/players/editor/social-identities/${encodeURIComponent(socialId)}`, { method: "PATCH", body: JSON.stringify({ display_name: socialDisplayName, linked_player_id: socialLinkedPlayerId ? Number(socialLinkedPlayerId) : null, confirmation_text: confirmationText, source: "next_player_editor_social_identity" }) });
      const savedName = payload.club_person?.display_name || socialId;
      const completion = actionSuccess("Social identity saved", `The social identity ${savedName} was saved.`);
      if (!actionRequest.isCurrent(generation)) return completion;
      await loadSocialIdentities(`Saved social identity ${payload.club_person?.display_name || socialId} and refreshed the list.`);
      return completion;
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to save social identity.");
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setSaving(false);
    }
  }

  async function autoLinkSocialIdentities(confirmationText: string) {
    setMessage(null);
    if (!requireReady()) throw new Error("Player Editor is not ready for this action.");
    const generation = actionRequest.begin();
    setSaving(true);
    try {
      const payload = await requestJson<AdminPlayerEditorWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/players/editor/social-identities/auto-link`, { method: "POST", body: JSON.stringify({ confirmation_text: confirmationText, source: "next_player_editor_social_auto_link" }) });
      const linkedCount = payload.linked_count ?? 0;
      const completion = actionSuccess("Social identities linked", `${linkedCount} exact-name social identit${linkedCount === 1 ? "y was" : "ies were"} linked.`);
      if (!actionRequest.isCurrent(generation)) return completion;
      await loadSocialIdentities(`Auto-linked ${payload.linked_count ?? 0} social identit${payload.linked_count === 1 ? "y" : "ies"}; skipped ${payload.skipped_count ?? 0}. The list is refreshed.`);
      return completion;
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to auto-link social identities.");
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setSaving(false);
    }
  }

  async function previewMerge() {
    setMessage(null); setMergePreview(null); setMergeAttempted(false); setMergeRecovery(null); setMergeOperationId("");
    if (!mergeSourceId || !mergeTargetId || !requireReady()) return;
    const generation = actionRequest.begin();
    const sourceId = mergeSourceId;
    const targetId = mergeTargetId;
    setSaving(true);
    try {
      const payload = await requestJson<AdminPlayerMergePreview>(`/admin/clubs/${encodeURIComponent(clubId)}/players/editor/merge/preview`, { method: "POST", body: JSON.stringify({ source_player_id: Number(sourceId), target_player_id: Number(targetId), source: "next_player_editor_merge_preview" }) });
      if (!actionRequest.isCurrent(generation)) return;
      setMergePreview(payload);
      setMergeOperationId(globalThis.crypto.randomUUID());
      setMessage(`Preview ready: ${sumRecord(payload.match_reference_counts)} source match reference(s), ${(payload.league_rating_plan?.move_ids || []).length} league row(s) to move, ${(payload.league_rating_plan?.delete_ids || []).length} conflicting row(s) to delete.`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to preview merge.");
    } finally {
      if (actionRequest.isCurrent(generation)) setSaving(false);
    }
  }
  async function executeMerge(confirmationText: string) {
    setMessage(null);
    if (!mergeSourceId || !mergeTargetId || !mergePreview?.preview_fingerprint || !mergeOperationId || !requireReady()) throw new Error("Review a valid player merge before continuing.");
    const operationId = mergeOperationId;
    const sourceId = mergeSourceId;
    const targetId = mergeTargetId;
    const selectedPlayerNeedsReload = selectedId === sourceId || selectedId === targetId;
    const generation = actionRequest.begin();
    setMergeAttempted(true);
    setSaving(true);
    try {
      const payload = await requestJson<AdminPlayerEditorWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/players/editor/merge`, {
        method: "POST",
        body: JSON.stringify({
          source_player_id: Number(sourceId),
          target_player_id: Number(targetId),
          preview_fingerprint: mergePreview.preview_fingerprint,
          operation_id: operationId,
          confirmation_text: confirmationText,
          source: "next_player_editor_merge"
        })
      });
      const completion = actionSuccess("Player merge recorded", `Player #${payload.source_player_id} was merged into #${payload.target_player_id}. Full Replay History evidence is still required.`);
      if (!actionRequest.isCurrent(generation)) return completion;
      setMergePreview(null);
      setMergeRecovery(payload);
      await loadPlayers();
      if (!actionRequest.isCurrent(generation)) return completion;
      if (selectedPlayerNeedsReload) {
        detailRequest.invalidate();
        setSelectedId(targetId);
        clearPlayerDetail();
        await loadDetail(targetId);
        if (!actionRequest.isCurrent(generation)) return completion;
      }
      setMessage(`Merged player #${payload.source_player_id} into #${payload.target_player_id}. Operation ${payload.operation_id || operationId} is pending full replay evidence.`);
      return completion;
    } catch (error) {
      if (actionRequest.isCurrent(generation)) {
        setMergePreview(null);
        setMergeRecovery(null);
        setMessage(`${error instanceof Error ? error.message : "Unable to execute merge."} Outcome unknown for operation ${operationId}; check that operation before retrying.`);
      }
      return actionUncertain(
        "Player merge needs verification",
        `The request ended without a confirmed result. Check merge operation ${operationId} before retrying.`,
        operationId,
        "Check merge operation",
        async () => {
          const payload = await requestJson<{ operation: { status?: string; source_player_id?: number; target_player_id?: number }; recovery?: AdminPlayerEditorWriteResponse["recovery"] }>(`/admin/clubs/${encodeURIComponent(clubId)}/players/editor/merge/${encodeURIComponent(operationId)}`);
          const operationStatus = payload.operation?.status || "unknown";
          setMergeRecovery({ ok: true, operation_id: operationId, operation_status: operationStatus, source_player_id: payload.operation?.source_player_id, target_player_id: payload.operation?.target_player_id, recovery: payload.recovery });
          setMessage(`Operation ${operationId} status: ${operationStatus}.`);
          return actionSuccess("Merge operation checked", `Operation ${operationId} is ${operationStatus}. Review its recovery requirements before continuing.`);
        }
      );
    } finally {
      if (actionRequest.isCurrent(generation)) setSaving(false);
    }
  }
  async function lookupMergeOperation() {
    const operationId = mergeRecovery?.operation_id || mergeOperationId;
    if (!operationId || !requireReady()) return;
    const generation = actionRequest.begin();
    setSaving(true);
    try {
      const payload = await requestJson<{ operation: { status?: string; source_player_id?: number; target_player_id?: number }; recovery?: AdminPlayerEditorWriteResponse["recovery"] }>(`/admin/clubs/${encodeURIComponent(clubId)}/players/editor/merge/${encodeURIComponent(operationId)}`);
      if (!actionRequest.isCurrent(generation)) return;
      setMergeRecovery({ ok: true, operation_id: operationId, operation_status: payload.operation?.status, source_player_id: payload.operation?.source_player_id, target_player_id: payload.operation?.target_player_id, recovery: payload.recovery });
      setMessage(`Operation ${operationId} status: ${payload.operation?.status || "unknown"}.`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(`${error instanceof Error ? error.message : "Unable to look up merge operation."} If not found, refresh the merge preview before retrying.`);
    } finally {
      if (actionRequest.isCurrent(generation)) setSaving(false);
    }
  }

  async function attachReplayEvidence(confirmationText: string) {
    const operationId = mergeRecovery?.operation_id;
    if (!operationId || !requireReady()) throw new Error("No merge operation is ready for Replay History evidence.");
    const generation = actionRequest.begin();
    setSaving(true);
    try {
      const payload = await requestJson<AdminPlayerEditorWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/players/editor/merge/${encodeURIComponent(operationId)}/replay-evidence`, { method: "POST", body: JSON.stringify({ replay_job_id: replayJobId.trim(), confirmation_text: confirmationText, source: "next_player_editor_merge_replay_evidence" }) });
      const completion = actionSuccess("Replay evidence attached", "Full Replay History evidence is attached and merge recovery is complete.");
      if (!actionRequest.isCurrent(generation)) return completion;
      setMergeRecovery(payload);
      setReplayJobId("");
      setMessage("Succeeded full Replay History evidence attached. Merge recovery is complete.");
      return completion;
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to attach replay evidence.");
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setSaving(false);
    }
  }

  async function compensateMerge(confirmationText: string) {
    const operationId = mergeRecovery?.operation_id;
    if (!operationId || !requireReady()) throw new Error("No merge operation is ready to compensate.");
    const generation = actionRequest.begin();
    setSaving(true);
    try {
      const payload = await requestJson<AdminPlayerEditorWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/players/editor/merge/${encodeURIComponent(operationId)}/compensate`, { method: "POST", body: JSON.stringify({ confirmation_text: confirmationText, source: "next_player_editor_merge_compensation" }) });
      const completion = actionSuccess("Merge compensated", "The pre-merge player, match, league, and social-link state was restored.");
      if (!actionRequest.isCurrent(generation)) return completion;
      setMergeRecovery(payload);
      await loadPlayers();
      if (!actionRequest.isCurrent(generation)) return completion;
      setMessage("Merge compensated: pre-merge player, match, league, and social-link state was restored; timestamps may advance.");
      return completion;
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to compensate merge.");
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setSaving(false);
    }
  }

  useAuthenticatedAutoLoad(status.enabled ? accessToken : "", loadWorkspace);

  if (!status.enabled) return <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Next Player Editor is disabled</h2><p style={{ color: "#475569" }}>{status.warnings?.[0] || "Enable the Player Editor pilot flag on FastAPI."}</p></article>;

  return <section style={{ display: "grid", gap: "1rem" }}>
    <InteractionDialog
      open={Boolean(resultDialog)}
      phase="success"
      title="Player update complete"
      onRequestClose={() => setResultDialog(null)}
      actions={<button type="button" onClick={() => setResultDialog(null)} style={buttonStyle}>OK</button>}
    >
      <div tabIndex={-1} data-dialog-focus>
        <StaticActionFeedback tone="success" title="Player saved" description={resultDialog || "The player update was saved."} />
      </div>
    </InteractionDialog>
    {message ? (
      <div
        role={isErrorMessage(message) ? "alert" : "status"}
        aria-live="polite"
        style={{
          position: "sticky",
          top: "0.75rem",
          zIndex: 10,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          gap: "0.75rem",
          border: `1px solid ${isErrorMessage(message) ? "#fca5a5" : "#86efac"}`,
          borderRadius: "12px",
          padding: "0.75rem",
          background: isErrorMessage(message) ? "#fef2f2" : "#f0fdf4",
          color: isErrorMessage(message) ? "#b91c1c" : "#166534",
          boxShadow: "0 4px 14px rgba(15, 23, 42, 0.12)"
        }}
      >
        <span>{message}</span>
        <button type="button" onClick={() => setMessage(null)} aria-label="Dismiss Player Editor message">Dismiss</button>
      </div>
    ) : null}
    {writeRecovery ? (
      <article aria-live="polite" style={{ ...cardStyle, background: "#fffbeb", borderColor: "#f59e0b" }}>
        <h2 style={{ marginTop: 0 }}>Player write needs exact-operation recovery</h2>
        <p style={{ color: "#92400e" }}>The response was not conclusive. Creating or changing another player or league rating stays blocked until this exact operation is reconciled.</p>
        <p><strong>Operation key:</strong> <code style={{ overflowWrap: "anywhere" }}>{writeRecovery.operationKey}</code><br /><strong>Last known status:</strong> {writeRecovery.status.replace(/_/g, " ")}</p>
        <p>{writeRecovery.message}</p>
        <button type="button" onClick={() => void reconcilePlayerWrite()} disabled={checkingWriteRecovery || !accessToken} style={ghostButtonStyle}>
          {checkingWriteRecovery ? "Checking and reconciling…" : "Check and reconcile exact operation"}
        </button>
      </article>
    ) : null}
    {mergeAttempted && mergeOperationId && !mergePreview && !mergeRecovery ? <article style={{ ...cardStyle, background: "#fef2f2", borderColor: "#fca5a5" }}><h2 style={{ marginTop: 0 }}>Merge outcome unknown</h2><p>Do not retry with a new operation. Check operation <code>{mergeOperationId}</code> first; an idempotent server retry uses this same ID.</p><button type="button" onClick={lookupMergeOperation} disabled={saving || !accessToken} style={ghostButtonStyle}>Check merge operation</button></article> : null}
    <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Player Editor admin session</h2><p style={{ color: "#475569" }}>This route supports roster/detail read, add player, basic player updates, guarded league-rating edits, Club Social identity linking, and guarded player merge. Merges require Replay History after execution.</p><div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: accessToken ? "#f0fdf4" : "#fffbeb", marginBottom: "1rem" }}><strong>{accessToken ? `Admin session: ${adminSessionLabel(session)}` : "Admin session required"}</strong><p style={{ margin: "0.35rem 0 0", color: accessToken ? "#166534" : "#92400e" }}>{accessToken ? "Ready to send authorized Player Editor requests." : sessionLoading ? "Checking admin session…" : "Sign in before using the Player Editor."}</p>{sessionMessage ? <p style={{ color: "#b91c1c", marginBottom: 0 }}>{sessionMessage}</p> : null}{!accessToken && !sessionLoading ? <p style={{ marginBottom: 0 }}><Link href="/admin/login">Open admin login</Link></p> : null}</div><button type="button" onClick={loadWorkspace} disabled={saving || !accessToken} style={buttonStyle}>{saving ? "Refreshing…" : "Refresh players and identities"}</button>{status.warnings?.length ? <ul style={{ color: "#92400e" }}>{status.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}</article>
    <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Add new player</h2><div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", alignItems: "end" }}><label><strong>Name</strong><br /><input value={newName} onChange={(event) => setNewName(event.target.value)} style={inputStyle} /></label><label><strong>Starting JUPR</strong><br /><input value={newStartingJupr} onChange={(event) => setNewStartingJupr(event.target.value)} type="number" min={1} max={7} step={0.1} style={inputStyle} /></label><button type="button" onClick={createPlayer} disabled={saving || !accessToken || Boolean(writeRecovery)} style={ghostButtonStyle}>Add player</button></div></article>
    <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Select player</h2><select value={selectedId} onChange={(event) => loadDetail(event.target.value)} style={inputStyle} disabled={saving || !accessToken}><option value="">Choose a player</option>{players.map((player) => <option key={player.id} value={String(player.id)}>{playerOptionLabel(player)}</option>)}</select></article>
    {detail ? <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Manage player</h2><div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.75rem" }}><label><strong>Name</strong><br /><input value={editName} onChange={(event) => setEditName(event.target.value)} style={inputStyle} /></label><label><strong>Overall JUPR</strong><br /><input value={editRating} onChange={(event) => setEditRating(event.target.value)} type="number" min={1} max={7} step={0.01} style={inputStyle} /></label><label><strong>Starting JUPR</strong><br /><input value={editStartingRating} onChange={(event) => setEditStartingRating(event.target.value)} type="number" min={1} max={7} step={0.01} style={inputStyle} /></label><label><strong>Active</strong><br /><select value={editActive ? "yes" : "no"} onChange={(event) => setEditActive(event.target.value === "yes")} style={inputStyle}><option value="yes">Active</option><option value="no">Inactive</option></select></label></div><p><button type="button" onClick={savePlayer} disabled={saving || !accessToken || Boolean(writeRecovery)} style={buttonStyle}>{saving ? "Saving…" : "Save player"}</button></p><div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "0.75rem", marginTop: "1rem" }}><div><strong>Wins</strong><br />{detail.player.wins ?? 0}</div><div><strong>Losses</strong><br />{detail.player.losses ?? 0}</div><div><strong>Matches</strong><br />{detail.player.matches_played ?? 0}</div><div><strong>Match refs</strong><br />{detail.match_reference_counts?.total ?? 0}</div></div></article> : null}
    {detail?.league_ratings?.length ? (
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>League ratings</h2>
        <p style={{ color: "#475569" }}>League-rating edits are audit-flagged because they can diverge from replayed history. Use them for targeted corrections only, then run Replay History if needed.</p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label><strong>League rating row</strong><br /><select value={selectedLeagueRatingId} onChange={(event) => seedLeagueRatingForm(detail.league_ratings.find((row) => String(row.id) === event.target.value) || null)} style={inputStyle}><option value="">Choose league rating…</option>{detail.league_ratings.map((row) => <option key={row.id} value={String(row.id)}>{row.league_name} · {juprLabel(row.rating_jupr)}</option>)}</select></label>
          <label><strong>League JUPR</strong><br /><input value={editLeagueRating} onChange={(event) => setEditLeagueRating(event.target.value)} type="number" min={1} max={7} step={0.01} style={inputStyle} /></label>
          <label><strong>League starting JUPR</strong><br /><input value={editLeagueStartingRating} onChange={(event) => setEditLeagueStartingRating(event.target.value)} type="number" min={1} max={7} step={0.01} style={inputStyle} /></label>
          <label><strong>League active</strong><br /><select value={editLeagueActive ? "yes" : "no"} onChange={(event) => setEditLeagueActive(event.target.value === "yes")} style={inputStyle}><option value="yes">Active</option><option value="no">Inactive</option></select></label>
        </div>
        <p>
          <ConfirmAction
            triggerLabel="Save league rating"
            title="Save this league rating correction?"
            description="This targeted correction can diverge from replayed history and will be audit-flagged. Run Replay History afterward if derived history must be rebuilt."
            confirmLabel="Yes, save league rating"
            confirmationText={leagueRatingConfirmText}
            disabled={saving || !accessToken || !selectedLeagueRatingId || Boolean(writeRecovery)}
            busy={saving}
            onConfirm={saveLeagueRating}
          />
        </p>
      </article>
    ) : null}
    <article style={cardStyle}>
      <h2 style={{ marginTop: 0 }}>Club Social identity linking</h2>
      <p style={{ color: "#475569" }}>Link social-only Club Social identities to official players so future social rows resolve consistently. This does not rewrite rated match history.</p>
      <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
        <button type="button" onClick={() => void loadSocialIdentities()} disabled={saving || !accessToken} style={ghostButtonStyle}>Refresh social identities</button>
        <ConfirmAction
          triggerLabel="Auto-link exact names"
          title="Auto-link exact Club Social names?"
          description="This will link only exact Club Social identity names to official players. Rated match history is not rewritten."
          confirmLabel="Yes, auto-link exact names"
          confirmationText={socialConfirmText}
          disabled={saving || !accessToken}
          busy={saving}
          onConfirm={autoLinkSocialIdentities}
        />
      </p>
      {socialPeople.length ? (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
            <label><strong>Social identity</strong><br /><select value={selectedSocialId} onChange={(event) => seedSocialForm(socialPeople.find((row) => row.id === event.target.value) || null)} style={inputStyle}><option value="">Choose identity…</option>{socialPeople.map((row) => <option key={row.id} value={row.id}>{socialLabel(row)}</option>)}</select></label>
            <label><strong>Display name</strong><br /><input value={socialDisplayName} onChange={(event) => setSocialDisplayName(event.target.value)} style={inputStyle} /></label>
            <label><strong>Linked player</strong><br /><select value={socialLinkedPlayerId} onChange={(event) => setSocialLinkedPlayerId(event.target.value)} style={inputStyle}><option value="">Unlinked</option>{socialPlayers.map((player) => <option key={player.id} value={String(player.id)}>{player.name} #{player.id}{player.active === false ? " [inactive]" : ""}</option>)}</select></label>
          </div>
          <p>
            <ConfirmAction
              triggerLabel="Save social link"
              title="Save this Club Social identity link?"
              description={<>This will link <strong>{socialDisplayName || "the selected social identity"}</strong> to {socialLinkedPlayerId ? `player #${socialLinkedPlayerId}` : "no official player"}. Rated match history is not rewritten.</>}
              confirmLabel="Yes, save social link"
              confirmationText={socialConfirmText}
              disabled={saving || !accessToken || !selectedSocialId}
              busy={saving}
              onConfirm={saveSocialIdentity}
            />
          </p>
        </>
      ) : <p style={{ color: "#64748b" }}>{saving ? "Loading Club Social identities…" : "No Club Social identities are available."}</p>}
    </article>
    <article style={{ ...cardStyle, background: "#fff7ed", borderColor: "#fed7aa" }}>
      <h2 style={{ marginTop: 0 }}>Merge player accounts</h2>
      <p style={{ color: "#7c2d12" }}>Merge rewires Source → Target in one guarded database transaction, records a recovery operation, then deactivates the source. A succeeded full Replay History job must be attached afterward.</p>
      {status.transactional_merge_ready === false ? <p style={{ color: "#b91c1c" }}><strong>Merge write unavailable:</strong> FastAPI has not confirmed its server-only service role and transaction contract.</p> : null}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
        <label><strong>Source duplicate</strong><br /><select value={mergeSourceId} onChange={(event) => { setMergeSourceId(event.target.value); setMergePreview(null); }} style={inputStyle}><option value="">Choose source…</option>{players.map((player) => <option key={player.id} value={String(player.id)}>{playerOptionLabel(player)}</option>)}</select></label>
        <label><strong>Target keeper</strong><br /><select value={mergeTargetId} onChange={(event) => { setMergeTargetId(event.target.value); setMergePreview(null); }} style={inputStyle}><option value="">Choose target…</option>{players.map((player) => <option key={player.id} value={String(player.id)}>{playerOptionLabel(player)}</option>)}</select></label>
        <button type="button" onClick={previewMerge} disabled={saving || !accessToken || !mergeSourceId || !mergeTargetId || mergeSourceId === mergeTargetId} style={ghostButtonStyle}>Preview merge</button>
      </div>
      {mergePreview ? (
        <div style={{ marginTop: "0.75rem", border: "1px solid #fed7aa", borderRadius: "12px", padding: "0.75rem", background: "white" }}>
          <strong>Preview · {mergePreview.can_merge === false ? "blocked" : "ready"}</strong>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.75rem", marginTop: "0.5rem" }}>
            <div>Match refs<br /><strong>{mergePreview.match_reference_counts?.total ?? 0}</strong></div>
            <div>League rows to move<br /><strong>{mergePreview.league_rating_plan?.move_ids?.length ?? 0}</strong></div>
            <div>Conflicting league rows to delete<br /><strong>{mergePreview.league_rating_plan?.delete_ids?.length ?? 0}</strong></div>
            <div>Source social links<br /><strong>{mergePreview.social_identity_counts?.source_linked ?? 0}</strong></div>
          </div>
          {mergePreview.collision_match_ids?.length ? <p style={{ color: "#b91c1c" }}>Blocked by match collision(s): {mergePreview.collision_match_ids.join(", ")}</p> : null}
          {mergePreview.league_rating_plan?.conflicts?.length ? <p style={{ color: "#92400e" }}>League conflicts: {mergePreview.league_rating_plan.conflicts.join(", ")}</p> : null}
          {mergePreview.warnings?.length ? <ul style={{ color: "#92400e" }}>{mergePreview.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
          <p>
            <ConfirmAction
              triggerLabel="Execute atomic merge"
              title="Merge these player accounts?"
              description={<>This will permanently rewire player #{mergeSourceId} into player #{mergeTargetId}, delete any reviewed conflicting league rows, deactivate the source, and require a succeeded full replay.</>}
              confirmLabel="Yes, merge accounts"
              confirmationText={mergeConfirmText}
              tone="danger"
              disabled={saving || !accessToken || status.transactional_merge_ready === false || mergePreview.can_merge === false || !mergePreview.preview_fingerprint}
              busy={saving}
              onConfirm={executeMerge}
            />
          </p>
        </div>
      ) : null}
    </article>
    {mergeRecovery?.operation_id ? (
      <article style={{ ...cardStyle, background: mergeRecovery.operation_status === "merged_pending_replay" ? "#fffbeb" : "#f0fdf4", borderColor: mergeRecovery.operation_status === "merged_pending_replay" ? "#f59e0b" : "#86efac" }}>
        <h2 style={{ marginTop: 0 }}>Merge recovery</h2>
        <p><strong>Operation:</strong> <code>{mergeRecovery.operation_id}</code><br /><strong>Status:</strong> {mergeRecovery.operation_status || mergeRecovery.recovery?.status || "unknown"}</p>
        {mergeRecovery.operation_status === "merged_pending_replay" ? (
          <>
            <p>{mergeRecovery.recovery?.operator_rule || "Run Replay History ALL, then attach its succeeded job ID."} <Link href={mergeRecovery.recovery?.replay_route || "/admin/replay-history"}>Open Replay History</Link>{mergeRecovery.recovery?.tracked_replay_fallback_url ? <> · <a href={mergeRecovery.recovery.tracked_replay_fallback_url}>Open tracked Streamlit fallback</a></> : null}</p>
            <label><strong>Succeeded replay job UUID</strong><br /><input value={replayJobId} onChange={(event) => setReplayJobId(event.target.value)} style={inputStyle} /></label>
            <p>
              <ConfirmAction
                triggerLabel="Attach replay evidence"
                title="Attach this replay job as merge recovery evidence?"
                description={<>This will attach succeeded replay job <code>{replayJobId.trim() || "not entered"}</code> to merge operation <code>{mergeRecovery.operation_id}</code> and complete its recovery record.</>}
                confirmLabel="Yes, attach replay evidence"
                confirmationText={replayEvidenceConfirmText}
                disabled={saving || !replayJobId.trim()}
                busy={saving}
                onConfirm={attachReplayEvidence}
              />
            </p>
            <details>
              <summary>Emergency pre-replay compensation</summary>
              <p style={{ color: "#92400e" }}>Use only before replay and only when the merge must be reversed. The server refuses stale compensation.</p>
              <p>
                <ConfirmAction
                  triggerLabel="Compensate merge"
                  title="Reverse this merge before replay?"
                  description={<>This emergency action attempts to restore the pre-merge player, match, league, and social-link state for operation <code>{mergeRecovery.operation_id}</code>. The server refuses stale compensation.</>}
                  confirmLabel="Yes, compensate merge"
                  confirmationText={compensateConfirmText}
                  tone="danger"
                  disabled={saving}
                  busy={saving}
                  onConfirm={compensateMerge}
                />
              </p>
            </details>
          </>
        ) : <p>Recovery is complete for this operation. No replay evidence or compensation action remains.</p>}
      </article>
    ) : null}
  </section>;
}
