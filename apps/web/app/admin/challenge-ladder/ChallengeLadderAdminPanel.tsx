"use client";

import { useRef, useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";
import { deriveLiveLadderOperationKey, idempotencyKeyFor, rotateIdempotencyKey } from "@/lib/liveLadderOperations";

type Challenge = { id: number; tier_id: string; status: string; bucket: string; challenger_id?: number | null; defender_id?: number | null; challenger_name: string; defender_name: string; winner_name?: string | null; created_at?: string | null; accept_by?: string | null; play_by?: string | null; resolution_notes?: string | null };
type Player = { player_id: number; player_name: string; rank?: number; status?: string; rating_jupr?: number | null };
type RosterRow = Player & { tier_id: string; is_active: boolean; joined_at?: string | null; left_at?: string | null; notes?: string | null };
type PlayerFlag = { player_id: number; player_name: string; vacation_until?: string | null; reinstate_required: boolean; reinstate_notes?: string | null; tier_move_flag?: boolean; tier_move_dest_tier?: string | null; tier_move_count?: number };
type Tier = { tier_id: string; label: string; range: string; players: Player[] };
type ChallengeNotice = { email_full: string; sms: string };
type StatusResponse = { enabled: boolean; writes_enabled?: boolean; status: string; summary?: Record<string, number>; warnings?: string[] };
type DashboardResponse = { ok: boolean; state_version: string; authority?: string; summary: Record<string, number>; settings: Record<string, unknown>; settings_row?: Record<string, unknown>; tiers: Tier[]; challenges: Challenge[]; bucket_counts: Record<string, number>; player_options?: Player[]; roster_rows?: RosterRow[]; player_flags?: PlayerFlag[] };
type Recovery = { match_log_url?: string; replay_history_url?: string; instructions?: string };
type ActionResponse = { ok: boolean; operation_key?: string; idempotent_replay?: boolean; recovery?: Recovery; correction?: Recovery; challenge?: Challenge; roster?: RosterRow | RosterRow[]; player_flags?: PlayerFlag; notice?: ChallengeNotice; warnings?: string[]; rank_result?: Record<string, unknown>; official_matches?: Record<string, unknown>; preview?: Record<string, unknown> };
type ResultDraft = { challenge_id: string; a_chal: string; a_def: string; b_chal: string; b_def: string; match_a_games: string; match_b_games: string; match_date: string; winner_override: string; publish_official_matches: boolean };
type ResultPreviewResponse = ActionResponse & { mode: "challenge_ladder_result_preview"; preview_fingerprint: string; challenge: Challenge; preview: { final_winner_side: string; final_winner_id: number; winner_summary: Record<string, string | number>; scores: Record<string, unknown> }; partner_names: Record<string, string>; match_date: string; would_publish_official_matches: boolean; rank_result: { would_swap: boolean; reason: string } };
type TierMovementTrigger = { player_id: number; player_name: string; current_tier: string; destination_tier: string; consecutive_match_count: number; latest_match_at?: string | null };
type TierMovementResponse = { ok: boolean; mode: "challenge_ladder_tier_movement_review"; summary: { evaluated_player_count: number; match_count: number; trigger_count: number; required_consecutive_matches: number }; triggers: TierMovementTrigger[] };
type TierRosterPreviewPlayer = { rank: number; player_id: number; player_name: string; previous_tier?: string | null; previous_rank?: number | null; change?: string };
type TierRosterReplacePreview = { ok: boolean; mode: "challenge_ladder_roster_replace_preview"; tier_id: string; can_apply: boolean; preview_fingerprint: string; summary: Record<string, number>; current_roster: TierRosterPreviewPlayer[]; proposed_roster: TierRosterPreviewPlayer[]; removed_players: TierRosterPreviewPlayer[]; moved_from_other_tiers: TierRosterPreviewPlayer[]; source_tier_recompressions: Array<{ tier_id: string; player_id: number; player_name: string; old_rank: number; new_rank: number }>; open_challenge_blockers: Array<{ challenge_id: number; status: string; affected_player_names: string[] }>; warnings: string[] };

type Props = { apiBase: string | null; clubId: string; status: StatusResponse | null };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string { return `${apiBase.replace(/\/$/, "")}${path}`; }
async function apiError(response: Response): Promise<string> { const text = await response.text().catch(() => ""); try { return String((JSON.parse(text) as { detail?: unknown }).detail || text); } catch { return text || `API error (${response.status}).`; } }
function mutationError(error: unknown, fallback: string): string { const detail = error instanceof Error ? error.message : fallback; return `${detail} If the response was lost, the outcome may be uncertain; reconcile the operation before attempting another write.`; }
function Pre({ value }: { value: unknown }) { return <pre style={{ whiteSpace: "pre-wrap", background: "#0f172a", color: "white", padding: "1rem", borderRadius: "12px", overflowX: "auto", fontSize: "0.82rem" }}>{JSON.stringify(value, null, 2)}</pre>; }
function parseGames(raw: string): number[][] { return raw.split(/[|,;]/).map((part) => part.trim()).filter(Boolean).map((part) => { const bits = part.split(/[-–—:\/]/).map((x) => Number(x.trim())); if (bits.length !== 2 || !Number.isFinite(bits[0]) || !Number.isFinite(bits[1])) throw new Error(`Invalid score: ${part}`); return [bits[0], bits[1]]; }); }
function activePlayers(tiers: Tier[] | undefined): Player[] { const rows: Player[] = []; for (const tier of tiers || []) for (const player of tier.players || []) rows.push(player); return rows; }
function resultDraftFingerprint(draft: ResultDraft): string { return JSON.stringify(draft); }
function tierRosterDraftFingerprint(draft: { tier_id: string; ranked_names: string }): string { return JSON.stringify({ tier_id: draft.tier_id, ranked_names: draft.ranked_names }); }
function resultPayload(draft: ResultDraft, confirmationText?: string): Record<string, unknown> {
  return {
    partner_a_challenger_id: Number(draft.a_chal), partner_a_defender_id: Number(draft.a_def), partner_b_challenger_id: Number(draft.b_chal), partner_b_defender_id: Number(draft.b_def),
    match_a_games: parseGames(draft.match_a_games), match_b_games: parseGames(draft.match_b_games), match_date: draft.match_date, winner_override: draft.winner_override, publish_official_matches: draft.publish_official_matches,
    ...(confirmationText ? { confirmation_text: confirmationText } : {}),
  };
}

const RECORDABLE_STATUSES = new Set(["ACCEPTED_SCHEDULING", "ACCEPTED", "IN_PROGRESS", "AWAITING_VERIFICATION", "OVERDUE_PLAY"]);
const OPEN_STATUSES = new Set(["PENDING_ACCEPTANCE", ...RECORDABLE_STATUSES]);

export default function ChallengeLadderAdminPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const operationKeys = useRef<Record<string, string>>({});
  const [dashboard, setDashboard] = useState<DashboardResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [notes, setNotes] = useState<Record<number, string>>({});
  const [createDraft, setCreateDraft] = useState({ challenger_id: "", defender_id: "", tier_id: "ADV", challenger_contact: "", ledger_ref: "", override: false, start_clock: false });
  const [lastNotice, setLastNotice] = useState<ChallengeNotice | null>(null);
  const [resultDraft, setResultDraft] = useState<ResultDraft>({ challenge_id: "", a_chal: "", a_def: "", b_chal: "", b_def: "", match_a_games: "11-0,11-0", match_b_games: "11-0,11-0", match_date: new Date().toISOString(), winner_override: "computed", publish_official_matches: true });
  const [resultPreview, setResultPreview] = useState<ResultPreviewResponse | null>(null);
  const [previewedDraftFingerprint, setPreviewedDraftFingerprint] = useState<string | null>(null);
  const [forfeitDraft, setForfeitDraft] = useState({ challenge_id: "", forfeited_by_id: "", admin_note: "" });
  const [passDraft, setPassDraft] = useState({ challenge_id: "", player_id: "" });
  const [rosterAddDraft, setRosterAddDraft] = useState({ player_id: "", tier_id: "ADV", admin_note: "" });
  const [rosterMoveDraft, setRosterMoveDraft] = useState({ player_id: "", destination_tier: "INT", recompress_old: true, admin_note: "" });
  const [rosterReplaceDraft, setRosterReplaceDraft] = useState({ tier_id: "ADV", ranked_names: "", admin_note: "" });
  const [rosterReplacePreview, setRosterReplacePreview] = useState<TierRosterReplacePreview | null>(null);
  const [previewedRosterDraftFingerprint, setPreviewedRosterDraftFingerprint] = useState<string | null>(null);
  const [overrideDraft, setOverrideDraft] = useState({ player_id: "", vacation_until: "", reinstate_required: false, reinstate_notes: "" });
  const [tierMovementReview, setTierMovementReview] = useState<TierMovementResponse | null>(null);
  const [lastResult, setLastResult] = useState<ActionResponse | null>(null);
  const [lastOperationKey, setLastOperationKey] = useState("");

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("Missing JUPR API base URL.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using Challenge Ladder Admin.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    if (!response.ok) throw new Error(await apiError(response));
    return (await response.json()) as T;
  }

  async function durableFields(scope: string, operationType: string, entityId: string) {
    if (status?.writes_enabled !== true) throw new Error("Next Challenge Ladder writes are guarded off; use the Streamlit fallback.");
    if (!dashboard?.state_version) throw new Error("Load the authoritative Python dashboard before writing.");
    const idempotencyKey = idempotencyKeyFor(operationKeys.current, scope);
    const operationKey = await deriveLiveLadderOperationKey({ clubId, surface: "challenge_ladder", operationType, entityId, idempotencyKey });
    setLastOperationKey(operationKey);
    return { expected_version: dashboard.state_version, idempotency_key: idempotencyKey };
  }

  function completeScope(scope: string) { rotateIdempotencyKey(operationKeys.current, scope); }

  async function loadDashboard() {
    setBusy(true); setMessage(null);
    try { const payload = await requestJson<DashboardResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/challenge-ladder/dashboard`); setDashboard(payload); setMessage("Challenge Ladder dashboard loaded."); }
    catch (error) { setMessage(error instanceof Error ? error.message : "Unable to load dashboard."); }
    finally { setBusy(false); }
  }

  async function loadTierMovementReview() {
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<TierMovementResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/challenge-ladder/tier-movement-review`);
      setTierMovementReview(payload); setMessage(payload.triggers.length ? `${payload.triggers.length} tier-movement review item${payload.triggers.length === 1 ? "" : "s"} found.` : "No tier-movement triggers found.");
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to load tier-movement review."); }
    finally { setBusy(false); }
  }

  function prepareReviewedTierMove(trigger: TierMovementTrigger) {
    setRosterMoveDraft({ player_id: String(trigger.player_id), destination_tier: trigger.destination_tier, recompress_old: true, admin_note: `Tier movement review: ${trigger.consecutive_match_count} consecutive matches toward ${trigger.destination_tier}.` });
    setMessage(`Prepared the guarded move draft for ${trigger.player_name}. Review it below before applying.`);
  }

  async function copyNotice(value: string, label: string) {
    try { await navigator.clipboard.writeText(value); setMessage(`${label} copied.`); }
    catch { setMessage(`Unable to copy ${label.toLowerCase()}; select the text manually.`); }
  }

  async function updateChallenge(challenge: Challenge, nextStatus: string, confirmationText: string) {
    setBusy(true); setMessage(null);
    try {
      const scope = `update:${challenge.id}:${nextStatus}`;
      const fields = await durableFields(scope, "update_challenge", String(challenge.id));
      const payload = await requestJson<ActionResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/challenge-ladder/challenges/${challenge.id}`, { method: "PATCH", body: JSON.stringify({ status: nextStatus, admin_note: notes[challenge.id] || "", confirmation_text: confirmationText, ...fields }) });
      completeScope(scope); setLastResult(payload); setMessage(`Challenge #${challenge.id} saved as ${nextStatus}.`); await loadDashboard();
    } catch (error) { setMessage(mutationError(error, "Unable to update challenge.")); }
    finally { setBusy(false); }
  }

  async function simpleAction(challenge: Challenge, action: "start-clock" | "accept", confirmationText: string) {
    setBusy(true); setMessage(null);
    try {
      const operationType = action === "start-clock" ? "start_clock" : "accept_challenge";
      const scope = `${operationType}:${challenge.id}`;
      const fields = await durableFields(scope, operationType, String(challenge.id));
      const payload = await requestJson<ActionResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/challenge-ladder/challenges/${challenge.id}/${action}`, { method: "POST", body: JSON.stringify({ confirmation_text: confirmationText, ...fields }) });
      completeScope(scope); setLastResult(payload); setMessage(`Challenge #${challenge.id} updated.`); await loadDashboard();
    } catch (error) { setMessage(mutationError(error, "Unable to update challenge.")); }
    finally { setBusy(false); }
  }

  async function createChallenge(confirmationText: string) {
    setBusy(true); setMessage(null);
    try {
      const entityId = `${createDraft.challenger_id}:${createDraft.defender_id}:${createDraft.tier_id}`;
      const scope = `create:${entityId}`;
      const fields = await durableFields(scope, "create_challenge", entityId);
      const payload = await requestJson<ActionResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/challenge-ladder/challenges`, { method: "POST", body: JSON.stringify({ ...createDraft, challenger_id: Number(createDraft.challenger_id), defender_id: Number(createDraft.defender_id), confirmation_text: confirmationText, ...fields }) });
      completeScope(scope); setLastResult(payload); setLastNotice(payload.notice || null); setMessage("Challenge created. Copy the notice, send it, then start the acceptance clock."); await loadDashboard();
    } catch (error) { setMessage(mutationError(error, "Unable to create challenge.")); }
    finally { setBusy(false); }
  }

  async function recordForfeit(confirmationText: string) {
    setBusy(true); setMessage(null);
    try {
      const scope = `forfeit:${forfeitDraft.challenge_id}`;
      const fields = await durableFields(scope, "record_forfeit", forfeitDraft.challenge_id);
      const payload = await requestJson<ActionResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/challenge-ladder/challenges/${Number(forfeitDraft.challenge_id)}/forfeit`, { method: "POST", body: JSON.stringify({ forfeited_by_id: Number(forfeitDraft.forfeited_by_id), admin_note: forfeitDraft.admin_note, confirmation_text: confirmationText, ...fields }) });
      completeScope(scope); setLastResult(payload); setMessage("Forfeit recorded."); await loadDashboard();
    } catch (error) { setMessage(mutationError(error, "Unable to record forfeit.")); }
    finally { setBusy(false); }
  }

  async function recordPass(confirmationText: string) {
    setBusy(true); setMessage(null);
    try {
      const scope = `pass:${passDraft.challenge_id}`;
      const fields = await durableFields(scope, "record_pass", passDraft.challenge_id);
      const payload = await requestJson<ActionResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/challenge-ladder/challenges/${Number(passDraft.challenge_id)}/pass`, { method: "POST", body: JSON.stringify({ player_id: Number(passDraft.player_id), confirmation_text: confirmationText, ...fields }) });
      completeScope(scope); setLastResult(payload); setPassDraft({ challenge_id: "", player_id: "" }); setMessage("Monthly pass recorded and challenge closed."); await loadDashboard();
    } catch (error) { setMessage(mutationError(error, "Unable to record monthly pass.")); }
    finally { setBusy(false); }
  }

  async function addRosterPlayer(confirmationText: string) {
    setBusy(true); setMessage(null);
    try {
      const scope = `roster-add:${rosterAddDraft.player_id}`;
      const fields = await durableFields(scope, "add_roster_player", rosterAddDraft.player_id);
      const payload = await requestJson<ActionResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/challenge-ladder/roster`, { method: "POST", body: JSON.stringify({ ...rosterAddDraft, player_id: Number(rosterAddDraft.player_id), confirmation_text: confirmationText, ...fields }) });
      completeScope(scope); setLastResult(payload); setRosterAddDraft((current) => ({ ...current, player_id: "", admin_note: "" })); setMessage("Ladder player added at the bottom of the selected tier."); await loadDashboard();
    } catch (error) { setMessage(mutationError(error, "Unable to add ladder player.")); }
    finally { setBusy(false); }
  }

  async function moveRosterPlayer(confirmationText: string) {
    setBusy(true); setMessage(null);
    try {
      const scope = `roster-move:${rosterMoveDraft.player_id}:${rosterMoveDraft.destination_tier}`;
      const fields = await durableFields(scope, "move_roster_player", rosterMoveDraft.player_id);
      const payload = await requestJson<ActionResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/challenge-ladder/roster/${Number(rosterMoveDraft.player_id)}/move`, { method: "POST", body: JSON.stringify({ destination_tier: rosterMoveDraft.destination_tier, recompress_old: rosterMoveDraft.recompress_old, admin_note: rosterMoveDraft.admin_note, confirmation_text: confirmationText, ...fields }) });
      completeScope(scope); setLastResult(payload); setRosterMoveDraft((current) => ({ ...current, player_id: "", admin_note: "" })); setMessage("Ladder player moved to the bottom of the destination tier."); await loadDashboard();
    } catch (error) { setMessage(mutationError(error, "Unable to move ladder player.")); }
    finally { setBusy(false); }
  }

  function loadCurrentTierRosterForReplacement() {
    const names = (dashboard?.roster_rows || [])
      .filter((row) => row.is_active && row.tier_id === rosterReplaceDraft.tier_id)
      .sort((a, b) => (a.rank ?? 999999) - (b.rank ?? 999999))
      .map((row) => row.player_name)
      .join("\n");
    setRosterReplaceDraft((current) => ({ ...current, ranked_names: names }));
    setRosterReplacePreview(null); setPreviewedRosterDraftFingerprint(null);
    setMessage(names ? "Loaded the current tier order. Edit it, then preview the complete replacement." : "This tier currently has no active roster players.");
  }

  async function previewRosterReplacement() {
    const rankedNames = rosterReplaceDraft.ranked_names.split("\n").map((name) => name.trim()).filter(Boolean);
    if (!rankedNames.length) { setMessage("Paste at least one player name before previewing the tier replacement."); return; }
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<TierRosterReplacePreview>(`/admin/clubs/${encodeURIComponent(clubId)}/challenge-ladder/roster/replace-tier/preview`, { method: "POST", body: JSON.stringify({ tier_id: rosterReplaceDraft.tier_id, ranked_names: rankedNames }) });
      setRosterReplacePreview(payload); setPreviewedRosterDraftFingerprint(tierRosterDraftFingerprint(rosterReplaceDraft));
      setMessage(payload.can_apply ? "Tier replacement preview is ready. Review every change before applying." : "Preview found blockers. Resolve them and preview again before applying.");
    } catch (error) {
      setRosterReplacePreview(null); setPreviewedRosterDraftFingerprint(null); setMessage(error instanceof Error ? error.message : "Unable to preview the tier replacement.");
    } finally { setBusy(false); }
  }

  async function applyRosterReplacement(confirmationText: string) {
    if (!rosterReplacePreview || previewedRosterDraftFingerprint !== tierRosterDraftFingerprint(rosterReplaceDraft)) { setMessage("Preview the current ranked list before applying it."); return; }
    if (!rosterReplacePreview.can_apply) { setMessage("Resolve open challenge blockers and preview the roster again before applying it."); return; }
    setBusy(true); setMessage(null);
    try {
      const scope = `roster-replace:${rosterReplaceDraft.tier_id}`;
      const fields = await durableFields(scope, "replace_tier_roster", rosterReplaceDraft.tier_id);
      const payload = await requestJson<ActionResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/challenge-ladder/roster/replace-tier`, { method: "POST", body: JSON.stringify({ tier_id: rosterReplaceDraft.tier_id, ranked_player_ids: rosterReplacePreview.proposed_roster.map((row) => row.player_id), preview_fingerprint: rosterReplacePreview.preview_fingerprint, admin_note: rosterReplaceDraft.admin_note, confirmation_text: confirmationText, ...fields }) });
      completeScope(scope); setLastResult(payload); setRosterReplacePreview(null); setPreviewedRosterDraftFingerprint(null); setRosterReplaceDraft((current) => ({ ...current, admin_note: "" })); setMessage("Reviewed tier roster replacement applied and audited."); await loadDashboard();
    } catch (error) { setMessage(mutationError(error, "Unable to apply the tier replacement.")); }
    finally { setBusy(false); }
  }

  async function savePlayerOverrides(confirmationText: string) {
    setBusy(true); setMessage(null);
    try {
      const scope = `overrides:${overrideDraft.player_id}`;
      const fields = await durableFields(scope, "save_player_overrides", overrideDraft.player_id);
      const payload = await requestJson<ActionResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/challenge-ladder/roster/${Number(overrideDraft.player_id)}/overrides`, { method: "PATCH", body: JSON.stringify({ vacation_until: overrideDraft.vacation_until.trim() || null, reinstate_required: overrideDraft.reinstate_required, reinstate_notes: overrideDraft.reinstate_notes, confirmation_text: confirmationText, ...fields }) });
      completeScope(scope); setLastResult(payload); setMessage("Vacation and reinstate overrides saved."); await loadDashboard();
    } catch (error) { setMessage(mutationError(error, "Unable to save ladder overrides.")); }
    finally { setBusy(false); }
  }

  async function previewResult() {
    if (!resultDraft.challenge_id) { setMessage("Choose a recordable challenge before previewing a result."); return; }
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<ResultPreviewResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/challenge-ladder/challenges/${Number(resultDraft.challenge_id)}/result/preview`, { method: "POST", body: JSON.stringify(resultPayload(resultDraft)) });
      setResultPreview(payload); setPreviewedDraftFingerprint(resultDraftFingerprint(resultDraft)); setMessage("Result preview is ready. Review it before publishing.");
    } catch (error) {
      setResultPreview(null); setPreviewedDraftFingerprint(null); setMessage(error instanceof Error ? error.message : "Unable to preview ladder result.");
    } finally { setBusy(false); }
  }

  async function publishResult(confirmationText: string) {
    if (!resultPreview || previewedDraftFingerprint !== resultDraftFingerprint(resultDraft)) { setMessage("Preview the current result draft before publishing it."); return; }
    setBusy(true); setMessage(null);
    try {
      const scope = `publish-result:${resultDraft.challenge_id}`;
      const fields = await durableFields(scope, "publish_result", resultDraft.challenge_id);
      const payload = await requestJson<ActionResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/challenge-ladder/challenges/${Number(resultDraft.challenge_id)}/result`, { method: "POST", body: JSON.stringify({ ...resultPayload(resultDraft, confirmationText), preview_fingerprint: resultPreview.preview_fingerprint, ...fields }) });
      completeScope(scope); setLastResult(payload); setResultPreview(null); setPreviewedDraftFingerprint(null); setMessage("Ladder result published."); await loadDashboard();
    } catch (error) { setMessage(mutationError(error, "Unable to publish ladder result.")); }
    finally { setBusy(false); }
  }

  async function reconcileLastOperation(confirmationText: string) {
    if (!lastOperationKey) { setMessage("No operation is available to reconcile."); return; }
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<ActionResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/challenge-ladder/operations/${encodeURIComponent(lastOperationKey)}/reconcile`, { method: "POST", body: JSON.stringify({ confirmation_text: confirmationText }) });
      setLastResult(payload); setMessage(payload.ok ? "Operation reconciled from the durable result." : "The operation is still uncertain. Inspect Match Log and Replay before any correction.");
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to reconcile the operation."); }
    finally { setBusy(false); }
  }

  if (!status?.enabled) return <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Challenge Ladder Admin is disabled</h2><p>{status?.warnings?.[0] || "Enable JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER on FastAPI."}</p></article>;

  const rosterPlayers = activePlayers(dashboard?.tiers);
  const createTierPlayers = dashboard?.tiers.find((tier) => tier.tier_id === createDraft.tier_id)?.players || [];
  const allPlayerOptions = dashboard?.player_options?.length ? dashboard.player_options : rosterPlayers;
  const openChallengeOptions = (dashboard?.challenges || []).filter((challenge) => OPEN_STATUSES.has(challenge.status));
  const recordableChallengeOptions = openChallengeOptions.filter((challenge) => RECORDABLE_STATUSES.has(challenge.status));
  const selectedResultChallenge = recordableChallengeOptions.find((challenge) => String(challenge.id) === resultDraft.challenge_id);
  const selectedForfeitChallenge = openChallengeOptions.find((challenge) => String(challenge.id) === forfeitDraft.challenge_id);
  const selectedPassChallenge = openChallengeOptions.find((challenge) => String(challenge.id) === passDraft.challenge_id);
  const activeRosterRows = (dashboard?.roster_rows || []).filter((row) => row.is_active);
  const inactiveRosterRows = (dashboard?.roster_rows || []).filter((row) => !row.is_active);
  const currentOverrideRows = (dashboard?.player_flags || []).filter((flags) => flags.vacation_until || flags.reinstate_required || flags.reinstate_notes);
  const activeRosterPlayerIds = new Set(activeRosterRows.map((row) => row.player_id));
  const addRosterOptions = allPlayerOptions.filter((player) => !activeRosterPlayerIds.has(player.player_id));
  const selectedMovePlayer = activeRosterRows.find((player) => String(player.player_id) === rosterMoveDraft.player_id);
  const partnerPlayers = allPlayerOptions.filter((player) => player.player_id !== selectedResultChallenge?.challenger_id && player.player_id !== selectedResultChallenge?.defender_id);
  const currentDraftIsPreviewed = Boolean(resultPreview && previewedDraftFingerprint === resultDraftFingerprint(resultDraft));
  const currentRosterReplacementIsPreviewed = Boolean(rosterReplacePreview && previewedRosterDraftFingerprint === tierRosterDraftFingerprint(rosterReplaceDraft));
  const writesGuarded = busy || status.writes_enabled !== true;
  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Admin session</h2><p style={{ color: "#475569" }}>{adminSessionLabel(session)}</p>{sessionLoading ? <p>Checking session…</p> : null}{sessionMessage ? <p>{sessionMessage}</p> : null}
        <p style={{ color: "#475569" }}>Players: {status.summary?.active_player_count ?? "—"} · Active challenges: {status.summary?.active_challenge_count ?? "—"}</p>
      </article>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Dashboard</h2>
        <button type="button" onClick={loadDashboard} disabled={busy || !accessToken} style={buttonStyle}>{busy ? "Loading…" : "Load Challenge Ladder"}</button>
        {status.writes_enabled !== true ? <p style={{ color: "#92400e", fontWeight: 700 }}>Next.js writes are guarded off. Use the Streamlit Challenge Ladder fallback for staging writes; this dashboard remains read-only.</p> : null}
        {message ? <p role="status" aria-live="polite" style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("type") || message.toLowerCase().includes("invalid") || message.toLowerCase().includes("uncertain") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
      </article>
      {dashboard ? <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Create challenge</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label>Tier<br /><select value={createDraft.tier_id} onChange={(e) => setCreateDraft((c) => ({ ...c, tier_id: e.target.value, challenger_id: "", defender_id: "" }))} style={inputStyle}>{dashboard.tiers.map((tier) => <option key={tier.tier_id} value={tier.tier_id}>{tier.label}</option>)}</select></label>
          <label>Challenger<br /><select value={createDraft.challenger_id} onChange={(e) => setCreateDraft((c) => ({ ...c, challenger_id: e.target.value }))} style={inputStyle}><option value="">Choose</option>{createTierPlayers.map((p) => <option key={p.player_id} value={p.player_id}>{p.player_name} · rank {p.rank ?? "—"}</option>)}</select></label>
          <label>Defender<br /><select value={createDraft.defender_id} onChange={(e) => setCreateDraft((c) => ({ ...c, defender_id: e.target.value }))} style={inputStyle}><option value="">Choose</option>{createTierPlayers.filter((player) => String(player.player_id) !== createDraft.challenger_id).map((p) => <option key={p.player_id} value={p.player_id}>{p.player_name} · rank {p.rank ?? "—"}</option>)}</select></label>
          <label>Challenger contact (optional)<br /><input value={createDraft.challenger_contact} onChange={(e) => setCreateDraft((c) => ({ ...c, challenger_contact: e.target.value }))} placeholder="Email, text, or WhatsApp" style={inputStyle} /></label>
          <label>Ledger/ref<br /><input value={createDraft.ledger_ref} onChange={(e) => setCreateDraft((c) => ({ ...c, ledger_ref: e.target.value }))} style={inputStyle} /></label>
          <label style={{ display: "inline-flex", gap: "0.5rem", alignItems: "center" }}><input type="checkbox" checked={createDraft.override} onChange={(e) => setCreateDraft((c) => ({ ...c, override: e.target.checked }))} /> Override eligibility</label>
          <label style={{ display: "inline-flex", gap: "0.5rem", alignItems: "center" }}><input type="checkbox" checked={createDraft.start_clock} onChange={(e) => setCreateDraft((c) => ({ ...c, start_clock: e.target.checked }))} /> Start clock now (notice already sent)</label>
          <ConfirmAction
            triggerLabel="Create challenge"
            title="Create this ladder challenge?"
            description="This creates the selected challenge and records its eligibility and clock settings."
            confirmLabel="Yes, create challenge"
            confirmationText="CREATE LADDER CHALLENGE"
            disabled={writesGuarded || !createDraft.challenger_id || !createDraft.defender_id}
            busy={busy}
            onConfirm={createChallenge}
          />
        </div>
      </article> : null}
      {lastNotice ? <article style={{ ...cardStyle, borderColor: "#86efac", background: "#f0fdf4" }}>
        <h2 style={{ marginTop: 0 }}>Copy/paste challenge notice</h2>
        <p style={{ color: "#475569" }}>The 48-hour window is based on the timestamp of the message you send. If you did not start the clock during creation, send the notice first and then use “Start clock” on the challenge below.</p>
        <label>Email<br /><textarea value={lastNotice.email_full} readOnly rows={14} style={{ ...inputStyle, resize: "vertical" }} /></label>
        <p><button type="button" onClick={() => copyNotice(lastNotice.email_full, "Email notice")} style={ghostButtonStyle}>Copy email notice</button></p>
        <label>Text/SMS<br /><textarea value={lastNotice.sms} readOnly rows={4} style={{ ...inputStyle, resize: "vertical" }} /></label>
        <p><button type="button" onClick={() => copyNotice(lastNotice.sms, "SMS notice")} style={ghostButtonStyle}>Copy SMS notice</button></p>
      </article> : null}
      {dashboard ? <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Publish played result</h2>
        <p style={{ color: "#475569" }}>Played ladder results insert two official rated matches and apply direct rank swap when the challenger wins. Forfeits do not create match rows.</p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label>Challenge<br /><select value={resultDraft.challenge_id} onChange={(e) => setResultDraft((c) => ({ ...c, challenge_id: e.target.value, a_chal: "", a_def: "", b_chal: "", b_def: "" }))} style={inputStyle}><option value="">Choose</option>{recordableChallengeOptions.map((ch) => <option key={ch.id} value={ch.id}>#{ch.id} {ch.challenger_name} vs {ch.defender_name} ({ch.status})</option>)}</select></label>
          <label>A challenger partner<br /><select value={resultDraft.a_chal} onChange={(e) => setResultDraft((c) => ({ ...c, a_chal: e.target.value }))} style={inputStyle}><option value="">Choose</option>{partnerPlayers.map((p) => <option key={p.player_id} value={p.player_id}>{p.player_name}</option>)}</select></label>
          <label>A defender partner<br /><select value={resultDraft.a_def} onChange={(e) => setResultDraft((c) => ({ ...c, a_def: e.target.value }))} style={inputStyle}><option value="">Choose</option>{partnerPlayers.filter((player) => String(player.player_id) !== resultDraft.a_chal).map((p) => <option key={p.player_id} value={p.player_id}>{p.player_name}</option>)}</select></label>
          <label>B challenger partner<br /><select value={resultDraft.b_chal} onChange={(e) => setResultDraft((c) => ({ ...c, b_chal: e.target.value }))} style={inputStyle}><option value="">Choose</option>{partnerPlayers.filter((player) => String(player.player_id) !== resultDraft.a_chal).map((p) => <option key={p.player_id} value={p.player_id}>{p.player_name}</option>)}</select></label>
          <label>B defender partner<br /><select value={resultDraft.b_def} onChange={(e) => setResultDraft((c) => ({ ...c, b_def: e.target.value }))} style={inputStyle}><option value="">Choose</option>{partnerPlayers.filter((player) => String(player.player_id) !== resultDraft.a_def && String(player.player_id) !== resultDraft.b_chal).map((p) => <option key={p.player_id} value={p.player_id}>{p.player_name}</option>)}</select></label>
          <label>Match A games<br /><input value={resultDraft.match_a_games} onChange={(e) => setResultDraft((c) => ({ ...c, match_a_games: e.target.value }))} placeholder="11-7,8-11,11-6" style={inputStyle} /></label>
          <label>Match B games<br /><input value={resultDraft.match_b_games} onChange={(e) => setResultDraft((c) => ({ ...c, match_b_games: e.target.value }))} placeholder="11-7,8-11,11-6" style={inputStyle} /></label>
          <label>Winner override<br /><select value={resultDraft.winner_override} onChange={(e) => setResultDraft((c) => ({ ...c, winner_override: e.target.value }))} style={inputStyle}><option value="computed">Computed</option><option value="challenger">Challenger</option><option value="defender">Defender</option></select></label>
          <label>Match date ISO<br /><input value={resultDraft.match_date} onChange={(e) => setResultDraft((c) => ({ ...c, match_date: e.target.value }))} style={inputStyle} /></label>
          <label style={{ display: "inline-flex", gap: "0.5rem", alignItems: "center" }}><input type="checkbox" checked={resultDraft.publish_official_matches} onChange={(e) => setResultDraft((c) => ({ ...c, publish_official_matches: e.target.checked }))} /> Publish official matches</label>
          <button type="button" onClick={previewResult} disabled={busy || !resultDraft.challenge_id || !resultDraft.a_chal || !resultDraft.a_def || !resultDraft.b_chal || !resultDraft.b_def} style={ghostButtonStyle}>Preview result</button>
          <ConfirmAction
            triggerLabel="Publish reviewed result"
            title="Publish this reviewed ladder result?"
            description="This finalizes the ladder result, may swap ranks, and can create two official rated matches."
            confirmLabel="Yes, publish result"
            confirmationText="PUBLISH LADDER RESULT"
            tone="danger"
            disabled={writesGuarded || !currentDraftIsPreviewed}
            busy={busy}
            onConfirm={publishResult}
          />
        </div>
        {resultPreview ? <section style={{ border: `1px solid ${currentDraftIsPreviewed ? "#86efac" : "#fbbf24"}`, borderRadius: "12px", padding: "1rem", marginTop: "1rem", background: currentDraftIsPreviewed ? "#f0fdf4" : "#fffbeb" }}>
          <h3 style={{ marginTop: 0 }}>{currentDraftIsPreviewed ? "Reviewed result preview" : "Preview is out of date"}</h3>
          {currentDraftIsPreviewed ? <>
            <p><strong>Winner:</strong> {resultPreview.preview.final_winner_side === "challenger" ? resultPreview.challenge.challenger_name : resultPreview.challenge.defender_name} ({resultPreview.preview.final_winner_side})</p>
            <p><strong>Rank movement:</strong> {resultPreview.rank_result.would_swap ? "Challenger and defender ranks will swap." : "Defender holds; no rank swap."}</p>
            <p><strong>Official matches:</strong> {resultPreview.would_publish_official_matches ? "Two rated Challenge Ladder matches will be submitted." : "No official match rows will be submitted."}</p>
            <p style={{ color: "#475569" }}>Any change to the challenge, partners, scores, date, winner choice, or publish setting invalidates this review.</p>
            <Pre value={{ partners: resultPreview.partner_names, winner: resultPreview.preview.winner_summary, scores: resultPreview.preview.scores, match_date: resultPreview.match_date }} />
          </> : <p>Preview the edited draft again before publishing.</p>}
        </section> : null}
      </article> : null}
      {dashboard ? <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Record forfeit</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label>Challenge<br /><select value={forfeitDraft.challenge_id} onChange={(e) => setForfeitDraft((c) => ({ ...c, challenge_id: e.target.value, forfeited_by_id: "" }))} style={inputStyle}><option value="">Choose</option>{openChallengeOptions.map((ch) => <option key={ch.id} value={ch.id}>#{ch.id} {ch.challenger_name} vs {ch.defender_name}</option>)}</select></label>
          <label>Forfeited by<br /><select value={forfeitDraft.forfeited_by_id} onChange={(e) => setForfeitDraft((c) => ({ ...c, forfeited_by_id: e.target.value }))} style={inputStyle}><option value="">Choose</option>{selectedForfeitChallenge?.challenger_id ? <option value={selectedForfeitChallenge.challenger_id}>{selectedForfeitChallenge.challenger_name} · challenger</option> : null}{selectedForfeitChallenge?.defender_id ? <option value={selectedForfeitChallenge.defender_id}>{selectedForfeitChallenge.defender_name} · defender</option> : null}</select></label>
          <label>Note<br /><input value={forfeitDraft.admin_note} onChange={(e) => setForfeitDraft((c) => ({ ...c, admin_note: e.target.value }))} style={inputStyle} /></label>
          <ConfirmAction
            triggerLabel="Record forfeit"
            title="Record this ladder forfeit?"
            description="This closes the selected challenge as a forfeit without creating official match rows."
            confirmLabel="Yes, record forfeit"
            confirmationText="RECORD LADDER FORFEIT"
            tone="danger"
            disabled={writesGuarded || !forfeitDraft.challenge_id || !forfeitDraft.forfeited_by_id}
            busy={busy}
            onConfirm={recordForfeit}
          />
        </div>
      </article> : null}
      {dashboard ? <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Record monthly pass</h2>
        <p style={{ color: "#475569" }}>A pass closes the selected challenge without publishing matches or changing ranks. The server rejects a second pass for the same player in the current UTC month.</p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label>Challenge<br /><select value={passDraft.challenge_id} onChange={(e) => setPassDraft({ challenge_id: e.target.value, player_id: "" })} style={inputStyle}><option value="">Choose</option>{openChallengeOptions.map((ch) => <option key={ch.id} value={ch.id}>#{ch.id} {ch.challenger_name} vs {ch.defender_name}</option>)}</select></label>
          <label>Pass used by<br /><select value={passDraft.player_id} onChange={(e) => setPassDraft((current) => ({ ...current, player_id: e.target.value }))} style={inputStyle}><option value="">Choose</option>{selectedPassChallenge?.challenger_id ? <option value={selectedPassChallenge.challenger_id}>{selectedPassChallenge.challenger_name} · challenger</option> : null}{selectedPassChallenge?.defender_id ? <option value={selectedPassChallenge.defender_id}>{selectedPassChallenge.defender_name} · defender</option> : null}</select></label>
          <ConfirmAction
            triggerLabel="Record monthly pass"
            title="Record this monthly pass?"
            description="This closes the selected challenge without publishing matches or changing ranks."
            confirmLabel="Yes, record pass"
            confirmationText="RECORD LADDER PASS"
            tone="danger"
            disabled={writesGuarded || !passDraft.challenge_id || !passDraft.player_id}
            busy={busy}
            onConfirm={recordPass}
          />
        </div>
      </article> : null}
      {dashboard ? <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Tier-movement review</h2>
        <p style={{ color: "#475569" }}>Read-only review of active players with 10 consecutive rated matches whose post-match tier is the same tier above or below their assigned ladder tier.</p>
        <button type="button" onClick={loadTierMovementReview} disabled={busy || !accessToken} style={ghostButtonStyle}>{busy ? "Loading…" : "Evaluate tier movement"}</button>
        {tierMovementReview ? <section style={{ marginTop: "1rem" }}>
          <p style={{ color: "#475569" }}>Evaluated {tierMovementReview.summary.evaluated_player_count} active players across {tierMovementReview.summary.match_count} recent matches · {tierMovementReview.summary.trigger_count} trigger{tierMovementReview.summary.trigger_count === 1 ? "" : "s"}.</p>
          {tierMovementReview.triggers.length ? <div style={{ display: "grid", gap: "0.75rem" }}>{tierMovementReview.triggers.map((trigger) => <div key={trigger.player_id} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.85rem", display: "flex", justifyContent: "space-between", alignItems: "center", gap: "1rem", flexWrap: "wrap" }}><div><strong>{trigger.player_name}</strong><br /><span style={{ color: "#475569" }}>{trigger.current_tier} → {trigger.destination_tier} · {trigger.consecutive_match_count} consecutive matches · latest {trigger.latest_match_at || "—"}</span></div><button type="button" onClick={() => prepareReviewedTierMove(trigger)} disabled={busy} style={ghostButtonStyle}>Prepare guarded move</button></div>)}</div> : <p>No players currently meet the tier-movement trigger.</p>}
        </section> : null}
      </article> : null}
      {dashboard ? <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Roster operations</h2>
        <p style={{ color: "#475569" }}>Add or reactivate a club player at the bottom of a tier, or move an active player to another tier. These actions do not replace an entire tier roster.</p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "1rem" }}>
          <section style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "1rem" }}>
            <h3 style={{ marginTop: 0 }}>Add or reactivate</h3>
            <div style={{ display: "grid", gap: "0.75rem" }}>
              <label>Club player<br /><select value={rosterAddDraft.player_id} onChange={(e) => setRosterAddDraft((current) => ({ ...current, player_id: e.target.value }))} style={inputStyle}><option value="">Choose a player not currently active</option>{addRosterOptions.map((player) => <option key={player.player_id} value={player.player_id}>{player.player_name}{inactiveRosterRows.some((row) => row.player_id === player.player_id) ? " · reactivate" : ""}</option>)}</select></label>
              <label>Tier<br /><select value={rosterAddDraft.tier_id} onChange={(e) => setRosterAddDraft((current) => ({ ...current, tier_id: e.target.value }))} style={inputStyle}>{dashboard.tiers.map((tier) => <option key={tier.tier_id} value={tier.tier_id}>{tier.label}</option>)}</select></label>
              <label>Audit note<br /><input value={rosterAddDraft.admin_note} onChange={(e) => setRosterAddDraft((current) => ({ ...current, admin_note: e.target.value }))} style={inputStyle} /></label>
              <ConfirmAction
                triggerLabel="Add to bottom"
                title="Add this player to the ladder?"
                description="This adds or reactivates the selected player at the bottom of the chosen tier."
                confirmLabel="Yes, add player"
                confirmationText="ADD LADDER PLAYER"
                disabled={writesGuarded || !rosterAddDraft.player_id}
                busy={busy}
                onConfirm={addRosterPlayer}
              />
            </div>
          </section>
          <section style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "1rem" }}>
            <h3 style={{ marginTop: 0 }}>Move active player</h3>
            <div style={{ display: "grid", gap: "0.75rem" }}>
              <label>Active player<br /><select value={rosterMoveDraft.player_id} onChange={(e) => { const player = activeRosterRows.find((row) => String(row.player_id) === e.target.value); setRosterMoveDraft((current) => ({ ...current, player_id: e.target.value, destination_tier: player?.tier_id === current.destination_tier ? (dashboard.tiers.find((tier) => tier.tier_id !== player.tier_id)?.tier_id || current.destination_tier) : current.destination_tier })); }} style={inputStyle}><option value="">Choose</option>{activeRosterRows.map((player) => <option key={player.player_id} value={player.player_id}>{player.player_name} · {player.tier_id} rank {player.rank ?? "—"}</option>)}</select></label>
              <label>Destination tier<br /><select value={rosterMoveDraft.destination_tier} onChange={(e) => setRosterMoveDraft((current) => ({ ...current, destination_tier: e.target.value }))} style={inputStyle}>{dashboard.tiers.filter((tier) => tier.tier_id !== selectedMovePlayer?.tier_id).map((tier) => <option key={tier.tier_id} value={tier.tier_id}>{tier.label}</option>)}</select></label>
              <label style={{ display: "inline-flex", gap: "0.5rem", alignItems: "center" }}><input type="checkbox" checked={rosterMoveDraft.recompress_old} onChange={(e) => setRosterMoveDraft((current) => ({ ...current, recompress_old: e.target.checked }))} /> Recompress the former tier</label>
              <label>Audit note<br /><input value={rosterMoveDraft.admin_note} onChange={(e) => setRosterMoveDraft((current) => ({ ...current, admin_note: e.target.value }))} style={inputStyle} /></label>
              <ConfirmAction
                triggerLabel="Move to tier bottom"
                title="Move this player to another tier?"
                description="This moves the selected player to the bottom of the destination tier and may recompress the former tier."
                confirmLabel="Yes, move player"
                confirmationText="MOVE LADDER PLAYER"
                disabled={writesGuarded || !rosterMoveDraft.player_id || rosterMoveDraft.destination_tier === selectedMovePlayer?.tier_id}
                busy={busy}
                onConfirm={moveRosterPlayer}
              />
            </div>
          </section>
        </div>
        <section style={{ border: "1px solid #f59e0b", borderRadius: "12px", padding: "1rem", marginTop: "1rem", background: "#fffbeb" }}>
          <h3 style={{ marginTop: 0 }}>Initialize or replace a complete tier</h3>
          <p style={{ color: "#475569" }}>Paste exact club player names, one per line, in rank order. Preview is read-only. Apply preserves removed players as inactive, recompresses any source tiers, blocks affected open challenges, and rejects the operation if the live roster changed after preview.</p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
            <label>Tier<br /><select value={rosterReplaceDraft.tier_id} onChange={(e) => { setRosterReplaceDraft((current) => ({ ...current, tier_id: e.target.value })); setRosterReplacePreview(null); setPreviewedRosterDraftFingerprint(null); }} style={inputStyle}>{dashboard.tiers.map((tier) => <option key={tier.tier_id} value={tier.tier_id}>{tier.label}</option>)}</select></label>
            <button type="button" onClick={loadCurrentTierRosterForReplacement} disabled={busy} style={ghostButtonStyle}>Load current tier order</button>
          </div>
          <label style={{ display: "block", marginTop: "0.75rem" }}>Ranked names (top to bottom)<br /><textarea value={rosterReplaceDraft.ranked_names} onChange={(e) => { setRosterReplaceDraft((current) => ({ ...current, ranked_names: e.target.value })); setRosterReplacePreview(null); setPreviewedRosterDraftFingerprint(null); }} rows={10} placeholder={"Player One\nPlayer Two\nPlayer Three"} style={{ ...inputStyle, resize: "vertical" }} /></label>
          <p><button type="button" onClick={previewRosterReplacement} disabled={busy || !rosterReplaceDraft.ranked_names.trim()} style={ghostButtonStyle}>Preview complete replacement</button></p>
          {rosterReplacePreview && currentRosterReplacementIsPreviewed ? <div style={{ borderTop: "1px solid #fcd34d", paddingTop: "1rem", marginTop: "1rem" }}>
            <h4 style={{ marginTop: 0 }}>Reviewed change set</h4>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: "0.5rem" }}>{Object.entries(rosterReplacePreview.summary).map(([key, value]) => <div key={key} style={{ background: "white", border: "1px solid #fde68a", borderRadius: "8px", padding: "0.55rem" }}><strong>{key.replace(/_/g, " ")}</strong><br />{value}</div>)}</div>
            {rosterReplacePreview.warnings.map((warning) => <p key={warning} style={{ color: "#92400e", fontWeight: 700 }}>{warning}</p>)}
            {rosterReplacePreview.open_challenge_blockers.length ? <div style={{ background: "#fef2f2", border: "1px solid #fecaca", borderRadius: "8px", padding: "0.75rem" }}><strong>Blocking open challenges</strong><ul>{rosterReplacePreview.open_challenge_blockers.map((blocker) => <li key={blocker.challenge_id}>#{blocker.challenge_id} · {blocker.status} · {blocker.affected_player_names.join(", ")}</li>)}</ul></div> : null}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1rem", marginTop: "0.75rem" }}>
              <div><strong>Proposed order</strong><ol>{rosterReplacePreview.proposed_roster.map((row) => <li key={row.player_id}>{row.player_name} · {row.change}{row.previous_tier && row.previous_tier !== rosterReplacePreview.tier_id ? ` from ${row.previous_tier} rank ${row.previous_rank ?? "—"}` : ""}</li>)}</ol></div>
              <div><strong>Players made inactive</strong>{rosterReplacePreview.removed_players.length ? <ul>{rosterReplacePreview.removed_players.map((row) => <li key={row.player_id}>{row.player_name} · former rank {row.rank}</li>)}</ul> : <p>None.</p>}<strong>Source-tier rank updates</strong>{rosterReplacePreview.source_tier_recompressions.length ? <ul>{rosterReplacePreview.source_tier_recompressions.map((row) => <li key={row.player_id}>{row.player_name} · {row.tier_id} {row.old_rank} → {row.new_rank}</li>)}</ul> : <p>None.</p>}</div>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
              <label>Audit note<br /><input value={rosterReplaceDraft.admin_note} onChange={(e) => setRosterReplaceDraft((current) => ({ ...current, admin_note: e.target.value }))} style={inputStyle} /></label>
              <ConfirmAction
                triggerLabel="Apply reviewed replacement"
                title="Replace this complete tier roster?"
                description="This applies the reviewed order, makes removed players inactive, and may recompress source tiers."
                confirmLabel="Yes, replace tier roster"
                confirmationText="REPLACE LADDER TIER"
                tone="danger"
                disabled={writesGuarded || !rosterReplacePreview.can_apply || !currentRosterReplacementIsPreviewed}
                busy={busy}
                onConfirm={applyRosterReplacement}
              />
            </div>
          </div> : null}
        </section>
        {inactiveRosterRows.length ? <section style={{ marginTop: "1rem" }}><h3>Inactive roster</h3><p style={{ color: "#475569" }}>{inactiveRosterRows.map((row) => `${row.player_name} (${row.tier_id}, former rank ${row.rank ?? "—"})`).join(" · ")}</p></section> : null}
      </article> : null}
      {dashboard ? <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Vacation and reinstate overrides</h2>
        <p style={{ color: "#475569" }}>Vacation temporarily marks an active player unavailable. Reinstate-required takes priority in the public ladder status until an administrator clears it.</p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label>Active player<br /><select value={overrideDraft.player_id} onChange={(e) => { const flags = dashboard.player_flags?.find((row) => String(row.player_id) === e.target.value); setOverrideDraft({ player_id: e.target.value, vacation_until: flags?.vacation_until || "", reinstate_required: Boolean(flags?.reinstate_required), reinstate_notes: flags?.reinstate_notes || "" }); }} style={inputStyle}><option value="">Choose</option>{activeRosterRows.map((player) => <option key={player.player_id} value={player.player_id}>{player.player_name} · {player.tier_id} rank {player.rank ?? "—"}</option>)}</select></label>
          <label>Vacation until (ISO with Z/offset)<br /><input value={overrideDraft.vacation_until} onChange={(e) => setOverrideDraft((current) => ({ ...current, vacation_until: e.target.value }))} placeholder="2026-08-01T17:00:00Z" style={inputStyle} /></label>
          <label style={{ display: "inline-flex", gap: "0.5rem", alignItems: "center" }}><input type="checkbox" checked={overrideDraft.reinstate_required} onChange={(e) => setOverrideDraft((current) => ({ ...current, reinstate_required: e.target.checked }))} /> Reinstate required</label>
          <label>Reinstate notes<br /><input value={overrideDraft.reinstate_notes} onChange={(e) => setOverrideDraft((current) => ({ ...current, reinstate_notes: e.target.value }))} style={inputStyle} /></label>
          <ConfirmAction
            triggerLabel="Save overrides"
            title="Save these ladder overrides?"
            description="This updates the selected player's vacation and reinstatement status."
            confirmLabel="Yes, save overrides"
            confirmationText="SAVE LADDER OVERRIDES"
            disabled={writesGuarded || !overrideDraft.player_id}
            busy={busy}
            onConfirm={savePlayerOverrides}
          />
        </div>
        {currentOverrideRows.length ? <section style={{ marginTop: "1rem" }}><h3>Current overrides</h3><div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse" }}><thead><tr><th align="left">Player</th><th align="left">Vacation until</th><th align="left">Reinstate</th><th align="left">Notes</th></tr></thead><tbody>{currentOverrideRows.map((flags) => <tr key={flags.player_id}><td>{flags.player_name}</td><td>{flags.vacation_until || "—"}</td><td>{flags.reinstate_required ? "Required" : "No"}</td><td>{flags.reinstate_notes || "—"}</td></tr>)}</tbody></table></div></section> : null}
      </article> : null}
      {dashboard ? <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Summary</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "0.75rem" }}>
          {Object.entries(dashboard.summary || {}).map(([key, value]) => <div key={key} style={{ border: "1px solid #e2e8f0", borderRadius: "10px", padding: "0.75rem" }}><strong>{key.replace(/_/g, " ")}</strong><br />{String(value)}</div>)}
        </div>
      </article> : null}
      {dashboard?.tiers?.length ? <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Roster by tier</h2>{dashboard.tiers.filter((tier) => tier.players.length).map((tier) => <section key={tier.tier_id} style={{ marginTop: "1rem" }}><h3>{tier.label} · {tier.range}</h3><div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse" }}><tbody>{tier.players.map((player) => <tr key={player.player_id}><td>{player.rank ?? "—"}</td><td>{player.player_name}</td><td>{player.rating_jupr ? player.rating_jupr.toFixed(3) : "—"}</td><td>{player.status || "Ready"}</td></tr>)}</tbody></table></div></section>)}</article> : null}
      {dashboard?.challenges?.length ? <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Challenges</h2>
        {dashboard.challenges.map((challenge) => <section key={challenge.id} style={{ borderTop: "1px solid #e2e8f0", paddingTop: "0.75rem", marginTop: "0.75rem" }}>
          <h3 style={{ marginTop: 0 }}>#{challenge.id} · {challenge.bucket}</h3>
          <p>{challenge.challenger_name} vs {challenge.defender_name} · {challenge.status} · {challenge.tier_id}</p>
          <label>Admin note<br /><input value={notes[challenge.id] || ""} onChange={(e) => setNotes((current) => ({ ...current, [challenge.id]: e.target.value }))} style={inputStyle} /></label>
          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginTop: "0.75rem" }}>
            <ConfirmAction
              triggerLabel="Start clock"
              title={`Start the clock for challenge #${challenge.id}?`}
              description="This starts the ladder challenge acceptance or play deadline clock."
              confirmLabel="Yes, start clock"
              confirmationText="START LADDER CLOCK"
              disabled={writesGuarded}
              busy={busy}
              onConfirm={(confirmationText) => simpleAction(challenge, "start-clock", confirmationText)}
            />
            <ConfirmAction
              triggerLabel="Accept"
              title={`Accept challenge #${challenge.id}?`}
              description="This records the challenge as accepted and advances its workflow."
              confirmLabel="Yes, accept challenge"
              confirmationText="ACCEPT LADDER CHALLENGE"
              disabled={writesGuarded}
              busy={busy}
              onConfirm={(confirmationText) => simpleAction(challenge, "accept", confirmationText)}
            />
            <ConfirmAction
              triggerLabel="Cancel"
              title={`Cancel challenge #${challenge.id}?`}
              description="This closes the challenge as cancelled. The admin note above will be retained in the audit trail."
              confirmLabel="Yes, cancel challenge"
              confirmationText="SAVE LADDER"
              tone="danger"
              disabled={writesGuarded}
              busy={busy}
              onConfirm={(confirmationText) => updateChallenge(challenge, "CANCELLED", confirmationText)}
            />
          </div>
        </section>)}
      </article> : null}
      {lastOperationKey ? <article style={{ ...cardStyle, borderColor: "#f59e0b", background: "#fffbeb" }}>
        <h2 style={{ marginTop: 0 }}>Operation recovery</h2>
        <p>Operation key: <code>{lastOperationKey}</code>. If a response was lost, reconcile this record before retrying or entering a correction.</p>
        <p><a href={lastResult?.correction?.match_log_url || lastResult?.recovery?.match_log_url || "/admin/match-log"}>Open Match Log</a> · <a href={lastResult?.correction?.replay_history_url || lastResult?.recovery?.replay_history_url || "/admin/replay-history"}>Open Replay History</a></p>
        <ConfirmAction
          triggerLabel="Reconcile durable operation"
          title="Reconcile this ladder operation?"
          description="This inspects the durable operation record and recovers its stored response without replaying the write."
          confirmLabel="Yes, reconcile operation"
          confirmationText="RECONCILE LADDER OPERATION"
          disabled={busy}
          busy={busy}
          onConfirm={reconcileLastOperation}
        />
      </article> : null}
      {lastResult ? <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Last action result</h2><Pre value={lastResult} /></article> : null}
    </div>
  );
}
