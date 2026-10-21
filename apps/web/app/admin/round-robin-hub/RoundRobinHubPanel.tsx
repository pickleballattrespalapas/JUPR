"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { InteractionDialog } from "@/components/interaction";
import GeneratorRoundRunner from "@/app/admin/play-generators/GeneratorRoundRunner";
import GeneratorStandings from "@/app/admin/play-generators/GeneratorStandings";
import { useAdminSession } from "@/lib/useAdminSession";
import { matchesPlayerSearch } from "@/lib/playerSearch";
import HubAddPlayerDialog from "./HubAddPlayerDialog";
import HubRemovePlayerDialog from "./HubRemovePlayerDialog";
import HubFinishDialog from "./HubFinishDialog";
import HubRegistrationPanel from "./HubRegistrationPanel";
import {
  hubMatchLabel, hubPlayerLabel, hubOperationKey, hubWaitLabel, isUncertainHubError, requestRoundRobinHub, RoundRobinHubApiError,
  type CreateHubInput, type HubCommand, type HubCommandAction, type HubCourtConfig, type HubDetailResponse,
  type HubGroup, type HubGroupPlayer, type HubIdentityParticipant, type HubIdentityReview, type HubListResponse, type HubMatch, type HubWaitingPlayer
} from "@/lib/adminRoundRobinHubApi";
import styles from "./RoundRobinHubPanel.module.css";

type Props = { apiBase: string | null; clubId: string; initialHubId?: string; initialAddSessionKey?: string; initialView?: string };
type Tab = "board" | "queue" | "groups" | "standings" | "players" | "courts" | "registration";
type Pending = { hubId: string; body: HubCommand | CreateHubInput };
type CreatedSession = { hubId: string; sessionKey: string };
type ReturnIntent = CreatedSession & { attempted: boolean };
type ScoreEditor = { courtId: string; match: HubMatch; version: string | number; scoreA: string; scoreB: string };
type CourtEditor = { match: HubMatch; sourceCourtId: string; targetCourtId: string; version: string | number };
type SessionView = { sessionKey: string; roundNumber: number };
type StandingsSelection = { hubId: string; sessionKey: string };

const emptyList: HubListResponse = { ok: true, hubs: [], sessions: [] };
const commandMessages: Partial<Record<HubCommandAction, string>> = {
  add_group: "Round robin added to the hub.", remove_group: "Round robin removed from the hub. Its results are preserved.",
  configure_courts: "Court setup saved.", assign_next_court: "Match assigned to an open court.", assign_game: "Match assigned.",
  reserve_game_for_court: "Match reserved next on this court.", cancel_reservation: "Court reservation canceled.",
  move_game: "Match moved.", release_game: "Match returned to the queue.", score_and_release: "Score saved and court released.",
  mark_played: "Match marked played and court released.", pause_player: "Player paused. Waiting games checked for available replacements. Their waiting time is held.",
  resume_player: "Player is ready again.", advance_round: "Next round is ready for the court queue.", finish_hub: "Event finished.",
  finish_group: "Round robin finish request saved.",
  set_registration: "Public check-in updated.", review_checkin: "Check-in saved.",
  auto_fill_courts: "Available courts filled with eligible matches.",
  add_player: "Player added to the round robin. Games already assigned keep their players.",
  remove_player: "Player departure saved. Games already assigned keep their players.",
  resolve_identities: "Player identity decisions saved. The court queue has been updated."
};

function newCourts(count: number): HubCourtConfig[] {
  return Array.from({ length: count }, (_, index) => ({ id: `court-${index + 1}`, label: `Court ${index + 1}`, dedicated_session_key: null, open: true }));
}
function courtConfig(court: HubCourtConfig): HubCourtConfig {
  return { id: court.id, label: court.label, dedicated_session_key: court.dedicated_session_key || null, open: court.open };
}
function formatName(value?: string): string { return value === "moneyball_expectations" ? "Moneyball" : "Round robin"; }
function defaultStandingsGroup(groups: HubGroup[]): HubGroup | undefined {
  return groups.find(group => group.status === "active" && group.scoring_mode !== "unscored")
    || groups.find(group => group.scoring_mode !== "unscored") || groups[0];
}
function identityParticipantKey(participant: HubIdentityParticipant): string {
  return JSON.stringify([participant.session_key, participant.participant_id]);
}
function groupPlayers(group: HubGroup, players: HubWaitingPlayer[]): HubGroupPlayer[] {
  if (group.players) return group.players;
  // Older snapshots still show the roster while the API deploy catches up.
  return players.filter(player => player.session_keys.includes(group.session_key)).map(player => ({
    participant_id: Array.isArray(player.participant_ids) ? player.identity : player.participant_ids[group.session_key] || player.identity,
    identity: player.identity, name: player.name, display_name: player.display_name, profile_id: null,
    paused: player.paused, wait_seconds: player.wait_seconds, on_court: false, reserved: false, active: true
  }));
}
function pendingStorageKey(clubId: string): string { return `rr-hub-pending:${clubId}`; }
function readPending(clubId: string): Pending | null {
  try {
    const value = JSON.parse(sessionStorage.getItem(pendingStorageKey(clubId)) || "null");
    return value && typeof value.hubId === "string" && value.body && typeof value.body.idempotency_key === "string" ? value as Pending : null;
  } catch { return null; }
}

export default function RoundRobinHubPanel({ apiBase, clubId, initialHubId = "", initialAddSessionKey = "", initialView = "" }: Props) {
  const { accessToken } = useAdminSession();
  const [list, setList] = useState<HubListResponse>(emptyList);
  const [hubId, setHubId] = useState(initialHubId);
  const [detail, setDetail] = useState<HubDetailResponse | null>(null);
  const [tab, setTab] = useState<Tab>(initialAddSessionKey ? "groups" : initialView === "registration" ? "registration" : "board");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [pending, setPending] = useState<Pending | null>(null);
  const [pendingHydrated, setPendingHydrated] = useState(false);
  const [createdSession, setCreatedSession] = useState<CreatedSession | null>(null);
  const [attaching, setAttaching] = useState(false);
  const [creating, setCreating] = useState(false);
  const [title, setTitle] = useState("");
  const [createCourts, setCreateCourts] = useState<HubCourtConfig[]>(() => newCourts(6));
  const [editCourts, setEditCourts] = useState<HubCourtConfig[]>([]);
  const [courtDraftVersion, setCourtDraftVersion] = useState<string | number>("");
  const [addSessionKey, setAddSessionKey] = useState(initialAddSessionKey);
  const [playerSearch, setPlayerSearch] = useState("");
  const [addPlayerOpen, setAddPlayerOpen] = useState(false);
  const [removePlayerOpen, setRemovePlayerOpen] = useState(false);
  const [scoreEditor, setScoreEditor] = useState<ScoreEditor | null>(null);
  const [courtEditor, setCourtEditor] = useState<CourtEditor | null>(null);
  const [sessionView, setSessionView] = useState<SessionView | null>(null);
  const [standingsSelection, setStandingsSelection] = useState<StandingsSelection | null>(null);
  const [finishReview, setFinishReview] = useState(false);
  const [finishGroupOpen, setFinishGroupOpen] = useState(false);
  const [identityReviewOpen, setIdentityReviewOpen] = useState(false);
  const [savedIdentityReviewOpen, setSavedIdentityReviewOpen] = useState(false);
  const [identitySelections, setIdentitySelections] = useState<Record<string, string[]>>({});
  const selectedRef = useRef(hubId);
  const loadSequence = useRef(0);
  const listLoadSequence = useRef(0);
  const mutationLock = useRef(false);
  const returnIntent = useRef<ReturnIntent | null>(null);
  const attachmentInFlight = useRef<ReturnIntent | null>(null);
  const scoreInput = useRef<HTMLInputElement>(null);
  const ready = Boolean(apiBase && accessToken);
  const workspace = detail?.workspace;
  const identityReviews = workspace?.identity_review || [];
  const pendingIdentityReviews = identityReviews.filter(review => review.status === "pending");
  const savedIdentityReviews = identityReviews.filter(review => review.status !== "pending");
  const identityPendingCount = workspace?.identity_pending_count ?? pendingIdentityReviews.length;
  const createDraftToken = useMemo(() => hubOperationKey(), [hubId, workspace?.groups.length]);
  const disabled = busy || attaching || Boolean(pending) || detail?.hub.status === "completed";
  const groupMap = useMemo(() => new Map((workspace?.groups || []).map(group => [group.session_key, group])), [workspace?.groups]);
  const standingsGroup = standingsSelection?.hubId === detail?.hub.id ? groupMap.get(standingsSelection?.sessionKey || "") : undefined;
  const availableSessions = list.sessions.filter(session => !session.round_robin_hub_id && session.status === "active" && !groupMap.has(session.session_key));
  const playerGroups = (workspace?.groups || []).map(group => ({
    group,
    players: groupPlayers(group, workspace?.players || [])
      .filter(player => matchesPlayerSearch(`${player.name} ${group.title}`, playerSearch))
      .sort((left, right) => left.name.localeCompare(right.name, undefined, { numeric: true, sensitivity: "base" }))
  })).filter(({ players }) => !playerSearch.trim() || players.length > 0)
    .sort((left, right) => left.group.title.localeCompare(right.group.title, undefined, { numeric: true, sensitivity: "base" }));
  const byeGroups = (workspace?.groups || []).map(group => ({ group, players: (group.players || []).filter(player => player.active && player.on_bye) }))
    .filter(({ players }) => players.length > 0);
  const activeGroups = (workspace?.groups || []).filter(group => group.status !== "completed");
  const canFillCourts = activeGroups.some(group => !group.results_locked)
    && Boolean(workspace?.courts.some(court => court.open && !court.current && !court.next));
  const finishingGroups = activeGroups.filter(group => group.finish_requested);

  const request = useCallback(<T,>(suffix = "", body?: unknown) => {
    if (!apiBase || !accessToken) throw new Error("Sign in before managing a round robin hub.");
    return requestRoundRobinHub<T>(apiBase, clubId, accessToken, suffix, body);
  }, [apiBase, clubId, accessToken]);

  const savePending = useCallback((value: Pending | null) => {
    setPending(value);
    try {
      if (value) sessionStorage.setItem(pendingStorageKey(clubId), JSON.stringify(value));
      else sessionStorage.removeItem(pendingStorageKey(clubId));
    } catch { /* The in-memory exact action remains available for retry. */ }
  }, [clubId]);

  const loadList = useCallback(async () => {
    if (!ready) return;
    const sequence = ++listLoadSequence.current;
    const value = await request<HubListResponse>();
    if (sequence === listLoadSequence.current) setList(value);
    return value;
  }, [ready, request]);

  const applyHubDetail = useCallback((value: HubDetailResponse) => {
    setDetail(value);
    const groups = value.workspace.groups;
    setStandingsSelection(current => {
      if (current?.hubId === value.hub.id && groups.some(group => group.session_key === current.sessionKey)) return current;
      const group = defaultStandingsGroup(groups);
      return group ? { hubId: value.hub.id, sessionKey: group.session_key } : null;
    });
    setSessionView(current => current && groups.some(group => group.session_key === current.sessionKey) ? current : null);
  }, []);

  const loadHub = useCallback(async (id: string, silent = false) => {
    if (!ready || !id) return;
    const sequence = ++loadSequence.current;
    if (!silent) setLoading(true);
    try {
      let value = await request<HubDetailResponse>(`/${encodeURIComponent(id)}`);
      if (value.registration?.automatic_pending_count && !mutationLock.current) {
        mutationLock.current = true;
        try {
          value = await request<HubDetailResponse>(`/${encodeURIComponent(id)}/commands`, {
            action: "process_checkins", payload: {}, expected_version: value.hub.version,
            idempotency_key: `automatic-checkins-v${value.hub.version}`
          });
        } catch (cause) {
          if (!(cause instanceof RoundRobinHubApiError && cause.status === 409)) throw cause;
        } finally { mutationLock.current = false; }
      }
      if (String(value.hub.id) !== id) throw new Error("The server returned a different event. Reload the hub before continuing.");
      if (sequence === loadSequence.current && selectedRef.current === id) {
        applyHubDetail(value);
        if (!silent) setError("");
      }
      return value;
    } catch (cause) {
      if (sequence === loadSequence.current && selectedRef.current === id) setError(cause instanceof Error ? cause.message : "Unable to load this event.");
    } finally {
      if (sequence === loadSequence.current && selectedRef.current === id) setLoading(false);
    }
  }, [ready, request, applyHubDetail]);

  const clearReturnIntent = useCallback(() => {
    returnIntent.current = null;
    setCreatedSession(null);
    // Navigation may cancel preparation, but never discards a sent command or
    // its durable pending request. That command still needs confirmation.
    if (!mutationLock.current) {
      attachmentInFlight.current = null;
      setAttaching(false);
    }
  }, []);

  function selectHub(id: string): void {
    clearReturnIntent();
    selectedRef.current = id;
    loadSequence.current += 1;
    setHubId(id); setDetail(null); setCreating(false); setSessionView(null); setScoreEditor(null); setCourtEditor(null); setError(""); setMessage("");
    setStandingsSelection(null);
    setAddPlayerOpen(false); setRemovePlayerOpen(false);
    setFinishGroupOpen(false); setFinishReview(false);
    setIdentityReviewOpen(false); setSavedIdentityReviewOpen(false); setIdentitySelections({});
    if (typeof window !== "undefined") window.history.replaceState(null, "", `/admin/round-robin-hub${id ? `?hub=${encodeURIComponent(id)}` : ""}`);
  }

  useEffect(() => {
    const retained = readPending(clubId);
    setPending(retained);
    setPendingHydrated(true);
    if (retained) setError("A previous action needs confirmation. Retry it below before making another change.");
  }, [clubId]);

  // Next preserves this panel's state on return navigation. The URL, rather than
  // the previously selected event, owns the destination of a newly created RR.
  useEffect(() => {
    if (!initialHubId || !initialAddSessionKey || returnIntent.current?.hubId !== initialHubId || returnIntent.current.sessionKey !== initialAddSessionKey) clearReturnIntent();
    if (!initialHubId) return;
    if (selectedRef.current !== initialHubId) {
      selectedRef.current = initialHubId;
      loadSequence.current += 1;
      setHubId(initialHubId); setDetail(null); setCreating(false); setSessionView(null); setScoreEditor(null); setCourtEditor(null);
      setStandingsSelection(null);
      setAddPlayerOpen(false); setRemovePlayerOpen(false);
      setFinishGroupOpen(false); setFinishReview(false);
      setIdentityReviewOpen(false); setSavedIdentityReviewOpen(false); setIdentitySelections({});
    }
    if (!initialAddSessionKey) return;
    if (returnIntent.current?.hubId === initialHubId && returnIntent.current.sessionKey === initialAddSessionKey) return;
    returnIntent.current = { hubId: initialHubId, sessionKey: initialAddSessionKey, attempted: false };
    setCreatedSession({ hubId: initialHubId, sessionKey: initialAddSessionKey });
    setAddSessionKey(initialAddSessionKey); setTab("board"); setCreating(false); setMessage("");
    setSessionView(null); setScoreEditor(null); setCourtEditor(null);
  }, [initialHubId, initialAddSessionKey, clearReturnIntent]);

  useEffect(() => {
    if (!ready) return;
    let live = true;
    void loadList().then(value => {
      if (!live) return;
      if (!selectedRef.current && value?.hubs.length) selectHub((value.hubs.find(hub => hub.status !== "completed") || value.hubs[0]).id);
      else if (!selectedRef.current) { setCreating(true); setLoading(false); }
    }).catch(cause => { if (live) { setError(cause instanceof Error ? cause.message : "Unable to load hubs."); setLoading(false); } });
    return () => { live = false; };
  }, [ready, loadList, initialHubId, initialAddSessionKey]);

  useEffect(() => { selectedRef.current = hubId; if (hubId && ready) void loadHub(hubId); }, [hubId, ready, loadHub]);
  useEffect(() => {
    if (!ready) return;
    let refreshing = false;
    const refresh = async () => {
      if (document.visibilityState !== "visible" || mutationLock.current || refreshing) return;
      refreshing = true;
      try {
        await Promise.all([loadList(), hubId ? loadHub(hubId, true) : Promise.resolve()]);
      } catch { /* Keep the current choices; manual refresh can report a connection failure. */ }
      finally { refreshing = false; }
    };
    const timer = window.setInterval(refresh, 15_000);
    window.addEventListener("focus", refresh); document.addEventListener("visibilitychange", refresh);
    return () => { window.clearInterval(timer); window.removeEventListener("focus", refresh); document.removeEventListener("visibilitychange", refresh); };
  }, [hubId, ready, loadHub, loadList]);

  const completeCreatedSession = useCallback((value: HubDetailResponse): boolean => {
    const intent = returnIntent.current;
    if (!intent || value.hub.id !== intent.hubId || !value.workspace.groups.some(group => group.session_key === intent.sessionKey)) return false;
    returnIntent.current = null;
    setCreatedSession(null); setAddSessionKey("");
    selectedRef.current = intent.hubId;
    setHubId(intent.hubId); applyHubDetail(value); setCreating(false); setTab("board"); setError("");
    setMessage("Round robin added. Choose Next open court or Fill open courts to start play.");
    window.history.replaceState(null, "", `/admin/round-robin-hub?hub=${encodeURIComponent(intent.hubId)}`);
    return true;
  }, [applyHubDetail]);

  const execute = useCallback(async (operation: Pending): Promise<boolean> => {
    if (mutationLock.current) return false;
    mutationLock.current = true; setBusy(true); setError(""); setMessage(""); savePending(operation);
    loadSequence.current += 1;
    try {
      const value = await request<HubDetailResponse>(operation.hubId ? `/${encodeURIComponent(operation.hubId)}/commands` : "", operation.body);
      if (operation.hubId && String(value.hub.id) !== operation.hubId) throw new Error("The action returned a different event. Retry this exact action to confirm its result.");
      if (returnIntent.current?.hubId === operation.hubId && "action" in operation.body && operation.body.action === "add_group" && operation.body.payload.session_key === returnIntent.current.sessionKey && !value.workspace.groups.some(group => group.session_key === returnIntent.current?.sessionKey)) {
        throw new Error("The round robin's addition was not confirmed. Retry this exact action to check its result.");
      }
      savePending(null);
      if (!operation.hubId) {
        selectedRef.current = value.hub.id; setHubId(value.hub.id); setCreating(false);
        window.history.replaceState(null, "", `/admin/round-robin-hub?hub=${encodeURIComponent(value.hub.id)}`);
      }
      if (selectedRef.current === value.hub.id) applyHubDetail(value);
      setMessage("action" in operation.body ? commandMessages[operation.body.action] || "Hub updated." : "Event created. Add your round robins below.");
      setScoreEditor(null); setCourtEditor(null); setFinishReview(false);
      if ("action" in operation.body && operation.body.action === "add_player") {
        setAddPlayerOpen(false);
        if (Number(value.workspace.identity_pending_count || 0) > 0) {
          setIdentityReviewOpen(true); setSavedIdentityReviewOpen(false);
          setMessage("Player added. Review the matching player names below; other ready matches can keep playing.");
        }
      }
      if ("action" in operation.body && operation.body.action === "remove_player") setRemovePlayerOpen(false);
      if ("action" in operation.body && operation.body.action === "finish_group") {
        setFinishGroupOpen(false);
        const sessionKey = operation.body.payload.session_key;
        const finishedGroup = value.workspace.groups.find(group => group.session_key === sessionKey);
        setMessage(`${finishedGroup?.title || "Round robin"}${finishedGroup?.status === "completed" ? " finished. Results remain available." : ` is finishing${finishedGroup?.finish_after_round ? ` after round ${finishedGroup.finish_after_round}` : ""}. Remaining matches stay in the queue.`}${value.hub.status === "completed" ? " All round robins are finished. Event closed." : ""}`);
      }
      if (!completeCreatedSession(value) && (!operation.hubId || ("action" in operation.body && operation.body.action === "add_group"))) setTab("groups");
      if ("action" in operation.body && operation.body.action === "add_group" && Number(value.workspace.identity_pending_count || 0) > 0) {
        setIdentityReviewOpen(true); setSavedIdentityReviewOpen(false); setTab("players");
        setMessage("Round robin added. Review the matching player names below; other ready matches can keep playing.");
      }
      if ("action" in operation.body && operation.body.action === "resolve_identities") setIdentitySelections({});
      void loadList().catch(() => undefined);
      return true;
    } catch (cause) {
      const text = cause instanceof Error ? cause.message : "Unable to complete this action.";
      if (isUncertainHubError(cause)) setError(`${text} The result is not yet confirmed. Retry the same action below.`);
      else {
        savePending(null); setError(text);
        if (cause instanceof RoundRobinHubApiError && cause.status === 409 && operation.hubId) await loadHub(operation.hubId, true);
      }
      return false;
    } finally { mutationLock.current = false; setBusy(false); setLoading(false); }
  }, [request, savePending, loadList, loadHub, completeCreatedSession, applyHubDetail]);

  const attachCreatedSession = useCallback(async (intent: ReturnIntent): Promise<void> => {
    if (mutationLock.current || pending || returnIntent.current !== intent) return;
    // Set before awaiting so Strict Mode and rerenders cannot start a second request.
    intent.attempted = true;
    attachmentInFlight.current = intent;
    setAttaching(true); setError("");
    selectedRef.current = intent.hubId;
    setHubId(intent.hubId); setCreating(false);
    loadSequence.current += 1;
    try {
      const value = await request<HubDetailResponse>(`/${encodeURIComponent(intent.hubId)}`);
      if (returnIntent.current !== intent) return;
      if (value.hub.id !== intent.hubId) throw new Error("The server returned a different event. Retry adding your saved round robin.");
      applyHubDetail(value);
      if (completeCreatedSession(value)) return;
      await execute({ hubId: intent.hubId, body: { action: "add_group", payload: { session_key: intent.sessionKey }, expected_version: value.hub.version, idempotency_key: hubOperationKey() } });
    } catch (cause) {
      if (returnIntent.current === intent) setError(cause instanceof Error ? cause.message : "Unable to add your saved round robin to this event.");
    } finally {
      if (attachmentInFlight.current === intent) {
        attachmentInFlight.current = null;
        setAttaching(false); setLoading(false);
      }
    }
  }, [pending, request, execute, completeCreatedSession, applyHubDetail]);

  useEffect(() => {
    const intent = returnIntent.current;
    if (ready && pendingHydrated && createdSession && !pending && !busy && intent && !intent.attempted) void attachCreatedSession(intent);
  }, [ready, pendingHydrated, createdSession, pending, busy, attachCreatedSession]);

  async function command(action: HubCommandAction, payload: Record<string, unknown> = {}, version?: string | number): Promise<boolean> {
    if (!detail || disabled) return false;
    return execute({ hubId: detail.hub.id, body: { action, payload, expected_version: version ?? detail.hub.version, idempotency_key: hubOperationKey() } });
  }

  function openCourtPicker(match: HubMatch, sourceCourtId = ""): void {
    if (!detail) return;
    const choice = workspace?.courts.find(court => match.allowed_court_ids.includes(court.id) && court.id !== sourceCourtId && court.open && !court.next && (!sourceCourtId || !court.current));
    setCourtEditor({ match, sourceCourtId, targetCourtId: choice?.id || "", version: detail.hub.version }); setError("");
  }

  function openCourtSetup(): void {
    if (!detail) return;
    setEditCourts(detail.workspace.courts.map(courtConfig)); setCourtDraftVersion(detail.hub.version); setTab("courts");
  }

  function openFinishGroup(): void {
    setFinishGroupOpen(true); setFinishReview(false); setAddPlayerOpen(false); setRemovePlayerOpen(false); setError("");
  }

  function courtEditorRows(courts: HubCourtConfig[], update: (value: HubCourtConfig[]) => void, existing: boolean) {
    return <div className={styles.courtEditor}>{courts.map((court, index) => {
      const occupied = Boolean(existing && workspace?.courts.find(item => item.id === court.id && (item.current || item.next)));
      return <div className={styles.courtEditorRow} key={court.id}>
        <label className={styles.field}>Court name<input aria-label={`Court ${index + 1} name`} value={court.label} maxLength={80} onChange={event => update(courts.map(item => item.id === court.id ? { ...item, label: event.target.value } : item))} /></label>
        <label className={styles.field}>Court use<select aria-label={`Court ${index + 1} use`} value={court.dedicated_session_key || ""} disabled={occupied} onChange={event => update(courts.map(item => item.id === court.id ? { ...item, dedicated_session_key: event.target.value || null } : item))}>
          <option value="">Shared courts</option>{(existing ? workspace?.groups || [] : []).map(group => <option key={group.session_key} value={group.session_key}>{group.title} only</option>)}
        </select></label>
        <label className={styles.checkbox}><input type="checkbox" checked={court.open} disabled={occupied} onChange={event => update(courts.map(item => item.id === court.id ? { ...item, open: event.target.checked } : item))} />Open</label>
        <button type="button" className={styles.button} disabled={occupied || courts.length <= 1} onClick={() => update(courts.filter(item => item.id !== court.id))} aria-label={`Remove ${court.label}`}>Remove</button>
      </div>;
    })}<div><button type="button" className={styles.button} disabled={courts.length >= 40} onClick={() => update([...courts, { id: hubOperationKey(), label: `Court ${courts.length + 1}`, dedicated_session_key: null, open: true }])}>Add court</button></div></div>;
  }

  function showSession(group: HubGroup, view: "round" | "standings" = "round"): void {
    if (view === "standings") {
      setStandingsSelection({ hubId, sessionKey: group.session_key });
      setSessionView(null); setTab("standings");
    } else setSessionView({ sessionKey: group.session_key, roundNumber: group.current_round_number });
  }

  function navigateSession(sessionKey: string, target: { view: "round" | "standings"; roundNumber?: number }): void {
    const group = groupMap.get(sessionKey);
    if (!group) return;
    if (target.view === "standings") showSession(group, "standings");
    else {
      setSessionView({ sessionKey, roundNumber: target.roundNumber || group.current_round_number });
      if (tab === "standings") setTab("groups");
    }
  }

  function openIdentityReview(): void {
    setIdentityReviewOpen(true); setSavedIdentityReviewOpen(false); setTab("players");
  }

  function selectedIdentityParticipants(review: HubIdentityReview): HubIdentityParticipant[] {
    const selected = identitySelections[review.id];
    return selected ? review.participants.filter(participant => selected.includes(identityParticipantKey(participant))) : review.participants;
  }

  function identityReviewCards(reviews: HubIdentityReview[]) {
    return <div className={styles.root}>{reviews.map(review => {
        const selected = selectedIdentityParticipants(review);
        const profileIds = selected.map(participant => participant.profile_id).filter((id): id is number => id != null);
        const distinctProfiles = new Set(profileIds);
        const selectedHasSameProfile = distinctProfiles.size < profileIds.length;
        const selectedIsOneProfile = profileIds.length === selected.length && distinctProfiles.size === 1;
        const canMergeSelection = distinctProfiles.size <= 1 && (selected.length !== review.participants.length || review.can_merge);
        const reviewDisabled = disabled || selected.length < 2;
        const selectedLabel = selected.map(hubPlayerLabel).join(" and ");
        return <article key={review.id} className={styles.groupCard} data-review-id={review.id} aria-label={`Identity review for ${review.name}`}>
          <div className={styles.heading}><h4 style={{ margin: 0 }}>{review.name}</h4><span className={`${styles.badge} ${review.status === "pending" ? styles.waiting : ""}`}>{review.status === "pending" ? "Needs review" : review.status === "same" ? "Same person" : review.status === "different" ? "Different people" : "Some entries share a person"}</span></div>
          <div className={styles.playerList}>{review.participants.map(participant => {
            const label = hubPlayerLabel(participant);
            const key = identityParticipantKey(participant);
            return <div key={key} className={styles.player}><div>{review.participants.length > 2 ? <label className={styles.checkbox}><input type="checkbox" checked={selected.some(item => identityParticipantKey(item) === key)} disabled={disabled} aria-label={`Select ${label} from ${participant.group_title} for identity decision`} onChange={event => {
              const keys = selected.map(identityParticipantKey);
              setIdentitySelections(current => ({ ...current, [review.id]: event.target.checked ? [...keys, key] : keys.filter(item => item !== key) }));
            }} /><strong>{label}</strong></label> : <strong>{label}</strong>}<p className={styles.small}>{participant.group_title} · {participant.profile_id == null ? "Guest player" : "Linked club player"}</p></div></div>;
          })}</div>
          {review.participants.length > 2 ? <p className={styles.small}>Select at least two entries to decide together. You can make separate decisions for the remaining entries. To split a previous same-person decision, select all entries that share that person.</p> : null}
          {review.assigned ? <p className={styles.small}>Finish or release affected assignments before marking these as the same person.</p> : null}
          {!canMergeSelection && selected.length >= 2 ? <p className={styles.small}>These entries link to different club profiles and must stay separate.</p> : null}
          {selectedHasSameProfile ? <p className={styles.small}>{selectedIsOneProfile ? "Entries linked to the same club profile belong to one person and cannot be split here." : "Entries linked to the same club profile stay together when other selected entries are separated."}</p> : null}
          <div className={styles.actions}><button type="button" className={styles.button} aria-label={`Same person: ${selectedLabel}`} disabled={reviewDisabled || review.assigned || !canMergeSelection} onClick={() => void command("resolve_identities", { resolution: "same", participants: selected.map(participant => ({ session_key: participant.session_key, participant_id: participant.participant_id })) })}>Same person</button><button type="button" className={styles.button} aria-label={`Different people: ${selectedLabel}`} disabled={reviewDisabled || selectedIsOneProfile} onClick={() => void command("resolve_identities", { resolution: "different", participants: selected.map(participant => ({ session_key: participant.session_key, participant_id: participant.participant_id })) })}>Different people</button></div>
        </article>;
      })}</div>;
  }

  function reviewIdentityNames() {
    return <section className={styles.card} aria-label="Matching player names">
      <div className={styles.heading}><div><h3>Matching player names</h3><p className={styles.muted}>Choose whether these entries belong to the same person or to different people. Decisions apply to this event’s court queue and pauses; round robin scores stay unchanged.</p></div><button type="button" className={styles.button} onClick={() => setIdentityReviewOpen(false)} disabled={identityPendingCount > 0}>Hide identity review</button></div>
      {identityPendingCount > 0 ? <div className={styles.notice}><p><strong>{identityPendingCount} matching {identityPendingCount === 1 ? "name needs" : "names need"} review.</strong> Only queued matches involving these entries are waiting for a decision.</p><div className={styles.actions}><button type="button" className={styles.primary} disabled={disabled} onClick={() => void command("resolve_identities", { resolution: "all_different" })}>These are all different people</button></div><p className={styles.small}>This resolves all pending matching names as different people. Previously confirmed same-person entries and linked player profiles stay together.</p></div> : <p className={styles.small}>No matching names need review. Your decisions are saved.</p>}
      {pendingIdentityReviews.length > 0 ? identityReviewCards(pendingIdentityReviews) : null}
      {savedIdentityReviews.length > 0 ? <div style={{ marginTop: "1rem" }}><button type="button" className={styles.button} aria-expanded={savedIdentityReviewOpen} aria-controls="hub-saved-identity-decisions" onClick={() => setSavedIdentityReviewOpen(open => !open)}>Saved decisions ({savedIdentityReviews.length})</button>{savedIdentityReviewOpen ? <section id="hub-saved-identity-decisions" aria-label="Saved identity decisions"><p className={styles.small}>Correct a previous decision here. Merging entries requires their court assignments to be finished or released first.</p>{identityReviewCards(savedIdentityReviews)}</section> : null}</div> : null}
    </section>;
  }

  function queueRows(matches: HubMatch[], kind: "ready" | "waiting" | "blocked") {
    return <ol className={styles.queue}>{matches.map((match, index) => {
      const openCourt = workspace?.courts.some(court => court.open && !court.current && !court.next && match.allowed_court_ids.includes(court.id));
      const courtChoice = workspace?.courts.some(court => court.open && !court.next && match.allowed_court_ids.includes(court.id));
      return <li key={`${match.session_key}:${match.match_id}`}>
        <span className={styles.position}>{kind === "ready" ? index + 1 : "•"}</span>
        <div className={styles.queueCopy}><strong>{hubMatchLabel(match)}</strong><span className={styles.small}>{groupMap.get(match.session_key)?.title || match.label} · Round {match.round_number}</span>
          <span className={styles.small}>Longest player wait: {hubWaitLabel(match.wait_seconds)}{kind === "waiting" ? ` · Next on ${workspace?.courts.find(court => court.id === match.reserved_court_id)?.label || "reserved court"}` : ""}</span>
          {match.blockers.length > 0 ? <span className={styles.small}>{match.blockers.join(" · ")}</span> : null}
        </div>
        <div className={styles.actions}>{kind === "ready" ? <><button type="button" className={styles.primary} disabled={disabled || !openCourt} onClick={() => void command("assign_next_court", { session_key: match.session_key, match_id: match.match_id })}>Next open court</button><button type="button" className={styles.button} disabled={disabled || !courtChoice} onClick={() => openCourtPicker(match)}>Choose court</button></> : kind === "waiting" ? <button type="button" className={styles.button} disabled={disabled} onClick={() => void command("cancel_reservation", { court_id: match.reserved_court_id, session_key: match.session_key, match_id: match.match_id })}>Cancel wait</button> : <span className={`${styles.badge} ${styles.waiting}`}>Waiting</span>}</div>
      </li>;
    })}</ol>;
  }

  const selectedScoreCourt = scoreEditor && workspace?.courts.find(court => court.id === scoreEditor.courtId);
  const scoreMatchChanged = Boolean(scoreEditor && (!selectedScoreCourt?.current || selectedScoreCourt.current.match_id !== scoreEditor.match.match_id || selectedScoreCourt.current.session_key !== scoreEditor.match.session_key));
  const scoreStale = Boolean(scoreEditor && String(scoreEditor.version) !== String(detail?.hub.version));
  const scoreA = scoreEditor?.scoreA.trim() || "", scoreB = scoreEditor?.scoreB.trim() || "";
  const scoreValid = /^\d+$/.test(scoreA) && /^\d+$/.test(scoreB) && Number(scoreA) <= 99 && Number(scoreB) <= 99 && Number(scoreA) !== Number(scoreB);
  const targetCourt = courtEditor && workspace?.courts.find(court => court.id === courtEditor.targetCourtId);
  const choiceCourts = courtEditor ? (workspace?.courts || []).filter(court => courtEditor.match.allowed_court_ids.includes(court.id) && court.open && !court.next && court.id !== courtEditor.sourceCourtId && (!courtEditor.sourceCourtId || !court.current)) : [];
  const unresolved = (workspace?.eligible_queue.length || 0) + (workspace?.blocked_queue.length || 0) + (workspace?.reserved_queue.length || 0) + (workspace?.courts.filter(court => court.current).length || 0);

  if (!apiBase) return <p role="alert">The round robin service is not configured.</p>;
  if (!accessToken) return <p>Sign in to manage a round robin hub.</p>;

  return <div className={styles.root}>
    <section className={styles.scope} aria-label="Choose event">
      <label className={styles.field}>Hub event<select aria-label="Select hub" value={hubId} disabled={busy || attaching || Boolean(pending)} onChange={event => selectHub(event.target.value)}><option value="">Choose an event</option>{list.hubs.map(hub => <option key={hub.id} value={hub.id}>{hub.title}{hub.status === "completed" ? " · Finished" : ""}</option>)}</select></label>
      <button type="button" className={styles.button} disabled={busy || attaching || Boolean(pending)} onClick={() => { setCreating(true); setSessionView(null); setError(""); }}>New event</button>
      <button type="button" className={styles.button} disabled={busy || attaching || loading} onClick={() => { void loadList().catch(cause => setError(cause.message)); if (hubId) void loadHub(hubId); }}>Refresh</button>
    </section>

    {error ? <div role="alert" className={`${styles.notice} ${styles.error}`}><p>{error}</p></div> : null}
    {message ? <div role="status" className={`${styles.notice} ${styles.success}`}><p>{message}</p></div> : null}
    {pending ? <section className={`${styles.notice} ${styles.error}`} aria-label="Confirm pending action"><strong>Confirm the previous action</strong><p>Its exact request is saved. Retrying checks the same action without creating a second one.</p><button type="button" className={styles.button} disabled={busy} onClick={() => void execute(pending)}>{busy ? "Checking…" : "Retry saved action"}</button></section> : null}
    {createdSession ? <section className={styles.notice} aria-label="Add created round robin"><strong>Round robin created</strong><p>{attaching ? "Adding your round robin to this event…" : "Your round robin is saved. Complete its addition to the hub to make its matches available for courts."}</p><div className={styles.actions}><Link className={styles.button} href={`/admin/round-robin-generator/sessions/${encodeURIComponent(createdSession.sessionKey)}/rounds/1`}>Open created round robin</Link>{!pending && !attaching ? <button type="button" className={styles.primary} disabled={busy} onClick={() => { if (returnIntent.current) void attachCreatedSession(returnIntent.current); }}>Retry adding to hub</button> : null}</div></section> : null}

    {creating ? <section className={styles.card} aria-label="Create event"><div className={styles.heading}><div><h2>Create an event</h2><p className={styles.muted}>Name this gathering and choose its courts. Then add your existing round robins from the Round robins tab.</p></div>{detail ? <button type="button" className={styles.button} onClick={() => setCreating(false)}>Cancel</button> : null}</div>
      <div className={styles.inlineForm}><label className={styles.field}>Event name<input value={title} maxLength={160} placeholder="Friday Pickleball Night" onChange={event => setTitle(event.target.value)} /></label><label className={styles.field} style={{ flex: "0 1 10rem" }}>Number of courts<input type="number" min={1} max={40} value={createCourts.length} onChange={event => { const count = Math.max(1, Math.min(40, Number(event.target.value) || 1)); setCreateCourts(current => count <= current.length ? current.slice(0, count) : [...current, ...newCourts(count).slice(current.length).map(court => ({ ...court, id: hubOperationKey() }))]); }} /></label></div>
      {courtEditorRows(createCourts, setCreateCourts, false)}
      <button type="button" className={styles.primary} disabled={busy || Boolean(pending) || !title.trim() || createCourts.some(court => !court.label.trim())} onClick={() => void execute({ hubId: "", body: { title: title.trim(), courts: createCourts.map(courtConfig), idempotency_key: hubOperationKey() } })}>{busy ? "Creating…" : "Create event"}</button>
    </section> : null}

    {loading && !detail ? <p role="status">Loading court board…</p> : null}
    {detail && workspace && !creating ? <>
      <div className={styles.heading}><div><h2 style={{ marginBottom: ".4rem" }}>{detail.hub.title}</h2><div className={styles.summary}><span><strong>{workspace.groups.length}</strong> round robins</span><span><strong>{workspace.courts.filter(court => court.open && !court.current).length}</strong> open courts</span><span><strong>{workspace.eligible_queue.length}</strong> ready matches</span><span><strong>{workspace.players.filter(player => player.paused).length}</strong> paused players</span></div></div><div className={styles.actions}><button type="button" className={styles.button} disabled={disabled || !activeGroups.length} onClick={openFinishGroup}>Finish round robin</button><span className={`${styles.badge} ${detail.hub.status === "completed" ? "" : styles.good}`}>{detail.hub.status === "completed" ? "Finished" : "Live event"}</span></div></div>
      <nav className={styles.nav} role="tablist" aria-label="Hub views">{([['board', 'Court board'], ['queue', 'Match queue'], ['groups', 'Round robins'], ['standings', 'Standings'], ['players', 'Players'], ['registration', 'Registration'], ['courts', 'Court setup']] as const).map(([value, label]) => <button type="button" key={value} role="tab" id={`hub-tab-${value}`} aria-selected={tab === value} aria-controls={`hub-panel-${value}`} onClick={() => { window.history.replaceState(null, "", `/admin/round-robin-hub?hub=${encodeURIComponent(hubId)}${value === "registration" ? "&view=registration" : ""}`); if (value === "courts") openCourtSetup(); else { if (value === "standings") setSessionView(null); if (value === "groups") void loadList().catch(cause => setError(cause instanceof Error ? cause.message : "Unable to refresh round robins.")); setTab(value); } }}>{label}{value === "registration" && detail.registration?.pending_count ? ` (${detail.registration.pending_count})` : ""}</button>)}</nav>

      {identityPendingCount > 0 ? <section className={styles.notice} role="status" aria-label="Player identity review needed"><div className={styles.heading}><div><strong>Review matching player names</strong><p className={styles.small}>Confirm who is playing across groups. Only their queued matches wait for review; other ready matches can keep playing.</p></div><button type="button" className={styles.button} onClick={openIdentityReview}>Review matching names</button></div></section> : null}

      {finishingGroups.length ? <section className={styles.notice} aria-label="Finishing round robins"><strong>Finishing round robins</strong>{finishingGroups.map(group => <p key={group.session_key} className={styles.small}><strong>{group.title}</strong> · Finishing after round {group.finish_after_round}{typeof group.finish_pending_matches === "number" ? ` · ${group.finish_pending_matches} ${group.finish_pending_matches === 1 ? "match" : "matches"} left` : ""}. Keep scoring or marking those matches played; results are preserved.</p>)}</section> : null}

      {tab === "board" ? <section id="hub-panel-board" role="tabpanel" aria-labelledby="hub-tab-board" className={styles.root}>
        <div className={styles.card}><div className={styles.heading}><div><h3>Ready for a court</h3><p className={styles.small}>Longest waiting players first, including time on bye.</p></div><div className={styles.actions}><button type="button" className={styles.primary} disabled={disabled || !workspace.groups.length} onClick={() => { setAddPlayerOpen(true); setRemovePlayerOpen(false); setError(""); }}>Add player</button><button type="button" className={styles.button} disabled={disabled || !workspace.groups.length} onClick={() => { setRemovePlayerOpen(true); setAddPlayerOpen(false); setError(""); }}>Remove player</button><button type="button" className={styles.button} disabled={disabled || !activeGroups.length} onClick={openFinishGroup}>Finish</button><button type="button" className={styles.primary} disabled={disabled || !canFillCourts} onClick={() => void command("auto_fill_courts")}>Fill open courts</button></div></div>{workspace.eligible_queue.length ? queueRows(workspace.eligible_queue.slice(0, 4), "ready") : <p className={styles.muted}>{workspace.groups.some(group => group.can_advance) ? "Fill open courts to refresh the queue and assign available matches." : "Waiting for enough available players. Check-ins automatically prepare matches; also check paused players and court availability."}</p>}{workspace.eligible_queue.length > 4 ? <button type="button" className={styles.button} style={{ marginTop: ".7rem" }} onClick={() => setTab("queue")}>View all {workspace.eligible_queue.length} ready matches</button> : null}{byeGroups.length > 0 ? <section className={styles.next} style={{ marginTop: ".85rem" }} aria-label="Byes this round"><h4 style={{ margin: 0 }}>Byes this round</h4>{byeGroups.map(({ group, players }) => <p key={group.session_key} className={styles.small} style={{ margin: 0 }}><strong>{group.title}</strong> · {players.map(player => player.name).join(", ")}</p>)}</section> : null}</div>
        <div className={styles.grid}>{workspace.courts.map(court => <article key={court.id} className={`${styles.court} ${court.current ? styles.occupied : ""} ${!court.open ? styles.closed : ""}`} aria-label={court.label}>
          <div className={styles.heading}><h3>{court.label}</h3><span className={`${styles.badge} ${court.current ? "" : court.open ? styles.good : ""}`}>{court.current ? "Playing" : court.open ? "Available" : "Closed"}</span></div>
          <p className={styles.small}>{court.dedicated_session_key ? `${groupMap.get(court.dedicated_session_key)?.title || "Dedicated group"} only` : "Shared court"}</p>
          {court.current ? <><p className={styles.eyebrow}>{groupMap.get(court.current.session_key)?.title} · Round {court.current.round_number}</p><strong>{hubMatchLabel(court.current)}</strong><div className={styles.actions}>{groupMap.get(court.current.session_key)?.scoring_mode === "unscored" ? <button type="button" className={styles.primary} disabled={disabled} onClick={() => void command("mark_played", { court_id: court.id, session_key: court.current!.session_key, match_id: court.current!.match_id })}>Match played</button> : <button type="button" className={styles.primary} disabled={disabled} onClick={() => { setScoreEditor({ courtId: court.id, match: court.current!, version: detail.hub.version, scoreA: "", scoreB: "" }); setError(""); }}>Enter score</button>}<button type="button" className={styles.button} disabled={disabled} onClick={() => openCourtPicker(court.current!, court.id)}>Move or remove</button></div></> : <p className={styles.muted}>{court.open ? "Assign a ready match from the queue." : "Open this court in Court setup when it is available."}</p>}
          {court.next ? <div className={styles.next}><span className={`${styles.badge} ${styles.waiting}`}>Next on this court</span><strong>{hubMatchLabel(court.next)}</strong><span className={styles.small}>{groupMap.get(court.next.session_key)?.title} · Round {court.next.round_number}</span><div><button type="button" className={styles.button} disabled={disabled} onClick={() => void command("cancel_reservation", { court_id: court.id, session_key: court.next!.session_key, match_id: court.next!.match_id })}>Cancel wait</button></div></div> : null}
        </article>)}</div>
      </section> : null}

      {tab === "queue" ? <section id="hub-panel-queue" role="tabpanel" aria-labelledby="hub-tab-queue" className={styles.root}><div className={styles.card}><h2>Ready matches · {workspace.eligible_queue.length}</h2><p className={styles.small}>Priority follows the longest waiting player in each eligible match. Scheduled byes count toward waiting time.</p>{workspace.eligible_queue.length ? queueRows(workspace.eligible_queue, "ready") : <p className={styles.empty}>No matches are ready.</p>}</div>{workspace.reserved_queue.length ? <div className={styles.card}><h2>Reserved next · {workspace.reserved_queue.length}</h2>{queueRows(workspace.reserved_queue, "waiting")}</div> : null}{workspace.blocked_queue.length ? <div className={styles.card}><h2>Waiting for players or courts · {workspace.blocked_queue.length}</h2>{queueRows(workspace.blocked_queue, "blocked")}</div> : null}</section> : null}

      {tab === "groups" ? <section id="hub-panel-groups" role="tabpanel" aria-labelledby="hub-tab-groups" className={styles.root}>
        <div className={styles.card}><div className={styles.heading}><div><h2>Add a round robin</h2><p className={styles.muted}>Each group keeps its roster, format, rating choice, and results.</p></div><Link className={styles.button} href={`/admin/round-robin-generator?hub=${encodeURIComponent(detail.hub.id)}&draft=${encodeURIComponent(createDraftToken)}`}>Create round robin</Link></div><div className={styles.inlineForm}><label className={styles.field}>Existing active round robin<select value={addSessionKey} onChange={event => setAddSessionKey(event.target.value)}><option value="">Choose a round robin</option>{availableSessions.map(session => <option key={session.session_key} value={session.session_key}>{session.title} · {formatName(session.event_format)} · {session.rating_mode || "unrated"}</option>)}</select></label><button type="button" className={styles.primary} disabled={disabled || !availableSessions.some(session => session.session_key === addSessionKey)} onClick={() => void command("add_group", { session_key: addSessionKey }).then(ok => { if (ok) setAddSessionKey(""); })}>Add to hub</button></div>{!availableSessions.length ? <p className={styles.small}>No other active round robins are available. Create a round robin to add it to this event and queue its matches.</p> : null}</div>
        <div className={styles.grid}>{workspace.groups.map(group => {
          const dedicated = workspace.courts.filter(court => court.dedicated_session_key === group.session_key);
          const groupBusy = workspace.courts.some(court => court.current?.session_key === group.session_key || court.next?.session_key === group.session_key);
          const meta = list.sessions.find(session => session.session_key === group.session_key);
          return <article className={styles.groupCard} key={group.session_key}><div className={styles.heading}><h3>{group.title}</h3><span className={styles.badge}>{formatName(group.event_format)}</span></div>{group.registration_driven && group.status === "active" ? <p className={styles.notice}>{group.active_player_count || 0} checked in · {(group.active_player_count || 0) < (group.minimum_players || 4) ? `Waiting for ${(group.minimum_players || 4) - (group.active_player_count || 0)} more to prepare a match` : "Matches prepare automatically as players become available"}</p> : null}<p className={styles.small}>{group.scoring_mode === "unscored" ? "Unscored" : (group.rating_mode || meta?.rating_mode) === "rated" ? "Rated · approval required" : "Unrated"} · {dedicated.length ? dedicated.map(court => court.label).join(", ") : "Shared courts"}</p><p><strong>{group.status === "completed" ? "Round" : "Completing round"} {group.current_round_number}</strong> · {group.completed_matches} of {group.total_matches} matches complete</p><div className={styles.progress} role="progressbar" aria-label={`${group.title} round completion`} aria-valuemin={0} aria-valuemax={group.total_matches || 1} aria-valuenow={group.completed_matches}><span style={{ width: `${group.total_matches ? Math.min(100, group.completed_matches / group.total_matches * 100) : 0}%` }} /></div><p className={styles.small}>{group.status === "completed" ? "Session finished. Results remain available." : group.can_advance && group.advance_label === "Finish session" ? "Final round complete. Finish this session when ready." : group.scoring_mode === "unscored" ? "Match completion is saved without scores. Players can take their next eligible match as courts open." : ["saved", "played"].includes(group.round_status) ? "Round complete. These scores count toward standings." : "Scores are saved as games finish. Standings update when this round is complete. Players can take their next eligible match as courts open."}</p><div className={styles.actions}>{group.can_advance ? <button type="button" className={styles.primary} disabled={disabled} onClick={() => void command("advance_round", { session_key: group.session_key })}>{group.advance_label || "Next round"}</button> : null}<button type="button" className={styles.button} onClick={() => showSession(group)}>Manage round robin</button>{group.scoring_mode !== "unscored" ? <button type="button" className={styles.button} onClick={() => showSession(group, "standings")}>Standings</button> : null}</div><div><button type="button" className={styles.button} disabled={disabled || groupBusy || dedicated.length > 0} title={dedicated.length ? "Return dedicated courts to shared use before removing this group." : undefined} onClick={() => void command("remove_group", { session_key: group.session_key })}>Remove from hub</button></div></article>;
        })}</div>
      </section> : null}

      {tab === "standings" ? <section id="hub-panel-standings" role="tabpanel" aria-labelledby="hub-tab-standings" className={styles.root}>
        <div className={styles.card}><h2>Standings</h2><p className={styles.muted}>Choose a round robin to view results and share its standings.</p>
          <label className={styles.field}>Round robin<select aria-label="Standings round robin" value={standingsGroup?.session_key || ""} disabled={!workspace.groups.length} onChange={event => { const group = groupMap.get(event.target.value); if (group) showSession(group, "standings"); }}>
            {!workspace.groups.length ? <option value="">No round robins</option> : null}{workspace.groups.map(group => <option key={group.session_key} value={group.session_key}>{group.title}{group.scoring_mode === "unscored" ? " · Unscored" : ""}{group.status === "completed" ? " · Finished" : ""}</option>)}
          </select></label>
        </div>
        {standingsGroup ? <div inert={busy || attaching || Boolean(pending)}><GeneratorStandings key={`${detail.hub.id}:${standingsGroup.session_key}:standings`} apiBase={apiBase} clubId={clubId} sessionKey={standingsGroup.session_key} embedded refreshKey={detail.hub.version} onNavigate={target => navigateSession(standingsGroup.session_key, target)} onSessionChanged={async () => { await loadHub(hubId, true); }} /></div> : <p className={styles.card}>Add a round robin to this event to view its standings here.</p>}
      </section> : null}

      {tab === "players" ? <section id="hub-panel-players" role="tabpanel" aria-labelledby="hub-tab-players" className={styles.card}>
        <div className={styles.heading}><div><h2>Players</h2><p className={styles.muted}>Pause for a break and resume when ready; waiting time is held. Standard round robins use available players to refill affected waiting games. Games already started stay intact. Pausing a shared player applies across this event.</p></div><div className={styles.actions}><button type="button" className={styles.primary} disabled={disabled || !workspace.groups.length} onClick={() => { setAddPlayerOpen(true); setRemovePlayerOpen(false); setError(""); }}>Add player</button><button type="button" className={styles.button} disabled={disabled || !workspace.groups.length} onClick={() => { setRemovePlayerOpen(true); setAddPlayerOpen(false); setError(""); }}>Remove player</button>{identityReviews.length ? <button type="button" className={styles.button} onClick={openIdentityReview}>Review player identities</button> : null}<label className={styles.field}>Find player<input type="search" value={playerSearch} onChange={event => setPlayerSearch(event.target.value)} /></label></div></div>
        {identityReviews.length > 0 && (identityReviewOpen || identityPendingCount > 0) ? reviewIdentityNames() : null}
        <div className={styles.root}>{playerGroups.map(({ group, players }) => <section key={group.session_key} className={styles.groupCard} aria-label={`Players in ${group.title}`} data-player-group={group.session_key}>
          <div className={styles.heading}><div><h3>{group.title}</h3><p className={styles.small}>{players.length} {players.length === 1 ? "player" : "players"}{playerSearch.trim() ? " matching this search" : ""}</p></div><div className={styles.actions}><button type="button" className={styles.button} onClick={() => showSession(group)}>Manage round robin</button></div></div>
          {!group.can_add_player ? <p className={styles.small}>{group.add_player_unavailable_reason || "Open this round robin to manage its roster."}</p> : null}
          <div className={styles.playerList}>{players.map(player => {
            const playingCourt = workspace.courts.find(court => court.current && [...court.current.side_a, ...court.current.side_b].some(person => person.identity === player.identity));
            const reservedCourt = workspace.courts.find(court => court.next && [...court.next.side_a, ...court.next.side_b].some(person => person.identity === player.identity));
            const joiningLater = !player.active && (player.active_from_round || 0) > group.current_round_number;
            const rosterStatus = player.active ? "" : joiningLater ? `Joins round ${player.active_from_round}` : player.inactive_from_round != null && player.inactive_from_round <= group.current_round_number ? "Left round robin" : "Not playing this round";
            return <div key={player.participant_id} className={styles.player} aria-label={`${player.name} · ${group.title}`} data-player-identity={player.identity} data-participant-id={player.participant_id}>
              <div><strong>{player.name}</strong><p className={styles.small}>{rosterStatus ? `${rosterStatus} · ` : ""}{playingCourt ? `Playing on ${playingCourt.label}` : reservedCourt ? `Reserved next on ${reservedCourt.label}` : player.active && player.on_bye ? `On bye this round · Waiting ${hubWaitLabel(player.wait_seconds)}` : `Waiting ${hubWaitLabel(player.wait_seconds)}`}{player.paused ? " · Paused" : ""}</p></div>
              <button type="button" className={player.paused ? styles.primary : styles.button} disabled={disabled || (!player.active && !joiningLater)} onClick={() => void command(player.paused ? "resume_player" : "pause_player", { identity: player.identity })}>{player.paused ? "Ready again" : "Pause player"}</button>
            </div>;
          })}</div>{!players.length ? <p className={styles.small}>No players in this round robin yet.</p> : null}
        </section>)}</div>
        {!playerGroups.length ? <p className={styles.small}>{playerSearch.trim() ? "No players match this search." : "Add a round robin to manage its players here."}</p> : null}
        <p className={styles.small}>Use Remove player for early departures. Open Manage round robin for substitutions and other roster changes.</p>
      </section> : null}

      {tab === "registration" ? <section id="hub-panel-registration" role="tabpanel" aria-labelledby="hub-tab-registration">
        <HubRegistrationPanel key={hubId} apiBase={apiBase} clubId={clubId} hubId={hubId} groups={workspace.groups} registration={detail.registration}
          disabled={disabled} busy={busy} pending={Boolean(pending)} error={error} onCommand={command} onRetry={() => { if (pending) void execute(pending); }} />
      </section> : null}

      {tab === "courts" ? <section id="hub-panel-courts" role="tabpanel" aria-labelledby="hub-tab-courts" className={styles.card}><h2>Court setup</h2><p className={styles.muted}>A group with dedicated courts uses only those courts. Other groups use the shared courts.</p>{courtEditorRows(editCourts, setEditCourts, true)}{String(courtDraftVersion) !== String(detail.hub.version) ? <p className={styles.small}>The court board has changed since this editor opened. Review current courts before saving.</p> : null}<div className={styles.actions}><button type="button" className={styles.primary} disabled={disabled || !editCourts.length || editCourts.some(court => !court.label.trim())} onClick={() => void command("configure_courts", { courts: editCourts.map(courtConfig) }, courtDraftVersion).then(ok => { if (ok) setTab("board"); })}>Save court setup</button><button type="button" className={styles.button} disabled={busy} onClick={openCourtSetup}>Reload current courts</button><button type="button" className={styles.button} disabled={disabled} onClick={() => setFinishReview(true)}>Finish event</button></div></section> : null}

      {sessionView && groupMap.has(sessionView.sessionKey) ? <section className={styles.embedded} aria-label="Manage selected round robin"><div className={styles.heading}><div><p className={styles.eyebrow}>Round robin controls</p><h2>{groupMap.get(sessionView.sessionKey)?.title || "Round robin"}</h2></div><button type="button" className={styles.button} onClick={() => setSessionView(null)}>Close round robin controls</button></div><div className={styles.embeddedBody} inert={busy || attaching || Boolean(pending)}><GeneratorRoundRunner key={`${detail.hub.id}:${sessionView.sessionKey}:round:${sessionView.roundNumber}`} apiBase={apiBase} clubId={clubId} generatorKind="round_robin" sessionKey={sessionView.sessionKey} roundNumber={sessionView.roundNumber} embedded refreshKey={detail.hub.version} onNavigate={target => navigateSession(sessionView.sessionKey, target)} onSessionChanged={async () => { await loadHub(hubId, true); }} /></div></section> : null}
    </> : null}

    {finishGroupOpen && workspace ? <HubFinishDialog key={hubId} groups={workspace.groups}
      busy={busy} disabled={disabled} pending={Boolean(pending)} error={error}
      onClose={() => setFinishGroupOpen(false)} onRetry={() => { if (pending) void execute(pending); }}
      onSubmit={sessionKey => { void command("finish_group", { session_key: sessionKey }); }} /> : null}

    {addPlayerOpen && workspace ? <HubAddPlayerDialog key={hubId} apiBase={apiBase} clubId={clubId} groups={workspace.groups}
      busy={busy} disabled={disabled} pending={Boolean(pending)} error={error}
      onClose={() => setAddPlayerOpen(false)} onRetry={() => { if (pending) void execute(pending); }}
      onSubmit={(sessionKey, name, playerId) => { void command("add_player", { session_key: sessionKey, name, ...(playerId == null ? {} : { player_id: playerId }) }); }} /> : null}

    {removePlayerOpen && workspace ? <HubRemovePlayerDialog key={hubId} groups={workspace.groups}
      busy={busy} disabled={disabled} pending={Boolean(pending)} error={error}
      onClose={() => setRemovePlayerOpen(false)} onRetry={() => { if (pending) void execute(pending); }}
      onSubmit={(sessionKey, participantId) => { void command("remove_player", { session_key: sessionKey, participant_id: participantId }); }} /> : null}

    {scoreEditor ? <InteractionDialog open phase={busy ? "working" : "ready"} title={`Enter score · ${selectedScoreCourt?.label || "Court"}`} description={<p>{groupMap.get(scoreEditor.match.session_key)?.title} · Round {scoreEditor.match.round_number}</p>} initialFocusRef={scoreInput} dirty={Boolean(scoreEditor.scoreA || scoreEditor.scoreB)} onRequestClose={() => { if (!busy) setScoreEditor(null); }} actions={<><button type="button" className={styles.button} disabled={busy} onClick={() => setScoreEditor(null)}>Cancel</button><button type="submit" form="hub-score-form" className={styles.primary} disabled={disabled || !scoreValid || scoreMatchChanged || scoreStale}>Save score &amp; release court</button></>}>
      <form id="hub-score-form" onSubmit={event => { event.preventDefault(); if (scoreValid && !scoreStale && !scoreMatchChanged) void command("score_and_release", { court_id: scoreEditor.courtId, session_key: scoreEditor.match.session_key, match_id: scoreEditor.match.match_id, score_a: Number(scoreA), score_b: Number(scoreB) }, scoreEditor.version); }}><div className={styles.scoreGrid}><label className={styles.field}>{scoreEditor.match.side_a.map(player => player.name).join(" / ")}<input ref={scoreInput} aria-label="Side A score" inputMode="numeric" type="number" min={0} max={99} value={scoreEditor.scoreA} onChange={event => setScoreEditor({ ...scoreEditor, scoreA: event.target.value })} /></label><label className={styles.field}>{scoreEditor.match.side_b.map(player => player.name).join(" / ")}<input aria-label="Side B score" inputMode="numeric" type="number" min={0} max={99} value={scoreEditor.scoreB} onChange={event => setScoreEditor({ ...scoreEditor, scoreB: event.target.value })} /></label></div></form><p className={styles.small}>This saves the match and frees its court. Players can take their next eligible match as courts open. Standings update only when the whole round is complete.</p>{scoreA && scoreB && !scoreValid ? <p role="alert">Enter two different whole-number scores between 0 and 99.</p> : null}{scoreMatchChanged ? <p role="alert">This court now has a different match. Close this score entry and review the court board.</p> : scoreStale ? <div className={styles.notice}><p>The court board changed while this score was open. The same match is still on court; review it before saving.</p><button type="button" className={styles.button} onClick={() => setScoreEditor({ ...scoreEditor, version: detail!.hub.version })}>Reviewed — keep these scores</button></div> : null}{error ? <p role="alert">{error}</p> : null}{pending ? <button type="button" className={styles.button} disabled={busy} onClick={() => void execute(pending)}>Retry saved action</button> : null}
    </InteractionDialog> : null}

    {courtEditor ? <InteractionDialog open phase={busy ? "working" : "ready"} title={courtEditor.sourceCourtId ? "Move or remove match" : "Choose court"} description={<><p>{groupMap.get(courtEditor.match.session_key)?.title} · Round {courtEditor.match.round_number}</p><p>{hubMatchLabel(courtEditor.match)}</p></>} onRequestClose={() => { if (!busy) setCourtEditor(null); }} actions={<><button type="button" className={styles.button} disabled={busy} onClick={() => setCourtEditor(null)}>Cancel</button>{courtEditor.sourceCourtId ? <button type="button" className={styles.button} disabled={disabled} onClick={() => void command("release_game", { court_id: courtEditor.sourceCourtId, session_key: courtEditor.match.session_key, match_id: courtEditor.match.match_id }, courtEditor.version)}>Return to queue</button> : null}<button type="button" className={styles.primary} disabled={disabled || !targetCourt || !choiceCourts.some(court => court.id === targetCourt.id)} onClick={() => void command(courtEditor.sourceCourtId ? "move_game" : targetCourt?.current ? "reserve_game_for_court" : "assign_game", courtEditor.sourceCourtId ? { court_id: courtEditor.sourceCourtId, target_court_id: courtEditor.targetCourtId, session_key: courtEditor.match.session_key, match_id: courtEditor.match.match_id } : { session_key: courtEditor.match.session_key, match_id: courtEditor.match.match_id, court_id: courtEditor.targetCourtId }, courtEditor.version)}>{courtEditor.sourceCourtId ? "Move match" : targetCourt?.current ? "Reserve next" : "Assign to court"}</button></>}>
      <label className={styles.field}>Court<select value={courtEditor.targetCourtId} onChange={event => setCourtEditor({ ...courtEditor, targetCourtId: event.target.value })}><option value="">Choose a court</option>{choiceCourts.map(court => <option key={court.id} value={court.id}>{court.label}{court.current ? " · Reserve next" : " · Available"}</option>)}</select></label><p className={styles.small}>Only courts available to this group are shown. Dedicated courts are exclusive.</p>{!choiceCourts.length ? <p>No eligible court is currently available for this action.</p> : null}{error ? <p role="alert">{error}</p> : null}{pending ? <button type="button" className={styles.button} disabled={busy} onClick={() => void execute(pending)}>Retry saved action</button> : null}
    </InteractionDialog> : null}

    {finishReview ? <InteractionDialog open phase={busy ? "working" : "ready"} title="Finish this event?" description={<p>{detail?.hub.title}</p>} onRequestClose={() => { if (!busy) setFinishReview(false); }} actions={<><button type="button" className={styles.button} disabled={busy} onClick={() => setFinishReview(false)}>Keep event open</button><button type="button" className={styles.primary} disabled={disabled || unresolved > 0} onClick={() => void command("finish_hub")}>Finish event</button></>}><p>{unresolved ? `${unresolved} matches are playing, waiting, or reserved. Resolve these before finishing.` : "The court queue is clear. Individual round robin results remain available."}</p><p className={styles.small}>Use Finish round robin to end each group without losing its queued matches. The event closes automatically when all groups are finished.</p>{error ? <p role="alert">{error}</p> : null}{pending ? <button type="button" className={styles.button} disabled={busy} onClick={() => void execute(pending)}>Retry saved action</button> : null}</InteractionDialog> : null}
  </div>;
}
