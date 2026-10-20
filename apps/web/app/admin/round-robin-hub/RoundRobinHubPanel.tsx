"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { InteractionDialog } from "@/components/interaction";
import GeneratorRoundRunner from "@/app/admin/play-generators/GeneratorRoundRunner";
import GeneratorStandings from "@/app/admin/play-generators/GeneratorStandings";
import { useAdminSession } from "@/lib/useAdminSession";
import {
  hubMatchLabel, hubOperationKey, hubWaitLabel, isUncertainHubError, requestRoundRobinHub, RoundRobinHubApiError,
  type CreateHubInput, type HubCommand, type HubCommandAction, type HubCourtConfig, type HubDetailResponse,
  type HubGroup, type HubListResponse, type HubMatch
} from "@/lib/adminRoundRobinHubApi";
import styles from "./RoundRobinHubPanel.module.css";

type Props = { apiBase: string | null; clubId: string; initialHubId?: string; initialAddSessionKey?: string };
type Tab = "board" | "queue" | "groups" | "players" | "courts";
type Pending = { hubId: string; body: HubCommand | CreateHubInput };
type CreatedSession = { hubId: string; sessionKey: string };
type ReturnIntent = CreatedSession & { attempted: boolean };
type ScoreEditor = { courtId: string; match: HubMatch; version: string | number; scoreA: string; scoreB: string };
type CourtEditor = { match: HubMatch; sourceCourtId: string; targetCourtId: string; version: string | number };
type SessionView = { sessionKey: string; view: "round" | "standings"; roundNumber: number };

const emptyList: HubListResponse = { ok: true, hubs: [], sessions: [] };
const commandMessages: Partial<Record<HubCommandAction, string>> = {
  add_group: "Round robin added to the hub.", remove_group: "Round robin removed from the hub. Its results are preserved.",
  configure_courts: "Court setup saved.", assign_next_court: "Match assigned to an open court.", assign_game: "Match assigned.",
  reserve_game_for_court: "Match reserved next on this court.", cancel_reservation: "Court reservation canceled.",
  move_game: "Match moved.", release_game: "Match returned to the queue.", score_and_release: "Score saved and court released.",
  mark_played: "Match marked played and court released.", pause_player: "Player paused. Their waiting time is held.",
  resume_player: "Player is ready again.", advance_round: "Next round is ready for the court queue.", finish_hub: "Event finished.",
  auto_fill_courts: "Available courts filled with eligible matches."
};

function newCourts(count: number): HubCourtConfig[] {
  return Array.from({ length: count }, (_, index) => ({ id: `court-${index + 1}`, label: `Court ${index + 1}`, dedicated_session_key: null, open: true }));
}
function courtConfig(court: HubCourtConfig): HubCourtConfig {
  return { id: court.id, label: court.label, dedicated_session_key: court.dedicated_session_key || null, open: court.open };
}
function formatName(value?: string): string { return value === "moneyball_expectations" ? "Moneyball" : "Round robin"; }
function pendingStorageKey(clubId: string): string { return `rr-hub-pending:${clubId}`; }
function readPending(clubId: string): Pending | null {
  try {
    const value = JSON.parse(sessionStorage.getItem(pendingStorageKey(clubId)) || "null");
    return value && typeof value.hubId === "string" && value.body && typeof value.body.idempotency_key === "string" ? value as Pending : null;
  } catch { return null; }
}

export default function RoundRobinHubPanel({ apiBase, clubId, initialHubId = "", initialAddSessionKey = "" }: Props) {
  const { accessToken } = useAdminSession();
  const [list, setList] = useState<HubListResponse>(emptyList);
  const [hubId, setHubId] = useState(initialHubId);
  const [detail, setDetail] = useState<HubDetailResponse | null>(null);
  const [tab, setTab] = useState<Tab>(initialAddSessionKey ? "groups" : "board");
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
  const [scoreEditor, setScoreEditor] = useState<ScoreEditor | null>(null);
  const [courtEditor, setCourtEditor] = useState<CourtEditor | null>(null);
  const [sessionView, setSessionView] = useState<SessionView | null>(null);
  const [finishReview, setFinishReview] = useState(false);
  const selectedRef = useRef(hubId);
  const loadSequence = useRef(0);
  const mutationLock = useRef(false);
  const returnIntent = useRef<ReturnIntent | null>(null);
  const attachmentInFlight = useRef<ReturnIntent | null>(null);
  const scoreInput = useRef<HTMLInputElement>(null);
  const ready = Boolean(apiBase && accessToken);
  const workspace = detail?.workspace;
  const disabled = busy || attaching || Boolean(pending) || detail?.hub.status === "completed";
  const groupMap = useMemo(() => new Map((workspace?.groups || []).map(group => [group.session_key, group])), [workspace?.groups]);
  const availableSessions = list.sessions.filter(session => !session.round_robin_hub_id && session.status === "active" && !groupMap.has(session.session_key));

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
    const value = await request<HubListResponse>();
    setList(value);
    return value;
  }, [ready, request]);

  const loadHub = useCallback(async (id: string, silent = false) => {
    if (!ready || !id) return;
    const sequence = ++loadSequence.current;
    if (!silent) setLoading(true);
    try {
      const value = await request<HubDetailResponse>(`/${encodeURIComponent(id)}`);
      if (String(value.hub.id) !== id) throw new Error("The server returned a different event. Reload the hub before continuing.");
      if (sequence === loadSequence.current && selectedRef.current === id) {
        setDetail(value);
        if (!silent) setError("");
      }
      return value;
    } catch (cause) {
      if (sequence === loadSequence.current && selectedRef.current === id) setError(cause instanceof Error ? cause.message : "Unable to load this event.");
    } finally {
      if (sequence === loadSequence.current && selectedRef.current === id) setLoading(false);
    }
  }, [ready, request]);

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
    if (!hubId || !ready) return;
    const refresh = () => { if (document.visibilityState === "visible" && !mutationLock.current) void loadHub(hubId, true); };
    const timer = window.setInterval(refresh, 15_000);
    window.addEventListener("focus", refresh); document.addEventListener("visibilitychange", refresh);
    return () => { window.clearInterval(timer); window.removeEventListener("focus", refresh); document.removeEventListener("visibilitychange", refresh); };
  }, [hubId, ready, loadHub]);

  const completeCreatedSession = useCallback((value: HubDetailResponse): boolean => {
    const intent = returnIntent.current;
    if (!intent || value.hub.id !== intent.hubId || !value.workspace.groups.some(group => group.session_key === intent.sessionKey)) return false;
    returnIntent.current = null;
    setCreatedSession(null); setAddSessionKey("");
    selectedRef.current = intent.hubId;
    setHubId(intent.hubId); setDetail(value); setCreating(false); setTab("board"); setError("");
    setMessage("Round robin added. Choose Next open court or Fill open courts to start play.");
    window.history.replaceState(null, "", `/admin/round-robin-hub?hub=${encodeURIComponent(intent.hubId)}`);
    return true;
  }, []);

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
      if (selectedRef.current === value.hub.id) setDetail(value);
      setMessage("action" in operation.body ? commandMessages[operation.body.action] || "Hub updated." : "Event created. Add your round robins below.");
      setScoreEditor(null); setCourtEditor(null); setFinishReview(false);
      if (!completeCreatedSession(value) && (!operation.hubId || ("action" in operation.body && operation.body.action === "add_group"))) setTab("groups");
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
  }, [request, savePending, loadList, loadHub, completeCreatedSession]);

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
      setDetail(value);
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
  }, [pending, request, execute, completeCreatedSession]);

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
    setSessionView({ sessionKey: group.session_key, view, roundNumber: group.current_round_number });
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
      <label className={styles.field}>Event<select aria-label="Select hub" value={hubId} disabled={busy || attaching || Boolean(pending)} onChange={event => selectHub(event.target.value)}><option value="">Choose an event</option>{list.hubs.map(hub => <option key={hub.id} value={hub.id}>{hub.title}{hub.status === "completed" ? " · Finished" : ""}</option>)}</select></label>
      <button type="button" className={styles.button} disabled={busy || attaching || Boolean(pending)} onClick={() => { setCreating(true); setSessionView(null); setError(""); }}>New event</button>
      <button type="button" className={styles.button} disabled={busy || attaching || loading} onClick={() => { void loadList().catch(cause => setError(cause.message)); if (hubId) void loadHub(hubId); }}>Refresh</button>
    </section>

    {error ? <div role="alert" className={`${styles.notice} ${styles.error}`}><p>{error}</p></div> : null}
    {message ? <div role="status" className={`${styles.notice} ${styles.success}`}><p>{message}</p></div> : null}
    {pending ? <section className={`${styles.notice} ${styles.error}`} aria-label="Confirm pending action"><strong>Confirm the previous action</strong><p>Its exact request is saved. Retrying checks the same action without creating a second one.</p><button type="button" className={styles.button} disabled={busy} onClick={() => void execute(pending)}>{busy ? "Checking…" : "Retry saved action"}</button></section> : null}
    {createdSession ? <section className={styles.notice} aria-label="Add created round robin"><strong>Round robin created</strong><p>{attaching ? "Adding your round robin to this event…" : "Your round robin is saved. Complete its addition to the hub to make its matches available for courts."}</p><div className={styles.actions}><Link className={styles.button} href={`/admin/round-robin-generator/sessions/${encodeURIComponent(createdSession.sessionKey)}/rounds/1`}>Open created round robin</Link>{!pending && !attaching ? <button type="button" className={styles.primary} disabled={busy} onClick={() => { if (returnIntent.current) void attachCreatedSession(returnIntent.current); }}>Retry adding to hub</button> : null}</div></section> : null}

    {creating ? <section className={styles.card} aria-label="Create event"><div className={styles.heading}><div><h2>Create an event</h2><p className={styles.muted}>Name this gathering and choose the courts available to its round robins.</p></div>{detail ? <button type="button" className={styles.button} onClick={() => setCreating(false)}>Cancel</button> : null}</div>
      <div className={styles.inlineForm}><label className={styles.field}>Event name<input value={title} maxLength={160} placeholder="Friday Pickleball Night" onChange={event => setTitle(event.target.value)} /></label><label className={styles.field} style={{ flex: "0 1 10rem" }}>Number of courts<input type="number" min={1} max={40} value={createCourts.length} onChange={event => { const count = Math.max(1, Math.min(40, Number(event.target.value) || 1)); setCreateCourts(current => count <= current.length ? current.slice(0, count) : [...current, ...newCourts(count).slice(current.length).map(court => ({ ...court, id: hubOperationKey() }))]); }} /></label></div>
      {courtEditorRows(createCourts, setCreateCourts, false)}
      <button type="button" className={styles.primary} disabled={busy || Boolean(pending) || !title.trim() || createCourts.some(court => !court.label.trim())} onClick={() => void execute({ hubId: "", body: { title: title.trim(), courts: createCourts.map(courtConfig), idempotency_key: hubOperationKey() } })}>{busy ? "Creating…" : "Create event"}</button>
    </section> : null}

    {loading && !detail ? <p role="status">Loading court board…</p> : null}
    {detail && workspace && !creating ? <>
      <div className={styles.heading}><div><h2 style={{ marginBottom: ".4rem" }}>{detail.hub.title}</h2><div className={styles.summary}><span><strong>{workspace.groups.length}</strong> round robins</span><span><strong>{workspace.courts.filter(court => court.open && !court.current).length}</strong> open courts</span><span><strong>{workspace.eligible_queue.length}</strong> ready matches</span><span><strong>{workspace.players.filter(player => player.paused).length}</strong> paused players</span></div></div><span className={`${styles.badge} ${detail.hub.status === "completed" ? "" : styles.good}`}>{detail.hub.status === "completed" ? "Finished" : "Live event"}</span></div>
      <nav className={styles.nav} role="tablist" aria-label="Hub views">{([['board', 'Court board'], ['queue', 'Match queue'], ['groups', 'Round robins'], ['players', 'Players'], ['courts', 'Court setup']] as const).map(([value, label]) => <button type="button" key={value} role="tab" id={`hub-tab-${value}`} aria-selected={tab === value} aria-controls={`hub-panel-${value}`} onClick={() => value === "courts" ? openCourtSetup() : setTab(value)}>{label}</button>)}</nav>

      {tab === "board" ? <section id="hub-panel-board" role="tabpanel" aria-labelledby="hub-tab-board" className={styles.root}>
        <div className={styles.card}><div className={styles.heading}><div><h3>Ready for a court</h3><p className={styles.small}>Longest waiting players first, including time on bye.</p></div><button type="button" className={styles.primary} disabled={disabled || !workspace.eligible_queue.length || !workspace.courts.some(court => court.open && !court.current)} onClick={() => void command("auto_fill_courts")}>Fill open courts</button></div>{workspace.eligible_queue.length ? queueRows(workspace.eligible_queue.slice(0, 4), "ready") : <p className={styles.muted}>No matches are ready. Check paused players, group progress, and court availability.</p>}{workspace.eligible_queue.length > 4 ? <button type="button" className={styles.button} style={{ marginTop: ".7rem" }} onClick={() => setTab("queue")}>View all {workspace.eligible_queue.length} ready matches</button> : null}</div>
        <div className={styles.grid}>{workspace.courts.map(court => <article key={court.id} className={`${styles.court} ${court.current ? styles.occupied : ""} ${!court.open ? styles.closed : ""}`} aria-label={court.label}>
          <div className={styles.heading}><h3>{court.label}</h3><span className={`${styles.badge} ${court.current ? "" : court.open ? styles.good : ""}`}>{court.current ? "Playing" : court.open ? "Available" : "Closed"}</span></div>
          <p className={styles.small}>{court.dedicated_session_key ? `${groupMap.get(court.dedicated_session_key)?.title || "Dedicated group"} only` : "Shared court"}</p>
          {court.current ? <><p className={styles.eyebrow}>{groupMap.get(court.current.session_key)?.title} · Round {court.current.round_number}</p><strong>{hubMatchLabel(court.current)}</strong><div className={styles.actions}>{groupMap.get(court.current.session_key)?.scoring_mode === "unscored" ? <button type="button" className={styles.primary} disabled={disabled} onClick={() => void command("mark_played", { court_id: court.id, session_key: court.current!.session_key, match_id: court.current!.match_id })}>Match played</button> : <button type="button" className={styles.primary} disabled={disabled} onClick={() => { setScoreEditor({ courtId: court.id, match: court.current!, version: detail.hub.version, scoreA: "", scoreB: "" }); setError(""); }}>Enter score</button>}<button type="button" className={styles.button} disabled={disabled} onClick={() => openCourtPicker(court.current!, court.id)}>Move or remove</button></div></> : <p className={styles.muted}>{court.open ? "Assign a ready match from the queue." : "Open this court in Court setup when it is available."}</p>}
          {court.next ? <div className={styles.next}><span className={`${styles.badge} ${styles.waiting}`}>Next on this court</span><strong>{hubMatchLabel(court.next)}</strong><span className={styles.small}>{groupMap.get(court.next.session_key)?.title} · Round {court.next.round_number}</span><div><button type="button" className={styles.button} disabled={disabled} onClick={() => void command("cancel_reservation", { court_id: court.id, session_key: court.next!.session_key, match_id: court.next!.match_id })}>Cancel wait</button></div></div> : null}
        </article>)}</div>
      </section> : null}

      {tab === "queue" ? <section id="hub-panel-queue" role="tabpanel" aria-labelledby="hub-tab-queue" className={styles.root}><div className={styles.card}><h2>Ready matches · {workspace.eligible_queue.length}</h2><p className={styles.small}>Priority follows the longest waiting player in each eligible match. Scheduled byes count toward waiting time.</p>{workspace.eligible_queue.length ? queueRows(workspace.eligible_queue, "ready") : <p className={styles.empty}>No matches are ready.</p>}</div>{workspace.reserved_queue.length ? <div className={styles.card}><h2>Reserved next · {workspace.reserved_queue.length}</h2>{queueRows(workspace.reserved_queue, "waiting")}</div> : null}{workspace.blocked_queue.length ? <div className={styles.card}><h2>Waiting for players or courts · {workspace.blocked_queue.length}</h2>{queueRows(workspace.blocked_queue, "blocked")}</div> : null}</section> : null}

      {tab === "groups" ? <section id="hub-panel-groups" role="tabpanel" aria-labelledby="hub-tab-groups" className={styles.root}>
        <div className={styles.card}><div className={styles.heading}><div><h2>Add a round robin</h2><p className={styles.muted}>Each group keeps its roster, format, rating choice, and results.</p></div><Link className={styles.button} href={`/admin/round-robin-generator?hub=${encodeURIComponent(detail.hub.id)}`}>Create round robin</Link></div><div className={styles.inlineForm}><label className={styles.field}>Existing active round robin<select value={addSessionKey} onChange={event => setAddSessionKey(event.target.value)}><option value="">Choose a round robin</option>{availableSessions.map(session => <option key={session.session_key} value={session.session_key}>{session.title} · {formatName(session.event_format)} · {session.rating_mode || "unrated"}</option>)}</select></label><button type="button" className={styles.primary} disabled={disabled || !availableSessions.some(session => session.session_key === addSessionKey)} onClick={() => void command("add_group", { session_key: addSessionKey }).then(ok => { if (ok) setAddSessionKey(""); })}>Add to hub</button></div>{!availableSessions.length ? <p className={styles.small}>No other active round robins are available. Create a round robin to add it to this event and queue its matches.</p> : null}</div>
        <div className={styles.grid}>{workspace.groups.map(group => {
          const dedicated = workspace.courts.filter(court => court.dedicated_session_key === group.session_key);
          const groupBusy = workspace.courts.some(court => court.current?.session_key === group.session_key || court.next?.session_key === group.session_key);
          const meta = list.sessions.find(session => session.session_key === group.session_key);
          return <article className={styles.groupCard} key={group.session_key}><div className={styles.heading}><h3>{group.title}</h3><span className={styles.badge}>{formatName(group.event_format)}</span></div><p className={styles.small}>{group.scoring_mode === "unscored" ? "Unscored" : (group.rating_mode || meta?.rating_mode) === "rated" ? "Rated · approval required" : "Unrated"} · {dedicated.length ? dedicated.map(court => court.label).join(", ") : "Shared courts"}</p><p><strong>Round {group.current_round_number}</strong> · {group.completed_matches} of {group.total_matches} matches complete</p><div className={styles.progress} role="progressbar" aria-label={`${group.title} round completion`} aria-valuemin={0} aria-valuemax={group.total_matches || 1} aria-valuenow={group.completed_matches}><span style={{ width: `${group.total_matches ? Math.min(100, group.completed_matches / group.total_matches * 100) : 0}%` }} /></div><p className={styles.small}>{group.scoring_mode === "unscored" ? "Match completion is saved without scores." : ["saved", "played"].includes(group.round_status) ? "Round complete. These scores count toward standings." : "Scores are saved as games finish. Standings update when this round is complete."}</p><div className={styles.actions}><button type="button" className={styles.primary} disabled={disabled || !group.can_advance} onClick={() => void command("advance_round", { session_key: group.session_key })}>{group.advance_label || "Next round"}</button><button type="button" className={styles.button} onClick={() => showSession(group)}>Manage round robin</button>{group.scoring_mode !== "unscored" ? <button type="button" className={styles.button} onClick={() => showSession(group, "standings")}>Standings</button> : null}</div><div><button type="button" className={styles.button} disabled={disabled || groupBusy || dedicated.length > 0} title={dedicated.length ? "Return dedicated courts to shared use before removing this group." : undefined} onClick={() => void command("remove_group", { session_key: group.session_key })}>Remove from hub</button></div></article>;
        })}</div>
      </section> : null}

      {tab === "players" ? <section id="hub-panel-players" role="tabpanel" aria-labelledby="hub-tab-players" className={styles.card}><div className={styles.heading}><div><h2>Players</h2><p className={styles.muted}>Pause for a break and resume when ready. No reason needed. Waiting time is held while paused.</p></div><label className={styles.field}>Find player<input type="search" value={playerSearch} onChange={event => setPlayerSearch(event.target.value)} /></label></div><div className={styles.playerList}>{workspace.players.filter(player => player.name.toLowerCase().includes(playerSearch.toLowerCase())).map(player => {
        const playingCourt = workspace.courts.find(court => court.current && [...court.current.side_a, ...court.current.side_b].some(person => person.identity === player.identity));
        return <div key={player.identity} className={styles.player}><div><strong>{player.name}</strong><p className={styles.small}>{player.session_keys.map(key => groupMap.get(key)?.title).filter(Boolean).join(" · ")}</p><span className={styles.small}>{playingCourt ? `Playing on ${playingCourt.label}` : `Waiting ${hubWaitLabel(player.wait_seconds)}`}{player.paused ? " · Paused" : ""}</span></div><button type="button" className={player.paused ? styles.primary : styles.button} disabled={disabled} onClick={() => void command(player.paused ? "resume_player" : "pause_player", { identity: player.identity })}>{player.paused ? "Ready again" : "Pause player"}</button></div>;
      })}</div><p className={styles.small}>To leave a round robin early or change a roster, open that group’s Manage round robin controls.</p></section> : null}

      {tab === "courts" ? <section id="hub-panel-courts" role="tabpanel" aria-labelledby="hub-tab-courts" className={styles.card}><h2>Court setup</h2><p className={styles.muted}>A group with dedicated courts uses only those courts. Other groups use the shared courts.</p>{courtEditorRows(editCourts, setEditCourts, true)}{String(courtDraftVersion) !== String(detail.hub.version) ? <p className={styles.small}>The court board has changed since this editor opened. Review current courts before saving.</p> : null}<div className={styles.actions}><button type="button" className={styles.primary} disabled={disabled || !editCourts.length || editCourts.some(court => !court.label.trim())} onClick={() => void command("configure_courts", { courts: editCourts.map(courtConfig) }, courtDraftVersion).then(ok => { if (ok) setTab("board"); })}>Save court setup</button><button type="button" className={styles.button} disabled={busy} onClick={openCourtSetup}>Reload current courts</button><button type="button" className={styles.button} disabled={disabled} onClick={() => setFinishReview(true)}>Finish event</button></div></section> : null}

      {sessionView ? <section className={styles.embedded} aria-label="Manage selected round robin"><div className={styles.heading}><div><p className={styles.eyebrow}>Round robin controls</p><h2>{groupMap.get(sessionView.sessionKey)?.title || "Round robin"}</h2></div><button type="button" className={styles.button} onClick={() => setSessionView(null)}>Close round robin controls</button></div><div className={styles.embeddedBody} inert={busy || attaching || Boolean(pending)}>{sessionView.view === "standings" ? <GeneratorStandings key={`${sessionView.sessionKey}:standings`} apiBase={apiBase} clubId={clubId} sessionKey={sessionView.sessionKey} embedded refreshKey={detail.hub.version} onNavigate={target => setSessionView({ ...sessionView, view: target.view, roundNumber: target.roundNumber || sessionView.roundNumber })} onSessionChanged={() => { void loadHub(hubId, true); }} /> : <GeneratorRoundRunner key={`${sessionView.sessionKey}:round:${sessionView.roundNumber}`} apiBase={apiBase} clubId={clubId} generatorKind="round_robin" sessionKey={sessionView.sessionKey} roundNumber={sessionView.roundNumber} embedded refreshKey={detail.hub.version} onNavigate={target => setSessionView({ ...sessionView, view: target.view, roundNumber: target.roundNumber || sessionView.roundNumber })} onSessionChanged={() => { void loadHub(hubId, true); }} />}</div></section> : null}
    </> : null}

    {scoreEditor ? <InteractionDialog open phase={busy ? "working" : "ready"} title={`Enter score · ${selectedScoreCourt?.label || "Court"}`} description={<p>{groupMap.get(scoreEditor.match.session_key)?.title} · Round {scoreEditor.match.round_number}</p>} initialFocusRef={scoreInput} dirty={Boolean(scoreEditor.scoreA || scoreEditor.scoreB)} onRequestClose={() => { if (!busy) setScoreEditor(null); }} actions={<><button type="button" className={styles.button} disabled={busy} onClick={() => setScoreEditor(null)}>Cancel</button><button type="submit" form="hub-score-form" className={styles.primary} disabled={disabled || !scoreValid || scoreMatchChanged || scoreStale}>Save score &amp; release court</button></>}>
      <form id="hub-score-form" onSubmit={event => { event.preventDefault(); if (scoreValid && !scoreStale && !scoreMatchChanged) void command("score_and_release", { court_id: scoreEditor.courtId, session_key: scoreEditor.match.session_key, match_id: scoreEditor.match.match_id, score_a: Number(scoreA), score_b: Number(scoreB) }, scoreEditor.version); }}><div className={styles.scoreGrid}><label className={styles.field}>{scoreEditor.match.side_a.map(player => player.name).join(" / ")}<input ref={scoreInput} aria-label="Side A score" inputMode="numeric" type="number" min={0} max={99} value={scoreEditor.scoreA} onChange={event => setScoreEditor({ ...scoreEditor, scoreA: event.target.value })} /></label><label className={styles.field}>{scoreEditor.match.side_b.map(player => player.name).join(" / ")}<input aria-label="Side B score" inputMode="numeric" type="number" min={0} max={99} value={scoreEditor.scoreB} onChange={event => setScoreEditor({ ...scoreEditor, scoreB: event.target.value })} /></label></div></form><p className={styles.small}>This saves the match and frees its court. The score counts toward standings when the entire round is complete.</p>{scoreA && scoreB && !scoreValid ? <p role="alert">Enter two different whole-number scores between 0 and 99.</p> : null}{scoreMatchChanged ? <p role="alert">This court now has a different match. Close this score entry and review the court board.</p> : scoreStale ? <div className={styles.notice}><p>The court board changed while this score was open. The same match is still on court; review it before saving.</p><button type="button" className={styles.button} onClick={() => setScoreEditor({ ...scoreEditor, version: detail!.hub.version })}>Reviewed — keep these scores</button></div> : null}{error ? <p role="alert">{error}</p> : null}{pending ? <button type="button" className={styles.button} disabled={busy} onClick={() => void execute(pending)}>Retry saved action</button> : null}
    </InteractionDialog> : null}

    {courtEditor ? <InteractionDialog open phase={busy ? "working" : "ready"} title={courtEditor.sourceCourtId ? "Move or remove match" : "Choose court"} description={<p>{hubMatchLabel(courtEditor.match)}</p>} onRequestClose={() => { if (!busy) setCourtEditor(null); }} actions={<><button type="button" className={styles.button} disabled={busy} onClick={() => setCourtEditor(null)}>Cancel</button>{courtEditor.sourceCourtId ? <button type="button" className={styles.button} disabled={disabled} onClick={() => void command("release_game", { court_id: courtEditor.sourceCourtId, session_key: courtEditor.match.session_key, match_id: courtEditor.match.match_id }, courtEditor.version)}>Return to queue</button> : null}<button type="button" className={styles.primary} disabled={disabled || !targetCourt || !choiceCourts.some(court => court.id === targetCourt.id)} onClick={() => void command(courtEditor.sourceCourtId ? "move_game" : targetCourt?.current ? "reserve_game_for_court" : "assign_game", courtEditor.sourceCourtId ? { court_id: courtEditor.sourceCourtId, target_court_id: courtEditor.targetCourtId, session_key: courtEditor.match.session_key, match_id: courtEditor.match.match_id } : { session_key: courtEditor.match.session_key, match_id: courtEditor.match.match_id, court_id: courtEditor.targetCourtId }, courtEditor.version)}>{courtEditor.sourceCourtId ? "Move match" : targetCourt?.current ? "Reserve next" : "Assign to court"}</button></>}>
      <label className={styles.field}>Court<select value={courtEditor.targetCourtId} onChange={event => setCourtEditor({ ...courtEditor, targetCourtId: event.target.value })}><option value="">Choose a court</option>{choiceCourts.map(court => <option key={court.id} value={court.id}>{court.label}{court.current ? " · Reserve next" : " · Available"}</option>)}</select></label><p className={styles.small}>Only courts available to this group are shown. Dedicated courts are exclusive.</p>{!choiceCourts.length ? <p>No eligible court is currently available for this action.</p> : null}{error ? <p role="alert">{error}</p> : null}{pending ? <button type="button" className={styles.button} disabled={busy} onClick={() => void execute(pending)}>Retry saved action</button> : null}
    </InteractionDialog> : null}

    {finishReview ? <InteractionDialog open phase={busy ? "working" : "ready"} title="Finish this event?" description={<p>{detail?.hub.title}</p>} onRequestClose={() => { if (!busy) setFinishReview(false); }} actions={<><button type="button" className={styles.button} disabled={busy} onClick={() => setFinishReview(false)}>Keep event open</button><button type="button" className={styles.primary} disabled={disabled || unresolved > 0} onClick={() => void command("finish_hub")}>Finish event</button></>}><p>{unresolved ? `${unresolved} matches are playing, waiting, or reserved. Resolve these before finishing.` : "The court queue is clear. Individual round robin results remain available."}</p><p className={styles.small}>Finish each round robin through its own controls to review results and submission options.</p>{error ? <p role="alert">{error}</p> : null}{pending ? <button type="button" className={styles.button} disabled={busy} onClick={() => void execute(pending)}>Retry saved action</button> : null}</InteractionDialog> : null}
  </div>;
}
