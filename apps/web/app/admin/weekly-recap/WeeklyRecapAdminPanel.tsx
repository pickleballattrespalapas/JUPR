"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  useAuthenticatedAutoLoad,
  useLatestRequestGuard,
} from "@/lib/useAuthenticatedAutoLoad";
import SearchablePlayerSelect from "@/components/SearchablePlayerSelect";
import { ConfirmAction } from "@/components/ConfirmAction";
import { actionSuccess } from "@/components/interaction";
import { useAdminSession } from "@/lib/useAdminSession";
import { weeklyReportDatePreset } from "@/lib/clubDate";
import type {
  AdminWeeklyRecapCandidate,
  AdminWeeklyRecapDetailResponse,
  AdminWeeklyRecapListResponse,
  AdminWeeklyRecapRow,
  AdminWeeklyRecapStatusResponse,
} from "@/lib/adminWeeklyRecapApi";
import type { WeeklyRecapJson } from "@/lib/weeklyRecapApi";
import {
  layoutTemplate,
  normalizeBuilder,
  previewRecap,
  type BuilderConfig,
  type Editorial,
  type RecapCatalog,
  type RecapEdits,
  type RecapPreferences,
} from "@/lib/recapBuilder";
import { buildWeeklyReportEmail, emailUrl } from "@/lib/weeklyReportEmail";
import { RecapContentEditor, RecapStoryEditor } from "./RecapContentEditor";
import AdminWeeklyRecapPreview from "./AdminWeeklyRecapPreview";
import WeeklyReportExport from "./WeeklyReportExport";
import RecapLayoutManager from "./RecapLayoutManager";
import styles from "./recap-builder.module.css";

type Props = {
  apiBase: string | null;
  clubId: string;
  clubSlug?: string;
  status: AdminWeeklyRecapStatusResponse;
  initialWeekStart?: string;
  printMode?: boolean;
};
const steps = ["Setup", "Content", "Personalize", "Preview", "Publish & share"];
const emptyEditorial: Editorial = {
  headline: "This week on the courts",
  opening: "",
  event_notes: "",
  upcoming: [],
};
const emptyPreferences: RecapPreferences = {
  revision: 0,
  settings: { templates: [], default_template_id: "" },
  leagues: [],
};
const blankEdits = (): RecapEdits => ({
  editorial: emptyEditorial,
  spotlight_overrides: {},
  looking_ahead: [],
  builder: normalizeBuilder(),
  scope: { league_names: [] },
});
function rowEdits(
  row: AdminWeeklyRecapRow,
  candidates: Record<string, AdminWeeklyRecapCandidate[]>,
): RecapEdits {
  const raw = row.edits_json || {};
  const report = (raw.editorial ||
    row.generated_json?.report ||
    {}) as Partial<Editorial>;
  const generated = (row.generated_json?.spotlight || []) as Array<{
    key: string;
    candidate_ids?: string[];
    description?: string;
    order?: number;
  }>;
  const overrides = (raw.spotlight_overrides || {}) as Partial<
    RecapEdits["spotlight_overrides"]
  >;
  const spotlight = Object.fromEntries(
    Object.entries(candidates).map(([key, items], index) => {
      const original = generated.find((item) => item.key === key);
      return [
        key,
        {
          include: true,
          order: original?.order || index + 1,
          description: original?.description || "",
          players:
            original?.candidate_ids ||
            items.slice(0, 1).map((i) => i.candidate_id),
          ...overrides[key],
        },
      ];
    }),
  );
  return {
    editorial: {
      headline: report.headline || emptyEditorial.headline,
      opening: report.opening || "",
      event_notes: report.event_notes || "",
      upcoming: report.upcoming || [],
    },
    spotlight_overrides: spotlight,
    looking_ahead: Array.isArray(raw.looking_ahead)
      ? (raw.looking_ahead as string[])
      : ((row.generated_json?.looking_ahead || []) as string[]),
    builder: normalizeBuilder(raw.builder as BuilderConfig),
    scope: (raw.scope || { league_names: [] }) as RecapEdits["scope"],
  };
}
export default function WeeklyRecapAdminPanel({
  apiBase,
  clubId,
  clubSlug = "",
  status,
  initialWeekStart = "",
  printMode = false,
}: Props) {
  const { accessToken, loading: sessionLoading } = useAdminSession();
  const [range, setRange] = useState(() => weeklyReportDatePreset("last_week"));
  const [rows, setRows] = useState<AdminWeeklyRecapRow[]>([]),
    [row, setRow] = useState<AdminWeeklyRecapRow | null>(null);
  const [candidates, setCandidates] = useState<
    Record<string, AdminWeeklyRecapCandidate[]>
  >({});
  const [edits, setEdits] = useState<RecapEdits>(blankEdits),
    [lastSaved, setLastSaved] = useState("");
  const [preferences, setPreferences] =
      useState<RecapPreferences>(emptyPreferences),
    [preferencesReady, setPreferencesReady] = useState(false);
  const [templateName, setTemplateName] = useState(""),
    [step, setStep] = useState(0),
    [previewMode, setPreviewMode] = useState("website");
  const [revisionOpen, setRevisionOpen] = useState(true),
    [busy, setBusy] = useState(false),
    [saving, setSaving] = useState(false);
  const [saveFailed, setSaveFailed] = useState(false),
    [canDelete, setCanDelete] = useState(false);
  const [message, setMessage] = useState(""),
    [error, setError] = useState(""),
    [origin, setOrigin] = useState("");
  const requestScope = `${clubId}\u0000${accessToken}`;
  const scope = requestScope;
  const workspaceRequest = useLatestRequestGuard(
    requestScope,
    clearProtectedRecapState,
  );
  const detailRequest = useLatestRequestGuard(requestScope);
  const writeRequest = useLatestRequestGuard(requestScope);
  const saveLock = useRef(false);
  const latest = useRef({ row, edits, scope });
  latest.current = { row, edits, scope };
  const serialized = JSON.stringify(edits),
    dirty = Boolean(row && serialized !== lastSaved);
  const disabled =
    !status.mutations_enabled || busy || !accessToken || !revisionOpen;
  const preview = useMemo(
    () =>
      previewRecap(
        (row?.generated_json || {}) as Partial<WeeklyRecapJson>,
        edits,
        candidates,
      ),
    [row?.generated_json, edits, candidates],
  );
  const generatedCatalog = (preview.builder_catalog || {}) as RecapCatalog;
  const coverage = (row?.generated_json as Partial<WeeklyRecapJson>)?.report
    ?.coverage;
  const isLive = row?.status === "published",
    hasChanges = dirty || Boolean(row?.has_unpublished_changes);
  const publicReady = Boolean(
    isLive && !hasChanges && row?.edits_json?.builder,
  );
  const emailHtml = buildWeeklyReportEmail(preview, {
    origin,
    clubSlug,
    published: publicReady,
    compact: true,
    document: true,
  });
  const invalidLinks = [
    ...edits.editorial.upcoming.map((v) => v.url),
    ...edits.builder.custom_blocks.flatMap((v) => [v.url || ""]),
  ].filter((url) => url && !emailUrl(url));
  invalidLinks.push(
    ...edits.builder.custom_blocks
      .map((v) => v.image_url || "")
      .filter((url) => url && !/^https:\/\//i.test(url)),
  );
  async function request<T>(
    path: string,
    method = "GET",
    body?: unknown,
  ): Promise<T> {
    if (scope !== latest.current.scope)
      throw new Error(
        "Your sign-in or club changed. Reload before continuing.",
      );
    if (!apiBase || !accessToken)
      throw new Error("Sign in before editing recaps.");
    const headers = new Headers({ Authorization: `Bearer ${accessToken}` });
    if (body) headers.set("Content-Type", "application/json");
    const response = await fetch(
      `${apiBase.replace(/\/$/, "")}/admin/clubs/${encodeURIComponent(clubId)}/weekly-recap${path}`,
      { method, headers, ...(body ? { body: JSON.stringify(body) } : {}) },
    );
    const payload = await response.json();
    if (!response.ok)
      throw new Error(
        typeof payload.detail === "string"
          ? payload.detail
          : "Could not complete this action. Reload and try again.",
      );
    return payload;
  }
  function apply(payload: AdminWeeklyRecapDetailResponse) {
    const nextCandidates = payload.candidates || {},
      next = rowEdits(payload.recap, nextCandidates);
    setRow(payload.recap);
    setCandidates(nextCandidates);
    setEdits(next);
    setLastSaved(JSON.stringify(next));
    setRange({ start: payload.recap.week_start, end: payload.recap.week_end });
    setRevisionOpen(payload.recap.status !== "published");
    setStep(payload.recap.status === "published" ? 3 : next.builder.step);
    setSaveFailed(false);
  }
  async function load(target: string) {
    const generation = detailRequest.begin();
    setRow(null);
    setLastSaved("");
    setBusy(true);
    setError("");
    try {
      const payload = await request<AdminWeeklyRecapDetailResponse>(
        `/recaps/${encodeURIComponent(target)}?include_candidates=true`,
      );
      if (!detailRequest.isCurrent(generation)) return;
      apply(payload);
    } catch (e) {
      if (detailRequest.isCurrent(generation))
        setError(e instanceof Error ? e.message : "Unable to load recap.");
    } finally {
      if (detailRequest.isCurrent(generation)) setBusy(false);
    }
  }
  function clearProtectedRecapState() {
    detailRequest.invalidate();
    setRow(null);
    setRows([]);
    setCandidates({});
    setEdits(blankEdits());
    setLastSaved("");
    setPreferences(emptyPreferences);
    setPreferencesReady(false);
    setStep(0);
    setError("");
    setMessage("");
    setSaving(false);
    saveLock.current = false;
    setCanDelete(false);
    setBusy(false);
  }
  async function loadWorkspace() {
    const generation = workspaceRequest.begin();
    clearProtectedRecapState();
    if (!accessToken || !status.enabled) return;
    setBusy(true);
    const [list, prefs] = await Promise.allSettled([
      request<AdminWeeklyRecapListResponse>("/recaps?limit=100"),
      request<RecapPreferences>("/preferences"),
    ]);
    if (!workspaceRequest.isCurrent(generation)) return;
    if (list.status === "fulfilled") {
      setRows(list.value.recaps);
      setCanDelete(Boolean(list.value.can_delete_drafts));
    } else setError(String(list.reason?.message || "Could not load recaps."));
    if (prefs.status === "fulfilled") {
      setPreferences(prefs.value);
      setPreferencesReady(true);
      const saved = prefs.value.settings.templates.find(
        (t) => t.id === prefs.value.settings.default_template_id,
      );
      setEdits({ ...blankEdits(), builder: normalizeBuilder(saved?.builder) });
    } else
      setError(
        "Club templates could not be loaded. Reload the workspace before saving defaults.",
      );
    setBusy(false);
    if (initialWeekStart) await load(initialWeekStart);
  }
  useAuthenticatedAutoLoad(
    status.enabled ? accessToken : "",
    loadWorkspace,
    `${clubId}:${initialWeekStart}`,
  );
  useEffect(() => {
    if (typeof window !== "undefined") setOrigin(window.location.origin);
  }, []);
  useEffect(() => {
    if (typeof window === "undefined") return;
    const warn = (event: BeforeUnloadEvent) => {
      if (dirty) {
        event.preventDefault();
        event.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);
  async function saveDraft() {
    if (saveLock.current) throw new Error("A save is still in progress.");
    const target = latest.current.row,
      snapshot = latest.current.edits;
    if (!target) throw new Error("Generate or load a recap first.");
    const generation = writeRequest.begin();
    saveLock.current = true;
    setSaving(true);
    setError("");
    try {
      const payload = await request<AdminWeeklyRecapDetailResponse>(
        `/recaps/${target.week_start}`,
        "PATCH",
        {
          edits_json: snapshot,
          confirmation_text: "SAVE RECAP",
          expected_recap_id: target.id,
          expected_row_version: target.row_version,
        },
      );
      if (!writeRequest.isCurrent(generation)) return;
      setRow(payload.recap);
      setLastSaved(JSON.stringify(snapshot));
      setSaveFailed(false);
      setMessage("Draft saved.");
      setRows((current) =>
        current.map((r) => (r.id === payload.recap.id ? payload.recap : r)),
      );
    } catch (e) {
      if (writeRequest.isCurrent(generation)) {
        setSaveFailed(true);
        setError(e instanceof Error ? e.message : "Draft could not be saved.");
      }
      throw e;
    } finally {
      if (writeRequest.isCurrent(generation)) {
        saveLock.current = false;
        setSaving(false);
      }
    }
  }
  useEffect(() => {
    if (
      !dirty ||
      !revisionOpen ||
      !status.mutations_enabled ||
      busy ||
      saving ||
      saveFailed ||
      invalidLinks.length
    )
      return;
    const timer = setTimeout(() => {
      void saveDraft().catch(() => undefined);
    }, 1400);
    return () => clearTimeout(timer);
    // Serialize each edit snapshot; a failed save requires an explicit retry.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    serialized,
    dirty,
    busy,
    saving,
    saveFailed,
    revisionOpen,
    status.mutations_enabled,
    scope,
  ]);
  async function operation(
    action: "generate" | "publish" | "unpublish" | "delete",
    confirmationText: string,
  ) {
    if (saveLock.current)
      throw new Error("The draft is still saving. Try again in a moment.");
    const generation = writeRequest.begin();
    setBusy(true);
    setError("");
    try {
      const current = latest.current;
      if (row?.id !== current.row?.id)
        throw new Error(
          "The selected recap changed. Close this dialog and review it again.",
        );
      const target =
        action === "generate"
          ? current.row?.week_start === range.start
            ? current.row
            : rows.find((r) => r.week_start === range.start)
          : current.row;
      if (action === "generate" && target && target.id !== row?.id) {
        await load(target.week_start);
        return actionSuccess(
          "Recap loaded",
          "Review the existing edition before refreshing it.",
        );
      }
      if (action !== "generate" && !target)
        throw new Error("Load a recap first.");
      const identity = {
        expected_recap_id: target?.id || null,
        expected_row_version: target?.row_version || null,
        confirmation_text: confirmationText,
      };
      const path =
        action === "generate"
          ? "/generate"
          : `/recaps/${target!.week_start}${action === "publish" || action === "unpublish" ? "/publish" : ""}`;
      const body =
        action === "generate"
          ? {
              ...identity,
              week_start: range.start,
              week_end: range.end,
              preserve_edits: true,
              edits_json: current.edits,
            }
          : action === "delete"
            ? identity
            : { ...identity, action, edits_json: current.edits };
      const payload = await request<AdminWeeklyRecapDetailResponse>(
        path,
        action === "delete" ? "DELETE" : "POST",
        body,
      );
      if (!writeRequest.isCurrent(generation))
        return actionSuccess("Action complete", "");
      if (action === "delete") {
        setRow(null);
        setLastSaved("");
        setStep(0);
      } else {
        apply(payload);
        if (action === "generate") {
          setStep(1);
          setRevisionOpen(true);
        }
        if (action === "publish") setStep(4);
      }
      const list =
        await request<AdminWeeklyRecapListResponse>("/recaps?limit=100");
      if (writeRequest.isCurrent(generation)) {
        setRows(list.recaps);
        setMessage(
          action === "publish"
            ? "Recap published."
            : action === "unpublish"
              ? "Recap returned to draft."
              : action === "delete"
                ? "Draft deleted."
                : "Results refreshed; your selections and stories were preserved.",
        );
      }
      return actionSuccess(
        "Recap updated",
        "The requested action is complete.",
      );
    } catch (e) {
      if (writeRequest.isCurrent(generation))
        setError(e instanceof Error ? e.message : "Unable to update recap.");
      throw e;
    } finally {
      if (writeRequest.isCurrent(generation)) setBusy(false);
    }
  }
  async function chooseRecap(value: string) {
    try {
      if (dirty && revisionOpen) await saveDraft();
      if (value) await load(value);
      else {
        setRow(null);
        setStep(0);
        setLastSaved("");
        setRevisionOpen(true);
        setEdits({
          ...blankEdits(),
          builder: normalizeBuilder(
            preferences.settings.templates.find(
              (t) => t.id === preferences.settings.default_template_id,
            )?.builder,
          ),
        });
      }
    } catch {
      /* Keep the current draft and its error. */
    }
  }
  function changeBuilder(builder: BuilderConfig) {
    setEdits((current) => ({ ...current, builder }));
  }
  function go(next: number) {
    setStep(next);
    if (revisionOpen && row)
      setEdits((current) => ({
        ...current,
        builder: { ...current.builder, step: next },
      }));
  }
  function editorial(patch: Partial<Editorial>) {
    setEdits((current) => ({
      ...current,
      editorial: { ...current.editorial, ...patch },
    }));
  }
  function spotlight(
    key: string,
    patch: Partial<RecapEdits["spotlight_overrides"][string]>,
  ) {
    setEdits((current) => ({
      ...current,
      spotlight_overrides: {
        ...current.spotlight_overrides,
        [key]: { ...current.spotlight_overrides[key], ...patch },
      },
    }));
  }
  async function saveTemplate(asDefault: boolean) {
    if (saveLock.current) return;
    const name = templateName.trim();
    if (!name) {
      setError("Give the layout a name.");
      return;
    }
    try {
      const prior = preferences.settings.templates.find(
        (t) => t.name.toLowerCase() === name.toLowerCase(),
      );
      const template = {
        id: prior?.id || crypto.randomUUID(),
        name,
        builder: layoutTemplate(edits.builder),
      };
      const settings = {
        templates: [
          ...preferences.settings.templates.filter((t) => t.id !== template.id),
          template,
        ],
        default_template_id: asDefault
          ? template.id
          : preferences.settings.default_template_id,
      };
      await updateLayouts(
        settings,
        preferences.revision,
        asDefault ? "Club default layout saved." : "Layout template saved.",
      );
      setTemplateName("");
    } catch {
      /* updateLayouts keeps the error visible. */
    }
  }
  async function updateLayouts(
    settings?: RecapPreferences["settings"],
    revision?: number,
    success = "Saved layouts reloaded.",
  ) {
    if (scope !== latest.current.scope)
      throw new Error(
        "Your sign-in or club changed. Reload before continuing.",
      );
    if (saveLock.current)
      throw new Error(
        "A save is still in progress. Try again when it finishes.",
      );
    const generation = writeRequest.begin();
    saveLock.current = true;
    setBusy(true);
    setError("");
    try {
      const result = await request<RecapPreferences>(
        "/preferences",
        settings ? "PUT" : "GET",
        settings ? { revision, settings } : undefined,
      );
      if (!writeRequest.isCurrent(generation))
        throw new Error(
          "Your sign-in or club changed. Reload before continuing.",
        );
      setPreferences(result);
      setPreferencesReady(true);
      setMessage(success);
    } catch (e) {
      if (writeRequest.isCurrent(generation))
        setError(
          e instanceof Error ? e.message : "Could not update saved layouts.",
        );
      throw e;
    } finally {
      if (writeRequest.isCurrent(generation)) {
        saveLock.current = false;
        setBusy(false);
      }
    }
  }
  if (!status.enabled) return <p>Weekly recaps are unavailable.</p>;
  return (
    <div className={styles.builder} data-testid="recap-builder">
      <div className={styles.toolbar}>
        <div>
          <strong>Weekly Recap Builder</strong>
          <div role="status">
            {saving
              ? "Saving draft…"
              : dirty
                ? "Unsaved changes"
                : row
                  ? "Draft saved"
                  : "Start a new edition"}
            {isLive
              ? hasChanges
                ? " · Published version remains live"
                : " · Published"
              : ""}
          </div>
        </div>
        <div className={styles.actions}>
          {row && !revisionOpen ? (
            <button
              disabled={!status.mutations_enabled || busy}
              onClick={() => {
                setRevisionOpen(true);
                setMessage(
                  "Editing a draft revision. The published edition stays live.",
                );
              }}
            >
              Edit draft revision
            </button>
          ) : null}
          {row && revisionOpen ? (
            <button
              disabled={disabled || saving || !dirty || !!invalidLinks.length}
              onClick={() => void saveDraft().catch(() => undefined)}
            >
              {saveFailed ? "Retry save" : "Save draft"}
            </button>
          ) : null}
          {!row ? (
            <button disabled={busy} onClick={() => void loadWorkspace()}>
              Refresh workspace
            </button>
          ) : null}
          {row ? (
            <button
              disabled={busy || saving || (dirty && !saveFailed)}
              onClick={() => void load(row.week_start)}
            >
              {saveFailed
                ? "Reload latest (discard unsaved edits)"
                : "Reload saved recap"}
            </button>
          ) : null}
        </div>
      </div>
      {!status.mutations_enabled ? (
        <p>Read-only: you can review saved recaps.</p>
      ) : null}
      {sessionLoading ? <p>Checking sign-in…</p> : null}
      {error ? (
        <p role="alert">{error}</p>
      ) : message ? (
        <p role="status">{message}</p>
      ) : null}
      <nav className={styles.nav} aria-label="Recap steps">
        {steps.map((label, index) => (
          <button
            key={label}
            aria-current={step === index ? "step" : undefined}
            disabled={busy || (index > 0 && !row)}
            onClick={() => go(index)}
          >
            {index + 1}. {label}
          </button>
        ))}
      </nav>
      <section hidden={step !== 0} className={styles.section}>
        <h2>Set up your edition</h2>
        {!busy && !rows.length ? (
          <p>No saved recaps yet. Generate your first edition below.</p>
        ) : null}
        <label>
          Existing recaps
          <select
            disabled={busy || saving}
            value={row?.week_start || ""}
            onChange={(e) => void chooseRecap(e.target.value)}
          >
            <option value="">New edition</option>
            {rows.map((r) => (
              <option key={r.id || r.week_start} value={r.week_start}>
                {r.week_start} → {r.week_end} · {r.status}
              </option>
            ))}
          </select>
        </label>
        <div className={styles.actions}>
          {(["last_week", "this_week", "last_seven"] as const).map((preset) => (
            <button
              key={preset}
              disabled={disabled || Boolean(row)}
              onClick={() => setRange(weeklyReportDatePreset(preset))}
            >
              {
                {
                  last_week: "Last completed week",
                  this_week: "This week so far",
                  last_seven: "Last 7 days",
                }[preset]
              }
            </button>
          ))}
        </div>
        <div className={styles.fields}>
          <label>
            Start date
            <input
              type="date"
              value={range.start}
              disabled={disabled || Boolean(row)}
              onChange={(e) =>
                setRange((current) => ({ ...current, start: e.target.value }))
              }
            />
          </label>
          <label>
            End date
            <input
              type="date"
              value={range.end}
              disabled={disabled}
              onChange={(e) =>
                setRange((current) => ({ ...current, end: e.target.value }))
              }
            />
          </label>
        </div>
        <label>
          Report scope
          <select
            disabled={disabled || Boolean(row)}
            value={edits.scope.league_names.length ? "league" : "club"}
            onChange={(e) =>
              setEdits((current) => ({
                ...current,
                scope: {
                  league_names:
                    e.target.value === "league"
                      ? [preferences.leagues[0] || ""]
                      : [],
                },
              }))
            }
          >
            <option value="club">Whole club</option>
            <option value="league" disabled={!preferences.leagues.length}>
              One league
            </option>
          </select>
        </label>
        {edits.scope.league_names.length ? (
          <label>
            League
            <select
              disabled={disabled || Boolean(row)}
              value={edits.scope.league_names[0]}
              onChange={(e) =>
                setEdits((current) => ({
                  ...current,
                  scope: { league_names: [e.target.value] },
                }))
              }
            >
              {preferences.leagues.map((name) => (
                <option key={name}>{name}</option>
              ))}
            </select>
          </label>
        ) : null}
        <p className={styles.muted}>
          Scope determines games, totals, and eligible highlights. Content
          controls choose which stories appear. Choose New edition to change the
          start date or scope.
        </p>
        <label>
          Layout template
          <select
            defaultValue=""
            disabled={disabled}
            onChange={(e) =>
              changeBuilder({
                ...normalizeBuilder(
                  preferences.settings.templates.find(
                    (t) => t.id === e.target.value,
                  )?.builder,
                ),
                custom_blocks: edits.builder.custom_blocks,
              })
            }
          >
            <option value="">Standard recap</option>
            {preferences.settings.templates.map((t) => (
              <option key={t.id} value={t.id}>
                {t.name}
                {t.id === preferences.settings.default_template_id
                  ? " · Club default"
                  : ""}
              </option>
            ))}
          </select>
        </label>
        <RecapLayoutManager
          key={`${scope}:${preferences.revision}`}
          preferences={preferences}
          disabled={
            !status.mutations_enabled ||
            busy ||
            saving ||
            !accessToken ||
            !preferencesReady
          }
          onSave={updateLayouts}
          onReload={() => updateLayouts()}
        />
        <ConfirmAction
          triggerLabel={
            rows.some((r) => r.week_start === range.start)
              ? "Refresh results"
              : "Generate draft"
          }
          title="Generate this recap?"
          description="Load recorded results for this period while preserving your selections and editorial work."
          confirmationText="GENERATE RECAP"
          confirmLabel="Generate draft"
          disabled={
            disabled || saving || !range.start || range.end < range.start
          }
          busy={busy}
          onConfirm={(confirmation) => operation("generate", confirmation)}
        />
        {canDelete && row?.status === "draft" ? (
          <ConfirmAction
            triggerLabel="Delete draft"
            title="Delete this recap draft?"
            description={`Permanently delete the draft for ${row.week_start} through ${row.week_end}? Recorded games and player results stay as they are.`}
            confirmationText="DELETE RECAP"
            confirmLabel="Delete draft"
            tone="danger"
            disabled={disabled || saving}
            busy={busy}
            onConfirm={(confirmation) => operation("delete", confirmation)}
          />
        ) : null}
      </section>
      {row ? (
        <>
          <section hidden={step !== 1}>
            <h2>Choose and arrange content</h2>
            <RecapContentEditor
              builder={edits.builder}
              catalog={generatedCatalog}
              disabled={disabled}
              onChange={changeBuilder}
            />
          </section>
          <section hidden={step !== 2} className={styles.section}>
            <h2>Make it your club’s recap</h2>
            <label>
              Report headline
              <input
                maxLength={160}
                value={edits.editorial.headline}
                disabled={disabled}
                onChange={(e) => editorial({ headline: e.target.value })}
              />
            </label>
            <label>
              Opening message
              <textarea
                maxLength={3000}
                value={edits.editorial.opening}
                disabled={disabled}
                onChange={(e) => editorial({ opening: e.target.value })}
              />
            </label>
            <label>
              Club news
              <textarea
                maxLength={3000}
                value={edits.editorial.event_notes}
                disabled={disabled}
                onChange={(e) => editorial({ event_notes: e.target.value })}
              />
            </label>
            <details>
              <summary>Player spotlights</summary>
              {Object.entries(edits.spotlight_overrides).map(
                ([key, config]) => (
                  <fieldset
                    key={key}
                    className={styles.section}
                    disabled={disabled}
                  >
                    <legend>{candidates[key]?.[0]?.label || key}</legend>
                    <label className={styles.check}>
                      <input
                        type="checkbox"
                        checked={config.include}
                        onChange={(e) =>
                          spotlight(key, { include: e.target.checked })
                        }
                      />
                      Include this spotlight
                    </label>
                    <div className={styles.fields}>
                      <label>
                        Order
                        <input
                          type="number"
                          min={1}
                          max={99}
                          value={config.order}
                          onChange={(e) =>
                            spotlight(key, {
                              order: Number(e.target.value) || 1,
                            })
                          }
                        />
                      </label>
                      <label>
                        Description
                        <input
                          value={config.description}
                          maxLength={1000}
                          onChange={(e) =>
                            spotlight(key, { description: e.target.value })
                          }
                        />
                      </label>
                    </div>
                    {[0, 1, 2].map((index) => (
                      <label key={index}>
                        Candidate {index + 1}
                        <SearchablePlayerSelect
                          aria-label={`Candidate ${index + 1} for ${key}`}
                          value={config.players[index] || ""}
                          onValueChange={(value) => {
                            const players = [...config.players];
                            players[index] = value;
                            spotlight(key, { players });
                          }}
                        >
                          <option value="">None</option>
                          {(candidates[key] || []).map((c) => (
                            <option key={c.candidate_id} value={c.candidate_id}>
                              {c.display}
                            </option>
                          ))}
                        </SearchablePlayerSelect>
                      </label>
                    ))}
                    <label className={styles.check}>
                      <input
                        type="checkbox"
                        checked={(
                          edits.builder.sections.spotlight.email_item_ids ||
                          (
                            preview.report?.edition?.find(
                              (b) => b.key === "spotlight",
                            )?.items || []
                          )
                            .slice(0, 2)
                            .map((i) => i.id)
                        ).includes(
                          generatedCatalog.spotlight?.find(
                            (i) => i.kind === key,
                          )?.id || `spotlight-${key}`,
                        )}
                        onChange={(e) => {
                          const id =
                            generatedCatalog.spotlight?.find(
                              (i) => i.kind === key,
                            )?.id || `spotlight-${key}`;
                          const current =
                            edits.builder.sections.spotlight.email_item_ids ||
                            (
                              preview.report?.edition?.find(
                                (b) => b.key === "spotlight",
                              )?.items || []
                            )
                              .slice(0, 2)
                              .map((i) => i.id);
                          changeBuilder({
                            ...edits.builder,
                            sections: {
                              ...edits.builder.sections,
                              spotlight: {
                                ...edits.builder.sections.spotlight,
                                email_item_ids: e.target.checked
                                  ? [...current.filter((v) => v !== id), id]
                                  : current.filter((v) => v !== id),
                              },
                            },
                          });
                        }}
                      />
                      Feature this spotlight in email
                    </label>
                  </fieldset>
                ),
              )}
            </details>
            <RecapStoryEditor
              builder={edits.builder}
              disabled={disabled}
              onChange={changeBuilder}
            />
            <h3>Coming up</h3>
            {edits.editorial.upcoming.map((item, index) => (
              <div key={index} className={styles.fields}>
                <label>
                  Event or announcement
                  <input
                    value={item.text}
                    maxLength={240}
                    disabled={disabled}
                    onChange={(e) =>
                      editorial({
                        upcoming: edits.editorial.upcoming.map((v, i) =>
                          i === index ? { ...v, text: e.target.value } : v,
                        ),
                      })
                    }
                  />
                </label>
                <label>
                  Registration URL
                  <input
                    type="url"
                    value={item.url}
                    disabled={disabled}
                    onChange={(e) =>
                      editorial({
                        upcoming: edits.editorial.upcoming.map((v, i) =>
                          i === index ? { ...v, url: e.target.value } : v,
                        ),
                      })
                    }
                  />
                </label>
                <button
                  disabled={disabled}
                  onClick={() =>
                    editorial({
                      upcoming: edits.editorial.upcoming.filter(
                        (_, i) => i !== index,
                      ),
                    })
                  }
                >
                  Remove event {index + 1}
                </button>
              </div>
            ))}
            <button
              disabled={disabled || edits.editorial.upcoming.length >= 10}
              onClick={() =>
                editorial({
                  upcoming: [
                    ...edits.editorial.upcoming,
                    { text: "", url: "" },
                  ],
                })
              }
            >
              Add event link
            </button>
            {edits.looking_ahead.map((line, index) => (
              <label key={index}>
                Looking ahead {index + 1}
                <input
                  value={line}
                  disabled={disabled}
                  onChange={(e) =>
                    setEdits((current) => ({
                      ...current,
                      looking_ahead: current.looking_ahead.map((v, i) =>
                        i === index ? e.target.value : v,
                      ),
                    }))
                  }
                />
              </label>
            ))}
          </section>
          <section hidden={step !== 3}>
            <h2>Preview your edition</h2>
            {invalidLinks.length ? (
              <p role="alert">
                Fix {invalidLinks.length} invalid link(s) in Personalize before
                saving or publishing. Photo URLs must use HTTPS.
              </p>
            ) : null}
            {coverage ? (
              <details>
                <summary>
                  Results coverage · {coverage.games || 0} games ·{" "}
                  {coverage.players || 0} players
                </summary>
                <ul>
                  {coverage.included?.map((v, i) => (
                    <li key={i}>
                      {v.title} · {v.date} · {v.status}
                    </li>
                  ))}
                </ul>
                {coverage.pending?.map((v, i) => (
                  <p key={i}>
                    Awaiting approval: {v.title}. These games are excluded until
                    approved.
                  </p>
                ))}
                {coverage.warnings?.map((v, i) => (
                  <p key={i}>{v}</p>
                ))}
              </details>
            ) : null}
            <div className={styles.previewTabs}>
              {["website", "email", "print"].map((view) => (
                <button
                  key={view}
                  aria-pressed={previewMode === view}
                  onClick={() => setPreviewMode(view)}
                >
                  {view === "website"
                    ? "Website"
                    : view === "email"
                      ? "Member email"
                      : "Print / PDF"}
                </button>
              ))}
            </div>
            {previewMode === "email" ? (
              <iframe
                title="Live member email preview"
                sandbox="allow-same-origin"
                srcDoc={emailHtml}
                style={{ width: "100%", height: 650, border: 0 }}
              />
            ) : (
              <AdminWeeklyRecapPreview
                recap={{
                  ...row,
                  status: publicReady ? "published" : "draft",
                  final_json: preview as Record<string, unknown>,
                }}
                printMode={printMode}
              />
            )}
            {previewMode === "print" ? (
              <button onClick={() => window.print()}>Print / save PDF</button>
            ) : null}
          </section>
          <section hidden={step !== 4} className={styles.section}>
            <h2>Publish and share</h2>
            <p>
              {isLive && hasChanges
                ? "Your published edition stays live until you publish this revision."
                : "Publish the reviewed edition to your club website."}
            </p>
            <ConfirmAction
              triggerLabel="Publish recap"
              title={isLive ? "Publish this revision?" : "Publish this recap?"}
              description="Publish the content selected in Preview, including your email selections."
              confirmationText="PUBLISH RECAP"
              confirmLabel="Publish recap"
              disabled={
                disabled ||
                saving ||
                !!invalidLinks.length ||
                (isLive && !hasChanges)
              }
              busy={busy}
              onConfirm={(confirmation) => operation("publish", confirmation)}
            />
            {isLive ? (
              <ConfirmAction
                triggerLabel="Unpublish recap"
                title="Unpublish this recap?"
                description="Remove this edition from the public page and keep its draft."
                confirmationText="UNPUBLISH RECAP"
                confirmLabel="Unpublish recap"
                tone="danger"
                disabled={busy || saving || !status.mutations_enabled}
                busy={busy}
                onConfirm={(confirmation) =>
                  operation("unpublish", confirmation)
                }
              />
            ) : null}
            <WeeklyReportExport
              recap={preview}
              clubSlug={clubSlug}
              published={publicReady}
            />
            <h3>Save this layout for future weeks</h3>
            <p className={styles.muted}>
              Saves section headings, order, detail, and email settings. New
              weeks get fresh results and stories.
            </p>
            <label>
              Layout name
              <input
                maxLength={80}
                value={templateName}
                disabled={busy}
                onChange={(e) => setTemplateName(e.target.value)}
              />
            </label>
            <div className={styles.actions}>
              <button
                disabled={
                  busy ||
                  saving ||
                  !status.mutations_enabled ||
                  !preferencesReady
                }
                onClick={() => void saveTemplate(false)}
              >
                Save template
              </button>
              <button
                disabled={
                  busy ||
                  saving ||
                  !status.mutations_enabled ||
                  !preferencesReady
                }
                onClick={() => void saveTemplate(true)}
              >
                Save as club default
              </button>
            </div>
          </section>
          <div className={styles.actions} style={{ marginTop: "1rem" }}>
            <button disabled={step === 0 || busy} onClick={() => go(step - 1)}>
              Back
            </button>
            {step < 4 ? (
              <button disabled={busy} onClick={() => go(step + 1)}>
                Next: {steps[step + 1]}
              </button>
            ) : null}
          </div>
        </>
      ) : null}
    </div>
  );
}
