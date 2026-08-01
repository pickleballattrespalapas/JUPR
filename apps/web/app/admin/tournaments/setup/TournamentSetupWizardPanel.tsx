"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useMemo, useState, type ReactNode } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import TournamentSetupWizardNav, {
  TOURNAMENT_SETUP_STEPS,
  tournamentSetupStepHref,
  type TournamentSetupStep,
  type TournamentSetupStepState
} from "@/components/TournamentSetupWizardNav";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { useAdminSession } from "@/lib/useAdminSession";
import TournamentCommercePanel from "../commerce/TournamentCommercePanel";
import { TournamentSetupDayCard } from "../../tournament-setup/TournamentSetupDayCard";
import {
  appendBuilderRow,
  configurationPayload,
  dayLabel,
  dayReference,
  distinctFamilyNames,
  draftSignature,
  eventDayReference,
  eventDivisionName,
  eventUsesLabelDayReference,
  issuesForPath,
  moveBuilderRow,
  newDayRow,
  newEventOptionRow,
  publishConfigurationPayload,
  recordBoolean,
  removeBuilderRow,
  replaceBuilderRow,
  setRecordString,
  validateSetupConfiguration,
  wrapBuilderRows,
  type SetupConfiguration,
  type SetupRecord,
  type ValidationIssue
} from "../../tournament-setup/tournamentSetupBuilder";
import TournamentSetupEventCard from "./TournamentSetupEventCard";

type SetupStatus = {
  enabled: boolean;
  status: string;
  warnings?: string[];
  confirmation_text?: Record<string, string>;
};

type SetupTemplate = {
  key: string;
  label: string;
  description?: string;
  days: SetupRecord[];
  event_families: SetupRecord[];
  event_options: SetupRecord[];
};

type SetupDetail = {
  ok: boolean;
  tournament: Record<string, unknown>;
  settings: Record<string, unknown>;
  days: SetupRecord[];
  event_options: SetupRecord[];
  builder_draft?: Record<string, unknown> | null;
  publish_impact?: Record<string, unknown> | null;
  publish_impact_warning?: string | null;
  registration_count?: number;
  state_fingerprint: string;
  templates?: SetupTemplate[];
};

type WriteResponse = {
  ok: boolean;
  idempotent_replay?: boolean;
  reconciled?: boolean;
  operation_key?: string;
  request_fingerprint?: string;
};

type ImpactResponse = {
  ok: boolean;
  mode: string;
  dry_run: true;
  write_count: 0;
  state_fingerprint: string;
  impact_fingerprint: string;
  publish_impact: Record<string, unknown>;
};

type BasicsDraft = {
  name: string;
  startDate: string;
  endDate: string;
};

type Props = {
  apiBase: string | null;
  clubId: string;
  status: SetupStatus | null;
  tournamentId: string;
  tournamentName: string;
  step: TournamentSetupStep;
};

const emptyConfiguration: SetupConfiguration = {
  days: [],
  eventFamilies: [],
  eventOptions: []
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white",
  minWidth: 0
};

const inputStyle = {
  width: "100%",
  minWidth: 0,
  boxSizing: "border-box" as const,
  padding: "0.55rem",
  border: "1px solid #cbd5e1",
  borderRadius: "8px",
  font: "inherit"
};

const buttonStyle = {
  display: "inline-block",
  padding: "0.6rem 0.9rem",
  borderRadius: "999px",
  border: "1px solid #0f172a",
  background: "#0f172a",
  color: "white",
  fontWeight: 800,
  textDecoration: "none",
  cursor: "pointer"
};

const ghostButtonStyle = {
  ...buttonStyle,
  background: "white",
  color: "#0f172a"
};

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function safeString(value: unknown): string {
  return value == null ? "" : String(value);
}

function dateValue(value: unknown): string {
  return value ? String(value).slice(0, 10) : "";
}

function objectValue(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function listValue(value: unknown): SetupRecord[] {
  return Array.isArray(value)
    ? value.filter(
        (row): row is SetupRecord =>
          Boolean(row) && typeof row === "object" && !Array.isArray(row)
      )
    : [];
}

function stepDefinition(step: TournamentSetupStep) {
  return TOURNAMENT_SETUP_STEPS.find((row) => row.key === step)!;
}

function stepMessageColor(message: string): string {
  return /unable|error|required|blocked|reload|cannot|missing|invalid/i.test(message)
    ? "#b91c1c"
    : "#166534";
}

function setupState(
  basics: BasicsDraft,
  settings: Record<string, unknown>,
  configuration: SetupConfiguration
): Partial<Record<TournamentSetupStep, TournamentSetupStepState>> {
  const issues = validateSetupConfiguration(configuration);
  const basicsComplete = Boolean(
    basics.name.trim() && basics.startDate && basics.endDate
  );
  const eventsComplete =
    configuration.eventOptions.length > 0 &&
    !issues.some((issue) => issue.path.startsWith("events"));
  const rulesComplete = Boolean(
    safeString(settings.registration_slug).trim() &&
      safeString(settings.registration_close_at).trim()
  );
  const scheduleComplete =
    configuration.days.length > 0 &&
    !issues.some((issue) => issue.path.startsWith("days"));
  const reviewComplete =
    basicsComplete && eventsComplete && rulesComplete && scheduleComplete;

  return {
    basics: basicsComplete ? "complete" : "in-progress",
    events: eventsComplete
      ? "complete"
      : configuration.eventOptions.length
        ? "in-progress"
        : "not-started",
    "registration-rules": rulesComplete ? "complete" : "in-progress",
    pricing: "in-progress",
    schedule: scheduleComplete
      ? "complete"
      : configuration.days.length
        ? "in-progress"
        : "not-started",
    review: reviewComplete ? "in-progress" : "blocked"
  };
}

function initialDaysFromTournament(tournament: Record<string, unknown>): SetupRecord[] {
  const start = dateValue(tournament.start_date);
  const end = dateValue(tournament.end_date);
  if (!start) return [];
  const startDate = new Date(`${start}T00:00:00Z`);
  const endDate = end ? new Date(`${end}T00:00:00Z`) : startDate;
  if (Number.isNaN(startDate.valueOf()) || Number.isNaN(endDate.valueOf())) {
    return [];
  }
  const rows: SetupRecord[] = [];
  const cursor = new Date(startDate);
  while (cursor <= endDate && rows.length < 14) {
    const row = newDayRow(rows.length + 1, `Day ${rows.length + 1}`);
    row.event_date = cursor.toISOString().slice(0, 10);
    rows.push(row);
    cursor.setUTCDate(cursor.getUTCDate() + 1);
  }
  return rows;
}

function formatImpactItem(value: unknown): string {
  if (value == null) return "No detail supplied.";
  if (typeof value !== "object") return String(value);
  const row = value as Record<string, unknown>;
  for (const key of ["message", "detail", "reason", "warning", "name"]) {
    if (row[key] != null && String(row[key]).trim()) return String(row[key]);
  }
  return JSON.stringify(row);
}

function footerRow(children: ReactNode) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        gap: "0.75rem",
        flexWrap: "wrap",
        paddingTop: "0.25rem"
      }}
    >
      {children}
    </div>
  );
}

export default function TournamentSetupWizardPanel({
  apiBase,
  clubId,
  status,
  tournamentId,
  tournamentName,
  step
}: Props) {
  const router = useRouter();
  const { accessToken, loading: sessionLoading } = useAdminSession();
  const [detail, setDetail] = useState<SetupDetail | null>(null);
  const [basics, setBasics] = useState<BasicsDraft>({
    name: tournamentName,
    startDate: "",
    endDate: ""
  });
  const [settings, setSettings] = useState<Record<string, unknown>>({});
  const [configuration, setConfiguration] =
    useState<SetupConfiguration>(emptyConfiguration);
  const [impactReview, setImpactReview] = useState<ImpactResponse | null>(null);
  const [reviewedDraftSignature, setReviewedDraftSignature] = useState("");
  const [setupPublishedThisSession, setSetupPublishedThisSession] =
    useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const detailRequest = useLatestRequestGuard(
    `${accessToken}\u0000${tournamentId}`,
    clearProtectedState
  );
  const actionRequest = useLatestRequestGuard(accessToken);

  function clearProtectedState() {
    actionRequest.invalidate();
    setDetail(null);
    setBasics({ name: tournamentName, startDate: "", endDate: "" });
    setSettings({});
    setConfiguration(emptyConfiguration);
    setImpactReview(null);
    setReviewedDraftSignature("");
    setSetupPublishedThisSession(false);
    setBusy(false);
    setMessage(null);
  }

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("Tournament Setup API is not configured.");
    if (!accessToken) throw new Error("Sign in before editing tournament setup.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(String(payload?.detail || `API error (${response.status})`));
    }
    return payload as T;
  }

  function hydrate(payload: SetupDetail) {
    const draft = objectValue(payload.builder_draft);
    const publishedDays = payload.days || [];
    const draftDays = listValue(draft.days);
    const days = draftDays.length
      ? draftDays
      : publishedDays.length
        ? publishedDays
        : initialDaysFromTournament(payload.tournament);
    const draftEvents = listValue(
      Array.isArray(draft.event_options) ? draft.event_options : draft.divisions
    );
    const events = draftEvents.length ? draftEvents : payload.event_options || [];
    const families = listValue(draft.event_families);

    setDetail(payload);
    setBasics({
      name: safeString(payload.tournament.name) || tournamentName,
      startDate: dateValue(payload.tournament.start_date),
      endDate: dateValue(payload.tournament.end_date)
    });
    setSettings(payload.settings || {});
    setConfiguration({
      days: wrapBuilderRows(days, "day"),
      eventFamilies: wrapBuilderRows(families, "family"),
      eventOptions: wrapBuilderRows(events, "event")
    });
    setImpactReview(null);
    setReviewedDraftSignature("");
  }

  async function loadDetail() {
    const generation = detailRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<SetupDetail>(
        `/admin/clubs/${encodeURIComponent(
          clubId
        )}/tournaments/setup/tournaments/${encodeURIComponent(tournamentId)}`
      );
      if (!detailRequest.isCurrent(generation)) return;
      hydrate(payload);
    } catch (error) {
      if (detailRequest.isCurrent(generation)) {
        setMessage(
          error instanceof Error
            ? error.message
            : "Unable to load tournament setup."
        );
      }
    } finally {
      if (detailRequest.isCurrent(generation)) setBusy(false);
    }
  }

  useAuthenticatedAutoLoad(
    status?.enabled ? `${accessToken}\u0000${tournamentId}` : "",
    loadDetail
  );

  function goTo(nextStep: TournamentSetupStep) {
    router.push(
      tournamentSetupStepHref(
        nextStep,
        tournamentId,
        basics.name.trim() || tournamentName
      )
    );
  }

  async function saveBasics(confirmationText: string) {
    if (!detail) return;
    if (!basics.name.trim()) {
      setMessage("Tournament name is required.");
      return;
    }
    if (!basics.startDate || !basics.endDate) {
      setMessage("Start and end dates are required before continuing.");
      return;
    }
    if (basics.endDate < basics.startDate) {
      setMessage("Tournament end date cannot be before its start date.");
      return;
    }
    const expectedUpdatedAt = safeString(detail.tournament.updated_at);
    if (!expectedUpdatedAt) {
      setMessage("Reload the tournament before saving its basics.");
      return;
    }

    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      await requestJson<WriteResponse>(
        `/admin/clubs/${encodeURIComponent(
          clubId
        )}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}`,
        {
          method: "PATCH",
          body: JSON.stringify({
            name: basics.name.trim(),
            start_date: basics.startDate,
            end_date: basics.endDate,
            expected_updated_at: expectedUpdatedAt,
            confirmation_text: confirmationText,
            source: "next_tournament_setup_wizard_basics"
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) return;
      await loadDetail();
      if (!actionRequest.isCurrent(generation)) return;
      goTo("events");
    } catch (error) {
      if (actionRequest.isCurrent(generation)) {
        setMessage(
          error instanceof Error ? error.message : "Unable to save tournament basics."
        );
      }
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function saveDraftAndContinue(
    nextStep: TournamentSetupStep,
    confirmationText: string
  ) {
    if (!detail) return;
    const issues = validateSetupConfiguration(configuration);
    const relevantIssues =
      step === "events"
        ? issues.filter((issue) => issue.path.startsWith("events"))
        : step === "schedule"
          ? issues.filter(
              (issue) =>
                issue.path.startsWith("days") ||
                issue.path.endsWith("registration_day_id")
            )
          : issues;
    if (relevantIssues.length) {
      setMessage(relevantIssues[0].message);
      return;
    }

    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const draft = configurationPayload(configuration);
      await requestJson<WriteResponse>(
        `/admin/clubs/${encodeURIComponent(
          clubId
        )}/tournaments/setup/tournaments/${encodeURIComponent(
          tournamentId
        )}/draft`,
        {
          method: "PUT",
          body: JSON.stringify({
            ...draft,
            expected_state_fingerprint: detail.state_fingerprint,
            confirmation_text: confirmationText
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) return;
      await loadDetail();
      if (!actionRequest.isCurrent(generation)) return;
      goTo(nextStep);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) {
        setMessage(
          error instanceof Error ? error.message : "Unable to save setup draft."
        );
      }
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function saveRegistrationRules(confirmationText: string) {
    if (!detail) return;
    if (!safeString(settings.registration_slug).trim()) {
      setMessage("Registration slug is required before continuing.");
      return;
    }
    if (!safeString(settings.registration_close_at).trim()) {
      setMessage("Registration close date and time are required before continuing.");
      return;
    }

    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      await requestJson<WriteResponse>(
        `/admin/clubs/${encodeURIComponent(
          clubId
        )}/tournaments/setup/tournaments/${encodeURIComponent(
          tournamentId
        )}/settings`,
        {
          method: "PATCH",
          body: JSON.stringify({
            ...settings,
            registration_status:
              safeString(settings.registration_status) || "draft",
            expected_state_fingerprint: detail.state_fingerprint,
            confirmation_text: confirmationText
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) return;
      await loadDetail();
      if (!actionRequest.isCurrent(generation)) return;
      goTo("pricing");
    } catch (error) {
      if (actionRequest.isCurrent(generation)) {
        setMessage(
          error instanceof Error
            ? error.message
            : "Unable to save registration rules."
        );
      }
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function reviewImpact() {
    if (!detail) return;
    const issues = validateSetupConfiguration(configuration);
    if (issues.length) {
      setMessage(issues[0].message);
      return;
    }
    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const draft = publishConfigurationPayload(configuration);
      const payload = await requestJson<ImpactResponse>(
        `/admin/clubs/${encodeURIComponent(
          clubId
        )}/tournaments/setup/tournaments/${encodeURIComponent(
          tournamentId
        )}/impact`,
        {
          method: "POST",
          body: JSON.stringify({
            days: draft.days,
            event_options: draft.event_options,
            expected_state_fingerprint: detail.state_fingerprint
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) return;
      setImpactReview(payload);
      setReviewedDraftSignature(draftSignature(configuration));
      setMessage("Setup review completed. No rows were changed.");
    } catch (error) {
      if (actionRequest.isCurrent(generation)) {
        setImpactReview(null);
        setReviewedDraftSignature("");
        setMessage(
          error instanceof Error ? error.message : "Unable to review setup."
        );
      }
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function publishSetup(confirmationText: string) {
    if (!detail) return;
    if (
      !impactReview ||
      reviewedDraftSignature !== draftSignature(configuration)
    ) {
      setMessage("Review the current setup before publishing it.");
      return;
    }
    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const draft = publishConfigurationPayload(configuration);
      await requestJson<WriteResponse>(
        `/admin/clubs/${encodeURIComponent(
          clubId
        )}/tournaments/setup/tournaments/${encodeURIComponent(
          tournamentId
        )}/publish`,
        {
          method: "POST",
          body: JSON.stringify({
            days: draft.days,
            event_options: draft.event_options,
            expected_state_fingerprint: detail.state_fingerprint,
            reviewed_impact_fingerprint: impactReview.impact_fingerprint,
            confirmation_text: confirmationText
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) return;
      setSetupPublishedThisSession(true);
      await loadDetail();
      if (actionRequest.isCurrent(generation)) {
        setMessage("Tournament setup published. Registration can now be opened.");
      }
    } catch (error) {
      if (actionRequest.isCurrent(generation)) {
        setMessage(
          error instanceof Error ? error.message : "Unable to publish setup."
        );
      }
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function openRegistration(confirmationText: string) {
    if (!detail) return;
    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      await requestJson<WriteResponse>(
        `/admin/clubs/${encodeURIComponent(
          clubId
        )}/tournaments/setup/tournaments/${encodeURIComponent(
          tournamentId
        )}/settings`,
        {
          method: "PATCH",
          body: JSON.stringify({
            ...settings,
            registration_status: "open",
            expected_state_fingerprint: detail.state_fingerprint,
            confirmation_text: confirmationText
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) return;
      router.push(
        `/admin/tournaments/registration?${new URLSearchParams({
          tournament: tournamentId,
          name: basics.name.trim() || tournamentName
        }).toString()}`
      );
    } catch (error) {
      if (actionRequest.isCurrent(generation)) {
        setMessage(
          error instanceof Error ? error.message : "Unable to open registration."
        );
      }
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  const states = setupState(basics, settings, configuration);
  const definition = stepDefinition(step);
  const issues = validateSetupConfiguration(configuration);
  const familyNames = distinctFamilyNames(configuration);
  const registrationStatus = safeString(settings.registration_status || "draft");
  const settingsConfirmation =
    status?.confirmation_text?.settings || "SAVE SETUP";
  const draftConfirmation =
    status?.confirmation_text?.draft || "SAVE SETUP DRAFT";
  const publishConfirmation =
    status?.confirmation_text?.publish || "PUBLISH SETUP";

  const eventsByDay = useMemo(() => {
    return configuration.days.map((day) => {
      const references = new Set(
        [dayReference(day.value), dayLabel(day.value)].filter(Boolean)
      );
      return {
        key: day.key,
        label: dayLabel(day.value) || "Untitled day",
        date: dateValue(day.value.event_date),
        events: configuration.eventOptions.filter((event) =>
          references.has(eventDayReference(event.value))
        )
      };
    });
  }, [configuration.days, configuration.eventOptions]);

  if (!status?.enabled) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Tournament Setup is unavailable</h2>
        <p>{status?.warnings?.[0] || "Tournament Setup is disabled."}</p>
      </article>
    );
  }
  if (sessionLoading && !accessToken) {
    return <p role="status">Loading tournament setup…</p>;
  }
  if (!accessToken) {
    return (
      <article style={{ ...cardStyle, background: "#fffbeb" }}>
        <h2 style={{ marginTop: 0 }}>Admin sign-in required</h2>
        <p>
          <Link href="/admin/login">Open admin login</Link>
        </p>
      </article>
    );
  }

  function renderBasics() {
    return (
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>1. Tournament basics</h2>
        <p style={{ color: "#475569" }}>
          Start with the tournament identity and dates. Saving this step takes you
          directly to Events and formats.
        </p>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))",
            gap: "0.75rem"
          }}
        >
          <label style={{ minWidth: 0 }}>
            <strong>Tournament name</strong>
            <br />
            <input
              value={basics.name}
              onChange={(event) =>
                setBasics((current) => ({
                  ...current,
                  name: event.target.value
                }))
              }
              disabled={busy}
              style={inputStyle}
            />
          </label>
          <label style={{ minWidth: 0 }}>
            <strong>Start date</strong>
            <br />
            <input
              type="date"
              value={basics.startDate}
              onChange={(event) =>
                setBasics((current) => ({
                  ...current,
                  startDate: event.target.value
                }))
              }
              disabled={busy}
              style={inputStyle}
            />
          </label>
          <label style={{ minWidth: 0 }}>
            <strong>End date</strong>
            <br />
            <input
              type="date"
              min={basics.startDate || undefined}
              value={basics.endDate}
              onChange={(event) =>
                setBasics((current) => ({
                  ...current,
                  endDate: event.target.value
                }))
              }
              disabled={busy}
              style={inputStyle}
            />
          </label>
        </div>
        <div style={{ marginTop: "1rem" }}>
          <ConfirmAction
            triggerLabel={busy ? "Saving…" : "Save and continue"}
            title="Save tournament basics and continue?"
            description="Save the tournament name and dates, then open Step 2: Events and formats."
            confirmLabel="Yes, save and continue"
            confirmationText="SAVE TOURNAMENT"
            disabled={!basics.name.trim() || !basics.startDate || !basics.endDate}
            busy={busy}
            onConfirm={saveBasics}
          />
        </div>
      </article>
    );
  }

  function renderEvents() {
    const eventIssues = issues.filter((issue) => issue.path.startsWith("events"));
    return (
      <div style={{ display: "grid", gap: "1rem" }}>
        <article style={cardStyle}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              gap: "0.75rem",
              alignItems: "flex-start",
              flexWrap: "wrap"
            }}
          >
            <div>
              <h2 style={{ marginTop: 0 }}>2. Events and formats</h2>
              <p style={{ color: "#475569", marginBottom: 0 }}>
                Review the current divisions, add new ones, and choose one clear
                format for each event. Day assignments can be refined in Step 5.
              </p>
            </div>
            <button
              type="button"
              style={buttonStyle}
              disabled={busy || !configuration.days.length}
              onClick={() =>
                setConfiguration((current) => ({
                  ...current,
                  eventOptions: appendBuilderRow(
                    current.eventOptions,
                    "event",
                    newEventOptionRow(current)
                  )
                }))
              }
            >
              Add event
            </button>
          </div>
          {!configuration.days.length ? (
            <p role="alert" style={{ color: "#b91c1c" }}>
              Add tournament dates in Step 1 before creating events.
            </p>
          ) : null}
        </article>

        {configuration.eventOptions.map((row, index) => (
          <TournamentSetupEventCard
            key={row.key}
            row={row}
            position={index}
            total={configuration.eventOptions.length}
            days={configuration.days}
            familyNames={familyNames}
            disabled={busy}
            issues={issuesForPath(issues, `events.${index}`)}
            onChange={(value) => {
              setConfiguration((current) => ({
                ...current,
                eventOptions: replaceBuilderRow(
                  current.eventOptions,
                  row.key,
                  value
                )
              }));
              setImpactReview(null);
            }}
            onMove={(direction) =>
              setConfiguration((current) => ({
                ...current,
                eventOptions: moveBuilderRow(
                  current.eventOptions,
                  row.key,
                  direction
                )
              }))
            }
            onRemove={() =>
              setConfiguration((current) => ({
                ...current,
                eventOptions: removeBuilderRow(current.eventOptions, row.key)
              }))
            }
          />
        ))}

        {!configuration.eventOptions.length ? (
          <article style={cardStyle}>
            <p style={{ margin: 0, color: "#64748b" }}>
              No events yet. Click Add event to create the first division.
            </p>
          </article>
        ) : null}

        {eventIssues.length ? (
          <article style={{ ...cardStyle, borderColor: "#fecaca" }}>
            <h3 style={{ marginTop: 0 }}>Items to fix before continuing</h3>
            <ul>
              {eventIssues.map((issue) => (
                <li key={`${issue.path}-${issue.message}`}>{issue.message}</li>
              ))}
            </ul>
          </article>
        ) : null}

        {footerRow(
          <>
            <Link
              href={tournamentSetupStepHref(
                "basics",
                tournamentId,
                basics.name || tournamentName
              )}
              style={ghostButtonStyle}
            >
              Back
            </Link>
            <ConfirmAction
              triggerLabel={busy ? "Saving…" : "Save and continue"}
              title="Save events and formats?"
              description="Save the current setup draft, then continue to Registration rules. Published registration does not change until Step 6."
              confirmLabel="Yes, save and continue"
              confirmationText={draftConfirmation}
              disabled={!configuration.eventOptions.length || eventIssues.length > 0}
              busy={busy}
              onConfirm={(confirmationText) =>
                saveDraftAndContinue("registration-rules", confirmationText)
              }
            />
          </>
        )}
      </div>
    );
  }

  function renderRegistrationRules() {
    return (
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>3. Registration rules</h2>
        <p style={{ color: "#475569" }}>
          Set when registration is available and how waitlists, partner requests,
          and public policies behave. Registration remains closed until Step 6.
        </p>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))",
            gap: "0.75rem"
          }}
        >
          <label>
            <strong>Registration link</strong>
            <br />
            <input
              value={safeString(settings.registration_slug)}
              onChange={(event) =>
                setSettings((current) => ({
                  ...current,
                  registration_slug: event.target.value
                }))
              }
              disabled={busy}
              style={inputStyle}
            />
          </label>
          <label>
            <strong>Registration opens</strong>
            <br />
            <input
              type="datetime-local"
              value={safeString(settings.registration_open_at).slice(0, 16)}
              onChange={(event) =>
                setSettings((current) => ({
                  ...current,
                  registration_open_at: event.target.value
                }))
              }
              disabled={busy}
              style={inputStyle}
            />
          </label>
          <label>
            <strong>Registration closes</strong>
            <br />
            <input
              type="datetime-local"
              value={safeString(settings.registration_close_at).slice(0, 16)}
              onChange={(event) =>
                setSettings((current) => ({
                  ...current,
                  registration_close_at: event.target.value
                }))
              }
              disabled={busy}
              style={inputStyle}
            />
          </label>
          <label
            style={{
              display: "flex",
              gap: "0.5rem",
              alignItems: "center",
              alignSelf: "end"
            }}
          >
            <input
              type="checkbox"
              checked={Boolean(settings.waitlist_enabled)}
              onChange={(event) =>
                setSettings((current) => ({
                  ...current,
                  waitlist_enabled: event.target.checked
                }))
              }
              disabled={busy}
            />
            Waitlist enabled
          </label>
          <label
            style={{
              display: "flex",
              gap: "0.5rem",
              alignItems: "center",
              alignSelf: "end"
            }}
          >
            <input
              type="checkbox"
              checked={Boolean(settings.partner_board_enabled)}
              onChange={(event) =>
                setSettings((current) => ({
                  ...current,
                  partner_board_enabled: event.target.checked
                }))
              }
              disabled={busy}
            />
            Partner Board enabled
          </label>
        </div>

        <label style={{ display: "block", marginTop: "0.75rem" }}>
          <strong>Registration rules</strong>
          <br />
          <textarea
            value={safeString(settings.rules_markdown)}
            onChange={(event) =>
              setSettings((current) => ({
                ...current,
                rules_markdown: event.target.value
              }))
            }
            rows={5}
            disabled={busy}
            style={inputStyle}
          />
        </label>
        <label style={{ display: "block", marginTop: "0.75rem" }}>
          <strong>Cancellation and refund policy</strong>
          <br />
          <textarea
            value={safeString(settings.refund_policy_markdown)}
            onChange={(event) =>
              setSettings((current) => ({
                ...current,
                refund_policy_markdown: event.target.value
              }))
            }
            rows={4}
            disabled={busy}
            style={inputStyle}
          />
        </label>

        <div style={{ marginTop: "1rem" }}>
          {footerRow(
            <>
              <Link
                href={tournamentSetupStepHref(
                  "events",
                  tournamentId,
                  basics.name || tournamentName
                )}
                style={ghostButtonStyle}
              >
                Back
              </Link>
              <ConfirmAction
                triggerLabel={busy ? "Saving…" : "Save and continue"}
                title="Save registration rules?"
                description="Save the registration window and policies, then continue to Pricing, extras, and fulfillment."
                confirmLabel="Yes, save and continue"
                confirmationText={settingsConfirmation}
                disabled={
                  !safeString(settings.registration_slug).trim() ||
                  !safeString(settings.registration_close_at).trim()
                }
                busy={busy}
                onConfirm={saveRegistrationRules}
              />
            </>
          )}
        </div>
      </article>
    );
  }

  function renderPricing() {
    return (
      <div style={{ display: "grid", gap: "1rem" }}>
        <article style={{ ...cardStyle, background: "#eff6ff", borderColor: "#bfdbfe" }}>
          <h2 style={{ marginTop: 0 }}>4. Pricing, extras, and fulfillment</h2>
          <p style={{ color: "#475569", marginBottom: 0 }}>
            Set entry fees on the event cards, then build merchandise, bundles,
            inventory, and pickup options below. Save catalog changes before
            continuing.
          </p>
        </article>
        <TournamentCommercePanel
          clubId={clubId}
          tournamentId={tournamentId}
          tournamentName={basics.name || tournamentName}
        />
        {footerRow(
          <>
            <Link
              href={tournamentSetupStepHref(
                "registration-rules",
                tournamentId,
                basics.name || tournamentName
              )}
              style={ghostButtonStyle}
            >
              Back
            </Link>
            <button
              type="button"
              style={buttonStyle}
              onClick={() => goTo("schedule")}
            >
              Continue to schedule and courts
            </button>
          </>
        )}
      </div>
    );
  }

  function renderSchedule() {
    const dayIssues = issues.filter(
      (issue) =>
        issue.path.startsWith("days") ||
        issue.path.endsWith("registration_day_id")
    );
    const dayOptions = configuration.days.map((day) => ({
      id: dayReference(day.value),
      label: dayLabel(day.value) || dayReference(day.value),
      enabled: recordBoolean(day.value.enabled, true)
    }));
    return (
      <div style={{ display: "grid", gap: "1rem" }}>
        <article style={cardStyle}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              gap: "0.75rem",
              alignItems: "flex-start",
              flexWrap: "wrap"
            }}
          >
            <div>
              <h2 style={{ marginTop: 0 }}>5. Schedule and courts</h2>
              <p style={{ color: "#475569", marginBottom: 0 }}>
                Confirm the tournament days and assign every event to a day.
                Exact court numbers and match times are finalized in Live
                Operations after registration closes.
              </p>
            </div>
            <button
              type="button"
              style={buttonStyle}
              disabled={busy}
              onClick={() =>
                setConfiguration((current) => ({
                  ...current,
                  days: appendBuilderRow(
                    current.days,
                    "day",
                    newDayRow(current.days.length + 1)
                  )
                }))
              }
            >
              Add tournament day
            </button>
          </div>
        </article>

        {configuration.days.map((row, index) => (
          <TournamentSetupDayCard
            key={row.key}
            row={row}
            position={index}
            total={configuration.days.length}
            disabled={busy}
            issues={issuesForPath(issues, `days.${index}`)}
            onChange={(value) =>
              setConfiguration((current) => {
                const previous = current.days.find(
                  (day) => day.key === row.key
                )?.value;
                const previousLabel = dayLabel(previous || {});
                const nextLabel = dayLabel(value);
                const eventOptions =
                  previousLabel && previousLabel !== nextLabel
                    ? current.eventOptions.map((event) =>
                        eventUsesLabelDayReference(event.value) &&
                        eventDayReference(event.value) === previousLabel
                          ? {
                              ...event,
                              value: setRecordString(
                                event.value,
                                ["assigned_day"],
                                nextLabel
                              )
                            }
                          : event
                      )
                    : current.eventOptions;
                return {
                  ...current,
                  days: replaceBuilderRow(current.days, row.key, value),
                  eventOptions
                };
              })
            }
            onMove={(direction) =>
              setConfiguration((current) => ({
                ...current,
                days: moveBuilderRow(current.days, row.key, direction)
              }))
            }
            onRemove={() => {
              const references = new Set([
                dayReference(row.value),
                dayLabel(row.value)
              ]);
              const attached = configuration.eventOptions.find((event) =>
                references.has(eventDayReference(event.value))
              );
              if (attached) {
                setMessage(
                  `Move ${eventDivisionName(attached.value) || "the attached event"} before removing this day.`
                );
                return;
              }
              setConfiguration((current) => ({
                ...current,
                days: removeBuilderRow(current.days, row.key)
              }));
            }}
          />
        ))}

        <article style={cardStyle}>
          <h3 style={{ marginTop: 0 }}>Event day assignments</h3>
          <div style={{ display: "grid", gap: "0.65rem" }}>
            {configuration.eventOptions.map((event) => {
              const usesLabel = eventUsesLabelDayReference(event.value);
              return (
                <label
                  key={event.key}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "minmax(180px, 1fr) minmax(180px, 1fr)",
                    gap: "0.75rem",
                    alignItems: "center"
                  }}
                >
                  <strong>{eventDivisionName(event.value) || "Untitled event"}</strong>
                  <select
                    value={eventDayReference(event.value)}
                    disabled={busy}
                    style={inputStyle}
                    onChange={(changeEvent) =>
                      setConfiguration((current) => ({
                        ...current,
                        eventOptions: replaceBuilderRow(
                          current.eventOptions,
                          event.key,
                          setRecordString(
                            event.value,
                            usesLabel
                              ? ["assigned_day"]
                              : ["registration_day_id"],
                            changeEvent.target.value
                          )
                        )
                      }))
                    }
                  >
                    <option value="">Choose a day</option>
                    {dayOptions.map((option) => (
                      <option
                        key={option.id}
                        value={usesLabel ? option.label : option.id}
                        disabled={!option.enabled}
                      >
                        {option.label}
                        {option.enabled ? "" : " (disabled)"}
                      </option>
                    ))}
                  </select>
                </label>
              );
            })}
          </div>
        </article>

        <article style={{ ...cardStyle, background: "#f8fafc" }}>
          <h3 style={{ marginTop: 0 }}>Schedule summary</h3>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
              gap: "0.75rem"
            }}
          >
            {eventsByDay.map((day) => (
              <div key={day.key}>
                <strong>{day.label}</strong>
                <br />
                <small>{day.date || "Date not set"}</small>
                <ul>
                  {day.events.map((event) => (
                    <li key={event.key}>
                      {eventDivisionName(event.value) || "Untitled event"}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </article>

        {dayIssues.length ? (
          <article style={{ ...cardStyle, borderColor: "#fecaca" }}>
            <h3 style={{ marginTop: 0 }}>Items to fix before continuing</h3>
            <ul>
              {dayIssues.map((issue) => (
                <li key={`${issue.path}-${issue.message}`}>{issue.message}</li>
              ))}
            </ul>
          </article>
        ) : null}

        {footerRow(
          <>
            <Link
              href={tournamentSetupStepHref(
                "pricing",
                tournamentId,
                basics.name || tournamentName
              )}
              style={ghostButtonStyle}
            >
              Back
            </Link>
            <ConfirmAction
              triggerLabel={busy ? "Saving…" : "Save and continue"}
              title="Save the tournament schedule?"
              description="Save the tournament days and event assignments, then continue to final review."
              confirmLabel="Yes, save and continue"
              confirmationText={draftConfirmation}
              disabled={!configuration.days.length || dayIssues.length > 0}
              busy={busy}
              onConfirm={(confirmationText) =>
                saveDraftAndContinue("review", confirmationText)
              }
            />
          </>
        )}
      </div>
    );
  }

  function renderReview() {
    const impact = impactReview?.publish_impact || {};
    const blocked = Array.isArray(impact.blocked) ? impact.blocked : [];
    const warnings = Array.isArray(impact.warnings) ? impact.warnings : [];
    const basicsReady = Boolean(
      basics.name.trim() && basics.startDate && basics.endDate
    );
    const eventsReady = configuration.eventOptions.length > 0;
    const scheduleReady = configuration.days.length > 0;
    const rulesReady = Boolean(
      safeString(settings.registration_slug).trim() &&
        safeString(settings.registration_close_at).trim()
    );
    const ready =
      basicsReady &&
      eventsReady &&
      scheduleReady &&
      rulesReady &&
      issues.length === 0;

    return (
      <div style={{ display: "grid", gap: "1rem" }}>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>6. Review and open registration</h2>
          <p style={{ color: "#475569" }}>
            Review the complete tournament before anything becomes public. The
            three actions below are deliberate: review changes, publish setup,
            then open registration.
          </p>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))",
              gap: "0.75rem"
            }}
          >
            {[
              ["Tournament basics", basicsReady, `${basics.startDate || "No start"} – ${basics.endDate || "No end"}`],
              ["Events and formats", eventsReady, `${configuration.eventOptions.length} event(s)`],
              ["Registration rules", rulesReady, `Status: ${registrationStatus}`],
              ["Pricing and extras", true, "Review the saved catalog from Step 4"],
              ["Schedule and courts", scheduleReady, `${configuration.days.length} tournament day(s)`]
            ].map(([label, complete, note]) => (
              <div
                key={String(label)}
                style={{
                  padding: "0.75rem",
                  border: `1px solid ${complete ? "#bbf7d0" : "#fecaca"}`,
                  borderRadius: "12px",
                  background: complete ? "#f0fdf4" : "#fef2f2"
                }}
              >
                <strong>{complete ? "✓" : "!"} {String(label)}</strong>
                <br />
                <small>{String(note)}</small>
              </div>
            ))}
          </div>
        </article>

        {issues.length ? (
          <article style={{ ...cardStyle, borderColor: "#fecaca" }}>
            <h3 style={{ marginTop: 0 }}>Setup issues</h3>
            <ul>
              {issues.map((issue: ValidationIssue) => (
                <li key={`${issue.path}-${issue.message}`}>{issue.message}</li>
              ))}
            </ul>
          </article>
        ) : null}

        <article style={cardStyle}>
          <h3 style={{ marginTop: 0 }}>A. Review setup changes</h3>
          <p style={{ color: "#475569" }}>
            This calculates the exact impact without writing any rows.
          </p>
          <button
            type="button"
            style={buttonStyle}
            disabled={!ready || busy}
            onClick={() => void reviewImpact()}
          >
            {busy ? "Reviewing…" : "Review setup impact"}
          </button>
          {impactReview ? (
            <div style={{ marginTop: "0.75rem" }}>
              <p style={{ color: "#166534", fontWeight: 800 }}>
                Review complete. No rows were written.
              </p>
              {warnings.length ? (
                <>
                  <strong>Warnings</strong>
                  <ul>
                    {warnings.map((warning, index) => (
                      <li key={index}>{formatImpactItem(warning)}</li>
                    ))}
                  </ul>
                </>
              ) : null}
              {blocked.length ? (
                <>
                  <strong>Blocked changes</strong>
                  <ul>
                    {blocked.map((item, index) => (
                      <li key={index}>{formatImpactItem(item)}</li>
                    ))}
                  </ul>
                </>
              ) : null}
            </div>
          ) : null}
        </article>

        <article style={cardStyle}>
          <h3 style={{ marginTop: 0 }}>B. Publish tournament setup</h3>
          <p style={{ color: "#475569" }}>
            Publish the reviewed days and events. Existing registrations are
            protected by the impact review and guarded write.
          </p>
          <ConfirmAction
            triggerLabel={busy ? "Publishing…" : "Publish reviewed setup"}
            title="Publish this reviewed tournament setup?"
            description="Apply the exact reviewed day and event configuration. Registration remains closed until the next action."
            confirmLabel="Yes, publish setup"
            confirmationText={publishConfirmation}
            disabled={
              !impactReview ||
              reviewedDraftSignature !== draftSignature(configuration) ||
              blocked.length > 0
            }
            busy={busy}
            onConfirm={publishSetup}
          />
        </article>

        <article style={cardStyle}>
          <h3 style={{ marginTop: 0 }}>C. Open registration</h3>
          <p style={{ color: "#475569" }}>
            Make the published tournament available to registrants. Offline
            payment remains the only payment mode.
          </p>
          {registrationStatus.toLowerCase() === "open" ? (
            <p style={{ color: "#166534", fontWeight: 800 }}>
              Registration is already open.
            </p>
          ) : (
            <ConfirmAction
              triggerLabel={busy ? "Opening…" : "Open registration"}
              title="Open tournament registration?"
              description="Open registration using the saved window, rules, events, prices, and Partner Board settings."
              confirmLabel="Yes, open registration"
              confirmationText={settingsConfirmation}
              disabled={!setupPublishedThisSession}
              busy={busy}
              onConfirm={openRegistration}
            />
          )}
          {!setupPublishedThisSession &&
          registrationStatus.toLowerCase() !== "open" ? (
            <p style={{ color: "#64748b" }}>
              Publish the reviewed setup above before opening registration.
            </p>
          ) : null}
        </article>

        {footerRow(
          <>
            <Link
              href={tournamentSetupStepHref(
                "schedule",
                tournamentId,
                basics.name || tournamentName
              )}
              style={ghostButtonStyle}
            >
              Back
            </Link>
            <Link
              href={`/admin/tournaments/tournament?${new URLSearchParams({
                tournament: tournamentId,
                name: basics.name || tournamentName
              }).toString()}`}
              style={ghostButtonStyle}
            >
              Return to Tournament Home
            </Link>
          </>
        )}
      </div>
    );
  }

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <TournamentSetupWizardNav
        currentStep={step}
        tournamentId={tournamentId}
        tournamentName={basics.name || tournamentName}
        states={states}
      />

      <article style={{ ...cardStyle, background: "#eff6ff", borderColor: "#bfdbfe" }}>
        <p
          style={{
            margin: "0 0 0.35rem",
            color: "#1d4ed8",
            fontWeight: 900
          }}
        >
          Step {definition.number} of 6
        </p>
        <h1 style={{ margin: 0 }}>{definition.label}</h1>
        <p style={{ color: "#475569", marginBottom: 0 }}>
          {definition.description}
        </p>
      </article>

      {message ? (
        <p role="status" style={{ color: stepMessageColor(message) }}>
          {message}
        </p>
      ) : null}
      {busy && !detail ? (
        <p role="status">Loading {tournamentName} setup…</p>
      ) : null}

      {detail ? (
        step === "basics" ? (
          renderBasics()
        ) : step === "events" ? (
          renderEvents()
        ) : step === "registration-rules" ? (
          renderRegistrationRules()
        ) : step === "pricing" ? (
          renderPricing()
        ) : step === "schedule" ? (
          renderSchedule()
        ) : (
          renderReview()
        )
      ) : null}
    </div>
  );
}
