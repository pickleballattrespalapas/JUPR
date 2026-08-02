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
  draftSignature,
  eventDayReference,
  eventDayReferences,
  eventDivisionName,
  eventFamilyName,
  eventUsesLabelDayReference,
  issuesForPath,
  moveBuilderRow,
  newDayRow,
  newEventFamilyRow,
  newEventOptionRow,
  publishConfigurationPayload,
  recordBoolean,
  removeBuilderRow,
  replaceBuilderRow,
  setEventDayReferences,
  setRecordString,
  validateSetupConfiguration,
  wrapBuilderRows,
  type SetupConfiguration,
  type SetupRecord,
  type ValidationIssue
} from "../../tournament-setup/tournamentSetupBuilder";
import TournamentSetupEventFamilyCard from "./TournamentSetupEventFamilyCard";
import TournamentSetupDivisionCard from "./TournamentSetupDivisionCard";
import TournamentSetupDivisionDialog from "./TournamentSetupDivisionDialog";
import TournamentSetupPolicies, { withDefaultTournamentPolicies } from "./TournamentSetupPolicies";

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

type SponsorDraft = {
  id: string;
  name: string;
  level: string;
  website: string;
  notes: string;
};

type BasicsDraft = {
  name: string;
  startDate: string;
  endDate: string;
  locationName: string;
  timezone: string;
  sponsors: SponsorDraft[];
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

const TIMEZONE_OPTIONS = [
  ["America/Mazatlan", "Baja California Sur · America/Mazatlan"],
  ["America/Los_Angeles", "Pacific · America/Los_Angeles"],
  ["America/Denver", "Mountain · America/Denver"],
  ["America/Phoenix", "Arizona · America/Phoenix"],
  ["America/Chicago", "Central · America/Chicago"],
  ["America/New_York", "Eastern · America/New_York"],
  ["UTC", "UTC"]
] as const;

function sponsorId(): string {
  return globalThis.crypto?.randomUUID?.() || `sponsor-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function newSponsor(): SponsorDraft {
  return { id: sponsorId(), name: "", level: "", website: "", notes: "" };
}

function normalizeSponsors(value: unknown): SponsorDraft[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((row): row is Record<string, unknown> => Boolean(row) && typeof row === "object" && !Array.isArray(row))
    .map((row) => ({
      id: safeString(row.id) || sponsorId(),
      name: safeString(row.name),
      level: safeString(row.level),
      website: safeString(row.website),
      notes: safeString(row.notes)
    }));
}

function emptyBasics(name: string): BasicsDraft {
  return {
    name,
    startDate: "",
    endDate: "",
    locationName: "",
    timezone: "America/Mazatlan",
    sponsors: []
  };
}

function derivedEventFamilies(events: SetupRecord[], days: SetupRecord[]): SetupRecord[] {
  const rows = new Map<string, SetupRecord>();
  for (const event of events) {
    const family = safeString(event.event_family_label ?? event.event_family);
    if (!family) continue;
    const key = family.toLowerCase();
    const schedule = eventDayReferences(event);
    const existing = rows.get(key);
    if (existing) {
      rows.set(
        key,
        setEventDayReferences(existing, [
          ...eventDayReferences(existing),
          ...schedule
        ])
      );
      continue;
    }
    const base = newEventFamilyRow(rows.size + 1, family);
    rows.set(
      key,
      setEventDayReferences(
        {
          ...base,
          event_family: family,
          participant_type: safeString(event.event_type ?? event.participant_type) || "GENDER_DOUBLES",
          gender_restriction: safeString(event.gender_restriction) || "ANY",
          default_format: safeString(event.event_format_default) || "ROUND_ROBIN_PLUS_PLAYOFF",
          default_scoring: safeString(event.scoring_default) || "GAME_TO_15",
          default_waitlist: recordBoolean(event.waitlist_enabled, true),
          default_partner_board: recordBoolean(event.partner_board_enabled, true),
          default_capacity_teams: Number(event.capacity_teams) || 16,
          default_price_usd: Number(event.price_usd) || 0
        },
        schedule
      )
    );
  }
  if (rows.size) return [...rows.values()];
  const firstDay = days.find((row) => recordBoolean(row.enabled, true)) || days[0] || {};
  return [
    setEventDayReferences(
      newEventFamilyRow(1, "Event 1"),
      [dayReference(firstDay)].filter(Boolean)
    )
  ];
}


function globalDivisionStatus(registrationStatus: unknown): string {
  const status = safeString(registrationStatus).toLowerCase();
  if (status === "open") return "open";
  if (status === "closed") return "closed";
  return "draft";
}

function configurationWithGlobalStatus(
  configuration: SetupConfiguration,
  registrationStatus: unknown
): SetupConfiguration {
  const status = globalDivisionStatus(registrationStatus);
  return {
    ...configuration,
    eventOptions: configuration.eventOptions.map((row) => ({
      ...row,
      value: { ...row.value, status }
    }))
  };
}

function divisionForFirstEvent(configuration: SetupConfiguration): SetupRecord {
  const value = newEventOptionRow(configuration);
  const event = configuration.eventFamilies[0]?.value || {};
  const family = eventFamilyName(event);
  const day = eventDayReference(event);
  return {
    ...value,
    event_family_label: family,
    event_family: family,
    registration_day_id: day || value.registration_day_id,
    assigned_day: day || value.assigned_day,
    event_type: safeString(event.participant_type) || value.event_type,
    participant_type: safeString(event.participant_type) || value.participant_type,
    gender_restriction: safeString(event.gender_restriction) || value.gender_restriction,
    capacity_teams: event.default_capacity_teams ?? value.capacity_teams,
    price_usd: event.default_price_usd ?? value.price_usd,
    waitlist_enabled: recordBoolean(event.default_waitlist, true),
    partner_board_enabled: recordBoolean(event.default_partner_board, true)
  };
}

function setupState(
  basics: BasicsDraft,
  settings: Record<string, unknown>,
  configuration: SetupConfiguration
): Partial<Record<TournamentSetupStep, TournamentSetupStepState>> {
  const issues = validateSetupConfiguration(configuration);
  const policiesComplete = Boolean(
    safeString(settings.registration_slug).trim() &&
      safeString(settings.registration_open_at).trim() &&
      safeString(settings.registration_close_at).trim() &&
      safeString(settings.rules_markdown).trim() &&
      safeString(settings.refund_policy_markdown).trim() &&
      safeString(settings.weather_policy_markdown).trim()
  );
  const basicsComplete = Boolean(
    basics.name.trim() &&
      basics.startDate &&
      basics.endDate &&
      basics.locationName.trim() &&
      basics.timezone &&
      policiesComplete
  );
  const scheduleComplete =
    configuration.days.length > 0 &&
    !issues.some((issue) => issue.path.startsWith("days"));
  const eventsComplete =
    configuration.eventFamilies.length > 0 &&
    !issues.some((issue) => issue.path.startsWith("families"));
  const divisionsComplete =
    configuration.eventOptions.length > 0 &&
    !issues.some((issue) => issue.path.startsWith("events"));
  const reviewComplete =
    basicsComplete && scheduleComplete && eventsComplete && divisionsComplete;

  return {
    basics: basicsComplete ? "complete" : "in-progress",
    schedule: scheduleComplete
      ? "complete"
      : configuration.days.length
        ? "in-progress"
        : "not-started",
    events: eventsComplete
      ? "complete"
      : configuration.eventFamilies.length
        ? "in-progress"
        : "not-started",
    divisions: divisionsComplete
      ? "complete"
      : configuration.eventOptions.length
        ? "in-progress"
        : "not-started",
    pricing: "in-progress",
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
  const [basics, setBasics] = useState<BasicsDraft>(() =>
    emptyBasics(tournamentName)
  );
  const [settings, setSettings] = useState<Record<string, unknown>>({});
  const [configuration, setConfiguration] =
    useState<SetupConfiguration>(emptyConfiguration);
  const [impactReview, setImpactReview] = useState<ImpactResponse | null>(null);
  const [reviewedDraftSignature, setReviewedDraftSignature] = useState("");
  const [setupPublishedThisSession, setSetupPublishedThisSession] =
    useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [divisionDialogOpen, setDivisionDialogOpen] = useState(false);

  const detailRequest = useLatestRequestGuard(
    `${accessToken}\u0000${tournamentId}`,
    clearProtectedState
  );
  const actionRequest = useLatestRequestGuard(accessToken);

  function clearProtectedState() {
    actionRequest.invalidate();
    setDetail(null);
    setBasics(emptyBasics(tournamentName));
    setSettings({});
    setConfiguration(emptyConfiguration);
    setImpactReview(null);
    setReviewedDraftSignature("");
    setSetupPublishedThisSession(false);
    setBusy(false);
    setMessage(null);
    setDivisionDialogOpen(false);
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
  const draftFamilies = listValue(draft.event_families);
  const families = draftFamilies.length
    ? draftFamilies
    : derivedEventFamilies(events, days);
  const loadedSettings = withDefaultTournamentPolicies(payload.settings || {});

  setDetail(payload);
  setBasics({
    name: safeString(payload.tournament.name) || tournamentName,
    startDate: dateValue(payload.tournament.start_date),
    endDate: dateValue(payload.tournament.end_date),
    locationName: safeString(loadedSettings.location_name),
    timezone: safeString(loadedSettings.timezone) || "America/Mazatlan",
    sponsors: normalizeSponsors(loadedSettings.sponsors_json)
  });
  setSettings(loadedSettings);
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

  async function saveBasics() {
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
  if (!basics.locationName.trim()) {
    setMessage("Tournament location is required before continuing.");
    return;
  }
  if (!basics.timezone) {
    setMessage("Tournament timezone is required before continuing.");
    return;
  }
  if (basics.sponsors.some((sponsor) => !sponsor.name.trim())) {
    setMessage("Every sponsor needs a name or should be removed.");
    return;
  }
  if (!safeString(settings.registration_slug).trim()) {
    setMessage("Registration link is required before continuing.");
    return;
  }
  if (!safeString(settings.registration_open_at).trim() || !safeString(settings.registration_close_at).trim()) {
    setMessage("Registration opening and closing dates are required before continuing.");
    return;
  }
  if (!safeString(settings.rules_markdown).trim()) {
    setMessage("Choose or write registration rules before continuing.");
    return;
  }
  if (!safeString(settings.refund_policy_markdown).trim()) {
    setMessage("Choose or write a cancellation policy before continuing.");
    return;
  }
  if (!safeString(settings.weather_policy_markdown).trim()) {
    setMessage("Choose or write a weather policy before continuing.");
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
      `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/setup/tournaments/${encodeURIComponent(tournamentId)}/settings`,
      {
        method: "PATCH",
        body: JSON.stringify({
          registration_slug: safeString(settings.registration_slug).trim(),
          locale: safeString(settings.locale) || "en",
          registration_status: safeString(settings.registration_status) || "draft",
          registration_open_at: settings.registration_open_at || null,
          registration_close_at: settings.registration_close_at || null,
          waitlist_enabled: Boolean(settings.waitlist_enabled),
          partner_board_enabled: Boolean(settings.partner_board_enabled),
          rules_markdown: safeString(settings.rules_markdown),
          refund_policy_markdown: safeString(settings.refund_policy_markdown),
          weather_policy_markdown: safeString(settings.weather_policy_markdown),
          sponsor_markdown: safeString(settings.sponsor_markdown),
          location_name: basics.locationName.trim(),
          timezone: basics.timezone,
          sponsors_json: basics.sponsors.map((sponsor) => ({
            id: sponsor.id,
            name: sponsor.name.trim(),
            level: sponsor.level.trim(),
            website: sponsor.website.trim(),
            notes: sponsor.notes.trim()
          })),
          expected_state_fingerprint: detail.state_fingerprint,
          confirmation_text: "SAVE SETUP",
          source: "next_tournament_setup_wizard_basics_and_policies"
        })
      }
    );
    if (!actionRequest.isCurrent(generation)) return;
    await requestJson<WriteResponse>(
      `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}`,
      {
        method: "PATCH",
        body: JSON.stringify({
          name: basics.name.trim(),
          start_date: basics.startDate,
          end_date: basics.endDate,
          expected_updated_at: expectedUpdatedAt,
          confirmation_text: "SAVE TOURNAMENT",
          source: "next_tournament_setup_wizard_basics"
        })
      }
    );
    if (!actionRequest.isCurrent(generation)) return;
    await loadDetail();
    if (!actionRequest.isCurrent(generation)) return;
    goTo("schedule");
  } catch (error) {
    if (actionRequest.isCurrent(generation)) {
      setMessage(error instanceof Error ? error.message : "Unable to save tournament basics and policies.");
    }
  } finally {
    if (actionRequest.isCurrent(generation)) setBusy(false);
  }
}


async function saveDraftAndContinue(nextStep: TournamentSetupStep) {
  if (!detail) return;
  const issues = validateSetupConfiguration(configuration);
  const relevantIssues =
    step === "events"
      ? issues.filter((issue) => issue.path.startsWith("families"))
      : step === "divisions"
        ? issues.filter((issue) => issue.path.startsWith("events"))
        : step === "schedule"
          ? issues.filter((issue) => issue.path.startsWith("days"))
          : issues;
  if (relevantIssues.length) {
    setMessage(relevantIssues[0].message);
    return;
  }

  const generation = actionRequest.begin();
  setBusy(true);
  setMessage(null);
  try {
    const normalized = configurationWithGlobalStatus(
      configuration,
      settings.registration_status
    );
    const draft = configurationPayload(normalized);
    await requestJson<WriteResponse>(
      `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/setup/tournaments/${encodeURIComponent(tournamentId)}/draft`,
      {
        method: "PUT",
        body: JSON.stringify({
          ...draft,
          saved_step: step,
          expected_state_fingerprint: detail.state_fingerprint,
          confirmation_text: draftConfirmation
        })
      }
    );
    if (!actionRequest.isCurrent(generation)) return;
    await loadDetail();
    if (!actionRequest.isCurrent(generation)) return;
    goTo(nextStep);
  } catch (error) {
    if (actionRequest.isCurrent(generation)) {
      setMessage(error instanceof Error ? error.message : "Unable to save setup draft.");
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
      const normalized = configurationWithGlobalStatus(
        configuration,
        settings.registration_status
      );
      const draft = publishConfigurationPayload(normalized);
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
      const normalized = configurationWithGlobalStatus(
        configuration,
        settings.registration_status
      );
      const draft = publishConfigurationPayload(normalized);
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

  const states = detail ? setupState(basics, settings, configuration) : {};
  const definition = stepDefinition(step);
  const issues = validateSetupConfiguration(configuration);
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
      <div style={{ display: "grid", gap: "1rem" }}>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>1. Tournament basics and policies</h2>
          <p style={{ color: "#475569" }}>
            Set the tournament identity, registration window, public policies,
            location, timezone, and sponsors. Save and continue moves directly to
            Schedule and courts without another confirmation.
          </p>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))",
              gap: "0.75rem"
            }}
          >
            <label><strong>Tournament name</strong><br /><input value={basics.name} onChange={(event) => setBasics((current) => ({ ...current, name: event.target.value }))} disabled={busy} style={inputStyle} /></label>
            <label><strong>Start date</strong><br /><input type="date" value={basics.startDate} onChange={(event) => setBasics((current) => ({ ...current, startDate: event.target.value }))} disabled={busy} style={inputStyle} /></label>
            <label><strong>End date</strong><br /><input type="date" min={basics.startDate || undefined} value={basics.endDate} onChange={(event) => setBasics((current) => ({ ...current, endDate: event.target.value }))} disabled={busy} style={inputStyle} /></label>
            <label><strong>Location or venue</strong><br /><input value={basics.locationName} onChange={(event) => setBasics((current) => ({ ...current, locationName: event.target.value }))} placeholder="Tres Palapas Baja Pickleball Resort" disabled={busy} style={inputStyle} /></label>
            <label><strong>Timezone</strong><br /><select value={basics.timezone} onChange={(event) => setBasics((current) => ({ ...current, timezone: event.target.value }))} disabled={busy} style={inputStyle}>{TIMEZONE_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
          </div>
        </article>

        <article style={cardStyle}>
          <h3 style={{ marginTop: 0 }}>Registration window and policies</h3>
          <TournamentSetupPolicies
            settings={settings}
            registrationStatus={registrationStatus}
            disabled={busy}
            inputStyle={inputStyle}
            onChange={setSettings}
          />
        </article>

        <article style={cardStyle}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", alignItems: "center", flexWrap: "wrap" }}>
            <div><h3 style={{ margin: 0 }}>Sponsors</h3><p style={{ margin: "0.25rem 0 0", color: "#64748b" }}>Add, edit, or remove sponsors shown with this tournament.</p></div>
            <button type="button" style={ghostButtonStyle} disabled={busy} onClick={() => setBasics((current) => ({ ...current, sponsors: [...current.sponsors, newSponsor()] }))}>Add sponsor</button>
          </div>
          <div style={{ display: "grid", gap: "0.75rem", marginTop: "0.75rem" }}>
            {basics.sponsors.map((sponsor, index) => (
              <article key={sponsor.id} style={{ ...cardStyle, background: "#f8fafc" }}>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(175px, 1fr))", gap: "0.65rem" }}>
                  <label><strong>Sponsor name</strong><br /><input value={sponsor.name} onChange={(event) => setBasics((current) => ({ ...current, sponsors: current.sponsors.map((row) => row.id === sponsor.id ? { ...row, name: event.target.value } : row) }))} disabled={busy} style={inputStyle} /></label>
                  <label><strong>Level or label</strong><br /><input value={sponsor.level} onChange={(event) => setBasics((current) => ({ ...current, sponsors: current.sponsors.map((row) => row.id === sponsor.id ? { ...row, level: event.target.value } : row) }))} placeholder="Title sponsor" disabled={busy} style={inputStyle} /></label>
                  <label><strong>Website</strong><br /><input type="url" value={sponsor.website} onChange={(event) => setBasics((current) => ({ ...current, sponsors: current.sponsors.map((row) => row.id === sponsor.id ? { ...row, website: event.target.value } : row) }))} placeholder="https://" disabled={busy} style={inputStyle} /></label>
                  <label><strong>Notes</strong><br /><input value={sponsor.notes} onChange={(event) => setBasics((current) => ({ ...current, sponsors: current.sponsors.map((row) => row.id === sponsor.id ? { ...row, notes: event.target.value } : row) }))} disabled={busy} style={inputStyle} /></label>
                </div>
                <button type="button" style={{ ...ghostButtonStyle, marginTop: "0.65rem", color: "#991b1b", borderColor: "#fecaca" }} disabled={busy} onClick={() => setBasics((current) => ({ ...current, sponsors: current.sponsors.filter((row) => row.id !== sponsor.id) }))}>Remove sponsor {index + 1}</button>
              </article>
            ))}
            {!basics.sponsors.length ? <p style={{ color: "#64748b" }}>No sponsors added yet.</p> : null}
          </div>
        </article>

        <div>
          <button type="button" style={buttonStyle} disabled={busy} onClick={() => void saveBasics()}>{busy ? "Saving…" : "Save and continue"}</button>
        </div>
      </div>
    );
  }

function renderEvents() {
  const familyIssues = issues.filter((issue) => issue.path.startsWith("families"));
  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <article style={cardStyle}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", alignItems: "flex-start", flexWrap: "wrap" }}>
          <div>
            <h2 style={{ marginTop: 0 }}>3. Events</h2>
            <p style={{ color: "#475569", marginBottom: 0 }}>
              Create event families such as Gender Doubles on Tuesday or Mixed Doubles on Wednesday. Skill and age divisions are created separately in Step 3.
            </p>
          </div>
          <button
            type="button"
            style={buttonStyle}
            disabled={busy || !configuration.days.length}
            onClick={() =>
              setConfiguration((current) => {
                const day =
                  current.days.find((row) => recordBoolean(row.value.enabled, true))?.value ||
                  current.days[0]?.value ||
                  {};
                return {
                  ...current,
                  eventFamilies: appendBuilderRow(
                    current.eventFamilies,
                    "family",
                    {
                      ...newEventFamilyRow(current.eventFamilies.length + 1),
                      registration_day_id: dayReference(day)
                    }
                  )
                };
              })
            }
          >
            Add event
          </button>
        </div>
      </article>

      {configuration.eventFamilies.map((row, index) => {
        const family = eventFamilyName(row.value);
        const divisions = configuration.eventOptions.filter(
          (division) =>
            eventFamilyName(division.value).toLowerCase() === family.toLowerCase()
        );
        return (
          <TournamentSetupEventFamilyCard
            key={row.key}
            row={row}
            position={index}
            total={configuration.eventFamilies.length}
            days={configuration.days}
            disabled={busy}
            issues={issuesForPath(issues, `families.${index}`)}
            divisionCount={divisions.length}
            onChange={(value) =>
              setConfiguration((current) => {
                const previousName = eventFamilyName(row.value);
                const nextName = eventFamilyName(value);
                const nextDays = eventDayReferences(value);
                return {
                  ...current,
                  eventFamilies: replaceBuilderRow(current.eventFamilies, row.key, value),
                  eventOptions: current.eventOptions.map((division) => {
                    if (
                      eventFamilyName(division.value).toLowerCase() !==
                      previousName.toLowerCase()
                    ) {
                      return division;
                    }
                    let nextValue: SetupRecord = {
                      ...division.value,
                      event_family_label: nextName,
                      event_family: nextName
                    };
                    if (safeString(division.value.schedule_mode || "INHERIT_EVENT") !== "CUSTOM") {
                      nextValue = setEventDayReferences(
                        { ...nextValue, schedule_mode: "INHERIT_EVENT" },
                        nextDays
                      );
                    }
                    return { ...division, value: nextValue };
                  })
                };
              })
            }
            onMove={(direction) =>
              setConfiguration((current) => ({
                ...current,
                eventFamilies: moveBuilderRow(
                  current.eventFamilies,
                  row.key,
                  direction
                )
              }))
            }
            onRemove={() =>
              setConfiguration((current) => ({
                ...current,
                eventFamilies: removeBuilderRow(current.eventFamilies, row.key)
              }))
            }
          />
        );
      })}

      {familyIssues.length ? (
        <article style={{ ...cardStyle, borderColor: "#fecaca" }}>
          <h3 style={{ marginTop: 0 }}>Items to fix before continuing</h3>
          <ul>
            {familyIssues.map((issue) => (
              <li key={`${issue.path}-${issue.message}`}>{issue.message}</li>
            ))}
          </ul>
        </article>
      ) : null}

      {footerRow(
        <>
          <Link
            href={tournamentSetupStepHref("schedule", tournamentId, basics.name || tournamentName)}
            style={ghostButtonStyle}
          >
            Back
          </Link>
          <button
            type="button"
            style={buttonStyle}
            disabled={busy || !configuration.eventFamilies.length || familyIssues.length > 0}
            onClick={() => void saveDraftAndContinue("divisions")}
          >
            {busy ? "Saving…" : "Save and continue"}
          </button>
        </>
      )}
    </div>
  );
}

function renderDivisions() {
  const divisionIssues = issues.filter((issue) => issue.path.startsWith("events"));
  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <TournamentSetupDivisionDialog
        open={divisionDialogOpen}
        initialValue={divisionForFirstEvent(configuration)}
        eventFamilies={configuration.eventFamilies}
        days={configuration.days}
        onCancel={() => setDivisionDialogOpen(false)}
        onConfirm={(value) => {
          const divisionId = safeString(value.id);
          setConfiguration((current) => ({
            ...current,
            eventOptions: appendBuilderRow(current.eventOptions, "event", value)
          }));
          setImpactReview(null);
          setDivisionDialogOpen(false);
          setMessage(`Division ${eventDivisionName(value) || "added"} added to the setup list.`);
          globalThis.setTimeout(() => {
            document.getElementById(`division-${divisionId}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
          }, 50);
        }}
      />

      <article style={cardStyle}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", alignItems: "flex-start", flexWrap: "wrap" }}>
          <div>
            <h2 style={{ marginTop: 0 }}>4. Divisions</h2>
            <p style={{ color: "#475569", marginBottom: 0 }}>
              Create skill and age divisions inside each event. A division may use every parent-event day or a selected subset.
            </p>
          </div>
          <button
            type="button"
            style={buttonStyle}
            disabled={busy || !configuration.eventFamilies.length}
            onClick={() => setDivisionDialogOpen(true)}
          >
            Add division
          </button>
        </div>
        {!configuration.eventFamilies.length ? (
          <p role="alert" style={{ color: "#b91c1c" }}>
            Create an event in Step 3 before adding divisions.
          </p>
        ) : null}
      </article>

      {configuration.eventOptions.map((row, index) => (
        <div key={row.key} id={`division-${safeString(row.value.id)}`}>
          <TournamentSetupDivisionCard
            row={row}
            position={index}
            total={configuration.eventOptions.length}
            eventFamilies={configuration.eventFamilies}
            days={configuration.days}
            disabled={busy}
            issues={issuesForPath(issues, `events.${index}`)}
            onChange={(value) => {
              setConfiguration((current) => ({
                ...current,
                eventOptions: replaceBuilderRow(current.eventOptions, row.key, value)
              }));
              setImpactReview(null);
            }}
            onMove={(direction) =>
              setConfiguration((current) => ({
                ...current,
                eventOptions: moveBuilderRow(current.eventOptions, row.key, direction)
              }))
            }
            onRemove={() =>
              setConfiguration((current) => ({
                ...current,
                eventOptions: removeBuilderRow(current.eventOptions, row.key)
              }))
            }
          />
        </div>
      ))}

      {!configuration.eventOptions.length ? (
        <article style={cardStyle}>
          <p style={{ margin: 0, color: "#64748b" }}>
            No divisions yet. Click Add division to open the focused setup dialog.
          </p>
        </article>
      ) : null}
      {divisionIssues.length ? (
        <article style={{ ...cardStyle, borderColor: "#fecaca" }}>
          <h3 style={{ marginTop: 0 }}>Items to fix before continuing</h3>
          <ul>
            {divisionIssues.map((issue) => (
              <li key={`${issue.path}-${issue.message}`}>{issue.message}</li>
            ))}
          </ul>
        </article>
      ) : null}

      {footerRow(
        <>
          <Link
            href={tournamentSetupStepHref("events", tournamentId, basics.name || tournamentName)}
            style={ghostButtonStyle}
          >
            Back
          </Link>
          <button
            type="button"
            style={buttonStyle}
            disabled={busy || !configuration.eventOptions.length || divisionIssues.length > 0}
            onClick={() => void saveDraftAndContinue("pricing")}
          >
            {busy ? "Saving…" : "Save and continue"}
          </button>
        </>
      )}
    </div>
  );
}

  function renderPricing() {
    return (
      <div style={{ display: "grid", gap: "1rem" }}>
        <article style={{ ...cardStyle, background: "#eff6ff", borderColor: "#bfdbfe" }}>
          <h2 style={{ marginTop: 0 }}>5. Pricing, extras, and fulfillment</h2>
          <p style={{ color: "#475569", marginBottom: 0 }}>
            Set entry fees on the division cards, then build merchandise, bundles,
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
                "divisions",
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
              onClick={() => goTo("review")}
            >
              Continue to final review
            </button>
          </>
        )}
      </div>
    );
  }

  function renderSchedule() {
    const dayIssues = issues.filter((issue) => issue.path.startsWith("days"));
    return (
      <div style={{ display: "grid", gap: "1rem" }}>
        <article style={cardStyle}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", alignItems: "flex-start", flexWrap: "wrap" }}>
            <div>
              <h2 style={{ marginTop: 0 }}>2. Schedule and courts</h2>
              <p style={{ color: "#475569", marginBottom: 0 }}>
                Create every tournament day before Events and Divisions. Events can span multiple days, and each division can inherit all event days or use a selected subset.
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
                const previousReferences = new Set([
                  dayReference(row.value),
                  dayLabel(row.value)
                ].filter(Boolean));
                const nextReference = dayReference(value) || dayLabel(value);
                const replaceReferences = (record: SetupRecord) => {
                  const refs = eventDayReferences(record).map((reference) =>
                    previousReferences.has(reference) ? nextReference : reference
                  );
                  return setEventDayReferences(record, refs);
                };
                return {
                  ...current,
                  days: replaceBuilderRow(current.days, row.key, value),
                  eventFamilies: current.eventFamilies.map((family) => ({
                    ...family,
                    value: replaceReferences(family.value)
                  })),
                  eventOptions: current.eventOptions.map((division) => ({
                    ...division,
                    value: replaceReferences(division.value)
                  }))
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
              ].filter(Boolean));
              const attachedEvent = configuration.eventFamilies.find((event) =>
                eventDayReferences(event.value).some((reference) => references.has(reference))
              );
              const attachedDivision = configuration.eventOptions.find((event) =>
                eventDayReferences(event.value).some((reference) => references.has(reference))
              );
              if (attachedEvent || attachedDivision) {
                setMessage("Remove this day from its events and divisions before deleting it.");
                return;
              }
              setConfiguration((current) => ({
                ...current,
                days: removeBuilderRow(current.days, row.key)
              }));
            }}
          />
        ))}

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
              href={tournamentSetupStepHref("basics", tournamentId, basics.name || tournamentName)}
              style={ghostButtonStyle}
            >
              Back
            </Link>
            <button
              type="button"
              style={buttonStyle}
              disabled={busy || !configuration.days.length || dayIssues.length > 0}
              onClick={() => void saveDraftAndContinue("events")}
            >
              {busy ? "Saving…" : "Save and continue"}
            </button>
          </>
        )}
      </div>
    );
  }

  function renderReview() {
    const impact = impactReview?.publish_impact || {};
    const blocked = Array.isArray(impact.blocked) ? impact.blocked : [];
    const warnings = Array.isArray(impact.warnings) ? impact.warnings : [];
    const basicsReady = states.basics === "complete";
    const scheduleReady = states.schedule === "complete";
    const eventFamiliesReady = states.events === "complete";
    const divisionsReady = states.divisions === "complete";
    const ready =
      basicsReady &&
      scheduleReady &&
      eventFamiliesReady &&
      divisionsReady &&
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
              ["Tournament basics and policies", basicsReady, `${basics.startDate || "No start"} – ${basics.endDate || "No end"} · registration ${registrationStatus}`],
              ["Schedule and courts", scheduleReady, `${configuration.days.length} tournament day(s)`],
              ["Events", eventFamiliesReady, `${configuration.eventFamilies.length} event(s)`],
              ["Divisions", divisionsReady, `${configuration.eventOptions.length} division(s)`],
              ["Pricing and extras", true, "Review the saved catalog from Step 5"]
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
                "pricing",
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
        ) : step === "schedule" ? (
          renderSchedule()
        ) : step === "events" ? (
          renderEvents()
        ) : step === "divisions" ? (
          renderDivisions()
        ) : step === "pricing" ? (
          renderPricing()
        ) : (
          renderReview()
        )
      ) : null}
    </div>
  );
}
