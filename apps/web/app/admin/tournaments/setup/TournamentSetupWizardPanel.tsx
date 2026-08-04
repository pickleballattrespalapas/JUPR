"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useMemo, useRef, useState, type ReactNode } from "react";
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
  eventDayReference,
  eventDayReferences,
  eventDivisionName,
  eventFamilyName,
  eventUsesLabelDayReference,
  MAX_TOURNAMENT_DAYS,
  issuesForPath,
  newDayRow,
  newEventFamilyRow,
  newEventOptionRow,
  publishConfigurationPayload,
  recordBoolean,
  removeBuilderRow,
  replaceBuilderRow,
  setEventDayReferences,
  setRecordString,
  sortDivisionsByEventAndName,
  sortEventFamiliesByTournamentDay,
  syncTournamentDays,
  validateSetupConfiguration,
  withDefaultDayCourts,
  wrapBuilderRows,
  type SetupConfiguration,
  type SetupRecord,
  type ValidationIssue
} from "../../tournament-setup/tournamentSetupBuilder";
import TournamentSetupEventFamilyCard from "./TournamentSetupEventFamilyCard";
import TournamentSetupEventFamilyDialog from "./TournamentSetupEventFamilyDialog";
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
  resolveDivisionId?: string;
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

function basicsDraftPayload(basics: BasicsDraft): Record<string, unknown> {
  return {
    name: basics.name.trim(),
    start_date: basics.startDate || null,
    end_date: basics.endDate || null,
    location_name: basics.locationName.trim(),
    timezone: basics.timezone,
    sponsors_json: basics.sponsors.map((sponsor) => ({
      id: sponsor.id,
      name: sponsor.name.trim(),
      level: sponsor.level.trim(),
      website: sponsor.website.trim(),
      notes: sponsor.notes.trim()
    }))
  };
}

function settingsDraftPayload(settings: Record<string, unknown>): Record<string, unknown> {
  return {
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
    sponsor_markdown: safeString(settings.sponsor_markdown)
  };
}

function fullDraftSignature(
  basics: BasicsDraft,
  settings: Record<string, unknown>,
  configuration: SetupConfiguration
): string {
  return JSON.stringify({
    basics: basicsDraftPayload(basics),
    settings: settingsDraftPayload(settings),
    configuration: configurationPayload(configuration)
  });
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
  while (cursor <= endDate && rows.length < 31) {
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

type BlockedImpactDetail = {
  message: string;
  entity_type?: string;
  entity_id?: string;
  entity_label?: string;
  step?: TournamentSetupStep;
};

function blockedImpactDetail(value: unknown): BlockedImpactDetail {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const row = value as Record<string, unknown>;
    return {
      message: formatImpactItem(row),
      entity_type: safeString(row.entity_type),
      entity_id: safeString(row.entity_id),
      entity_label: safeString(row.entity_label),
      step: safeString(row.step) as TournamentSetupStep
    };
  }
  const message = formatImpactItem(value);
  const labelMatch = message.match(/division '([^']+)'/i);
  return {
    message,
    entity_type: labelMatch ? "division" : "",
    entity_label: labelMatch?.[1] || "",
    step: labelMatch ? "divisions" : "review"
  };
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
  step,
  resolveDivisionId = ""
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
  const [eventDialogKey, setEventDialogKey] = useState<string | null | undefined>(undefined);
  const [divisionDialogKey, setDivisionDialogKey] = useState<string | null | undefined>(undefined);
  const [publishedBasics, setPublishedBasics] = useState<BasicsDraft>(() => emptyBasics(tournamentName));
  const [publishedSettings, setPublishedSettings] = useState<Record<string, unknown>>({});
  const [publishedConfiguration, setPublishedConfiguration] = useState<SetupConfiguration>(emptyConfiguration);
  const openedResolutionRef = useRef(false);

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
    setEventDialogKey(undefined);
    setDivisionDialogKey(undefined);
    setPublishedBasics(emptyBasics(tournamentName));
    setPublishedSettings({});
    setPublishedConfiguration(emptyConfiguration);
    openedResolutionRef.current = false;
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
    const publishedSettingsValue = withDefaultTournamentPolicies(payload.settings || {});
    const publishedBasicsValue: BasicsDraft = {
      name: safeString(payload.tournament.name) || tournamentName,
      startDate: dateValue(payload.tournament.start_date),
      endDate: dateValue(payload.tournament.end_date),
      locationName: safeString(publishedSettingsValue.location_name),
      timezone: safeString(publishedSettingsValue.timezone) || "America/Mazatlan",
      sponsors: normalizeSponsors(publishedSettingsValue.sponsors_json)
    };
    const draftBasicsRecord = objectValue(draft.basics);
    const draftBasicsValue: BasicsDraft = {
      name: safeString(draftBasicsRecord.name) || publishedBasicsValue.name,
      startDate: dateValue(draftBasicsRecord.start_date) || publishedBasicsValue.startDate,
      endDate: dateValue(draftBasicsRecord.end_date) || publishedBasicsValue.endDate,
      locationName: safeString(draftBasicsRecord.location_name) || publishedBasicsValue.locationName,
      timezone: safeString(draftBasicsRecord.timezone) || publishedBasicsValue.timezone,
      sponsors: Array.isArray(draftBasicsRecord.sponsors_json)
        ? normalizeSponsors(draftBasicsRecord.sponsors_json)
        : publishedBasicsValue.sponsors
    };
    const draftSettingsValue = withDefaultTournamentPolicies({
      ...publishedSettingsValue,
      ...objectValue(draft.settings)
    });

    const publishedDays = (payload.days || []).map(withDefaultDayCourts);
    const rawDraftDays = listValue(draft.days).map(withDefaultDayCourts);
    const baseDays = rawDraftDays.length
      ? rawDraftDays
      : publishedDays.length
        ? publishedDays
        : initialDaysFromTournament({
            start_date: draftBasicsValue.startDate,
            end_date: draftBasicsValue.endDate
          });
    const syncedDays = syncTournamentDays(
      draftBasicsValue.startDate,
      draftBasicsValue.endDate,
      wrapBuilderRows(baseDays, "day")
    );

    const draftEvents = listValue(
      Array.isArray(draft.event_options) ? draft.event_options : draft.divisions
    );
    const events = draftEvents.length ? draftEvents : payload.event_options || [];
    const draftFamilies = listValue(draft.event_families);
    const families = draftFamilies.length
      ? draftFamilies
      : derivedEventFamilies(events, syncedDays.map((row) => row.value));
    const wrappedFamilies = sortEventFamiliesByTournamentDay(
      wrapBuilderRows(families, "family"),
      syncedDays
    );
    const wrappedEvents = sortDivisionsByEventAndName(
      wrapBuilderRows(events, "event"),
      wrappedFamilies,
      syncedDays
    );

    const publishedDayRows = wrapBuilderRows(
      publishedDays.length
        ? publishedDays
        : initialDaysFromTournament(payload.tournament).map(withDefaultDayCourts),
      "published-day"
    );
    const publishedFamilyRows = sortEventFamiliesByTournamentDay(
      wrapBuilderRows(derivedEventFamilies(payload.event_options || [], publishedDayRows.map((row) => row.value)), "published-family"),
      publishedDayRows
    );
    const publishedEventRows = sortDivisionsByEventAndName(
      wrapBuilderRows(payload.event_options || [], "published-event"),
      publishedFamilyRows,
      publishedDayRows
    );

    setDetail(payload);
    setBasics(draftBasicsValue);
    setSettings(draftSettingsValue);
    setConfiguration({ days: syncedDays, eventFamilies: wrappedFamilies, eventOptions: wrappedEvents });
    setPublishedBasics(publishedBasicsValue);
    setPublishedSettings(publishedSettingsValue);
    setPublishedConfiguration({
      days: publishedDayRows,
      eventFamilies: publishedFamilyRows,
      eventOptions: publishedEventRows
    });
    setImpactReview(null);
    setReviewedDraftSignature("");
    if (
      step === "divisions" &&
      resolveDivisionId &&
      !openedResolutionRef.current
    ) {
      const resolutionRow = wrappedEvents.find(
        (row) => safeString(row.value.id) === resolveDivisionId
      );
      if (resolutionRow) {
        openedResolutionRef.current = true;
        setDivisionDialogKey(resolutionRow.key);
        setMessage(
          `Opened ${eventDivisionName(
            resolutionRow.value
          ) || "the affected division"} to resolve the blocked draft change.`
        );
      }
    }
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

  function updateTournamentDate(field: "startDate" | "endDate", value: string) {
    const nextBasics = { ...basics, [field]: value };
    setBasics(nextBasics);
    if (nextBasics.startDate && nextBasics.endDate && nextBasics.endDate >= nextBasics.startDate) {
      setConfiguration((current) => ({
        ...current,
        days: syncTournamentDays(nextBasics.startDate, nextBasics.endDate, current.days)
      }));
    }
    setImpactReview(null);
  }

  function moveSponsor(index: number, direction: -1 | 1) {
    setBasics((current) => {
      const nextIndex = index + direction;
      if (nextIndex < 0 || nextIndex >= current.sponsors.length) return current;
      const sponsors = [...current.sponsors];
      const [moved] = sponsors.splice(index, 1);
      sponsors.splice(nextIndex, 0, moved);
      return { ...current, sponsors };
    });
    setImpactReview(null);
  }

  function saveEventDialog(value: SetupRecord) {
    setConfiguration((current) => {
      const existing =
        typeof eventDialogKey === "string"
          ? current.eventFamilies.find((row) => row.key === eventDialogKey)
          : undefined;
      const previousName = existing ? eventFamilyName(existing.value) : "";
      const nextName = eventFamilyName(value);
      const nextDays = eventDayReferences(value);
      const eventFamilies = sortEventFamiliesByTournamentDay(
        existing
          ? replaceBuilderRow(current.eventFamilies, existing.key, value)
          : appendBuilderRow(current.eventFamilies, "family", value),
        current.days
      );
      const eventOptions = sortDivisionsByEventAndName(
        current.eventOptions.map((division) => {
          if (!existing || eventFamilyName(division.value).toLowerCase() !== previousName.toLowerCase()) {
            return division;
          }
          let nextValue: SetupRecord = {
            ...division.value,
            event_family_label: nextName,
            event_family: nextName,
            participant_type: value.participant_type,
            event_type: value.participant_type,
            gender_restriction:
              safeString(value.participant_type) === "MIXED_DOUBLES"
                ? "MIXED"
                : value.gender_restriction
          };
          if (safeString(division.value.schedule_mode || "INHERIT_EVENT") !== "CUSTOM") {
            nextValue = setEventDayReferences(
              { ...nextValue, schedule_mode: "INHERIT_EVENT" },
              nextDays
            );
          }
          return { ...division, value: nextValue };
        }),
        eventFamilies,
        current.days
      );
      return { ...current, eventFamilies, eventOptions };
    });
    setEventDialogKey(undefined);
    setImpactReview(null);
    setMessage(`Event ${eventFamilyName(value) || "saved"} saved to the unpublished setup draft.`);
  }

  function saveDivisionDialog(value: SetupRecord) {
    setConfiguration((current) => {
      const existing =
        typeof divisionDialogKey === "string"
          ? current.eventOptions.find((row) => row.key === divisionDialogKey)
          : undefined;
      const eventOptions = sortDivisionsByEventAndName(
        existing
          ? replaceBuilderRow(current.eventOptions, existing.key, value)
          : appendBuilderRow(current.eventOptions, "event", value),
        current.eventFamilies,
        current.days
      );
      return { ...current, eventOptions };
    });
    setDivisionDialogKey(undefined);
    setImpactReview(null);
    setMessage(`Division ${eventDivisionName(value) || "saved"} saved to the unpublished setup draft.`);
  }

  function keepPublishedValueForBlockedChange(raw: unknown) {
    const item = blockedImpactDetail(raw);
    if (item.entity_type === "division" || item.entity_label) {
      const published = publishedConfiguration.eventOptions.find((row) => {
        if (item.entity_id && safeString(row.value.id) === item.entity_id) return true;
        return item.entity_label && eventDivisionName(row.value).toLowerCase() === item.entity_label.toLowerCase();
      });
      if (!published) {
        setMessage("The published division could not be matched. Open Divisions and revert the affected fields manually.");
        return;
      }
      setConfiguration((current) => {
        const currentRow = current.eventOptions.find((row) => {
          if (item.entity_id && safeString(row.value.id) === item.entity_id) return true;
          return item.entity_label && eventDivisionName(row.value).toLowerCase() === item.entity_label.toLowerCase();
        });
        if (!currentRow) return current;
        return {
          ...current,
          eventOptions: sortDivisionsByEventAndName(
            replaceBuilderRow(current.eventOptions, currentRow.key, published.value),
            current.eventFamilies,
            current.days
          )
        };
      });
      setImpactReview(null);
      setReviewedDraftSignature("");
      setMessage(`Reverted ${item.entity_label || "the affected division"} to its published value. Save the draft and review impact again.`);
      return;
    }
    setMessage("Open the affected setup step and restore the published value before reviewing again.");
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
    const start = new Date(`${basics.startDate}T00:00:00Z`);
    const end = new Date(`${basics.endDate}T00:00:00Z`);
    const span = Math.floor((end.valueOf() - start.valueOf()) / 86_400_000) + 1;
    if (!Number.isFinite(span) || span < 1 || span > MAX_TOURNAMENT_DAYS) {
      setMessage(
        `Tournament Setup supports between 1 and ${MAX_TOURNAMENT_DAYS} consecutive tournament days.`
      );
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

    const nextConfiguration: SetupConfiguration = {
      ...configuration,
      days: syncTournamentDays(basics.startDate, basics.endDate, configuration.days)
    };
    const normalized = configurationWithGlobalStatus(
      nextConfiguration,
      settings.registration_status
    );
    const draft = configurationPayload(normalized);
    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      await requestJson<WriteResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/setup/tournaments/${encodeURIComponent(tournamentId)}/draft`,
        {
          method: "PUT",
          body: JSON.stringify({
            ...draft,
            basics: basicsDraftPayload(basics),
            settings: settingsDraftPayload(settings),
            saved_step: "basics",
            expected_state_fingerprint: detail.state_fingerprint,
            confirmation_text: draftConfirmation
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) return;
      setConfiguration(nextConfiguration);
      await loadDetail();
      if (!actionRequest.isCurrent(generation)) return;
      setMessage("Unpublished setup draft saved. Nothing public changed.");
      goTo("schedule");
    } catch (error) {
      if (actionRequest.isCurrent(generation)) {
        setMessage(error instanceof Error ? error.message : "Unable to save tournament basics and policies draft.");
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
          basics: basicsDraftPayload(basics),
          settings: settingsDraftPayload(settings),
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
            basics: basicsDraftPayload(basics),
            settings: settingsDraftPayload(settings),
            expected_state_fingerprint: detail.state_fingerprint
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) return;
      setImpactReview(payload);
      setReviewedDraftSignature(fullDraftSignature(basics, settings, configuration));
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
      reviewedDraftSignature !== fullDraftSignature(basics, settings, configuration)
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
            basics: basicsDraftPayload(basics),
            settings: settingsDraftPayload(settings),
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
  const currentDraftSignature = fullDraftSignature(basics, settings, configuration);
  const publishedSignature = fullDraftSignature(
    publishedBasics,
    publishedSettings,
    publishedConfiguration
  );
  const hasUnpublishedChanges = Boolean(detail && currentDraftSignature !== publishedSignature);
  const publishedSetupState = setupState(
    publishedBasics,
    publishedSettings,
    publishedConfiguration
  );
  const publishedSetupReady = publishedSetupState.review === "in-progress";
  const registrationCanOpen = Boolean(
    detail &&
      !hasUnpublishedChanges &&
      publishedSetupReady
  );
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
            location, timezone, and sponsors. Save draft and continue stores an
            unpublished draft and moves directly to Schedule and courts. Nothing
            becomes public until the reviewed setup is published in Step 6.
          </p>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))",
              gap: "0.75rem"
            }}
          >
            <label><strong>Tournament name</strong><br /><input value={basics.name} onChange={(event) => setBasics((current) => ({ ...current, name: event.target.value }))} disabled={busy} style={inputStyle} /></label>
            <label><strong>Start date</strong><br /><input type="date" value={basics.startDate} onChange={(event) => updateTournamentDate("startDate", event.target.value)} disabled={busy} style={inputStyle} /></label>
            <label><strong>End date</strong><br /><input type="date" min={basics.startDate || undefined} value={basics.endDate} onChange={(event) => updateTournamentDate("endDate", event.target.value)} disabled={busy} style={inputStyle} /></label>
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
                <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginTop: "0.65rem" }}>
                  <button type="button" style={ghostButtonStyle} disabled={busy || index === 0} onClick={() => moveSponsor(index, -1)}>Move up</button>
                  <button type="button" style={ghostButtonStyle} disabled={busy || index === basics.sponsors.length - 1} onClick={() => moveSponsor(index, 1)}>Move down</button>
                  <button type="button" style={{ ...ghostButtonStyle, color: "#991b1b", borderColor: "#fecaca" }} disabled={busy} onClick={() => setBasics((current) => ({ ...current, sponsors: current.sponsors.filter((row) => row.id !== sponsor.id) }))}>Remove sponsor {index + 1}</button>
                </div>
              </article>
            ))}
            {!basics.sponsors.length ? <p style={{ color: "#64748b" }}>No sponsors added yet.</p> : null}
          </div>
        </article>

        <div>
          <button type="button" style={buttonStyle} disabled={busy} onClick={() => void saveBasics()}>{busy ? "Saving draft…" : "Save draft and continue"}</button>
        </div>
      </div>
    );
  }

function renderEvents() {
  const familyIssues = issues.filter((issue) => issue.path.startsWith("families"));
  const firstDay =
    configuration.days.find((row) => recordBoolean(row.value.enabled, true))?.value ||
    configuration.days[0]?.value ||
    {};
  const dialogValue =
    eventDialogKey === null
      ? setEventDayReferences(
          newEventFamilyRow(configuration.eventFamilies.length + 1),
          [dayReference(firstDay)].filter(Boolean)
        )
      : typeof eventDialogKey === "string"
        ? configuration.eventFamilies.find((row) => row.key === eventDialogKey)?.value || {}
        : {};
  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <TournamentSetupEventFamilyDialog
        open={eventDialogKey !== undefined}
        mode={eventDialogKey === null ? "add" : "edit"}
        initialValue={dialogValue}
        days={configuration.days}
        onCancel={() => setEventDialogKey(undefined)}
        onConfirm={saveEventDialog}
      />
      <article style={cardStyle}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", alignItems: "flex-start", flexWrap: "wrap" }}>
          <div>
            <h2 style={{ marginTop: 0 }}>3. Events</h2>
            <p style={{ color: "#475569", marginBottom: 0 }}>
              Create parent events in a focused dialog. Saved events appear as compact summaries and are sorted automatically by tournament day. Skill and age divisions are created separately in Step 4.
            </p>
          </div>
          <button
            type="button"
            style={buttonStyle}
            disabled={busy || !configuration.days.length}
            onClick={() => setEventDialogKey(null)}
          >
            Add event
          </button>
        </div>
      </article>

      {sortEventFamiliesByTournamentDay(configuration.eventFamilies, configuration.days).map((row, index) => {
        const family = eventFamilyName(row.value);
        const divisions = configuration.eventOptions.filter(
          (division) => eventFamilyName(division.value).toLowerCase() === family.toLowerCase()
        );
        const originalIndex = configuration.eventFamilies.findIndex((candidate) => candidate.key === row.key);
        return (
          <TournamentSetupEventFamilyCard
            key={row.key}
            row={row}
            position={index}
            days={configuration.days}
            disabled={busy}
            issues={issuesForPath(issues, `families.${Math.max(0, originalIndex)}`)}
            divisionCount={divisions.length}
            onEdit={() => setEventDialogKey(row.key)}
            onRemove={() => {
              setConfiguration((current) => ({
                ...current,
                eventFamilies: sortEventFamiliesByTournamentDay(
                  removeBuilderRow(current.eventFamilies, row.key),
                  current.days
                )
              }));
              setImpactReview(null);
            }}
          />
        );
      })}

      {!configuration.eventFamilies.length ? (
        <article style={cardStyle}>
          <p style={{ margin: 0, color: "#64748b" }}>No events yet. Click Add event to open the setup dialog.</p>
        </article>
      ) : null}

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
            {busy ? "Saving draft…" : "Save draft and continue"}
          </button>
        </>
      )}
    </div>
  );
}

function renderDivisions() {
  const divisionIssues = issues.filter((issue) => issue.path.startsWith("events"));
  const dialogValue =
    divisionDialogKey === null
      ? divisionForFirstEvent(configuration)
      : typeof divisionDialogKey === "string"
        ? configuration.eventOptions.find((row) => row.key === divisionDialogKey)?.value || {}
        : {};
  const sortedDivisions = sortDivisionsByEventAndName(
    configuration.eventOptions,
    configuration.eventFamilies,
    configuration.days
  );
  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <TournamentSetupDivisionDialog
        open={divisionDialogKey !== undefined}
        mode={divisionDialogKey === null ? "add" : "edit"}
        initialValue={dialogValue}
        eventFamilies={configuration.eventFamilies}
        days={configuration.days}
        onCancel={() => setDivisionDialogKey(undefined)}
        onConfirm={saveDivisionDialog}
      />

      <article style={cardStyle}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", alignItems: "flex-start", flexWrap: "wrap" }}>
          <div>
            <h2 style={{ marginTop: 0 }}>4. Divisions</h2>
            <p style={{ color: "#475569", marginBottom: 0 }}>
              Create skill and age divisions inside each event. Add and Edit open a focused dialog; saved divisions remain compact and read-only. A division may use every parent-event day or a selected subset.
            </p>
          </div>
          <button
            type="button"
            style={buttonStyle}
            disabled={busy || !configuration.eventFamilies.length}
            onClick={() => setDivisionDialogKey(null)}
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

      {sortedDivisions.map((row, index) => {
        const originalIndex = configuration.eventOptions.findIndex((candidate) => candidate.key === row.key);
        return (
          <div key={row.key} id={`division-${safeString(row.value.id)}`}>
            <TournamentSetupDivisionCard
              row={row}
              position={index}
              eventFamilies={configuration.eventFamilies}
              days={configuration.days}
              disabled={busy}
              issues={issuesForPath(issues, `events.${Math.max(0, originalIndex)}`)}
              onEdit={() => setDivisionDialogKey(row.key)}
              onRemove={() => {
                setConfiguration((current) => ({
                  ...current,
                  eventOptions: sortDivisionsByEventAndName(
                    removeBuilderRow(current.eventOptions, row.key),
                    current.eventFamilies,
                    current.days
                  )
                }));
                setImpactReview(null);
              }}
            />
          </div>
        );
      })}

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
            {busy ? "Saving draft…" : "Save draft and continue"}
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
          <h2 style={{ marginTop: 0 }}>2. Schedule and courts</h2>
          <p style={{ color: "#475569", marginBottom: 0 }}>
            Tournament days are generated automatically from the start and end dates saved in Step 1. Dates and chronological order are fixed here; edit the day label, available courts, court names, hours, and notes. Return to Step 1 to change the date range.
          </p>
        </article>

        {configuration.days.map((row, index) => (
          <TournamentSetupDayCard
            key={row.key}
            row={row}
            position={index}
            total={configuration.days.length}
            disabled={busy}
            structureLocked
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
              {busy ? "Saving draft…" : "Save draft and continue"}
            </button>
          </>
        )}
      </div>
    );
  }

  function renderReview() {
    const impact = impactReview?.publish_impact || {};
    const blocked = Array.isArray(impact.blocked) ? impact.blocked : [];
    const blockedDetails = Array.isArray(impact.blocked_details) && impact.blocked_details.length
      ? impact.blocked_details
      : blocked;
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
            Review the unpublished draft against the currently published tournament.
            Setup changes saved in Steps 1–4 remain private. Step 5 uses a separate
            reviewed catalog save. The three actions below are deliberate: review
            setup changes, publish setup, then open registration.
          </p>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))",
              gap: "0.75rem"
            }}
          >
            {[
              { key: "basics" as TournamentSetupStep, label: "Tournament basics and policies", complete: basicsReady, draft: `${basics.startDate || "No start"} – ${basics.endDate || "No end"}`, published: `${publishedBasics.startDate || "No start"} – ${publishedBasics.endDate || "No end"}` },
              { key: "schedule" as TournamentSetupStep, label: "Schedule and courts", complete: scheduleReady, draft: `${configuration.days.length} day(s) · ${configuration.days.reduce((total, day) => total + Number(day.value.court_count || 0), 0)} daily-court slots`, published: `${publishedConfiguration.days.length} day(s)` },
              { key: "events" as TournamentSetupStep, label: "Events", complete: eventFamiliesReady, draft: `${configuration.eventFamilies.length} event(s)`, published: `${publishedConfiguration.eventFamilies.length} event(s)` },
              { key: "divisions" as TournamentSetupStep, label: "Divisions", complete: divisionsReady, draft: `${configuration.eventOptions.length} division(s)`, published: `${publishedConfiguration.eventOptions.length} division(s)` },
              { key: "pricing" as TournamentSetupStep, label: "Pricing and extras", complete: true, draft: "Saved catalog draft", published: "Published catalog remains active until saved changes are applied" }
            ].map((item) => (
              <Link
                key={item.key}
                href={tournamentSetupStepHref(item.key, tournamentId, basics.name || tournamentName)}
                style={{
                  padding: "0.75rem",
                  border: `1px solid ${item.complete ? "#bbf7d0" : "#fecaca"}`,
                  borderRadius: "12px",
                  background: item.complete ? "#f0fdf4" : "#fef2f2",
                  color: "#0f172a",
                  textDecoration: "none"
                }}
              >
                <strong>{item.complete ? "✓" : "!"} {item.label}</strong>
                <br />
                <small><strong>Draft:</strong> {item.draft}</small>
                <br />
                <small><strong>Published:</strong> {item.published}</small>
              </Link>
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
              {blockedDetails.length ? (
                <>
                  <strong>Blocked changes — resolve each before publishing</strong>
                  <div style={{ display: "grid", gap: "0.65rem", marginTop: "0.55rem" }}>
                    {blockedDetails.map((raw, index) => {
                      const item = blockedImpactDetail(raw);
                      const editHref = item.step === "divisions"
                        ? `${tournamentSetupStepHref("divisions", tournamentId, basics.name || tournamentName)}&resolveDivision=${encodeURIComponent(item.entity_id || "")}`
                        : tournamentSetupStepHref(item.step || "review", tournamentId, basics.name || tournamentName);
                      return (
                        <article key={`${item.message}-${index}`} style={{ padding: "0.75rem", border: "1px solid #fecaca", borderRadius: "12px", background: "#fef2f2" }}>
                          <p style={{ marginTop: 0 }}>{item.message}</p>
                          <div style={{ display: "flex", gap: "0.55rem", flexWrap: "wrap" }}>
                            <button type="button" style={ghostButtonStyle} onClick={() => keepPublishedValueForBlockedChange(raw)}>
                              Keep published value
                            </button>
                            <Link href={editHref} style={ghostButtonStyle}>
                              Edit affected draft
                            </Link>
                          </div>
                        </article>
                      );
                    })}
                  </div>
                </>
              ) : null}
            </div>
          ) : null}
        </article>

        <article style={cardStyle}>
          <h3 style={{ marginTop: 0 }}>B. Publish tournament setup</h3>
          <p style={{ color: "#475569" }}>
            Publish the reviewed tournament identity, policies, schedule/courts,
            events, and divisions in one guarded operation. Existing registrations
            are protected by the impact review.
          </p>
          <ConfirmAction
            triggerLabel={busy ? "Publishing…" : "Publish reviewed setup"}
            title="Publish this reviewed tournament setup?"
            description="Apply the exact reviewed draft—tournament basics, policies, courts, events, and divisions—to the published tournament. Registration status remains a separate action."
            confirmLabel="Yes, publish setup"
            confirmationText={publishConfirmation}
            disabled={
              !impactReview ||
              reviewedDraftSignature !== fullDraftSignature(basics, settings, configuration) ||
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
              disabled={!(setupPublishedThisSession || registrationCanOpen)}
              busy={busy}
              onConfirm={openRegistration}
            />
          )}
          {!setupPublishedThisSession &&
          !registrationCanOpen &&
          registrationStatus.toLowerCase() !== "open" ? (
            <p style={{ color: "#64748b" }}>
              Resolve unpublished changes and publish a complete setup before
              opening registration.
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
      <article
        role="status"
        style={{
          ...cardStyle,
          background: hasUnpublishedChanges ? "#fffbeb" : "#f0fdf4",
          borderColor: hasUnpublishedChanges ? "#fde68a" : "#bbf7d0"
        }}
      >
        <strong>{hasUnpublishedChanges ? "Unpublished setup draft" : "Published setup is current"}</strong>
        <p style={{ margin: "0.35rem 0 0", color: "#475569" }}>
          {hasUnpublishedChanges
            ? "Save draft and continue on Steps 1–4 only preserves this private admin draft. Public tournament pages continue using the currently published setup until Publish reviewed setup succeeds in Step 6. The extras catalog in Step 5 has its own separate Review and Save action."
            : "No unpublished setup changes are waiting. New setup edits remain private until final review and publication; extras catalog changes use their separate reviewed save in Step 5."}
        </p>
      </article>
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
