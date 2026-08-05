"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import TournamentSetupWizardNav, {
  TOURNAMENT_SETUP_STEPS,
  TOURNAMENT_SETUP_DOMAINS,
  tournamentSetupStepHref,
  type TournamentSetupStep,
  tournamentSetupDomainForStep,
  type TournamentSetupStepState
} from "@/components/TournamentSetupWizardNav";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { useAdminSession } from "@/lib/useAdminSession";
import TournamentCommercePanel from "../commerce/TournamentCommercePanel";
import {
  appendBuilderRow,
  configurationPayload,
  dayLabel,
  dayReference,
  eventDayReference,
  eventDayReferences,
  eventDivisionName,
  eventFamilyAgeLabel,
  eventFamilyAgeMode,
  eventFamilyAgeRules,
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
import {
  EVENT_AGE_POLICY_FIELDS,
  agePolicySummary,
  readAgePolicy
} from "./TournamentAgePolicyEditor";

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

type AgeSplitPreviewResponse = {
  ok: boolean;
  dry_run: true;
  write_count: 0;
  event_family: string;
  policy: Record<string, unknown>;
  total_entries: number;
  brackets: Array<{
    id: string;
    label: string;
    count: number;
    viable: boolean;
    entries: Array<{ registration_id: string; selection_id?: string | null; display_name: string; age?: number | null; partner_age?: number | null; effective_age?: number | null }>;
  }>;
  recommendations: string[];
  unassigned_entries: Array<Record<string, unknown>>;
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
    sponsor_markdown: safeString(settings.sponsor_markdown),
    venue_court_count: Number(settings.venue_court_count) || 10,
    venue_court_labels: Array.isArray(settings.venue_court_labels)
      ? settings.venue_court_labels.map((value) => safeString(value).trim()).filter(Boolean)
      : [],
    forced_change_resolutions: objectValue(settings.forced_change_resolutions)
  };
}

function venueCourtCount(settings: Record<string, unknown>, configuration?: SetupConfiguration): number {
  const configured = Number(settings.venue_court_count);
  if (Number.isInteger(configured) && configured >= 1 && configured <= 100) return configured;
  const inherited = Number(configuration?.days[0]?.value.court_count);
  return Number.isInteger(inherited) && inherited >= 1 && inherited <= 100 ? inherited : 10;
}

function venueCourtLabels(settings: Record<string, unknown>, configuration?: SetupConfiguration): string[] {
  const configured = Array.isArray(settings.venue_court_labels)
    ? settings.venue_court_labels.map((value) => safeString(value).trim()).filter(Boolean)
    : [];
  if (configured.length) return configured;
  const inherited = Array.isArray(configuration?.days[0]?.value.court_labels)
    ? (configuration?.days[0]?.value.court_labels as unknown[]).map((value) => safeString(value).trim()).filter(Boolean)
    : [];
  return inherited;
}

function configurationWithVenue(
  configuration: SetupConfiguration,
  settings: Record<string, unknown>
): SetupConfiguration {
  const courtCount = venueCourtCount(settings, configuration);
  const labels = venueCourtLabels(settings, configuration).slice(0, courtCount);
  return {
    ...configuration,
    days: configuration.days.map((row) => ({
      ...row,
      value: {
        ...row.value,
        court_count: courtCount,
        court_labels: labels,
        court_open_time: null,
        court_close_time: null,
        court_notes: ""
      }
    }))
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

function impactContentSignature(
  basics: BasicsDraft,
  settings: Record<string, unknown>,
  configuration: SetupConfiguration
): string {
  const impactSettings = settingsDraftPayload(settings);
  delete impactSettings.forced_change_resolutions;
  return JSON.stringify({
    basics: basicsDraftPayload(basics),
    settings: impactSettings,
    configuration: configurationPayload(configuration)
  });
}

function comparablePublishedStateSignature(
  basics: BasicsDraft,
  settings: Record<string, unknown>,
  configuration: SetupConfiguration
): string {
  const comparableSettings = settingsDraftPayload(settings);
  delete comparableSettings.forced_change_resolutions;
  const normalized = configurationWithGlobalStatus(
    configurationWithVenue(configuration, settings),
    settings.registration_status
  );
  const publishable = publishConfigurationPayload(normalized);
  const builder = configurationPayload(normalized);
  return JSON.stringify({
    basics: basicsDraftPayload(basics),
    settings: comparableSettings,
    days: publishable.days,
    event_families: builder.event_families,
    event_options: publishable.event_options
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
          default_price_usd: Number(event.price_usd) || 0,
          competition_format: safeString(event.competition_format) || "STANDARD",
          team_roster_size: Number(event.team_roster_size) || 2,
          team_gender_rule: safeString(event.team_gender_rule) || "NONE",
          team_tiebreak_mode: safeString(event.team_tiebreak_mode) || "SINGLES",
          team_playoff_format: safeString(event.team_playoff_format) || "NONE",
          team_allow_substitutes: recordBoolean(event.team_allow_substitutes, false),
          default_age_mode: safeString(event.age_mode) || "ALL_AGES",
          default_age_label: safeString(event.age_label) || "All Ages",
          default_age_rules: event.age_rules || { mode: safeString(event.age_mode) || "ALL_AGES" }
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
      policiesComplete
  );
  const venueComplete = Boolean(
    basics.locationName.trim() &&
      basics.timezone &&
      Number.isInteger(venueCourtCount(settings, configuration)) &&
      venueCourtCount(settings, configuration) >= 1
  );
  const scheduleComplete =
    venueComplete &&
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

type AffectedRegistration = {
  registration_id: string;
  selection_id?: string;
  display_name: string;
  email?: string;
  registration_status?: string;
  current_value?: unknown;
  proposed_value?: unknown;
};

type BlockedImpactDetail = {
  block_id: string;
  message: string;
  entity_type?: string;
  entity_id?: string;
  entity_label?: string;
  field?: string;
  current_value?: unknown;
  proposed_value?: unknown;
  step?: TournamentSetupStep;
  resolution_options?: string[];
  affected_registrations: AffectedRegistration[];
};

function blockedImpactDetail(value: unknown): BlockedImpactDetail {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const row = value as Record<string, unknown>;
    return {
      block_id: safeString(row.block_id) || `${safeString(row.entity_id)}:${safeString(row.field)}`,
      message: formatImpactItem(row),
      entity_type: safeString(row.entity_type),
      entity_id: safeString(row.entity_id),
      entity_label: safeString(row.entity_label),
      field: safeString(row.field),
      current_value: row.current_value,
      proposed_value: row.proposed_value,
      step: safeString(row.step) as TournamentSetupStep,
      resolution_options: Array.isArray(row.resolution_options) ? row.resolution_options.map((value) => safeString(value)) : [],
      affected_registrations: Array.isArray(row.affected_registrations)
        ? row.affected_registrations
            .filter((value) => value && typeof value === "object" && !Array.isArray(value))
            .map((value) => {
              const affected = value as Record<string, unknown>;
              return {
                registration_id: safeString(affected.registration_id),
                selection_id: safeString(affected.selection_id),
                display_name: safeString(affected.display_name) || safeString(affected.email) || safeString(affected.registration_id),
                email: safeString(affected.email),
                registration_status: safeString(affected.registration_status),
                current_value: affected.current_value,
                proposed_value: affected.proposed_value
              };
            })
        : []
    };
  }
  const message = formatImpactItem(value);
  const labelMatch = message.match(/division '([^']+)'/i);
  return {
    block_id: `${labelMatch?.[1] || "setup"}:unknown`,
    message,
    entity_type: labelMatch ? "division" : "",
    entity_label: labelMatch?.[1] || "",
    step: labelMatch ? "divisions" : "review",
    affected_registrations: []
  };
}

function reviewValue(value: unknown): string {
  if (value == null || value === "") return "Not set";
  if (Array.isArray(value)) return value.map((row) => reviewValue(row)).join(", ") || "None";
  if (typeof value === "object") return JSON.stringify(value);
  if (typeof value === "boolean") return value ? "Yes" : "No";
  return String(value);
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
  const [ageSplitPreviews, setAgeSplitPreviews] = useState<Record<string, AgeSplitPreviewResponse>>({});
  const [ageSplitPreviewBusy, setAgeSplitPreviewBusy] = useState("");
  const [resolutionDraftDirty, setResolutionDraftDirty] = useState(false);
  const openedResolutionRef = useRef(false);
  const autoReviewSignatureRef = useRef("");

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
    setAgeSplitPreviews({});
    setAgeSplitPreviewBusy("");
    setResolutionDraftDirty(false);
    openedResolutionRef.current = false;
    autoReviewSignatureRef.current = "";
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
    const publishedVenueCount = Number(publishedSettingsValue.venue_court_count)
      || Number(publishedDays[0]?.court_count)
      || 10;
    const publishedVenueLabels = Array.isArray(publishedSettingsValue.venue_court_labels)
      ? publishedSettingsValue.venue_court_labels
      : Array.isArray(publishedDays[0]?.court_labels)
        ? publishedDays[0]?.court_labels
        : [];
    publishedSettingsValue.venue_court_count = publishedVenueCount;
    publishedSettingsValue.venue_court_labels = publishedVenueLabels;

    const rawDraftDays = listValue(draft.days).map(withDefaultDayCourts);
    const draftVenueCount = Number(draftSettingsValue.venue_court_count)
      || Number(rawDraftDays[0]?.court_count)
      || publishedVenueCount;
    const draftVenueLabels = Array.isArray(draftSettingsValue.venue_court_labels)
      ? draftSettingsValue.venue_court_labels
      : Array.isArray(rawDraftDays[0]?.court_labels)
        ? rawDraftDays[0]?.court_labels
        : publishedVenueLabels;
    draftSettingsValue.venue_court_count = draftVenueCount;
    draftSettingsValue.venue_court_labels = draftVenueLabels;
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
    ).map((row) => ({
      ...row,
      value: {
        ...row.value,
        court_count: draftVenueCount,
        court_labels: Array.isArray(draftVenueLabels)
          ? draftVenueLabels.map((value) => safeString(value).trim()).filter(Boolean).slice(0, draftVenueCount)
          : [],
        court_open_time: null,
        court_close_time: null,
        court_notes: ""
      }
    }));

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
      (publishedDays.length
        ? publishedDays
        : initialDaysFromTournament(payload.tournament).map(withDefaultDayCourts)
      ).map((row) => ({
        ...row,
        court_count: publishedVenueCount,
        court_labels: Array.isArray(publishedVenueLabels)
          ? publishedVenueLabels.map((value) => safeString(value).trim()).filter(Boolean).slice(0, publishedVenueCount)
          : [],
        court_open_time: null,
        court_close_time: null,
        court_notes: ""
      })),
      "published-day"
    );
    const publishedFamilyPayload = listValue(draft.published_event_families);
    const publishedFamilyRows = sortEventFamiliesByTournamentDay(
      wrapBuilderRows(
        publishedFamilyPayload.length
          ? publishedFamilyPayload
          : derivedEventFamilies(payload.event_options || [], publishedDayRows.map((row) => row.value)),
        "published-family"
      ),
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
    setAgeSplitPreviews({});
    setResolutionDraftDirty(false);
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

  function updateVenueSettings(patch: Record<string, unknown>) {
    setSettings((current) => {
      const next = { ...current, ...patch };
      setConfiguration((configuration) => configurationWithVenue(configuration, next));
      return next;
    });
    setImpactReview(null);
  }

  async function previewAgeSplit(value: SetupRecord) {
    const family = eventFamilyName(value);
    if (!family) return;
    const policy = readAgePolicy(value, EVENT_AGE_POLICY_FIELDS);
    setAgeSplitPreviewBusy(family);
    setMessage(null);
    try {
      const published = publishConfigurationPayload(
        configurationWithVenue(configuration, settings)
      );
      const payload = await requestJson<AgeSplitPreviewResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/setup/tournaments/${encodeURIComponent(tournamentId)}/age-split-preview`,
        {
          method: "POST",
          body: JSON.stringify({
            event_family: family,
            policy: {
              mode: policy.mode,
              label: policy.label,
              min_age: policy.min_age,
              max_age: policy.max_age,
              split_age_threshold: policy.split_age_threshold,
              min_teams_per_age_group: policy.min_teams_per_age_group,
              team_age_rule: policy.team_age_rule,
              merge_strategy: policy.merge_strategy,
              brackets: policy.brackets
            },
            event_options: published.event_options
          })
        }
      );
      setAgeSplitPreviews((current) => ({ ...current, [family]: payload }));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to preview the age split.");
    } finally {
      setAgeSplitPreviewBusy("");
    }
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
          const inheritsAge = safeString(division.value.age_policy_source).toUpperCase() !== "OVERRIDE";
          let nextValue: SetupRecord = {
            ...division.value,
            event_family_label: nextName,
            event_family: nextName,
            participant_type: value.participant_type,
            event_type: value.participant_type,
            gender_restriction:
              safeString(value.participant_type) === "MIXED_DOUBLES"
                ? "MIXED"
                : value.gender_restriction,
            competition_format: value.competition_format || "STANDARD",
            team_roster_size: value.team_roster_size ?? 2,
            team_gender_rule: value.team_gender_rule || "NONE",
            team_tiebreak_mode: value.team_tiebreak_mode || "SINGLES",
            team_playoff_format: value.team_playoff_format || "NONE",
            team_allow_substitutes: recordBoolean(value.team_allow_substitutes, false),
            event_format_default: value.default_format || division.value.event_format_default,
            scoring_default: value.default_scoring || division.value.scoring_default,
            ...(inheritsAge
              ? {
                  age_policy_source: "INHERIT_EVENT",
                  age_mode: eventFamilyAgeMode(value),
                  age_label: eventFamilyAgeLabel(value),
                  age_rules: eventFamilyAgeRules(value)
                }
              : {})
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
    if (item.entity_type !== "division" && !item.entity_label) {
      setMessage("Open the affected setup step and restore the published value before reviewing again.");
      return;
    }
    const published = publishedConfiguration.eventOptions.find((row) => {
      if (item.entity_id && safeString(row.value.id) === item.entity_id) return true;
      return item.entity_label && eventDivisionName(row.value).toLowerCase() === item.entity_label.toLowerCase();
    });
    if (!published) {
      setMessage("The published division could not be matched. Open Divisions and revert the affected fields manually.");
      return;
    }

    let reverted = false;
    setConfiguration((current) => {
      const currentRow = current.eventOptions.find((row) => {
        if (item.entity_id && safeString(row.value.id) === item.entity_id) return true;
        return item.entity_label && eventDivisionName(row.value).toLowerCase() === item.entity_label.toLowerCase();
      });
      if (!currentRow) return current;
      const publishedValue = published.value;
      let nextValue: SetupRecord = { ...currentRow.value };
      if (item.field === "registration_day_id") {
        nextValue.registration_day_id = publishedValue.registration_day_id;
      } else if (item.field === "scheduled_day_ids") {
        nextValue = setEventDayReferences(nextValue, eventDayReferences(publishedValue));
        nextValue.schedule_mode = "CUSTOM";
      } else if (item.field === "event_type") {
        nextValue.event_type = publishedValue.event_type;
        nextValue.participant_type = publishedValue.event_type;
        nextValue.competition_format = publishedValue.competition_format || "STANDARD";
      } else if (item.field === "gender_restriction") {
        nextValue.gender_restriction = publishedValue.gender_restriction;
      } else if (item.field === "skill_age_rules") {
        for (const field of ["skill_label", "skill_mode", "age_label", "age_mode", "age_rules"] as const) {
          nextValue[field] = publishedValue[field];
        }
        nextValue.age_policy_source = "OVERRIDE";
      } else if (item.field === "capacity_teams") {
        nextValue.capacity_teams = publishedValue.capacity_teams;
      } else {
        setMessage("This blocker does not yet support a field-level revert. Open Divisions and restore the published field manually.");
        return current;
      }
      reverted = true;
      return {
        ...current,
        eventOptions: sortDivisionsByEventAndName(
          replaceBuilderRow(current.eventOptions, currentRow.key, nextValue),
          current.eventFamilies,
          current.days
        )
      };
    });
    if (!reverted) return;
    setImpactReview(null);
    setReviewedDraftSignature("");
    setMessage(`Restored only the blocked ${item.field || "field"} for ${item.entity_label || "the affected division"}. Other draft changes were preserved. Save the draft and review impact again.`);
  }

  function forcedResolutionPlans(): Record<string, unknown> {
    return objectValue(settings.forced_change_resolutions);
  }

  function forcedResolutionPlan(item: BlockedImpactDetail): Record<string, unknown> | null {
    const plan = forcedResolutionPlans()[item.block_id];
    return plan && typeof plan === "object" && !Array.isArray(plan)
      ? (plan as Record<string, unknown>)
      : null;
  }

  function beginForcedResolution(raw: unknown) {
    const item = blockedImpactDetail(raw);
    if (!item.affected_registrations.length) {
      setMessage("This blocker does not yet include affected-registration details. Refresh the impact review before forcing the change.");
      return;
    }
    const plans = forcedResolutionPlans();
    const existing = forcedResolutionPlan(item);
    const registrations = item.affected_registrations.map((registration) => {
      const prior = Array.isArray(existing?.affected_registrations)
        ? (existing?.affected_registrations as Array<Record<string, unknown>>).find(
            (row) => safeString(row.registration_id) === registration.registration_id && safeString(row.selection_id) === safeString(registration.selection_id)
          )
        : null;
      return {
        ...registration,
        action: safeString(prior?.action),
        notes: safeString(prior?.notes),
        resolved: Boolean(prior?.resolved)
      };
    });
    setSettings((current) => ({
      ...current,
      forced_change_resolutions: {
        ...plans,
        [item.block_id]: {
          block_id: item.block_id,
          entity_type: item.entity_type,
          entity_id: item.entity_id,
          entity_label: item.entity_label,
          field: item.field,
          current_value: item.current_value,
          proposed_value: item.proposed_value,
          status: "IN_PROGRESS",
          affected_registrations: registrations
        }
      }
    }));
    setResolutionDraftDirty(true);
    setReviewedDraftSignature("");
    setMessage(`Created a manual resolution queue for ${item.entity_label || "the blocked change"}. Publication remains blocked until every affected registration is resolved.`);
  }

  function updateForcedRegistration(
    item: BlockedImpactDetail,
    registration: AffectedRegistration,
    patch: Record<string, unknown>
  ) {
    const plans = forcedResolutionPlans();
    const plan = forcedResolutionPlan(item);
    if (!plan) return;
    const rows = Array.isArray(plan.affected_registrations)
      ? (plan.affected_registrations as Array<Record<string, unknown>>)
      : [];
    const nextRows = rows.map((row) => {
      const match = safeString(row.registration_id) === registration.registration_id
        && safeString(row.selection_id) === safeString(registration.selection_id);
      if (!match) return row;
      const next = { ...row, ...patch };
      if (Object.prototype.hasOwnProperty.call(patch, "action") || Object.prototype.hasOwnProperty.call(patch, "notes")) {
        next.resolved = false;
      }
      if (Object.prototype.hasOwnProperty.call(patch, "resolved")) {
        next.resolved = Boolean(patch.resolved) && Boolean(safeString(next.action) && safeString(next.notes));
      }
      return next;
    });
    const complete = nextRows.length > 0 && nextRows.every((row) => Boolean(row.resolved));
    setSettings((current) => ({
      ...current,
      forced_change_resolutions: {
        ...plans,
        [item.block_id]: {
          ...plan,
          status: complete ? "RESOLVED" : "IN_PROGRESS",
          resolved_at: complete ? new Date().toISOString() : null,
          affected_registrations: nextRows
        }
      }
    }));
    setResolutionDraftDirty(true);
    setReviewedDraftSignature("");
  }

  function forcedResolutionComplete(item: BlockedImpactDetail): boolean {
    const plan = forcedResolutionPlan(item);
    if (!plan || safeString(plan.status) !== "RESOLVED") return false;
    const rows = Array.isArray(plan.affected_registrations)
      ? (plan.affected_registrations as Array<Record<string, unknown>>)
      : [];
    return rows.length > 0 && rows.every((row) => Boolean(row.resolved) && safeString(row.action) && safeString(row.notes));
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

    const nextConfiguration = configurationWithVenue({
      ...configuration,
      days: syncTournamentDays(basics.startDate, basics.endDate, configuration.days)
    }, settings);
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
  if (step === "schedule") {
    if (!basics.locationName.trim()) {
      setMessage("Venue name is required before continuing.");
      return;
    }
    if (!basics.timezone) {
      setMessage("Venue timezone is required before continuing.");
      return;
    }
    const courtCount = venueCourtCount(settings, configuration);
    if (!Number.isInteger(courtCount) || courtCount < 1 || courtCount > 100) {
      setMessage("Venue court count must be a whole number from 1 to 100.");
      return;
    }
  }
  const venueConfiguration = configurationWithVenue(configuration, settings);
  const issues = validateSetupConfiguration(venueConfiguration);
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
      venueConfiguration,
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

async function saveResolutionDraft() {
    if (!detail) return;
    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const normalized = configurationWithGlobalStatus(
        configurationWithVenue(configuration, settings),
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
            saved_step: "review",
            expected_state_fingerprint: detail.state_fingerprint,
            confirmation_text: draftConfirmation
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) return;
      setResolutionDraftDirty(false);
      autoReviewSignatureRef.current = "";
      await loadDetail();
      if (!actionRequest.isCurrent(generation)) return;
      setMessage("Conflict-resolution queue saved to the unpublished tournament draft. The impact review is refreshing against the saved queue.");
    } catch (error) {
      if (actionRequest.isCurrent(generation)) {
        setMessage(error instanceof Error ? error.message : "Unable to save the conflict-resolution queue.");
      }
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function reviewImpact() {
    if (!detail) return;
    const venueConfiguration = configurationWithVenue(configuration, settings);
    const issues = validateSetupConfiguration(venueConfiguration);
    if (issues.length) {
      setMessage(issues[0].message);
      return;
    }
    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const normalized = configurationWithGlobalStatus(
        venueConfiguration,
        settings.registration_status
      );
      const draft = publishConfigurationPayload(normalized);
      const builderDraft = configurationPayload(normalized);
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
            event_families: builderDraft.event_families,
            event_options: draft.event_options,
            builder_event_options: builderDraft.event_options,
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
          error instanceof Error ? `${error.message} Use Refresh review to try again.` : "Unable to review setup. Use Refresh review to try again."
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
    if (resolutionDraftDirty) {
      setMessage("Save the registration-resolution queue before publishing.");
      return;
    }
    const impact = impactReview.publish_impact || {};
    const rawBlockedDetails = Array.isArray(impact.blocked_details) && impact.blocked_details.length
      ? impact.blocked_details
      : (Array.isArray(impact.blocked) ? impact.blocked : []);
    const unresolved = rawBlockedDetails
      .map(blockedImpactDetail)
      .filter((item) => !forcedResolutionComplete(item));
    if (unresolved.length) {
      setMessage(`Resolve ${unresolved.length} blocked change${unresolved.length === 1 ? "" : "s"} before publishing.`);
      return;
    }
    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const normalized = configurationWithGlobalStatus(
        configurationWithVenue(configuration, settings),
        settings.registration_status
      );
      const draft = publishConfigurationPayload(normalized);
      const builderDraft = configurationPayload(normalized);
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
            event_families: builderDraft.event_families,
            event_options: draft.event_options,
            builder_event_options: builderDraft.event_options,
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
  const currentDomain = tournamentSetupDomainForStep(step);
  const domainDefinition = TOURNAMENT_SETUP_DOMAINS.find((row) => row.key === currentDomain)!;
  const domainSectionIndex = domainDefinition.steps.indexOf(step) + 1;
  const issues = validateSetupConfiguration(configuration);
  const currentDraftSignature = comparablePublishedStateSignature(
    basics,
    settings,
    configuration
  );
  const setupReady = Boolean(
    states.basics === "complete" &&
      states.schedule === "complete" &&
      states.events === "complete" &&
      states.divisions === "complete" &&
      issues.length === 0
  );
  const publishedSignature = comparablePublishedStateSignature(
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

  useEffect(() => {
    if (step !== "review" || !detail || !setupReady || busy || resolutionDraftDirty) return;
    const signature = `${detail.state_fingerprint}:${impactContentSignature(basics, settings, configuration)}`;
    if (autoReviewSignatureRef.current === signature) return;
    autoReviewSignatureRef.current = signature;
    void reviewImpact();
  }, [step, detail, setupReady, busy, resolutionDraftDirty, basics, settings, configuration]);


  function renderBasics() {
    return (
      <div style={{ display: "grid", gap: "1rem" }}>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Tournament · Basics, registration, and policies</h2>
          <p style={{ color: "#475569" }}>
            Set the tournament identity, date range, registration window, sponsors,
            and public policies. Venue and court capacity are configured in the next
            Tournament section. Saving preserves a private draft; nothing public
            changes until Review publishes the complete tournament.
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
          </div>
        </article>

        <article style={cardStyle}>
          <h3 style={{ marginTop: 0 }}>Registration and public policies</h3>
          <TournamentSetupPolicies
            settings={settings}
            registrationStatus={registrationStatus}
            disabled={busy}
            inputStyle={inputStyle}
            onChange={(next) => {
              setSettings(next);
              setImpactReview(null);
            }}
          />
        </article>

        <article style={cardStyle}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", alignItems: "center", flexWrap: "wrap" }}>
            <div><h3 style={{ margin: 0 }}>Sponsors</h3><p style={{ margin: "0.25rem 0 0", color: "#64748b" }}>Sponsors appear publicly in the order shown here.</p></div>
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
          <button type="button" style={buttonStyle} disabled={busy} onClick={() => void saveBasics()}>{busy ? "Saving draft…" : "Save draft and continue to Venue"}</button>
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
            <h2 style={{ marginTop: 0 }}>Competition · Events and event policies</h2>
            <p style={{ color: "#475569", marginBottom: 0 }}>
              Define the competition intent once: Singles, Gender Doubles, Mixed Doubles, or Four-player team, together with age policy, draw, scoring, capacity, and schedule defaults. Divisions inherit these policies unless deliberately overridden.
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
        const policy = readAgePolicy(row.value, EVENT_AGE_POLICY_FIELDS);
        const preview = ageSplitPreviews[family];
        return (
          <div key={row.key} style={{ display: "grid", gap: "0.65rem" }}>
            <TournamentSetupEventFamilyCard
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
            {policy.mode !== "ALL_AGES" ? (
              <article style={{ ...cardStyle, background: "#f8fafc" }}>
                <div style={{ display: "flex", justifyContent: "space-between", gap: "0.65rem", alignItems: "center", flexWrap: "wrap" }}>
                  <div>
                    <strong>Age-split preview</strong>
                    <p style={{ margin: "0.2rem 0 0", color: "#64748b" }}>
                      See how current registrations would be grouped before any divisions or draws change.
                    </p>
                  </div>
                  <button
                    type="button"
                    style={ghostButtonStyle}
                    disabled={ageSplitPreviewBusy === family}
                    onClick={() => void previewAgeSplit(row.value)}
                  >
                    {ageSplitPreviewBusy === family ? "Calculating…" : "Preview age split"}
                  </button>
                </div>
                {preview ? (
                  <div style={{ marginTop: "0.75rem" }}>
                    <p><strong>{preview.total_entries}</strong> current registration entr{preview.total_entries === 1 ? "y" : "ies"} evaluated. No rows were changed.</p>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.55rem" }}>
                      {preview.brackets.map((bracket) => (
                        <article key={bracket.id} style={{ border: `1px solid ${bracket.viable ? "#bbf7d0" : "#fde68a"}`, borderRadius: "10px", padding: "0.65rem", background: bracket.viable ? "#f0fdf4" : "#fffbeb" }}>
                          <strong>{bracket.label}</strong><br />
                          {bracket.count} entr{bracket.count === 1 ? "y" : "ies"} · {bracket.viable ? "Create" : "Below minimum"}
                        </article>
                      ))}
                    </div>
                    {preview.recommendations.length ? (
                      <ul>
                        {preview.recommendations.map((recommendation) => <li key={recommendation}>{recommendation}</li>)}
                      </ul>
                    ) : null}
                  </div>
                ) : null}
              </article>
            ) : null}
          </div>
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
            {busy ? "Saving draft…" : "Save draft and continue to Divisions"}
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
            <h2 style={{ marginTop: 0 }}>Competition · Divisions</h2>
            <p style={{ color: "#475569", marginBottom: 0 }}>
              Build the final competitive groups inside each event. Divisions inherit event structure, age policy, draw, and scoring by default; use explicit overrides only when a specific division needs different rules.
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
            Create an Event and event policy before adding divisions.
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
            {busy ? "Saving draft…" : "Save draft and continue to Commerce"}
          </button>
        </>
      )}
    </div>
  );
}

  function renderPricing() {
    const sortedFamilies = sortEventFamiliesByTournamentDay(configuration.eventFamilies, configuration.days);
    const sortedDivisions = sortDivisionsByEventAndName(configuration.eventOptions, configuration.eventFamilies, configuration.days);
    return (
      <div style={{ display: "grid", gap: "1rem" }}>
        <article style={{ ...cardStyle, background: "#eff6ff", borderColor: "#bfdbfe" }}>
          <h2 style={{ marginTop: 0 }}>Commerce · Fees, extras, bundles, and giveaways</h2>
          <p style={{ color: "#475569", marginBottom: 0 }}>
            Commerce is the consolidated place to review every event and division fee,
            then configure extras, options, bundles, inventory, fulfillment, and
            giveaways. Tournament fees remain part of the unpublished setup draft;
            catalog changes retain their separate reviewed save.
          </p>
        </article>

        <article style={cardStyle}>
          <h3 style={{ marginTop: 0 }}>Entry fees</h3>
          <p style={{ color: "#64748b" }}>
            Event fees provide defaults for new divisions. Division fees are the amounts registrants will see for those final competitive groups.
          </p>
          <h4>Event defaults</h4>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.65rem" }}>
            {sortedFamilies.map((row) => (
              <label key={`fee-family-${row.key}`} style={{ border: "1px solid #e2e8f0", borderRadius: "10px", padding: "0.65rem", background: "#f8fafc" }}>
                <strong>{eventFamilyName(row.value)}</strong><br />
                <span style={{ display: "flex", alignItems: "center", gap: "0.4rem", marginTop: "0.35rem" }}>
                  <span>$</span>
                  <input
                    type="number"
                    min="0"
                    step="0.01"
                    inputMode="decimal"
                    value={Number(row.value.default_price_usd || 0)}
                    disabled={busy}
                    style={inputStyle}
                    onChange={(event) => {
                      const amount = Math.max(0, Number(event.target.value) || 0);
                      setConfiguration((current) => ({
                        ...current,
                        eventFamilies: replaceBuilderRow(current.eventFamilies, row.key, { ...row.value, default_price_usd: amount })
                      }));
                      setImpactReview(null);
                    }}
                  />
                </span>
              </label>
            ))}
          </div>

          <h4 style={{ marginTop: "1rem" }}>Division registration fees</h4>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "0.65rem" }}>
            {sortedDivisions.map((row) => (
              <label key={`fee-division-${row.key}`} style={{ border: "1px solid #e2e8f0", borderRadius: "10px", padding: "0.65rem", background: "#f8fafc" }}>
                <strong>{eventDivisionName(row.value)}</strong>
                <small style={{ display: "block", color: "#64748b" }}>{eventFamilyName(row.value)}</small>
                <span style={{ display: "flex", alignItems: "center", gap: "0.4rem", marginTop: "0.35rem" }}>
                  <span>$</span>
                  <input
                    type="number"
                    min="0"
                    step="0.01"
                    inputMode="decimal"
                    value={Number(row.value.price_usd || 0)}
                    disabled={busy}
                    style={inputStyle}
                    onChange={(event) => {
                      const amount = Math.max(0, Number(event.target.value) || 0);
                      setConfiguration((current) => ({
                        ...current,
                        eventOptions: replaceBuilderRow(current.eventOptions, row.key, { ...row.value, price_usd: amount })
                      }));
                      setImpactReview(null);
                    }}
                  />
                </span>
              </label>
            ))}
          </div>
        </article>

        <TournamentCommercePanel
          clubId={clubId}
          tournamentId={tournamentId}
          tournamentName={basics.name || tournamentName}
        />
        {footerRow(
          <>
            <Link
              href={tournamentSetupStepHref("divisions", tournamentId, basics.name || tournamentName)}
              style={ghostButtonStyle}
            >
              Back to Competition
            </Link>
            <button
              type="button"
              style={buttonStyle}
              disabled={busy}
              onClick={() => void saveDraftAndContinue("review")}
            >
              {busy ? "Saving draft…" : "Save fees draft and continue to Review"}
            </button>
          </>
        )}
      </div>
    );
  }

  function renderSchedule() {
    const dayIssues = issues.filter((issue) => issue.path.startsWith("days"));
    const courtCount = venueCourtCount(settings, configuration);
    const courtLabels = venueCourtLabels(settings, configuration).slice(0, courtCount);
    const duplicateCourtTitles = new Set(
      courtLabels
        .map((label) => label.toLowerCase())
        .filter((label, index, values) => values.indexOf(label) !== index)
    );
    return (
      <div style={{ display: "grid", gap: "1rem" }}>
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Tournament · Venue and tournament days</h2>
          <p style={{ color: "#475569", marginBottom: 0 }}>
            Venue information is stored once and applies to every tournament day.
            Court count is required; court titles are optional. Event and division
            scheduling controls play start times, so tournament-level court hours
            are intentionally not collected here.
          </p>
        </article>

        <article style={cardStyle}>
          <h3 style={{ marginTop: 0 }}>Venue</h3>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "0.75rem" }}>
            <label>
              <strong>Venue name</strong><br />
              <input
                value={basics.locationName}
                placeholder="Tres Palapas Baja Pickleball Resort"
                disabled={busy}
                style={inputStyle}
                onChange={(event) => {
                  setBasics((current) => ({ ...current, locationName: event.target.value }));
                  setImpactReview(null);
                }}
              />
            </label>
            <label>
              <strong>Timezone</strong><br />
              <select
                value={basics.timezone}
                disabled={busy}
                style={inputStyle}
                onChange={(event) => {
                  setBasics((current) => ({ ...current, timezone: event.target.value }));
                  setImpactReview(null);
                }}
              >
                {TIMEZONE_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </select>
            </label>
            <label>
              <strong>Total venue courts</strong><br />
              <input
                type="number"
                min="1"
                max="100"
                step="1"
                required
                value={courtCount}
                disabled={busy}
                style={inputStyle}
                onChange={(event) => {
                  const nextCount = Math.max(1, Math.min(100, Math.trunc(Number(event.target.value) || 1)));
                  updateVenueSettings({
                    venue_court_count: nextCount,
                    venue_court_labels: courtLabels.slice(0, nextCount)
                  });
                }}
              />
              <small>This capacity applies to every tournament day.</small>
            </label>
          </div>
        </article>

        <article style={cardStyle}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", alignItems: "center", flexWrap: "wrap" }}>
            <div>
              <h3 style={{ margin: 0 }}>Optional court titles</h3>
              <p style={{ margin: "0.25rem 0 0", color: "#64748b" }}>
                Name only the courts that need public or operational labels. Unnamed courts remain available by number.
              </p>
            </div>
            <button
              type="button"
              style={ghostButtonStyle}
              disabled={busy || courtLabels.length >= courtCount}
              onClick={() => updateVenueSettings({ venue_court_labels: [...courtLabels, `Court ${courtLabels.length + 1}`] })}
            >
              Add court title
            </button>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.65rem", marginTop: "0.75rem" }}>
            {courtLabels.map((label, index) => (
              <label key={`venue-court-${index}`}>
                <strong>Court {index + 1} title</strong><br />
                <div style={{ display: "flex", gap: "0.4rem" }}>
                  <input
                    value={label}
                    disabled={busy}
                    style={{ ...inputStyle, borderColor: duplicateCourtTitles.has(label.toLowerCase()) ? "#ef4444" : "#cbd5e1" }}
                    onChange={(event) => updateVenueSettings({ venue_court_labels: courtLabels.map((row, rowIndex) => rowIndex === index ? event.target.value : row) })}
                  />
                  <button
                    type="button"
                    disabled={busy}
                    aria-label={`Remove court ${index + 1} title`}
                    style={{ ...ghostButtonStyle, padding: "0.45rem 0.65rem", color: "#991b1b" }}
                    onClick={() => updateVenueSettings({ venue_court_labels: courtLabels.filter((_, rowIndex) => rowIndex !== index) })}
                  >
                    Remove
                  </button>
                </div>
              </label>
            ))}
          </div>
          {!courtLabels.length ? <p style={{ color: "#64748b", marginBottom: 0 }}>No court titles are required.</p> : null}
          {duplicateCourtTitles.size ? <p role="alert" style={{ color: "#b91c1c" }}>Court titles must be unique.</p> : null}
        </article>

        <article style={cardStyle}>
          <h3 style={{ marginTop: 0 }}>Tournament days</h3>
          <p style={{ color: "#64748b" }}>
            Dates are generated automatically from the tournament start and end dates. Dates and chronological order are fixed; only the public day label can be edited here.
          </p>
          <div style={{ display: "grid", gap: "0.65rem" }}>
            {configuration.days.map((row, index) => (
              <article key={row.key} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: "#f8fafc" }}>
                <div style={{ display: "grid", gridTemplateColumns: "minmax(210px, 2fr) minmax(170px, 1fr)", gap: "0.75rem", alignItems: "end" }}>
                  <label>
                    <strong>Day {index + 1} label</strong><br />
                    <input
                      value={dayLabel(row.value)}
                      disabled={busy}
                      style={inputStyle}
                      onChange={(event) => {
                        const previousReferences = new Set([dayReference(row.value), dayLabel(row.value)].filter(Boolean));
                        const nextValue = setRecordString(row.value, ["label"], event.target.value);
                        const nextReference = dayReference(nextValue) || dayLabel(nextValue);
                        const replaceReferences = (record: SetupRecord) => setEventDayReferences(
                          record,
                          eventDayReferences(record).map((reference) => previousReferences.has(reference) ? nextReference : reference)
                        );
                        setConfiguration((current) => ({
                          ...current,
                          days: replaceBuilderRow(current.days, row.key, nextValue),
                          eventFamilies: current.eventFamilies.map((family) => ({ ...family, value: replaceReferences(family.value) })),
                          eventOptions: current.eventOptions.map((division) => ({ ...division, value: replaceReferences(division.value) }))
                        }));
                        setImpactReview(null);
                      }}
                    />
                  </label>
                  <label>
                    <strong>Fixed tournament date</strong><br />
                    <input value={dateValue(row.value.event_date)} readOnly disabled style={inputStyle} />
                  </label>
                </div>
              </article>
            ))}
          </div>
        </article>

        {dayIssues.length || duplicateCourtTitles.size ? (
          <article style={{ ...cardStyle, borderColor: "#fecaca" }}>
            <h3 style={{ marginTop: 0 }}>Items to fix before continuing</h3>
            <ul>
              {dayIssues.map((issue) => <li key={`${issue.path}-${issue.message}`}>{issue.message}</li>)}
              {duplicateCourtTitles.size ? <li>Court titles must be unique.</li> : null}
            </ul>
          </article>
        ) : null}

        {footerRow(
          <>
            <Link href={tournamentSetupStepHref("basics", tournamentId, basics.name || tournamentName)} style={ghostButtonStyle}>Back to Basics</Link>
            <button
              type="button"
              style={buttonStyle}
              disabled={busy || !configuration.days.length || dayIssues.length > 0 || duplicateCourtTitles.size > 0}
              onClick={() => void saveDraftAndContinue("events")}
            >
              {busy ? "Saving draft…" : "Save draft and continue to Competition"}
            </button>
          </>
        )}
      </div>
    );
  }

  function renderReview() {
    const impact = impactReview?.publish_impact || {};
    const blocked = Array.isArray(impact.blocked) ? impact.blocked : [];
    const rawBlockedDetails = Array.isArray(impact.blocked_details) && impact.blocked_details.length
      ? impact.blocked_details
      : blocked;
    const blockedDetails = rawBlockedDetails.map(blockedImpactDetail);
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
    const unresolvedBlockers = blockedDetails.filter((item) => !forcedResolutionComplete(item));
    const forcePlans = blockedDetails.filter((item) => Boolean(forcedResolutionPlan(item)));

    const valueChanged = (current: unknown, proposed: unknown) =>
      JSON.stringify(current ?? null) !== JSON.stringify(proposed ?? null);
    const comparisons = [
      { field: "Tournament name", current: publishedBasics.name, proposed: basics.name },
      {
        field: "Tournament dates",
        current: `${publishedBasics.startDate || "Not set"} – ${publishedBasics.endDate || "Not set"}`,
        proposed: `${basics.startDate || "Not set"} – ${basics.endDate || "Not set"}`
      },
      { field: "Venue", current: publishedBasics.locationName, proposed: basics.locationName },
      { field: "Timezone", current: publishedBasics.timezone, proposed: basics.timezone },
      {
        field: "Venue court count",
        current: venueCourtCount(publishedSettings, publishedConfiguration),
        proposed: venueCourtCount(settings, configuration)
      },
      {
        field: "Optional court titles",
        current: venueCourtLabels(publishedSettings, publishedConfiguration),
        proposed: venueCourtLabels(settings, configuration)
      },
      {
        field: "Registration window",
        current: `${safeString(publishedSettings.registration_open_at) || "Not set"} → ${safeString(publishedSettings.registration_close_at) || "Not set"}`,
        proposed: `${safeString(settings.registration_open_at) || "Not set"} → ${safeString(settings.registration_close_at) || "Not set"}`
      },
      {
        field: "Sponsors",
        current: publishedBasics.sponsors.map((row) => row.name),
        proposed: basics.sponsors.map((row) => row.name)
      },
      {
        field: "Events and policies",
        current: publishedConfiguration.eventFamilies.map((row) => ({
          event: eventFamilyName(row.value),
          format: safeString(row.value.competition_format) === "FOUR_PLAYER_TEAM"
            ? "Four-player team"
            : safeString(row.value.participant_type),
          age: agePolicySummary(readAgePolicy(row.value, EVENT_AGE_POLICY_FIELDS)),
          days: eventDayReferences(row.value)
        })),
        proposed: configuration.eventFamilies.map((row) => ({
          event: eventFamilyName(row.value),
          format: safeString(row.value.competition_format) === "FOUR_PLAYER_TEAM"
            ? "Four-player team"
            : safeString(row.value.participant_type),
          age: agePolicySummary(readAgePolicy(row.value, EVENT_AGE_POLICY_FIELDS)),
          days: eventDayReferences(row.value)
        }))
      },
      {
        field: "Divisions",
        current: publishedConfiguration.eventOptions.map((row) => ({
          division: eventDivisionName(row.value),
          event: eventFamilyName(row.value),
          skill: row.value.skill_label,
          age: row.value.age_label,
          fee: row.value.price_usd
        })),
        proposed: configuration.eventOptions.map((row) => ({
          division: eventDivisionName(row.value),
          event: eventFamilyName(row.value),
          skill: row.value.skill_label,
          age: row.value.age_label,
          fee: row.value.price_usd
        }))
      }
    ].filter((row) => valueChanged(row.current, row.proposed));

    const domainCards = [
      {
        key: "basics" as TournamentSetupStep,
        label: "Tournament",
        complete: basicsReady && scheduleReady,
        draft: `${basics.name || "Untitled"} · ${basics.locationName || "No venue"} · ${venueCourtCount(settings, configuration)} courts`,
        published: `${publishedBasics.name || "Untitled"} · ${publishedBasics.locationName || "No venue"} · ${venueCourtCount(publishedSettings, publishedConfiguration)} courts`
      },
      {
        key: "events" as TournamentSetupStep,
        label: "Competition",
        complete: eventFamiliesReady && divisionsReady,
        draft: `${configuration.eventFamilies.length} event(s) · ${configuration.eventOptions.length} division(s)`,
        published: `${publishedConfiguration.eventFamilies.length} event(s) · ${publishedConfiguration.eventOptions.length} division(s)`
      },
      {
        key: "pricing" as TournamentSetupStep,
        label: "Commerce",
        complete: true,
        draft: "Fees and catalog reviewed in the Commerce domain",
        published: "Published catalog remains active until its reviewed save succeeds"
      },
      {
        key: "review" as TournamentSetupStep,
        label: "Review",
        complete: ready && unresolvedBlockers.length === 0,
        draft: impactReview
          ? `${warnings.length} warning(s) · ${unresolvedBlockers.length} unresolved blocker(s)`
          : "Impact review calculating",
        published: registrationStatus.toLowerCase() === "open" ? "Registration open" : "Registration not open"
      }
    ];

    return (
      <div style={{ display: "grid", gap: "1rem" }}>
        <article style={{ ...cardStyle, background: "#eff6ff", borderColor: "#bfdbfe" }}>
          <h2 style={{ marginTop: 0 }}>Review · Preview, conflicts, publish, and registration</h2>
          <p style={{ color: "#475569", marginBottom: 0 }}>
            This domain answers one question: <strong>If I publish now, what tournament will exist?</strong> The impact review runs automatically, compares the unpublished draft with the live tournament, and never writes data. Publishing setup and opening registration remain separate consequential actions.
          </p>
        </article>

        <article style={cardStyle}>
          <h3 style={{ marginTop: 0 }}>Four-domain readiness</h3>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "0.75rem" }}>
            {domainCards.map((item) => (
              <Link
                key={item.label}
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

        <article style={cardStyle}>
          <h3 style={{ marginTop: 0 }}>Tournament preview</h3>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.65rem" }}>
            <div><strong>Tournament</strong><br />{basics.name || "Untitled tournament"}</div>
            <div><strong>Dates</strong><br />{basics.startDate || "Not set"} – {basics.endDate || "Not set"}</div>
            <div><strong>Venue</strong><br />{basics.locationName || "Not set"}</div>
            <div><strong>Courts</strong><br />{venueCourtCount(settings, configuration)}</div>
            <div><strong>Registration window</strong><br />{safeString(settings.registration_open_at) || "Not set"}<br />to {safeString(settings.registration_close_at) || "Not set"}</div>
          </div>
          <div style={{ display: "grid", gap: "0.75rem", marginTop: "1rem" }}>
            {configuration.eventFamilies.map((family) => {
              const familyName = eventFamilyName(family.value);
              const divisions = configuration.eventOptions.filter((division) => eventFamilyName(division.value).toLowerCase() === familyName.toLowerCase());
              const structure = safeString(family.value.competition_format) === "FOUR_PLAYER_TEAM"
                ? "Four-player team"
                : safeString(family.value.participant_type).replaceAll("_", " ");
              return (
                <article key={family.key} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: "#f8fafc" }}>
                  <strong>{familyName}</strong>
                  <p style={{ margin: "0.25rem 0", color: "#475569" }}>
                    {structure} · {agePolicySummary(readAgePolicy(family.value, EVENT_AGE_POLICY_FIELDS))} · {eventDayReferences(family.value).length} tournament day(s)
                  </p>
                  {divisions.length ? (
                    <ul style={{ marginBottom: 0 }}>
                      {divisions.map((division) => (
                        <li key={division.key}>
                          {eventDivisionName(division.value)} · {safeString(division.value.skill_label) || "Open"} · ${Number(division.value.price_usd || 0).toFixed(2)}
                        </li>
                      ))}
                    </ul>
                  ) : <p style={{ color: "#b91c1c", marginBottom: 0 }}>No divisions</p>}
                </article>
              );
            })}
          </div>
        </article>

        {comparisons.length ? (
          <article style={cardStyle}>
            <h3 style={{ marginTop: 0 }}>Published versus proposed</h3>
            <p style={{ color: "#64748b" }}>Only fields that differ are shown.</p>
            <div style={{ display: "grid", gap: "0.65rem" }}>
              {comparisons.map((comparison) => (
                <article key={comparison.field} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem" }}>
                  <strong>{comparison.field}</strong>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))", gap: "0.65rem", marginTop: "0.45rem" }}>
                    <div style={{ padding: "0.6rem", borderRadius: "10px", background: "#f8fafc" }}><small>Current published value</small><br />{reviewValue(comparison.current)}</div>
                    <div style={{ padding: "0.6rem", borderRadius: "10px", background: "#eff6ff" }}><small>Proposed draft value</small><br />{reviewValue(comparison.proposed)}</div>
                  </div>
                </article>
              ))}
            </div>
          </article>
        ) : (
          <article style={{ ...cardStyle, background: "#f0fdf4", borderColor: "#bbf7d0" }}>
            <strong>No unpublished setup differences are currently detected.</strong>
          </article>
        )}

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
          <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", alignItems: "flex-start", flexWrap: "wrap" }}>
            <div>
              <h3 style={{ marginTop: 0 }}>Impact review and conflict resolution</h3>
              <p style={{ color: "#475569", marginBottom: 0 }}>
                The read-only review runs automatically when this page opens or the draft changes. Refresh it manually at any time. No rows are written.
              </p>
            </div>
            <button type="button" style={ghostButtonStyle} disabled={!ready || busy} onClick={() => void reviewImpact()}>
              {busy ? "Reviewing…" : "Refresh review"}
            </button>
          </div>
          {!impactReview && ready ? <p role="status" style={{ color: "#1d4ed8" }}>Calculating tournament impact…</p> : null}
          {impactReview ? (
            <div style={{ marginTop: "0.75rem" }}>
              <p style={{ color: "#166534", fontWeight: 800 }}>Review complete. No rows were written.</p>
              {warnings.length ? (
                <>
                  <strong>Warnings</strong>
                  <ul>{warnings.map((warning, index) => <li key={index}>{formatImpactItem(warning)}</li>)}</ul>
                </>
              ) : null}
              {blockedDetails.length ? (
                <>
                  <strong>Blocked changes — resolve each before publishing</strong>
                  <div style={{ display: "grid", gap: "0.75rem", marginTop: "0.55rem" }}>
                    {blockedDetails.map((item) => {
                      const editHref = item.step === "divisions"
                        ? `${tournamentSetupStepHref("divisions", tournamentId, basics.name || tournamentName)}&resolveDivision=${encodeURIComponent(item.entity_id || "")}`
                        : tournamentSetupStepHref(item.step || "review", tournamentId, basics.name || tournamentName);
                      const plan = forcedResolutionPlan(item);
                      const planRows = Array.isArray(plan?.affected_registrations)
                        ? (plan?.affected_registrations as Array<Record<string, unknown>>)
                        : [];
                      const forceComplete = forcedResolutionComplete(item);
                      return (
                        <article key={item.block_id} style={{ padding: "0.8rem", border: `1px solid ${forceComplete ? "#bbf7d0" : "#fecaca"}`, borderRadius: "12px", background: forceComplete ? "#f0fdf4" : "#fef2f2" }}>
                          <strong>{item.entity_label || "Blocked setup change"}</strong>
                          <p>{item.message}</p>
                          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.65rem" }}>
                            <div style={{ padding: "0.6rem", borderRadius: "10px", background: "white" }}><small>Current published {item.field || "value"}</small><br />{reviewValue(item.current_value)}</div>
                            <div style={{ padding: "0.6rem", borderRadius: "10px", background: "#eff6ff" }}><small>Proposed draft {item.field || "value"}</small><br />{reviewValue(item.proposed_value)}</div>
                          </div>
                          <div style={{ display: "flex", gap: "0.55rem", flexWrap: "wrap", marginTop: "0.7rem" }}>
                            <button type="button" style={ghostButtonStyle} onClick={() => keepPublishedValueForBlockedChange(item)}>
                              Keep published value
                            </button>
                            <Link href={editHref} style={ghostButtonStyle}>Edit affected draft</Link>
                            {!plan && item.resolution_options?.includes("FORCE_CHANGE_WITH_RESOLUTION") ? (
                              <button type="button" style={{ ...ghostButtonStyle, borderColor: "#b91c1c", color: "#991b1b" }} disabled={!item.affected_registrations.length} onClick={() => beginForcedResolution(item)}>
                                Force change with registration resolution
                              </button>
                            ) : null}
                          </div>
                          {!plan && item.resolution_options?.includes("FORCE_CHANGE_WITH_RESOLUTION") && !item.affected_registrations.length ? <p style={{ color: "#92400e" }}>Refresh the review to load affected-registration details before forcing this change.</p> : null}
                          {!plan && !item.resolution_options?.includes("FORCE_CHANGE_WITH_RESOLUTION") ? <p style={{ color: "#b91c1c" }}>This change cannot be forced after draws, teams, or games exist. Edit the draft or keep the published value.</p> : null}
                          {plan ? (
                            <div style={{ marginTop: "0.8rem", paddingTop: "0.8rem", borderTop: "1px solid #fecaca" }}>
                              <p style={{ marginTop: 0 }}>
                                <strong>Manual registration-resolution queue</strong><br />
                                Publication remains blocked until every row has an action and audit note. Complete the actual registration change through the linked editor, then record the resolution here.
                              </p>
                              <div style={{ display: "grid", gap: "0.65rem" }}>
                                {planRows.map((row) => {
                                  const registration: AffectedRegistration = {
                                    registration_id: safeString(row.registration_id),
                                    selection_id: safeString(row.selection_id),
                                    display_name: safeString(row.display_name),
                                    email: safeString(row.email),
                                    registration_status: safeString(row.registration_status)
                                  };
                                  const editorHref = `/admin/tournaments/registration/registrants/${encodeURIComponent(registration.registration_id)}?${new URLSearchParams({ tournament: tournamentId, name: basics.name || tournamentName }).toString()}`;
                                  const resolved = Boolean(row.resolved);
                                  return (
                                    <article key={`${registration.registration_id}-${registration.selection_id || "registration"}`} style={{ border: "1px solid #e2e8f0", borderRadius: "10px", padding: "0.65rem", background: "white" }}>
                                      <div style={{ display: "flex", justifyContent: "space-between", gap: "0.55rem", flexWrap: "wrap" }}>
                                        <div><strong>{registration.display_name || registration.email || registration.registration_id}</strong><br /><small>{registration.email || "No email"} · {registration.registration_status || "Unknown status"}</small></div>
                                        <Link href={editorHref} style={ghostButtonStyle}>Open registration editor</Link>
                                      </div>
                                      <div style={{ display: "grid", gridTemplateColumns: "minmax(190px, 1fr) minmax(260px, 2fr)", gap: "0.65rem", marginTop: "0.6rem" }}>
                                        <label><strong>Resolution action</strong><br />
                                          <select value={safeString(row.action)} style={inputStyle} onChange={(event) => updateForcedRegistration(item, registration, { action: event.target.value })}>
                                            <option value="">Choose action…</option>
                                            <option value="MOVE_REGISTRATION">Move registration</option>
                                            <option value="CANCEL_REFUND">Cancel and refund</option>
                                            <option value="CREDIT">Cancel or move with credit</option>
                                            <option value="GRANDFATHER">Grandfather only when explicitly safe</option>
                                            <option value="OTHER">Other manual resolution</option>
                                          </select>
                                        </label>
                                        <label><strong>Audit note</strong><br />
                                          <textarea value={safeString(row.notes)} style={{ ...inputStyle, minHeight: "76px" }} placeholder="Describe the actual registration change and why it resolves the conflict." onChange={(event) => updateForcedRegistration(item, registration, { notes: event.target.value })} />
                                        </label>
                                      </div>
                                      <label style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginTop: "0.6rem", fontWeight: 800 }}>
                                        <input
                                          type="checkbox"
                                          checked={resolved}
                                          disabled={!safeString(row.action) || !safeString(row.notes)}
                                          onChange={(event) => updateForcedRegistration(item, registration, { resolved: event.target.checked })}
                                        />
                                        I completed and verified this registration action
                                      </label>
                                      <small style={{ color: resolved ? "#166534" : "#92400e", fontWeight: 800 }}>{resolved ? "Resolved for publication" : "Action, audit note, and completion confirmation required"}</small>
                                    </article>
                                  );
                                })}
                              </div>
                              <p style={{ color: forceComplete ? "#166534" : "#92400e", fontWeight: 800 }}>
                                {forceComplete ? "All affected registrations are documented as resolved." : "This force-change queue is still incomplete."}
                              </p>
                            </div>
                          ) : null}
                        </article>
                      );
                    })}
                  </div>
                  {forcePlans.length ? (
                    <div style={{ marginTop: "0.75rem" }}>
                      <button type="button" style={buttonStyle} disabled={busy || !resolutionDraftDirty} onClick={() => void saveResolutionDraft()}>
                        {busy ? "Saving queue…" : resolutionDraftDirty ? "Save registration-resolution queue" : "Resolution queue saved"}
                      </button>
                      {resolutionDraftDirty ? <p style={{ color: "#92400e" }}>Save the queue before publishing or refreshing the browser.</p> : null}
                    </div>
                  ) : null}
                </>
              ) : (
                <p style={{ color: "#166534", fontWeight: 800 }}>No blocked changes.</p>
              )}
            </div>
          ) : null}
        </article>

        <article style={cardStyle}>
          <h3 style={{ marginTop: 0 }}>Publish tournament</h3>
          <p style={{ color: "#475569" }}>
            Publish the exact reviewed Tournament, Competition, and Commerce setup. Existing registrations remain protected; a forced change is accepted only when its affected-registration queue is complete and saved.
          </p>
          <ConfirmAction
            triggerLabel={busy ? "Publishing…" : "Publish reviewed tournament"}
            title="Publish this reviewed tournament?"
            description="Apply the exact reviewed draft to the published tournament. Registration status remains a separate action. Forced changes and their registration resolutions are written to the audit record."
            confirmLabel="Yes, publish tournament"
            confirmationText={publishConfirmation}
            disabled={!impactReview || reviewedDraftSignature !== fullDraftSignature(basics, settings, configuration) || unresolvedBlockers.length > 0 || resolutionDraftDirty}
            busy={busy}
            onConfirm={publishSetup}
          />
          {unresolvedBlockers.length ? <p style={{ color: "#b91c1c" }}>Resolve {unresolvedBlockers.length} blocked change{unresolvedBlockers.length === 1 ? "" : "s"} before publishing.</p> : null}
          {resolutionDraftDirty ? <p style={{ color: "#92400e" }}>Save the registration-resolution queue before publishing.</p> : null}
        </article>

        <article style={cardStyle}>
          <h3 style={{ marginTop: 0 }}>Open registration</h3>
          <p style={{ color: "#475569" }}>Make the published tournament available to registrants. Offline payment remains the only payment mode.</p>
          {registrationStatus.toLowerCase() === "open" ? (
            <p style={{ color: "#166534", fontWeight: 800 }}>Registration is already open.</p>
          ) : (
            <ConfirmAction
              triggerLabel={busy ? "Opening…" : "Open registration"}
              title="Open tournament registration?"
              description="Open registration using the published tournament, registration window, policies, divisions, prices, and Partner Board settings."
              confirmLabel="Yes, open registration"
              confirmationText={settingsConfirmation}
              disabled={!(setupPublishedThisSession || registrationCanOpen)}
              busy={busy}
              onConfirm={openRegistration}
            />
          )}
          {!setupPublishedThisSession && !registrationCanOpen && registrationStatus.toLowerCase() !== "open" ? (
            <p style={{ color: "#64748b" }}>Publish a complete tournament with no unresolved conflicts before opening registration.</p>
          ) : null}
        </article>

        {footerRow(
          <>
            <Link href={tournamentSetupStepHref("pricing", tournamentId, basics.name || tournamentName)} style={ghostButtonStyle}>Back to Commerce</Link>
            <Link href={`/admin/tournaments/tournament?${new URLSearchParams({ tournament: tournamentId, name: basics.name || tournamentName }).toString()}`} style={ghostButtonStyle}>Return to Tournament Home</Link>
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
            ? "Changes in the Tournament and Competition domains are saved only to this private admin draft. Public tournament pages continue using the currently published configuration until the Review domain publishes the complete tournament. Commerce catalog changes retain their separate reviewed save."
            : "No unpublished setup changes are waiting. New Tournament and Competition edits remain private until the Review domain publishes them; Commerce catalog changes retain their separate reviewed save."}
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
          Domain {domainDefinition.number} of 4 · {domainDefinition.label}
          {domainDefinition.steps.length > 1 ? ` · Section ${domainSectionIndex} of ${domainDefinition.steps.length}` : ""}
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
