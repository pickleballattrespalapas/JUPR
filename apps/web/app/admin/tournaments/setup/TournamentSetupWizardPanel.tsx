"use client";

import Link from "next/link";
import TournamentSponsorEditor from "./TournamentSponsorEditor";
import { sponsorTiers, type SponsorDraft, type SponsorTier } from "@/lib/tournamentSponsors";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import { actionSuccess, InteractionActionError, type ActionSuccess } from "@/components/interaction";
import { tournamentSetupActionError } from "@/lib/tournamentSetupActionError";
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
import { tournamentRouteHref } from "@/lib/tournamentRouteContext";
import TournamentCommercePanel from "../commerce/TournamentCommercePanel";
import {
  appendBuilderRow,
  comparablePublishedConfigurationPayload,
  configurationPayload,
  dayLabel,
  dayReference,
  editableString,
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
  stableSetupJsonStringify,
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
import TournamentDivisionPresetDialog from "./TournamentDivisionPresetDialog";
import TournamentBulkAddCourtsDialog from "./TournamentBulkAddCourtsDialog";
import TournamentDivisionBulkEditDialog from "./TournamentDivisionBulkEditDialog";
import TournamentSetupDivisionCard from "./TournamentSetupDivisionCard";
import TournamentSetupDivisionDialog from "./TournamentSetupDivisionDialog";
import TournamentSetupPolicies, { withDefaultTournamentPolicies } from "./TournamentSetupPolicies";
import {
  EVENT_AGE_POLICY_FIELDS,
  agePolicySummary,
  readAgePolicy
} from "./TournamentAgePolicyEditor";
import {
  configurationWithVenueInventory,
  courtDisplayName,
  dayAvailableCourtIds as venueDayAvailableCourtIds,
  newVenueCourt,
  normalizeVenueCourts,
  settingsWithVenueCourts,
  venueIssues,
  withVenueCourtAvailability,
  type VenueCourt
} from "./TournamentVenueModel";
import {
  ReviewComparisonDisplay,
  ReviewValueDisplay,
  humanReviewFieldLabel
} from "./TournamentReviewValue";
import {
  setupPublicationStatus,
  type SetupDetailLoadState,
  type SetupPublicationStatus
} from "./tournamentSetupPublicationStatus";

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
  sponsor_logo_urls?: Record<string, string>;
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
  tournament_status?: string;
  activated_from_draft?: boolean;
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
    provisional_count: number;
    viable: boolean;
    entries: Array<{ registration_id: string; selection_id?: string | null; display_name: string; age?: number | null; partner_age?: number | null; effective_age?: number | null }>;
  }>;
  recommendations: string[];
  pending_entries?: Array<Record<string, unknown>>;
  unassigned_entries: Array<Record<string, unknown>>;
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
  drawId: string;
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

const publicationBanners: Record<
  SetupPublicationStatus,
  { background: string; borderColor: string; title: string; description: string }
> = {
  unpublished: {
    background: "#fffbeb",
    borderColor: "#fde68a",
    title: "Unpublished setup draft",
    description:
      "Changes in the Tournament and Competition domains are saved only to this private admin draft. Public tournament pages continue using the currently published configuration until the Review domain publishes the complete tournament. Commerce catalog changes retain their separate reviewed save."
  },
  current: {
    background: "#f0fdf4",
    borderColor: "#bbf7d0",
    title: "Published setup is current",
    description:
      "No unpublished setup changes are waiting. New Tournament and Competition edits remain private until the Review domain publishes them; Commerce catalog changes retain their separate reviewed save."
  },
  unavailable: {
    background: "#fef2f2",
    borderColor: "#fecaca",
    title: "Published setup status unavailable",
    description:
      "The app could not load both the private draft and published configuration, so it cannot verify whether unpublished setup changes are waiting. Review the error below and retry."
  },
  checking: {
    background: "#eff6ff",
    borderColor: "#bfdbfe",
    title: "Checking published setup…",
    description:
      "Loading the private draft and published configuration before comparing them."
  }
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

function normalizeSponsors(value: unknown, urls: Record<string, string> = {}): SponsorDraft[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((row): row is Record<string, unknown> => Boolean(row) && typeof row === "object" && !Array.isArray(row))
    .map((row, index) => ({
      id: safeString(row.id) || sponsorId(),
      name: safeString(row.name),
      sort_order: index,
      tier: sponsorTiers.includes(row.tier as SponsorTier) ? row.tier as SponsorTier : "supporting",
      logo_path: safeString(row.logo_path),
      logo_url: urls[safeString(row.logo_path)] || "",
      is_visible: row.is_visible !== false,
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
      tier: sponsor.tier,
      logo_path: sponsor.logo_path,
      is_visible: sponsor.is_visible,
      name: sponsor.name.trim(),
      level: sponsor.level.trim(),
      website: sponsor.website.trim(),
      notes: sponsor.notes.trim()
    }))
  };
}

function settingsDraftPayload(settings: Record<string, unknown>): Record<string, unknown> {
  const courts = normalizeVenueCourts(settings);
  const normalized = settingsWithVenueCourts(settings, courts);
  const registrationSlug = safeString(settings.registration_slug)
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/-{2,}/g, "-")
    .replace(/^-|-$/g, "");
  return {
    registration_slug: registrationSlug,
    locale: safeString(settings.locale).trim() || "en",
    registration_status: safeString(settings.registration_status) || "draft",
    registration_open_at: settings.registration_open_at || null,
    registration_close_at: settings.registration_close_at || null,
    waitlist_enabled: Boolean(settings.waitlist_enabled),
    partner_board_enabled: Boolean(settings.partner_board_enabled),
    rules_markdown: safeString(settings.rules_markdown),
    refund_policy_markdown: safeString(settings.refund_policy_markdown),
    weather_policy_markdown: safeString(settings.weather_policy_markdown),
    sponsor_markdown: safeString(settings.sponsor_markdown),
    venue_address: safeString(settings.venue_address).trim(),
    venue_directions: safeString(settings.venue_directions).trim(),
    venue_courts_json: normalized.venue_courts_json,
    venue_court_count: normalized.venue_court_count,
    venue_court_labels: normalized.venue_court_labels,
    timezone: safeString(settings.timezone).trim() || "America/Mazatlan",
    forced_change_resolutions: objectValue(settings.forced_change_resolutions),
    communication_change_acknowledgements: objectValue(settings.communication_change_acknowledgements)
  };
}

function venueCourts(settings: Record<string, unknown>, configuration?: SetupConfiguration): VenueCourt[] {
  return normalizeVenueCourts(settings, configuration?.days || []);
}

function venueCourtCount(settings: Record<string, unknown>, configuration?: SetupConfiguration): number {
  return venueCourts(settings, configuration).length;
}

function venueCourtLabels(settings: Record<string, unknown>, configuration?: SetupConfiguration): string[] {
  return venueCourts(settings, configuration).map(courtDisplayName);
}

function configurationWithVenue(
  configuration: SetupConfiguration,
  settings: Record<string, unknown>
): SetupConfiguration {
  return configurationWithVenueInventory(configuration, settings);
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
  delete impactSettings.communication_change_acknowledgements;
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
  delete comparableSettings.communication_change_acknowledgements;
  // Registration status is controlled by the separate Open/Close action. It
  // must not make an otherwise published setup look like an unpublished draft.
  delete comparableSettings.registration_status;
  const normalized = configurationWithVenue(configuration, settings);
  const comparableConfiguration = comparablePublishedConfigurationPayload(normalized);
  const builder = configurationPayload(normalized);
  return stableSetupJsonStringify({
    basics: basicsDraftPayload(basics),
    settings: comparableSettings,
    days: comparableConfiguration.days,
    event_families: builder.event_families,
    event_options: comparableConfiguration.event_options
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
      !venueIssues(settings, configuration).length
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
  current_source?: string;
  proposed_source?: string;
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
  current_source?: string;
  proposed_source?: string;
  step?: TournamentSetupStep;
  resolution_options?: string[];
  affected_registrations: AffectedRegistration[];
};

type CommunicationImpactDetail = {
  impact_id: string;
  impact_type: string;
  message: string;
  entity_type?: string;
  entity_id?: string;
  entity_label?: string;
  field?: string;
  current_value?: unknown;
  proposed_value?: unknown;
  current_source?: string;
  proposed_source?: string;
  step?: TournamentSetupStep;
  resolution_options?: string[];
  requires_acknowledgement: boolean;
  requires_data_completion: boolean;
  data_completion_registrations: AffectedRegistration[];
  affected_registrations: AffectedRegistration[];
};

function affectedRegistrations(value: unknown): AffectedRegistration[] {
  return Array.isArray(value)
    ? value
        .filter((row) => row && typeof row === "object" && !Array.isArray(row))
        .map((row) => {
          const affected = row as Record<string, unknown>;
          return {
            registration_id: safeString(affected.registration_id),
            selection_id: safeString(affected.selection_id),
            display_name: safeString(affected.display_name) || safeString(affected.email) || safeString(affected.registration_id),
            email: safeString(affected.email),
            registration_status: safeString(affected.registration_status),
            current_value: affected.current_value,
            proposed_value: affected.proposed_value,
            current_source: safeString(affected.current_source),
            proposed_source: safeString(affected.proposed_source)
          };
        })
    : [];
}

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
      current_source: safeString(row.current_source),
      proposed_source: safeString(row.proposed_source),
      step: safeString(row.step) as TournamentSetupStep,
      resolution_options: Array.isArray(row.resolution_options) ? row.resolution_options.map((value) => safeString(value)) : [],
      affected_registrations: affectedRegistrations(row.affected_registrations)
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

function communicationImpactDetail(value: unknown): CommunicationImpactDetail {
  const row = value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
  return {
    impact_id: safeString(row.impact_id) || `${safeString(row.entity_id)}:${safeString(row.field)}:communication`,
    impact_type: safeString(row.impact_type) || "SCHEDULE_COMMUNICATION",
    message: formatImpactItem(value),
    entity_type: safeString(row.entity_type),
    entity_id: safeString(row.entity_id),
    entity_label: safeString(row.entity_label),
    field: safeString(row.field),
    current_value: row.current_value,
    proposed_value: row.proposed_value,
    current_source: safeString(row.current_source),
    proposed_source: safeString(row.proposed_source),
    step: safeString(row.step) as TournamentSetupStep,
    resolution_options: Array.isArray(row.resolution_options) ? row.resolution_options.map((item) => safeString(item)) : [],
    requires_acknowledgement: row.requires_acknowledgement !== false,
    requires_data_completion: Boolean(row.requires_data_completion),
    data_completion_registrations: affectedRegistrations(row.data_completion_registrations),
    affected_registrations: affectedRegistrations(row.affected_registrations)
  };
}

function communicationImpactTitle(item: CommunicationImpactDetail): string {
  const impactType = item.impact_type.toUpperCase();
  if (impactType === "AGE_GROUPING_COMMUNICATION") {
    return "Age-grouping change — no registration conflict";
  }
  if (impactType === "SCHEDULE_COMMUNICATION") {
    return "Schedule change — no registration conflict";
  }
  if (impactType === "ELIGIBILITY_COMMUNICATION") {
    return "Eligibility-rule change — no known registration conflict";
  }
  return "Registration-preserving change — no registration conflict";
}

function affectedRegistrationImpactSummary(
  registration: AffectedRegistration,
  item: CommunicationImpactDetail
): string {
  const proposed = objectValue(registration.proposed_value);
  const impactType = item.impact_type.toUpperCase();
  if (["AGE_GROUPING_COMMUNICATION", "ELIGIBILITY_COMMUNICATION"].includes(impactType)) {
    const age = objectValue(proposed.age_eligibility);
    const ageGroupingImpact = impactType === "AGE_GROUPING_COMMUNICATION";
    const preferredAgeLabel = safeString(proposed.preferred_age_group);
    const ageLabel = safeString(preferredAgeLabel || proposed.age_label);
    const effectiveAgeValue = proposed.effective_age;
    const effectiveAge = effectiveAgeValue == null || effectiveAgeValue === ""
      ? Number.NaN
      : Number(effectiveAgeValue);
    const playerAgeValue = age.player_age ?? proposed.player_age;
    const playerAge = playerAgeValue == null || playerAgeValue === ""
      ? Number.NaN
      : Number(playerAgeValue);
    const provisionalAgePlacement = ageGroupingImpact && Boolean(
      age.provisional
      || proposed.age_placement_provisional
    );
    const eligibilityStatus = safeString(
      ageGroupingImpact ? age.status : proposed.eligibility_status
    );
    const skill = objectValue(proposed.skill_eligibility);
    const skillStatus = safeString(skill.status);
    const ceiling = Number(skill.skill_ceiling_exclusive);
    const controllingRating = Number(skill.controlling_rating);
    const combinedRating = Number(skill.combined_rating);
    const combinedCap = Number(skill.combined_rating_cap);
    const issues = ageGroupingImpact
      ? [safeString(age.issue)].filter(Boolean)
      : Array.isArray(proposed.eligibility_issues)
        ? proposed.eligibility_issues.map((value) => safeString(value)).filter(Boolean)
        : [];
    const assignmentIssue = ageGroupingImpact ? "" : safeString(proposed.assignment_issue);
    const parts: string[] = [];
    if (provisionalAgePlacement) {
      const registrantAge = Number.isFinite(playerAge)
        ? ` using registrant age ${Number.isInteger(playerAge) ? playerAge.toFixed(0) : playerAge.toFixed(1)}`
        : "";
      parts.push(preferredAgeLabel
        ? `Provisional age group: ${preferredAgeLabel}${registrantAge}; recalculated when a partner is assigned.`
        : `Partner-based age placement is pending${registrantAge}; recalculated when a partner is assigned.`
      );
    } else {
      if (eligibilityStatus) parts.push(eligibilityStatus === "ELIGIBLE" ? "Eligible" : eligibilityStatus.replaceAll("_", " "));
      if (ageLabel && ageLabel !== "All ages") parts.push(`Preferred age group: ${ageLabel}`);
      if (Number.isFinite(effectiveAge)) parts.push(`team age ${Number.isInteger(effectiveAge) ? effectiveAge.toFixed(0) : effectiveAge.toFixed(1)}`);
    }
    if (skillStatus === "ELIGIBLE" && Number.isFinite(ceiling)) {
      parts.push(`skill eligible below ${ceiling.toFixed(2)}`);
    }
    if (Number.isFinite(controllingRating)) parts.push(`controlling rating ${controllingRating.toFixed(2)}`);
    if (Number.isFinite(combinedRating) && Number.isFinite(combinedCap)) {
      parts.push(`combined ${combinedRating.toFixed(2)} below ${combinedCap.toFixed(2)}`);
    }
    for (const issue of issues) if (!parts.includes(issue)) parts.push(issue);
    if (assignmentIssue && !parts.includes(assignmentIssue)) parts.push(assignmentIssue);
    return parts.join(" · ");
  }
  return "";
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
  drawId,
  step,
  resolveDivisionId = ""
}: Props) {
  const router = useRouter();
  const { accessToken, loading: sessionLoading } = useAdminSession();
  const [detail, setDetail] = useState<SetupDetail | null>(null);
  const [detailLoadState, setDetailLoadState] =
    useState<SetupDetailLoadState>("idle");
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
  const [divisionPresetFamilyKey, setDivisionPresetFamilyKey] = useState<string | undefined>(undefined);
  const [divisionDialogKey, setDivisionDialogKey] = useState<string | null | undefined>(undefined);
  const [bulkCourtDialogOpen, setBulkCourtDialogOpen] = useState(false);
  const [bulkDivisionSelecting, setBulkDivisionSelecting] = useState(false);
  const [selectedDivisionKeys, setSelectedDivisionKeys] = useState<string[]>([]);
  const [bulkDivisionDialogOpen, setBulkDivisionDialogOpen] = useState(false);
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
    setDetailLoadState("idle");
    setBasics(emptyBasics(tournamentName));
    setSettings({});
    setConfiguration(emptyConfiguration);
    setImpactReview(null);
    setReviewedDraftSignature("");
    setSetupPublishedThisSession(false);
    setBusy(false);
    setMessage(null);
    setEventDialogKey(undefined);
    setDivisionPresetFamilyKey(undefined);
    setDivisionDialogKey(undefined);
    setBulkCourtDialogOpen(false);
    setBulkDivisionSelecting(false);
    setSelectedDivisionKeys([]);
    setBulkDivisionDialogOpen(false);
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
    if (!accessToken) throw new InteractionActionError("Sign in before editing tournament setup.", { kind: "forbidden" });
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      throw tournamentSetupActionError(response, payload);
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
      sponsors: normalizeSponsors(publishedSettingsValue.sponsors_json, payload.sponsor_logo_urls)
    };
    const draftBasicsRecord = objectValue(draft.basics);
    const draftBasicsValue: BasicsDraft = {
      name: safeString(draftBasicsRecord.name) || publishedBasicsValue.name,
      startDate: dateValue(draftBasicsRecord.start_date) || publishedBasicsValue.startDate,
      endDate: dateValue(draftBasicsRecord.end_date) || publishedBasicsValue.endDate,
      locationName: safeString(draftBasicsRecord.location_name) || publishedBasicsValue.locationName,
      timezone: safeString(draftBasicsRecord.timezone) || publishedBasicsValue.timezone,
      sponsors: Array.isArray(draftBasicsRecord.sponsors_json)
        ? normalizeSponsors(draftBasicsRecord.sponsors_json, payload.sponsor_logo_urls)
        : publishedBasicsValue.sponsors
    };
    const draftSettingsValue = withDefaultTournamentPolicies({
      ...publishedSettingsValue,
      ...objectValue(draft.settings)
    });
    // Open/Close Registration is authoritative outside the setup draft. An
    // older draft snapshot must never override the current registration state.
    draftSettingsValue.registration_status = publishedSettingsValue.registration_status;

    const publishedDays = (payload.days || []).map(withDefaultDayCourts);
    const publishedCourts = normalizeVenueCourts(publishedSettingsValue, publishedDays);
    Object.assign(publishedSettingsValue, settingsWithVenueCourts(publishedSettingsValue, publishedCourts));

    const rawDraftDays = listValue(draft.days).map(withDefaultDayCourts);
    const draftCourts = normalizeVenueCourts(
      draftSettingsValue,
      rawDraftDays.length ? rawDraftDays : publishedDays
    );
    Object.assign(draftSettingsValue, settingsWithVenueCourts(draftSettingsValue, draftCourts));
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
      value: withVenueCourtAvailability(row.value, draftCourts)
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
      ).map((row) => withVenueCourtAvailability(row, publishedCourts)),
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
    setDetailLoadState("loading");
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
      setDetailLoadState("loaded");
    } catch (error) {
      if (detailRequest.isCurrent(generation)) {
        setDetailLoadState("failed");
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
        basics.name.trim() || tournamentName,
        drawId
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
            participant_type: safeString(value.participant_type) || "GENDER_DOUBLES",
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

  function nextConfigurationForEvent(value: SetupRecord): SetupConfiguration {
    const existing =
      typeof eventDialogKey === "string"
        ? configuration.eventFamilies.find((row) => row.key === eventDialogKey)
        : undefined;
    const previousName = existing ? eventFamilyName(existing.value) : "";
    const nextName = eventFamilyName(value);
    const nextDays = eventDayReferences(value);
    const eventFamilies = sortEventFamiliesByTournamentDay(
      existing
        ? replaceBuilderRow(configuration.eventFamilies, existing.key, value)
        : appendBuilderRow(configuration.eventFamilies, "family", value),
      configuration.days
    );
    const eventOptions = sortDivisionsByEventAndName(
      configuration.eventOptions.map((division) => {
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
      configuration.days
    );
    return { ...configuration, eventFamilies, eventOptions };
  }

  async function persistConfigurationDraft(
    nextConfiguration: SetupConfiguration,
    savedStep: TournamentSetupStep,
    successMessage: string,
    settingsOverride: Record<string, unknown> = settings,
    basicsOverride: BasicsDraft = basics,
    propagateError = false
  ): Promise<boolean> {
    if (!detail) return false;
    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const normalized = configurationWithGlobalStatus(
        configurationWithVenue(nextConfiguration, settingsOverride),
        settingsOverride.registration_status
      );
      const draft = configurationPayload(normalized);
      await requestJson<WriteResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/setup/tournaments/${encodeURIComponent(tournamentId)}/draft`,
        {
          method: "PUT",
          body: JSON.stringify({
            ...draft,
            basics: basicsDraftPayload(basicsOverride),
            settings: settingsDraftPayload(settingsOverride),
            saved_step: savedStep,
            expected_state_fingerprint: detail.state_fingerprint,
            confirmation_text: draftConfirmation
          })
        }
      );
      if (!actionRequest.isCurrent(generation)) return false;
      setConfiguration(nextConfiguration);
      await loadDetail();
      if (!actionRequest.isCurrent(generation)) return false;
      setMessage(successMessage);
      return true;
    } catch (error) {
      if (actionRequest.isCurrent(generation)) {
        setMessage(error instanceof Error ? error.message : "Unable to save the unpublished tournament draft.");
      }
      // Dialog actions must receive the actual rejection instead of a boolean
      // that hides the actionable message behind the open modal.
      if (propagateError) throw error;
      return false;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function saveEventDialog(value: SetupRecord) {
    const nextConfiguration = nextConfigurationForEvent(value);
    const saved = await persistConfigurationDraft(
      nextConfiguration,
      "events",
      `Event ${eventFamilyName(value) || "saved"} saved to the private admin draft. Nothing public changed.`
    );
    if (!saved) throw new Error("The event was not saved. Review the draft and try again.");
    setImpactReview(null);
    return actionSuccess("Event saved", `${eventFamilyName(value) || "The event"} was saved to the private admin draft. Nothing public changed.`);
  }

  async function saveDivisionDialog(value: SetupRecord) {
    const existing =
      typeof divisionDialogKey === "string"
        ? configuration.eventOptions.find((row) => row.key === divisionDialogKey)
        : undefined;
    const eventOptions = sortDivisionsByEventAndName(
      existing
        ? replaceBuilderRow(configuration.eventOptions, existing.key, value)
        : appendBuilderRow(configuration.eventOptions, "event", value),
      configuration.eventFamilies,
      configuration.days
    );
    const nextConfiguration = { ...configuration, eventOptions };
    const saved = await persistConfigurationDraft(
      nextConfiguration,
      "divisions",
      `Division ${eventDivisionName(value) || "saved"} saved to the private admin draft. Nothing public changed.`
    );
    if (!saved) throw new Error("The division was not saved. Review the draft and try again.");
    setImpactReview(null);
    return actionSuccess("Division saved", `${eventDivisionName(value) || "The division"} was saved to the private admin draft. Nothing public changed.`);
  }


  async function saveGeneratedDivisions(values: SetupRecord[]) {
    let rows = configuration.eventOptions;
    values.forEach((value) => {
      rows = appendBuilderRow(rows, "event", value);
    });
    const nextConfiguration = {
      ...configuration,
      eventOptions: sortDivisionsByEventAndName(
        rows,
        configuration.eventFamilies,
        configuration.days
      )
    };
    const saved = await persistConfigurationDraft(
      nextConfiguration,
      "events",
      `${values.length} division${values.length === 1 ? "" : "s"} generated and saved to the private admin draft. Nothing public changed.`
    );
    if (!saved) throw new Error("The generated divisions were not saved. Review the draft and try again.");
    setImpactReview(null);
    return actionSuccess("Divisions generated", `${values.length} division${values.length === 1 ? " was" : "s were"} generated and saved to the private admin draft. Nothing public changed.`);
  }

  async function saveBulkDivisionEdits(rows: Array<{ key: string; value: SetupRecord }>) {
    const replacements = new Map(rows.map((row) => [row.key, row.value]));
    const nextConfiguration = {
      ...configuration,
      eventOptions: sortDivisionsByEventAndName(
        configuration.eventOptions.map((row) => {
          const value = replacements.get(row.key);
          return value ? { ...row, value } : row;
        }),
        configuration.eventFamilies,
        configuration.days
      )
    };
    const saved = await persistConfigurationDraft(
      nextConfiguration,
      "divisions",
      `${rows.length} division${rows.length === 1 ? "" : "s"} updated together in the private admin draft. Nothing public changed.`
    );
    if (!saved) throw new Error("The selected divisions were not saved. Review the draft and try again.");
    setImpactReview(null);
    return actionSuccess("Divisions updated", `${rows.length} division${rows.length === 1 ? " was" : "s were"} updated together in the private admin draft. Nothing public changed.`);
  }

  async function removeEventFamily(rowKey: string) {
    const nextConfiguration = {
      ...configuration,
      eventFamilies: sortEventFamiliesByTournamentDay(
        removeBuilderRow(configuration.eventFamilies, rowKey),
        configuration.days
      )
    };
    await persistConfigurationDraft(
      nextConfiguration,
      "events",
      "Event removed from the private admin draft. Nothing public changed."
    );
  }


  async function removeDivision(rowKey: string) {
    const nextConfiguration = {
      ...configuration,
      eventOptions: sortDivisionsByEventAndName(
        removeBuilderRow(configuration.eventOptions, rowKey),
        configuration.eventFamilies,
        configuration.days
      )
    };
    await persistConfigurationDraft(
      nextConfiguration,
      "divisions",
      "Division removed from the private admin draft. Nothing public changed."
    );
  }

  async function keepPublishedValueForBlockedChange(raw: unknown) {
    const rawRecord = raw && typeof raw === "object" && !Array.isArray(raw)
      ? (raw as Record<string, unknown>)
      : {};
    const item = safeString(rawRecord.impact_id)
      ? communicationImpactDetail(raw)
      : blockedImpactDetail(raw);
    const actionKey = "impact_id" in item ? item.impact_id : item.block_id;
    if (item.entity_type !== "division" && !item.entity_label) {
      setMessage("Open the affected setup step and restore the published value before reviewing again.");
      return;
    }
    const published = publishedConfiguration.eventOptions.find((row) => {
      if (item.entity_id && safeString(row.value.id) === item.entity_id) return true;
      return item.entity_label && eventDivisionName(row.value).toLowerCase() === item.entity_label.toLowerCase();
    });
    const currentRow = configuration.eventOptions.find((row) => {
      if (item.entity_id && safeString(row.value.id) === item.entity_id) return true;
      return item.entity_label && eventDivisionName(row.value).toLowerCase() === item.entity_label.toLowerCase();
    });
    if (!published || !currentRow) {
      setMessage("The published Division could not be matched. Open Divisions and revert the affected fields manually.");
      return;
    }

    const publishedValue = published.value;
    let nextValue: SetupRecord = { ...currentRow.value };
    let nextFamilies = configuration.eventFamilies;
    let restoredParentRelationship = false;
    let restoredParentDays: string[] = [];
    const currentFamilyName = eventFamilyName(currentRow.value);
    const publishedFamilyName = eventFamilyName(publishedValue);
    const currentParent = configuration.eventFamilies.find(
      (row) => eventFamilyName(row.value).toLowerCase() === currentFamilyName.toLowerCase()
    );
    const publishedParent = publishedConfiguration.eventFamilies.find(
      (row) => eventFamilyName(row.value).toLowerCase() === publishedFamilyName.toLowerCase()
    );

    if (publishedFamilyName && publishedFamilyName.toLowerCase() !== currentFamilyName.toLowerCase()) {
      // Preserve unrelated inherited age and schedule edits before moving the
      // Division back to its published parent Event.
      if (item.field !== "skill_age_rules" && currentParent && safeString(nextValue.age_policy_source).toUpperCase() !== "OVERRIDE") {
        nextValue = {
          ...nextValue,
          age_policy_source: "OVERRIDE",
          age_mode: eventFamilyAgeMode(currentParent.value),
          age_label: eventFamilyAgeLabel(currentParent.value),
          age_rules: eventFamilyAgeRules(currentParent.value)
        };
      }
      if (item.field !== "registration_day_id" && item.field !== "scheduled_day_ids") {
        nextValue = setEventDayReferences(
          { ...nextValue, schedule_mode: "CUSTOM" },
          eventDayReferences(currentRow.value)
        );
      }
      nextValue.event_family_label = publishedFamilyName;
      nextValue.event_family = publishedFamilyName;
      nextValue.event_type = publishedValue.event_type;
      nextValue.participant_type = publishedValue.event_type;
      nextValue.gender_restriction = publishedValue.gender_restriction;
      nextValue.competition_format = publishedValue.competition_format || "STANDARD";
      for (const field of [
        "team_roster_size",
        "team_gender_rule",
        "team_tiebreak_mode",
        "team_playoff_format",
        "team_allow_substitutes"
      ] as const) {
        if (publishedValue[field] != null) nextValue[field] = publishedValue[field];
      }
      restoredParentRelationship = true;

      const existingTargetParent = nextFamilies.find(
        (row) => eventFamilyName(row.value).toLowerCase() === publishedFamilyName.toLowerCase()
      );
      if (!existingTargetParent) {
        const fallbackParent: SetupRecord = {
          ...newEventFamilyRow(nextFamilies.length + 1, publishedFamilyName),
          event_family: publishedFamilyName,
          participant_type: safeString(publishedValue.event_type) || "GENDER_DOUBLES",
          gender_restriction: safeString(publishedValue.event_type).toUpperCase() === "MIXED_DOUBLES"
            ? "MIXED"
            : "ANY",
          default_format: safeString(publishedValue.event_format_default) || "ROUND_ROBIN_PLUS_PLAYOFF",
          default_scoring: safeString(publishedValue.scoring_default) || "GAME_TO_15",
          default_capacity_teams: Number(publishedValue.capacity_teams) || 16,
          default_price_usd: Number(publishedValue.price_usd) || 0,
          default_waitlist: recordBoolean(publishedValue.waitlist_enabled, true),
          default_partner_board: recordBoolean(publishedValue.partner_board_enabled, true),
          default_age_mode: safeString(publishedValue.age_mode) || "ALL_AGES",
          default_age_label: safeString(publishedValue.age_label) || "All Ages",
          default_age_rules: publishedValue.age_rules || { mode: safeString(publishedValue.age_mode) || "ALL_AGES" }
        };
        nextFamilies = appendBuilderRow(
          nextFamilies,
          "family",
          publishedParent ? { ...publishedParent.value } : fallbackParent
        );
      }
    }

    if (item.field === "registration_day_id" || item.field === "scheduled_day_ids") {
      nextValue = setEventDayReferences(nextValue, eventDayReferences(publishedValue));
      nextValue.schedule_mode = "CUSTOM";
    } else if (item.field === "event_type") {
      nextValue.event_type = publishedValue.event_type;
      nextValue.participant_type = publishedValue.event_type;
      nextValue.competition_format = publishedValue.competition_format || "STANDARD";
      for (const field of [
        "team_roster_size",
        "team_gender_rule",
        "team_tiebreak_mode",
        "team_playoff_format",
        "team_allow_substitutes"
      ] as const) {
        nextValue[field] = publishedValue[field];
      }
    } else if (item.field === "gender_restriction") {
      nextValue.gender_restriction = publishedValue.gender_restriction;
    } else if (item.field === "skill_age_rules") {
      for (const field of [
        "skill_label",
        "skill_mode",
        "eligibility_mode",
        "skill_min_rating",
        "skill_max_rating",
        "combined_rating_cap",
        "age_label",
        "age_mode",
        "age_rules"
      ] as const) {
        nextValue[field] = publishedValue[field];
      }
      nextValue.age_policy_source = "OVERRIDE";
    } else if (item.field === "capacity_teams") {
      nextValue.capacity_teams = publishedValue.capacity_teams;
    } else {
      setMessage("This impact does not yet support a field-level revert. Open Divisions and restore the published field manually.");
      return;
    }

    const targetFamilyName = eventFamilyName(nextValue);
    let targetFamily = nextFamilies.find(
      (row) => eventFamilyName(row.value).toLowerCase() === targetFamilyName.toLowerCase()
    );
    if (!targetFamily) {
      setMessage("The required parent Event could not be restored. Open Events and restore the published parent relationship manually.");
      return;
    }

    // Keep-published is dependency-aware: a child Division cannot be restored
    // to a published gender/participant shape while its parent Event still
    // rejects that shape. Preserve unrelated Event edits, but restore the
    // minimum parent fields needed to make the published Division valid.
    const publishedParentValue = publishedParent?.value || {};
    const desiredParticipantType = safeString(
      publishedParentValue.participant_type || publishedValue.event_type || nextValue.event_type
    ).toUpperCase();
    const desiredDivisionGender = safeString(
      publishedValue.gender_restriction || nextValue.gender_restriction
    ).toUpperCase();
    const currentParentGender = safeString(targetFamily.value.gender_restriction || "ANY").toUpperCase();
    const publishedParentGender = safeString(publishedParentValue.gender_restriction).toUpperCase();
    const parentGenderCompatible = (
      currentParentGender === "ANY"
      || currentParentGender === desiredDivisionGender
      || (desiredParticipantType === "MIXED_DOUBLES" && currentParentGender === "MIXED")
    );
    const requiredParentGender = publishedParentGender
      || (desiredParticipantType === "MIXED_DOUBLES" ? "MIXED" : parentGenderCompatible ? currentParentGender : "ANY");
    let nextTargetFamilyValue: SetupRecord = { ...targetFamily.value };
    let targetParentChanged = false;
    if (desiredParticipantType && safeString(nextTargetFamilyValue.participant_type).toUpperCase() !== desiredParticipantType) {
      nextTargetFamilyValue.participant_type = desiredParticipantType;
      targetParentChanged = true;
    }
    if (requiredParentGender && safeString(nextTargetFamilyValue.gender_restriction || "ANY").toUpperCase() !== requiredParentGender) {
      nextTargetFamilyValue.gender_restriction = requiredParentGender;
      targetParentChanged = true;
    }
    if (publishedParentValue.competition_format != null
      && nextTargetFamilyValue.competition_format !== publishedParentValue.competition_format) {
      nextTargetFamilyValue.competition_format = publishedParentValue.competition_format;
      targetParentChanged = true;
    }
    for (const field of [
      "team_roster_size",
      "team_gender_rule",
      "team_tiebreak_mode",
      "team_playoff_format",
      "team_allow_substitutes"
    ] as const) {
      if (publishedParentValue[field] != null && nextTargetFamilyValue[field] !== publishedParentValue[field]) {
        nextTargetFamilyValue[field] = publishedParentValue[field];
        targetParentChanged = true;
      }
    }
    if (targetParentChanged) {
      nextFamilies = sortEventFamiliesByTournamentDay(
        replaceBuilderRow(nextFamilies, targetFamily.key, nextTargetFamilyValue),
        configuration.days
      );
      targetFamily = nextFamilies.find((row) => row.key === targetFamily?.key) || targetFamily;
      restoredParentRelationship = true;
    }

    const requiredSchedule = eventDayReferences(nextValue);
    const familyDays = eventDayReferences(targetFamily.value);
    const dayOrder = new Map(
      configuration.days.map((row, index) => [dayReference(row.value), index] as const)
    );
    const mergedFamilyDays = [...new Set([...familyDays, ...requiredSchedule])]
      .sort((left, right) => (dayOrder.get(left) ?? 9999) - (dayOrder.get(right) ?? 9999));
    restoredParentDays = mergedFamilyDays.filter((dayId) => !familyDays.includes(dayId));
    if (restoredParentDays.length) {
      nextFamilies = sortEventFamiliesByTournamentDay(
        replaceBuilderRow(
          nextFamilies,
          targetFamily.key,
          setEventDayReferences(targetFamily.value, mergedFamilyDays)
        ),
        configuration.days
      );
    }

    const nextConfiguration: SetupConfiguration = {
      ...configuration,
      eventFamilies: nextFamilies,
      eventOptions: sortDivisionsByEventAndName(
        replaceBuilderRow(configuration.eventOptions, currentRow.key, nextValue),
        nextFamilies,
        configuration.days
      )
    };
    const forced = { ...objectValue(settings.forced_change_resolutions) };
    const communications = { ...objectValue(settings.communication_change_acknowledgements) };
    delete forced[actionKey];
    delete communications[actionKey];
    const nextSettings = {
      ...settings,
      forced_change_resolutions: forced,
      communication_change_acknowledgements: communications
    };
    const parentMessages: string[] = [];
    if (restoredParentRelationship) parentMessages.push(`restored the published parent Event ${publishedFamilyName || targetFamilyName}`);
    if (restoredParentDays.length) {
      parentMessages.push(
        `The parent Event also regained ${restoredParentDays.length} required tournament day${restoredParentDays.length === 1 ? "" : "s"}`
      );
    }
    const dependencyMessage = parentMessages.length
      ? ` The action also ${parentMessages.join(" and ")} so the Division remains valid.`
      : "";
    const saved = await persistConfigurationDraft(
      nextConfiguration,
      "review",
      `Restored the published ${humanReviewFieldLabel(item.field || "value")} for ${item.entity_label || "the affected Division"}.${dependencyMessage} Other draft changes were preserved.`,
      nextSettings
    );
    if (saved) {
      setResolutionDraftDirty(false);
      setImpactReview(null);
      setReviewedDraftSignature("");
      autoReviewSignatureRef.current = "";
    }
  }

  function communicationAcknowledgementPlans(): Record<string, unknown> {
    return objectValue(settings.communication_change_acknowledgements);
  }

  function communicationAcknowledgementPlan(item: CommunicationImpactDetail): Record<string, unknown> | null {
    const plan = communicationAcknowledgementPlans()[item.impact_id];
    return plan && typeof plan === "object" && !Array.isArray(plan)
      ? (plan as Record<string, unknown>)
      : null;
  }

  function updateCommunicationAcknowledgement(
    item: CommunicationImpactDetail,
    patch: Record<string, unknown>
  ) {
    setSettings((current) => {
      const plans = { ...objectValue(current.communication_change_acknowledgements) };
      const existing = plans[item.impact_id] && typeof plans[item.impact_id] === "object" && !Array.isArray(plans[item.impact_id])
        ? (plans[item.impact_id] as Record<string, unknown>)
        : {};
      const next = {
        impact_id: item.impact_id,
        impact_type: item.impact_type,
        entity_type: item.entity_type,
        entity_id: item.entity_id,
        entity_label: item.entity_label,
        field: item.field,
        current_value: item.current_value,
        proposed_value: item.proposed_value,
        affected_registrations: item.affected_registrations,
        action: safeString(existing.action),
        notes: safeString(existing.notes),
        acknowledged: Boolean(existing.acknowledged),
        status: safeString(existing.status) || "IN_PROGRESS",
        ...patch
      } as Record<string, unknown>;
      if (Object.prototype.hasOwnProperty.call(patch, "action") || Object.prototype.hasOwnProperty.call(patch, "notes")) {
        next.acknowledged = false;
        next.status = "IN_PROGRESS";
        next.acknowledged_at = null;
      }
      if (Object.prototype.hasOwnProperty.call(patch, "acknowledged")) {
        const action = safeString(next.action).toUpperCase();
        const acknowledged = Boolean(patch.acknowledged)
          && !item.requires_data_completion
          && item.data_completion_registrations.length === 0
          && ["NOTIFY_AFFECTED", "ACKNOWLEDGE_NO_NOTICE"].includes(action);
        next.acknowledged = acknowledged;
        next.status = acknowledged ? "ACKNOWLEDGED" : "IN_PROGRESS";
        next.acknowledged_at = acknowledged ? new Date().toISOString() : null;
      }
      plans[item.impact_id] = next;
      return { ...current, communication_change_acknowledgements: plans };
    });
    setResolutionDraftDirty(true);
    setReviewedDraftSignature("");
  }

  function communicationAcknowledgementComplete(item: CommunicationImpactDetail): boolean {
    if (!item.requires_acknowledgement) return true;
    const plan = communicationAcknowledgementPlan(item);
    const action = safeString(plan?.action).toUpperCase();
    return !item.requires_data_completion
      && item.data_completion_registrations.length === 0
      && Boolean(plan?.acknowledged)
      && safeString(plan?.status).toUpperCase() === "ACKNOWLEDGED"
      && ["NOTIFY_AFFECTED", "ACKNOWLEDGE_NO_NOTICE"].includes(action);
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
        const action = safeString(next.action).toUpperCase();
        const noteRequired = action === "OTHER";
        next.resolved = Boolean(patch.resolved)
          && Boolean(action)
          && (!noteRequired || Boolean(safeString(next.notes)));
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
    return rows.length > 0 && rows.every((row) => {
      const action = safeString(row.action).toUpperCase();
      return Boolean(row.resolved)
        && Boolean(action)
        && (action !== "OTHER" || Boolean(safeString(row.notes)));
    });
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
    const venueValidation = venueIssues(settings, configuration);
    if (venueValidation.length) {
      setMessage(venueValidation[0]);
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
      setMessage("Review actions saved to the unpublished tournament draft. The impact review is refreshing against the saved resolutions and communication acknowledgements.");
    } catch (error) {
      if (actionRequest.isCurrent(generation)) {
        setMessage(error instanceof Error ? error.message : "Unable to save the Review actions.");
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

  async function publishSetup(confirmationText: string): Promise<ActionSuccess> {
    function rejectPublish(message: string): never {
      setMessage(message);
      throw new Error(message);
    }

    if (!detail) {
      rejectPublish("Reload the tournament setup before publishing it.");
    }
    if (
      !impactReview ||
      reviewedDraftSignature !== fullDraftSignature(basics, settings, configuration)
    ) {
      rejectPublish("Review the current setup before publishing it.");
    }
    if (resolutionDraftDirty) {
      rejectPublish("Save the Review actions before publishing.");
    }
    const impact = impactReview.publish_impact || {};
    const rawBlockedDetails = Array.isArray(impact.blocked_details) && impact.blocked_details.length
      ? impact.blocked_details
      : (Array.isArray(impact.blocked) ? impact.blocked : []);
    const unresolved = rawBlockedDetails
      .map(blockedImpactDetail)
      .filter((item) => !forcedResolutionComplete(item));
    if (unresolved.length) {
      rejectPublish(`Resolve ${unresolved.length} blocked change${unresolved.length === 1 ? "" : "s"} before publishing.`);
    }
    const rawCommunicationDetails = Array.isArray(impact.communication_impact_details)
      ? impact.communication_impact_details
      : [];
    const unresolvedCommunications = rawCommunicationDetails
      .map(communicationImpactDetail)
      .filter((item) => !communicationAcknowledgementComplete(item));
    if (unresolvedCommunications.length) {
      rejectPublish(`Acknowledge ${unresolvedCommunications.length} registration-preserving communication impact${unresolvedCommunications.length === 1 ? "" : "s"} before publishing.`);
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
      const publishResult = await requestJson<WriteResponse>(
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
      const registrationWasOpen = safeString(settings.registration_status).toLowerCase() === "open";
      const completion = actionSuccess(
        publishResult.activated_from_draft
          ? "Tournament published and activated"
          : "Tournament published",
        publishResult.activated_from_draft
          ? "The reviewed setup is published, and the tournament is now active on the public site. Registration status was left unchanged."
          : "The reviewed tournament setup is now published. Its existing lifecycle and registration status were left unchanged.",
        "Done"
      );
      if (!actionRequest.isCurrent(generation)) return completion;
      setSetupPublishedThisSession(true);
      await loadDetail();
      if (actionRequest.isCurrent(generation)) {
        setMessage(
          publishResult.activated_from_draft
            ? `Tournament setup published and tournament activated. Registration ${registrationWasOpen ? "remains open." : "can now be opened."}`
            : `Tournament setup published. Registration ${registrationWasOpen ? "remains open." : "can now be opened."}`
        );
      }
      return completion;
    } catch (error) {
      const publishError = error instanceof Error ? error : new Error("Unable to publish setup.");
      if (actionRequest.isCurrent(generation)) {
        setMessage(publishError.message);
      }
      throw publishError;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function openRegistration(confirmationText: string) {
    if (!detail) throw new Error("Reload the tournament before opening registration.");
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
      const completion = actionSuccess("Registration opened", "The published tournament is now available to registrants.");
      if (actionRequest.isCurrent(generation)) {
        router.push(
          tournamentRouteHref("/admin/tournaments/registration", {
            tournamentId,
            tournamentName: basics.name.trim() || tournamentName,
            drawId
          })
        );
      }
      return completion;
    } catch (error) {
      if (actionRequest.isCurrent(generation)) {
        setMessage(
          error instanceof Error ? error.message : "Unable to open registration."
        );
      }
      throw error;
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
  const publicationStatus = setupPublicationStatus({
    detailLoadState,
    hasAuthoritativeDetail: Boolean(detail),
    hasUnpublishedChanges
  });
  const publicationBanner = publicationBanners[publicationStatus];
  const publishedSetupState = setupState(
    publishedBasics,
    publishedSettings,
    publishedConfiguration
  );
  const publishedSetupReady = publishedSetupState.review === "in-progress";
  const registrationCanOpen = Boolean(
    publicationStatus === "current" &&
      publishedSetupReady
  );
  const registrationStatus = safeString(publishedSettings.registration_status || "draft");
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

        <TournamentSponsorEditor
          sponsors={basics.sponsors}
          tournamentName={basics.name}
          disabled={busy}
          onUpload={(image_base64) => requestJson(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/setup/tournaments/${encodeURIComponent(tournamentId)}/sponsor-logos`, { method: "POST", body: JSON.stringify({ image_base64 }) })}
          onSave={(sponsors) => persistConfigurationDraft(configuration, "basics", "Sponsor draft saved. Publish from Review when you’re ready.", settings, { ...basics, sponsors }, true)}
        />

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
        onAcknowledge={() => setEventDialogKey(undefined)}
      />
      <TournamentDivisionPresetDialog
        open={divisionPresetFamilyKey !== undefined}
        family={configuration.eventFamilies.find((row) => row.key === divisionPresetFamilyKey) || null}
        configuration={configuration}
        onCancel={() => setDivisionPresetFamilyKey(undefined)}
        onConfirm={saveGeneratedDivisions}
        onAcknowledge={() => setDivisionPresetFamilyKey(undefined)}
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
              onGenerateDivisions={() => setDivisionPresetFamilyKey(row.key)}
              onRemove={() => void removeEventFamily(row.key)}
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
                          {bracket.count} entr{bracket.count === 1 ? "y" : "ies"}
                          {bracket.provisional_count ? ` (${bracket.provisional_count} provisional)` : ""}
                          {` · ${bracket.viable ? "Create" : "Below minimum"}`}
                        </article>
                      ))}
                    </div>
                    {preview.recommendations.length ? (
                      <ul>
                        {preview.recommendations.map((recommendation) => <li key={recommendation}>{recommendation}</li>)}
                      </ul>
                    ) : null}
                    {preview.pending_entries?.length ? (
                      <div style={{ marginTop: "0.65rem", padding: "0.65rem", border: "1px solid #bfdbfe", borderRadius: "10px", background: "#eff6ff" }}>
                        <strong>Entries awaiting partner-based placement</strong>
                        <ul style={{ marginBottom: 0 }}>
                          {preview.pending_entries.map((entry, index) => (
                            <li key={safeString(entry.selection_id) || safeString(entry.registration_id) || index}>
                              {safeString(entry.display_name) || `Entry ${index + 1}`}
                              {" — placement remains open and will be recalculated when a partner is assigned"}
                            </li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                    {preview.unassigned_entries.length ? (
                      <div style={{ marginTop: "0.65rem", padding: "0.65rem", border: "1px solid #fecaca", borderRadius: "10px", background: "#fef2f2" }}>
                        <strong>Entries needing manual resolution</strong>
                        <ul style={{ marginBottom: 0 }}>
                          {preview.unassigned_entries.map((entry, index) => (
                            <li key={safeString(entry.selection_id) || safeString(entry.registration_id) || index}>
                              {safeString(entry.display_name) || `Entry ${index + 1}`}
                              {safeString(entry.assignment_issue) ? ` — ${safeString(entry.assignment_issue)}` : " — age information is incomplete or outside the proposed policy"}
                            </li>
                          ))}
                        </ul>
                      </div>
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
            href={tournamentSetupStepHref("schedule", tournamentId, basics.name || tournamentName, drawId)}
            style={ghostButtonStyle}
          >
            Back
          </Link>
          <button
            type="button"
            style={buttonStyle}
            disabled={busy || !configuration.eventFamilies.length || familyIssues.length > 0}
            onClick={() => goTo("divisions")}
          >
            Continue to Divisions
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
  const selectedDivisions = configuration.eventOptions.filter((row) => selectedDivisionKeys.includes(row.key));
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
        onAcknowledge={() => setDivisionDialogKey(undefined)}
      />
      <TournamentDivisionBulkEditDialog
        open={bulkDivisionDialogOpen}
        divisions={selectedDivisions}
        eventFamilies={configuration.eventFamilies}
        days={configuration.days}
        disabled={busy}
        onCancel={() => setBulkDivisionDialogOpen(false)}
        onConfirm={saveBulkDivisionEdits}
        onAcknowledge={() => {
          setBulkDivisionDialogOpen(false);
          setBulkDivisionSelecting(false);
          setSelectedDivisionKeys([]);
        }}
      />

      <article style={cardStyle}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", alignItems: "flex-start", flexWrap: "wrap" }}>
          <div>
            <h2 style={{ marginTop: 0 }}>Competition · Divisions</h2>
            <p style={{ color: "#475569", marginBottom: 0 }}>
              Build the final competitive groups inside each event. Divisions inherit event structure, age policy, draw, and scoring by default; use explicit overrides only when a specific division needs different rules.
            </p>
          </div>
          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", alignItems: "center" }}>
            {bulkDivisionSelecting ? (
              <>
                <button
                  type="button"
                  style={ghostButtonStyle}
                  disabled={busy}
                  onClick={() => setSelectedDivisionKeys(
                    selectedDivisionKeys.length === configuration.eventOptions.length
                      ? []
                      : configuration.eventOptions.map((row) => row.key)
                  )}
                >
                  {selectedDivisionKeys.length === configuration.eventOptions.length ? "Clear all" : "Select all"}
                </button>
                <button
                  type="button"
                  style={buttonStyle}
                  disabled={busy || !selectedDivisionKeys.length}
                  onClick={() => setBulkDivisionDialogOpen(true)}
                >
                  Edit selected ({selectedDivisionKeys.length})
                </button>
                <button
                  type="button"
                  style={ghostButtonStyle}
                  disabled={busy}
                  onClick={() => {
                    setBulkDivisionSelecting(false);
                    setSelectedDivisionKeys([]);
                  }}
                >
                  Cancel bulk edit
                </button>
              </>
            ) : (
              <>
                <button
                  type="button"
                  style={ghostButtonStyle}
                  disabled={busy || !configuration.eventOptions.length}
                  onClick={() => {
                    setBulkDivisionSelecting(true);
                    setSelectedDivisionKeys([]);
                  }}
                >
                  Bulk edit
                </button>
                <button
                  type="button"
                  style={buttonStyle}
                  disabled={busy || !configuration.eventFamilies.length}
                  onClick={() => setDivisionDialogKey(null)}
                >
                  Add division
                </button>
              </>
            )}
          </div>
        </div>
        {!configuration.eventFamilies.length ? (
          <p role="alert" style={{ color: "#b91c1c" }}>
            Create an Event and event policy before adding divisions.
          </p>
        ) : null}
      </article>

      {sortedDivisions.map((row, index) => {
        const originalIndex = configuration.eventOptions.findIndex((candidate) => candidate.key === row.key);
        const selected = selectedDivisionKeys.includes(row.key);
        return (
          <div
            key={row.key}
            id={`division-${safeString(row.value.id)}`}
            style={bulkDivisionSelecting ? { border: `2px solid ${selected ? "#2563eb" : "#cbd5e1"}`, borderRadius: "14px", padding: "0.45rem", background: selected ? "#eff6ff" : "transparent" } : undefined}
          >
            {bulkDivisionSelecting ? (
              <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", padding: "0.25rem 0.25rem 0.55rem", fontWeight: 800 }}>
                <input
                  type="checkbox"
                  checked={selected}
                  disabled={busy}
                  onChange={(event) => setSelectedDivisionKeys((current) =>
                    event.target.checked
                      ? [...current, row.key]
                      : current.filter((key) => key !== row.key)
                  )}
                />
                Select {eventDivisionName(row.value) || `division ${index + 1}`} for bulk editing
              </label>
            ) : null}
            <TournamentSetupDivisionCard
              row={row}
              position={index}
              eventFamilies={configuration.eventFamilies}
              days={configuration.days}
              disabled={busy || bulkDivisionSelecting}
              issues={issuesForPath(issues, `events.${Math.max(0, originalIndex)}`)}
              onEdit={() => setDivisionDialogKey(row.key)}
              onRemove={() => void removeDivision(row.key)}
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
            href={tournamentSetupStepHref("events", tournamentId, basics.name || tournamentName, drawId)}
            style={ghostButtonStyle}
          >
            Back
          </Link>
          <button
            type="button"
            style={buttonStyle}
            disabled={busy || !configuration.eventOptions.length || divisionIssues.length > 0}
            onClick={() => goTo("pricing")}
          >
            Continue to Commerce
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
              href={tournamentSetupStepHref("divisions", tournamentId, basics.name || tournamentName, drawId)}
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
    const courts = venueCourts(settings, configuration);
    const titledCourts = courts.map((court) => court.title.trim()).filter(Boolean);
    const duplicateCourtTitles = new Set(
      titledCourts
        .map((label) => label.toLowerCase())
        .filter((label, index, values) => values.indexOf(label) !== index)
    );
    const venueValidation = venueIssues(settings, configuration);

    function updateCourts(nextCourts: VenueCourt[]) {
      const patch = settingsWithVenueCourts({}, nextCourts);
      updateVenueSettings(patch);
    }

    function updateDayCourts(rowKey: string, selectedIds: readonly string[]) {
      setConfiguration((current) => ({
        ...current,
        days: current.days.map((row) =>
          row.key === rowKey
            ? { ...row, value: withVenueCourtAvailability(row.value, courts, selectedIds) }
            : row
        )
      }));
      setImpactReview(null);
    }

    return (
      <div style={{ display: "grid", gap: "1rem" }}>
        <TournamentBulkAddCourtsDialog
          open={bulkCourtDialogOpen}
          existingCount={courts.length}
          disabled={busy}
          onCancel={() => setBulkCourtDialogOpen(false)}
          onConfirm={async (count) => {
            updateCourts([
              ...courts,
              ...Array.from({ length: count }, () => newVenueCourt())
            ]);
            return actionSuccess("Courts added", `${count} court${count === 1 ? " was" : "s were"} added to the unpublished venue inventory.`);
          }}
          onAcknowledge={() => setBulkCourtDialogOpen(false)}
        />
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Tournament · Venue and tournament days</h2>
          <p style={{ color: "#475569", marginBottom: 0 }}>
            Store the venue once, maintain a stable court inventory, and choose the exact courts available on each tournament day. Event and division scheduling controls play start times, so tournament-level court hours are intentionally not collected here.
          </p>
        </article>

        <article style={cardStyle}>
          <h3 style={{ marginTop: 0 }}>Venue</h3>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))", gap: "0.75rem" }}>
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
              <strong>Venue address</strong><br />
              <input
                value={safeString(settings.venue_address)}
                placeholder="Street, city, state/province, postal code, country"
                disabled={busy}
                style={inputStyle}
                onChange={(event) => updateVenueSettings({ venue_address: event.target.value })}
              />
              <small>Required for public directions and map links.</small>
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
            <label style={{ gridColumn: "1 / -1" }}>
              <strong>Directions to the venue (optional)</strong><br />
              <textarea
                value={safeString(settings.venue_directions)}
                placeholder="Parking, gate, building, check-in, or arrival instructions"
                disabled={busy}
                style={{ ...inputStyle, minHeight: "84px" }}
                onChange={(event) => updateVenueSettings({ venue_directions: event.target.value })}
              />
            </label>
          </div>
        </article>

        <article style={cardStyle}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", alignItems: "center", flexWrap: "wrap" }}>
            <div>
              <h3 style={{ margin: 0 }}>Venue court inventory · {courts.length} total</h3>
              <p style={{ margin: "0.25rem 0 0", color: "#64748b" }}>
                Every court has a stable identity. Titles are optional; an untitled court remains available by number. Only Remove court deletes it.
              </p>
            </div>
            <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", alignItems: "center" }}>
              <output
                aria-label="Total venue courts"
                style={{ padding: "0.55rem 0.75rem", border: "1px solid #cbd5e1", borderRadius: "999px", background: "#f8fafc", fontWeight: 800 }}
              >
                Total venue courts: {courts.length}
              </output>
              <small style={{ color: "#64748b" }}>This read-only count is derived from the court inventory.</small>
              <button
                type="button"
                style={ghostButtonStyle}
                disabled={busy || courts.length >= 100}
                onClick={() => setBulkCourtDialogOpen(true)}
              >
                Bulk add courts
              </button>
              <button
                type="button"
                style={ghostButtonStyle}
                disabled={busy || courts.length >= 100}
                onClick={() => updateCourts([...courts, newVenueCourt()])}
              >
                Add court
              </button>
            </div>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "0.65rem", marginTop: "0.75rem" }}>
            {courts.map((court, index) => {
              const usedDays = configuration.days.filter((row) => venueDayAvailableCourtIds(row.value, courts).includes(court.id));
              const duplicate = court.title.trim() && duplicateCourtTitles.has(court.title.trim().toLowerCase());
              return (
                <article key={court.id} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.65rem", background: "#f8fafc" }}>
                  <label>
                    <strong>Court {index + 1} title (optional)</strong><br />
                    <input
                      value={court.title}
                      placeholder={`Court ${index + 1}`}
                      disabled={busy}
                      style={{ ...inputStyle, borderColor: duplicate ? "#ef4444" : "#cbd5e1" }}
                      onChange={(event) => updateCourts(courts.map((row) => row.id === court.id ? { ...row, title: event.target.value } : row))}
                    />
                  </label>
                  <small style={{ color: "#64748b" }}>Displayed as {courtDisplayName(court, index)} when left blank.</small>
                  <div style={{ marginTop: "0.55rem" }}>
                    <ConfirmAction
                      triggerLabel="Remove court"
                      title={`Remove ${courtDisplayName(court, index)}?`}
                      description={usedDays.length
                        ? `This court is currently available on ${usedDays.length} tournament day${usedDays.length === 1 ? "" : "s"}. Removing it updates those unpublished day selections. Published data remains unchanged until Review.`
                        : "This removes the court from the unpublished venue inventory. Published data remains unchanged until Review."}
                      confirmLabel="Yes, remove court"
                      cancelLabel="No, keep court"
                      confirmationText=""
                      tone="danger"
                      disabled={busy || courts.length <= 1}
                      onConfirm={async () => {
                        updateCourts(courts.filter((row) => row.id !== court.id));
                        return actionSuccess("Court removed", `${courtDisplayName(court, index)} was removed from the unpublished venue inventory.`);
                      }}
                    />
                  </div>
                </article>
              );
            })}
          </div>
          {duplicateCourtTitles.size ? <p role="alert" style={{ color: "#b91c1c" }}>Optional court titles must be unique.</p> : null}
        </article>

        <article style={cardStyle}>
          <h3 style={{ marginTop: 0 }}>Tournament days</h3>
          <p style={{ color: "#64748b" }}>
            Dates are generated automatically from the tournament start and end dates. Edit the public day label and choose all venue courts or an exact subset available that day.
          </p>
          <div style={{ display: "grid", gap: "0.75rem" }}>
            {configuration.days.map((row, index) => {
              const selectedIds = venueDayAvailableCourtIds(row.value, courts);
              const allCourts = selectedIds.length === courts.length && courts.every((court) => selectedIds.includes(court.id));
              return (
                <article key={row.key} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: "#f8fafc" }}>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
                    <label>
                      <strong>Day {index + 1} label</strong><br />
                      <input
                        value={editableString(row.value.label)}
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
                    <label>
                      <strong>Available courts this day</strong><br />
                      <input
                        type="number"
                        min="1"
                        max={courts.length}
                        step="1"
                        value={selectedIds.length}
                        disabled={busy}
                        style={inputStyle}
                        onChange={(event) => {
                          const requested = Math.max(1, Math.min(courts.length, Math.trunc(Number(event.target.value) || 1)));
                          const kept = selectedIds.slice(0, requested);
                          const additions = courts.map((court) => court.id).filter((id) => !kept.includes(id)).slice(0, requested - kept.length);
                          updateDayCourts(row.key, [...kept, ...additions]);
                        }}
                      />
                      <small>Maximum {courts.length} venue courts.</small>
                    </label>
                  </div>
                  <fieldset style={{ marginTop: "0.75rem", padding: "0.7rem", border: "1px solid #cbd5e1", borderRadius: "10px" }}>
                    <legend style={{ fontWeight: 800 }}>Which courts are available?</legend>
                    <label style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
                      <input
                        type="checkbox"
                        checked={allCourts}
                        disabled={busy}
                        onChange={(event) => updateDayCourts(
                          row.key,
                          event.target.checked ? courts.map((court) => court.id) : selectedIds.slice(0, Math.max(1, selectedIds.length - 1))
                        )}
                      />
                      Use all venue courts
                    </label>
                    {!allCourts ? (
                      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: "0.4rem", marginTop: "0.55rem" }}>
                        {courts.map((court, courtIndex) => (
                          <label key={court.id} style={{ display: "flex", gap: "0.45rem", alignItems: "center" }}>
                            <input
                              type="checkbox"
                              checked={selectedIds.includes(court.id)}
                              disabled={busy || (selectedIds.length === 1 && selectedIds.includes(court.id))}
                              onChange={(event) => updateDayCourts(
                                row.key,
                                event.target.checked
                                  ? [...selectedIds, court.id]
                                  : selectedIds.filter((id) => id !== court.id)
                              )}
                            />
                            {courtDisplayName(court, courtIndex)}
                          </label>
                        ))}
                      </div>
                    ) : null}
                  </fieldset>
                </article>
              );
            })}
          </div>
        </article>

        {dayIssues.length || duplicateCourtTitles.size || venueValidation.length ? (
          <article style={{ ...cardStyle, borderColor: "#fecaca" }}>
            <h3 style={{ marginTop: 0 }}>Items to fix before continuing</h3>
            <ul>
              {venueValidation.map((issue) => <li key={issue}>{issue}</li>)}
              {dayIssues.map((issue) => <li key={`${issue.path}-${issue.message}`}>{issue.message}</li>)}
              {duplicateCourtTitles.size ? <li>Optional court titles must be unique.</li> : null}
            </ul>
          </article>
        ) : null}

        {footerRow(
          <>
            <Link href={tournamentSetupStepHref("basics", tournamentId, basics.name || tournamentName, drawId)} style={ghostButtonStyle}>Back to Basics</Link>
            <button
              type="button"
              style={buttonStyle}
              disabled={busy || !configuration.days.length || dayIssues.length > 0 || duplicateCourtTitles.size > 0 || venueValidation.length > 0}
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
    const communicationDetails = (Array.isArray(impact.communication_impact_details)
      ? impact.communication_impact_details
      : []).map(communicationImpactDetail);
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
    const unresolvedCommunications = communicationDetails.filter((item) => !communicationAcknowledgementComplete(item));
    const forcePlans = blockedDetails.filter((item) => Boolean(forcedResolutionPlan(item)));
    const communicationPlans = communicationDetails.filter((item) => Boolean(communicationAcknowledgementPlan(item)));
    const hasReviewActionPlans = forcePlans.length > 0 || communicationPlans.length > 0;

    const valueChanged = (current: unknown, proposed: unknown) =>
      JSON.stringify(current ?? null) !== JSON.stringify(proposed ?? null);
    const publishedCourts = venueCourts(publishedSettings, publishedConfiguration);
    const draftCourts = venueCourts(settings, configuration);
    const dayComparisonRows = (rows: typeof configuration.days, courts: VenueCourt[]) => rows.map((row) => ({
      id: dayReference(row.value),
      label: dayLabel(row.value),
      event_date: row.value.event_date,
      available_courts: venueDayAvailableCourtIds(row.value, courts).map((id) => {
        const index = courts.findIndex((court) => court.id === id);
        return index >= 0 ? courtDisplayName(courts[index], index) : id;
      })
    }));
    const comparisons = [
      { field: "Tournament name", current: publishedBasics.name, proposed: basics.name },
      {
        field: "Tournament dates",
        current: { start_date: publishedBasics.startDate, end_date: publishedBasics.endDate },
        proposed: { start_date: basics.startDate, end_date: basics.endDate }
      },
      {
        field: "Venue",
        current: {
          name: publishedBasics.locationName,
          address: publishedSettings.venue_address,
          directions: publishedSettings.venue_directions,
          timezone: publishedBasics.timezone
        },
        proposed: {
          name: basics.locationName,
          address: settings.venue_address,
          directions: settings.venue_directions,
          timezone: basics.timezone
        }
      },
      {
        field: "Venue courts",
        current: publishedCourts,
        proposed: draftCourts
      },
      {
        field: "Tournament days",
        current: dayComparisonRows(publishedConfiguration.days, publishedCourts),
        proposed: dayComparisonRows(configuration.days, draftCourts)
      },
      {
        field: "Registration window",
        current: {
          registration_open_at: publishedSettings.registration_open_at,
          registration_close_at: publishedSettings.registration_close_at
        },
        proposed: {
          registration_open_at: settings.registration_open_at,
          registration_close_at: settings.registration_close_at
        }
      },
      {
        field: "Sponsors",
        current: publishedBasics.sponsors,
        proposed: basics.sponsors
      },
      {
        field: "Events and policies",
        current: publishedConfiguration.eventFamilies.map((row) => ({
          id: row.value.id,
          event: eventFamilyName(row.value),
          format: safeString(row.value.competition_format) === "FOUR_PLAYER_TEAM"
            ? "Four-player team"
            : safeString(row.value.participant_type),
          age: agePolicySummary(readAgePolicy(row.value, EVENT_AGE_POLICY_FIELDS)),
          days: eventDayReferences(row.value)
        })),
        proposed: configuration.eventFamilies.map((row) => ({
          id: row.value.id,
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
          id: row.value.id,
          division: eventDivisionName(row.value),
          event: eventFamilyName(row.value),
          skill: row.value.skill_label,
          age: row.value.age_label,
          fee: row.value.price_usd
        })),
        proposed: configuration.eventOptions.map((row) => ({
          id: row.value.id,
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
        complete: ready && unresolvedBlockers.length === 0 && unresolvedCommunications.length === 0,
        draft: impactReview
          ? `${warnings.length} warning(s) · ${unresolvedBlockers.length} registration blocker(s) · ${unresolvedCommunications.length} communication impact(s)`
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
                href={tournamentSetupStepHref(item.key, tournamentId, basics.name || tournamentName, drawId)}
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
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "0.65rem" }}>
            <div><strong>Tournament</strong><br />{basics.name || "Untitled tournament"}</div>
            <div><strong>Dates</strong><ReviewValueDisplay field="Tournament dates" value={{ start_date: basics.startDate, end_date: basics.endDate }} days={configuration.days} timezone={basics.timezone} technical={false} /></div>
            <div><strong>Venue</strong><br />{basics.locationName || "Not set"}<br /><small>{safeString(settings.venue_address) || "No address"}</small>{safeString(settings.venue_directions) ? <><br /><small>{safeString(settings.venue_directions)}</small></> : null}</div>
            <div><strong>Courts</strong><br />{venueCourtCount(settings, configuration)} total</div>
            <div><strong>Registration window</strong><ReviewValueDisplay field="Registration window" value={{ registration_open_at: settings.registration_open_at, registration_close_at: settings.registration_close_at }} days={configuration.days} timezone={basics.timezone} technical={false} /></div>
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
                  <ReviewComparisonDisplay
                    field={comparison.field}
                    current={comparison.current}
                    proposed={comparison.proposed}
                    days={configuration.days}
                    timezone={basics.timezone}
                  />
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
              {communicationDetails.length ? (
                <>
                  <strong>Registration-preserving impacts — acknowledge before publishing</strong>
                  <p style={{ color: "#475569" }}>
                    These changes do not invalidate registrations. Review the affected registrants, complete any missing information, record one communication decision for the Division, and continue without a grandfather, move, cancellation, or refund queue.
                  </p>
                  <div style={{ display: "grid", gap: "0.75rem", marginTop: "0.55rem" }}>
                    {communicationDetails.map((item) => {
                      const plan = communicationAcknowledgementPlan(item);
                      const acknowledged = communicationAcknowledgementComplete(item);
                      const dataCompletionPending = item.requires_data_completion || item.data_completion_registrations.length > 0;
                      const editHref = item.step === "divisions"
                        ? `${tournamentSetupStepHref("divisions", tournamentId, basics.name || tournamentName, drawId)}&resolveDivision=${encodeURIComponent(item.entity_id || "")}`
                        : tournamentSetupStepHref(item.step || "review", tournamentId, basics.name || tournamentName, drawId);
                      return (
                        <article key={item.impact_id} style={{ padding: "0.8rem", border: `1px solid ${acknowledged ? "#bbf7d0" : dataCompletionPending ? "#fecaca" : "#fde68a"}`, borderRadius: "12px", background: acknowledged ? "#f0fdf4" : dataCompletionPending ? "#fef2f2" : "#fffbeb" }}>
                          <strong>{communicationImpactTitle(item)}</strong>
                          <p><strong>{item.entity_label || "Affected Division"}</strong><br />{item.message}</p>
                          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.65rem" }}>
                            <div style={{ padding: "0.6rem", borderRadius: "10px", background: "white", minWidth: 0, overflowWrap: "anywhere" }}>
                              <small>Current published {humanReviewFieldLabel(item.field || "value")}{item.current_source ? ` · ${item.current_source}` : ""}</small>
                              <ReviewValueDisplay field={item.field || "value"} value={item.current_value} days={configuration.days} timezone={basics.timezone} />
                            </div>
                            <div style={{ padding: "0.6rem", borderRadius: "10px", background: "#eff6ff", minWidth: 0, overflowWrap: "anywhere" }}>
                              <small>Proposed draft {humanReviewFieldLabel(item.field || "value")}{item.proposed_source ? ` · ${item.proposed_source}` : ""}</small>
                              <ReviewValueDisplay field={item.field || "value"} value={item.proposed_value} days={configuration.days} timezone={basics.timezone} />
                            </div>
                          </div>
                          <p style={{ color: dataCompletionPending ? "#9a3412" : "#166534", fontWeight: 800 }}>
                            {dataCompletionPending
                              ? `No known registration is ineligible. ${item.data_completion_registrations.length} registration${item.data_completion_registrations.length === 1 ? " needs" : "s need"} missing information before eligibility or preferred placement can be confirmed.`
                              : `All ${item.affected_registrations.length} affected registration${item.affected_registrations.length === 1 ? " remains" : "s remain"} valid. No registration-level cancellation or eligibility resolution is required.`}
                          </p>
                          {item.affected_registrations.length ? (
                            <details>
                              <summary style={{ cursor: "pointer", fontWeight: 800 }}>Affected registrants ({item.affected_registrations.length})</summary>
                              <ul style={{ marginBottom: 0 }}>
                                {item.affected_registrations.map((registration) => {
                                  const summary = affectedRegistrationImpactSummary(registration, item);
                                  return (
                                    <li key={`${registration.registration_id}-${registration.selection_id || "registration"}`}>
                                      {registration.display_name || registration.email || "Registration needs details"}{registration.email && registration.display_name ? ` · ${registration.email}` : ""}{summary ? ` — ${summary}` : ""}
                                    </li>
                                  );
                                })}
                              </ul>
                            </details>
                          ) : null}
                          {item.data_completion_registrations.length ? (
                            <article style={{ marginTop: "0.7rem", padding: "0.7rem", border: "1px solid #fecaca", borderRadius: "10px", background: "#fff7ed" }}>
                              <strong>Required eligibility information</strong>
                              <p style={{ margin: "0.35rem 0", color: "#9a3412" }}>
                                No known rule makes these registrations ineligible, but the missing information must be completed before final eligibility or preferred placement can be confirmed. Publication stays blocked only for these targeted data-completion tasks.
                              </p>
                              <ul style={{ marginBottom: 0 }}>
                                {item.data_completion_registrations.map((registration) => {
                                  const proposed = objectValue(registration.proposed_value);
                                  const eligibilityIssues = Array.isArray(proposed.eligibility_issues)
                                    ? proposed.eligibility_issues.map((value) => safeString(value)).filter(Boolean)
                                    : [];
                                  const issue = eligibilityIssues.join(" · ") || safeString(proposed.assignment_issue) || "Complete the missing eligibility information.";
                                  const editorHref = tournamentRouteHref(`/admin/tournaments/registration/registrants/${encodeURIComponent(registration.registration_id)}`, { tournamentId, tournamentName: basics.name || tournamentName, drawId });
                                  return (
                                    <li key={`data-${registration.registration_id}-${registration.selection_id || "registration"}`}>
                                      <Link href={editorHref}>{registration.display_name || registration.email || "Registration needs details"}</Link> — {issue}
                                    </li>
                                  );
                                })}
                              </ul>
                            </article>
                          ) : null}
                          <div style={{ display: "flex", gap: "0.55rem", flexWrap: "wrap", marginTop: "0.7rem" }}>
                            <button type="button" style={ghostButtonStyle} onClick={() => void keepPublishedValueForBlockedChange(item)}>
                              Keep published value
                            </button>
                            <Link href={editHref} style={ghostButtonStyle}>Edit affected draft</Link>
                          </div>
                          {item.requires_acknowledgement ? (
                            <div style={{ marginTop: "0.8rem", paddingTop: "0.8rem", borderTop: "1px solid #fde68a" }}>
                              <div style={{ display: "grid", gridTemplateColumns: "minmax(220px, 1fr) minmax(260px, 2fr)", gap: "0.65rem" }}>
                                <label><strong>Communication action</strong><br />
                                  <select
                                    value={safeString(plan?.action)}
                                    style={inputStyle}
                                    onChange={(event) => updateCommunicationAcknowledgement(item, { action: event.target.value })}
                                  >
                                    <option value="">Choose action…</option>
                                    <option value="NOTIFY_AFFECTED">Notify affected registrants</option>
                                    <option value="ACKNOWLEDGE_NO_NOTICE">No separate notice needed</option>
                                  </select>
                                </label>
                                <label><strong>Communication note (optional)</strong><br />
                                  <textarea
                                    value={safeString(plan?.notes)}
                                    style={{ ...inputStyle, minHeight: "76px" }}
                                    placeholder="Optional record of the message, channel, timing, or why no separate notice is needed."
                                    onChange={(event) => updateCommunicationAcknowledgement(item, { notes: event.target.value })}
                                  />
                                </label>
                              </div>
                              <label style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginTop: "0.6rem", fontWeight: 800 }}>
                                <input
                                  type="checkbox"
                                  checked={acknowledged}
                                  disabled={!safeString(plan?.action) || dataCompletionPending}
                                  onChange={(event) => updateCommunicationAcknowledgement(item, { acknowledged: event.target.checked })}
                                />
                                I completed and verified this communication action
                              </label>
                              <small style={{ color: acknowledged ? "#166534" : "#92400e", fontWeight: 800 }}>
                                {acknowledged
                                  ? "Communication impact acknowledged for publication"
                                  : dataCompletionPending
                                    ? "Complete the required eligibility information before acknowledging this impact"
                                    : "Choose an action and confirm completion"}
                              </small>
                            </div>
                          ) : (
                            <p style={{ color: "#166534", fontWeight: 800 }}>Informational only; no acknowledgement is required.</p>
                          )}
                        </article>
                      );
                    })}
                  </div>
                </>
              ) : (
                <p style={{ color: "#166534", fontWeight: 800 }}>No registration-preserving communication impacts.</p>
              )}
              {blockedDetails.length ? (
                <>
                  <strong>Blocked changes — resolve each before publishing</strong>
                  <div style={{ display: "grid", gap: "0.75rem", marginTop: "0.55rem" }}>
                    {blockedDetails.map((item) => {
                      const editHref = item.step === "divisions"
                        ? `${tournamentSetupStepHref("divisions", tournamentId, basics.name || tournamentName, drawId)}&resolveDivision=${encodeURIComponent(item.entity_id || "")}`
                        : tournamentSetupStepHref(item.step || "review", tournamentId, basics.name || tournamentName, drawId);
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
                            <div style={{ padding: "0.6rem", borderRadius: "10px", background: "white", minWidth: 0, overflowWrap: "anywhere" }}><small>Current published {humanReviewFieldLabel(item.field || "value")}{item.current_source ? ` · ${item.current_source}` : ""}</small><ReviewValueDisplay field={item.field || "value"} value={item.current_value} days={configuration.days} timezone={basics.timezone} /></div>
                            <div style={{ padding: "0.6rem", borderRadius: "10px", background: "#eff6ff", minWidth: 0, overflowWrap: "anywhere" }}><small>Proposed draft {humanReviewFieldLabel(item.field || "value")}{item.proposed_source ? ` · ${item.proposed_source}` : ""}</small><ReviewValueDisplay field={item.field || "value"} value={item.proposed_value} days={configuration.days} timezone={basics.timezone} /></div>
                          </div>
                          <div style={{ display: "flex", gap: "0.55rem", flexWrap: "wrap", marginTop: "0.7rem" }}>
                            <button type="button" style={ghostButtonStyle} onClick={() => void keepPublishedValueForBlockedChange(item)}>
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
                                Publication remains blocked until every row has a structured action and completion confirmation. Audit notes are optional for standard actions and required only for Other. Complete the actual registration change through the linked editor, then record the resolution here.
                              </p>
                              <div style={{ display: "grid", gap: "0.65rem" }}>
                                {planRows.map((row) => {
                                  const registration: AffectedRegistration = {
                                    registration_id: safeString(row.registration_id),
                                    selection_id: safeString(row.selection_id),
                                    display_name: safeString(row.display_name),
                                    email: safeString(row.email),
                                    registration_status: safeString(row.registration_status),
                                    current_source: safeString(row.current_source),
                                    proposed_source: safeString(row.proposed_source)
                                  };
                                  const editorHref = tournamentRouteHref(`/admin/tournaments/registration/registrants/${encodeURIComponent(registration.registration_id)}`, { tournamentId, tournamentName: basics.name || tournamentName, drawId });
                                  const resolved = Boolean(row.resolved);
                                  return (
                                    <article key={`${registration.registration_id}-${registration.selection_id || "registration"}`} style={{ border: "1px solid #e2e8f0", borderRadius: "10px", padding: "0.65rem", background: "white" }}>
                                      <div style={{ display: "flex", justifyContent: "space-between", gap: "0.55rem", flexWrap: "wrap" }}>
                                        <div><strong>{registration.display_name || registration.email || "Registration needs details"}</strong><br /><small>{registration.email || "No email"} · {registration.registration_status || "Unknown status"}</small></div>
                                        <Link href={editorHref} style={ghostButtonStyle}>Open registration editor</Link>
                                      </div>
                                      {(row.current_value != null || row.proposed_value != null) ? (
                                        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.55rem", marginTop: "0.6rem" }}>
                                          <div style={{ padding: "0.55rem", borderRadius: "8px", background: "#f8fafc", minWidth: 0 }}><small>Current registration value{safeString(row.current_source) ? ` · ${safeString(row.current_source)}` : ""}</small><ReviewValueDisplay field={item.field || "value"} value={row.current_value} days={configuration.days} timezone={basics.timezone} /></div>
                                          <div style={{ padding: "0.55rem", borderRadius: "8px", background: "#eff6ff", minWidth: 0 }}><small>Proposed registration value{safeString(row.proposed_source) ? ` · ${safeString(row.proposed_source)}` : ""}</small><ReviewValueDisplay field={item.field || "value"} value={row.proposed_value} days={configuration.days} timezone={basics.timezone} /></div>
                                        </div>
                                      ) : null}
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
                                        <label><strong>Audit note {safeString(row.action).toUpperCase() === "OTHER" ? "(required)" : "(optional)"}</strong><br />
                                          <textarea value={safeString(row.notes)} style={{ ...inputStyle, minHeight: "76px" }} placeholder={safeString(row.action).toUpperCase() === "OTHER" ? "Describe the custom resolution and why it resolves the conflict." : "Optional context beyond the recorded action, actor, timestamp, and before/after values."} onChange={(event) => updateForcedRegistration(item, registration, { notes: event.target.value })} />
                                        </label>
                                      </div>
                                      <label style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginTop: "0.6rem", fontWeight: 800 }}>
                                        <input
                                          type="checkbox"
                                          checked={resolved}
                                          disabled={!safeString(row.action) || (safeString(row.action).toUpperCase() === "OTHER" && !safeString(row.notes))}
                                          onChange={(event) => updateForcedRegistration(item, registration, { resolved: event.target.checked })}
                                        />
                                        I completed and verified this registration action
                                      </label>
                                      <small style={{ color: resolved ? "#166534" : "#92400e", fontWeight: 800 }}>{resolved ? "Resolved for publication" : safeString(row.action).toUpperCase() === "OTHER" ? "Action, audit note, and completion confirmation required" : "Action and completion confirmation required"}</small>
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
                </>
              ) : (
                <p style={{ color: "#166534", fontWeight: 800 }}>No blocked changes.</p>
              )}
              {hasReviewActionPlans ? (
                <div style={{ marginTop: "0.75rem" }}>
                  <button type="button" style={buttonStyle} disabled={busy || !resolutionDraftDirty} onClick={() => void saveResolutionDraft()}>
                    {busy ? "Saving Review actions…" : resolutionDraftDirty ? "Save Review actions" : "Review actions saved"}
                  </button>
                  {resolutionDraftDirty ? <p style={{ color: "#92400e" }}>Save the registration resolutions and communication acknowledgements before publishing or refreshing the browser.</p> : null}
                </div>
              ) : null}
            </div>
          ) : null}
        </article>

        <article style={cardStyle}>
          <h3 style={{ marginTop: 0 }}>Publish tournament</h3>
          <p style={{ color: "#475569" }}>
            Publish the exact reviewed Tournament, Competition, and Commerce setup. Existing registrations remain protected; hard conflicts require completed registration resolutions, while registration-preserving changes require a saved communication acknowledgement.
          </p>
          <p style={{ color: "#475569" }}>
            On first publish, a Draft tournament becomes Active and appears on the public site. Republishing preserves an existing Active, Paused, Inactive, Completed, or Archived lifecycle state. Registration status remains a separate action.
          </p>
          <ConfirmAction
            triggerLabel={busy ? "Publishing…" : "Publish reviewed tournament"}
            title="Publish this reviewed tournament?"
            description="Apply the exact reviewed draft to the published tournament. A new Draft becomes Active on first publish; every other lifecycle state is preserved. Registration status remains a separate action. Registration resolutions and communication acknowledgements are written to the audit record."
            confirmLabel="Yes, publish tournament"
            confirmationText={publishConfirmation}
            disabled={!impactReview || reviewedDraftSignature !== fullDraftSignature(basics, settings, configuration) || unresolvedBlockers.length > 0 || unresolvedCommunications.length > 0 || resolutionDraftDirty}
            busy={busy}
            onConfirm={publishSetup}
          />
          {unresolvedBlockers.length ? <p style={{ color: "#b91c1c" }}>Resolve {unresolvedBlockers.length} blocked change{unresolvedBlockers.length === 1 ? "" : "s"} before publishing.</p> : null}
          {unresolvedCommunications.length ? <p style={{ color: "#92400e" }}>Acknowledge {unresolvedCommunications.length} registration-preserving communication impact{unresolvedCommunications.length === 1 ? "" : "s"} before publishing.</p> : null}
          {resolutionDraftDirty ? <p style={{ color: "#92400e" }}>Save the Review actions before publishing.</p> : null}
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
              description="Open registration using the published tournament, registration window, policies, divisions, prices, and Players Needing Partners settings."
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
            <Link href={tournamentSetupStepHref("pricing", tournamentId, basics.name || tournamentName, drawId)} style={ghostButtonStyle}>Back to Commerce</Link>
            <Link href={tournamentRouteHref("/admin/tournaments/tournament", { tournamentId, tournamentName: basics.name || tournamentName, drawId })} style={ghostButtonStyle}>Return to Tournament Home</Link>
          </>
        )}
      </div>
    );
  }
  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <article
        style={{
          ...cardStyle,
          background: publicationBanner.background,
          borderColor: publicationBanner.borderColor
        }}
      >
        <div
          role={publicationStatus === "unavailable" ? "alert" : "status"}
          aria-busy={publicationStatus === "checking"}
        >
          <strong>{publicationBanner.title}</strong>
          <p style={{ margin: "0.35rem 0 0", color: "#475569" }}>
            {publicationBanner.description}
          </p>
        </div>
        {publicationStatus === "unavailable" ? (
          <button
            type="button"
            style={{ ...ghostButtonStyle, marginTop: "0.75rem" }}
            disabled={busy}
            onClick={() => void loadDetail()}
          >
            {busy ? "Retrying…" : "Retry setup status"}
          </button>
        ) : null}
      </article>
      <TournamentSetupWizardNav
        currentStep={step}
        tournamentId={tournamentId}
        tournamentName={basics.name || tournamentName}
        drawId={drawId}
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
