from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text()


def write(path: str, text: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def replace_once(path: str, old: str, new: str) -> None:
    text = read(path)
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one match, found {count}: {old[:100]!r}")
    write(path, text.replace(old, new, 1))


def replace_between(path: str, start: str, end: str, replacement: str) -> None:
    text = read(path)
    start_index = text.find(start)
    if start_index < 0:
        raise SystemExit(f"{path}: start marker not found: {start!r}")
    end_index = text.find(end, start_index + len(start))
    if end_index < 0:
        raise SystemExit(f"{path}: end marker not found: {end!r}")
    write(path, text[:start_index] + replacement + text[end_index:])


# ---------------------------------------------------------------------------
# Builder: ordered multi-day schedules for event families and divisions.
# ---------------------------------------------------------------------------
builder_path = "apps/web/app/admin/tournament-setup/tournamentSetupBuilder.ts"
replace_once(
    builder_path,
    '''export function eventDayReference(row: SetupRecord): string {
  return cleanString(row.assigned_day ?? row.registration_day_id);
}

export function eventUsesLabelDayReference(row: SetupRecord): boolean {
''',
    '''export function eventDayReference(row: SetupRecord): string {
  return cleanString(row.assigned_day ?? row.registration_day_id);
}

function cleanStringList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return [...new Set(value.map(cleanString).filter(Boolean))];
}

export function eventDayReferences(row: SetupRecord): string[] {
  const scheduled = cleanStringList(
    row.scheduled_day_ids ?? row.registration_day_ids ?? row.assigned_days
  );
  if (scheduled.length) return scheduled;
  const primary = eventDayReference(row);
  return primary ? [primary] : [];
}

export function setEventDayReferences(
  row: SetupRecord,
  references: readonly string[]
): SetupRecord {
  const scheduled = [...new Set(references.map(cleanString).filter(Boolean))];
  const primary = scheduled[0] || "";
  const next: SetupRecord = {
    ...row,
    scheduled_day_ids: scheduled,
    registration_day_id: primary
  };
  if (Object.prototype.hasOwnProperty.call(row, "assigned_day")) {
    next.assigned_day = primary;
  }
  return next;
}

export function eventUsesLabelDayReference(row: SetupRecord): boolean {
''',
)
replace_once(
    builder_path,
    '''    registration_day_id: registrationDayId,
    participant_type: "GENDER_DOUBLES",
''',
    '''    registration_day_id: registrationDayId,
    scheduled_day_ids: registrationDayId ? [registrationDayId] : [],
    participant_type: "GENDER_DOUBLES",
''',
)
replace_once(
    builder_path,
    '''  const defaults = eventFamilyDefaults(configuration.eventFamilies, familyName);
  const existingNames = new Set(configuration.eventOptions.map((row) => eventDivisionName(row.value).toLowerCase()));
''',
    '''  const defaults = eventFamilyDefaults(configuration.eventFamilies, familyName);
  const inheritedSchedule = eventDayReferences(defaults || {});
  const fallbackSchedule = [dayReference(firstDay)].filter(Boolean);
  const scheduledDayIds = inheritedSchedule.length ? inheritedSchedule : fallbackSchedule;
  const existingNames = new Set(configuration.eventOptions.map((row) => eventDivisionName(row.value).toLowerCase()));
''',
)
replace_once(
    builder_path,
    '''      assigned_day: dayLabel(firstDay),
      capacity_teams: Number(defaults?.default_capacity_teams ?? 16),
''',
    '''      assigned_day: scheduledDayIds[0] || dayLabel(firstDay),
      registration_day_id: scheduledDayIds[0] || dayReference(firstDay),
      scheduled_day_ids: scheduledDayIds,
      schedule_mode: "INHERIT_EVENT",
      capacity_teams: Number(defaults?.default_capacity_teams ?? 16),
''',
)
replace_once(
    builder_path,
    '''    registration_day_id: dayReference(firstDay),
    event_family_label: familyName,
''',
    '''    registration_day_id: scheduledDayIds[0] || dayReference(firstDay),
    scheduled_day_ids: scheduledDayIds,
    schedule_mode: "INHERIT_EVENT",
    event_family_label: familyName,
''',
)
replace_once(
    builder_path,
    '''    const usesLegacyShape = eventUsesLegacyBuilderShape(row);
    if (!usesLegacyShape) return projectCanonicalAgeRuleEdits(row);

    const familyName = eventFamilyName(row);
''',
    '''    const usesLegacyShape = eventUsesLegacyBuilderShape(row);
    if (!usesLegacyShape) {
      const projected = projectCanonicalAgeRuleEdits(row);
      const scheduledDayIds = eventDayReferences(projected)
        .map((reference) =>
          dayIdsByLabel.get(normalizedLookupKey(reference)) || reference
        )
        .filter((reference) => dayIdsByLabel.has(normalizedLookupKey(reference)) || days.some((day) => cleanString(day.id) === reference));
      const primary = scheduledDayIds[0] || cleanString(projected.registration_day_id);
      return {
        ...projected,
        registration_day_id: primary,
        scheduled_day_ids: scheduledDayIds.length ? scheduledDayIds : (primary ? [primary] : [])
      };
    }

    const familyName = eventFamilyName(row);
''',
)
replace_once(
    builder_path,
    '''    const assignedDay = eventDayReference(row);
    const registrationDayId = dayIdsByLabel.get(normalizedLookupKey(assignedDay))
      || cleanString(row.registration_day_id)
      || assignedDay;
''',
    '''    const assignedDay = eventDayReference(row);
    const scheduleReferences = eventDayReferences(row).length
      ? eventDayReferences(row)
      : eventDayReferences(defaults);
    const scheduledDayIds = scheduleReferences
      .map((reference) =>
        dayIdsByLabel.get(normalizedLookupKey(reference)) || reference
      )
      .filter(Boolean);
    const registrationDayId = scheduledDayIds[0]
      || dayIdsByLabel.get(normalizedLookupKey(assignedDay))
      || cleanString(row.registration_day_id)
      || assignedDay;
''',
)
replace_once(
    builder_path,
    '''      registration_day_id: registrationDayId,
      sort_order: row.sort_order || index + 1,
''',
    '''      registration_day_id: registrationDayId,
      scheduled_day_ids: scheduledDayIds.length ? scheduledDayIds : (registrationDayId ? [registrationDayId] : []),
      sort_order: row.sort_order || index + 1,
''',
)
replace_once(
    builder_path,
    '''  families.forEach((row, index) => {
    if (!eventFamilyName(row)) issues.push({ path: `families.${index}.event_family`, message: "Event name is required." });
''',
    '''  families.forEach((row, index) => {
    if (!eventFamilyName(row)) issues.push({ path: `families.${index}.event_family`, message: "Event name is required." });
    const scheduledDays = eventDayReferences(row);
    if (!scheduledDays.length) {
      issues.push({ path: `families.${index}.scheduled_day_ids`, message: "Choose at least one tournament day for this event." });
    }
    for (const scheduledDay of scheduledDays) {
      if (!enabledDayReferences.has(scheduledDay)) {
        issues.push({ path: `families.${index}.scheduled_day_ids`, message: "Every event day must be an enabled tournament day." });
        break;
      }
    }
''',
)
replace_once(
    builder_path,
    '''  const conflictKeys = events.map((row) => [
    eventDayReference(row),
    eventFamilyName(row),
    eventDivisionName(row)
  ].map((part) => part.trim().toLowerCase()).join("|"));
''',
    '''  const conflictKeys = events.map((row) => [
    eventFamilyName(row),
    eventDivisionName(row)
  ].map((part) => part.trim().toLowerCase()).join("|"));
''',
)
replace_once(
    builder_path,
    '''    const day = eventDayReference(row);
    const usesLegacyShape = eventUsesLegacyBuilderShape(row);
''',
    '''    const scheduledDays = eventDayReferences(row);
    const usesLegacyShape = eventUsesLegacyBuilderShape(row);
''',
)
replace_once(
    builder_path,
    '''    if (!day) issues.push({ path: `events.${index}.registration_day_id`, message: "Assigned day is required." });
    else if (!enabledDayReferences.has(day)) {
      issues.push({ path: `events.${index}.registration_day_id`, message: "Choose an enabled tournament day." });
    }
''',
    '''    if (!scheduledDays.length) {
      issues.push({ path: `events.${index}.scheduled_day_ids`, message: "Choose at least one tournament day for this division." });
    } else if (scheduledDays.some((day) => !enabledDayReferences.has(day))) {
      issues.push({ path: `events.${index}.scheduled_day_ids`, message: "Choose only enabled tournament days." });
    }
    const familyDefaults = families.find(
      (familyRow) => normalizedLookupKey(eventFamilyName(familyRow)) === normalizedLookupKey(family)
    );
    const familyDays = eventDayReferences(familyDefaults || {});
    if (familyDays.length && scheduledDays.some((day) => !familyDays.includes(day))) {
      issues.push({ path: `events.${index}.scheduled_day_ids`, message: "Division days must be selected on the parent event." });
    }
''',
)
replace_once(
    builder_path,
    '''      issues.push({ path: `events.${index}.division_name`, message: "This day, event, and division combination is duplicated." });
''',
    '''      issues.push({ path: `events.${index}.division_name`, message: "This event and division combination is duplicated." });
''',
)

# ---------------------------------------------------------------------------
# Event-family card: one event may be scheduled on multiple tournament days.
# ---------------------------------------------------------------------------
event_card_path = "apps/web/app/admin/tournaments/setup/TournamentSetupEventFamilyCard.tsx"
replace_once(
    event_card_path,
    '''  eventDayReference,
  eventFamilyName,
''',
    '''  eventDayReferences,
  eventFamilyName,
''',
)
replace_once(
    event_card_path,
    '''  setRecordNumber,
  setRecordString,
''',
    '''  setEventDayReferences,
  setRecordNumber,
  setRecordString,
''',
)
replace_once(
    event_card_path,
    '''  const currentDay = eventDayReference(value);
''',
    '''  const currentDays = eventDayReferences(value);
''',
)
old_day_select = '''        <label className={styles.label}>
          Tournament day
          <select
            className={styles.select}
            value={currentDay}
            disabled={disabled}
            onChange={(event) =>
              onChange(
                setRecordString(
                  value,
                  ["registration_day_id", "assigned_day"],
                  event.target.value
                )
              )
            }
          >
            <option value="">Choose a day</option>
            {dayOptions.map((option) => (
              <option
                key={option.value}
                value={option.value}
                disabled={!option.enabled}
              >
                {option.label}
                {option.enabled ? "" : " (disabled)"}
              </option>
            ))}
          </select>
        </label>
'''
new_day_select = '''        <fieldset className={`${styles.wide} ${styles.rowCard}`} style={{ padding: "0.75rem" }}>
          <legend style={{ fontWeight: 800 }}>Tournament days</legend>
          <p style={{ margin: "0 0 0.55rem", color: "#64748b" }}>
            Select every day on which this event may be played. A division may use all of these days or a selected subset.
          </p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.45rem" }}>
            {dayOptions.map((option) => (
              <label key={option.value} className={styles.checkbox}>
                <input
                  type="checkbox"
                  checked={currentDays.includes(option.value)}
                  disabled={disabled || !option.enabled}
                  onChange={(event) => {
                    const next = event.target.checked
                      ? [...currentDays, option.value]
                      : currentDays.filter((value) => value !== option.value);
                    onChange(setEventDayReferences(value, next));
                  }}
                />
                {option.label}{option.enabled ? "" : " (disabled)"}
              </label>
            ))}
          </div>
        </fieldset>
'''
replace_once(event_card_path, old_day_select, new_day_select)

# ---------------------------------------------------------------------------
# Division card: inherit all parent-event days or choose a subset.
# ---------------------------------------------------------------------------
division_card_path = "apps/web/app/admin/tournaments/setup/TournamentSetupDivisionCard.tsx"
replace_once(
    division_card_path,
    '''  cleanString,
  eventAgeMode,
  eventDayReference,
''',
    '''  cleanString,
  dayLabel,
  dayReference,
  eventAgeMode,
  eventDayReferences,
''',
)
replace_once(
    division_card_path,
    '''  setAgeRuleNumber,
  setEventAgeMode,
''',
    '''  setAgeRuleNumber,
  setEventAgeMode,
  setEventDayReferences,
''',
)
replace_once(
    division_card_path,
    '''  eventFamilies: BuilderRow[];
  disabled: boolean;
''',
    '''  eventFamilies: BuilderRow[];
  days: BuilderRow[];
  disabled: boolean;
''',
)
replace_between(
    division_card_path,
    "function applyFamily(",
    "\n\nexport default function TournamentSetupDivisionCard",
    '''function applyFamily(value: SetupRecord, eventFamilies: BuilderRow[], familyName: string): SetupRecord {
  const defaults = eventFamilyDefaults(eventFamilies, familyName) || {};
  const schedule = eventDayReferences(defaults);
  const next: SetupRecord = {
    ...value,
    event_family_label: familyName,
    event_family: familyName,
    event_type: cleanString(defaults.participant_type) || value.event_type,
    participant_type: cleanString(defaults.participant_type) || value.participant_type,
    gender_restriction:
      cleanString(defaults.gender_restriction) || value.gender_restriction,
    waitlist_enabled: recordBoolean(
      value.waitlist_enabled,
      recordBoolean(defaults.default_waitlist, true)
    ),
    partner_board_enabled: recordBoolean(
      value.partner_board_enabled,
      recordBoolean(defaults.default_partner_board, true)
    ),
    schedule_mode: "INHERIT_EVENT"
  };
  if (value.capacity_teams == null && defaults.default_capacity_teams != null) {
    next.capacity_teams = defaults.default_capacity_teams;
  }
  if (value.price_usd == null && defaults.default_price_usd != null) {
    next.price_usd = defaults.default_price_usd;
  }
  return setEventDayReferences(next, schedule);
}
''',
)
replace_once(
    division_card_path,
    '''  eventFamilies,
  disabled,
''',
    '''  eventFamilies,
  days,
  disabled,
''',
)
replace_once(
    division_card_path,
    '''  const familyDefaults = eventFamilyDefaults(eventFamilies, family) || {};
  const eventSummary = [
''',
    '''  const familyDefaults = eventFamilyDefaults(eventFamilies, family) || {};
  const parentDays = eventDayReferences(familyDefaults);
  const scheduleMode = cleanString(value.schedule_mode) || "INHERIT_EVENT";
  const selectedDays = scheduleMode === "CUSTOM"
    ? eventDayReferences(value)
    : parentDays;
  const dayLabels = new Map(
    days.map((day) => [dayReference(day.value), dayLabel(day.value) || dayReference(day.value)])
  );
  const eventSummary = [
''',
)
event_select_end = '''        </label>

        <label className={styles.label}>
          Skill division
'''
schedule_ui = '''        </label>

        <fieldset className={`${styles.wide} ${styles.rowCard}`} style={{ padding: "0.75rem" }}>
          <legend style={{ fontWeight: 800 }}>Division schedule</legend>
          <div style={{ display: "grid", gap: "0.45rem" }}>
            <label className={styles.checkbox}>
              <input
                type="radio"
                name={`schedule-mode-${row.key}`}
                checked={scheduleMode === "INHERIT_EVENT"}
                disabled={disabled}
                onChange={() =>
                  onChange(
                    setEventDayReferences(
                      { ...value, schedule_mode: "INHERIT_EVENT" },
                      parentDays
                    )
                  )
                }
              />
              Use every day selected for the parent event
            </label>
            <label className={styles.checkbox}>
              <input
                type="radio"
                name={`schedule-mode-${row.key}`}
                checked={scheduleMode === "CUSTOM"}
                disabled={disabled}
                onChange={() =>
                  onChange({ ...value, schedule_mode: "CUSTOM" })
                }
              />
              Choose specific event days
            </label>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.45rem", marginTop: "0.6rem" }}>
            {parentDays.map((dayId) => (
              <label key={dayId} className={styles.checkbox}>
                <input
                  type="checkbox"
                  checked={selectedDays.includes(dayId)}
                  disabled={disabled || scheduleMode !== "CUSTOM"}
                  onChange={(event) => {
                    const next = event.target.checked
                      ? [...selectedDays, dayId]
                      : selectedDays.filter((value) => value !== dayId);
                    onChange(setEventDayReferences(value, next));
                  }}
                />
                {dayLabels.get(dayId) || dayId}
              </label>
            ))}
          </div>
          {!parentDays.length ? (
            <p style={{ color: "#b91c1c" }}>Return to Events and select at least one tournament day.</p>
          ) : null}
        </fieldset>

        <label className={styles.label}>
          Skill division
'''
replace_once(division_card_path, event_select_end, schedule_ui)

# ---------------------------------------------------------------------------
# Wizard panel: policies in Basics, Schedule before Events, modal division add.
# ---------------------------------------------------------------------------
panel_path = "apps/web/app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx"
replace_once(
    panel_path,
    '''  eventDayReference,
  eventDivisionName,
  eventFamilyName,
''',
    '''  eventDayReference,
  eventDayReferences,
  eventDivisionName,
  eventFamilyName,
''',
)
replace_once(
    panel_path,
    '''  setRecordString,
  validateSetupConfiguration,
''',
    '''  setEventDayReferences,
  setRecordString,
  validateSetupConfiguration,
''',
)
replace_once(
    panel_path,
    '''import TournamentSetupEventFamilyCard from "./TournamentSetupEventFamilyCard";
import TournamentSetupDivisionCard from "./TournamentSetupDivisionCard";
''',
    '''import TournamentSetupEventFamilyCard from "./TournamentSetupEventFamilyCard";
import TournamentSetupDivisionCard from "./TournamentSetupDivisionCard";
import TournamentSetupDivisionDialog from "./TournamentSetupDivisionDialog";
import TournamentSetupPolicies, { withDefaultTournamentPolicies } from "./TournamentSetupPolicies";
''',
)
replace_between(
    panel_path,
    "function derivedEventFamilies(",
    "\n\nfunction globalDivisionStatus",
    '''function derivedEventFamilies(events: SetupRecord[], days: SetupRecord[]): SetupRecord[] {
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
''',
)
replace_between(
    panel_path,
    "function setupState(",
    "\n\nfunction initialDaysFromTournament",
    '''function setupState(
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
''',
)
replace_once(
    panel_path,
    '''  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
''',
    '''  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [divisionDialogOpen, setDivisionDialogOpen] = useState(false);
''',
)
replace_once(
    panel_path,
    '''    setBusy(false);
    setMessage(null);
  }
''',
    '''    setBusy(false);
    setMessage(null);
    setDivisionDialogOpen(false);
  }
''',
)
replace_once(
    panel_path,
    '''  const loadedSettings = payload.settings || {};

  setDetail(payload);
''',
    '''  const loadedSettings = withDefaultTournamentPolicies(payload.settings || {});

  setDetail(payload);
''',
)
replace_between(
    panel_path,
    "async function saveBasics()",
    "\n\nasync function saveDraftAndContinue",
    '''async function saveBasics() {
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
''',
)
replace_once(
    panel_path,
    '''        : step === "schedule"
          ? issues.filter(
              (issue) =>
                issue.path.startsWith("days") ||
                issue.path.endsWith("registration_day_id")
            )
''',
    '''        : step === "schedule"
          ? issues.filter((issue) => issue.path.startsWith("days"))
''',
)
replace_between(
    panel_path,
    "async function saveRegistrationRules()",
    "\n\nasync function reviewImpact()",
    "async function reviewImpact()",
)
replace_between(
    panel_path,
    "  function renderBasics()",
    "\nfunction renderEvents()",
    '''  function renderBasics() {
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

function renderEvents()''',
)
# Events now follow Schedule and support inherited multi-day schedules.
replace_once(panel_path, "<h2 style={{ marginTop: 0 }}>2. Events</h2>", "<h2 style={{ marginTop: 0 }}>3. Events</h2>")
replace_once(
    panel_path,
    '''            href={tournamentSetupStepHref("basics", tournamentId, basics.name || tournamentName)}
''',
    '''            href={tournamentSetupStepHref("schedule", tournamentId, basics.name || tournamentName)}
''',
)
replace_once(
    panel_path,
    '''                const previousDay = eventDayReference(row.value);
                const nextDay = eventDayReference(value);
''',
    '''                const nextDays = eventDayReferences(value);
''',
)
replace_once(
    panel_path,
    '''                    let nextValue: SetupRecord = {
                      ...division.value,
                      event_family_label: nextName,
                      event_family: nextName
                    };
                    if (previousDay !== nextDay) {
                      nextValue = {
                        ...nextValue,
                        registration_day_id: nextDay,
                        assigned_day: nextDay
                      };
                    }
                    return { ...division, value: nextValue };
''',
    '''                    let nextValue: SetupRecord = {
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
''',
)
# Replace Divisions renderer with modal creation and multi-day-aware cards.
replace_between(
    panel_path,
    "function renderDivisions()",
    "\nfunction renderRegistrationRules()",
    '''function renderDivisions() {
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

function renderRegistrationRules()''',
)
# Remove legacy Registration rules function entirely.
replace_between(
    panel_path,
    "function renderRegistrationRules()",
    "\n\n  function renderPricing()",
    "  function renderPricing()",
)
replace_once(panel_path, "<h2 style={{ marginTop: 0 }}>5. Pricing, extras, and fulfillment</h2>", "<h2 style={{ marginTop: 0 }}>5. Pricing, extras, and fulfillment</h2>")
replace_once(
    panel_path,
    '''                "registration-rules",
''',
    '''                "divisions",
''',
)
replace_once(
    panel_path,
    '''              onClick={() => goTo("schedule")}
            >
              Continue to schedule and courts
''',
    '''              onClick={() => goTo("review")}
            >
              Continue to final review
''',
)
# Replace Schedule renderer and place it before Events in the guided flow.
replace_between(
    panel_path,
    "  function renderSchedule()",
    "\n\n  function renderReview()",
    '''  function renderSchedule() {
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

  function renderReview()''',
)
# Review is now Step 6 with policies included in Basics.
replace_once(panel_path, "<h2 style={{ marginTop: 0 }}>7. Review and open registration</h2>", "<h2 style={{ marginTop: 0 }}>6. Review and open registration</h2>")
replace_once(
    panel_path,
    '''    const basicsReady = Boolean(
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
''',
    '''    const basicsReady = states.basics === "complete";
    const scheduleReady = states.schedule === "complete";
    const eventFamiliesReady = states.events === "complete";
    const divisionsReady = states.divisions === "complete";
    const ready =
      basicsReady &&
      scheduleReady &&
      eventFamiliesReady &&
      divisionsReady &&
      issues.length === 0;
''',
)
replace_once(
    panel_path,
    '''              ["Tournament basics", basicsReady, `${basics.startDate || "No start"} – ${basics.endDate || "No end"}`],
              ["Events", configuration.eventFamilies.length > 0, `${configuration.eventFamilies.length} event(s)`],
              ["Divisions", eventsReady, `${configuration.eventOptions.length} division(s)`],
              ["Registration rules", rulesReady, `Status: ${registrationStatus}`],
              ["Pricing and extras", true, "Review the saved catalog from Step 4"],
              ["Schedule and courts", scheduleReady, `${configuration.days.length} tournament day(s)`]
''',
    '''              ["Tournament basics and policies", basicsReady, `${basics.startDate || "No start"} – ${basics.endDate || "No end"} · registration ${registrationStatus}`],
              ["Schedule and courts", scheduleReady, `${configuration.days.length} tournament day(s)`],
              ["Events", eventFamiliesReady, `${configuration.eventFamilies.length} event(s)`],
              ["Divisions", divisionsReady, `${configuration.eventOptions.length} division(s)`],
              ["Pricing and extras", true, "Review the saved catalog from Step 5"]
''',
)
replace_once(
    panel_path,
    '''                "schedule",
''',
    '''                "pricing",
''',
)
replace_once(panel_path, "Step {definition.number} of 7", "Step {definition.number} of 6")
replace_once(
    panel_path,
    '''        ) : step === "events" ? (
          renderEvents()
        ) : step === "divisions" ? (
          renderDivisions()
) : step === "registration-rules" ? (
          renderRegistrationRules()
        ) : step === "pricing" ? (
          renderPricing()
        ) : step === "schedule" ? (
          renderSchedule()
''',
    '''        ) : step === "schedule" ? (
          renderSchedule()
        ) : step === "events" ? (
          renderEvents()
        ) : step === "divisions" ? (
          renderDivisions()
        ) : step === "pricing" ? (
          renderPricing()
''',
)

# Wizard page copy and legacy Registration-rules route.
replace_once(
    "apps/web/app/admin/tournaments/setup/TournamentSetupWizardPage.tsx",
    "Complete the seven setup steps in order.",
    "Complete the six setup steps in order.",
)
write(
    "apps/web/app/admin/tournaments/setup/registration-rules/page.tsx",
    '''import { redirect } from "next/navigation";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

function first(value: string | string[] | undefined): string {
  return Array.isArray(value) ? String(value[0] || "") : String(value || "");
}

export default function TournamentSetupRegistrationRulesPage({ searchParams }: Props) {
  const tournament = first(searchParams?.tournament).trim();
  const name = first(searchParams?.name).trim();
  const params = new URLSearchParams();
  if (tournament) params.set("tournament", tournament);
  if (name) params.set("name", name);
  redirect(`/admin/tournaments/setup/basics?${params.toString()}`);
}
''',
)

# Export policy-default hydrator from the new component.
policies_path = "apps/web/app/admin/tournaments/setup/TournamentSetupPolicies.tsx"
replace_once(
    policies_path,
    '''function text(value: unknown): string {
''',
    '''export function withDefaultTournamentPolicies(settings: SetupRecord): SetupRecord {
  return {
    ...settings,
    rules_markdown: text(settings.rules_markdown) || REGISTRATION_RULE_TEMPLATES[0].text,
    refund_policy_markdown:
      text(settings.refund_policy_markdown) || CANCELLATION_TEMPLATES[1].text,
    weather_policy_markdown:
      text(settings.weather_policy_markdown) || WEATHER_TEMPLATES[0].text
  };
}

function text(value: unknown): string {
''',
)

# ---------------------------------------------------------------------------
# API/service/repository schema support.
# ---------------------------------------------------------------------------
route_path = "services/api/admin_tournament_setup_routes.py"
replace_once(
    route_path,
    '''    refund_policy_markdown: str | None = None
    sponsor_markdown: str | None = None
''',
    '''    refund_policy_markdown: str | None = None
    weather_policy_markdown: str | None = None
    sponsor_markdown: str | None = None
''',
)
service_path = "jupr_app/services/admin_tournament_setup_service.py"
replace_once(
    service_path,
    '''        "refund_policy_markdown": row.get("refund_policy_markdown"),
        "sponsor_markdown": row.get("sponsor_markdown"),
''',
    '''        "refund_policy_markdown": row.get("refund_policy_markdown"),
        "weather_policy_markdown": row.get("weather_policy_markdown"),
        "sponsor_markdown": row.get("sponsor_markdown"),
''',
)
replace_once(
    service_path,
    '''        "registration_day_id": row.get("registration_day_id"),
        "event_family_label": row.get("event_family_label"),
''',
    '''        "registration_day_id": row.get("registration_day_id"),
        "scheduled_day_ids": list(row.get("scheduled_day_ids") or []),
        "event_family_label": row.get("event_family_label"),
''',
)
repo_path = "jupr_app/domain/tournament_registration_repo.py"
replace_once(
    repo_path,
    '''        "sponsors_json",
    ),
''',
    '''        "sponsors_json",
        "weather_policy_markdown",
    ),
''',
)
replace_once(
    repo_path,
    '''        "registration_day_id",
        "event_family_label",
''',
    '''        "registration_day_id",
        "scheduled_day_ids",
        "event_family_label",
''',
)
replace_once(
    repo_path,
    '''        "refund_policy_markdown": "",
        "sponsor_markdown": "",
''',
    '''        "refund_policy_markdown": "",
        "weather_policy_markdown": "",
        "sponsor_markdown": "",
''',
)
replace_once(
    repo_path,
    '''        "refund_policy_markdown": str(payload.get("refund_policy_markdown") or ""),
        "sponsor_markdown": str(payload.get("sponsor_markdown") or ""),
''',
    '''        "refund_policy_markdown": str(payload.get("refund_policy_markdown") or ""),
        "weather_policy_markdown": str(payload.get("weather_policy_markdown") or ""),
        "sponsor_markdown": str(payload.get("sponsor_markdown") or ""),
''',
)
replace_once(
    repo_path,
    '''    "registration_day_id",
    "sort_order",
''',
    '''    "registration_day_id",
    "scheduled_day_ids",
    "sort_order",
''',
)
replace_once(
    repo_path,
    '''        registration_day_id = str(raw.get("registration_day_id") or "").strip()
        registration_day_id = day_aliases.get(registration_day_id, registration_day_id)
        if not registration_day_id and len(normalized_days) == 1:
            registration_day_id = str(normalized_days[0]["id"])
        if registration_day_id not in day_ids:
            raise ValueError(
                f"Invalid event payload at row {index}: registration_day_id '{registration_day_id}' is not present in day payload."
            )
''',
    '''        raw_scheduled = raw.get("scheduled_day_ids")
        scheduled_day_ids = [
            day_aliases.get(str(value or "").strip(), str(value or "").strip())
            for value in (raw_scheduled if isinstance(raw_scheduled, list) else [])
            if str(value or "").strip()
        ]
        registration_day_id = str(raw.get("registration_day_id") or "").strip()
        registration_day_id = day_aliases.get(registration_day_id, registration_day_id)
        if not scheduled_day_ids and registration_day_id:
            scheduled_day_ids = [registration_day_id]
        if not scheduled_day_ids and len(normalized_days) == 1:
            scheduled_day_ids = [str(normalized_days[0]["id"])]
        scheduled_day_ids = list(dict.fromkeys(scheduled_day_ids))
        invalid_scheduled = [day_id for day_id in scheduled_day_ids if day_id not in day_ids]
        if invalid_scheduled:
            raise ValueError(
                f"Invalid event payload at row {index}: scheduled day '{invalid_scheduled[0]}' is not present in day payload."
            )
        registration_day_id = scheduled_day_ids[0] if scheduled_day_ids else registration_day_id
        if registration_day_id not in day_ids:
            raise ValueError(
                f"Invalid event payload at row {index}: registration_day_id '{registration_day_id}' is not present in day payload."
            )
''',
)
replace_once(
    repo_path,
    '''                "registration_day_id": registration_day_id,
                "sort_order": raw.get("sort_order") or index,
''',
    '''                "registration_day_id": registration_day_id,
                "scheduled_day_ids": scheduled_day_ids,
                "sort_order": raw.get("sort_order") or index,
''',
)
replace_once(
    repo_path,
    '''        if has_usage and existing_day_id != draft_day_id:
            blocked.append(f"Cannot move populated division '{label}' to a different day.")
''',
    '''        if has_usage and existing_day_id != draft_day_id:
            blocked.append(f"Cannot move populated division '{label}' to a different primary day.")
        existing_schedule = list(existing.get("scheduled_day_ids") or ([existing_day_id] if existing_day_id else []))
        draft_schedule = list(event.get("scheduled_day_ids") or ([draft_day_id] if draft_day_id else []))
        if has_usage and existing_schedule != draft_schedule:
            blocked.append(f"Cannot change the multi-day schedule for populated division '{label}'.")
''',
)

# Migration source guard latest migration expectations.
prod_guard = "tests/test_production_deployment_hardening.py"
text = read(prod_guard)
text = re.sub(
    r'assert migrations\[-1\] == "202610\d{8}"',
    'assert migrations[-1] == "20261022000000"',
    text,
    count=1,
)
write(prod_guard, text)

# Existing Setup refinement tests now reflect six steps and merged policies.
test_path = "tests/test_api_contract_tournament_setup_event_division_refine.py"
test_text = read(test_path)
test_text = test_text.replace('assert "Step {definition.number} of 7" in panel', 'assert "Step {definition.number} of 6" in panel')
test_text = test_text.replace('assert \'saveDraftAndContinue("registration-rules")\' in panel', 'assert \'saveDraftAndContinue("pricing")\' in panel')
test_text = test_text.replace('assert "Tournament-wide registration status" in rules', 'assert "TournamentSetupPolicies" in rules')
test_text = test_text.replace('assert "Divisions do not have separate registration statuses" in rules', 'assert "weather_policy_markdown" in rules')
write(test_path, test_text)

write(
    "tests/test_api_contract_tournament_setup_flow_policies_multiday.py",
    '''from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "apps" / "web"


def read(path: str) -> str:
    return (WEB / path).read_text()


def test_guided_setup_order_and_policy_merge() -> None:
    nav = read("components/TournamentSetupWizardNav.tsx")
    panel = read("app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx")
    assert 'label: "Tournament basics and policies"' in nav
    assert 'key: "schedule"' in nav
    assert nav.index('key: "schedule"') < nav.index('key: "events"') < nav.index('key: "divisions"')
    assert '"registration-rules"' not in nav
    assert "Step {definition.number} of 6" in panel
    assert "TournamentSetupPolicies" in panel
    assert 'goTo("schedule")' in panel
    assert 'saveDraftAndContinue("events")' in panel
    assert 'saveDraftAndContinue("divisions")' in panel
    assert 'saveDraftAndContinue("pricing")' in panel


def test_policy_templates_and_weather_policy_are_required() -> None:
    policies = read("app/admin/tournaments/setup/TournamentSetupPolicies.tsx")
    panel = read("app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx")
    routes = (ROOT / "services/api/admin_tournament_setup_routes.py").read_text()
    repo = (ROOT / "jupr_app/domain/tournament_registration_repo.py").read_text()
    assert policies.count('Write a custom policy') == 1
    assert 'Flexible refund policy' in policies
    assert 'Refunds until registration closes' in policies
    assert 'Weather policy' in policies
    assert 'weather_policy_markdown' in panel
    assert 'weather_policy_markdown' in routes
    assert 'weather_policy_markdown' in repo


def test_add_division_uses_dialog_instead_of_appending_hidden_row() -> None:
    panel = read("app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx")
    dialog = read("app/admin/tournaments/setup/TournamentSetupDivisionDialog.tsx")
    assert "TournamentSetupDivisionDialog" in panel
    assert 'onClick={() => setDivisionDialogOpen(true)}' in panel
    assert 'role="dialog"' in dialog
    assert 'Add division' in dialog
    assert 'onConfirm(draft)' in dialog
    assert 'scrollIntoView' in panel


def test_event_and_division_schedules_support_multiple_days() -> None:
    builder = read("app/admin/tournament-setup/tournamentSetupBuilder.ts")
    event_card = read("app/admin/tournaments/setup/TournamentSetupEventFamilyCard.tsx")
    division_card = read("app/admin/tournaments/setup/TournamentSetupDivisionCard.tsx")
    repo = (ROOT / "jupr_app/domain/tournament_registration_repo.py").read_text()
    migration = (ROOT / "supabase/migrations/20261022000000_tournament_setup_policies_multiday.sql").read_text()
    assert "eventDayReferences" in builder
    assert "setEventDayReferences" in builder
    assert "scheduled_day_ids" in builder
    assert "Select every day on which this event may be played" in event_card
    assert "Use every day selected for the parent event" in division_card
    assert "Choose specific event days" in division_card
    assert '"scheduled_day_ids"' in repo
    assert "scheduled_day_ids jsonb" in migration
''',
)
