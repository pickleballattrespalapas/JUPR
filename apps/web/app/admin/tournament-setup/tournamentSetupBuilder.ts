export type SetupRecord = Record<string, unknown>;

export type BuilderRow = {
  key: string;
  value: SetupRecord;
};

export type SetupConfiguration = {
  days: BuilderRow[];
  eventFamilies: BuilderRow[];
  eventOptions: BuilderRow[];
};

export type SetupPayload = {
  days: SetupRecord[];
  event_families: SetupRecord[];
  event_options: SetupRecord[];
};

export type SetupPublishPayload = Pick<SetupPayload, "days" | "event_options">;

export type ValidationIssue = {
  path: string;
  message: string;
};

export const COMPETITION_FORMATS = [
  "ROUND_ROBIN",
  "SINGLE_ELIM",
  "DOUBLE_ELIM",
  "ROUND_ROBIN_PLUS_PLAYOFF"
] as const;

export const SCORING_OPTIONS = ["GAME_TO_11", "GAME_TO_15", "GAME_TO_21", "BEST_2_OF_3"] as const;
export const AGE_MODES = ["ALL_AGES", "FIXED_AGE_BRACKET", "AUTO_AGE_SPLIT", "SPLIT_AGE"] as const;
export const PARTICIPANT_TYPES = ["SINGLES", "GENDER_DOUBLES", "MIXED_DOUBLES"] as const;
export const GENDER_RESTRICTIONS = ["ANY", "MEN", "WOMEN", "MIXED"] as const;
export const DIVISION_STATUSES = ["draft", "open", "tentative", "confirmed", "closed"] as const;
export const SKILL_LABEL_OPTIONS = ["Open", "3.0", "3.5", "4.0", "4.5", "5.0", "5.5"] as const;

let builderKeySequence = 0;

function nextBuilderKey(prefix: string, row: SetupRecord, index: number): string {
  builderKeySequence += 1;
  const rowId = cleanString(row.id).replace(/[^a-zA-Z0-9_-]/g, "-").slice(0, 48);
  return `${prefix}-${rowId || index + 1}-${builderKeySequence}`;
}

function newContractId(prefix: string): string {
  const randomId = globalThis.crypto?.randomUUID?.();
  if (randomId) return `${prefix}_${randomId.replace(/-/g, "").slice(0, 16)}`;
  builderKeySequence += 1;
  return `${prefix}_${Date.now().toString(36)}${builderKeySequence.toString(36)}`;
}

export function cleanString(value: unknown): string {
  return value == null ? "" : String(value).trim();
}

export function recordBoolean(value: unknown, fallback = false): boolean {
  if (typeof value === "boolean") return value;
  if (value == null || value === "") return fallback;
  return ["1", "true", "yes", "on"].includes(String(value).trim().toLowerCase());
}

export function numberInputValue(value: unknown): string {
  if (value == null || value === "") return "";
  const parsed = Number(value);
  return Number.isFinite(parsed) ? String(parsed) : String(value);
}

export function wrapBuilderRows(rows: SetupRecord[] | null | undefined, prefix: string): BuilderRow[] {
  return (rows || []).map((row, index) => {
    const safeRow = row && typeof row === "object" && !Array.isArray(row) ? row : {};
    return { key: nextBuilderKey(prefix, safeRow, index), value: { ...safeRow } };
  });
}

export function rowsToPayload(rows: BuilderRow[]): SetupRecord[] {
  return rows.map((row) => ({ ...row.value }));
}

export function configurationPayload(configuration: SetupConfiguration): SetupPayload {
  return {
    days: rowsToPayload(configuration.days),
    event_families: rowsToPayload(configuration.eventFamilies),
    event_options: rowsToPayload(configuration.eventOptions)
  };
}

export function replaceBuilderRow(rows: BuilderRow[], key: string, value: SetupRecord): BuilderRow[] {
  return rows.map((row) => row.key === key ? { ...row, value: { ...value } } : row);
}

export function removeBuilderRow(rows: BuilderRow[], key: string): BuilderRow[] {
  return rows.filter((row) => row.key !== key);
}

export function moveBuilderRow(rows: BuilderRow[], key: string, direction: -1 | 1): BuilderRow[] {
  const currentIndex = rows.findIndex((row) => row.key === key);
  const nextIndex = currentIndex + direction;
  if (currentIndex < 0 || nextIndex < 0 || nextIndex >= rows.length) return rows;
  const nextRows = [...rows];
  const [moved] = nextRows.splice(currentIndex, 1);
  nextRows.splice(nextIndex, 0, moved);
  return nextRows.map((row, index) => ({
    ...row,
    value: { ...row.value, sort_order: index + 1 }
  }));
}

export function appendBuilderRow(rows: BuilderRow[], prefix: string, value: SetupRecord): BuilderRow[] {
  return [...rows, { key: nextBuilderKey(prefix, value, rows.length), value: { ...value } }];
}

export function setRecordString(
  row: SetupRecord,
  keys: readonly string[],
  value: string
): SetupRecord {
  const targetKey = keys.find((key) => Object.prototype.hasOwnProperty.call(row, key)) || keys[0];
  return { ...row, [targetKey]: value };
}

export function setRecordNumber(
  row: SetupRecord,
  key: string,
  value: string
): SetupRecord {
  if (value === "") return { ...row, [key]: null };
  const parsed = Number(value);
  return { ...row, [key]: Number.isFinite(parsed) ? parsed : value };
}

type AgeRulesState = {
  rules: SetupRecord;
  valid: boolean;
  preserveObjectShape: boolean;
};

function readAgeRules(row: SetupRecord): AgeRulesState {
  const raw = row.age_rules;
  if (raw == null || (typeof raw === "string" && !raw.trim())) {
    return { rules: {}, valid: true, preserveObjectShape: false };
  }
  if (typeof raw === "string") {
    try {
      const parsed: unknown = JSON.parse(raw);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return {
          rules: { ...(parsed as SetupRecord) },
          valid: true,
          preserveObjectShape: false
        };
      }
    } catch {
      // Validation reports malformed persisted rules; an intentional edit can repair them.
    }
    return { rules: {}, valid: false, preserveObjectShape: false };
  }
  if (typeof raw === "object" && !Array.isArray(raw)) {
    return {
      rules: { ...(raw as SetupRecord) },
      valid: true,
      preserveObjectShape: true
    };
  }
  return { rules: {}, valid: false, preserveObjectShape: false };
}

function eventUsesLegacyBuilderShape(row: SetupRecord): boolean {
  return Object.prototype.hasOwnProperty.call(row, "assigned_day")
    || Object.prototype.hasOwnProperty.call(row, "event_family");
}

function nestedRuleValue(rules: SetupRecord, key: string): unknown {
  const nested = rules[key];
  return nested && typeof nested === "object" && !Array.isArray(nested)
    ? nested
    : null;
}

export function ageRuleValue(
  row: SetupRecord,
  key: "min_teams_per_age_group" | "split_age_threshold"
): unknown {
  if (Object.prototype.hasOwnProperty.call(row, key)) return row[key];
  const state = readAgeRules(row);
  if (!state.valid) return undefined;
  if (Object.prototype.hasOwnProperty.call(state.rules, key)) return state.rules[key];
  if (key === "min_teams_per_age_group") {
    return state.rules.min_teams;
  }
  if (state.rules.one_over != null) return state.rules.one_over;
  if (state.rules.threshold != null) return state.rules.threshold;
  const splitRule = nestedRuleValue(state.rules, "split_age_rule") as SetupRecord | null;
  return splitRule?.one_player_over_or_equal;
}

export function eventAgeMode(row: SetupRecord): string {
  const state = readAgeRules(row);
  return cleanString(row.age_mode)
    || (state.valid ? cleanString(state.rules.mode) : "")
    || "ALL_AGES";
}

function encodeAgeRules(state: AgeRulesState, rules: SetupRecord): unknown {
  return state.preserveObjectShape ? rules : JSON.stringify(rules);
}

function setCanonicalAgeRuleNumber(
  row: SetupRecord,
  key: "min_teams_per_age_group" | "split_age_threshold",
  value: string
): SetupRecord {
  const state = readAgeRules(row);
  const rules = state.valid ? { ...state.rules } : {};
  const parsed = value === "" ? null : Number(value);
  const nextValue: unknown = value === ""
    ? null
    : (Number.isFinite(parsed) ? parsed : value);

  if (key === "min_teams_per_age_group") {
    delete rules.min_teams;
    if (value === "") delete rules.min_teams_per_age_group;
    else rules.min_teams_per_age_group = nextValue;
  } else {
    delete rules.one_over;
    delete rules.threshold;
    const existingSplitRule = nestedRuleValue(rules, "split_age_rule") as SetupRecord | null;
    const splitRule = existingSplitRule ? { ...existingSplitRule } : {};
    if (value === "") {
      delete rules.split_age_threshold;
      delete splitRule.one_player_over_or_equal;
      delete splitRule.one_player_under;
    } else {
      rules.split_age_threshold = nextValue;
      splitRule.one_player_over_or_equal = nextValue;
      splitRule.one_player_under = nextValue;
    }
    if (Object.keys(splitRule).length) rules.split_age_rule = splitRule;
    else delete rules.split_age_rule;
  }

  rules.mode = eventAgeMode(row);
  if (!Object.prototype.hasOwnProperty.call(rules, "younger_player_controls_age")) {
    rules.younger_player_controls_age = true;
  }
  if (!Object.prototype.hasOwnProperty.call(rules, "higher_skill_player_controls_skill")) {
    rules.higher_skill_player_controls_skill = true;
  }
  const next: SetupRecord = { ...row, age_rules: encodeAgeRules(state, rules) };
  delete next[key];
  return next;
}

export function setAgeRuleNumber(
  row: SetupRecord,
  key: "min_teams_per_age_group" | "split_age_threshold",
  value: string
): SetupRecord {
  if (eventUsesLegacyBuilderShape(row)) return setRecordNumber(row, key, value);
  return setCanonicalAgeRuleNumber(row, key, value);
}

function clearIncompatibleAgeRuleFields(rules: SetupRecord, mode: string): SetupRecord {
  const next = { ...rules };
  if (mode !== "AUTO_AGE_SPLIT") {
    delete next.min_teams_per_age_group;
    delete next.min_teams;
  }
  if (mode !== "SPLIT_AGE") {
    delete next.split_age_threshold;
    delete next.one_over;
    delete next.threshold;
    delete next.split_age_rule;
  }
  return next;
}

export function setEventAgeMode(row: SetupRecord, value: string): SetupRecord {
  const next = setRecordString(row, ["age_mode"], value);
  if (eventUsesLegacyBuilderShape(row)) {
    if (value !== "AUTO_AGE_SPLIT") delete next.min_teams_per_age_group;
    if (value !== "SPLIT_AGE") delete next.split_age_threshold;
    return next;
  }
  const state = readAgeRules(row);
  if (!Object.prototype.hasOwnProperty.call(row, "age_rules") && value === "ALL_AGES") {
    return next;
  }
  const existingRules = state.valid ? state.rules : {};
  const rules = {
    ...clearIncompatibleAgeRuleFields(existingRules, value),
    mode: value
  };
  return { ...next, age_rules: encodeAgeRules(state, rules) };
}

function projectCanonicalAgeRuleEdits(row: SetupRecord): SetupRecord {
  let projected = { ...row };
  for (const key of ["min_teams_per_age_group", "split_age_threshold"] as const) {
    if (Object.prototype.hasOwnProperty.call(projected, key)) {
      const value = projected[key];
      projected = setCanonicalAgeRuleNumber(
        projected,
        key,
        value == null ? "" : String(value)
      );
    }
  }
  return projected;
}

export function dayLabel(row: SetupRecord): string {
  return cleanString(row.label);
}

export function dayReference(row: SetupRecord): string {
  return cleanString(row.id) || dayLabel(row);
}

export function eventFamilyName(row: SetupRecord): string {
  return cleanString(row.event_family ?? row.event_family_label);
}

export function eventDivisionName(row: SetupRecord): string {
  return cleanString(row.division_name ?? row.label);
}

export function eventDayReference(row: SetupRecord): string {
  return cleanString(row.assigned_day ?? row.registration_day_id);
}

export function eventUsesLabelDayReference(row: SetupRecord): boolean {
  return Object.prototype.hasOwnProperty.call(row, "assigned_day");
}

export function newDayRow(position: number, label = `Day ${position}`): SetupRecord {
  return {
    id: newContractId("day"),
    label,
    event_date: "",
    enabled: true,
    sort_order: position
  };
}

export function newEventFamilyRow(
  position: number,
  name = `Event ${position}`,
  registrationDayId = ""
): SetupRecord {
  return {
    id: newContractId("family"),
    event_family: name,
    registration_day_id: registrationDayId,
    participant_type: "GENDER_DOUBLES",
    gender_restriction: "ANY",
    default_format: "ROUND_ROBIN_PLUS_PLAYOFF",
    default_scoring: "GAME_TO_15",
    default_waitlist: true,
    default_partner_board: true,
    default_capacity_teams: 16,
    default_price_usd: 0,
    default_status: "open",
    sort_order: position
  };
}

export function eventFamilyDefaults(
  families: BuilderRow[],
  familyName: string
): SetupRecord | null {
  const normalizedName = cleanString(familyName).toLowerCase();
  return families.find(
    (row) => eventFamilyName(row.value).toLowerCase() === normalizedName
  )?.value || null;
}

export function effectiveParticipantType(row: SetupRecord, families: BuilderRow[]): string {
  const defaults = eventFamilyDefaults(families, eventFamilyName(row));
  return firstCleanString(
    row.event_type,
    row.participant_type,
    defaults?.participant_type,
    "SINGLES"
  ).toUpperCase();
}

export function effectiveGenderRestriction(row: SetupRecord, families: BuilderRow[]): string {
  const defaults = eventFamilyDefaults(families, eventFamilyName(row));
  return firstCleanString(
    row.gender_restriction,
    defaults?.gender_restriction,
    "ANY"
  ).toUpperCase();
}

export function newEventOptionRow(configuration: SetupConfiguration): SetupRecord {
  const position = configuration.eventOptions.length + 1;
  const firstDay = configuration.days.find((row) => recordBoolean(row.value.enabled, true))?.value
    || configuration.days[0]?.value
    || {};
  const familyNames = distinctFamilyNames(configuration);
  const familyName = familyNames[0] || `Event ${position}`;
  const defaults = eventFamilyDefaults(configuration.eventFamilies, familyName);
  const existingNames = new Set(configuration.eventOptions.map((row) => eventDivisionName(row.value).toLowerCase()));
  let divisionName = `${familyName} Open`;
  let suffix = position;
  while (existingNames.has(divisionName.toLowerCase())) {
    divisionName = `${familyName} Division ${suffix}`;
    suffix += 1;
  }
  const usesDraftShape = configuration.eventOptions.some((row) =>
    Object.prototype.hasOwnProperty.call(row.value, "assigned_day")
    || Object.prototype.hasOwnProperty.call(row.value, "event_family")
  );

  if (usesDraftShape) {
    return {
      id: newContractId("division"),
      event_family: familyName,
      division_name: divisionName,
      skill_label: "Open",
      age_mode: "ALL_AGES",
      age_label: "All Ages",
      assigned_day: dayLabel(firstDay),
      capacity_teams: Number(defaults?.default_capacity_teams ?? 16),
      price_usd: Number(defaults?.default_price_usd ?? 0),
      waitlist_enabled: recordBoolean(defaults?.default_waitlist, true),
      partner_board_enabled: recordBoolean(defaults?.default_partner_board, true),
      status: cleanString(defaults?.default_status) || "open",
      division_format: "",
      division_scoring: "",
      notes: "",
      sort_order: position
    };
  }

  const eventType = cleanString(defaults?.participant_type) || "GENDER_DOUBLES";
  return {
    id: newContractId("event"),
    registration_day_id: dayReference(firstDay),
    event_family_label: familyName,
    division_name: divisionName,
    event_type: eventType,
    gender_restriction: cleanString(defaults?.gender_restriction) || "ANY",
    event_format_default: cleanString(defaults?.default_format) || "ROUND_ROBIN_PLUS_PLAYOFF",
    scoring_default: cleanString(defaults?.default_scoring) || "GAME_TO_15",
    skill_label: "Open",
    skill_mode: "OPEN",
    age_mode: "ALL_AGES",
    age_label: "All Ages",
    capacity_teams: Number(defaults?.default_capacity_teams ?? 16),
    price_usd: Number(defaults?.default_price_usd ?? 0),
    waitlist_enabled: recordBoolean(defaults?.default_waitlist, true),
    partner_board_enabled: recordBoolean(defaults?.default_partner_board, eventType !== "SINGLES"),
    status: cleanString(defaults?.default_status) || "open",
    enabled: true,
    sort_order: position
  };
}

export function distinctFamilyNames(configuration: SetupConfiguration): string[] {
  const values = [
    ...configuration.eventFamilies.map((row) => eventFamilyName(row.value)),
    ...configuration.eventOptions.map((row) => eventFamilyName(row.value))
  ];
  return [...new Set(values.filter(Boolean))];
}

export function parseAdvancedConfiguration(raw: string): SetupPayload {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new Error("Advanced import must be valid JSON.");
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Advanced import must be an object with days, event_families, and event_options arrays.");
  }
  const value = parsed as Record<string, unknown>;
  const output = {
    days: value.days,
    event_families: value.event_families,
    event_options: value.event_options
  };
  for (const [key, rows] of Object.entries(output)) {
    if (!Array.isArray(rows)) throw new Error(`${key} must be a JSON array.`);
    if (rows.some((row) => !row || typeof row !== "object" || Array.isArray(row))) {
      throw new Error(`${key} must contain only JSON objects.`);
    }
  }
  return {
    days: (output.days as SetupRecord[]).map((row) => ({ ...row })),
    event_families: (output.event_families as SetupRecord[]).map((row) => ({ ...row })),
    event_options: (output.event_options as SetupRecord[]).map((row) => ({ ...row }))
  };
}

export function formatAdvancedConfiguration(configuration: SetupConfiguration): string {
  return JSON.stringify(configurationPayload(configuration), null, 2);
}

function duplicateIndexes(values: string[]): Set<number> {
  const seen = new Map<string, number>();
  const duplicates = new Set<number>();
  values.forEach((value, index) => {
    const key = value.trim().toLowerCase();
    if (!key) return;
    const firstIndex = seen.get(key);
    if (firstIndex == null) seen.set(key, index);
    else {
      duplicates.add(firstIndex);
      duplicates.add(index);
    }
  });
  return duplicates;
}

function isIsoDate(value: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const parsed = new Date(`${value}T00:00:00Z`);
  return !Number.isNaN(parsed.valueOf()) && parsed.toISOString().slice(0, 10) === value;
}

function finiteNumber(value: unknown): number | null {
  if (value == null || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizedLookupKey(value: unknown): string {
  return cleanString(value).toLowerCase();
}

function firstCleanString(...values: unknown[]): string {
  for (const value of values) {
    const cleaned = cleanString(value);
    if (cleaned) return cleaned;
  }
  return "";
}

function hasValue(row: SetupRecord, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(row, key) && row[key] != null && row[key] !== "";
}

function booleanWithDefault(row: SetupRecord, key: string, fallback: boolean): boolean {
  return hasValue(row, key) ? recordBoolean(row[key], fallback) : fallback;
}

function legacyAgeRules(row: SetupRecord): unknown {
  const state = readAgeRules(row);
  const mode = eventAgeMode(row);
  const ageRanges = cleanString(row.age_ranges);
  const minimumTeams = finiteNumber(row.min_teams_per_age_group);
  const splitThreshold = finiteNumber(row.split_age_threshold);
  const notes = cleanString(row.notes);
  if (
    !Object.prototype.hasOwnProperty.call(row, "age_rules")
    && mode === "ALL_AGES"
    && !ageRanges
    && minimumTeams == null
    && splitThreshold == null
    && !notes
  ) {
    return null;
  }
  const rules: SetupRecord = {
    ...(state.valid ? clearIncompatibleAgeRuleFields(state.rules, mode) : {}),
    mode,
    younger_player_controls_age: state.rules.younger_player_controls_age ?? true,
    higher_skill_player_controls_skill: state.rules.higher_skill_player_controls_skill ?? true
  };
  const ageLabel = cleanString(row.age_label);
  if (ageLabel) rules.age_label = ageLabel;
  if (ageRanges) rules.age_ranges = ageRanges;
  if (Object.prototype.hasOwnProperty.call(row, "min_teams_per_age_group")) {
    if (minimumTeams != null) rules.min_teams_per_age_group = minimumTeams;
    else delete rules.min_teams_per_age_group;
  }
  if (Object.prototype.hasOwnProperty.call(row, "split_age_threshold")) {
    if (splitThreshold != null) {
      rules.split_age_threshold = splitThreshold;
      rules.split_age_rule = {
        ...((nestedRuleValue(rules, "split_age_rule") as SetupRecord | null) || {}),
        one_player_over_or_equal: splitThreshold,
        one_player_under: splitThreshold
      };
    } else {
      delete rules.split_age_threshold;
      delete rules.split_age_rule;
    }
  }
  if (notes) rules.notes = notes;
  return JSON.stringify(rules);
}

/**
 * Project a guided builder draft onto the published registration contract.
 *
 * Streamlit builder drafts use human-readable `assigned_day` and `event_family`
 * fields. Draft saves must retain that exact editable shape, while impact review
 * and publish require registration day IDs and canonical event-option fields.
 * Canonical rows pass through unchanged unless a guided top-level age-rule edit
 * must be folded back into their persisted `age_rules` JSON.
 */
export function publishConfigurationPayload(configuration: SetupConfiguration): SetupPublishPayload {
  const draft = configurationPayload(configuration);
  const days = draft.days.map((row) => ({ ...row }));
  const dayIdsByLabel = new Map<string, string>();
  days.forEach((row, index) => {
    const publishedReference = cleanString(row.id) || `day_${index + 1}`;
    const labelKey = normalizedLookupKey(dayLabel(row));
    if (labelKey) dayIdsByLabel.set(labelKey, publishedReference);
    const idKey = normalizedLookupKey(row.id);
    if (idKey) dayIdsByLabel.set(idKey, publishedReference);
  });

  const familiesByName = new Map<string, SetupRecord>();
  draft.event_families.forEach((row) => {
    const key = normalizedLookupKey(eventFamilyName(row));
    if (key) familiesByName.set(key, row);
  });

  const eventOptions = draft.event_options.map((row, index) => {
    const usesLegacyShape = eventUsesLegacyBuilderShape(row);
    if (!usesLegacyShape) return projectCanonicalAgeRuleEdits(row);

    const familyName = eventFamilyName(row);
    const defaults = familiesByName.get(normalizedLookupKey(familyName)) || {};
    const assignedDay = eventDayReference(row);
    const registrationDayId = dayIdsByLabel.get(normalizedLookupKey(assignedDay))
      || cleanString(row.registration_day_id)
      || assignedDay;
    const eventType = firstCleanString(
      row.event_type,
      row.participant_type,
      defaults.participant_type,
      "SINGLES"
    ).toUpperCase();
    const genderRestriction = firstCleanString(
      row.gender_restriction,
      defaults.gender_restriction,
      "ANY"
    ).toUpperCase();
    const divisionName = eventDivisionName(row);
    const skillLabel = cleanString(row.skill_label) || "Open";
    const ageMode = eventAgeMode(row);
    const ageLabel = cleanString(row.age_label) || "All Ages";
    const capacity = finiteNumber(row.capacity_teams)
      ?? finiteNumber(defaults.default_capacity_teams)
      ?? 16;
    const price = finiteNumber(row.price_usd)
      ?? finiteNumber(defaults.default_price_usd)
      ?? 0;
    const defaultWaitlist = recordBoolean(defaults.default_waitlist, true);
    const defaultPartnerBoard = recordBoolean(
      defaults.default_partner_board,
      true
    );
    const waitlistEnabled = booleanWithDefault(row, "waitlist_enabled", defaultWaitlist);
    const partnerBoardEnabled = booleanWithDefault(
      row,
      "partner_board_enabled",
      defaultPartnerBoard
    );
    const canonical: SetupRecord = {
      registration_day_id: registrationDayId,
      sort_order: row.sort_order || index + 1,
      label: cleanString(row.label) || divisionName || familyName || `Event ${index + 1}`,
      event_type: eventType,
      gender_restriction: genderRestriction,
      skill_label: skillLabel,
      skill_mode: cleanString(row.skill_mode)
        || (skillLabel.toLowerCase() === "open" ? "OPEN" : "SKILL_BRACKET"),
      age_label: ageLabel,
      age_mode: ageMode,
      age_rules: legacyAgeRules(row),
      partner_required: booleanWithDefault(row, "partner_required", eventType !== "SINGLES"),
      capacity_teams: capacity,
      public_partner_board: booleanWithDefault(
        row,
        "public_partner_board",
        partnerBoardEnabled
      ),
      price_usd: price,
      event_family_label: familyName,
      division_name: divisionName,
      event_format_default: firstCleanString(
        row.event_format_default,
        defaults.default_format,
        "ROUND_ROBIN_PLUS_PLAYOFF"
      ),
      scoring_default: firstCleanString(
        row.scoring_default,
        defaults.default_scoring,
        "GAME_TO_15"
      ),
      event_format_override: cleanString(row.division_format)
        || cleanString(row.event_format_override)
        || null,
      scoring_override: cleanString(row.division_scoring)
        || cleanString(row.scoring_override)
        || null,
      waitlist_enabled: waitlistEnabled,
      partner_board_enabled: partnerBoardEnabled,
      status: firstCleanString(row.status, "draft").toLowerCase(),
      enabled: booleanWithDefault(row, "enabled", true)
    };
    for (const key of ["id", "tournament_id"] as const) {
      if (hasValue(row, key)) canonical[key] = row[key];
    }
    return canonical;
  });

  return { days, event_options: eventOptions };
}

export function validateSetupConfiguration(configuration: SetupConfiguration): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  const days = rowsToPayload(configuration.days);
  const families = rowsToPayload(configuration.eventFamilies);
  const events = rowsToPayload(configuration.eventOptions);

  if (!days.length) issues.push({ path: "days", message: "Add at least one tournament day." });
  const dayLabels = days.map(dayLabel);
  const duplicateDayLabels = duplicateIndexes(dayLabels);
  const dayIds = days.map((row) => cleanString(row.id));
  const duplicateDayIds = duplicateIndexes(dayIds);
  const enabledDayReferences = new Set<string>();
  days.forEach((row, index) => {
    const label = dayLabel(row);
    if (!label) issues.push({ path: `days.${index}.label`, message: "Day label is required." });
    if (duplicateDayLabels.has(index)) issues.push({ path: `days.${index}.label`, message: "Day labels must be unique." });
    if (duplicateDayIds.has(index)) issues.push({ path: `days.${index}.id`, message: "Day IDs must be unique." });
    const dateValue = cleanString(row.event_date ?? row.date ?? row.start_date);
    if (dateValue && !isIsoDate(dateValue)) {
      issues.push({ path: `days.${index}.event_date`, message: "Use a valid date in YYYY-MM-DD format." });
    }
    if (recordBoolean(row.enabled, true)) {
      if (cleanString(row.id)) enabledDayReferences.add(cleanString(row.id));
      if (label) enabledDayReferences.add(label);
    }
  });
  if (days.length && !days.some((row) => recordBoolean(row.enabled, true))) {
    issues.push({ path: "days", message: "Enable at least one tournament day." });
  }

  const familyNames = families.map(eventFamilyName);
  const definedFamilyNames = new Set(familyNames.map(normalizedLookupKey).filter(Boolean));
  const duplicateFamilies = duplicateIndexes(familyNames);
  const familyIds = families.map((row) => cleanString(row.id));
  const duplicateFamilyIds = duplicateIndexes(familyIds);
  families.forEach((row, index) => {
    if (!eventFamilyName(row)) issues.push({ path: `families.${index}.event_family`, message: "Event name is required." });
    if (duplicateFamilies.has(index)) issues.push({ path: `families.${index}.event_family`, message: "Event names must be unique." });
    if (duplicateFamilyIds.has(index)) issues.push({ path: `families.${index}.id`, message: "Event-family IDs must be unique." });
    const capacity = finiteNumber(row.default_capacity_teams);
    if (capacity != null && (!Number.isInteger(capacity) || capacity < 1)) {
      issues.push({ path: `families.${index}.default_capacity_teams`, message: "Default capacity must be a whole number of at least 1." });
    }
    const price = finiteNumber(row.default_price_usd);
    if (price != null && price < 0) {
      issues.push({ path: `families.${index}.default_price_usd`, message: "Default price cannot be negative." });
    }
  });

  if (!events.length) issues.push({ path: "events", message: "Add at least one division." });
  const eventIds = events.map((row) => cleanString(row.id));
  const duplicateEventIds = duplicateIndexes(eventIds);
  const conflictKeys = events.map((row) => [
    eventDayReference(row),
    eventFamilyName(row),
    eventDivisionName(row)
  ].map((part) => part.trim().toLowerCase()).join("|"));
  const duplicateEvents = duplicateIndexes(conflictKeys);

  events.forEach((row, index) => {
    const name = eventDivisionName(row);
    const family = eventFamilyName(row);
    const day = eventDayReference(row);
    const usesLegacyShape = eventUsesLegacyBuilderShape(row);
    if (!name) issues.push({ path: `events.${index}.division_name`, message: "Division name is required." });
    if (!family) issues.push({ path: `events.${index}.event_family`, message: "Event family is required." });
    else if (usesLegacyShape && !definedFamilyNames.has(normalizedLookupKey(family))) {
      issues.push({
        path: `events.${index}.event_family`,
        message: "Choose an event family with defined defaults."
      });
    }
    if (!day) issues.push({ path: `events.${index}.registration_day_id`, message: "Assigned day is required." });
    else if (!enabledDayReferences.has(day)) {
      issues.push({ path: `events.${index}.registration_day_id`, message: "Choose an enabled tournament day." });
    }
    if (duplicateEventIds.has(index)) issues.push({ path: `events.${index}.id`, message: "Division IDs must be unique." });
    if (duplicateEvents.has(index)) {
      issues.push({ path: `events.${index}.division_name`, message: "This day, event, and division combination is duplicated." });
    }

    const capacity = finiteNumber(row.capacity_teams);
    if (capacity == null || !Number.isInteger(capacity) || capacity < 1) {
      issues.push({ path: `events.${index}.capacity_teams`, message: "Capacity must be a whole number of at least 1." });
    }
    const price = finiteNumber(row.price_usd);
    if (price == null || price < 0) {
      issues.push({ path: `events.${index}.price_usd`, message: "Price must be zero or greater." });
    }
    const ageRulesState = readAgeRules(row);
    if (hasValue(row, "age_rules") && !ageRulesState.valid) {
      issues.push({
        path: `events.${index}.age_rules`,
        message: "Age rules must be a valid JSON object."
      });
    }
    const ageMode = eventAgeMode(row);
    if (ageMode === "AUTO_AGE_SPLIT") {
      const minimum = finiteNumber(ageRuleValue(row, "min_teams_per_age_group"));
      if (minimum == null || !Number.isInteger(minimum) || minimum < 1) {
        issues.push({ path: `events.${index}.min_teams_per_age_group`, message: "Auto age split needs a whole-number minimum of at least 1 team per age group." });
      }
    }
    if (ageMode === "SPLIT_AGE") {
      const threshold = finiteNumber(ageRuleValue(row, "split_age_threshold"));
      if (threshold == null || !Number.isInteger(threshold) || threshold < 1) {
        issues.push({ path: `events.${index}.split_age_threshold`, message: "Split age needs a whole-number threshold of at least 1." });
      }
    }
  });

  return issues;
}

export function issuesForPath(issues: ValidationIssue[], pathPrefix: string): ValidationIssue[] {
  return issues.filter((issue) => issue.path === pathPrefix || issue.path.startsWith(`${pathPrefix}.`));
}

export function draftSignature(configuration: SetupConfiguration): string {
  const payload = configurationPayload(configuration);
  return JSON.stringify({ days: payload.days, events: payload.event_options });
}
