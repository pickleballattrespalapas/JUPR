import assert from "node:assert/strict";
import { resolve } from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";

const compiledModulePath = process.env.JUPR_TOURNAMENT_SETUP_BUILDER_MODULE;
const builderModuleUrl = compiledModulePath
  ? pathToFileURL(resolve(compiledModulePath)).href
  : new URL("./tournamentSetupBuilder.ts", import.meta.url).href;

const {
  ageRuleValue,
  appendBuilderRow,
  configurationPayload,
  draftSignature,
  effectiveGenderRestriction,
  effectiveParticipantType,
  moveBuilderRow,
  newDayRow,
  newEventOptionRow,
  parseAdvancedConfiguration,
  publishConfigurationPayload,
  removeBuilderRow,
  setAgeRuleNumber,
  setEventAgeMode,
  setRecordString,
  validateSetupConfiguration,
  wrapBuilderRows
} = await import(builderModuleUrl);

function validConfiguration() {
  return {
    days: wrapBuilderRows([
      {
        id: "day-1",
        label: "Friday",
        event_date: "2026-11-20",
        enabled: true,
        sort_order: 1,
        court_count: 10,
        court_labels: Array.from({ length: 10 }, (_, index) => `Court ${index + 1}`),
        court_open_time: "08:00",
        court_close_time: "20:00"
      }
    ], "day"),
    eventFamilies: wrapBuilderRows([
      {
        id: "family-1",
        event_family: "Mixed Doubles",
        participant_type: "MIXED_DOUBLES",
        gender_restriction: "MIXED",
        default_capacity_teams: 16,
        default_price_usd: 0
      }
    ], "family"),
    eventOptions: wrapBuilderRows([
      {
        id: "event-1",
        registration_day_id: "day-1",
        event_family_label: "Mixed Doubles",
        division_name: "Mixed Doubles Open",
        event_type: "MIXED_DOUBLES",
        gender_restriction: "MIXED",
        skill_label: "Open",
        age_mode: "ALL_AGES",
        age_label: "All Ages",
        capacity_teams: 16,
        price_usd: 0,
        status: "open",
        enabled: true
      }
    ], "division")
  };
}

test("guided rows preserve the API payload exactly until an operator edits them", () => {
  const source = {
    days: [{
      id: "day-1",
      tournament_id: "t-1",
      label: "Friday",
      event_date: "2026-11-20",
      enabled: true,
      extension: { retained: ["yes"] }
    }],
    event_families: [{
      id: "family-1",
      event_family: "Mixed Doubles",
      custom_default: "retained"
    }],
    event_options: [{
      id: "event-1",
      registration_day_id: "day-1",
      event_family_label: "Mixed Doubles",
      division_name: "Open",
      capacity_teams: 16,
      price_usd: 0,
      unknown_backend_field: { value: 42 }
    }]
  };
  const configuration = {
    days: wrapBuilderRows(source.days, "day"),
    eventFamilies: wrapBuilderRows(source.event_families, "family"),
    eventOptions: wrapBuilderRows(source.event_options, "division")
  };

  assert.deepEqual(configurationPayload(configuration), source);
  assert.deepEqual(publishConfigurationPayload(configuration), {
    days: source.days,
    event_options: source.event_options
  });
});

test("publish projection converts a multi-day legacy draft without changing its saved shape", () => {
  const source = {
    days: [
      { id: "day-friday", label: "Friday", event_date: "2026-11-20", enabled: true, sort_order: 1 },
      { id: "day-saturday", label: "Saturday", event_date: "2026-11-21", enabled: true, sort_order: 2 }
    ],
    event_families: [
      {
        id: "family-mixed",
        event_family: "Mixed Doubles",
        participant_type: "MIXED_DOUBLES",
        gender_restriction: "MIXED",
        default_format: "ROUND_ROBIN_PLUS_PLAYOFF",
        default_scoring: "GAME_TO_15",
        default_waitlist: false,
        default_partner_board: true,
        default_capacity_teams: 16,
        default_price_usd: 25,
        default_status: "open"
      },
      {
        id: "family-singles",
        event_family: "Singles",
        participant_type: "SINGLES",
        gender_restriction: "ANY",
        default_format: "SINGLE_ELIM",
        default_scoring: "GAME_TO_21",
        default_waitlist: true,
        default_partner_board: false,
        default_capacity_teams: 12,
        default_price_usd: 15,
        default_status: "tentative"
      }
    ],
    event_options: [
      {
        id: "division-mixed",
        event_family: "Mixed Doubles",
        division_name: "Mixed 3.5",
        skill_label: "3.5",
        age_mode: "ALL_AGES",
        age_label: "All Ages",
        assigned_day: "Friday",
        capacity_teams: 14,
        price_usd: 30,
        division_format: "DOUBLE_ELIM",
        division_scoring: "",
        sort_order: 1
      },
      {
        id: "division-singles",
        event_family: "Singles",
        division_name: "Singles 50+",
        skill_label: "Open",
        age_mode: "SPLIT_AGE",
        age_label: "50+",
        split_age_threshold: 50,
        assigned_day: "Saturday",
        capacity_teams: 10,
        price_usd: 20,
        waitlist_enabled: false,
        status: "closed",
        sort_order: 2
      }
    ]
  };
  const configuration = {
    days: wrapBuilderRows(source.days, "day"),
    eventFamilies: wrapBuilderRows(source.event_families, "family"),
    eventOptions: wrapBuilderRows(source.event_options, "division")
  };
  const savedDraftBeforeProjection = configurationPayload(configuration);

  assert.deepEqual(publishConfigurationPayload(configuration), {
    days: source.days,
    event_options: [
      {
        id: "division-mixed",
        registration_day_id: "day-friday",
        scheduled_day_ids: ["day-friday"],
        sort_order: 1,
        label: "Mixed 3.5",
        event_type: "MIXED_DOUBLES",
        gender_restriction: "MIXED",
        skill_label: "3.5",
        skill_mode: "SKILL_BRACKET",
        age_label: "All Ages",
        age_mode: "ALL_AGES",
        age_rules: null,
        partner_required: true,
        capacity_teams: 14,
        public_partner_board: true,
        price_usd: 30,
        event_family_label: "Mixed Doubles",
        division_name: "Mixed 3.5",
        event_format_default: "ROUND_ROBIN_PLUS_PLAYOFF",
        scoring_default: "GAME_TO_15",
        event_format_override: "DOUBLE_ELIM",
        scoring_override: null,
        waitlist_enabled: false,
        partner_board_enabled: true,
        status: "draft",
        enabled: true
      },
      {
        id: "division-singles",
        registration_day_id: "day-saturday",
        scheduled_day_ids: ["day-saturday"],
        sort_order: 2,
        label: "Singles 50+",
        event_type: "SINGLES",
        gender_restriction: "ANY",
        skill_label: "Open",
        skill_mode: "OPEN",
        age_label: "50+",
        age_mode: "SPLIT_AGE",
        age_rules: JSON.stringify({
          mode: "SPLIT_AGE",
          younger_player_controls_age: true,
          higher_skill_player_controls_skill: true,
          age_label: "50+",
          split_age_threshold: 50,
          split_age_rule: {
            one_player_over_or_equal: 50,
            one_player_under: 50
          }
        }),
        partner_required: false,
        capacity_teams: 10,
        public_partner_board: false,
        price_usd: 20,
        event_family_label: "Singles",
        division_name: "Singles 50+",
        event_format_default: "SINGLE_ELIM",
        scoring_default: "GAME_TO_21",
        event_format_override: null,
        scoring_override: null,
        waitlist_enabled: false,
        partner_board_enabled: false,
        status: "closed",
        enabled: true
      }
    ]
  });
  assert.deepEqual(configurationPayload(configuration), savedDraftBeforeProjection);
  assert.equal(Object.hasOwn(configuration.eventOptions[0].value, "registration_day_id"), false);
  assert.equal(configuration.eventOptions[1].value.assigned_day, "Saturday");
});

test("structured field changes preserve unrelated and legacy fields", () => {
  const row = {
    assigned_day: "Friday",
    division_name: "3.5",
    extension: { retained: true }
  };
  const changed = setRecordString(row, ["registration_day_id", "assigned_day"], "Saturday");

  assert.equal(changed.assigned_day, "Saturday");
  assert.equal(Object.hasOwn(changed, "registration_day_id"), false);
  assert.deepEqual(changed.extension, { retained: true });
});

test("canonical AUTO_AGE_SPLIT rules display, edit, save, and publish without data loss", () => {
  const configuration = validConfiguration();
  const originalRules = {
    mode: "AUTO_AGE_SPLIT",
    min_teams_per_age_group: 4,
    custom_policy: { retained: true }
  };
  const original = {
    ...configuration.eventOptions[0].value,
    age_mode: "AUTO_AGE_SPLIT",
    age_rules: JSON.stringify(originalRules)
  };
  configuration.eventOptions[0].value = original;

  assert.equal(ageRuleValue(original, "min_teams_per_age_group"), 4);
  assert.deepEqual(validateSetupConfiguration(configuration), []);

  const edited = setAgeRuleNumber(original, "min_teams_per_age_group", "6");
  const editedRules = JSON.parse(edited.age_rules);
  assert.equal(Object.hasOwn(edited, "min_teams_per_age_group"), false);
  assert.equal(editedRules.min_teams_per_age_group, 6);
  assert.deepEqual(editedRules.custom_policy, { retained: true });
  configuration.eventOptions[0].value = edited;

  assert.equal(
    configurationPayload(configuration).event_options[0].age_rules,
    edited.age_rules
  );
  assert.equal(
    publishConfigurationPayload(configuration).event_options[0].age_rules,
    edited.age_rules
  );
  assert.deepEqual(validateSetupConfiguration(configuration), []);
});

test("canonical SPLIT_AGE nested rules round-trip and mode edits preserve extensions", () => {
  const configuration = validConfiguration();
  const original = {
    ...configuration.eventOptions[0].value,
    age_mode: "SPLIT_AGE",
    age_rules: JSON.stringify({
      mode: "SPLIT_AGE",
      split_age_rule: {
        one_player_over_or_equal: 50,
        one_player_under: 50,
        custom_nested: "keep"
      },
      min_teams_per_age_group: 4,
      custom_policy: "keep"
    })
  };
  configuration.eventOptions[0].value = original;

  assert.equal(ageRuleValue(original, "split_age_threshold"), 50);
  assert.deepEqual(validateSetupConfiguration(configuration), []);

  const edited = setAgeRuleNumber(original, "split_age_threshold", "55");
  const editedRules = JSON.parse(edited.age_rules);
  assert.equal(editedRules.split_age_threshold, 55);
  assert.equal(editedRules.split_age_rule.one_player_over_or_equal, 55);
  assert.equal(editedRules.split_age_rule.one_player_under, 55);
  assert.equal(editedRules.split_age_rule.custom_nested, "keep");
  assert.equal(editedRules.custom_policy, "keep");

  const switched = setEventAgeMode(edited, "AUTO_AGE_SPLIT");
  const switchedRules = JSON.parse(switched.age_rules);
  assert.equal(switched.age_mode, "AUTO_AGE_SPLIT");
  assert.equal(switchedRules.mode, "AUTO_AGE_SPLIT");
  assert.equal(switchedRules.min_teams_per_age_group, 4);
  assert.equal(Object.hasOwn(switchedRules, "split_age_threshold"), false);
  assert.equal(Object.hasOwn(switchedRules, "split_age_rule"), false);
  assert.equal(switchedRules.custom_policy, "keep");

  const allAges = setEventAgeMode(switched, "ALL_AGES");
  const allAgesRules = JSON.parse(allAges.age_rules);
  assert.equal(allAgesRules.mode, "ALL_AGES");
  assert.equal(Object.hasOwn(allAgesRules, "min_teams_per_age_group"), false);
  assert.equal(Object.hasOwn(allAgesRules, "split_age_rule"), false);
  assert.equal(allAgesRules.custom_policy, "keep");
});

test("canonical age validation requires valid mode-specific persisted rules", () => {
  const auto = validConfiguration();
  auto.eventOptions[0].value = {
    ...auto.eventOptions[0].value,
    age_mode: "AUTO_AGE_SPLIT",
    age_rules: JSON.stringify({ mode: "AUTO_AGE_SPLIT" })
  };
  assert.ok(
    validateSetupConfiguration(auto)
      .map((issue) => issue.message)
      .includes("Auto age split needs a whole-number minimum of at least 1 team per age group.")
  );

  const split = validConfiguration();
  split.eventOptions[0].value = {
    ...split.eventOptions[0].value,
    age_mode: "SPLIT_AGE",
    age_rules: JSON.stringify({ mode: "SPLIT_AGE", split_age_threshold: 0 })
  };
  assert.ok(
    validateSetupConfiguration(split)
      .map((issue) => issue.message)
      .includes("Split age needs a whole-number threshold of at least 1.")
  );

  const malformed = validConfiguration();
  malformed.eventOptions[0].value = {
    ...malformed.eventOptions[0].value,
    age_mode: "AUTO_AGE_SPLIT",
    age_rules: "{not-json"
  };
  const malformedMessages = validateSetupConfiguration(malformed).map((issue) => issue.message);
  assert.ok(malformedMessages.includes("Age rules must be a valid JSON object."));
  assert.ok(malformedMessages.includes("Auto age split needs a whole-number minimum of at least 1 team per age group."));
});

test("canonical top-level age edits are folded into age_rules before publish", () => {
  const configuration = validConfiguration();
  configuration.eventOptions[0].value = {
    ...configuration.eventOptions[0].value,
    age_mode: "AUTO_AGE_SPLIT",
    age_rules: JSON.stringify({
      mode: "AUTO_AGE_SPLIT",
      min_teams_per_age_group: 4,
      extension: "keep"
    }),
    min_teams_per_age_group: 7
  };

  const published = publishConfigurationPayload(configuration).event_options[0];
  const publishedRules = JSON.parse(published.age_rules);
  assert.equal(Object.hasOwn(published, "min_teams_per_age_group"), false);
  assert.equal(publishedRules.min_teams_per_age_group, 7);
  assert.equal(publishedRules.extension, "keep");
});

test("legacy divisions display family defaults and explicit overrides win at publish", () => {
  const configuration = validConfiguration();
  const legacy = {
    id: "legacy-effective-defaults",
    assigned_day: "Friday",
    event_family: "Mixed Doubles",
    division_name: "Mixed Open",
    capacity_teams: 16,
    price_usd: 0
  };
  assert.equal(effectiveParticipantType(legacy, configuration.eventFamilies), "MIXED_DOUBLES");
  assert.equal(effectiveGenderRestriction(legacy, configuration.eventFamilies), "MIXED");

  const overridden = setRecordString(
    setRecordString(legacy, ["event_type"], "SINGLES"),
    ["gender_restriction"],
    "ANY"
  );
  assert.equal(effectiveParticipantType(overridden, configuration.eventFamilies), "SINGLES");
  assert.equal(effectiveGenderRestriction(overridden, configuration.eventFamilies), "ANY");
  configuration.eventOptions = wrapBuilderRows([overridden], "division");

  const published = publishConfigurationPayload(configuration).event_options[0];
  assert.equal(published.event_type, "SINGLES");
  assert.equal(published.gender_restriction, "ANY");
  assert.equal(published.partner_required, false);
});

test("legacy families without participant type retain the Streamlit SINGLES fallback", () => {
  const configuration = validConfiguration();
  configuration.eventFamilies = wrapBuilderRows([
    {
      id: "family-without-participant-type",
      event_family: "Open play",
      gender_restriction: "ANY"
    }
  ], "family");
  const legacy = {
    id: "legacy-without-participant-type",
    assigned_day: "Friday",
    event_family: "Open play",
    division_name: "Open",
    capacity_teams: 16,
    price_usd: 0
  };
  configuration.eventOptions = wrapBuilderRows([legacy], "division");

  assert.equal(effectiveParticipantType(legacy, configuration.eventFamilies), "SINGLES");
  const published = publishConfigurationPayload(configuration).event_options[0];
  assert.equal(published.event_type, "SINGLES");
  assert.equal(published.partner_required, false);
  assert.equal(published.partner_board_enabled, true);
  assert.equal(published.public_partner_board, true);
  assert.equal(published.status, "draft");
});

test("valid configuration passes and duplicate/conflicting rows are rejected", () => {
  const configuration = validConfiguration();
  assert.deepEqual(validateSetupConfiguration(configuration), []);

  const duplicateDay = {
    ...configuration.days[0],
    key: "duplicate-day",
    value: { ...configuration.days[0].value, id: "day-2", label: " friday " }
  };
  const duplicateEvent = {
    ...configuration.eventOptions[0],
    key: "duplicate-event",
    value: { ...configuration.eventOptions[0].value, id: "event-2" }
  };
  const invalid = {
    ...configuration,
    days: [...configuration.days, duplicateDay],
    eventOptions: [...configuration.eventOptions, duplicateEvent]
  };
  const messages = validateSetupConfiguration(invalid).map((issue) => issue.message);

  assert.ok(messages.includes("Day labels must be unique."));
  assert.ok(messages.includes("This event and division combination is duplicated."));
});

test("validation reports bad references and typed numeric errors", () => {
  const configuration = validConfiguration();
  configuration.eventOptions[0].value.registration_day_id = "missing-day";
  configuration.eventOptions[0].value.capacity_teams = 0;
  configuration.eventOptions[0].value.price_usd = -1;

  const messages = validateSetupConfiguration(configuration).map((issue) => issue.message);
  assert.ok(messages.includes("Choose an enabled tournament day."));
  assert.ok(messages.includes("Capacity must be a whole number of at least 1."));
  assert.ok(messages.includes("Price must be zero or greater."));
});

test("legacy divisions require matching defaults while canonical rows do not", () => {
  const legacyConfiguration = validConfiguration();
  legacyConfiguration.eventOptions = wrapBuilderRows([
    {
      id: "legacy-1",
      assigned_day: "Friday",
      event_family: "Missing family",
      division_name: "Open",
      capacity_teams: 16,
      price_usd: 0
    }
  ], "division");
  assert.ok(
    validateSetupConfiguration(legacyConfiguration)
      .map((issue) => issue.message)
      .includes("Choose an event family with defined defaults.")
  );

  const canonicalConfiguration = validConfiguration();
  canonicalConfiguration.eventFamilies = [];
  assert.deepEqual(validateSetupConfiguration(canonicalConfiguration), []);
});

test("add, remove, and reorder update only the intended local rows", () => {
  const original = wrapBuilderRows([
    { id: "one", label: "One", sort_order: 1 },
    { id: "two", label: "Two", sort_order: 2 }
  ], "day");
  const added = appendBuilderRow(original, "day", newDayRow(3));
  assert.equal(added.length, 3);
  assert.equal(original.length, 2);

  const moved = moveBuilderRow(added, added[2].key, -1);
  assert.equal(moved[1].value.label, "Day 3");
  assert.deepEqual(moved.map((row) => row.value.sort_order), [1, 2, 3]);

  const removed = removeBuilderRow(moved, moved[1].key);
  assert.deepEqual(removed.map((row) => row.value.id), ["one", "two"]);
});

test("advanced import is all-or-nothing and reports malformed data", () => {
  const previous = validConfiguration();
  const previousSignature = draftSignature(previous);
  assert.throws(
    () => parseAdvancedConfiguration('{"days":[],"event_families":[],"event_options":"bad"}'),
    /event_options must be a JSON array/
  );
  assert.equal(draftSignature(previous), previousSignature);

  const imported = parseAdvancedConfiguration(JSON.stringify({
    days: [{ id: "d" }],
    event_families: [],
    event_options: [{ id: "e" }]
  }));
  assert.deepEqual(imported, {
    days: [{ id: "d" }],
    event_families: [],
    event_options: [{ id: "e" }]
  });
});

test("new divisions use the current payload shape and sensible defaults", () => {
  const publishedConfiguration = validConfiguration();
  const published = newEventOptionRow(publishedConfiguration);
  assert.equal(published.registration_day_id, "day-1");
  assert.equal(published.event_family_label, "Mixed Doubles");
  assert.equal(published.capacity_teams, 16);
  assert.equal(published.division_name, "Mixed Doubles Division 2");

  const draftConfiguration = {
    ...publishedConfiguration,
    eventOptions: wrapBuilderRows([
      {
        id: "draft-1",
        event_family: "Mixed Doubles",
        assigned_day: "Friday",
        division_name: "Mixed 3.5"
      }
    ], "division")
  };
  const draft = newEventOptionRow(draftConfiguration);
  assert.equal(draft.assigned_day, "Friday");
  assert.deepEqual(draft.scheduled_day_ids, ["day-1"]);
  assert.equal(draft.event_family, "Mixed Doubles");
  assert.equal(Object.hasOwn(draft, "registration_day_id"), false);
});

test("draft signatures change after guided edits and remain stable otherwise", () => {
  const configuration = validConfiguration();
  const initial = draftSignature(configuration);
  assert.equal(initial, draftSignature(configuration));
  configuration.eventOptions[0].value.division_name = "Mixed 4.0";
  assert.notEqual(initial, draftSignature(configuration));
});
