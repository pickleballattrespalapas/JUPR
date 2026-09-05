"use client";

import { TournamentSetupAdvancedPanel } from "./TournamentSetupAdvancedPanel";
import { TournamentSetupDayCard } from "./TournamentSetupDayCard";
import { TournamentSetupDivisionCard } from "./TournamentSetupDivisionCard";
import { TournamentSetupFamilyCard } from "./TournamentSetupFamilyCard";
import {
  appendBuilderRow,
  cleanString,
  distinctFamilyNames,
  eventDayReference,
  eventDivisionName,
  eventFamilyName,
  issuesForPath,
  moveBuilderRow,
  newDayRow,
  newEventFamilyRow,
  newEventOptionRow,
  recordBoolean,
  removeBuilderRow,
  replaceBuilderRow,
  setRecordString,
  wrapBuilderRows,
  type SetupConfiguration,
  type SetupPayload,
  type SetupRecord,
  type ValidationIssue
} from "./tournamentSetupBuilder";
import styles from "./TournamentSetupBuilder.module.css";

type TournamentSetupBuilderProps = {
  configuration: SetupConfiguration;
  issues: ValidationIssue[];
  disabled: boolean;
  onChange: (configuration: SetupConfiguration) => void;
  onNotice: (message: string) => void;
};

export function TournamentSetupBuilder({
  configuration,
  issues,
  disabled,
  onChange,
  onNotice
}: TournamentSetupBuilderProps) {
  const familyNames = distinctFamilyNames(configuration);
  const enabledDays = configuration.days.filter((day) => recordBoolean(day.value.enabled, true));

  function nextUniqueName(prefix: string, existingNames: string[], startingPosition: number): string {
    const normalized = new Set(existingNames.map((name) => name.trim().toLowerCase()));
    let position = startingPosition;
    while (normalized.has(`${prefix} ${position}`.toLowerCase())) position += 1;
    return `${prefix} ${position}`;
  }

  function updateDay(key: string, nextValue: SetupRecord) {
    const previous = configuration.days.find((row) => row.key === key)?.value;
    const previousLabel = cleanString(previous?.label);
    const nextLabel = cleanString(nextValue.label);
    const nextEvents = previousLabel && previousLabel !== nextLabel
      ? configuration.eventOptions.map((row) => (
        cleanString(row.value.assigned_day) === previousLabel
          ? { ...row, value: setRecordString(row.value, ["assigned_day"], nextLabel) }
          : row
      ))
      : configuration.eventOptions;
    onChange({
      ...configuration,
      days: replaceBuilderRow(configuration.days, key, nextValue),
      eventOptions: nextEvents
    });
  }

  function removeDay(key: string) {
    const day = configuration.days.find((row) => row.key === key)?.value;
    if (!day) return;
    const references = new Set([cleanString(day.id), cleanString(day.label)].filter(Boolean));
    const dependent = configuration.eventOptions.find((row) => references.has(eventDayReference(row.value)));
    if (dependent) {
      onNotice(`Move or remove "${eventDivisionName(dependent.value) || "the attached division"}" before removing this day.`);
      return;
    }
    onChange({ ...configuration, days: removeBuilderRow(configuration.days, key) });
    onNotice("Day removed from the local draft.");
  }

  function updateFamily(key: string, nextValue: SetupRecord) {
    const previous = configuration.eventFamilies.find((row) => row.key === key)?.value;
    const previousName = previous ? eventFamilyName(previous) : "";
    const nextName = eventFamilyName(nextValue);
    const nextEvents = previousName && previousName !== nextName
      ? configuration.eventOptions.map((row) => (
        eventFamilyName(row.value).toLowerCase() === previousName.toLowerCase()
          ? { ...row, value: setRecordString(row.value, ["event_family", "event_family_label"], nextName) }
          : row
      ))
      : configuration.eventOptions;
    onChange({
      ...configuration,
      eventFamilies: replaceBuilderRow(configuration.eventFamilies, key, nextValue),
      eventOptions: nextEvents
    });
  }

  function removeFamily(key: string) {
    const family = configuration.eventFamilies.find((row) => row.key === key)?.value;
    if (!family) return;
    const name = eventFamilyName(family);
    const dependent = configuration.eventOptions.find((row) => eventFamilyName(row.value).toLowerCase() === name.toLowerCase());
    if (dependent) {
      onNotice(`Reassign or remove "${eventDivisionName(dependent.value) || "the attached division"}" before removing ${name || "this event"}.`);
      return;
    }
    onChange({ ...configuration, eventFamilies: removeBuilderRow(configuration.eventFamilies, key) });
    onNotice("Event defaults removed from the local draft.");
  }

  function applyAdvancedPayload(payload: SetupPayload) {
    onChange({
      days: wrapBuilderRows(payload.days, "day"),
      eventFamilies: wrapBuilderRows(payload.event_families, "family"),
      eventOptions: wrapBuilderRows(payload.event_options, "division")
    });
    onNotice("Advanced JSON imported into the local draft. Nothing has been saved yet.");
  }

  const schedule = configuration.days
    .filter((day) => recordBoolean(day.value.enabled, true))
    .map((day) => {
      const references = new Set([cleanString(day.value.id), cleanString(day.value.label)].filter(Boolean));
      return {
        key: day.key,
        label: cleanString(day.value.label) || "Untitled day",
        events: configuration.eventOptions.filter((event) => references.has(eventDayReference(event.value)))
      };
    });
  const assignedKeys = new Set(schedule.flatMap((day) => day.events.map((event) => event.key)));
  const unassignedEvents = configuration.eventOptions.filter((event) => !assignedKeys.has(event.key));

  return (
    <div className={styles.builder}>
      <section className={styles.section} aria-labelledby="setup-days-heading">
        <div className={styles.sectionHeader}>
          <div>
            <h3 id="setup-days-heading">Tournament days</h3>
            <p className={styles.sectionDescription}>Set the public day label, date, and order used by registration.</p>
          </div>
          <button
            type="button"
            className={styles.button}
            disabled={disabled}
            onClick={() => {
              const label = nextUniqueName(
                "Day",
                configuration.days.map((day) => cleanString(day.value.label)),
                configuration.days.length + 1
              );
              onChange({
                ...configuration,
                days: appendBuilderRow(configuration.days, "day", newDayRow(configuration.days.length + 1, label))
              });
            }}
          >
            Add day
          </button>
        </div>
        <div className={styles.rows}>
          {configuration.days.length ? configuration.days.map((row, index) => (
            <TournamentSetupDayCard
              key={row.key}
              row={row}
              position={index}
              total={configuration.days.length}
              disabled={disabled}
              issues={issuesForPath(issues, `days.${index}`)}
              onChange={(value) => updateDay(row.key, value)}
              onMove={(direction) => onChange({
                ...configuration,
                days: moveBuilderRow(configuration.days, row.key, direction)
              })}
              onRemove={() => removeDay(row.key)}
            />
          )) : <p className={styles.empty}>No days yet. Add a day before creating divisions.</p>}
        </div>
        {issuesForPath(issues, "days").filter((issue) => !/^days\.\d+\./.test(issue.path)).map((issue) => (
          <p key={`${issue.path}-${issue.message}`} className={styles.error}>{issue.message}</p>
        ))}
      </section>

      <section className={styles.section} aria-labelledby="setup-events-heading">
        <div className={styles.sectionHeader}>
          <div>
            <h3 id="setup-events-heading">Event defaults</h3>
            <p className={styles.sectionDescription}>
              Optional reusable defaults for singles, gender doubles, and mixed doubles. Existing payloads without event defaults remain supported.
            </p>
          </div>
          <button
            type="button"
            className={styles.button}
            disabled={disabled}
            onClick={() => {
              const name = nextUniqueName(
                "Event",
                configuration.eventFamilies.map((family) => eventFamilyName(family.value)),
                configuration.eventFamilies.length + 1
              );
              onChange({
                ...configuration,
                eventFamilies: appendBuilderRow(
                  configuration.eventFamilies,
                  "family",
                  newEventFamilyRow(configuration.eventFamilies.length + 1, name)
                )
              });
            }}
          >
            Add event
          </button>
        </div>
        <div className={styles.rows}>
          {configuration.eventFamilies.length ? configuration.eventFamilies.map((row, index) => (
            <TournamentSetupFamilyCard
              key={row.key}
              row={row}
              position={index}
              total={configuration.eventFamilies.length}
              disabled={disabled}
              issues={issuesForPath(issues, `families.${index}`)}
              onChange={(value) => updateFamily(row.key, value)}
              onMove={(direction) => onChange({
                ...configuration,
                eventFamilies: moveBuilderRow(configuration.eventFamilies, row.key, direction)
              })}
              onRemove={() => removeFamily(row.key)}
            />
          )) : <p className={styles.empty}>No separate event defaults. Division-level event settings below will be used.</p>}
        </div>
      </section>

      <section className={styles.section} aria-labelledby="setup-divisions-heading">
        <div className={styles.sectionHeader}>
          <div>
            <h3 id="setup-divisions-heading">Divisions and registration options</h3>
            <p className={styles.sectionDescription}>
              Assign each division to a day, then set participant rules, capacity, price, waitlist, and Players Needing Partners behavior.
            </p>
          </div>
          <button
            type="button"
            className={styles.button}
            disabled={disabled || !enabledDays.length}
            onClick={() => onChange({
              ...configuration,
              eventOptions: appendBuilderRow(configuration.eventOptions, "division", newEventOptionRow(configuration))
            })}
          >
            Add division
          </button>
        </div>
        {!enabledDays.length ? <p className={styles.error}>Enable at least one day before adding a division.</p> : null}
        <div className={styles.rows}>
          {configuration.eventOptions.length ? configuration.eventOptions.map((row, index) => (
            <TournamentSetupDivisionCard
              key={row.key}
              row={row}
              position={index}
              total={configuration.eventOptions.length}
              days={configuration.days}
              eventFamilies={configuration.eventFamilies}
              familyNames={familyNames}
              disabled={disabled}
              issues={issuesForPath(issues, `events.${index}`)}
              onChange={(value) => onChange({
                ...configuration,
                eventOptions: replaceBuilderRow(configuration.eventOptions, row.key, value)
              })}
              onMove={(direction) => onChange({
                ...configuration,
                eventOptions: moveBuilderRow(configuration.eventOptions, row.key, direction)
              })}
              onRemove={() => {
                onChange({ ...configuration, eventOptions: removeBuilderRow(configuration.eventOptions, row.key) });
                onNotice("Division removed from the local draft.");
              }}
            />
          )) : <p className={styles.empty}>No divisions yet. Add at least one division before saving or publishing.</p>}
        </div>
        {issuesForPath(issues, "events").filter((issue) => !/^events\.\d+\./.test(issue.path)).map((issue) => (
          <p key={`${issue.path}-${issue.message}`} className={styles.error}>{issue.message}</p>
        ))}
      </section>

      <section className={styles.summary} aria-labelledby="setup-summary-heading">
        <h3 id="setup-summary-heading">Draft summary</h3>
        <div className={styles.summaryCounts}>
          <div className={styles.summaryCount}><strong>{configuration.days.length}</strong><br />day(s)</div>
          <div className={styles.summaryCount}><strong>{configuration.eventFamilies.length}</strong><br />event default(s)</div>
          <div className={styles.summaryCount}><strong>{configuration.eventOptions.length}</strong><br />division(s)</div>
        </div>
        <div className={styles.schedule}>
          {schedule.map((day) => (
            <div key={day.key} className={styles.scheduleDay}>
              <h4>{day.label}</h4>
              {day.events.length ? (
                <ul>
                  {day.events.map((event) => (
                    <li key={event.key}>
                      {eventDivisionName(event.value) || "Untitled division"}
                      {eventFamilyName(event.value) ? ` · ${eventFamilyName(event.value)}` : ""}
                      {cleanString(event.value.skill_label) ? ` · ${cleanString(event.value.skill_label)}` : ""}
                    </li>
                  ))}
                </ul>
              ) : <p className={styles.sectionDescription}>No divisions assigned.</p>}
            </div>
          ))}
          {unassignedEvents.length ? (
            <div className={styles.scheduleDay}>
              <h4>Needs day assignment</h4>
              <ul>{unassignedEvents.map((event) => <li key={event.key}>{eventDivisionName(event.value) || "Untitled division"}</li>)}</ul>
            </div>
          ) : null}
        </div>
        {issues.length ? (
          <>
            <p className={styles.error}><strong>Resolve {issues.length} validation issue(s) before saving or reviewing publish impact.</strong></p>
            <ul className={styles.issues}>
              {[...new Set(issues.map((issue) => issue.message))].map((message) => <li key={message}>{message}</li>)}
            </ul>
          </>
        ) : <p className={styles.success}><strong>Draft is ready to save or review.</strong></p>}
      </section>

      <TournamentSetupAdvancedPanel
        configuration={configuration}
        disabled={disabled}
        onApply={applyAdvancedPayload}
      />
    </div>
  );
}
