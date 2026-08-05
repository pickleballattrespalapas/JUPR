import {
  cleanString,
  type BuilderRow,
  type SetupConfiguration,
  type SetupRecord
} from "../../tournament-setup/tournamentSetupBuilder";

export type VenueCourt = {
  id: string;
  title: string;
};

const MAX_VENUE_COURTS = 100;

function deterministicCourtId(index: number): string {
  return `venue-court-${index + 1}`;
}

function randomCourtId(): string {
  const random = globalThis.crypto?.randomUUID?.();
  return random
    ? `venue-court-${random.replaceAll("-", "").slice(0, 16)}`
    : `venue-court-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function objectRows(value: unknown): SetupRecord[] {
  return Array.isArray(value)
    ? value.filter(
        (row): row is SetupRecord =>
          Boolean(row) && typeof row === "object" && !Array.isArray(row)
      )
    : [];
}

function uniqueId(candidate: string, used: Set<string>, index: number): string {
  let next = candidate || deterministicCourtId(index);
  let suffix = 2;
  while (used.has(next)) {
    next = `${candidate || deterministicCourtId(index)}-${suffix}`;
    suffix += 1;
  }
  used.add(next);
  return next;
}

export function normalizeVenueCourts(
  settings: Record<string, unknown>,
  days: BuilderRow[] | SetupRecord[] = []
): VenueCourt[] {
  const explicitRows = objectRows(settings.venue_courts_json);
  const used = new Set<string>();
  if (explicitRows.length) {
    return explicitRows.slice(0, MAX_VENUE_COURTS).map((row, index) => ({
      id: uniqueId(cleanString(row.id), used, index),
      title: row.title == null ? "" : String(row.title)
    }));
  }

  const firstDay = days[0] && "value" in days[0]
    ? (days[0] as BuilderRow).value
    : ((days[0] as SetupRecord | undefined) || {});
  const labels = Array.isArray(settings.venue_court_labels)
    ? settings.venue_court_labels.map((value) => (value == null ? "" : String(value)))
    : Array.isArray(firstDay.court_labels)
      ? firstDay.court_labels.map((value) => (value == null ? "" : String(value)))
      : [];
  const rawCount = Number(settings.venue_court_count ?? firstDay.court_count ?? labels.length ?? 10);
  const count = Number.isInteger(rawCount)
    ? Math.max(1, Math.min(MAX_VENUE_COURTS, rawCount))
    : 10;
  return Array.from({ length: count }, (_, index) => ({
    id: deterministicCourtId(index),
    title: labels[index] == null ? "" : String(labels[index])
  }));
}

export function newVenueCourt(): VenueCourt {
  return { id: randomCourtId(), title: "" };
}

export function courtDisplayName(court: VenueCourt, index: number): string {
  return court.title.trim() || `Court ${index + 1}`;
}

export function settingsWithVenueCourts(
  settings: Record<string, unknown>,
  courts: VenueCourt[]
): Record<string, unknown> {
  const safeCourts = courts.slice(0, MAX_VENUE_COURTS).map((court, index) => ({
    id: cleanString(court.id) || deterministicCourtId(index),
    title: court.title == null ? "" : String(court.title)
  }));
  return {
    ...settings,
    venue_courts_json: safeCourts,
    venue_court_count: safeCourts.length,
    venue_court_labels: safeCourts.map((court) => court.title)
  };
}

export function dayAvailableCourtIds(
  day: SetupRecord,
  courts: VenueCourt[]
): string[] {
  const valid = new Set(courts.map((court) => court.id));
  const explicit = Array.isArray(day.available_court_ids)
    ? [...new Set(day.available_court_ids.map(cleanString).filter((value) => valid.has(value)))]
    : [];
  if (explicit.length) return explicit;

  const legacyCount = Number(day.court_count);
  if (Number.isInteger(legacyCount) && legacyCount >= 1 && legacyCount < courts.length) {
    return courts.slice(0, legacyCount).map((court) => court.id);
  }
  return courts.map((court) => court.id);
}

export function withVenueCourtAvailability(
  day: SetupRecord,
  courts: VenueCourt[],
  requestedIds?: readonly string[]
): SetupRecord {
  const valid = new Set(courts.map((court) => court.id));
  const ids = [...new Set((requestedIds || dayAvailableCourtIds(day, courts)).map(cleanString))]
    .filter((value) => valid.has(value));
  const selected = ids.length ? ids : courts.slice(0, 1).map((court) => court.id);
  const indexById = new Map(courts.map((court, index) => [court.id, index]));
  const labels = selected.map((id) => {
    const index = indexById.get(id) ?? 0;
    return courtDisplayName(courts[index], index);
  });
  return {
    ...day,
    available_court_ids: selected,
    court_count: selected.length,
    court_labels: labels,
    court_open_time: null,
    court_close_time: null,
    court_notes: ""
  };
}

export function configurationWithVenueInventory(
  configuration: SetupConfiguration,
  settings: Record<string, unknown>
): SetupConfiguration {
  const courts = normalizeVenueCourts(settings, configuration.days);
  return {
    ...configuration,
    days: configuration.days.map((row) => ({
      ...row,
      value: withVenueCourtAvailability(row.value, courts)
    }))
  };
}

export function venueIssues(
  settings: Record<string, unknown>,
  configuration: SetupConfiguration
): string[] {
  const courts = normalizeVenueCourts(settings, configuration.days);
  const issues: string[] = [];
  if (!cleanString(settings.venue_address)) {
    issues.push("Venue address is required.");
  }
  if (!courts.length) issues.push("Add at least one venue court.");
  if (courts.length > MAX_VENUE_COURTS) {
    issues.push(`Venue court inventory cannot exceed ${MAX_VENUE_COURTS} courts.`);
  }
  const ids = courts.map((court) => court.id);
  if (new Set(ids).size !== ids.length) issues.push("Venue court IDs must be unique.");
  const titled = courts.map((court) => court.title.trim().toLowerCase()).filter(Boolean);
  if (new Set(titled).size !== titled.length) issues.push("Optional court titles must be unique.");
  const valid = new Set(ids);
  configuration.days.forEach((row, index) => {
    const selected = dayAvailableCourtIds(row.value, courts);
    if (!selected.length) issues.push(`Tournament day ${index + 1} needs at least one available court.`);
    if (selected.some((id) => !valid.has(id))) {
      issues.push(`Tournament day ${index + 1} references a court that is no longer in the venue inventory.`);
    }
  });
  return [...new Set(issues)];
}
