from __future__ import annotations

import re
from pathlib import Path


def sub_once(path: str, pattern: str, replacement: str) -> None:
    target = Path(path)
    text = target.read_text()
    next_text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"{path}: expected one regex match, found {count}: {pattern[:120]!r}")
    target.write_text(next_text)


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one match, found {count}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1))


builder = "apps/web/app/admin/tournament-setup/tournamentSetupBuilder.ts"
sub_once(
    builder,
    r'(  const scheduledDayIds = inheritedSchedule\.length \? inheritedSchedule : fallbackSchedule;\n)(  const existingNames =)',
    r'''\1  const primaryDay = configuration.days.find(
    (row) => dayReference(row.value) === scheduledDayIds[0]
  )?.value || firstDay;
\2''',
)
sub_once(
    builder,
    r'''      assigned_day: scheduledDayIds\[0\] \|\| dayLabel\(firstDay\),
      registration_day_id: scheduledDayIds\[0\] \|\| dayReference\(firstDay\),
      scheduled_day_ids: scheduledDayIds,''',
    '''      assigned_day: dayLabel(primaryDay) || scheduledDayIds[0] || dayLabel(firstDay),
      scheduled_day_ids: scheduledDayIds,''',
)
sub_once(
    builder,
    r'''      const primary = scheduledDayIds\[0\] \|\| cleanString\(projected\.registration_day_id\);
      return \{
        \.\.\.projected,
        registration_day_id: primary,
        scheduled_day_ids: scheduledDayIds\.length \? scheduledDayIds : \(primary \? \[primary\] : \[\]\)
      \};''',
    '''      const primary = scheduledDayIds[0] || cleanString(projected.registration_day_id);
      const next: SetupRecord = {
        ...projected,
        registration_day_id: primary
      };
      if (
        Object.prototype.hasOwnProperty.call(projected, "scheduled_day_ids") ||
        scheduledDayIds.length > 1
      ) {
        next.scheduled_day_ids = scheduledDayIds.length
          ? scheduledDayIds
          : (primary ? [primary] : []);
      }
      return next;''',
)
sub_once(
    builder,
    r'''    const scheduledDays = eventDayReferences\(row\);
    if \(!scheduledDays\.length\) \{
      issues\.push\(\{ path: `families\.\$\{index\}\.scheduled_day_ids`, message: "Choose at least one tournament day for this event\." \}\);
    \}
    for \(const scheduledDay of scheduledDays\) \{''',
    '''    const explicitScheduledDays = eventDayReferences(row);
    const inferredScheduledDays = events
      .filter(
        (event) =>
          normalizedLookupKey(eventFamilyName(event)) ===
          normalizedLookupKey(eventFamilyName(row))
      )
      .flatMap(eventDayReferences);
    const scheduledDays = explicitScheduledDays.length
      ? explicitScheduledDays
      : [...new Set(inferredScheduledDays)];
    if (!scheduledDays.length) {
      issues.push({ path: `families.${index}.scheduled_day_ids`, message: "Choose at least one tournament day for this event." });
    }
    for (const scheduledDay of scheduledDays) {''',
)
replace_once(
    builder,
    'issues.push({ path: `events.${index}.scheduled_day_ids`, message: "Choose only enabled tournament days." });',
    'issues.push({ path: `events.${index}.scheduled_day_ids`, message: "Choose an enabled tournament day." });',
)

tests = "apps/web/app/admin/tournament-setup/tournamentSetupBuilder.test.mjs"
replace_once(
    tests,
    '''        registration_day_id: "day-friday",
        sort_order: 1,''',
    '''        registration_day_id: "day-friday",
        scheduled_day_ids: ["day-friday"],
        sort_order: 1,''',
)
replace_once(
    tests,
    '''        registration_day_id: "day-saturday",
        sort_order: 2,''',
    '''        registration_day_id: "day-saturday",
        scheduled_day_ids: ["day-saturday"],
        sort_order: 2,''',
)
replace_once(
    tests,
    'assert.ok(messages.includes("This day, event, and division combination is duplicated."));',
    'assert.ok(messages.includes("This event and division combination is duplicated."));',
)
replace_once(
    tests,
    '''  assert.equal(draft.assigned_day, "Friday");
  assert.equal(draft.event_family, "Mixed Doubles");
  assert.equal(Object.hasOwn(draft, "registration_day_id"), false);''',
    '''  assert.equal(draft.assigned_day, "Friday");
  assert.deepEqual(draft.scheduled_day_ids, ["day-1"]);
  assert.equal(draft.event_family, "Mixed Doubles");
  assert.equal(Object.hasOwn(draft, "registration_day_id"), false);''',
)

# The previously successful publisher refreshes these contracts after this script.
setup_test = "tests/test_api_contract_tournament_setup_event_division_refine.py"
replace_once(
    setup_test,
    "    assert 'onClick={() => void saveBasics()}' in panel\n",
    "    assert 'onClick={() => void saveRegistrationRules()}' in panel\n",
)
replace_once(
    setup_test,
    "    assert 'onClick={() => void saveDraftAndContinue(\"events\")}' in panel\n",
    "    assert 'onClick={() => void saveDraftAndContinue(\"review\")}' in panel\n",
)

production_test = "tests/test_production_deployment_hardening.py"
replace_once(production_test, "    assert len(versions) == 56\n", "    assert len(versions) == 55\n")
replace_once(production_test, "    assert len(names) == 56\n", "    assert len(names) == 55\n")
replace_once(
    production_test,
    '    assert len(contract["required_ledger_names"]) == 56\n',
    '    assert len(contract["required_ledger_names"]) == 55\n',
)
replace_once(
    production_test,
    '''    assert versions[-5:] == (
        "20260731033000",
        "20260731210000",
        "20261020000000",
        "20261021000000",
        "20261022000000",
    )
''',
    '''    assert versions[-5:] == (
        "20260728041000",
        "20260731033000",
        "20260731210000",
        "20261020000000",
        "20261021000000",
    )
''',
)

Path(".github/workflows/tmp-tournament-setup-builder-compat.yml").unlink(missing_ok=True)
