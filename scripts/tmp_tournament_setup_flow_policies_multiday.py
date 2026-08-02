from __future__ import annotations

from pathlib import Path


def replace_between(path: str, start: str, end: str, replacement: str) -> None:
    target = Path(path)
    text = target.read_text()
    start_index = text.find(start)
    if start_index < 0:
        raise SystemExit(f"{path}: start marker not found: {start!r}")
    end_index = text.find(end, start_index + len(start))
    if end_index < 0:
        raise SystemExit(f"{path}: end marker not found: {end!r}")
    target.write_text(text[:start_index] + replacement + text[end_index:])


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one match, found {count}: {old[:120]!r}")
    target.write_text(text.replace(old, new, 1))


overview = "apps/web/app/admin/tournaments/TournamentLifecycleOverviewPanel.tsx"
replace_between(
    overview,
    "function setupSteps(",
    "\n\nfunction registrationSteps(",
    '''function setupSteps(
  detail: AdminTournamentDetailResponse,
  tournamentId: string,
  tournamentName: string
): StepCard[] {
  const datesReady = Boolean(
    detail.tournament.start_date && detail.tournament.end_date
  );
  const policiesReady = Boolean(
    detail.settings?.registration_open_at &&
      detail.settings?.registration_close_at &&
      detail.settings?.rules_markdown &&
      detail.settings?.refund_policy_markdown &&
      detail.settings?.weather_policy_markdown
  );
  const basicsReady = Boolean(
    datesReady &&
      detail.settings?.location_name &&
      detail.settings?.timezone &&
      policiesReady
  );
  const daysReady = detail.days.length > 0;
  const eventFamilies = new Set(
    detail.event_options
      .map((event) => String(event.event_family_label || "").trim())
      .filter(Boolean)
  );
  const eventsReady = eventFamilies.size > 0;
  const divisionsReady = detail.event_options.length > 0;
  const reviewReady = basicsReady && daysReady && eventsReady && divisionsReady;
  return [
    {
      title: "1. Tournament basics and policies",
      description:
        "Name, dates, venue, timezone, sponsors, registration window, rules, cancellation policy, and weather policy.",
      href: selectedHref(
        "/admin/tournaments/setup/basics",
        tournamentId,
        tournamentName
      ),
      state: basicsReady ? "Complete" : "In progress"
    },
    {
      title: "2. Schedule and courts",
      description:
        "Create the tournament days first so events and divisions can use one or several days.",
      href: selectedHref(
        "/admin/tournaments/setup/schedule",
        tournamentId,
        tournamentName
      ),
      state: daysReady ? "Complete" : "Not started"
    },
    {
      title: "3. Events",
      description:
        "Create event families, choose every available day, and set draw, scoring, capacity, and pricing defaults.",
      href: selectedHref(
        "/admin/tournaments/setup/events",
        tournamentId,
        tournamentName
      ),
      state: eventsReady ? "Complete" : "Not started",
      note: eventsReady
        ? `${eventFamilies.size} event${eventFamilies.size === 1 ? "" : "s"}`
        : undefined
    },
    {
      title: "4. Divisions",
      description:
        "Create skill and age divisions within each event and inherit all event days or choose a subset.",
      href: selectedHref(
        "/admin/tournaments/setup/divisions",
        tournamentId,
        tournamentName
      ),
      state: divisionsReady ? "Complete" : "Not started",
      note: divisionsReady
        ? `${detail.event_options.length} division${detail.event_options.length === 1 ? "" : "s"}`
        : undefined
    },
    {
      title: "5. Pricing, extras, and fulfillment",
      description:
        "Entry fees, additional events, extras, bundles, inventory, pickup, and offline payment.",
      href: selectedHref(
        "/admin/tournaments/setup/pricing",
        tournamentId,
        tournamentName
      ),
      state: "In progress"
    },
    {
      title: "6. Review and open registration",
      description:
        "Resolve missing fields, conflicts, capacity, pricing, policies, and schedule warnings before opening.",
      href: selectedHref(
        "/admin/tournaments/setup/review",
        tournamentId,
        tournamentName
      ),
      state: reviewReady ? "Ready" : "Blocked",
      note: reviewReady
        ? "Basics, policies, days, events, and divisions are present."
        : "Complete basics and policies, tournament days, events, and divisions first."
    }
  ];
}
''',
)

test_path = "tests/test_api_contract_tournament_lifecycle_rebuild.py"
replace_once(
    test_path,
    '''  for label in (
      "Tournament basics",
      "Events and formats",
      "Registration rules",
      "Pricing, extras, and fulfillment",
      "Schedule and courts",
      "Review and open registration",
''',
    '''  for label in (
      "Tournament basics and policies",
      "Schedule and courts",
      "Events",
      "Divisions",
      "Pricing, extras, and fulfillment",
      "Review and open registration",
''',
)
replace_once(
    test_path,
    '''  for label in (
      "1. Basics",
      "2. Events & formats",
      "3. Registration rules",
      "4. Pricing & extras",
      "5. Schedule & courts",
      "6. Review & open",
''',
    '''  for label in (
      "1. Basics & policies",
      "2. Schedule & courts",
      "3. Events",
      "4. Divisions",
      "5. Pricing & extras",
      "6. Review & open",
''',
)
replace_once(
    test_path,
    '''  for key in (
      'key: "basics"',
      'key: "events"',
      'key: "registration-rules"',
      'key: "pricing"',
      'key: "schedule"',
      'key: "review"',
''',
    '''  for key in (
      'key: "basics"',
      'key: "schedule"',
      'key: "events"',
      'key: "divisions"',
      'key: "pricing"',
      'key: "review"',
''',
)
replace_once(test_path, '  assert \'goTo("events")\' in setup_panel\n', '  assert \'goTo("schedule")\' in setup_panel\n')

# The successful publisher refreshes these values after this script.
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
