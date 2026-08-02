from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"{path.relative_to(ROOT)}: expected one match, found {count}: {old[:120]!r}"
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def restore_registration_api() -> None:
    target = ROOT / "apps/web/lib/tournamentRegistrationApi.ts"
    subprocess.run(["git", "fetch", "origin", "staging"], cwd=ROOT, check=True)
    with target.open("wb") as handle:
        subprocess.run(
            [
                "git",
                "show",
                "origin/staging:apps/web/lib/tournamentRegistrationApi.ts",
            ],
            cwd=ROOT,
            stdout=handle,
            check=True,
        )
    replace_once(
        target,
        "  refund_policy_markdown?: string | null;\n  sponsor_markdown?: string | null;\n",
        "  refund_policy_markdown?: string | null;\n"
        "  weather_policy_markdown?: string | null;\n"
        "  sponsor_markdown?: string | null;\n",
    )
    replace_once(
        target,
        "  registration_day_id: string;\n  label: string;\n",
        "  registration_day_id: string;\n"
        "  scheduled_day_ids?: string[] | null;\n"
        "  label: string;\n",
    )


def fix_division_card() -> None:
    path = ROOT / "apps/web/app/admin/tournaments/setup/TournamentSetupDivisionCard.tsx"
    for import_line in (
        "  ageRuleText,\n",
        "  clearIncompatibleAgeRuleFields,\n",
        "  finiteNumber,\n",
        "  setEventDivisionName,\n",
        "  setEventSkillLabel,\n",
        "  setRecordBoolean,\n",
    ):
        replace_once(path, import_line, "")

    replace_once(
        path,
        '''function optionLabel(value: string): string {
  return value
    .toLowerCase()
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}
''',
        '''function optionLabel(value: string): string {
  return value
    .toLowerCase()
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function setRecordBoolean(
  value: SetupRecord,
  key: string,
  checked: boolean
): SetupRecord {
  return { ...value, [key]: checked };
}

function setEventDivisionName(value: SetupRecord, name: string): SetupRecord {
  const next: SetupRecord = { ...value };
  if (
    Object.prototype.hasOwnProperty.call(value, "division_name") ||
    !Object.prototype.hasOwnProperty.call(value, "label")
  ) {
    next.division_name = name;
  }
  if (Object.prototype.hasOwnProperty.call(value, "label")) {
    next.label = name;
  }
  return next;
}

function setEventSkillLabel(value: SetupRecord, skill: string): SetupRecord {
  const next: SetupRecord = { ...value, skill_label: skill };
  if (
    Object.prototype.hasOwnProperty.call(value, "skill_mode") ||
    Object.prototype.hasOwnProperty.call(value, "event_type")
  ) {
    next.skill_mode = skill.trim().toLowerCase() === "open"
      ? "OPEN"
      : "SKILL_BRACKET";
  }
  return next;
}

function ageRuleText(value: SetupRecord, key: string): string {
  const direct = key === "notes"
    ? value.division_notes ?? value.notes
    : value[key];
  if (direct != null && String(direct).trim()) return cleanString(direct);
  const raw = value.age_rules;
  if (raw && typeof raw === "object" && !Array.isArray(raw)) {
    return cleanString((raw as SetupRecord)[key]);
  }
  if (typeof raw === "string" && raw.trim()) {
    try {
      const parsed: unknown = JSON.parse(raw);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return cleanString((parsed as SetupRecord)[key]);
      }
    } catch {
      return "";
    }
  }
  return "";
}
''',
    )

    replace_once(
        path,
        '''              onChange(
                clearIncompatibleAgeRuleFields(
                  setRecordString(
                    value,
                    ["event_format_override", "division_format"],
                    event.target.value
                  ),
                  ageMode
                )
              )
''',
        '''              onChange(
                setRecordString(
                  value,
                  ["event_format_override", "division_format"],
                  event.target.value
                )
              )
''',
    )


def expose_weather_policy() -> None:
    path = ROOT / "apps/web/app/clubs/[clubSlug]/tournament-registration/page.tsx"
    replace_once(
        path,
        '''      {tournament && settings?.refund_policy_markdown ? (
        <article style={{ ...cardStyle, marginBottom: "1rem" }}>
          <h2 style={{ marginTop: 0 }}>Refund policy</h2>
          {markdownish(settings.refund_policy_markdown)}
        </article>
      ) : null}

      {tournament ? (
''',
        '''      {tournament && settings?.refund_policy_markdown ? (
        <article style={{ ...cardStyle, marginBottom: "1rem" }}>
          <h2 style={{ marginTop: 0 }}>Refund policy</h2>
          {markdownish(settings.refund_policy_markdown)}
        </article>
      ) : null}

      {tournament && settings?.weather_policy_markdown ? (
        <article style={{ ...cardStyle, marginBottom: "1rem" }}>
          <h2 style={{ marginTop: 0 }}>Weather policy</h2>
          {markdownish(settings.weather_policy_markdown)}
        </article>
      ) : null}

      {tournament ? (
''',
    )


def update_contracts() -> None:
    path = ROOT / "tests/test_api_contract_tournament_setup_flow_policies_multiday.py"
    text = path.read_text(encoding="utf-8")
    if "def test_public_registration_displays_required_weather_policy" in text:
        return
    addition = '''


def test_public_registration_displays_required_weather_policy() -> None:
    public_api = read("lib/tournamentRegistrationApi.ts")
    public_page = read("app/clubs/[clubSlug]/tournament-registration/page.tsx")
    assert "weather_policy_markdown?: string | null" in public_api
    assert "scheduled_day_ids?: string[] | null" in public_api
    assert "settings?.weather_policy_markdown" in public_page
    assert ">Weather policy<" in public_page
'''
    path.write_text((text.rstrip() + addition).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    restore_registration_api()
    fix_division_card()
    expose_weather_policy()
    update_contracts()


if __name__ == "__main__":
    main()
