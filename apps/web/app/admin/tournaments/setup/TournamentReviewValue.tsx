"use client";

import type { CSSProperties, ReactNode } from "react";
import Image from "next/image";
import { sponsorTierLabels, type SponsorTier } from "@/lib/tournamentSponsors";
import {
  cleanString,
  dayLabel,
  dayReference,
  type BuilderRow,
  type SetupRecord
} from "../../tournament-setup/tournamentSetupBuilder";
import { skillEligibilityLabel, skillEligibilityMode, skillEligibilitySummary } from "@/lib/tournamentSkillEligibility";

const valueBox: CSSProperties = {
  padding: "0.65rem",
  borderRadius: "10px",
  background: "#f8fafc",
  minWidth: 0,
  overflowWrap: "anywhere"
};

function parseStructured(value: unknown): unknown {
  if (typeof value !== "string") return value;
  const text = value.trim();
  if (!text || !["{", "["].includes(text[0])) return value;
  try {
    return JSON.parse(text);
  } catch {
    return value;
  }
}

function titleCase(value: unknown): string {
  const text = cleanString(value);
  if (!text) return "Not set";
  return text
    .replaceAll("_", " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .toLowerCase()
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function humanReviewFieldLabel(field: unknown): string {
  const key = cleanString(field).toLowerCase();
  const labels: Record<string, string> = {
    scheduled_day_ids: "Tournament days",
    registration_day_id: "Primary tournament day",
    skill_age_rules: "Skill and age rules",
    age_rules: "Age rules",
    venue_courts_json: "Venue courts",
    available_court_ids: "Available courts",
    venue_address: "Venue address",
    venue_directions: "Directions to venue",
    registration_open_at: "Registration opens",
    registration_close_at: "Registration closes",
    event_families: "Events",
    event_options: "Divisions"
  };
  return labels[key] || titleCase(field);
}

function formatter(timezone: string, includeTime: boolean): Intl.DateTimeFormat {
  return new Intl.DateTimeFormat("en", {
    timeZone: timezone || "UTC",
    year: "numeric",
    month: "long",
    day: "numeric",
    ...(includeTime ? { hour: "numeric", minute: "2-digit", timeZoneName: "short" } : {})
  });
}

function readableDate(value: unknown, timezone: string): string | null {
  const text = cleanString(value);
  if (!text) return null;
  if (/^\d{4}-\d{2}-\d{2}$/.test(text)) {
    const date = new Date(`${text}T12:00:00Z`);
    return Number.isNaN(date.valueOf()) ? null : formatter(timezone, false).format(date);
  }
  if (/^\d{4}-\d{2}-\d{2}T/.test(text)) {
    const date = new Date(text);
    return Number.isNaN(date.valueOf()) ? null : formatter(timezone, true).format(date);
  }
  return null;
}

function dayLookup(days: BuilderRow[]): Map<string, string> {
  const map = new Map<string, string>();
  days.forEach((row) => {
    const id = dayReference(row.value);
    const label = dayLabel(row.value) || id;
    const date = readableDate(row.value.event_date, "UTC");
    const display = date ? `${label} — ${date}` : label;
    if (id) map.set(id, display);
    if (label) map.set(label, display);
  });
  return map;
}

function skillAgeItems(value: SetupRecord): Array<[string, string]> {
  const parsedRules = parseStructured(value.age_rules) as SetupRecord;
  const ageRules = parsedRules && typeof parsedRules === "object" && !Array.isArray(parsedRules)
    ? parsedRules
    : {};
  const items: Array<[string, string]> = [];
  const skill = skillEligibilityLabel(value);
  if (skill) items.push(["Skill", skill]);
  items.push(["Skill eligibility", skillEligibilitySummary(value)]);
  if (!["OPEN", "COMBINED_RATING_CAP"].includes(skillEligibilityMode(value))) {
    items.push(["Doubles controlling rating", "Higher-rated partner"]);
  }
  const source = cleanString(value.policy_source);
  if (source) items.push(["Policy source", source]);
  const mode = cleanString(value.age_mode ?? ageRules.mode).toUpperCase();
  const brackets = Array.isArray(ageRules.brackets)
    ? ageRules.brackets.filter((row): row is SetupRecord => Boolean(row) && typeof row === "object" && !Array.isArray(row))
    : [];
  const ageLabel = cleanString(value.age_label ?? ageRules.age_label);
  if (mode === "AUTO_AGE_SPLIT" && brackets.length) {
    items.push(["Candidate age brackets", brackets.map((row) => cleanString(row.label)).filter(Boolean).join(", ")]);
    items.push(["Age eligibility", "Older players may play in younger groups; younger players may not play in older groups"]);
    items.push(["Preferred placement", "Oldest age group the player or team qualifies for"]);
    const minimum = value.min_teams_per_age_group ?? ageRules.min_teams_per_age_group;
    if (minimum != null && minimum !== "") items.push(["Minimum entries per bracket", String(minimum)]);
    const merge = cleanString(value.merge_strategy ?? ageRules.merge_strategy);
    if (merge) items.push(["Underfilled bracket fallback", titleCase(merge)]);
  } else if (ageLabel) {
    items.push(["Age", ageLabel]);
    if (mode === "FIXED_AGE_BRACKET") {
      items.push(["Age eligibility", "Minimum-age rule; older players may play down into this group"]);
    }
  }
  if (mode) items.push(["Age mode", titleCase(mode)]);
  const threshold = value.split_age_threshold ?? ageRules.split_age_threshold;
  if (threshold != null && threshold !== "") {
    items.push(["Split-age rule", `One player under ${threshold} and one player ${threshold}+`]);
  }
  const teamRule = cleanString(value.team_age_rule ?? ageRules.team_age_rule);
  if (teamRule) {
    const rules: Record<string, string> = {
      YOUNGER: "Younger player determines team age",
      OLDER: "Older player determines team age",
      AVERAGE: "Average player age",
      BOTH_QUALIFY: "Both players must qualify"
    };
    items.push(["Team age", rules[teamRule.toUpperCase()] || titleCase(teamRule)]);
  }
  const playerAge = value.player_age;
  const partnerAge = value.partner_age;
  const effectiveAge = value.effective_age;
  if (playerAge != null && playerAge !== "") items.push(["Player age", String(playerAge)]);
  if (partnerAge != null && partnerAge !== "") items.push(["Partner age", String(partnerAge)]);
  if (effectiveAge != null && effectiveAge !== "") items.push(["Effective team age", String(effectiveAge)]);
  const assignmentIssue = cleanString(value.assignment_issue);
  if (assignmentIssue) items.push(["Assignment issue", assignmentIssue]);
  return items;
}

function objectSummary(value: SetupRecord, field: string, days: BuilderRow[], timezone: string): ReactNode {
  if (/skill_age_rules|age_rules/i.test(field)) {
    const items = skillAgeItems(value);
    return items.length ? (
      <dl style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "0.5rem", margin: 0 }}>
        {items.map(([label, text]) => <div key={label}><dt style={{ fontWeight: 800 }}>{label}</dt><dd style={{ margin: 0 }}>{text}</dd></div>)}
      </dl>
    ) : "Not set";
  }

  const entries = Object.entries(value).filter(([, item]) => item != null && item !== "");
  if (!entries.length) return "Not set";
  return (
    <dl style={{ display: "grid", gap: "0.4rem", margin: 0 }}>
      {entries.map(([key, item]) => (
        <div key={key} style={{ minWidth: 0 }}>
          <dt style={{ fontWeight: 800 }}>{humanReviewFieldLabel(key)}</dt>
          <dd style={{ margin: 0 }}><ReviewValueDisplay field={key} value={item} days={days} timezone={timezone} technical={false} /></dd>
        </div>
      ))}
    </dl>
  );
}

function listSummary(values: unknown[], field: string, days: BuilderRow[], timezone: string): ReactNode {
  const map = dayLookup(days);
  if (/day|scheduled/i.test(field) && values.every((value) => typeof value === "string")) {
    return (
      <ul style={{ margin: 0, paddingLeft: "1.2rem" }}>
        {values.map((value) => <li key={String(value)}>{map.get(String(value)) || titleCase(value)}</li>)}
      </ul>
    );
  }

  const objects = values.filter((value): value is SetupRecord => Boolean(value) && typeof value === "object" && !Array.isArray(value));
  if (objects.length === values.length && objects.length) {
    if (/sponsor/i.test(field)) {
      return <ul style={{ margin: 0, paddingLeft: "1.2rem" }}>{objects.map((row, index) => <li key={cleanString(row.id) || `${cleanString(row.name)}-${index}`}>
        <strong>{cleanString(row.name) || `Sponsor ${index + 1}`}</strong>
        <div>{sponsorTierLabels[row.tier as SponsorTier] || sponsorTierLabels.supporting}{cleanString(row.level) ? ` · ${cleanString(row.level)}` : ""} · {row.is_visible === false ? "Hidden" : "Visible"} · Position {Number(row.sort_order ?? index) + 1}</div>
        <div>{cleanString(row.website) || "No website"} · {cleanString(row.logo_path) ? "Logo uploaded" : "Name only"}</div>
        {cleanString(row.logo_url) ? <Image unoptimized src={cleanString(row.logo_url)} alt="" width={120} height={48} style={{ objectFit: "contain", background: "white", maxWidth: "100%" }} /> : null}
      </li>)}</ul>;
    }
    if (/court/i.test(field)) {
      return <ul style={{ margin: 0, paddingLeft: "1.2rem" }}>{objects.map((row, index) => <li key={cleanString(row.id) || index}>{cleanString(row.title) || `Court ${index + 1}`}</li>)}</ul>;
    }
    if (/event/i.test(field) && !/division/i.test(field)) {
      return (
        <div style={{ display: "grid", gap: "0.45rem" }}>
          {objects.map((row, index) => (
            <article key={cleanString(row.id) || `${cleanString(row.event)}-${index}`} style={{ border: "1px solid #e2e8f0", borderRadius: "8px", padding: "0.5rem", background: "white" }}>
              <strong>{cleanString(row.event ?? row.event_family ?? row.event_family_label) || `Event ${index + 1}`}</strong>
              <div>{titleCase(row.format ?? row.participant_type ?? row.event_type)}</div>
              {row.age ? <small>Age: {cleanString(row.age)}</small> : null}
              {Array.isArray(row.days) ? <div><ReviewValueDisplay field="scheduled_day_ids" value={row.days} days={days} timezone={timezone} technical={false} /></div> : null}
            </article>
          ))}
        </div>
      );
    }
    if (/division/i.test(field)) {
      return (
        <div style={{ display: "grid", gap: "0.45rem" }}>
          {objects.map((row, index) => (
            <article key={cleanString(row.id) || `${cleanString(row.event)}-${cleanString(row.division)}-${index}`} style={{ border: "1px solid #e2e8f0", borderRadius: "8px", padding: "0.5rem", background: "white" }}>
              <strong>{cleanString(row.division ?? row.division_name ?? row.label) || `Division ${index + 1}`}</strong>
              <div>{cleanString(row.event ?? row.event_family ?? row.event_family_label) || "No parent event"}</div>
              <small>{cleanString(row.skill ?? row.skill_label) || "Open"}{cleanString(row.age ?? row.age_label) ? ` · ${cleanString(row.age ?? row.age_label)}` : ""}{row.fee != null || row.price_usd != null ? ` · $${Number(row.fee ?? row.price_usd ?? 0).toFixed(2)}` : ""}</small>
            </article>
          ))}
        </div>
      );
    }
    return <div style={{ display: "grid", gap: "0.45rem" }}>{objects.map((row, index) => <article key={cleanString(row.id) || index} style={{ border: "1px solid #e2e8f0", borderRadius: "8px", padding: "0.5rem", background: "white" }}>{objectSummary(row, field, days, timezone)}</article>)}</div>;
  }

  return <ul style={{ margin: 0, paddingLeft: "1.2rem" }}>{values.map((value, index) => <li key={`${String(value)}-${index}`}><ReviewValueDisplay field={field} value={value} days={days} timezone={timezone} technical={false} /></li>)}</ul>;
}

export function ReviewValueDisplay({
  field,
  value,
  days,
  timezone,
  technical = true
}: {
  field: string;
  value: unknown;
  days: BuilderRow[];
  timezone: string;
  technical?: boolean;
}) {
  const parsed = parseStructured(value);
  let content: ReactNode;
  if (parsed == null || parsed === "") {
    content = "Not set";
  } else if (typeof parsed === "boolean") {
    content = parsed ? "Yes" : "No";
  } else if (Array.isArray(parsed)) {
    content = parsed.length ? listSummary(parsed, field, days, timezone) : "None";
  } else if (typeof parsed === "object") {
    content = objectSummary(parsed as SetupRecord, field, days, timezone);
  } else {
    const dayMap = dayLookup(days);
    const text = cleanString(parsed);
    const date = readableDate(text, timezone);
    content = dayMap.get(text) || date || (/_/.test(text) ? titleCase(text) : text);
  }

  return (
    <div style={{ minWidth: 0, overflowWrap: "anywhere" }}>
      {content}
      {technical && (typeof parsed === "object" || (typeof value === "string" && value !== parsed)) ? (
        <details style={{ marginTop: "0.45rem" }}>
          <summary style={{ cursor: "pointer", color: "#64748b" }}>Technical details</summary>
          <pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere", fontSize: "0.75rem", marginBottom: 0 }}>{JSON.stringify(parsed, null, 2)}</pre>
        </details>
      ) : null}
    </div>
  );
}

function identity(field: string, row: SetupRecord, index: number): string {
  const stableId = cleanString(row.id).toLowerCase();
  if (stableId) return stableId;
  if (/division/i.test(field)) return `${cleanString(row.event ?? row.event_family ?? row.event_family_label).toLowerCase()}|${cleanString(row.division ?? row.division_name ?? row.label).toLowerCase()}`;
  if (/event/i.test(field)) return cleanString(row.event ?? row.event_family ?? row.event_family_label).toLowerCase() || String(index);
  if (/sponsor/i.test(field)) return cleanString(row.name).toLowerCase() || String(index);
  return JSON.stringify(row) || String(index);
}

export function ReviewComparisonDisplay({
  field,
  current,
  proposed,
  days,
  timezone
}: {
  field: string;
  current: unknown;
  proposed: unknown;
  days: BuilderRow[];
  timezone: string;
}) {
  const currentParsed = parseStructured(current);
  const proposedParsed = parseStructured(proposed);
  const structuredCollections = Array.isArray(currentParsed) && Array.isArray(proposedParsed)
    && [...currentParsed, ...proposedParsed].every((row) => Boolean(row) && typeof row === "object" && !Array.isArray(row));

  if (structuredCollections) {
    const currentRows = currentParsed as SetupRecord[];
    const proposedRows = proposedParsed as SetupRecord[];
    const currentMap = new Map(currentRows.map((row, index) => [identity(field, row, index), row]));
    const proposedMap = new Map(proposedRows.map((row, index) => [identity(field, row, index), row]));
    const keys = [...new Set([...currentMap.keys(), ...proposedMap.keys()])];
    return (
      <div style={{ display: "grid", gap: "0.55rem", marginTop: "0.5rem" }}>
        {keys.map((key) => {
          const before = currentMap.get(key);
          const after = proposedMap.get(key);
          const status = !before ? "Added" : !after ? "Removed" : JSON.stringify(before) !== JSON.stringify(after) ? "Changed" : "Unchanged";
          if (status === "Unchanged") return null;
          const badgeBackground = status === "Added" ? "#dcfce7" : status === "Removed" ? "#fee2e2" : "#fef3c7";
          return (
            <article key={key} style={{ border: "1px solid #e2e8f0", borderRadius: "10px", padding: "0.65rem", minWidth: 0 }}>
              <span style={{ display: "inline-block", padding: "0.15rem 0.45rem", borderRadius: "999px", background: badgeBackground, fontWeight: 800, fontSize: "0.78rem" }}>{status}</span>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))", gap: "0.65rem", marginTop: "0.45rem" }}>
                {before ? <div style={valueBox}><small>Current published</small><ReviewValueDisplay field={field} value={[before]} days={days} timezone={timezone} /></div> : null}
                {after ? <div style={{ ...valueBox, background: "#eff6ff" }}><small>Proposed draft</small><ReviewValueDisplay field={field} value={[after]} days={days} timezone={timezone} /></div> : null}
              </div>
            </article>
          );
        })}
      </div>
    );
  }

  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))", gap: "0.65rem", marginTop: "0.45rem" }}>
      <div style={valueBox}><small>Current published value</small><ReviewValueDisplay field={field} value={currentParsed} days={days} timezone={timezone} /></div>
      <div style={{ ...valueBox, background: "#eff6ff" }}><small>Proposed draft value</small><ReviewValueDisplay field={field} value={proposedParsed} days={days} timezone={timezone} /></div>
    </div>
  );
}
