"use client";

import { useState } from "react";
import type { AdminWeeklyRecapCandidate } from "@/lib/adminWeeklyRecapApi";
import type {
  BuilderConfig,
  RecapCatalog,
  RecapEdits,
  RecapItem,
  SectionKey,
} from "@/lib/recapBuilder";
import {
  LEADERBOARD_CARD_DETAILS,
  LEADERBOARD_CARD_LABELS,
  type LeaderboardCard,
} from "@/lib/clubSite";
import styles from "./recap-builder.module.css";

const playerCategories: Record<
  string,
  { label: string; group: string; description: string }
> = {
  BIGGEST_JUMP_WEEK: {
    label: "Rating increases",
    group: "Ratings & improvement",
    description: "Positive rating changes during these dates. Minimum 3 games.",
  },
  TOP_PERFORMER_WEEK: {
    label: "Top performances",
    group: "Results",
    description: "Players’ actual win–loss records. Minimum 4 games.",
  },
  ABOVE_EXPECTATIONS_WEEK: {
    label: "Wins above expectation",
    group: "Competition",
    description:
      "Wins above the expectation from pre-game ratings. Minimum 4 rated games.",
  },
  GIANT_SLAYER_WEEK: {
    label: "Biggest upsets",
    group: "Competition",
    description:
      "Winning players or partners and the rating gap they overcame.",
  },
  PERFECT_RUN: {
    label: "Undefeated performances",
    group: "Streaks",
    description: "Players who finished this period without a loss.",
  },
  GRIND_WEEK: {
    label: "Most games played",
    group: "Participation & partnerships",
    description: "Recorded games played during these dates.",
  },
  COMMUNITY_STANDOUT_WEEK: {
    label: "Social play standouts",
    group: "Results",
    description: "Win–loss records from saved social events.",
  },
  SOCIAL_GRIND_WEEK: {
    label: "Social games played",
    group: "Participation & partnerships",
    description: "Participation in saved social events during these dates.",
  },
};
const eventGroups = [
  {
    id: "rr",
    label: "Round-robin winners",
    description:
      "Recorded winners, Moneyball results, and any unresolved ties.",
    match: (item: RecapItem) => /round robin|moneyball/i.test(item.kind || ""),
  },
  {
    id: "courts",
    label: "Ladder court winners",
    description:
      "Individual courts and rounds, with names, records, and point differences.",
    match: (item: RecapItem) => item.kind === "Ladder court",
  },
  {
    id: "leagues",
    label: "League & ladder highlights",
    description: "Recorded league and ladder summaries for this period.",
    match: (item: RecapItem) =>
      /league|ladder/i.test(item.kind || "") && item.kind !== "Ladder court",
  },
  {
    id: "tournaments",
    label: "Tournament results",
    description: "Completed divisions and recorded podiums.",
    match: (item: RecapItem) => /tournament/i.test(item.kind || ""),
  },
  {
    id: "interclub",
    label: "Interclub results",
    description: "Published interclub results involving your club.",
    match: (item: RecapItem) => /interclub/i.test(item.kind || ""),
  },
];
const otherGroups: Array<{
  id: string;
  section: SectionKey;
  label: string;
  description: string;
}> = [
  {
    id: "races",
    section: "races",
    label: "Season awards races",
    description: "Current award standings and progress toward eligibility.",
  },
  {
    id: "achievements",
    section: "achievements",
    label: "New achievements",
    description: "Badges earned from games in this date range.",
  },
  {
    id: "welcome",
    section: "welcome",
    label: "New players",
    description: "First-time players identified in this period.",
  },
  {
    id: "numbers",
    section: "numbers",
    label: "The week in numbers",
    description: "Choose the totals you want to show members.",
  },
];
const emailLimits: Partial<Record<SectionKey, number>> = {
  events: 3,
  races: 2,
  achievements: 3,
  spotlight: 2,
};
type Spotlight = RecapEdits["spotlight_overrides"][string];

export default function RecapEvidencePicker({
  builder,
  catalog,
  candidates,
  spotlights,
  disabled,
  onBuilderChange,
  onSpotlightChange,
}: {
  builder: BuilderConfig;
  catalog: RecapCatalog;
  candidates: Record<string, AdminWeeklyRecapCandidate[]>;
  spotlights: RecapEdits["spotlight_overrides"];
  disabled: boolean;
  onBuilderChange: (value: BuilderConfig) => void;
  onSpotlightChange: (key: string, patch: Partial<Spotlight>) => void;
}) {
  const [search, setSearch] = useState("");
  const needle = search.trim().toLocaleLowerCase();
  const events = catalog.events || [];
  const groups = [
    ...eventGroups.map((group) => ({
      ...group,
      section: "events" as SectionKey,
      items: events.filter(group.match),
    })),
    {
      id: "other-events",
      section: "events" as SectionKey,
      label: "Other event results",
      description: "Other verified results from the selected dates.",
      items: events.filter(
        (item) => !eventGroups.some((group) => group.match(item)),
      ),
    },
    ...otherGroups.map((group) => ({
      ...group,
      items: catalog[group.section] || [],
    })),
  ];
  const playerGroups = Object.entries(candidates).map(([key, items]) => {
    const metric = key.startsWith("METRIC_")
      ? (key.slice(7) as LeaderboardCard)
      : null;
    const definition = metric && LEADERBOARD_CARD_DETAILS[metric];
    return {
      key,
      items,
      ...(definition
        ? {
            label:
              metric === "highest_rating"
                ? "Highest recorded rating"
                : LEADERBOARD_CARD_LABELS[metric!],
            group: definition.group,
            description:
              metric === "highest_rating"
                ? "Rating after the player’s last recorded game in this period."
                : definition.description,
          }
        : playerCategories[key] || {
            label: items[0]?.label || key,
            group: "Player highlights",
            description: "Choose the recorded performances to feature.",
          }),
    };
  });
  function selected(section: SectionKey) {
    return (
      builder.sections[section].item_ids ||
      (catalog[section] || []).map((item) => item.id)
    );
  }
  function playerCategorySelected(key: string) {
    return Boolean(
      spotlights[key]?.include &&
      builder.sections.spotlight.enabled &&
      (!builder.sections.spotlight.item_ids ||
        builder.sections.spotlight.item_ids.includes(`spotlight-${key}`)),
    );
  }
  function emailSelected(section: SectionKey) {
    return (
      builder.sections[section].email_item_ids ||
      selected(section).slice(0, emailLimits[section] || 150)
    );
  }
  function choose(
    section: SectionKey,
    ids: string[],
    include: boolean,
    email = false,
  ) {
    const current = email ? emailSelected(section) : selected(section);
    const next = include
      ? [...new Set([...current, ...ids])]
      : current.filter((id) => !ids.includes(id));
    onBuilderChange({
      ...builder,
      sections: {
        ...builder.sections,
        [section]: {
          ...builder.sections[section],
          ...(include
            ? { enabled: true, ...(email ? { email_enabled: true } : {}) }
            : {}),
          [email ? "email_item_ids" : "item_ids"]: next,
        },
      },
    });
  }
  function togglePlayers(key: string, include: boolean) {
    const config = spotlights[key];
    onSpotlightChange(key, {
      include,
      players: config?.players.length
        ? config.players
        : (candidates[key] || []).slice(0, 3).map((item) => item.candidate_id),
    });
    if (include) choose("spotlight", [`spotlight-${key}`], true);
  }
  return (
    <div className={styles.sections} data-testid="recap-evidence-picker">
      <section className={styles.section}>
        <h3>Choose what members see</h3>
        <p className={styles.muted}>
          Turn on the types of highlights you want, then choose the actual
          results and players below. Your choices are saved with this draft.
        </p>
        <a href="#recap-results">Choose individual results and players ↓</a>
        <fieldset className={styles.choiceGroup}>
          <legend>Event results & club activity</legend>
          <div className={styles.choiceGrid}>
            {groups
              .filter(
                (group) => group.id !== "other-events" || group.items.length,
              )
              .map((group) => {
                const count = group.items.filter((item) =>
                  selected(group.section).includes(item.id),
                ).length;
                const included =
                  builder.sections[group.section].enabled && count > 0;
                return (
                  <label
                    className={styles.choiceCard}
                    data-selected={included}
                    key={group.id}
                  >
                    <span className={styles.check}>
                      <input
                        type="checkbox"
                        checked={included}
                        disabled={disabled || !group.items.length}
                        aria-label={`Include ${group.label}`}
                        onChange={(e) =>
                          choose(
                            group.section,
                            group.items.map((item) => item.id),
                            e.target.checked,
                          )
                        }
                      />
                      <strong>{group.label}</strong>
                    </span>
                    <span className={styles.muted}>{group.description}</span>
                    <small>
                      {group.items.length
                        ? `${included ? count : 0} of ${group.items.length} selected`
                        : "No verified results in this draft"}
                    </small>
                  </label>
                );
              })}
          </div>
        </fieldset>
        {[...new Set(playerGroups.map((group) => group.group))].map((label) => (
          <fieldset className={styles.choiceGroup} key={label}>
            <legend>{label}</legend>
            <div className={styles.choiceGrid}>
              {playerGroups
                .filter((group) => group.group === label)
                .map((group) => {
                  const included = Boolean(
                    group.items.length && playerCategorySelected(group.key),
                  );
                  return (
                    <label
                      className={styles.choiceCard}
                      data-selected={included}
                      key={group.key}
                    >
                      <span className={styles.check}>
                        <input
                          type="checkbox"
                          checked={included}
                          disabled={disabled || !group.items.length}
                          aria-label={`Include ${group.label}`}
                          onChange={(e) =>
                            togglePlayers(group.key, e.target.checked)
                          }
                        />
                        <strong>{group.label}</strong>
                      </span>
                      <span className={styles.muted}>{group.description}</span>
                      <small>
                        {group.items.length
                          ? `${group.items.length} available`
                          : "No qualifying results in this draft"}
                      </small>
                    </label>
                  );
                })}
            </div>
          </fieldset>
        ))}
        <p className={styles.muted}>
          The additional statistics use recorded rated doubles games, with at
          least 3 games per player. Close-game records require 3 close games;
          partnerships require 3 games together. Up to 10 candidates are offered
          per statistic.
        </p>
      </section>
      <label id="recap-results">
        Find a result or player
        <input
          type="search"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search names, events, courts, or results"
        />
      </label>
      {groups
        .filter((group) => group.items.length)
        .map((group) => {
          const visible = group.items.filter((item) =>
            [item.title, item.text, item.kind, item.date, ...item.lines]
              .join(" ")
              .toLocaleLowerCase()
              .includes(needle),
          );
          if (!visible.length) return null;
          const included = builder.sections[group.section].enabled;
          return (
            <section
              className={styles.section}
              key={group.id}
              data-testid={`recap-evidence-${group.id}`}
            >
              <h3>{group.label}</h3>
              <p className={styles.muted}>
                Check each result you want in the recap. Use Email to choose
                which also appear in the member email.
              </p>
              {visible.map((item) => (
                <div
                  className={styles.item}
                  key={item.id}
                  data-testid={`recap-evidence-item-${item.id}`}
                >
                  <div className={styles.evidenceText}>
                    <label className={styles.check}>
                      <input
                        type="checkbox"
                        checked={
                          included && selected(group.section).includes(item.id)
                        }
                        disabled={disabled}
                        onChange={(e) =>
                          choose(group.section, [item.id], e.target.checked)
                        }
                      />
                      <strong>{item.title || item.text}</strong>
                    </label>
                    {item.kind || item.date ? (
                      <p className={styles.muted}>
                        {[item.kind, item.date].filter(Boolean).join(" · ")}
                      </p>
                    ) : null}
                    {item.title && item.text ? (
                      <p className={styles.resultSummary}>{item.text}</p>
                    ) : null}
                    {item.lines.length ? (
                      <ul className={styles.evidenceLines}>
                        {item.lines.map((line, index) => (
                          <li key={index}>{line}</li>
                        ))}
                      </ul>
                    ) : null}
                  </div>
                  <label className={styles.check}>
                    <input
                      type="checkbox"
                      aria-label={`Feature ${item.title || item.text} in email`}
                      checked={
                        included &&
                        builder.sections[group.section].email_enabled &&
                        selected(group.section).includes(item.id) &&
                        emailSelected(group.section).includes(item.id)
                      }
                      disabled={
                        disabled ||
                        !included ||
                        !selected(group.section).includes(item.id)
                      }
                      onChange={(e) =>
                        choose(group.section, [item.id], e.target.checked, true)
                      }
                    />
                    Email
                  </label>
                </div>
              ))}
            </section>
          );
        })}
      {playerGroups
        .filter((group) => playerCategorySelected(group.key))
        .map((group) => {
          const config = spotlights[group.key];
          const visible = group.items.filter((item) =>
            `${group.label} ${item.display}`
              .toLocaleLowerCase()
              .includes(needle),
          );
          if (needle && !visible.length) return null;
          const id = `spotlight-${group.key}`;
          return (
            <section
              className={styles.section}
              key={group.key}
              data-testid={`recap-evidence-${group.key}`}
            >
              <h3>{group.label}</h3>
              <p className={styles.muted}>
                Choose up to 10 highlights.{" "}
                {config.players.filter(Boolean).length} selected. Each checked
                result is included in the recap.
              </p>
              {visible.map((item) => (
                <label
                  className={styles.evidenceChoice}
                  key={item.candidate_id}
                >
                  <input
                    type="checkbox"
                    checked={config.players.includes(item.candidate_id)}
                    disabled={
                      disabled ||
                      (!config.players.includes(item.candidate_id) &&
                        config.players.filter(Boolean).length >= 10)
                    }
                    onChange={(e) =>
                      onSpotlightChange(group.key, {
                        players: e.target.checked
                          ? [
                              ...config.players.filter(Boolean),
                              item.candidate_id,
                            ]
                          : config.players.filter(
                              (value) => value !== item.candidate_id,
                            ),
                      })
                    }
                  />
                  <span>{item.display}</span>
                </label>
              ))}
              {!visible.length ? (
                <p>
                  No qualifying performances in this draft. Refresh results in
                  Setup to check for newly recorded games.
                </p>
              ) : null}
              <label className={styles.check}>
                <input
                  type="checkbox"
                  aria-label={`Include ${group.label} in email`}
                  disabled={disabled}
                  checked={
                    builder.sections.spotlight.email_enabled &&
                    emailSelected("spotlight").includes(id)
                  }
                  onChange={(e) =>
                    choose("spotlight", [id], e.target.checked, true)
                  }
                />
                Include these highlights in email
              </label>
              <details>
                <summary>Highlight description and order</summary>
                <div className={styles.fields}>
                  <label>
                    Description
                    <input
                      value={config.description}
                      maxLength={1000}
                      disabled={disabled}
                      onChange={(e) =>
                        onSpotlightChange(group.key, {
                          description: e.target.value,
                        })
                      }
                    />
                  </label>
                  <label>
                    Order
                    <input
                      type="number"
                      min={1}
                      max={99}
                      value={config.order}
                      disabled={disabled}
                      onChange={(e) =>
                        onSpotlightChange(group.key, {
                          order: Number(e.target.value) || 1,
                        })
                      }
                    />
                  </label>
                </div>
              </details>
            </section>
          );
        })}
      {needle &&
      !groups.some((group) =>
        group.items.some((item) =>
          [item.title, item.text, item.kind, item.date, ...item.lines]
            .join(" ")
            .toLocaleLowerCase()
            .includes(needle),
        ),
      ) &&
      !playerGroups.some(
        (group) =>
          playerCategorySelected(group.key) &&
          group.items.some((item) =>
            `${group.label} ${item.display}`
              .toLocaleLowerCase()
              .includes(needle),
          ),
      ) ? (
        <p role="status">
          No results match “{search}”. Try another player or event name.
        </p>
      ) : null}
      <p className={styles.muted}>
        Game and player totals still cover the full date range and scope
        selected in Setup. Excluding a story changes what members see.
      </p>
    </div>
  );
}
