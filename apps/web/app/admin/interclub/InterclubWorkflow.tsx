"use client";

import Link from "next/link";
import styles from "./workflow.module.css";

export type LeagueStep = "pool" | "availability" | "lineups" | "run" | "approve";
export type RegistrationStep = Extract<LeagueStep, "pool" | "availability" | "lineups">;

export function workflowHref(step: LeagueStep, seasonId: string, meetId = ""): string {
  const page = step === "run" || step === "approve" ? "competition" : "registrations";
  const query = new URLSearchParams({ season: seasonId, step });
  if (meetId) query.set("meet", meetId);
  return `/admin/interclub/${page}?${query}`;
}

const steps: { id: LeagueStep; label: string; hint: string }[] = [
  { id: "pool", label: "Player pool", hint: "Join once for the season" },
  { id: "availability", label: "Meet availability", hint: "Optional: collect replies" },
  { id: "lineups", label: "Lineups", hint: "Choose players for this meet" },
  { id: "run", label: "Run meet", hint: "Approved lineups required" },
  { id: "approve", label: "Approve results", hint: "Submitted scores required" },
];

export default function InterclubWorkflow({ seasonId, meetId, current, disabled = false, unavailable = [], onSelect, hints, meetPlanningOpen = false }: {
  seasonId: string; meetId?: string; current: LeagueStep; disabled?: boolean;
  meetPlanningOpen?: boolean;
  unavailable?: LeagueStep[]; onSelect?: (step: LeagueStep) => void;
  hints?: Partial<Record<LeagueStep, string>>;
}) {
  return <nav className={styles.workflow} aria-label="League workflow">
    <ol>{steps.map((step, index) => {
      const blocked = disabled || unavailable.includes(step.id) || (step.id !== "pool" && (!meetId || !meetPlanningOpen));
      const contents = <><span className={styles.number} aria-hidden="true">{index + 1}</span><span><strong>{step.label}</strong><small>{step.id !== "pool" && !meetPlanningOpen ? "After season registration" : hints?.[step.id] || step.hint}</small></span></>;
      return <li key={step.id}>
        {blocked ? <span className={styles.step} aria-disabled="true" aria-current={current === step.id ? "step" : undefined}>{contents}</span>
          : <Link className={styles.step} href={workflowHref(step.id, seasonId, meetId)} aria-label={step.label} aria-current={current === step.id ? "step" : undefined}
            onClick={onSelect && !["run", "approve"].includes(step.id) ? event => {
              if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
              event.preventDefault(); onSelect(step.id);
            } : undefined}>{contents}</Link>}
      </li>;
    })}</ol>
    {disabled && <p role="status">Save or discard your score changes before leaving this meet.</p>}
  </nav>;
}
