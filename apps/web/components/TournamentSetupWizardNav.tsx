import Link from "next/link";
import { tournamentRouteHref } from "@/lib/tournamentRouteContext";

export type TournamentSetupStep =
  | "basics"
  | "schedule"
  | "events"
  | "divisions"
  | "pricing"
  | "review";

export type TournamentSetupDomain =
  | "tournament"
  | "competition"
  | "commerce"
  | "review";

export type TournamentSetupStepState =
  | "not-started"
  | "in-progress"
  | "complete"
  | "blocked";

type Props = {
  currentStep: TournamentSetupStep;
  tournamentId: string;
  tournamentName: string;
  drawId?: string;
  states?: Partial<Record<TournamentSetupStep, TournamentSetupStepState>>;
};

export const TOURNAMENT_SETUP_STEPS: Array<{
  key: TournamentSetupStep;
  number: number;
  domain: TournamentSetupDomain;
  label: string;
  shortLabel: string;
  description: string;
}> = [
  {
    key: "basics",
    number: 1,
    domain: "tournament",
    label: "Basics, registration, and policies",
    shortLabel: "Basics & policies",
    description:
      "Tournament identity, dates, registration window, sponsors, and public policies."
  },
  {
    key: "schedule",
    number: 2,
    domain: "tournament",
    label: "Venue and tournament days",
    shortLabel: "Venue & days",
    description:
      "Venue, timezone, global court capacity, optional court titles, and fixed tournament dates."
  },
  {
    key: "events",
    number: 3,
    domain: "competition",
    label: "Events and event policies",
    shortLabel: "Events & policies",
    description:
      "Event structure, formats, age policy, draw defaults, scoring, and tournament-day availability."
  },
  {
    key: "divisions",
    number: 4,
    domain: "competition",
    label: "Divisions",
    shortLabel: "Divisions",
    description:
      "Generated or organizer-defined skill and age groups that inherit event policy by default."
  },
  {
    key: "pricing",
    number: 5,
    domain: "commerce",
    label: "Fees, extras, bundles, and giveaways",
    shortLabel: "Commerce",
    description:
      "Consolidated event and division fees, merchandise, options, bundles, inventory, and fulfillment."
  },
  {
    key: "review",
    number: 6,
    domain: "review",
    label: "Preview, conflicts, publish, and registration",
    shortLabel: "Review & publish",
    description:
      "Preview the tournament that will exist, resolve conflicts, publish setup, and open registration."
  }
];

export const TOURNAMENT_SETUP_DOMAINS: Array<{
  key: TournamentSetupDomain;
  number: number;
  label: string;
  description: string;
  steps: TournamentSetupStep[];
}> = [
  {
    key: "tournament",
    number: 1,
    label: "Tournament",
    description: "Basics, venue, registration, and policies.",
    steps: ["basics", "schedule"]
  },
  {
    key: "competition",
    number: 2,
    label: "Competition",
    description: "Events, event policies, and divisions.",
    steps: ["events", "divisions"]
  },
  {
    key: "commerce",
    number: 3,
    label: "Commerce",
    description: "Fees, extras, bundles, inventory, and giveaways.",
    steps: ["pricing"]
  },
  {
    key: "review",
    number: 4,
    label: "Review",
    description: "Tournament preview, conflict resolution, publish, and registration.",
    steps: ["review"]
  }
];

export function tournamentSetupDomainForStep(
  step: TournamentSetupStep
): TournamentSetupDomain {
  return TOURNAMENT_SETUP_STEPS.find((row) => row.key === step)?.domain || "tournament";
}

export function tournamentSetupStepHref(
  step: TournamentSetupStep,
  tournamentId: string,
  tournamentName: string,
  drawId = ""
): string {
  return tournamentRouteHref(`/admin/tournaments/setup/${step}`, {
    tournamentId,
    tournamentName,
    drawId
  });
}

function stateLabel(state: TournamentSetupStepState | undefined): string | null {
  if (state === "complete") return "Complete";
  if (state === "in-progress") return "In progress";
  if (state === "blocked") return "Needs attention";
  return null;
}

function stateColors(state: TournamentSetupStepState | undefined) {
  if (state === "complete") {
    return { color: "#166534", background: "#dcfce7", borderColor: "#bbf7d0" };
  }
  if (state === "blocked") {
    return { color: "#991b1b", background: "#fee2e2", borderColor: "#fecaca" };
  }
  if (state === "in-progress") {
    return { color: "#92400e", background: "#fef3c7", borderColor: "#fde68a" };
  }
  return { color: "#475569", background: "#f8fafc", borderColor: "#cbd5e1" };
}

function domainState(
  steps: TournamentSetupStep[],
  states: Partial<Record<TournamentSetupStep, TournamentSetupStepState>>
): TournamentSetupStepState {
  const values = steps.map((step) => states[step] || "not-started");
  if (values.includes("blocked")) return "blocked";
  if (values.every((value) => value === "complete")) return "complete";
  if (values.some((value) => value === "complete" || value === "in-progress")) {
    return "in-progress";
  }
  return "not-started";
}

export default function TournamentSetupWizardNav({
  currentStep,
  tournamentId,
  tournamentName,
  drawId = "",
  states = {}
}: Props) {
  const currentDomain = tournamentSetupDomainForStep(currentStep);
  const activeDomain = TOURNAMENT_SETUP_DOMAINS.find((domain) => domain.key === currentDomain)!;

  return (
    <nav aria-label="Tournament builder domains">
      <ol
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))",
          gap: "0.65rem",
          margin: 0,
          padding: 0,
          listStyle: "none"
        }}
      >
        {TOURNAMENT_SETUP_DOMAINS.map((domain) => {
          const active = domain.key === currentDomain;
          const state = domainState(domain.steps, states);
          const badge = stateLabel(state);
          return (
            <li key={domain.key} style={{ minWidth: 0 }}>
              <Link
                href={tournamentSetupStepHref(domain.steps[0], tournamentId, tournamentName, drawId)}
                aria-current={active ? "step" : undefined}
                style={{
                  display: "grid",
                  gridTemplateColumns: "2rem minmax(0, 1fr)",
                  gap: "0.55rem",
                  alignItems: "start",
                  minHeight: "100%",
                  padding: "0.8rem",
                  border: `2px solid ${active ? "#2563eb" : "#e2e8f0"}`,
                  borderRadius: "14px",
                  background: active ? "#eff6ff" : "white",
                  color: "#0f172a",
                  textDecoration: "none"
                }}
              >
                <span
                  aria-hidden="true"
                  style={{
                    display: "inline-grid",
                    placeItems: "center",
                    width: "2rem",
                    height: "2rem",
                    borderRadius: "999px",
                    background: active ? "#2563eb" : "#e2e8f0",
                    color: active ? "white" : "#334155",
                    fontWeight: 900
                  }}
                >
                  {domain.number}
                </span>
                <span style={{ minWidth: 0 }}>
                  <strong style={{ display: "block" }}>{domain.label}</strong>
                  <small style={{ display: "block", marginTop: "0.2rem", color: "#64748b" }}>
                    {domain.description}
                  </small>
                  {badge ? (
                    <small
                      style={{
                        display: "inline-block",
                        marginTop: "0.35rem",
                        padding: "0.12rem 0.4rem",
                        border: "1px solid",
                        borderRadius: "999px",
                        fontWeight: 800,
                        ...stateColors(state)
                      }}
                    >
                      {badge}
                    </small>
                  ) : null}
                </span>
              </Link>
            </li>
          );
        })}
      </ol>

      {activeDomain.steps.length > 1 ? (
        <div
          style={{
            display: "flex",
            gap: "0.55rem",
            flexWrap: "wrap",
            marginTop: "0.7rem",
            padding: "0.7rem",
            border: "1px solid #dbeafe",
            borderRadius: "12px",
            background: "#f8fafc"
          }}
        >
          {activeDomain.steps.map((step) => {
            const definition = TOURNAMENT_SETUP_STEPS.find((row) => row.key === step)!;
            const active = step === currentStep;
            return (
              <Link
                key={step}
                href={tournamentSetupStepHref(step, tournamentId, tournamentName, drawId)}
                aria-current={active ? "page" : undefined}
                style={{
                  padding: "0.45rem 0.7rem",
                  borderRadius: "999px",
                  border: `1px solid ${active ? "#2563eb" : "#cbd5e1"}`,
                  background: active ? "#dbeafe" : "white",
                  color: active ? "#1d4ed8" : "#334155",
                  textDecoration: "none",
                  fontWeight: 800
                }}
              >
                {definition.shortLabel}
              </Link>
            );
          })}
        </div>
      ) : null}
    </nav>
  );
}
