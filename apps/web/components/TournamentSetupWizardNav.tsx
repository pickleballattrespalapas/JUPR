import Link from "next/link";

export type TournamentSetupStep =
  | "basics"
  | "events"
  | "registration-rules"
  | "pricing"
  | "schedule"
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
  states?: Partial<Record<TournamentSetupStep, TournamentSetupStepState>>;
};

export const TOURNAMENT_SETUP_STEPS: Array<{
  key: TournamentSetupStep;
  number: number;
  label: string;
  shortLabel: string;
  description: string;
}> = [
  {
    key: "basics",
    number: 1,
    label: "Tournament basics",
    shortLabel: "Basics",
    description: "Name and tournament dates."
  },
  {
    key: "events",
    number: 2,
    label: "Events and formats",
    shortLabel: "Events",
    description: "Divisions, eligibility, formats, scoring, and capacity."
  },
  {
    key: "registration-rules",
    number: 3,
    label: "Registration rules",
    shortLabel: "Rules",
    description: "Registration window, waitlist, Partner Board, and policies."
  },
  {
    key: "pricing",
    number: 4,
    label: "Pricing, extras, and fulfillment",
    shortLabel: "Pricing",
    description: "Entry fees, merchandise, bundles, inventory, and pickup."
  },
  {
    key: "schedule",
    number: 5,
    label: "Schedule and courts",
    shortLabel: "Schedule",
    description: "Tournament days and event-day assignments."
  },
  {
    key: "review",
    number: 6,
    label: "Review and open registration",
    shortLabel: "Review",
    description: "Resolve warnings, publish setup, and open registration."
  }
];

export function tournamentSetupStepHref(
  step: TournamentSetupStep,
  tournamentId: string,
  tournamentName: string
): string {
  const params = new URLSearchParams({ tournament: tournamentId });
  if (tournamentName) params.set("name", tournamentName);
  return `/admin/tournaments/setup/${step}?${params.toString()}`;
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

export default function TournamentSetupWizardNav({
  currentStep,
  tournamentId,
  tournamentName,
  states = {}
}: Props) {
  return (
    <nav aria-label="Tournament setup steps">
      <ol
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(155px, 1fr))",
          gap: "0.65rem",
          margin: 0,
          padding: 0,
          listStyle: "none"
        }}
      >
        {TOURNAMENT_SETUP_STEPS.map((step) => {
          const active = step.key === currentStep;
          const state = states[step.key];
          const badge = stateLabel(state);
          return (
            <li key={step.key} style={{ minWidth: 0 }}>
              <Link
                href={tournamentSetupStepHref(
                  step.key,
                  tournamentId,
                  tournamentName
                )}
                aria-current={active ? "step" : undefined}
                style={{
                  display: "grid",
                  gridTemplateColumns: "2rem minmax(0, 1fr)",
                  gap: "0.55rem",
                  alignItems: "start",
                  minHeight: "100%",
                  padding: "0.75rem",
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
                  {step.number}
                </span>
                <span style={{ minWidth: 0 }}>
                  <strong style={{ display: "block" }}>{step.label}</strong>
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
    </nav>
  );
}
