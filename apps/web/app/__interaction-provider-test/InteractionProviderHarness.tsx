"use client";

import { useEffect, useState } from "react";

import { ConfirmAction } from "@/components/ConfirmAction";
import {
  actionSuccess,
  FormDialog,
  useInteraction,
  type ActionSuccess
} from "@/components/interaction";
import TournamentSetupDivisionDialog from "@/app/admin/tournaments/setup/TournamentSetupDivisionDialog";

function DisposableRecord({ onRemove }: { onRemove: () => void }) {
  return (
    <ConfirmAction
      triggerLabel="Remove record"
      title="Remove this record?"
      description="This harness removes its own trigger before returning success."
      confirmLabel="Remove record"
      confirmationText="REMOVE RECORD"
      tone="danger"
      onConfirm={async () => {
        onRemove();
        return actionSuccess(
          "Record removed",
          "The authoritative record no longer exists."
        );
      }}
    />
  );
}

function OneActionLockHarness() {
  const { openAction } = useInteraction();
  const [result, setResult] = useState("not run");

  return (
    <section>
      <h2>One-action lock</h2>
      <button
        type="button"
        onClick={(event) => {
          const origin = event.currentTarget;
          const firstAccepted = openAction(
            {
              id: "lock-first",
              title: "First action",
              description: "Only this interaction should open.",
              confirmLabel: "Confirm first",
              cancelLabel: "Cancel first",
              workingLabel: "Working…",
              confirmationText: "FIRST",
              tone: "default",
              onConfirm: async () => actionSuccess("First complete", "First action completed.")
            },
            origin
          );
          const secondAccepted = openAction(
            {
              id: "lock-second",
              title: "Second action",
              description: "This interaction must be rejected by the synchronous lock.",
              confirmLabel: "Confirm second",
              cancelLabel: "Cancel second",
              workingLabel: "Working…",
              confirmationText: "SECOND",
              tone: "default",
              onConfirm: async () => actionSuccess("Second complete", "Second action completed.")
            },
            origin
          );
          setResult(`${firstAccepted}:${secondAccepted}`);
        }}
      >
        Exercise one-action lock
      </button>
      <output data-testid="lock-result">{result}</output>
    </section>
  );
}

function explicitFocusSuccess(): ActionSuccess {
  return {
    status: "success",
    title: "Focus priority confirmed",
    description: "The explicit focus target must win over the connected trigger.",
    focusTargetId: "explicit-focus-target"
  };
}

function DisabledTriggerFocusHarness() {
  const [triggerDisabled, setTriggerDisabled] = useState(false);

  return (
    <section>
      <h2>Disabled trigger focus fallback</h2>
      <ConfirmAction
        triggerLabel="Disable trigger after success"
        title="Check disabled-trigger focus fallback?"
        description="The originating button remains connected but becomes disabled after success."
        confirmLabel="Complete disabled-trigger check"
        confirmationText="CHECK DISABLED FOCUS"
        disabled={triggerDisabled}
        onConfirm={async () => {
          setTriggerDisabled(true);
          return actionSuccess(
            "Disabled-trigger fallback confirmed",
            "Acknowledgement must focus the main content instead of the now-disabled trigger."
          );
        }}
      />
    </section>
  );
}

function DisabledFormTriggerFocusHarness() {
  const [open, setOpen] = useState(false);
  const [triggerDisabled, setTriggerDisabled] = useState(false);

  return (
    <section>
      <h2>Disabled form trigger focus fallback</h2>
      <button
        type="button"
        disabled={triggerDisabled}
        onClick={() => setOpen(true)}
      >
        Open form focus check
      </button>
      <FormDialog
        open={open}
        mode="edit"
        title="Edit form focus check"
        description="The form trigger remains connected but becomes disabled after success."
        dirty={false}
        submitLabel="Complete form focus check"
        onSubmit={async () => {
          setTriggerDisabled(true);
          return actionSuccess(
            "Form disabled-trigger fallback confirmed",
            "Acknowledgement must focus main instead of the disabled form trigger."
          );
        }}
        onCancel={() => setOpen(false)}
      >
        <label>
          Harness value
          <input defaultValue="unchanged" />
        </label>
      </FormDialog>
    </section>
  );
}

function ExplicitFormFocusHarness() {
  const [open, setOpen] = useState(false);

  return (
    <section>
      <h2>Form explicit focus precedence</h2>
      <button type="button" onClick={() => setOpen(true)}>
        Open explicit form focus check
      </button>
      <button id="form-explicit-focus-target" type="button">
        Form explicit focus target
      </button>
      <FormDialog
        open={open}
        mode="edit"
        title="Edit explicit form focus check"
        dirty={false}
        submitLabel="Complete explicit form focus check"
        onSubmit={async () => ({
          ...actionSuccess(
            "Form focus priority confirmed",
            "The explicit success focus target must win over the connected form trigger."
          ),
          focusTargetId: "form-explicit-focus-target"
        })}
        onCancel={() => setOpen(false)}
      >
        <label>
          Harness value
          <input defaultValue="unchanged" />
        </label>
      </FormDialog>
    </section>
  );
}

function DivisionDraftRetentionHarness() {
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<"add" | "edit">("add");
  const [parentRenderCount, setParentRenderCount] = useState(0);
  const [sourceName, setSourceName] = useState("Default division name");

  useEffect(() => {
    if (!open) return;
    const timer = window.setInterval(
      () => setParentRenderCount((current) => current + 1),
      50
    );
    return () => window.clearInterval(timer);
  }, [open]);

  const family = {
    event_family_label: "Harness doubles",
    event_family: "Harness doubles",
    participant_type: "DOUBLES",
    gender_restriction: "ANY",
    competition_format: "STANDARD",
    team_roster_size: 2,
    default_capacity_teams: 16,
    default_price_usd: 0,
    default_waitlist: true,
    default_partner_board: true,
    scheduled_day_ids: ["harness-day"],
    age_mode: "ALL_AGES",
    age_label: "All Ages",
    age_rules: { mode: "ALL_AGES" }
  };
  const initialValue = {
    id: mode === "add" ? `generated-${parentRenderCount}` : "existing-division",
    event_family_label: "Harness doubles",
    event_family: "Harness doubles",
    division_name: sourceName,
    label: sourceName,
    skill_label: "3.5",
    skill_mode: "STANDARD",
    eligibility_mode: "STANDARD",
    capacity_teams: 16,
    price_usd: 0,
    waitlist_enabled: true,
    partner_board_enabled: true,
    schedule_mode: "INHERIT_EVENT",
    scheduled_day_ids: ["harness-day"],
    age_policy_source: "INHERIT_EVENT"
  };

  return (
    <section>
      <h2>Division draft retention</h2>
      <button type="button" onClick={() => { setMode("add"); setOpen(true); }}>
        Open add division retention check
      </button>
      <button type="button" onClick={() => { setMode("edit"); setOpen(true); }}>
        Open edit division retention check
      </button>
      <button
        type="button"
        onClick={() => setSourceName("Updated authoritative division")}
      >
        Update authoritative division source
      </button>
      <output data-testid="division-parent-render-count">{parentRenderCount}</output>
      <TournamentSetupDivisionDialog
        open={open}
        mode={mode}
        initialValue={initialValue}
        eventFamilies={[{ key: "harness-family", value: family }]}
        days={[
          {
            key: "harness-day",
            value: {
              id: "harness-day",
              label: "Harness day",
              event_date: "2026-08-14",
              enabled: true
            }
          }
        ]}
        onCancel={() => setOpen(false)}
        onConfirm={async () =>
          actionSuccess(
            "Division retained",
            "The division draft survived parent rerenders."
          )
        }
        onAcknowledge={() => setOpen(false)}
      />
    </section>
  );
}

export function InteractionProviderHarness() {
  const [recordExists, setRecordExists] = useState(true);

  return (
    <div>
      <h1>Interaction provider harness</h1>
      <section>
        <h2>Consumer unmount</h2>
        {recordExists ? (
          <DisposableRecord onRemove={() => setRecordExists(false)} />
        ) : (
          <p id="record-gone">Record removed from the consumer tree.</p>
        )}
      </section>
      <OneActionLockHarness />
      <section>
        <h2>Focus precedence</h2>
        <ConfirmAction
          triggerLabel="Check focus precedence"
          title="Check explicit focus precedence?"
          description="The success outcome identifies a focus target."
          confirmLabel="Complete focus check"
          confirmationText="CHECK FOCUS"
          onConfirm={async () => explicitFocusSuccess()}
        />
        <button id="explicit-focus-target" type="button">Explicit focus target</button>
      </section>
      <DisabledTriggerFocusHarness />
      <DisabledFormTriggerFocusHarness />
      <ExplicitFormFocusHarness />
      <DivisionDraftRetentionHarness />
    </div>
  );
}
