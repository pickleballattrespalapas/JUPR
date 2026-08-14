# JUPR app-wide interaction standard

Status: normative application standard

Scope: Next.js public and administrative interfaces
Applies to: Create, Edit, Delete, Bulk Edit, Publish, and every guarded or consequential write

## Purpose

Every action must tell the user what will happen, visibly remain in progress while it happens, and state what actually happened. A successful request must never look like a silent dismissal. A failed request must never erase entered work or imply that nothing may have been written.

The standard consolidates the strongest behavior already proven during acceptance testing:

- Add and Edit use focused dialogs and return to compact, read-only cards.
- The complete tournament registration editor keeps related identity, eligibility, event-entry, partner, extras, payment, and status data together without hiding the authoritative result.
- Division setup presents the five supported eligibility modes in user language and previews the effective rule rather than exposing stored JSON.
- An unpaired player receives a provisional age placement from their own information; data that only a future partner can supply is recalculated later instead of being treated as a false setup conflict.
- Tournament publication keeps its success result in the confirmation dialog until the user acknowledges it, preserves the separate registration status, and distinguishes Publish setup from Open registration.

These are reusable product rules, not tournament-only exceptions.

## Normative language

**Must** and **must not** are release requirements. **Should** identifies the expected default; an exception requires an audit note explaining why it is safer or clearer. **May** identifies an optional enhancement.

## Core rules

1. **No silent completion.** A successful consequential action must end in a persistent success state with an acknowledgement control such as **Done** or **OK**.
2. **No disappearing errors.** Validation, conflict, authorization, server, and network errors must remain visible beside the action while the user's values remain intact.
3. **One action, one visible lifecycle.** Confirmation, Working, Error/Recovery, and Success belong to the same dialog or focused workflow.
4. **Authoritative truth wins.** After a write, reload or apply the authoritative response before claiming success. Draft, published, open, closed, archived, and similar labels must come from the authoritative field for that state.
5. **No accidental duplicates.** A double click or repeated key press must create at most one in-flight request. Guarded writes must retain their exact idempotency key or request identity until the result is known.
6. **Use human language.** Show names, dates, counts, before/after values, and consequences. Do not make IDs, raw JSON, fingerprints, or internal flags the main explanation.
7. **Preserve context.** Cancelling makes no change. A rejected save preserves input. A successful save updates the relevant card/list or navigates to an authoritative detail view with an explicit success result.
8. **Consequences determine friction.** Routine reversible edits do not need theatrical warnings. Destructive, public, financial, rating-affecting, bulk, and irreversible actions require review and explicit confirmation.
9. **Draft save, publication, and availability are distinct.** Saving a private draft does not imply publication. Publishing setup does not imply opening registration or another public availability switch.
10. **No native browser prompts.** `window.confirm`, `window.alert`, and `window.prompt` are not permitted product interaction surfaces.

## Action lifecycle

All shared action dialogs implement this state machine:

```text
closed
  -> ready
  -> working
  -> success -> acknowledged -> closed
  -> error -> ready/retry or cancel
  -> uncertain -> reconcile/inspect -> success or error
```

| State | Required presentation | Allowed exits |
| --- | --- | --- |
| Ready | Title, consequence, readable scope/preview, Cancel, explicit action label | Cancel; start action |
| Working | Same dialog, **Working…** or action-specific progressive label, controls disabled, `aria-busy="true"` | Completion only; do not dismiss or submit again |
| Success | Same dialog, success heading, exact effect/count, warnings or unchanged state, **Done/OK** | Acknowledge; optional safe follow-up link |
| Error | Same dialog, specific correction/retry guidance, original inputs/preview preserved | Retry; edit; cancel when safe |
| Uncertain | Same dialog, clear “outcome not yet confirmed,” retained operation reference, no blind new request | Reconcile/inspect exact request; safe close if recovery remains visible elsewhere |

An action callback must not catch an error and then return normally. Normal resolution means success. A callback that cannot prove success must throw a typed failure or return an explicit uncertain result.

## Choosing the interaction surface

### Focused form dialog

Use for adding or editing one contained record from a collection, card, or setup screen. Examples include an event, division, tournament day, sponsor, extra, roster member, or compact player field set.

- The underlying saved representation remains a compact read-only card or row.
- **Add** or **Edit** opens the dialog with a clear noun in the title.
- Save occurs directly to the private draft or authoritative record described by the dialog; navigation must not imply that a second save is required.
- A failed save leaves the dialog and all field values open.
- The form initializes its locally editable field values when the dialog opens. Session refreshes, background reads, and equivalent parent rerenders must never replace those unsaved values while it remains open. Current authoritative records and limits may still inform previews and final validation without overwriting the operator's inputs.
- A successful save shows completion, then **Done** returns to the updated card.
- If the form is dirty, backdrop/Escape dismissal must not silently discard it. Either disable those dismissals or ask the user to discard changes within the same dialog flow.

### Dedicated page

A dedicated page is appropriate for a long, multi-section workflow, live-operation surface, or record whose context is the page itself. The same lifecycle still applies to its final action: the action dialog or focused status area stays visible through Working, Error/Recovery, and Success.

### Inline action

A read-only refresh, filter, local preview, copy, expand/collapse, or other action with no durable side effect may be inline. It still needs a visible loading state and accessible error/status message when asynchronous.

## Pattern requirements

### Create

- Present a focused Add dialog for contained records.
- Identify what will be created and whether it is a private draft, public record, or inactive shell.
- Validate fields before sending; identify the field and explain the correction.
- During the request, disable duplicate submission and keep the dialog open.
- Success names the created record and its resulting state.
- After acknowledgement, show the created record as a read-only card/row with an explicit **Edit** action.
- Top-level creation that must navigate may navigate only after an acknowledged success, or land on the created record with a durable success notice announced to assistive technology.

### Edit

- Saved values are read-only until the user chooses **Edit**.
- The dialog starts from the authoritative loaded version.
- Preview meaningful before/after values for consequential changes.
- Do not overwrite omitted or “Multiple values” fields.
- A conflict/stale-version response stays in the dialog and offers reload/review; it must not silently replace either version.
- Success states what changed and refreshes the authoritative card/detail.

### Delete, remove, cancel, archive, and revoke

- Use the danger tone only when the action removes access/data, cancels participation, refunds, archives, replaces authoritative data, or is otherwise materially destructive.
- Name the exact target and explain downstream effects and reversibility.
- The confirm label must be specific: **Delete draft**, **Remove event entry**, **Cancel order**, not **Confirm**.
- Use an expected version/fingerprint when the API supports it.
- Success remains visible and states what was removed and whether recovery is possible.
- If the server blocks deletion because the record is in use, show the readable dependencies and keep the dialog open.

### Bulk Edit

- Start from an explicit selected count and identify the population.
- Offer only fields compatible across every selected record.
- Unchecked fields remain untouched. A mixed field displays **Multiple values** and is not normalized unless the user checks and changes it.
- Show a per-record before/after preview, including rows with **No change**.
- Require a fresh reviewed fingerprint/version before commit.
- Save atomically. Partial completion is an error/recovery state, never a success.
- Success reports updated, unchanged, skipped, and failed counts; a nonzero failure count cannot use a success-only treatment.

### Publish and other public-state changes

- A read-only review must precede publication and be bound to the exact reviewed fingerprint.
- Summarize affected public data and any registration, rating, communication, inventory, or availability impact.
- Keep separate controls for separate domain actions: for example, **Publish setup** and **Open registration**.
- The button and dialog use **Publishing…** or **Working…** until authoritative readback succeeds.
- Success remains in the same dialog and names what is published, what stayed unchanged, and the current public/registration status.
- A timeout or lost response is uncertain. Do not advise publishing again; reconcile the exact operation first.

### Guarded and high-consequence actions

Guarded actions include ratings writes/replays, match publication, player merges, bulk replacement, financial/status changes, email sends with duplicate-delivery risk, role changes, public publication, and database recovery operations.

- Show the reviewed scope and readable consequence before confirmation.
- Send the server-required confirmation phrase without requiring the user to memorize internal wording unless policy explicitly requires typed confirmation.
- Use expected row versions or a state fingerprint for compare-and-swap protection.
- Use a stable idempotency key for retryable writes. Do not rotate it until authoritative completion is confirmed.
- Record and retain the operation key where the backend provides one.
- Distinguish **failed** from **uncertain**. A network interruption after send is uncertain unless authoritative evidence proves failure.
- In uncertainty, disable a fresh mutation and present **Check operation**, **Reconcile**, or a similarly exact recovery action.
- Success includes the meaningful write and required downstream work (for example, rating replay or audit completion), not merely a `200` response.

## Validation, errors, and recovery

Use this failure taxonomy consistently:

| Kind | Meaning | Required response |
| --- | --- | --- |
| `validation` | User-entered value or required selection is invalid | Keep values; focus first invalid field; inline field message plus error summary |
| `conflict` | Authoritative data changed after review | Keep proposed values; explain the stale item; reload/review before retry |
| `forbidden` | Session or permission is missing | Preserve safe draft values; provide sign-in or permission guidance; do not imply retry will work unchanged |
| `failed` | Server proved the action did not complete | Keep values/preview; show specific error and a safe retry when applicable |
| `uncertain` | Request may have completed but response/readback is missing | Preserve exact operation identity; block blind repeat; offer reconcile/inspect |

Do not classify errors by searching message text for words such as “unable.” API helpers and action callbacks must provide an explicit kind.

Route/render failures belong in the nearest Next.js `error.tsx`; expected mutation failures belong inside the active action/dialog and must not tear down the route.

## Content and visual conventions

The existing PCS visual language remains the baseline:

- dialog: white surface, slate text, 16px radius, slate translucent backdrop;
- primary: slate `#0f172a` on white;
- danger: red `#991b1b` on white;
- success: green text/surface family (`#166534`, `#f0fdf4`);
- warning/recovery: amber text/surface family (`#92400e`, `#fffbeb`);
- error: red text/surface family (`#b91c1c`, `#fef2f2`);
- focus: visible 3px blue outline (`#60a5fa`);
- buttons: minimum 44px target, clear disabled state, action verb plus noun;
- mobile: dialog fits within the visual viewport; action buttons become full-width without reversing their logical tab order.

Color is supplementary. Every state includes text and, where useful, a heading or icon with an accessible name.

## Accessibility requirements

- Use the shared native `<dialog>` primitive rendered in a body portal. Do not hand-build a `role="dialog"` overlay.
- Every dialog has a unique accessible title and description.
- Opening focus goes to the first form field for Create/Edit, or the least destructive control for destructive confirmation.
- Native modal focus containment is retained. Background content is not keyboard reachable while open.
- Escape/backdrop closes only when closing is safe and the action is not working. Dirty forms require a discard decision.
- Working uses `aria-busy="true"` and a polite live status.
- Errors use `role="alert"`; focus moves to the error summary or first invalid field.
- Success uses a focused heading or polite live status and remains until acknowledged.
- Closing restores focus to the still-connected, eligible/focusable trigger; otherwise it focuses the updated card, next logical record, or `<main>`.
- Form controls have programmatic labels, help/error text linked with `aria-describedby`, and `aria-invalid` when invalid.
- Before/after comparisons use semantic lists or tables with headers, not color-only chips.
- Touch targets are at least 44 by 44 CSS pixels and all behavior is keyboard operable.
- Test at 320 CSS pixels wide, 200% browser zoom, and with reduced motion enabled.

## Data and async integrity

- Client callbacks may update optimistic local values only when rollback is reliable and the action is not high consequence.
- Consequential action success requires authoritative response data or a readback.
- `useLatestRequestGuard` remains appropriate for stale reads and selection changes. It does not replace an idempotency key for writes.
- A dialog owns its own in-flight lock. A page-level `busy` flag may additionally prevent incompatible writes, but must not be the only duplicate-submit guard.
- Initialize editable local dialog state once per explicit open session, not whenever an object or array prop receives a new JavaScript identity.
- Do not clear the form, rotate the idempotency key, close the dialog, or navigate until success is confirmed.
- Do not hide a partial result behind generic success. Report counts and warnings explicitly.
- Server confirmation, expected-version/fingerprint, authorization, and write flags remain server-enforced; the dialog is not a security boundary.

## Required action result contract

Every consequential client mutation resolves to an explicit result or throws a typed error. `void` is not a valid successful result.

```ts
type ActionSuccess = {
  status: "success";
  title: string;
  description: ReactNode;
  closeLabel?: string;
  focusTargetId?: string;
};

type ActionUncertain = {
  status: "uncertain";
  title: string;
  description: ReactNode;
  operationKey: string;
  recoveryLabel: string;
  onRecover: () => Promise<ActionSuccess | ActionUncertain>;
};

type ActionCompletion = ActionSuccess | ActionUncertain;

class InteractionActionError extends Error {
  kind: "validation" | "conflict" | "forbidden" | "failed";
  fieldErrors?: Record<string, string>;
  recovery?: ReactNode;
}
```

The shared lifecycle layer currently converts unknown exceptions into a generic `failed` message. That fallback is only safe when the caller knows the request did not start. Durable callers must classify transport uncertainty explicitly and return `ActionUncertain`; callers should use typed errors so the dialog can provide the correct recovery behavior.

## Release acceptance

An action conforms only when all applicable statements pass:

- The trigger clearly names the action and disabled state has a visible reason nearby.
- Confirmation/review identifies target, scope, consequences, and reversibility.
- Working state is visible, announced, non-dismissible, and duplicate-safe.
- Success remains visible in the same action surface until acknowledged.
- Validation/server/conflict errors remain visible and preserve user input.
- Uncertain outcomes retain operation identity and prevent blind repetition.
- Authoritative readback updates the card/page state.
- Keyboard, focus restoration, Escape, screen-reader naming, zoom, and mobile layout pass.
- The action has an automated happy-path test and at least one error/retry test.
- Destructive, bulk, publish, and guarded actions also have stale-version and duplicate-submit coverage.
