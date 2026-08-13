# App-wide interaction standard implementation plan

This plan turns `docs/interaction-standard.md` into a small shared interaction layer, then migrates every audited write without rewriting domain forms or APIs.

## Repository findings at audit start

Static inspection of `apps/web` found:

- 146 `ConfirmAction` instances across 43 consumer files.
- Only two consumer modules import `ConfirmActionSuccess`, so persistent in-dialog success is currently exceptional rather than the default.
- `ConfirmAction` already has important strengths: a body portal, native `<dialog>`, duplicate-submit ref, disabled Working state, error display for rejected callbacks, and trigger-focus restoration.
- Its current `onConfirm` contract accepts `void`. Normal resolution therefore closes the dialog. Many callers catch an error, set a page message, and return normally; from the dialog's perspective that is a successful callback and it closes.
- Nine dialog implementations appear across eight files. In addition to the shared native dialog, the tournament Create/Edit/Bulk flows and Player Editor include hand-built `role="dialog"` overlays without one common focus, Escape, dirty-state, error, or restoration contract.
- Four native `window.confirm` calls remain in the admin and public generator round runners.
- No dedicated behavioral test currently exercises `ConfirmAction` focus, keyboard, Working, rejected request, persistent success, or duplicate-submit behavior.
- Button, input, card, and feedback styles are repeatedly declared inline. Existing colors and sizing are consistent enough to extract without a visual redesign.

These counts are a starting snapshot, not the final action audit. The audit workbook is the authoritative row-by-row record.

## Minimal target architecture

Keep domain state and API calls inside their existing panels. Add one shared dialog primitive, one action lifecycle hook, a few small presentation components, and a compatibility export.

```text
apps/web/components/interaction/
  InteractionDialog.tsx
  InteractionDialog.module.css
  FormDialog.tsx
  ActionFeedback.tsx
  ChangeReview.tsx
  types.ts
  useActionLifecycle.ts
apps/web/components/ConfirmAction.tsx
apps/web/tests/interaction-standard.cjs
apps/web/e2e/interaction-standard.spec.ts
```

### `types.ts`

Define the strict shared contracts from the standard:

- `ActionSuccess`
- `ActionUncertain`
- `ActionCompletion`
- `ActionErrorKind`
- `InteractionActionError`
- `FieldErrors`

The public mutation signature becomes:

```ts
type ActionCallback = (confirmationText: string) => Promise<ActionCompletion>;
```

No `void` success is allowed. This deliberate compile-time break is the most reliable way to prove that every `ConfirmAction` caller was reviewed. A callback must either return a user-facing completion or throw.

### `useActionLifecycle.ts`

Own the lifecycle shared by confirmation and form dialogs:

- phases: `ready | working | success | error | uncertain`;
- synchronous duplicate-submit lock in addition to React state;
- typed error normalization;
- field-error and recovery metadata;
- `run`, `reset`, and `acknowledge` operations;
- no automatic close on normal callback resolution;
- no automatic retry for uncertainty.

The hook does not own domain data or issue fetches. It only orchestrates one supplied action.

Proposed public shape:

```ts
type ActionLifecycle = {
  phase: "ready" | "working" | "success" | "error" | "uncertain";
  completion: ActionCompletion | null;
  error: InteractionActionError | null;
  run: (action: () => Promise<ActionCompletion>) => Promise<void>;
  reset: () => void;
};

function useActionLifecycle(): ActionLifecycle;
```

### `InteractionDialog.tsx`

Provide the body-portal native `<dialog>` shell used by every modal flow:

- labelled title and description IDs;
- initial focus, focus containment, and trigger/fallback focus restoration;
- safe Escape/backdrop behavior;
- dirty-form close guard;
- `aria-busy`, alert, polite status, and success focus behavior;
- responsive content and action slots;
- optional danger styling without domain logic.

Do not use an App Router intercepting route for these local CRUD interactions. They are stateful actions inside already-loaded client panels, and the native dialog primitive avoids route/history churn. Dedicated editor pages remain pages.

Proposed public shape:

```ts
type InteractionDialogProps = {
  open: boolean;
  phase: ActionLifecycle["phase"];
  title: ReactNode;
  description?: ReactNode;
  children: ReactNode;
  actions: ReactNode;
  initialFocusRef?: RefObject<HTMLElement>;
  returnFocusRef?: RefObject<HTMLElement>;
  dirty?: boolean;
  onRequestClose: () => void;
};
```

### `FormDialog.tsx`

Compose `InteractionDialog` with `useActionLifecycle` for Add/Edit forms. It accepts form content, dirty state, a submit callback returning `ActionCompletion`, and optional first-invalid-field resolution. Existing event, division, court, preset, and bulk dialog bodies can move into it without changing their domain field state.

Proposed public shape:

```ts
type FormDialogProps = {
  open: boolean;
  mode: "create" | "edit" | "bulk";
  title: ReactNode;
  description?: ReactNode;
  dirty: boolean;
  submitLabel: string;
  workingLabel?: string;
  children: ReactNode;
  onSubmit: () => Promise<ActionCompletion>;
  onCancel: () => void;
  onAcknowledge?: (success: ActionSuccess) => void;
};
```

### `ActionFeedback.tsx`

Render the common Working, Error, Uncertain, and Success panels. This prevents each feature from guessing severity by parsing message text. It is usable in a dedicated-page workflow as well as inside a dialog.

### `ChangeReview.tsx`

Render accessible human-readable before/after rows for Edit, Bulk Edit, Publish, and guarded actions. It accepts already-formatted labels and values; it must not infer domain meaning from raw objects.

### `ConfirmAction.tsx`

Keep the existing import path so 43 consumer files do not need import churn. Reimplement it as a thin composition of `InteractionDialog`, `useActionLifecycle`, and `ActionFeedback`.

Proposed public shape:

```ts
type ConfirmActionProps = {
  triggerLabel: string;
  title: ReactNode;
  description: ReactNode;
  preview?: ReactNode;
  confirmLabel: string;
  cancelLabel?: string;
  workingLabel?: string;
  confirmationText: string;
  tone?: "default" | "danger";
  disabled?: boolean;
  disabledReason?: ReactNode;
  onConfirm: ActionCallback;
  onAcknowledge?: (success: ActionSuccess) => void;
};
```

Required changes:

- replace the `void | ConfirmActionSuccess` callback with strict `Promise<ActionCompletion>`;
- retain the synchronous double-submit guard;
- keep the dialog open for every resolved success;
- keep rejected errors visible;
- represent uncertain outcomes explicitly;
- allow action-specific working labels and success follow-ups;
- expose stable `data-state` values for tests;
- move shared visual rules into the interaction CSS module.

#### Persistent-success decision

Persistent success is an invariant, not an optional `defaultSuccess` prop. `ConfirmAction` must never invent “Action complete” merely because a legacy callback resolved: a handler may have caught an API error, set a page message, and returned `void`. Synthesizing success in that case would turn the existing silent-close defect into a false-success defect.

The safe migration is therefore intentionally strict:

1. remove `void` from the callback result type;
2. make TypeScript identify every consumer;
3. update each handler to return a specific `ActionCompletion` only after authoritative success/readback;
4. rethrow or translate every caught error to `InteractionActionError`;
5. let the shared component render every resolved success persistently until acknowledgement.

There is no legacy auto-close mode in the finished implementation. If batching is necessary during development, the compatibility branch may retain the old type only while an audit row is actively being migrated, but CI must reject any legacy consumer before merge.

## Migration rule for callbacks

Each audited action function follows one shape:

```ts
async function saveThing(confirmationText: string): Promise<ActionCompletion> {
  validateOrThrow();
  try {
    const response = await guardedRequest(/* existing payload */);
    const authoritative = await reloadOrUseAuthoritativeResponse(response);
    return {
      status: "success",
      title: "Thing saved",
      description: <>Readable exact result from {authoritative.name}.</>
    };
  } catch (error) {
    throw toInteractionActionError(error);
  }
}
```

Never use this shape:

```ts
try {
  await request();
  setMessage("Saved");
} catch (error) {
  setMessage(messageFor(error));
}
// implicit void closes the current dialog
```

Page notices may remain for durable context after acknowledgement, but they cannot substitute for the dialog lifecycle.

## Audit record schema

Every action receives one row with these columns before it is changed:

| Column | Purpose |
| --- | --- |
| Area / route / source file | Locate the surface and code owner |
| Object and action | Human-readable target and verb |
| Class | Create, Edit, Delete, Bulk Edit, Publish, Guarded, or non-write |
| Current surface | Inline, native browser prompt, hand-built dialog, shared dialog, dedicated page |
| Scope/consequence preview | None, partial, or complete |
| Busy/duplicate safety | Visible state and request-lock evidence |
| Success | Silent close, page-only message, or persistent same-surface result |
| Error/value preservation | Error location, retry path, and whether values survive |
| Concurrency guard | Expected timestamp/version/fingerprint |
| Idempotency/recovery | Key lifetime, uncertain state, reconcile path |
| Accessibility | Name, focus, keyboard, live region, touch/zoom |
| Finding / required change | Concrete defect or “conforms” with evidence |
| Implementation status | Not started, fixed, verified, or justified exception |
| Automated test | Test name covering the row or family |

The audit must include direct write buttons as well as `ConfirmAction` consumers. A simple search for confirmation components is not sufficient.

## Implementation sequence

### 1. Freeze and enumerate

1. Generate the workbook/action inventory from all client panels and route forms.
2. Search for `ConfirmAction`, mutation HTTP methods, write-like handlers, hand-built dialogs, and native browser prompts.
3. Classify read-only refresh/filter/export actions separately so they are not given unnecessary confirmation friction.
4. Record each action's API concurrency/idempotency support and any missing backend prerequisite.

Exit: every Create, Edit, Delete, Bulk Edit, Publish, and guarded action has an audit row.

### 2. Build the shared foundation

1. Add the shared types, lifecycle hook, dialog shell, feedback, review, and CSS.
2. Convert `ConfirmAction` while preserving its import path.
3. Add a small interaction harness to the existing admin Theme QA route or a test-only page available in local/CI builds.
4. Add Playwright behavioral tests and a static source guard.

Exit: the primitive passes accessibility/lifecycle tests before domain migrations begin.

### 3. Remove known escape hatches

1. Replace the four generator `window.confirm` calls with shared confirmations.
2. Migrate hand-built `role="dialog"` overlays to `InteractionDialog`/`FormDialog`.
3. Remove duplicated overlay and dialog button styling after visual parity is confirmed.

Exit: source guard finds no native browser prompt and no product-owned `role="dialog"` outside the shared primitive.

### 4. Migrate highest-consequence actions first

Convert Publish, Open/Close, ratings/match publication, merge, replay/recovery, email send/retry, destructive replacement, roles, refund/cancel, and Delete actions.

For each action:

1. return an explicit completion;
2. rethrow typed failure instead of swallowing it;
3. preserve exact idempotency/operation identity on uncertainty;
4. verify authoritative readback before success;
5. add action-family tests.

Exit: no high-consequence audit row is page-message-only or auto-closing.

### 5. Migrate Create and Edit surfaces

1. Move contained Add/Edit forms into `FormDialog`.
2. Leave compact read-only cards/rows behind with explicit Edit actions.
3. Preserve form values and focus on validation or API errors.
4. Show a persistent success result, then update/focus the authoritative card.
5. For justified complex dedicated pages, use the same lifecycle for final save and record the exception in the audit.

Exit: every Create/Edit row conforms or has a documented dedicated-page exception that still satisfies lifecycle requirements.

### 6. Migrate Bulk Edit

1. Use the existing tournament division bulk editor as the product baseline: compatible shared fields, untouched mixed values, and per-row preview.
2. Migrate its shell to `FormDialog` and add strict success/error behavior.
3. Apply the same review/atomicity/count rules to registrations, roster, match-log, and other bulk actions.

Exit: every bulk row documents shared-field compatibility, atomic commit, exact result counts, and stale-review protection.

### 7. Consolidate presentation and remove legacy paths

1. Replace page-specific success modals with the shared primitive.
2. Replace severity-by-message-regex with typed feedback.
3. Remove duplicated inline dialog styles and unused result state.
4. Keep domain-specific card/input layout local; do not turn this effort into a full design-system rewrite.

Exit: one shared modal implementation and one typed lifecycle serve all audited actions.

## Automated acceptance tests

Use Playwright, which is already installed, for behavior that requires a browser. Add a light static test for prohibited source patterns. Avoid regex-only tests for lifecycle behavior.

### Shared primitive tests

1. Trigger exposes `aria-haspopup="dialog"` and opens a labelled native dialog.
2. Create/Edit initial focus lands on the configured field; destructive confirmation lands on Cancel.
3. Tab/Shift+Tab remain within the modal.
4. Escape/backdrop closes a clean ready dialog and restores trigger focus.
5. Escape/backdrop cannot close Working or a dirty form without a discard decision.
6. Rapid double click and Enter create exactly one callback invocation.
7. Working state remains visible, announces status, disables actions, and sets `aria-busy`.
8. Typed validation error keeps values, shows alert, and focuses the first invalid field.
9. Rejected server action keeps the dialog open and permits safe retry.
10. Success replaces confirmation in the same dialog, is announced, and remains until Done/OK.
11. Uncertain result shows the operation reference, prevents a new mutation, and runs only reconciliation.
12. Closing success focuses the updated card/fallback if the original trigger was removed.
13. Layout remains usable at 320px width and 200% zoom; buttons meet 44px target size.

### Pattern tests

- **Create:** error preserves form; success names record; acknowledgement reveals/focuses read-only card.
- **Edit:** readable before/after; stale response stays open; reload does not silently overwrite proposed values.
- **Delete:** target/consequence named; failure in-use lists dependencies; success states recovery status.
- **Bulk:** mixed values untouched; preview covers all rows; double submit once; atomic failure is not success; exact counts displayed.
- **Publish:** exact reviewed fingerprint sent; Working persists through readback; success distinguishes publication from availability; lost response becomes uncertain and does not republish.
- **Guarded:** confirmation token, expected version, and idempotency key are stable; reconciliation reuses operation identity; success waits for required downstream evidence.

### Static source guard

Fail CI when application source contains:

- `window.confirm`, `window.alert`, or `window.prompt`;
- product-owned `role="dialog"` outside the shared interaction component/test harness;
- a `ConfirmAction` callback typed or inferred as returning `void`;
- a caught action error that resolves normally from a strict action callback.

The last rule is best enforced by TypeScript's strict callback result type; the source guard should cover obvious bypasses, not attempt to parse the language completely.

## Verification commands

Run focused checks continuously and the complete suite before merge:

```bash
cd apps/web
npx tsc --noEmit
npm run test:component
npx playwright test e2e/interaction-standard.spec.ts --retries=0 --forbid-only
npm run build
```

Then run the repository's focused Python/API tests for every changed action family so UI success claims remain aligned with backend confirmation, version, idempotency, and recovery contracts.

## Definition of done

- The workbook contains every in-scope action and links it to evidence.
- Every audit row is fixed and verified, or has a narrow documented exception that still satisfies the lifecycle.
- No native browser prompts or one-off modal overlays remain.
- Every consequential callback returns explicit success/uncertainty or throws typed failure; `void` cannot compile.
- Working, error, uncertainty, and success all remain in the action surface as specified.
- User input survives errors; authoritative state is shown after success.
- Publish and guarded actions preserve review fingerprints, versions, idempotency keys, and recovery paths.
- Shared and pattern-specific automated tests pass.
- Next.js typecheck and production build pass.
- Manual keyboard, screen-reader announcement, mobile-width, 200%-zoom, and reduced-motion checks are recorded.
- Production remains untouched until the normal promotion and acceptance process explicitly approves it.
