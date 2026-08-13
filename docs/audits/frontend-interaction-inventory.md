# Frontend interaction inventory

Static baseline audit of `apps/web` at revision `dcf08791`, completed 2026-08-13. This is the stable-ID implementation inventory for the app-wide Create/Edit/Delete/Bulk/Publish/guarded-action standard. No product code was changed during the inventory pass; the candidate remediation is summarized in [remediation-report.md](remediation-report.md).

The source counts and pattern grades below intentionally describe the pre-remediation baseline so the original gaps remain reproducible. They are not current-candidate conformance claims. The remediation report records what this candidate implements, what remains residual, and which acceptance stages have not occurred.

## Scope and counting rules

- Scanned all 256 `*.tsx` files under `apps/web/app` and `apps/web/components`; 109 are client components.
- Source contains 306 native `<button>` elements, 14 `<form>` elements, 146 `ConfirmAction` instances in 43 files, four `window.confirm()` calls, and 151 explicit `POST`/`PUT`/`PATCH`/`DELETE` method literals.
- Four of the 14 forms are read-only GET/search filters and are excluded from mutation findings: leaderboards, players, league players, and tournament roster.
- A “source instance” is one rendered control in source. A mapped control (for example, one league lifecycle `ConfirmAction` rendered for several available transitions) counts once in source metrics but every runtime label is named below.
- Local draft actions are included when they add/remove/edit data that will later be persisted, even if the immediate click does not call the API.
- Navigation, refresh/retry, print/download, copy-link, read-only preview, sorting/reordering, selection-only, and filter controls are excluded unless they discard data or trigger a write.
- Backend transactionality cannot be proved by a frontend-only audit. “Guarded/atomic” below means the UI supplies visible preview/version/idempotency/typed-confirmation evidence and blocks duplicate submission; backend enforcement still requires a separate API/database audit.

The action-level registers contain **232 audited action families/source instances**: 42 Create, 52 Edit, 21 Delete, 17 Bulk, 21 Publish, and 79 Guarded. Baseline-pattern counts are 2 `CA-S`, 144 `CA-L`, 4 `MD-S`, 8 `MD-L`, 55 `IN-L`, 4 `NC-L`, and 15 `LOCAL`.

Reproducible baseline source-count commands (run at `dcf08791`):

```sh
rg --files apps/web/app apps/web/components -g '*.tsx' | wc -l
rg -l --glob '*.tsx' '^"use client"|^'"'"'use client'"'"'' apps/web/app apps/web/components | wc -l
rg -o --glob '*.tsx' '<button\b' apps/web/app apps/web/components | wc -l
rg -o --glob '*.tsx' '<form\b' apps/web/app apps/web/components | wc -l
rg -o --glob '*.tsx' '<ConfirmAction\b' apps/web/app apps/web/components | wc -l
rg -l --glob '*.tsx' '<ConfirmAction\b' apps/web/app apps/web/components | wc -l
rg -n --glob '*.tsx' 'window\.confirm\(' apps/web/app apps/web/components
rg -o --glob '*.tsx' 'method:\s*"(POST|PUT|PATCH|DELETE)"' apps/web/app apps/web/components | wc -l
```

## Target standard used for grading

1. Create and Edit open a focused modal. The saved result returns to a compact read-only card with an explicit Edit action.
2. Destructive or consequential actions require an explicit confirmation with a human-readable impact preview.
3. The same dialog remains open through `Working…`, then shows persistent success until the user chooses `Done`/`OK`.
4. Errors remain in the dialog; entered data and reviewed choices are preserved for correction/retry.
5. Bulk changes expose only compatible shared fields, preserve mixed values unless selected, preview every affected record, and save atomically.
6. Duplicate submission is blocked in state and, for durable writes, supported by an operation/idempotency key or version/CAS evidence.
7. Dialogs have an accessible name/description, trap focus, support Escape where safe, and restore focus to the trigger. Inputs have programmatic labels and errors use `role="alert"`/appropriate live regions.
8. State and feedback are authoritative: no silent dismissal, stale success, or success inferred merely from a request ending.

## Behavior codes

| Code | Current behavior | Grade |
| --- | --- | --- |
| `CA-S` | Shared `ConfirmAction`; native modal focus handling, duplicate-submit ref guard, Working state, in-dialog thrown error, persistent success with OK. | Meets dialog lifecycle; still verify human-readable preview and server guard per action. |
| `CA-L` | Shared `ConfirmAction`; confirmation + Working + duplicate-submit guard, but callback returns `void`, so the dialog closes immediately. Most handlers catch and convert errors to page messages, which also makes the dialog close on failure. | Partial; systemic failure of success/error lifecycle. |
| `MD-S` | Purpose-built modal with Working and/or persistent result. | Partial; verify focus/escape/restore and error containment. |
| `MD-L` | Purpose-built role-dialog with local form and Working label, but closes after save; errors may be local. | Partial; no persistent success and no reusable focus contract. |
| `IN-L` | Inline editable form/card; pending state and page-level feedback, no focused modal and/or no confirmation. | Does not meet Create/Edit or consequential-action standard. |
| `NC-L` | Native `window.confirm()`. | Does not meet preview, persistent result, error, or accessible app-dialog standard. |
| `LOCAL` | Client-only draft mutation; consequence is deferred until a later save. | Must be read-only-card/modalized where it represents Create/Edit; destructive local removal still needs deliberate confirmation/undo. |

## Systemic findings

### Shared confirmation primitive

`apps/web/components/ConfirmAction.tsx` is a good foundation: native `<dialog>`, accessible title/description, `aria-busy`, initial cancel focus, focus restoration, Escape handling, disabled actions during work, and a synchronous `submittingRef` duplicate-submit guard.

The success contract is optional. `handleConfirm()` closes the dialog whenever `onConfirm` returns no `ConfirmActionSuccess`. Only two callbacks currently return that object:

- `apps/web/app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx` — Publish reviewed tournament (`CA-S`).
- `apps/web/app/admin/match-log/MatchLogApplyPanel.tsx` — Apply staged edits (`CA-S`).

Therefore 144 of 146 source instances are `CA-L`. An additional error-path defect is common: callbacks catch API errors, set page-level message state, and return normally. `ConfirmAction` interprets that as completion and closes instead of keeping the error in the dialog.

Required shared fix: make success presentation mandatory (required success descriptor/result), require callbacks to throw typed errors, and keep confirmation inputs/previews mounted until OK/Done. A migration helper should turn existing API result/page messages into `ConfirmActionSuccess` without bespoke dialogs.

### Create/Edit modal infrastructure

There is no shared Create/Edit dialog primitive. Tournament setup has six purpose-built role-dialogs; Match Uploader has two native dialogs; Player Editor has one custom success overlay. The role-dialog implementations do not trap focus, choose initial focus, handle Escape consistently, or restore focus. They do preserve form state on validation errors and disable submit during work.

Required shared fix: introduce an accessible `ActionDialog`/`EditDialog` foundation using native `<dialog>` or an equivalent tested focus manager, with `editing → working → success/error` states and a required human-readable result.

### Read-only-card pattern

The new Tournament Event and Division flows meet the strongest part of the intended pattern: Add/Edit open a focused-looking dialog and saving returns a compact, read-only card with explicit Edit and guarded Remove. Most other admin editors remain always-editable inline panels (players, league settings, commerce, registrations, match/social editors, team competition, and operations).

## Complete shared-confirmation inventory

Every `ConfirmAction` source instance is listed here. Unless marked `CA-S`, current behavior is `CA-L`: confirmation and Working exist, but success auto-dismisses and caught errors return to the page.

| ID | File | Count | Runtime action label(s) | Guard/preview notes | Result |
| --- | --- | ---: | --- | --- | --- |
| CA-001 | `apps/web/app/admin/badges/BadgeDiagnosticsPanel.tsx` | 3 | Update badge state; Run applying recompute; Revoke matched badge rows | Typed confirmation; applying/revoke are consequential; reviewed data shown on page. | `CA-L` |
| CA-002 | `apps/web/app/admin/challenge-ladder/ChallengeLadderAdminPanel.tsx` | 12 | Create challenge; Publish reviewed result; Record forfeit; Record monthly pass; Add to bottom; Move to tier bottom; Apply reviewed replacement; Save overrides; Start clock; Accept; Cancel; Reconcile durable operation | Several actions have previews/operation keys; Create and edits remain inline. | `CA-L` |
| CA-003 | `apps/web/app/admin/jupr-live/JuprLiveAdminPanel.tsx` | 8 | Create session; Save scores; Advance round; Publish official matches; Complete; Abandon; Archive; Reconcile stored response | Typed confirmations and retained recovery evidence; no persistent completion. | `CA-L` |
| CA-004 | `apps/web/app/admin/league-manager/GuidedLeagueSettingsEditor.tsx` | 1 | Save structured draft / Save description | Inline editor, preview available for schedule only. | `CA-L` |
| CA-005 | `apps/web/app/admin/league-manager/awards/LeagueAwardsPanel.tsx` | 3 | Freeze and save; Mint and verify; Archive completed league | Wizard/fingerprint/retry evidence is strong. | `CA-L` |
| CA-006 | `apps/web/app/admin/league-manager/create/LeagueCreatePanel.tsx` | 1 | Create league | Inline create form rather than focused modal. | `CA-L` |
| CA-007 | `apps/web/app/admin/league-manager/league/LeagueHomePanel.tsx` | 2 | Available lifecycle transition(s): activate/start/end/archive as supplied by API; Duplicate league | Consequence described, typed confirmation. | `CA-L` |
| CA-008 | `apps/web/app/admin/league-manager/live/LeagueLiveRoundPanel.tsx` | 6 | Create persisted session; Save session snapshot; Reconcile round; Verify round; Create guest; Publish complete league round | Version/recovery/round preview evidence present. | `CA-L` |
| CA-009 | `apps/web/app/admin/league-manager/roster/LeagueRosterPanel.tsx` | 1 | Add Player(s) / Remove Player(s) | Explicitly describes one atomic roster batch; selected rows visible. | `CA-L` |
| CA-010 | `apps/web/app/admin/league-manager/teams/TeamLeaguesPanel.tsx` | 6 | Save team league setup; Save result/forfeit; Reconcile from match; Pair/withdraw waitlist; Publish schedule/playoffs; Finalize/compensate recovery | Strong preview, versions, and recovery language. | `CA-L` |
| CA-011 | `apps/web/app/admin/match-canonical-audit/MatchCanonicalAuditPanel.tsx` | 1 | Apply exact reviewed plan | Exact dry-run plan precedes write. | `CA-L` |
| CA-012 | `apps/web/app/admin/match-log/MatchLogApplyPanel.tsx` | 4 | Apply staged edits; Complete mandatory replay; Mark group no issue; Soft-exclude duplicates | Staged human-readable patches/replay evidence. Apply staged edits returns persistent result; other three do not. | 1 `CA-S`, 3 `CA-L` |
| CA-013 | `apps/web/app/admin/match-log/MatchLogBulkExcludePanel.tsx` | 1 | Exclude selected matches | Selection and reason shown; guarded bulk action. | `CA-L` |
| CA-014 | `apps/web/app/admin/match-log/MatchLogExclusionRecoveryPanel.tsx` | 1 | Resume exact recovery / Continue exact operation | Exact operation retained. | `CA-L` |
| CA-015 | `apps/web/app/admin/match-log/MatchLogQuickReplayPanel.tsx` | 1 | Run Quick Replay | Scope summary shown. | `CA-L` |
| CA-016 | `apps/web/app/admin/match-log/MatchLogSocialPanel.tsx` | 1 | Delete selected Club Social row | Destructive scope described. | `CA-L` |
| CA-017 | `apps/web/app/admin/match-uploader/MatchUploaderForm.tsx` | 1 | Remove match | Local row removal; confirmation used only after row has data. | `CA-L` |
| CA-018 | `apps/web/app/admin/moneyball/MoneyballPanel.tsx` | 2 | Publish reviewed official matches; Reconcile stored response | Settlement preview and retained operation key. | `CA-L` |
| CA-019 | `apps/web/app/admin/player-updates/PlayerUpdatesPanel.tsx` | 6 | Queue digests; Replace atomically; Deactivate; Send selected pending; Reset selected to pending; Delete selected pending | Bulk selections visible; explicit atomic replacement. | `CA-L` |
| CA-020 | `apps/web/app/admin/player-updates/verified-requests/VerifiedRequestsPanel.tsx` | 3 | Approve; Reject; Unsubscribe | Request detail visible; no persistent result. | `CA-L` |
| CA-021 | `apps/web/app/admin/players/PlayerEditorPanel.tsx` | 6 | Save league rating; Auto-link exact names; Save social link; Execute atomic merge; Attach replay evidence; Compensate merge | Merge has preview/operation/recovery evidence; other edits are inline. | `CA-L` |
| CA-022 | `apps/web/app/admin/replay-history/ReplayHistoryForm.tsx` | 1 | Run replay | Scope and replay options shown. | `CA-L` |
| CA-023 | `apps/web/app/admin/support-requests/SupportRequestsPanel.tsx` | 1 | Save request status | Inline status/note editor. | `CA-L` |
| CA-024 | `apps/web/app/admin/tools/AdminToolsPanel.tsx` | 7 | Reconcile existing rows; Approve/Reject submission; Apply selected tournament matches; Run badge queue; Run badge recompute; Save role; Revoke role | Several have previews and typed confirmation; role form is inline. | `CA-L` |
| CA-025 | `apps/web/app/admin/tournament-live/TournamentLivePanel.tsx` | 7 | Reconcile operation; Save score; Generate games; Generate playoffs; Generate podium; Award podium; Publish official matches | Strong readiness/operation evidence and human descriptions. | `CA-L` |
| CA-026 | `apps/web/app/admin/tournament-setup/TournamentSetupAdvancedPanel.tsx` | 1 | Apply imported JSON | JSON is not a human-readable before/after preview. | `CA-L`; preview fails standard |
| CA-027 | `apps/web/app/admin/tournament-setup/TournamentSetupDayCard.tsx` | 1 | Remove day | Local draft destructive action. | `CA-L` |
| CA-028 | `apps/web/app/admin/tournament-setup/TournamentSetupDivisionCard.tsx` | 1 | Remove division | Local draft destructive action. | `CA-L` |
| CA-029 | `apps/web/app/admin/tournament-setup/TournamentSetupFamilyCard.tsx` | 1 | Remove event | Local draft destructive action. | `CA-L` |
| CA-030 | `apps/web/app/admin/tournament-setup/TournamentSetupPanel.tsx` | 6 | Create tournament; Save settings; Seed from published config; Generate standard divisions; Save draft; Publish setup | Legacy all-in-one inline editor; impact review only for publish. | `CA-L` |
| CA-031 | `apps/web/app/admin/tournaments/bulk/BulkRegistrationPanel.tsx` | 1 | Apply bulk update | Selected rows and patch fields visible; server atomicity not proved here. | `CA-L` |
| CA-032 | `apps/web/app/admin/tournaments/commerce/TournamentCommercePanel.tsx` | 3 | Save reviewed catalog; Cancel extras order; Save fulfillment | Catalog has review step; order/fulfillment target visible. | `CA-L` |
| CA-033 | `apps/web/app/admin/tournaments/create/TournamentCreatePanel.tsx` | 1 | Create tournament | Inline create form; no focused modal. | `CA-L` |
| CA-034 | `apps/web/app/admin/tournaments/delete-draft/DeleteDraftPanel.tsx` | 1 | Delete draft | Excellent destructive scope and dependency guard; no persistent result. | `CA-L` |
| CA-035 | `apps/web/app/admin/tournaments/editor/TournamentRegistrationEditorPanel.tsx` | 2 | Save registration; Save event entry | Always-editable inline controls, limited before/after preview. | `CA-L` |
| CA-036 | `apps/web/app/admin/tournaments/ops/TournamentOpsPanel.tsx` | 12 | Create draw; Import registrations; Import teams; Save teams; Generate games; Save score; Generate playoffs; Generate podium; Award podium; Commit reviewed results; Publish rating game; Publish official matches | Most operations have version/CAS/readiness or import preview evidence. | `CA-L` |
| CA-037 | `apps/web/app/admin/tournaments/registration/registrants/[registrationId]/TournamentRegistrantEditPanel.tsx` | 6 | Save registration; Add event entry; Save event entry; Save assigned partner; Remove event entry; Save reviewed extras | Complete inline editor; extras has preview, other before/after summaries are limited. | `CA-L` |
| CA-038 | `apps/web/app/admin/tournaments/setup/TournamentSetupDivisionCard.tsx` | 1 | Remove division | Compact read-only card with explicit Edit; local private-draft removal. | `CA-L`; card pattern passes |
| CA-039 | `apps/web/app/admin/tournaments/setup/TournamentSetupEventFamilyCard.tsx` | 1 | Remove event | Compact read-only card with Edit/Generate; local private-draft removal. | `CA-L`; card pattern passes |
| CA-040 | `apps/web/app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx` | 3 | Remove court; Publish reviewed tournament; Open registration | Publish has persistent success; remove/open auto-dismiss. Publish and Open are correctly separate. | 1 `CA-S`, 2 `CA-L` |
| CA-041 | `apps/web/app/admin/tournaments/status/TournamentStatusPanel.tsx` | 1 | Archive / Unarchive tournament | Reversible action, selected object and version visible. | `CA-L` |
| CA-042 | `apps/web/app/admin/tournaments/team-competition/TeamTournamentAdminPanel.tsx` | 14 | Save event rules; Verify rating; Save review; Finalize registration-close review; Create team/invitations; Reissue invitation; withdraw/cancel invitation; Build round robin; Build playoffs; Save score; Reconcile official match; Save podium draft; Publish podium/results; Lock/Update lineup | Strong domain guards but dense inline forms and no persistent completion. | `CA-L` |
| CA-043 | `apps/web/app/admin/weekly-recap/WeeklyRecapAdminPanel.tsx` | 4 | Generate/Regenerate draft; Save draft edits; Publish recap; Unpublish recap | Draft/public distinction is described. | `CA-L` |

Totals: 146 shared-confirmation source instances; two `CA-S`, 144 `CA-L`.

## Action-level shared-confirmation register

This ledger gives every shared-confirmation source instance a stable audit ID. `VOID` explicitly flags the missing success return. “Caught/page error” flags the contract hazard: if the callback catches instead of throwing, the modal closes and the error is rendered elsewhere.

| Action ID | Class | Source | Label | Pattern | Outcome/error behavior | Concrete gap |
| --- | --- | --- | --- | --- | --- | --- |
| A-CA-001 | Edit | `apps/web/app/admin/badges/BadgeDiagnosticsPanel.tsx:357` | Update badge state | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-002 | Guarded | `apps/web/app/admin/badges/BadgeDiagnosticsPanel.tsx:378` | Run applying recompute | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-003 | Delete | `apps/web/app/admin/badges/BadgeDiagnosticsPanel.tsx:394` | Revoke matched badge row(s) | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-004 | Create | `apps/web/app/admin/challenge-ladder/ChallengeLadderAdminPanel.tsx:444` | Create challenge | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-005 | Publish | `apps/web/app/admin/challenge-ladder/ChallengeLadderAdminPanel.tsx:482` | Publish reviewed result | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-006 | Guarded | `apps/web/app/admin/challenge-ladder/ChallengeLadderAdminPanel.tsx:511` | Record forfeit | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-007 | Guarded | `apps/web/app/admin/challenge-ladder/ChallengeLadderAdminPanel.tsx:530` | Record monthly pass | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-008 | Create | `apps/web/app/admin/challenge-ladder/ChallengeLadderAdminPanel.tsx:562` | Add to bottom | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-009 | Guarded | `apps/web/app/admin/challenge-ladder/ChallengeLadderAdminPanel.tsx:581` | Move to tier bottom | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-010 | Guarded | `apps/web/app/admin/challenge-ladder/ChallengeLadderAdminPanel.tsx:614` | Apply reviewed replacement | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-011 | Edit | `apps/web/app/admin/challenge-ladder/ChallengeLadderAdminPanel.tsx:638` | Save overrides | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-012 | Guarded | `apps/web/app/admin/challenge-ladder/ChallengeLadderAdminPanel.tsx:665` | Start clock | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-013 | Guarded | `apps/web/app/admin/challenge-ladder/ChallengeLadderAdminPanel.tsx:675` | Accept | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-014 | Guarded | `apps/web/app/admin/challenge-ladder/ChallengeLadderAdminPanel.tsx:685` | Cancel | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-015 | Guarded | `apps/web/app/admin/challenge-ladder/ChallengeLadderAdminPanel.tsx:703` | Reconcile durable operation | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-016 | Create | `apps/web/app/admin/jupr-live/JuprLiveAdminPanel.tsx:214` | Create session | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-017 | Edit | `apps/web/app/admin/jupr-live/JuprLiveAdminPanel.tsx:260` | Save scores | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-018 | Guarded | `apps/web/app/admin/jupr-live/JuprLiveAdminPanel.tsx:270` | Advance round | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-019 | Publish | `apps/web/app/admin/jupr-live/JuprLiveAdminPanel.tsx:281` | Publish official matches | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-020 | Guarded | `apps/web/app/admin/jupr-live/JuprLiveAdminPanel.tsx:296` | Complete | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-021 | Delete | `apps/web/app/admin/jupr-live/JuprLiveAdminPanel.tsx:306` | Abandon | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-022 | Guarded | `apps/web/app/admin/jupr-live/JuprLiveAdminPanel.tsx:317` | Archive | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-023 | Guarded | `apps/web/app/admin/jupr-live/JuprLiveAdminPanel.tsx:335` | Reconcile stored response | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-024 | Edit | `apps/web/app/admin/league-manager/GuidedLeagueSettingsEditor.tsx:342` | Save structured draft / Save description | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-025 | Edit | `apps/web/app/admin/league-manager/awards/LeagueAwardsPanel.tsx:518` | Freeze and save | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-026 | Guarded | `apps/web/app/admin/league-manager/awards/LeagueAwardsPanel.tsx:575` | Mint and verify / Retry mint and verification | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-027 | Guarded | `apps/web/app/admin/league-manager/awards/LeagueAwardsPanel.tsx:593` | Archive completed league | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-028 | Create | `apps/web/app/admin/league-manager/create/LeagueCreatePanel.tsx:144` | Create league | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-029 | Guarded | `apps/web/app/admin/league-manager/league/LeagueHomePanel.tsx:259` | League lifecycle transition | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-030 | Guarded | `apps/web/app/admin/league-manager/league/LeagueHomePanel.tsx:280` | Duplicate league | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-031 | Create | `apps/web/app/admin/league-manager/live/LeagueLiveRoundPanel.tsx:913` | Create persisted session | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-032 | Edit | `apps/web/app/admin/league-manager/live/LeagueLiveRoundPanel.tsx:970` | Save session snapshot | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-033 | Guarded | `apps/web/app/admin/league-manager/live/LeagueLiveRoundPanel.tsx:1013` | Reconcile round operation | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-034 | Guarded | `apps/web/app/admin/league-manager/live/LeagueLiveRoundPanel.tsx:1033` | Verify round operation | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-035 | Create | `apps/web/app/admin/league-manager/live/LeagueLiveRoundPanel.tsx:1096` | Create guest | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-036 | Publish | `apps/web/app/admin/league-manager/live/LeagueLiveRoundPanel.tsx:1133` | Publish complete league round | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-037 | Bulk | `apps/web/app/admin/league-manager/roster/LeagueRosterPanel.tsx:213` | Add / Remove selected players | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-038 | Edit | `apps/web/app/admin/league-manager/teams/TeamLeaguesPanel.tsx:206` | Save team league setup | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-039 | Edit | `apps/web/app/admin/league-manager/teams/TeamLeaguesPanel.tsx:306` | Save result / forfeit | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-040 | Guarded | `apps/web/app/admin/league-manager/teams/TeamLeaguesPanel.tsx:311` | Reconcile from match | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-041 | Bulk | `apps/web/app/admin/league-manager/teams/TeamLeaguesPanel.tsx:656` | Pair / withdraw waitlist entries | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-042 | Publish | `apps/web/app/admin/league-manager/teams/TeamLeaguesPanel.tsx:666` | Publish schedule / playoffs | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-043 | Guarded | `apps/web/app/admin/league-manager/teams/TeamLeaguesPanel.tsx:685` | Finalize / compensate recovery | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-044 | Guarded | `apps/web/app/admin/match-canonical-audit/MatchCanonicalAuditPanel.tsx:236` | Apply exact reviewed plan | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-045 | Guarded | `apps/web/app/admin/match-log/MatchLogApplyPanel.tsx:901` | Apply staged edits | `CA-S` | Persistent success/OK; thrown error remains in dialog | Lifecycle compliant; retain preview/guard tests |
| A-CA-046 | Guarded | `apps/web/app/admin/match-log/MatchLogApplyPanel.tsx:921` | Complete mandatory replay | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-047 | Guarded | `apps/web/app/admin/match-log/MatchLogApplyPanel.tsx:971` | Mark this group no issue | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-048 | Guarded | `apps/web/app/admin/match-log/MatchLogApplyPanel.tsx:1009` | Soft-exclude duplicate rows | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-049 | Bulk | `apps/web/app/admin/match-log/MatchLogBulkExcludePanel.tsx:258` | Exclude selected matches | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-050 | Guarded | `apps/web/app/admin/match-log/MatchLogExclusionRecoveryPanel.tsx:185` | recoveryRequired ? "Resume exact recovery" : "Continue exact operation" | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-051 | Guarded | `apps/web/app/admin/match-log/MatchLogQuickReplayPanel.tsx:145` | Run Quick Replay | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-052 | Bulk | `apps/web/app/admin/match-log/MatchLogSocialPanel.tsx:411` | Delete selected Club Social row | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-053 | Delete | `apps/web/app/admin/match-uploader/MatchUploaderForm.tsx:1381` | Remove match | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-054 | Publish | `apps/web/app/admin/moneyball/MoneyballPanel.tsx:104` | Publish reviewed official matches | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-055 | Guarded | `apps/web/app/admin/moneyball/MoneyballPanel.tsx:105` | Reconcile stored response | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-056 | Guarded | `apps/web/app/admin/player-updates/PlayerUpdatesPanel.tsx:285` | Queue digests | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-057 | Bulk | `apps/web/app/admin/player-updates/PlayerUpdatesPanel.tsx:296` | Replace atomically | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-058 | Delete | `apps/web/app/admin/player-updates/PlayerUpdatesPanel.tsx:297` | Deactivate | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-059 | Bulk | `apps/web/app/admin/player-updates/PlayerUpdatesPanel.tsx:312` | Send selected pending | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-060 | Bulk | `apps/web/app/admin/player-updates/PlayerUpdatesPanel.tsx:313` | Reset selected to pending | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-061 | Bulk | `apps/web/app/admin/player-updates/PlayerUpdatesPanel.tsx:314` | Delete selected pending | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-062 | Guarded | `apps/web/app/admin/player-updates/verified-requests/VerifiedRequestsPanel.tsx:134` | Approve | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-063 | Guarded | `apps/web/app/admin/player-updates/verified-requests/VerifiedRequestsPanel.tsx:144` | Reject | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-064 | Guarded | `apps/web/app/admin/player-updates/verified-requests/VerifiedRequestsPanel.tsx:155` | Unsubscribe | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-065 | Edit | `apps/web/app/admin/players/PlayerEditorPanel.tsx:376` | Save league rating | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-066 | Guarded | `apps/web/app/admin/players/PlayerEditorPanel.tsx:394` | Auto-link exact names | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-067 | Edit | `apps/web/app/admin/players/PlayerEditorPanel.tsx:413` | Save social link | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-068 | Guarded | `apps/web/app/admin/players/PlayerEditorPanel.tsx:449` | Execute atomic merge | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-069 | Guarded | `apps/web/app/admin/players/PlayerEditorPanel.tsx:473` | Attach replay evidence | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-070 | Guarded | `apps/web/app/admin/players/PlayerEditorPanel.tsx:488` | Compensate merge | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-071 | Guarded | `apps/web/app/admin/replay-history/ReplayHistoryForm.tsx:174` | Run replay | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-072 | Edit | `apps/web/app/admin/support-requests/SupportRequestsPanel.tsx:263` | Save request status | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-073 | Guarded | `apps/web/app/admin/tools/AdminToolsPanel.tsx:386` | Reconcile existing rows | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-074 | Bulk | `apps/web/app/admin/tools/AdminToolsPanel.tsx:433` | Approve / Reject selected submission | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-075 | Bulk | `apps/web/app/admin/tools/AdminToolsPanel.tsx:471` | Apply selected tournament matches | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-076 | Guarded | `apps/web/app/admin/tools/AdminToolsPanel.tsx:480` | Run badge queue | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-077 | Guarded | `apps/web/app/admin/tools/AdminToolsPanel.tsx:493` | Run badge recompute | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-078 | Edit | `apps/web/app/admin/tools/AdminToolsPanel.tsx:497` | Save role | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-079 | Delete | `apps/web/app/admin/tools/AdminToolsPanel.tsx:497` | Revoke role | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-080 | Guarded | `apps/web/app/admin/tournament-live/TournamentLivePanel.tsx:681` | Reconcile operation | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-081 | Edit | `apps/web/app/admin/tournament-live/TournamentLivePanel.tsx:712` | Save score | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-082 | Create | `apps/web/app/admin/tournament-live/TournamentLivePanel.tsx:724` | Generate games | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-083 | Create | `apps/web/app/admin/tournament-live/TournamentLivePanel.tsx:732` | Generate playoffs | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-084 | Create | `apps/web/app/admin/tournament-live/TournamentLivePanel.tsx:739` | Generate podium | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-085 | Guarded | `apps/web/app/admin/tournament-live/TournamentLivePanel.tsx:742` | Award podium | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-086 | Publish | `apps/web/app/admin/tournament-live/TournamentLivePanel.tsx:750` | Publish official matches | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-087 | Guarded | `apps/web/app/admin/tournament-setup/TournamentSetupAdvancedPanel.tsx:106` | Apply imported JSON | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-088 | Delete | `apps/web/app/admin/tournament-setup/TournamentSetupDayCard.tsx:83` | Remove day | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-089 | Delete | `apps/web/app/admin/tournament-setup/TournamentSetupDivisionCard.tsx:120` | Remove division | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-090 | Delete | `apps/web/app/admin/tournament-setup/TournamentSetupFamilyCard.tsx:79` | Remove event | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-091 | Create | `apps/web/app/admin/tournament-setup/TournamentSetupPanel.tsx:425` | Create tournament | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-092 | Edit | `apps/web/app/admin/tournament-setup/TournamentSetupPanel.tsx:452` | Save settings | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-093 | Publish | `apps/web/app/admin/tournament-setup/TournamentSetupPanel.tsx:459` | Seed from published config | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-094 | Create | `apps/web/app/admin/tournament-setup/TournamentSetupPanel.tsx:469` | Generate standard divisions | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-095 | Edit | `apps/web/app/admin/tournament-setup/TournamentSetupPanel.tsx:491` | Save draft | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-096 | Publish | `apps/web/app/admin/tournament-setup/TournamentSetupPanel.tsx:506` | Publish setup | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-097 | Bulk | `apps/web/app/admin/tournaments/bulk/BulkRegistrationPanel.tsx:230` | Apply bulk update | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-098 | Edit | `apps/web/app/admin/tournaments/commerce/TournamentCommercePanel.tsx:2018` | Save reviewed catalog | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-099 | Guarded | `apps/web/app/admin/tournaments/commerce/TournamentCommercePanel.tsx:2181` | Cancel extras order | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-100 | Edit | `apps/web/app/admin/tournaments/commerce/TournamentCommercePanel.tsx:2330` | Save fulfillment | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-101 | Create | `apps/web/app/admin/tournaments/create/TournamentCreatePanel.tsx:187` | Create tournament | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-102 | Delete | `apps/web/app/admin/tournaments/delete-draft/DeleteDraftPanel.tsx:133` | Delete draft | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-103 | Edit | `apps/web/app/admin/tournaments/editor/TournamentRegistrationEditorPanel.tsx:252` | busy ? "Saving…" : "Save registration" | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-104 | Edit | `apps/web/app/admin/tournaments/editor/TournamentRegistrationEditorPanel.tsx:267` | busy ? "Saving…" : "Save event entry" | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-105 | Create | `apps/web/app/admin/tournaments/ops/TournamentOpsPanel.tsx:678` | Create draw | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-106 | Bulk | `apps/web/app/admin/tournaments/ops/TournamentOpsPanel.tsx:719` | Import registrations | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-107 | Bulk | `apps/web/app/admin/tournaments/ops/TournamentOpsPanel.tsx:729` | Import teams | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-108 | Bulk | `apps/web/app/admin/tournaments/ops/TournamentOpsPanel.tsx:740` | Save teams | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-109 | Create | `apps/web/app/admin/tournaments/ops/TournamentOpsPanel.tsx:759` | Generate games | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-110 | Edit | `apps/web/app/admin/tournaments/ops/TournamentOpsPanel.tsx:760` | Save score | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-111 | Create | `apps/web/app/admin/tournaments/ops/TournamentOpsPanel.tsx:761` | Generate playoffs | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-112 | Create | `apps/web/app/admin/tournaments/ops/TournamentOpsPanel.tsx:762` | Generate podium | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-113 | Guarded | `apps/web/app/admin/tournaments/ops/TournamentOpsPanel.tsx:763` | Award podium | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-114 | Guarded | `apps/web/app/admin/tournaments/ops/TournamentOpsPanel.tsx:832` | Commit reviewed results | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-115 | Publish | `apps/web/app/admin/tournaments/ops/TournamentOpsPanel.tsx:863` | Publish rating game | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-116 | Publish | `apps/web/app/admin/tournaments/ops/TournamentOpsPanel.tsx:884` | Publish official matches | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-117 | Edit | `apps/web/app/admin/tournaments/registration/registrants/[registrationId]/TournamentRegistrantEditPanel.tsx:952` | Save registration | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-118 | Guarded | `apps/web/app/admin/tournaments/registration/registrants/[registrationId]/TournamentRegistrantEditPanel.tsx:974` | Add event entry | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-119 | Edit | `apps/web/app/admin/tournaments/registration/registrants/[registrationId]/TournamentRegistrantEditPanel.tsx:1155` | Save event entry | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-120 | Edit | `apps/web/app/admin/tournaments/registration/registrants/[registrationId]/TournamentRegistrantEditPanel.tsx:1164` | Save assigned partner | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-121 | Delete | `apps/web/app/admin/tournaments/registration/registrants/[registrationId]/TournamentRegistrantEditPanel.tsx:1179` | Remove event entry | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-122 | Edit | `apps/web/app/admin/tournaments/registration/registrants/[registrationId]/TournamentRegistrantEditPanel.tsx:1324` | Save reviewed extras | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-123 | Delete | `apps/web/app/admin/tournaments/setup/TournamentSetupDivisionCard.tsx:92` | Remove division | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-124 | Delete | `apps/web/app/admin/tournaments/setup/TournamentSetupEventFamilyCard.tsx:83` | Remove event | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-125 | Delete | `apps/web/app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx:2986` | Remove court | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-126 | Publish | `apps/web/app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx:3682` | Publish reviewed tournament | `CA-S` | Persistent success/OK; thrown error remains in dialog | Lifecycle compliant; retain preview/guard tests |
| A-CA-127 | Publish | `apps/web/app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx:3703` | Open registration | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-128 | Guarded | `apps/web/app/admin/tournaments/status/TournamentStatusPanel.tsx:163` | Archive / Unarchive tournament | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-129 | Edit | `apps/web/app/admin/tournaments/team-competition/TeamTournamentAdminPanel.tsx:633` | Save event rules | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-130 | Guarded | `apps/web/app/admin/tournaments/team-competition/TeamTournamentAdminPanel.tsx:753` | Verify rating | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-131 | Edit | `apps/web/app/admin/tournaments/team-competition/TeamTournamentAdminPanel.tsx:879` | Save review | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-132 | Guarded | `apps/web/app/admin/tournaments/team-competition/TeamTournamentAdminPanel.tsx:921` | Finalize all at registration close | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-133 | Create | `apps/web/app/admin/tournaments/team-competition/TeamTournamentAdminPanel.tsx:1063` | Create team and send invitations | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-134 | Guarded | `apps/web/app/admin/tournaments/team-competition/TeamTournamentAdminPanel.tsx:1145` | Reissue invitation | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-135 | Delete | `apps/web/app/admin/tournaments/team-competition/TeamTournamentAdminPanel.tsx:1246` | Withdraw team / Replace roster | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-136 | Create | `apps/web/app/admin/tournaments/team-competition/TeamTournamentAdminPanel.tsx:1386` | Build round robin | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-137 | Create | `apps/web/app/admin/tournaments/team-competition/TeamTournamentAdminPanel.tsx:1419` | Build playoffs | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-138 | Edit | `apps/web/app/admin/tournaments/team-competition/TeamTournamentAdminPanel.tsx:1604` | Save score | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-139 | Guarded | `apps/web/app/admin/tournaments/team-competition/TeamTournamentAdminPanel.tsx:1691` | Reconcile official match | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-140 | Edit | `apps/web/app/admin/tournaments/team-competition/TeamTournamentAdminPanel.tsx:1805` | Save podium draft | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-141 | Publish | `apps/web/app/admin/tournaments/team-competition/TeamTournamentAdminPanel.tsx:1830` | Publish podium and results | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-142 | Edit | `apps/web/app/admin/tournaments/team-competition/TeamTournamentAdminPanel.tsx:2049` | Lock / Update lineup | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-143 | Guarded | `apps/web/app/admin/weekly-recap/WeeklyRecapAdminPanel.tsx:346` | Generate / Regenerate draft | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-144 | Edit | `apps/web/app/admin/weekly-recap/WeeklyRecapAdminPanel.tsx:413` | Save draft edits | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-145 | Publish | `apps/web/app/admin/weekly-recap/WeeklyRecapAdminPanel.tsx:439` | Publish recap | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |
| A-CA-146 | Publish | `apps/web/app/admin/weekly-recap/WeeklyRecapAdminPanel.tsx:440` | Unpublish recap | `CA-L` | **VOID**: auto-close; thrown error stays, caught/page error closes | Return success descriptor; rethrow typed error into dialog |

Register count: **146** (`A-CA-001` through `A-CA-146`).


## Non-shared consequential-action inventory

These actions do not use `ConfirmAction`. The table includes direct writes and destructive client-draft changes. Ordinary read-only previews are named only when they are part of the write flow.

| ID | Surface and file(s) | Action labels / consequences | Current progress, success, error, and input behavior | Main deviations |
| --- | --- | --- | --- | --- |
| NS-001 | Admin auth: `apps/web/app/admin/login/AdminLoginForm.tsx`, `apps/web/components/AdminShell.tsx`, `apps/web/app/admin/AdminOperationsCockpit.tsx` | Sign in; Send magic link; Sign out | Busy disables controls; inline status/error; auth forms preserved on error. | Appropriate inline auth exception, but success is navigation/page feedback rather than acknowledged result. |
| NS-002 | Password recovery: `apps/web/app/admin/reset-password/AdminResetPasswordForm.tsx` | Send/resend recovery email; Update password | Busy guard and inline messages; fields remain on error. | Consequential credential update has no final acknowledged success dialog. |
| NS-003 | Player create/edit: `apps/web/app/admin/players/PlayerEditorPanel.tsx` | Add player; Save player | Always-editable inline panels. Busy guard; page error. Save player alone opens a persistent custom success overlay; Create reports page message. | Create/Edit not modal/read-only-card based. Create lacks persistent success. Save-result overlay lacks focus/Escape/restore management. |
| NS-004 | Match Uploader: `apps/web/app/admin/match-uploader/MatchUploaderForm.tsx` | Create player; Create and add player; Submit singles match; Submit batch; Submit scored round-robin games | Inline editors with busy labels and errors. Successful match writes open a persistent native-dialog result with OK and human-readable counts/ratings. | Submission result is strong (`MD-S`), but Create Player is inline with page feedback. No pre-submit confirm for official rated writes. |
| NS-005 | Match Uploader local removal: same file | Remove All; Keep rows with data; Remove blank/all; Remove unfilled row | Native modal for Remove All; per-row ConfirmAction only when row contains data; blank row removal is immediate. | Remove-All dialog has no result state/focus restoration; terminology mixes local draft removal with durable delete. |
| NS-006 | Club Social editor: `apps/web/app/admin/match-log/MatchLogSocialPanel.tsx` | Save Club Social row | Inline editable fields, Working label, page success/error, inputs retained. Delete uses CA-016. | Edit not modal/read-only; no before/after confirmation; no persistent result. |
| NS-007 | League Awards wizard: `apps/web/app/admin/league-manager/awards/LeagueAwardsPanel.tsx` | Save award category choices; Compute/Recompute and save preview; Confirm winners/reasons; Confirm empty preview | Inline wizard with busy guard, persisted revision/fingerprint and page message. | Consequential saves bypass confirmation dialog and persistent success. |
| NS-008 | League Live editor: `apps/web/app/admin/league-manager/live/LeagueLiveRoundPanel.tsx` | Add/remove court rows; Generate match slots; roster suggestion; movement preview | Add/remove are local session-draft edits; final snapshot/publish are guarded by CA-008. | Local destructive remove has no confirm/undo; editor stays inline rather than read-only card + Edit. |
| NS-009 | Admin play generators: `apps/web/app/admin/play-generators/GeneratorWorkspace.tsx`, `GeneratorRoundRunner.tsx`, `GeneratorStandings.tsx` | Start session; Save round scores / Round played; Skip round; Advance/finish; Save substitution/roster change; Publish official matches | Busy guards and page feedback. Skip-discard and publish use two native `window.confirm()` calls. | Four total native confirms across admin/public (`NC-L`); no persistent result; no structured impact preview; errors/page state outside dialog. |
| NS-010 | Public play generators: `apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorWorkspace.tsx`, `PublicGeneratorRoundRunner.tsx`, `PublicGeneratorStandings.tsx` | Start session; Save/mark/skip/advance/finish round; Save roster change | Same inline behavior; two native confirms for skip-discard/publish code path. | Same as NS-009; public official-publish code path should be explicitly role/permission gated and standardized. |
| NS-011 | Public live creator: `apps/web/app/clubs/[clubSlug]/live/PublicLiveCreator.tsx` | Create event | Inline setup, submitting guard, page error/success/navigation; operation key prevents repeat creation. | Create not focused modal/read-only; no persistent acknowledged result. |
| NS-012 | Public live runner: `apps/web/app/clubs/[clubSlug]/live/[sessionKey]/LiveSessionRunner.tsx` | Save scores; Advance; Complete (two entry points); Save/retry substitution | Busy guard and durable operation keys; inline feedback. | Consequential complete/advance/substitution lack human-readable confirm and persistent result. |
| NS-013 | Rated score entry: `apps/web/app/clubs/[clubSlug]/admin/score-entry/ScoreEntryForm.tsx` and `apps/web/app/admin/clubs/[clubId]/score-entry/page.tsx` | Save rated match | Inline fields, Saving state, page feedback. | Official rated write lacks preview/confirmation/persistent result. |
| NS-014 | Tournament Event dialog: `apps/web/app/admin/tournaments/setup/TournamentSetupEventFamilyDialog.tsx` | Add event; Save event | Focused-looking `role=dialog`; local validation errors and draft preserved; submit disabled; closes to compact read-only Event card with Edit. | Strong product pattern, but no focus trap/initial focus/Escape/restore and no persistent success. `MD-L`. |
| NS-015 | Tournament Division dialog: `apps/web/app/admin/tournaments/setup/TournamentSetupDivisionDialog.tsx` | Add division; Save division | Same as NS-014; closes to compact read-only Division card. | Same accessibility/result gaps. `MD-L`. |
| NS-016 | Tournament division presets: `apps/web/app/admin/tournaments/setup/TournamentDivisionPresetDialog.tsx` | Save selected generated divisions | Custom modal, selected-row preview, busy guard, inline errors. | No shared focus contract or persistent result; atomicity depends on parent draft save. `MD-L`. |
| NS-017 | Tournament bulk division edit: `apps/web/app/admin/tournaments/setup/TournamentDivisionBulkEditDialog.tsx` | Save selected divisions together | Best bulk implementation: compatible fields only, “Multiple values,” explicit opt-in per field, per-division preview, validation, one parent save, busy guard. | No focus/escape/restore and no persistent result. Backend atomicity not proved. `MD-L`. |
| NS-018 | Tournament bulk courts: `apps/web/app/admin/tournaments/setup/TournamentBulkAddCourtsDialog.tsx` | Add N courts | Custom modal, resulting-count and names preview, validation, busy guard. | Closes immediately to inline court editors; no persistent result/focus contract. `MD-L`. |
| NS-019 | Tournament age policy: `apps/web/app/admin/tournaments/setup/TournamentAgePolicyEditor.tsx` | Add age group; Remove age group | Inline nested controls inside Event/Division dialogs; removal is immediate local draft mutation. | Needs reversible removal or focused confirmation; programmatic labels are incomplete for repeated controls. |
| NS-020 | Tournament wizard private-draft saves: `apps/web/app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx` | Save basics and continue; save schedule/venue and continue; save fees draft and continue; save Review actions/resolutions | Inline page buttons, Working text, page-level feedback. Event/division dialog saves also call draft persistence before closing. | No persistent result; errors can appear away from trigger; save semantics vary between immediate dialog saves and domain footer saves. |
| NS-021 | Tournament wizard local lists: same file | Add/remove sponsor; Add single court; Add/edit/remove event/division through child dialogs/cards; Keep published value; Force change with registration resolution | Sponsors/courts are editable inline; remove sponsor is immediate; court removal uses CA-L; conflict resolutions are inline and later saved. | Sponsor/court records do not use read-only card + focused Edit consistently; remove sponsor lacks confirm/undo; resolution actions need a durable result. |
| NS-022 | Legacy tournament builder: `apps/web/app/admin/tournament-setup/TournamentSetupBuilder.tsx` plus Day/Family/Division cards | Add day; Add event; Add division; reorder; edit inline; remove | Always-editable inline builder; removals use CA-L cards; final writes use CA-030. | Entire legacy create/edit model diverges from the newer focused-dialog/read-only-card standard. Decide deprecation vs migration. |
| NS-023 | Tournament Commerce catalog draft: `apps/web/app/admin/tournaments/commerce/TournamentCommercePanel.tsx` | Add/remove extra; add/remove option; add presets; add/remove bundle/component; add/remove giveaway/promotion | Dense inline mutable catalog; removals are immediate local draft changes. Final reviewed save uses CA-032. | Create/Edit not focused/read-only; local destructive controls lack confirm/undo; preview is separate from edit surface. |
| NS-024 | Tournament Commerce payment: same file | Update payment status | Direct inline button/API write with busy guard and page result. | Consequential financial status change lacks confirmation, before/after preview, and persistent result. |
| NS-025 | Tournament Ops team draft: `apps/web/app/admin/tournaments/ops/TournamentOpsPanel.tsx` | Add/remove team editor rows; Reset from snapshot | Local inline draft before CA-036 Save teams. | Remove/reset can discard edits without confirm/undo; mutable rows are not compact cards. |
| NS-026 | Tournament registration: `apps/web/app/clubs/[clubSlug]/tournament-registration/TournamentRegistrationForm.tsx` | Start new / Edit existing wizard; Submit registration; Retry team setup | Multi-step inline wizard includes a review step; pending guard; errors preserve state; success routes to confirmation. | Create/Edit not modal, but review/confirmation page is a reasonable public-flow exception. Ensure final confirmation page remains explicit and retry-safe. |
| NS-027 | Public registration edit: `apps/web/app/clubs/[clubSlug]/tournament-registration/edit/EditTournamentRegistrationForm.tsx` | Save registration changes | Inline form, pending guard, error retained, success message/navigation. | No before/after confirmation or persistent dialog result. |
| NS-028 | Registration edit-link request: `apps/web/app/clubs/[clubSlug]/tournament-registration/EditLinkRequestForm.tsx` | Send secure edit link | Inline form, pending guard and feedback. | Low-risk request; modal not essential, but success should remain visible and announced. |
| NS-029 | Four-player recovery: `apps/web/app/clubs/[clubSlug]/tournament-registration/confirmation/FourPlayerTeamSetupRecovery.tsx` | Submit/retry team setup | Inline action with operation evidence and page feedback. | Consequential team creation/invitations lack confirm and persistent result. |
| NS-030 | Partner Board interest: `apps/web/app/clubs/[clubSlug]/tournament-partner-board/PairingInterestPanel.tsx` | Send interest | Inline per-card action; pending guard; button changes after sent. | No confirmation/result dialog; consequence should be explained before sending. |
| NS-031 | Partner request review: `apps/web/app/clubs/[clubSlug]/tournament-partner-board/PartnerRequestReviewPanel.tsx` | Accept; Decline; Cancel | Manual two-click inline confirmation, pending guard, page feedback. | Not a modal, no persistent result, and error is outside the confirmation state. |
| NS-032 | Team invitation: `apps/web/app/clubs/[clubSlug]/tournament-team-invitation/TeamInvitationReview.tsx` | Accept invitation; Decline invitation | Inline pending guard and page feedback. | No confirmation/persistent result; decline is consequential. |
| NS-033 | Team-league partner response: `apps/web/app/clubs/[clubSlug]/team-league-partner-confirmation/PartnerConfirmationPanel.tsx` | Accept; Decline | Inline pending guard and page feedback. | Same as NS-032. |
| NS-034 | Team-league registration: `apps/web/app/clubs/[clubSlug]/team-leagues/[leagueName]/TeamLeagueRegistrationForm.tsx` | Register team; Join partner waitlist | Inline form, busy guard, page feedback. | Create flow not modal/reviewed; no acknowledged success dialog. |
| NS-035 | Email preferences: `apps/web/app/email-preferences/EmailPreferencesPanel.tsx` | Unsubscribe / expand to global opt-out | Inline busy guard and persistent page state. | Global opt-out should show explicit scope confirmation and result. |
| NS-036 | Public request forms: `apps/web/app/support/SupportRequestForm.tsx`, `data-corrections/DataCorrectionForm.tsx`, `profile-privacy/ProfilePrivacyRequestForm.tsx`, `verified-updates/VerifiedUpdatesRequestForm.tsx` | Submit support, correction, privacy, and verified-update requests | Native forms with pending guard and inline success/error; inputs retained on error. | Low-risk submissions can remain page forms, but success must be durable/announced; privacy request needs especially clear scope. |

## Action-level non-shared register

This register splits the grouped `NS-*` findings into individually countable action families. Repeated row instances share one ID when they execute the same handler and contract; distinct consequences receive distinct IDs.

| Action ID | Class | Source | Action | Pattern | Outcome/error behavior | Concrete gap |
| --- | --- | --- | --- | --- | --- | --- |
| A-NS-001 | Guarded | `apps/web/app/admin/login/AdminLoginForm.tsx` | Sign in | `IN-L` | Busy guard; inline error/success or navigation | Document auth exception; announce durable outcome |
| A-NS-002 | Guarded | `apps/web/app/admin/login/AdminLoginForm.tsx` | Send magic link | `IN-L` | Busy guard; inline feedback | Keep persistent announced success |
| A-NS-003 | Guarded | `apps/web/components/AdminShell.tsx; apps/web/app/admin/AdminOperationsCockpit.tsx` | Sign out | `IN-L` | Busy guard; navigation/session state | Document auth exception |
| A-NS-004 | Guarded | `apps/web/app/admin/reset-password/AdminResetPasswordForm.tsx` | Send / resend recovery email | `IN-L` | Busy guard; inline feedback; input retained | Persistent announced success |
| A-NS-005 | Edit | `apps/web/app/admin/reset-password/AdminResetPasswordForm.tsx` | Update password | `IN-L` | Busy guard; inline feedback | Add acknowledged success for credential change |
| A-NS-006 | Create | `apps/web/app/admin/players/PlayerEditorPanel.tsx` | Add player | `IN-L` | Busy guard; page message | Focused Create dialog; persistent result |
| A-NS-007 | Edit | `apps/web/app/admin/players/PlayerEditorPanel.tsx` | Save player | `MD-S` | Busy guard; page error; custom success/OK overlay | Move edit into shared dialog; fix focus/Escape/restore |
| A-NS-008 | Create | `apps/web/app/admin/match-uploader/MatchUploaderForm.tsx` | Create player | `IN-L` | Busy guard; inline error/result | Focused Create dialog; persistent result |
| A-NS-009 | Create | `apps/web/app/admin/match-uploader/MatchUploaderForm.tsx` | Create and add player | `IN-L` | Busy guard; inline error/result | Focused Create dialog; persistent result |
| A-NS-010 | Publish | `apps/web/app/admin/match-uploader/MatchUploaderForm.tsx` | Submit singles rated match | `MD-S` | Busy guard; page error; persistent result dialog/OK | Add pre-submit human preview/confirm |
| A-NS-011 | Publish | `apps/web/app/admin/match-uploader/MatchUploaderForm.tsx` | Submit rated match batch | `MD-S` | Busy guard; row errors; persistent result dialog/OK | Add reviewed batch confirmation |
| A-NS-012 | Publish | `apps/web/app/admin/match-uploader/MatchUploaderForm.tsx` | Submit scored round-robin games | `MD-S` | Busy guard; page error; persistent result dialog/OK | Add reviewed games confirmation |
| A-NS-013 | Delete | `apps/web/app/admin/match-uploader/MatchUploaderForm.tsx` | Remove all entered rows | `MD-L` | Native confirmation modal; immediate local result | Add focus restoration and explicit resulting row count/undo |
| A-NS-014 | Delete | `apps/web/app/admin/match-uploader/MatchUploaderForm.tsx` | Remove blank row | `LOCAL` | Immediate local mutation | Provide undo or consistent draft-removal rule |
| A-NS-015 | Edit | `apps/web/app/admin/match-log/MatchLogSocialPanel.tsx` | Save Club Social row | `IN-L` | Working label; page success/error; values retained | Focused Edit with before/after and persistent result |
| A-NS-016 | Edit | `apps/web/app/admin/league-manager/awards/LeagueAwardsPanel.tsx` | Save award category choices | `IN-L` | Busy guard; page feedback/revision | Persistent result; consider focused editor |
| A-NS-017 | Guarded | `apps/web/app/admin/league-manager/awards/LeagueAwardsPanel.tsx` | Compute / recompute and save award preview | `IN-L` | Busy guard; persisted fingerprint/page feedback | Confirm replacement; persistent result |
| A-NS-018 | Edit | `apps/web/app/admin/league-manager/awards/LeagueAwardsPanel.tsx` | Confirm winners and reasons / save revisions | `IN-L` | Busy guard; page feedback; draft retained | Human before/after confirm; persistent result |
| A-NS-019 | Edit | `apps/web/app/admin/league-manager/awards/LeagueAwardsPanel.tsx` | Confirm empty preview | `IN-L` | Busy guard; page feedback | Explicit consequence dialog; persistent result |
| A-NS-020 | Delete | `apps/web/app/admin/league-manager/live/LeagueLiveRoundPanel.tsx` | Remove court row from session draft | `LOCAL` | Immediate local mutation | Confirm/undo populated row removal |
| A-NS-021 | Create | `apps/web/app/admin/play-generators/GeneratorWorkspace.tsx` | Start generator session | `IN-L` | Busy guard; page error/navigation | Focused review/confirmation and persistent created state |
| A-NS-022 | Edit | `apps/web/app/admin/play-generators/GeneratorRoundRunner.tsx` | Save round scores | `IN-L` | Busy guard; page feedback | Persistent result or explicit saved-state card |
| A-NS-023 | Edit | `apps/web/app/admin/play-generators/GeneratorRoundRunner.tsx` | Mark round played | `IN-L` | Busy guard; page feedback | Persistent result or explicit saved-state card |
| A-NS-024 | Guarded | `apps/web/app/admin/play-generators/GeneratorRoundRunner.tsx` | Skip round / discard draft scores | `NC-L` | Native confirm only when draft exists; page feedback | Replace native confirm; preview discarded scores; persistent result |
| A-NS-025 | Guarded | `apps/web/app/admin/play-generators/GeneratorRoundRunner.tsx; GeneratorStandings.tsx` | Advance / finish session | `IN-L` | Busy guard; page feedback/navigation | Confirm final completion; persistent result |
| A-NS-026 | Edit | `apps/web/app/admin/play-generators/GeneratorRoundRunner.tsx` | Save roster substitution/removal | `IN-L` | Busy guard; page feedback | Before/after confirm; persistent result |
| A-NS-027 | Publish | `apps/web/app/admin/play-generators/GeneratorRoundRunner.tsx` | Publish official rated matches | `NC-L` | Native confirm; busy/page feedback | Shared confirm with reviewed count and persistent result |
| A-NS-028 | Create | `apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorWorkspace.tsx` | Start public generator session | `IN-L` | Busy guard; page error/navigation | Document full-page public exception; persistent created state |
| A-NS-029 | Edit | `apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner.tsx` | Save public round scores | `IN-L` | Busy guard; page feedback | Persistent saved state |
| A-NS-030 | Edit | `apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner.tsx` | Mark public round played | `IN-L` | Busy guard; page feedback | Persistent saved state |
| A-NS-031 | Guarded | `apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner.tsx` | Skip public round / discard draft scores | `NC-L` | Native confirm; page feedback | Replace native confirm; preview discarded data |
| A-NS-032 | Guarded | `apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner.tsx; PublicGeneratorStandings.tsx` | Advance / finish public session | `IN-L` | Busy guard; page feedback/navigation | Confirm final completion; persistent result |
| A-NS-033 | Edit | `apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner.tsx` | Save public roster substitution/removal | `IN-L` | Busy guard; page feedback | Before/after confirm; persistent result |
| A-NS-034 | Publish | `apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner.tsx` | Publish official matches (dead/unrendered function) | `NC-L` | Native confirm exists in unreachable static function | Delete dead code or render only behind explicit admin role |
| A-NS-035 | Create | `apps/web/app/clubs/[clubSlug]/live/PublicLiveCreator.tsx` | Create public live event | `IN-L` | Operation key + busy guard; page error/navigation | Focused/full-page review; persistent created result |
| A-NS-036 | Edit | `apps/web/app/clubs/[clubSlug]/live/[sessionKey]/LiveSessionRunner.tsx` | Save live scores | `IN-L` | Busy guard; operation state/page feedback | Persistent result |
| A-NS-037 | Guarded | `apps/web/app/clubs/[clubSlug]/live/[sessionKey]/LiveSessionRunner.tsx` | Advance live round | `IN-L` | Busy guard; operation state/page feedback | Human confirmation and persistent result |
| A-NS-038 | Guarded | `apps/web/app/clubs/[clubSlug]/live/[sessionKey]/LiveSessionRunner.tsx` | Complete live session (two triggers) | `IN-L` | Busy guard; page feedback | Confirm terminal consequence; persistent result |
| A-NS-039 | Edit | `apps/web/app/clubs/[clubSlug]/live/[sessionKey]/LiveSessionRunner.tsx` | Save / retry substitution | `IN-L` | Retained operation key; page feedback | Before/after dialog and persistent result |
| A-NS-040 | Publish | `apps/web/app/clubs/[clubSlug]/admin/score-entry/ScoreEntryForm.tsx` | Save rated match | `IN-L` | Saving guard; page feedback | Review/confirm official match; persistent result |
| A-NS-041 | Create | `apps/web/app/admin/tournaments/setup/TournamentSetupEventFamilyDialog.tsx` | Add Event | `MD-L` | Dialog validation + Working; closes to read-only card | Add shared focus contract and persistent success |
| A-NS-042 | Edit | `apps/web/app/admin/tournaments/setup/TournamentSetupEventFamilyDialog.tsx` | Edit Event | `MD-L` | Dialog validation + Working; closes to read-only card | Add shared focus contract and persistent success |
| A-NS-043 | Create | `apps/web/app/admin/tournaments/setup/TournamentSetupDivisionDialog.tsx` | Add Division | `MD-L` | Dialog validation + Working; closes to read-only card | Add shared focus contract and persistent success |
| A-NS-044 | Edit | `apps/web/app/admin/tournaments/setup/TournamentSetupDivisionDialog.tsx` | Edit Division | `MD-L` | Dialog validation + Working; closes to read-only card | Add shared focus contract and persistent success |
| A-NS-045 | Bulk | `apps/web/app/admin/tournaments/setup/TournamentDivisionPresetDialog.tsx` | Save generated Division presets | `MD-L` | Selected-row preview + Working; inline error | Persistent success; shared focus/escape/restore |
| A-NS-046 | Bulk | `apps/web/app/admin/tournaments/setup/TournamentDivisionBulkEditDialog.tsx` | Save Divisions together | `MD-L` | Compatible fields + mixed values + per-row preview + Working | Persistent success; prove atomic API boundary; shared focus |
| A-NS-047 | Bulk | `apps/web/app/admin/tournaments/setup/TournamentBulkAddCourtsDialog.tsx` | Bulk add venue courts | `MD-L` | Resulting-count/name preview + Working | Persistent success; shared focus/escape/restore |
| A-NS-048 | Create | `apps/web/app/admin/tournaments/setup/TournamentAgePolicyEditor.tsx` | Add age group | `LOCAL` | Immediate nested draft mutation | Create through focused subdialog or explicit inline exception |
| A-NS-049 | Delete | `apps/web/app/admin/tournaments/setup/TournamentAgePolicyEditor.tsx` | Remove age group | `LOCAL` | Immediate nested draft mutation | Confirm/undo; stable repeated-field labels |
| A-NS-050 | Edit | `apps/web/app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx` | Save Basics draft and continue | `IN-L` | Working label; page feedback | Persistent result at action point |
| A-NS-051 | Edit | `apps/web/app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx` | Save schedule/venue draft and continue | `IN-L` | Working label; page feedback | Persistent result at action point |
| A-NS-052 | Edit | `apps/web/app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx` | Save fees draft and continue | `IN-L` | Working label; page feedback | Persistent result at action point |
| A-NS-053 | Edit | `apps/web/app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx` | Save Review actions/resolutions | `IN-L` | Working label; page feedback; choices retained | Persistent result and exact affected-registration count |
| A-NS-054 | Create | `apps/web/app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx` | Add sponsor | `LOCAL` | Immediate inline draft row | Focused Add/Edit or documented simple-row exception |
| A-NS-055 | Delete | `apps/web/app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx` | Remove sponsor | `LOCAL` | Immediate local removal | Confirm/undo |
| A-NS-056 | Create | `apps/web/app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx` | Add single court | `LOCAL` | Immediate inline draft row | Focused Add/Edit or read-only court card |
| A-NS-057 | Guarded | `apps/web/app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx` | Keep published value for blocked change | `IN-L` | Inline state mutation; later Review save | Persistent saved resolution state |
| A-NS-058 | Guarded | `apps/web/app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx` | Force change with registration resolution | `IN-L` | Inline plan editor; later Review save | Focused resolution dialog and persistent saved result |
| A-NS-059 | Create | `apps/web/app/admin/tournament-setup/TournamentSetupBuilder.tsx` | Add day | `LOCAL` | Immediate inline legacy-draft row | Migrate to focused dialog/read-only card or retire legacy flow |
| A-NS-060 | Create | `apps/web/app/admin/tournament-setup/TournamentSetupBuilder.tsx` | Add event | `LOCAL` | Immediate inline legacy-draft row | Migrate or retire |
| A-NS-061 | Create | `apps/web/app/admin/tournament-setup/TournamentSetupBuilder.tsx` | Add division | `LOCAL` | Immediate inline legacy-draft row | Migrate or retire |
| A-NS-062 | Create | `apps/web/app/admin/tournaments/commerce/TournamentCommercePanel.tsx` | Add extra / option / bundle / giveaway | `LOCAL` | Immediate always-editable catalog draft rows | Focused Add dialogs; return read-only cards |
| A-NS-063 | Delete | `apps/web/app/admin/tournaments/commerce/TournamentCommercePanel.tsx` | Remove extra / option / bundle / component / promotion | `LOCAL` | Immediate local removal | Confirm/undo and show scope |
| A-NS-064 | Edit | `apps/web/app/admin/tournaments/commerce/TournamentCommercePanel.tsx` | Update payment status | `IN-L` | Busy guard; page feedback | Financial before/after confirm; persistent result |
| A-NS-065 | Create | `apps/web/app/admin/tournaments/ops/TournamentOpsPanel.tsx` | Add team draft row | `LOCAL` | Immediate mutable row | Focused Add/Edit or read-only team card |
| A-NS-066 | Delete | `apps/web/app/admin/tournaments/ops/TournamentOpsPanel.tsx` | Remove team draft row | `LOCAL` | Immediate local removal | Confirm/undo |
| A-NS-067 | Guarded | `apps/web/app/admin/tournaments/ops/TournamentOpsPanel.tsx` | Reset teams from snapshot | `LOCAL` | Discards local edits immediately | Confirm exact discarded changes |
| A-NS-068 | Create | `apps/web/app/clubs/[clubSlug]/tournament-registration/TournamentRegistrationForm.tsx` | Submit tournament registration | `IN-L` | Multi-step review; pending guard; error preserves input; success page | Document full-page public exception; keep explicit confirmation page |
| A-NS-069 | Edit | `apps/web/app/clubs/[clubSlug]/tournament-registration/edit/EditTournamentRegistrationForm.tsx` | Save registration changes | `IN-L` | Pending guard; inline error/success | Before/after review and persistent result |
| A-NS-070 | Guarded | `apps/web/app/clubs/[clubSlug]/tournament-registration/EditLinkRequestForm.tsx` | Send secure edit link | `IN-L` | Pending guard; inline feedback | Persistent announced success |
| A-NS-071 | Create | `apps/web/app/clubs/[clubSlug]/tournament-registration/confirmation/FourPlayerTeamSetupRecovery.tsx` | Submit / retry four-player team setup | `IN-L` | Retained operation evidence; page feedback | Confirm invitations/team effect; persistent result |
| A-NS-072 | Guarded | `apps/web/app/clubs/[clubSlug]/tournament-partner-board/PairingInterestPanel.tsx` | Send partner interest | `IN-L` | Per-card pending/sent state | Explain consequence before send; persistent result |
| A-NS-073 | Guarded | `apps/web/app/clubs/[clubSlug]/tournament-partner-board/PartnerRequestReviewPanel.tsx` | Accept partner request | `IN-L` | Manual inline two-step; pending/page feedback | Shared confirmation and persistent result |
| A-NS-074 | Guarded | `apps/web/app/clubs/[clubSlug]/tournament-partner-board/PartnerRequestReviewPanel.tsx` | Decline partner request | `IN-L` | Manual inline two-step; pending/page feedback | Shared confirmation and persistent result |
| A-NS-075 | Guarded | `apps/web/app/clubs/[clubSlug]/tournament-partner-board/PartnerRequestReviewPanel.tsx` | Cancel partner request | `IN-L` | Manual inline two-step; pending/page feedback | Shared confirmation and persistent result |
| A-NS-076 | Guarded | `apps/web/app/clubs/[clubSlug]/tournament-team-invitation/TeamInvitationReview.tsx` | Accept team invitation | `IN-L` | Pending guard; page feedback | Shared confirmation and persistent result |
| A-NS-077 | Guarded | `apps/web/app/clubs/[clubSlug]/tournament-team-invitation/TeamInvitationReview.tsx` | Decline team invitation | `IN-L` | Pending guard; page feedback | Shared confirmation and persistent result |
| A-NS-078 | Guarded | `apps/web/app/clubs/[clubSlug]/team-league-partner-confirmation/PartnerConfirmationPanel.tsx` | Accept team-league partner | `IN-L` | Pending guard; page feedback | Shared confirmation and persistent result |
| A-NS-079 | Guarded | `apps/web/app/clubs/[clubSlug]/team-league-partner-confirmation/PartnerConfirmationPanel.tsx` | Decline team-league partner | `IN-L` | Pending guard; page feedback | Shared confirmation and persistent result |
| A-NS-080 | Create | `apps/web/app/clubs/[clubSlug]/team-leagues/[leagueName]/TeamLeagueRegistrationForm.tsx` | Register team | `IN-L` | Busy guard; page feedback | Review step and persistent result |
| A-NS-081 | Create | `apps/web/app/clubs/[clubSlug]/team-leagues/[leagueName]/TeamLeagueRegistrationForm.tsx` | Join partner waitlist | `IN-L` | Busy guard; page feedback | Review step and persistent result |
| A-NS-082 | Guarded | `apps/web/app/email-preferences/EmailPreferencesPanel.tsx` | Unsubscribe / expand global opt-out | `IN-L` | Busy guard; persistent page state | Confirm human-readable scope; acknowledged result |
| A-NS-083 | Create | `apps/web/app/support/SupportRequestForm.tsx` | Submit support request | `IN-L` | Pending guard; input retained on error; inline result | Document low-risk full-page exception; announce success |
| A-NS-084 | Create | `apps/web/app/data-corrections/DataCorrectionForm.tsx` | Submit correction request | `IN-L` | Pending guard; input retained on error; inline result | Document low-risk full-page exception; announce success |
| A-NS-085 | Guarded | `apps/web/app/profile-privacy/ProfilePrivacyRequestForm.tsx` | Submit privacy request | `IN-L` | Pending guard; input retained on error; inline result | Clarify scope/consequence; persistent acknowledgement |
| A-NS-086 | Create | `apps/web/app/verified-updates/VerifiedUpdatesRequestForm.tsx` | Submit verified-update request | `IN-L` | Pending guard; input retained on error; inline result | Document low-risk full-page exception; announce success |

Register count: **86** (`A-NS-001` through `A-NS-086`).


## Custom-dialog accessibility inventory

There are nine dialog primitives in product source including the shared `ConfirmAction`; eight are purpose-built dialogs/overlays.

| File / dialog | Native modal | Name | Focus trap | Initial focus | Escape | Restore trigger focus | Persistent result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `components/ConfirmAction.tsx` | Yes | Yes | Native | Cancel/OK | Yes unless busy | Yes | Only when callback returns result (2/146) |
| `admin/match-uploader/MatchUploaderForm.tsx` — Remove All | Yes | Yes | Native | Browser default | Yes | Not explicit | No |
| Same — Submission Result | Yes | Yes | Native | Browser default | Yes | Not explicit | Yes, OK |
| `admin/players/PlayerEditorPanel.tsx` — save result | No (`div role=dialog`) | Yes | No | No | No | No | Yes, OK |
| `TournamentSetupEventFamilyDialog.tsx` | No (`section role=dialog`) | Yes | No | No | No | No | No |
| `TournamentSetupDivisionDialog.tsx` | No | Yes | No | No | No | No | No |
| `TournamentDivisionPresetDialog.tsx` | No | Yes | No | No | No | No | No |
| `TournamentDivisionBulkEditDialog.tsx` | No | Yes | No | No | No | No | No |
| `TournamentBulkAddCourtsDialog.tsx` | No | Yes | No | No | No | No | No |

Inputs are usually wrapped in `<label>`, and most validation errors use `role="alert"`. Repeated inline editors are the main labeling risk: sponsor/court/catalog/team rows rely heavily on visual grouping and should receive stable `fieldset`/`legend`, `aria-describedby`, and per-row error IDs during migration.

## Guarded/atomic write findings

Strong frontend evidence exists in Match Log staged edits/replay, player merge, league live publishing/recovery, team-league schedule/recovery, tournament operations, Tournament Live, Moneyball, Tournament Review/publish, and bulk division edit. These surfaces use combinations of reviewed previews, fingerprints, revisions/CAS values, operation IDs, idempotency keys, or exact retained requests.

Weak or absent frontend guard evidence exists for raw player create/edit, Club Social save, rated score entry, league-award intermediate saves, payment-status update, public live progression/completion, partner/invitation responses, and public/team registrations. These may still be guarded server-side, but the UI does not consistently expose that fact or retain an exact retry state.

Duplicate submit protection is broadly present via `busy`/`pending` disabled controls. `ConfirmAction` additionally uses a synchronous ref guard. The custom Tournament dialogs use component state only; because React state updates are asynchronous, the shared modal primitive should provide the same synchronous submit lock to every dialog.

## Prioritized remediation plan

### P0 — shared lifecycle (unblocks most of the app)

1. Make `ConfirmAction` success mandatory and persistent. Add a reusable success formatter for action title, affected object/count, authoritative state, and recovery/audit link where present.
2. Require rejected promises/typed thrown errors; do not swallow an API error and return `void`. Keep errors and reviewed inputs inside the open dialog.
3. Add a shared `ActionDialog`/`EditDialog` with native modal behavior, initial focus, Escape rules, trigger-focus restoration, synchronous submit lock, `aria-describedby`, and `editing/working/success/error` states.
4. Replace all four `window.confirm()` calls.

### P1 — focused Create/Edit migrations

1. Migrate Player create/edit, League create/settings, Club Social edit, Tournament registration admin editors, Commerce items/bundles/giveaways, Team Competition records, and Tournament Ops teams to read-only cards plus explicit Add/Edit dialogs.
2. Promote the Tournament Event/Division card/dialog pattern into shared components, then move the six tournament-specific dialogs onto the shared focus/result lifecycle.
3. Decide whether the legacy `admin/tournament-setup` builder is retired or brought to parity; maintaining two interaction models will reintroduce drift.

### P2 — destructive/local draft consistency

1. Add confirmation or undo for sponsor, age-bracket, commerce item/option/bundle/component/promotion, team-row, live-court, and populated Match Uploader row removal.
2. Make “local draft only” explicit in confirmation text and reserve “Delete” for durable deletion.
3. Ensure Reset/Discard actions preview exactly which unsaved fields will be lost.

### P3 — public/consequential flows

1. Standardize rated match submission, public live Complete/Advance, registrations, team/partner invitations, and global unsubscribe around human-readable review and persistent result.
2. Preserve the public registration wizard/confirmation-page exception where a full-page workflow is more usable than a modal, but apply the same Working/error/result contract.

## Implemented remediation — play generators and Club Social editor

- `A-NS-024` and `A-NS-031` now use shared `ConfirmAction` dialogs with the exact unsaved-score count, skip reason, persistent authoritative success, and retained exact-request recovery on an uncertain response.
- `A-NS-027` now previews the exact unpublished saved-match count and effective match date, sends `PUBLISH MATCHES`, remains open through Working and acknowledged success, and retains the publish idempotency key for exact-request recovery when the outcome is uncertain.
- `A-NS-034` is closed by deleting the unreachable public official-publish state and handler. Public generator sessions continue to state that official publishing is an administrative action.
- `A-NS-015` now uses shared `ConfirmAction` with `SAVE SOCIAL MATCH`, a stable idempotency key, and `expected_current` values for every patched field. Known validation/conflict failures stay in the dialog; uncertain post-send outcomes retain and retry the unchanged request.
- `A-CA-052` now returns an authoritative persistent success result for Club Social deletion and rethrows known failures into the open confirmation dialog.
- Admin play-generator completion remains an automatic `/advance` outcome; there is no `/complete` UI call to migrate or label with `COMPLETE SESSION`.

## Acceptance criteria for closing all remediation findings

These are release criteria, not claims that the current candidate has closed every row. See [remediation-report.md](remediation-report.md) for the implemented, automated-ready, manual-ready, and formally accepted distinctions.

- Source has zero `window.confirm()` calls.
- Every `ConfirmAction` callback returns a persistent success descriptor or throws an in-dialog error; no guarded action closes on `void`.
- All Create/Edit surfaces are either focused dialogs returning to read-only cards or documented full-page workflow exceptions.
- Every destructive action has explicit scope and either confirmation or reversible undo.
- Every bulk action shows compatible fields, mixed-value preservation, per-record preview, and one atomic request/transaction boundary.
- Every modal passes keyboard focus entry, Tab containment, Escape policy, and trigger-focus restoration tests.
- Every write has synchronous duplicate-submit protection; durable retry paths retain operation/version evidence.
- Component tests cover working, success, error, retry, and double-click behavior; browser tests cover focus and one representative flow in each domain.
