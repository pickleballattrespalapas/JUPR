# App-wide interaction remediation report

Report date: 2026-08-13
Baseline revision: `dcf08791`
Candidate status: implementation working tree; not deployed and not formally accepted

## Outcome

The app now has one normative interaction standard and complete, stable-ID inventories for the audited frontend and backend surfaces. The candidate implements a shared confirmation/form lifecycle, replaces native and one-off modal paths, requires explicit completion results from shared confirmations, preserves success above consumer lifetimes, hardens the highest-priority backend gaps, and includes one forward-only Supabase trigger-function migration that corrects the tournament-team text/UUID comparison.

This is not a claim that all audited rows fully conform. The audit is complete; remediation is broad but intentionally records justified exceptions and residual work. In particular, transport-level uncertainty is not yet normalized across every durable client action, several application-ledgered writes still require an integrated SQL RPC for transaction-level atomicity, and `BE-184` still needs approved throttling/provider-reconciliation infrastructure.

Production, staging deployment state, runtime flags, DNS, and remote Supabase data were not changed by this implementation.

## Audited scope

| Inventory | Exact audited count | Coverage |
| --- | ---: | --- |
| Frontend action families/source instances | **232** | 42 Create, 52 Edit, 21 Delete, 17 Bulk Edit, 21 Publish, and 79 Guarded actions |
| Backend unsafe-method FastAPI contracts | **198** | 159 `POST`, 30 `PATCH`, 8 `PUT`, and 1 `DELETE`; 167 admin-prefixed and 31 public contracts |
| Total audited frontend + backend contracts | **430** | 232 frontend rows + 198 backend unsafe-method contracts |
| Staging write-wave coverage | **198 / 198** | 0 missing and 0 stale manifest entries; 12 additional named-wave assignments across 10 multi-wave contracts |

The frontend rows are recorded in [frontend-interaction-inventory.md](frontend-interaction-inventory.md). The backend routes, guards, durability families, and stable IDs `BE-001` through `BE-198` are recorded in [backend-guarded-action-inventory.md](backend-guarded-action-inventory.md). Existing rows `BE-001`–`BE-192` remain unchanged; five authenticated proof-only reconciliation routes are appended as `BE-193`–`BE-197`, followed by the server-retained League Live retry as `BE-198`. The governing requirements are in [interaction-standard.md](../interaction-standard.md); the implementation sequence and test model are in [implementation-plan.md](implementation-plan.md).

The counts describe audited action contracts, not unique buttons rendered at runtime. Repeated rows that invoke the same handler and contract share an action ID. Preview, plan, quote, and resolve endpoints remain in the backend inventory because the write boundary deliberately fails closed by unsafe HTTP method and some previews persist review or audit evidence.

## What is implemented

### Shared frontend lifecycle

- `ActionCompletion` is now a strict `ActionSuccess | ActionUncertain` contract. A shared confirmation callback cannot use `void` as success; a result that cannot prove completion is rejected by the lifecycle instead of silently closing.
- `InteractionDialog`, `FormDialog`, `ActionFeedback`, `ChangeReview`, `useActionLifecycle`, and typed `InteractionActionError` provide the Ready → Working → Success/Error/Uncertain lifecycle defined by the standard.
- `InteractionProvider` is mounted above the application route tree and owns an immutable active-action snapshot. Its result remains visible when the originating row or component is removed by the authoritative update.
- The provider admits only one active confirmation, preserves the synchronous in-flight lock, and restores focus in this order: eligible explicit result target, still-connected eligible/focusable trigger, then `<main>`.
- Shared success stays visible until **Done/OK**. Working state is non-dismissible, errors remain in the dialog, and an explicit uncertain result retains its operation key and recovery callback.
- Tournament Event, Division, Division Preset, Division Bulk Edit, and Bulk Add Courts forms now use the shared form-dialog shell. Match Uploader and Player Editor result/confirmation overlays use the shared modal primitive.
- Product source contains no `window.confirm`, `window.alert`, or `window.prompt`. The only product-owned native `<dialog>` primitive is `InteractionDialog`.
- The interaction harness demonstrates the regression-critical case in which a successful destructive action removes its own trigger before returning; the provider still owns and displays the success result.

The strict result contract is a real compile-time safeguard, but it does not by itself prove that every caller has correctly classified a network interruption after send. That residual is tracked below.

### Migrated interaction families

- Existing `ConfirmAction` consumers were migrated to explicit persistent completion results or typed failure/uncertainty paths instead of `void` completion.
- Admin and public play-generator skip/publish flows use human-readable shared confirmation. Admin official publication sends `PUBLISH MATCHES`; session completion sends `COMPLETE SESSION`. The unreachable public official-publish control was removed rather than presenting authority the public route does not have.
- Club Social edit uses `SAVE SOCIAL MATCH`, browser-loaded `expected_current` values, a stable idempotency key, and explicit stale/uncertain handling. Club Social deletion returns an authoritative persistent result.
- Tournament publication retains the reviewed publication action as distinct from opening registration and keeps its result visible for acknowledgement.
- Contained tournament Create/Edit/Bulk forms use a focused dialog while their saved representation remains a read-only card or row.
- The Samuel Patel age-placement correction remains a product rule in the standard: an unpaired player is provisionally grouped from their own age; partner-dependent placement is recalculated when a partner exists instead of blocking an unrelated setup publication.

### Backend guards implemented in this candidate

The following targeted application guards use existing private ledgers and guarded services. The candidate adds no new tables or columns, but it does include a forward-only migration that replaces the existing tournament-team eligibility trigger function with the type-correct comparison. No remote database mutation was made while preparing this candidate.

| IDs | Implemented application guarantee | Remaining limitation |
| --- | --- | --- |
| `BE-030`, `BE-198` | The exact round-publish request and idempotency key are retained before Match Uploader runs. Authenticated `RETRY LEAGUE ROUND` reuses that server-retained operation only for zero-evidence retry states and verifies deterministic contexts before writing; evidence-bearing states continue through Reconcile. | No new schema residual; the existing atomic direct-match RPC and League Live publish ledger remain authoritative |
| `BE-026`, `BE-196` | Stable create-session idempotency, intent before mutation, exact replay, changed-payload rejection, recovery-required state, authenticated inspection, and proof-only `RECONCILE LIVE SESSION` | Session/courts/audit/receipt do not commit in one SQL transaction |
| `BE-050`, `BE-195` | `SAVE SOCIAL MATCH`, field CAS, stable key, exact replay, structured stale/uncertain `409`, authenticated inspection, and proof-only `RECONCILE SOCIAL MATCH` | Domain update and completion audit are not one SQL transaction; rollback plus recovery ledger remains the fallback |
| `BE-059`, `BE-193` | `CREATE PLAYERS`, reviewed canonical fingerprint, stable key, one multi-row player insert, verified counts/readback, exact replay, and proof-only `RECONCILE PLAYER BATCH` | Insert is atomic, but intent/domain audit/ledger completion are separate transactions |
| `BE-073`, `BE-074` | Backend validation of `COMPLETE SESSION` and `PUBLISH MATCHES`; phrase participates in durable request fingerprint | No schema residual for these two actions |
| `BE-081`, `BE-082`, `BE-083`, `BE-194` | Create idempotency; canonical state fingerprints; edit/rating CAS; exact replay; retained recovery; authenticated inspection; `SAVE LEAGUE RATING`; proof-only `RECONCILE PLAYER OPERATION` | Player/rating row, domain audit, and ledger completion are separate transactions |
| `BE-111`, `BE-113` | `SAVE PAYMENT STATUS` and `SAVE FULFILLMENT STATUS` included in the existing transactional commerce request fingerprint | No schema residual for these two actions |
| `BE-118`, `BE-197` | Browser UUID and exact reviewed registration-import request are retained before send; structured uncertainty blocks all guarded Ops writes in that club tab; authenticated proof-only `RECONCILE REGISTRATION IMPORT` uses a stored result or an exact-request recovery tombstone and never repeats the import. The tombstone serializes an absent lookup against a late original request, while a competing original `intent` remains uncertain. | The atomic team-write RPC and private guarded-operation/audit ledger are separate transactions; a normal empty `recovery_required` row remains locked until commit-fence or manual evidence exists |
| `BE-184` | Stable edit-link request key, recipient/tournament/registration dedupe, durable delivery intent/receipt, 15-minute provider-uncertain resend suppression, and non-enumerating public response | Client-IP throttling, longer-lived exact-key replay, and provider receipt reconciliation are deferred |

At the audited boundary there are no P0 route bypasses. Production writes remain fail-closed unless separately enabled, and service-role credentials remain server-only.

## Justified interaction exceptions

An exception means the surface differs from the preferred focused-dialog presentation; it does not waive lifecycle, accessibility, or integrity requirements.

| Exception | Justification | Required behavior retained or still required |
| --- | --- | --- |
| Authentication and account recovery pages | Sign-in, magic link, password reset, and permission recovery are page-level security journeys | Persistent announced feedback, preserved safe inputs, explicit navigation/session result |
| Public registration and complete-registration editors | Long, multi-section identity, eligibility, partner, extras, payment, and consent workflows need full-page context | Reviewed final action, duplicate guard, retained inputs on failure, authoritative confirmation page or acknowledged result |
| Support, correction, privacy, and verified-update intake | Low-risk request forms are clearer as dedicated pages; privacy scope remains consequential | Persistent `status`/`alert` feedback, clear scope, anti-abuse controls, no silent completion |
| Live-operation workspaces | Scores, rounds, courts, and recovery evidence must remain visible together | Final guarded action still uses the shared confirmation/result lifecycle; routine score editing can remain inline |
| Local private-draft row changes | Some Add/Remove/Reset controls only mutate an unpublished in-memory draft and are committed by a later guarded Save | Clearly label draft-only state; populated/destructive removal still needs confirmation or undo; no public-success claim before save |
| Generator actions `BE-067`–`BE-072` | Expected version, idempotency, durable ledger, and reconciliation already protect routine Create/Edit/progression operations | Keep as routine exceptions unless product reclassifies a step as destructive/public/terminal; `BE-073`/`BE-074` are not exceptions |
| Public token/uniqueness-protected writes | Partner requests and some intake actions use scoped tokens, atomic uniqueness, and business-identity dedupe rather than a redundant typed phrase | Preserve the token, unique index/RPC, idempotent response, and rate-limit tests |
| Read-only previews on unsafe methods | The fail-closed method inventory intentionally includes them; some write review evidence | Do not add theatrical confirmation to a proven read-only preview, and do not remove it from boundary coverage |

## Explicit residual and deferred work

### 1. Transport uncertainty and reconciliation

The shared lifecycle can represent `uncertain`, and targeted generator, Club Social, guarded-operation, and communication flows retain an exact operation identity. Five appended authenticated `POST` contracts make proof-only recovery explicit: Match Uploader `BE-193`, Player Editor `BE-194`, Club Social `BE-195`, League Live `BE-196`, and Tournament Ops registration import `BE-197`. Each requires its action-specific reconciliation phrase and none reruns the original mutation. League Live `BE-198` is intentionally different: after a conclusive zero-write failure or reload, it repeats only the exact server-retained `BE-030` request and original key; it refuses evidence-bearing or completed states. `BE-197` retains the full reviewed browser request and reserves an exact-request tombstone before an absent lookup is considered non-applied, preventing a still-in-flight original request from arriving later and mutating. It closes completion only from a stored result and closes non-application only from that winning tombstone; a normal empty `recovery_required` row stays locked because a momentarily unchanged readback cannot fence a late PostgREST RPC commit. Ambiguous, incomplete, or still-`intent` evidence remains recovery-required.

Coverage is still not universal across all durable clients. Many legacy API helpers expose an ordinary thrown error without proving whether it occurred before or after the request was sent. The lifecycle currently treats an unknown exception as `failed`; therefore a caller may offer retry where the server may actually have committed. Full conformance requires a transport-aware, machine-readable error envelope with `kind`, stable `code`, `operation_key`, and an exact inspect/reconcile action across every durable family. Until that work is complete, do not claim universal prevention of blind post-timeout retries.

### 2. SQL transaction atomicity

Application-level idempotency, CAS, intent ledgers, readback, and recovery are implemented for the prioritized gaps, but integrated database transactions remain deferred for:

- `BE-026` / `BE-196`: live session, courts, audit, and operation completion;
- `BE-050` / `BE-195`: Club Social domain update, audit, and operation completion;
- `BE-059` / `BE-193`: player batch insert, audit, and operation completion;
- `BE-081`–`BE-083` / `BE-194`: player/rating mutation, audit, and operation completion.
- `BE-118` / `BE-197`: atomic registration-team write, guarded-operation result, and completion/recovery audit.

Each requires a separately reviewed Supabase migration/RPC. The current candidate must be described as **application guard implemented with atomic-RPC residual**, not fully transaction-atomic.

### 3. `BE-184` abuse control and provider evidence

Duplicate delivery is mitigated within the 15-minute recipient/tournament/registration bucket, including suppression after an uncertain provider result. Client-IP throttling and longer-lived exact-key replay need an approved distributed store or schema, and exact provider receipt reconciliation needs an approved provider contract. Provider acceptance and local ledger completion cannot currently be one transaction.

### 4. Other backend alignment findings

The backend inventory retains the following follow-up groups rather than presenting them as completed:

- `BE-034`/`BE-035`: league create/duplicate needs replayable creation receipt and atomic audit;
- `BE-045`/`BE-047`: league settings/roster should send browser-loaded version plus stable replay identity;
- `BE-157`: verified-request administration needs CAS and a replay receipt;
- `BE-158`–`BE-160`: weekly recap generation/save/publication needs stable caller keys;
- `BE-186`/`BE-187`: registration create/edit needs a top-level replayable receipt;
- `BE-176`/`BE-192`: document existing business-identity replay semantics in the API response contract;
- `BE-180`–`BE-183`: preserve the existing token, atomic RPC, uniqueness, and notification protections; a new phrase is not required.

### 5. Remaining frontend row-level alignment

The 232-row audit is complete, but row-level conformance is not universally closed. The inventory remains authoritative for:

- dedicated or inline Create/Edit pages that still need the acknowledged-result lifecycle or a narrower documented exception;
- local draft removals/resets that need an explicit draft-only explanation, confirmation, or reversible undo;
- public invitation, partner, registration, global-unsubscribe, and rated-match flows that need a readable consequence review and persistent result;
- bulk families that have not yet proved mixed-value preservation, a fresh reviewed fingerprint, a single atomic commit, and exact updated/unchanged/skipped/failed counts;
- domain-specific authoritative readback and stale-version tests not yet represented by the shared foundation tests.

### 6. Accessibility and release verification

The shared primitive implements native modal behavior, accessible names/descriptions, a 44px minimum action target, busy/error/success announcements, dirty-form discard handling, and explicit focus restoration. Formal acceptance still requires manual keyboard traversal, screen-reader announcements, 320 CSS-pixel width, 200% zoom, reduced motion, and representative domain flows on the exact staged commit.

## Verification evidence and limits

| Evidence | Recorded result | What it proves |
| --- | --- | --- |
| Unsafe-route AST/manifest comparison | 198 registered / 198 manifest; 0 missing; 0 stale; 12 additional assignments across 10 multi-wave contracts | Complete fail-closed staging boundary inventory |
| Staging boundary tests | 20 passed / 1 deprecation warning | Exact route/write-wave set equality, overlap classification, and staging boundary behavior |
| Combined interaction-inventory, registration-import recovery static contracts, and staging-boundary rerun | 31 passed / 1 deprecation warning | Covers the 232/198/430 count assertions, stable `BE-193`–`BE-198` mappings, retained-request and uncertainty source guards, and exact route/write-wave equality |
| Registration-import recovery contract | 57 focused backend/API/generic-guard/static tests passed / 3 framework warnings | Covers exact retained request validation, authentication, structured uncertainty, reference mismatch with zero writes, tombstone-wins/late-original refusal, same-UUID insert races, definite unrelated stale locks, original-intent-wins uncertainty, stored-result completion, audited proven-no-write closure, empty-recovery refusal, and idempotent reconciliation |
| Focused backend remediation suite | 146 passed / 3 deprecation warnings | Phrase, CAS, replay, duplicate suppression, durable intent, authorization, Unicode parity, and proof-only recovery/ambiguity behavior for targeted families |
| Static interaction guard | Product source scan: zero native browser prompts; only shared product modal primitive | Removal of native/one-off modal escape hatches in the candidate tree |
| Shared foundation contract | Source/component contract plus provider-unmount harness present | Strict completion, provider ownership, duplicate lock, persistent result, focus fallback, and uncertain-state API are testable |
| Web TypeScript and component checks | `tsc --noEmit` passed; 8 / 8 component contracts passed | The migrated interaction consumers satisfy the strict completion types, guarded-recovery contracts, and existing component contracts |
| Optimized Next.js production build | Passed; 92 / 92 pages generated | The candidate compiles, type-checks, and produces the complete web application bundle; existing hook warnings remain non-blocking |
| Local provider browser specification | 4 Playwright scenarios discovered; equivalent temporary jsdom harness passed its original 3 / 3 cases | Unmount persistence, the single-open lock, focus-target precedence, and connected-disabled-trigger fallback are encoded; a real browser run remains pending because no browser binary was available in this sandbox |
| Broad Python suite, final frozen candidate | 2835 passed / 9 failed / 40 warnings | All nine failures belong to the exact baseline's ten-failure set; one baseline failure now passes and no new interaction-standard regression remains |
| Exact baseline comparison (`dcf08791`) | 2756 passed / 10 failed / 40 warnings | Establishes the pre-existing failure set; one baseline failure now passes in the candidate, and no new failure was introduced |
| Newly introduced failure triage | 9 interaction-related static failures corrected; focused rerun 9 / 9 passed | The migrated strict-return, request-guard, exact-retry, persistent-result, and provider source contracts introduced no remaining Python-suite regression |

The current checks are necessary but not sufficient. Static tests do not replace a real browser execution, and focused backend tests do not prove the deferred cross-table SQL atomicity. The broad Python result establishes **zero new interaction-standard regressions relative to the baseline**, but the suite is not globally green because nine baseline failures remain. TypeScript, component, and production-build checks pass in this candidate; real-browser lifecycle execution and a manual staging pass remain required for formal acceptance, and the baseline failures must be resolved or explicitly waived through the normal release process.

## Acceptance status

The terms below are deliberately separate:

- **Implemented**: source/documentation exists in the candidate working tree.
- **Automated-ready**: an automated check or focused test exists and the recorded evidence is sufficient to run it against an exact commit; it does not imply the complete suite is green.
- **Manual-ready**: the behavior and test instructions are usable for staging validation after the exact candidate passes build/CI and is deployed to staging.
- **Formally accepted**: the exact commit has passed required automated and manual staging acceptance and an authorized reviewer has recorded acceptance.

| Scope | Implemented | Automated-ready | Manual-ready | Formally accepted |
| --- | --- | --- | --- | --- |
| Normative standard and 232/198 inventories (430 total contracts) | Yes | Yes — exact-count/static checks | Yes — review artifacts available | **No** |
| Shared confirmation/form lifecycle and root-owned persistent result | Yes | Yes — foundation/static contract and regression harness | Conditional — final build/CI and staging deployment still required | **No** |
| Native prompt and one-off modal removal | Yes in candidate source | Yes — prohibited-source guard | Conditional — keyboard/zoom/mobile check pending | **No** |
| Targeted frontend high-consequence migrations | Broadly implemented | Partial — strongest generator/Social/provider paths covered; domain coverage is not universal | Partial — remaining row findings and staging checks above apply | **No** |
| Targeted backend `BE-026`, `BE-030`, `BE-050`, `BE-059`, `BE-073`/`074`, `BE-081`–`083`, `BE-111`/`113`, `BE-118`, `BE-184`, and `BE-193`–`198` | Application guards, proof-only reconciliation routes, and retained League Live retry implemented; stated residuals remain | Yes for focused application behavior; no for deferred RPC/provider work | Conditional — staging verification required | **No** |
| All 232 frontend actions fully conforming | Audit complete; remediation not complete | No | No | **No** |
| All 198 backend contracts fully transaction-atomic and uniformly reconcilable | Boundary audit complete; not all contracts have that requirement or guarantee | No | No | **No** |

No line item in this report is formally accepted or production-deployed. Promotion remains a separate guarded decision after the exact candidate commit, final automated evidence, and manual staging acceptance are recorded.
