# Staging parity operator guide — 2026-07-22

> Protected write windows are now controlled through locked issue `#1062`.
> Joe never runs a workflow form. The connected agent applies the exact
> open/advance/close body and the controller restores `write_wave=none`.
> See `docs/staging-write-session-controller.md`.

This guide turns the remaining parity session into short behavior checks. The
operator should never need to copy a Git SHA, deployment ID, database row ID, or
audit ID. The evidence runner records those details and fails closed if the tested
candidate, project, or active write wave changes.

## Division of responsibility

The operator does only three things:

1. follow one browser instruction at a time;
2. report whether the expected result is visibly shown; and
3. stop immediately if the page, data, or result differs from the instruction.

The evidence runner owns candidate binding, migration and flag checks, disposable
fixture IDs, authoritative readback, audit attribution, deduplication counts,
restoration, and the final `write_wave=none` proof.

Every manual-only staging write uses one of two review modes:

- a second person witnesses the named browser action and visible result; or
- candidate-bound GitHub automation records the exact candidate, Actions run,
  artifact, authoritative readback, and recovery result.

The witness, when used, does not type, copy, or verify Git SHAs, deployment IDs,
database row IDs, or audit IDs. The evidence runner records and validates that
metadata. In automated-review mode, it also writes the exact structured sign-off
to the formal book. Automated-only waves remain covered by their candidate-bound
workflow artifact and do not acquire a manual-row witness requirement.

## Minimal repeatable fixture text

Use these values whenever a form permits them. Do not invent long dated names.

| Field | Value |
|---|---|
| Name | `Test` |
| Email | `test@x.invalid` |
| Subject / title | `smoke` |
| Details | `staging only` |
| Admin note | `test only` |
| Requested change / action | `none` |
| Player name, only when required | `Test Player` |

IDs, links, idempotency keys, scores, and before-state values come from the active
fixture packet; the operator does not type or choose them. Confirmation is by an
explicit Yes/No dialog, never a typed phrase.

## Session order

The evidence runner opens at most one write wave. Every mutating batch follows:

`none -> one named wave -> authoritative readback/recovery -> none`

The operator-facing work is grouped into three sessions:

1. **Non-mutating public and admin acceptance:** public ratings/discovery,
   policies, all admin read/filter/print/export views, authenticated automatic
   loading, guided Tournament Setup drafting without save, diagnostics, and
   mobile/narrow-screen checks.
2. **Bounded write and recovery batches:** support/identity intake; registration
   and pairing; league and communications; match/player recovery; ladder/live
   operations; and tournament operations. Each sub-batch uses one disposable
   fixture family and its own least-privilege wave, authoritative readback,
   recovery, and return to `none`. Each manual write row uses either a distinct
   human witness or candidate-bound automated review.
3. **Exact-candidate restoration:** remove all disposable fixtures that are
   required to be removed, deploy a same-candidate `write_wave=none` release,
   prove every controlled write flag false and email delivery safe, and run the
   canonical Staging Smoke against the exact Vercel/Fly candidate.

This keeps the 20 non-mutating page acceptances together, reuses fixtures inside
the 25 write/recovery rows, and avoids asking the operator to return for isolated
one-page checks.

## Operator prompt format

Each operator prompt must contain only:

- the exact link or navigation label;
- one action to perform;
- the visible result to look for; and
- the failure screenshot or text to send only if the result differs.

In human-review mode, the witness receives the same short behavior expectation
and confirms that the operator performed the named action and saw the expected
result or recovery. Neither person is asked to transcribe candidate or deployment
metadata. In automated-review mode, no witness prompt is sent; the evidence runner
must capture the exact candidate-bound GitHub run and artifact.

The evidence runner captures normal success evidence. No prompt should ask the
operator to inspect source code, network requests, database rows, environment
variables, workflow IDs, or deployment identifiers.

The formal book accepts only these exact sign-off shapes:

- `operator=<identity>; witness=<different-identity>`
- `operator=<identity>; automated=candidate=<40-sha>,run=https://github.com/pickleballattrespalapas/JUPR/actions/runs/<run-id>,artifact=<same-run-artifact-url|sha256:64-hex>`

Stop if the automation run or artifact is missing, points to another repository
or run, or names a candidate other than the recorded staging candidate. Do not
replace the structured record with prose such as `review=automated`.

## Stop and recovery rules

Stop the current batch on wrong-club data, private contact details, a missing or
unexpected record, an external email, a stale-state warning, a failed Yes/No
dialog, duplicate rows after an exact retry, or any result that cannot be restored.
Do not repeat the action until the evidence runner has reconciled the authoritative
state. Never proceed from one active write wave directly into another.

After the final batch, the evidence runner verifies the same candidate with every
controlled write flag false and email delivery still safe. Production cutover,
legal-copy approval, and live email delivery remain separate later decisions.
