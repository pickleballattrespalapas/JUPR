# Staging parity operator guide — 2026-07-22

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

For every manual-only staging write, a second person is the witness. The witness
observes or independently reviews the named browser action and visible result; they
do not type, copy, or verify Git SHAs, deployment IDs, database row IDs, or audit
IDs. The evidence runner records and validates that metadata. Automated-only waves
remain covered by their candidate-bound workflow artifact and do not acquire a
manual-row witness requirement.

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
   recovery, and return to `none`. A witness is required only for the manual
   write rows named by the formal book.
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

The witness receives the same short behavior expectation. They confirm that the
operator performed the named action and that the expected result or recovery was
visible. Neither person is asked to transcribe candidate or deployment metadata.

The evidence runner captures normal success evidence. No prompt should ask the
operator to inspect source code, network requests, database rows, environment
variables, workflow IDs, or deployment identifiers.

## Stop and recovery rules

Stop the current batch on wrong-club data, private contact details, a missing or
unexpected record, an external email, a stale-state warning, a failed Yes/No
dialog, duplicate rows after an exact retry, or any result that cannot be restored.
Do not repeat the action until the evidence runner has reconciled the authoritative
state. Never proceed from one active write wave directly into another.

After the final batch, the evidence runner verifies the same candidate with every
controlled write flag false and email delivery still safe. Production cutover,
legal-copy approval, and live email delivery remain separate later decisions.
