# Staging parity operator guide — 2026-07-22

This guide turns the remaining parity session into short behavior checks. The
operator should never need to copy a Git SHA, deployment ID, database row ID, or
audit ID. The evidence runner records those details and fails closed if the tested
candidate, project, or active write wave changes.

## Division of responsibility

The operator does only three things:

1. follow one browser instruction at a time;
2. report what is visibly shown, usually with one screenshot; and
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
| Details / note | `test only` |
| Requested change / action | `none` |
| Player name, only when required | `Test Player` |

IDs, links, idempotency keys, scores, and before-state values come from the active
fixture packet; the operator does not type or choose them. Confirmation is by an
explicit Yes/No dialog, never a typed phrase.

## Session order

The evidence runner opens at most one write wave. Every mutating batch follows:

`none -> one named wave -> authoritative readback/recovery -> none`

The operator-facing session is grouped into eight batches:

1. **Public ratings and discovery:** leaderboards, rules, league results/print,
   match explorer, players, badges, ladder, FAQ, support, and policy display.
2. **Admin read and export:** badge diagnostics/audit, canonical audit, Admin
   Guide, filters, print/CSV output, and the three auto-load UX changes.
3. **Support and identity intake:** support create/exact retry, correction,
   privacy, verified update, preference-link guard, login, and recovery messaging.
4. **Registration and pairing:** registration, confirmation, edit, roster,
   partner board, and pairing lifecycle on a disposable event.
5. **League and communications:** manager, awards, live round, weekly recap,
   player updates, subscription/outbox, and printable top players.
6. **Match and player recovery:** uploader, Match Log review, player editor,
   replay/canonical readback, and exact restoration.
7. **Ladder and live operations:** challenge ladder, Moneyball, JUPR Live admin,
   and public live using disposable sessions only.
8. **Tournament operations:** admin/setup, registration admin, operations, and
   live score/non-score actions using a disposable tournament and full recovery.

The first two batches cover the 20 non-mutating page acceptances. The last six
group the 25 bounded write/recovery acceptances so shared fixtures are created and
cleaned once instead of making the operator retype the same data.

## Operator prompt format

Each operator prompt must contain only:

- the exact link or navigation label;
- one action to perform;
- the visible result to look for; and
- what screenshot to send.

The witness receives the same short behavior expectation. They confirm that the
operator performed the named action and that the expected result or recovery was
visible. Neither person is asked to transcribe candidate or deployment metadata.

No prompt should ask the operator to inspect source code, network requests,
database rows, environment variables, workflow IDs, or deployment identifiers.

## Stop and recovery rules

Stop the current batch on wrong-club data, private contact details, a missing or
unexpected record, an external email, a stale-state warning, a failed Yes/No
dialog, duplicate rows after an exact retry, or any result that cannot be restored.
Do not repeat the action until the evidence runner has reconciled the authoritative
state. Never proceed from one active write wave directly into another.

After the final batch, the evidence runner verifies the same candidate with every
controlled write flag false and email delivery still safe. Production cutover,
legal-copy approval, and live email delivery remain separate later decisions.
