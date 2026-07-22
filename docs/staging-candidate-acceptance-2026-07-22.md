# Staging candidate acceptance record — 2026-07-22

This record binds the final confirmation-dialog candidate to its deployed staging
surfaces and preserves the bounded correction-intake evidence collected immediately
before that candidate. It is an acceptance workbench, not a completed 45-page parity
book and not a production-cutover approval.

## Final candidate identity

| Surface | Verified value |
|---|---|
| Application candidate | `9a0975f18d5d43b3f25e53872a80b04737c3a29c` |
| Merged change | PR `#1018`, “Replace typed admin confirmations with Yes/No dialogs” |
| Vercel staging alias | `https://jupr-git-staging-pickleballattrespalapas1.vercel.app` |
| Vercel deployment | `dpl_5rQuCqdPquvfnVmLiMHstJENDycS` |
| Vercel immutable origin | `https://jupr-mvw1gcqk4-pickleballattrespalapas1.vercel.app` |
| Vercel deployment state | `READY`; Git metadata attests the exact candidate SHA and branch `staging` |
| Fly final-`none` run | `https://github.com/pickleballattrespalapas/JUPR/actions/runs/29886383749` (job `88817686060`) |
| Fly image | `registry.fly.io/juprleagues-api-staging:deployment-01KY3V2M6M2DMR04ZEYSF9WE6V` |
| Fly deployment-time version | `01KY3V3T4C5JXGAJN6DCJG8VNJ` |
| Supabase staging project | `sijpxjxvdtrehmqvirfi` (`JUPR Staging`, `ACTIVE_HEALTHY`, `us-east-1`) |
| Migration inventory head | `20260720123402 baseline_worker_run_log` |
| Canonical Staging Smoke | `https://github.com/pickleballattrespalapas/JUPR/actions/runs/29886777122` |

The final Fly release completed before the smoke. Its verifier attested the exact
candidate SHA, `write_wave=none`,
`business_data_write_wave_active=false`, every controlled write flag false, and
email mode `dry_run`. The canonical smoke then completed successfully on the same
SHA with the strict Chromium public-read manifest: 56 passed, with no skipped,
flaky, retried, focused, or count-drifted tests.

Identity binding:

```text
candidate=9a0975f18d5d43b3f25e53872a80b04737c3a29c; vercel=dpl_5rQuCqdPquvfnVmLiMHstJENDycS; fly=registry.fly.io/juprleagues-api-staging:deployment-01KY3V2M6M2DMR04ZEYSF9WE6V; artifact=https://github.com/pickleballattrespalapas/JUPR/actions/runs/29886383749
```

## Bounded write evidence retained from the parent candidate

The approved data-correction create/exact-retry and dismissal completed on parent
hardening SHA `32cdd1fda80b542625e294e13e68a61d9c0e27aa`. The final candidate changed
the browser confirmation UX afterward. These rows are valid staging QA evidence,
but they are not represented as same-SHA formal write acceptance for `9a0975f…`.

| Step | Exact evidence |
|---|---|
| A — correction intake | Fly run `29840526376`, job `88667948087`, wave `public-intake-auth`, image `registry.fly.io/juprleagues-api-staging:deployment-01KY2J5BYDXDGJHS56XWFMEN5G`, deployment-time version `01KY2J6XKVC1K1KYZ0T40CVZSK` |
| A — durable row | `public.public_support_requests.id=req_4055accc54b34ab6977d`; type `data_correction`; requester `Test <test@x.invalid>`; subject `smoke`; created `2026-07-21T14:47:51.070955Z` |
| A — exact retry | Browser receipt reported the existing request; the durable readback contains exactly one matching row and one daily dedupe key |
| A — restoration | Fly run `29840964586`, job `88669460470`, wave `none`, image `registry.fly.io/juprleagues-api-staging:deployment-01KY2JEWEXGCQ30F9M8Q16DSG0`, deployment-time version `01KY2JFB7FGEXVTP429HHBBTNQ` |
| B — dismissal | Fly run `29841244823`, job `88670407361`, wave `support-requests`, image `registry.fly.io/juprleagues-api-staging:deployment-01KY2JN4W2V3BTX1QY99WZ7XFA`, deployment-time version `01KY2JP7YKT2VS2A8CB8Y40Q20` |
| B — durable readback | Request `req_4055accc54b34ab6977d` is `dismissed`; note `test only`; reviewed by `baumannjoe@yahoo.com` at `2026-07-22T01:41:05.192302Z` |
| B — audit | `public.admin_activity_log.id=4`; action `update_public_support_request_admin`; entity `public_support_request/req_4055accc54b34ab6977d`; before `new`; after `dismissed`; actor role `super_admin` |
| B — restoration | Fly run `29883968488`, job `88810575093`, wave `none`, image `registry.fly.io/juprleagues-api-staging:deployment-01KY3R03N2QG1C84NFSVV4QPJN`, deployment-time version `01KY3R14WDGHR2N0Y3XX2AE9YG`; the later final-candidate run `29886383749` supersedes it as the current all-writes-off release |

The dismissed request and audit row are intentionally retained as staging evidence.
No customer action, rating change, match change, player change, tournament change,
email delivery, or production mutation occurred.

The B wave remained enabled from its successful health check at approximately
`2026-07-21T14:54Z` until the dismissal and `none` restoration completed at
approximately `2026-07-22T01:47Z`. That roughly 10-hour-53-minute interval exceeded
the packet's immediate-close sequencing rule. The final state is safely restored,
but this deviation is another reason the parent-candidate write exercise remains
QA evidence rather than a formal write-wave `Pass` for the final candidate.

## Acceptance scope

Ready for operator and witness review:

- final Git/Vercel/Fly/Supabase identity equality;
- final Fly `none` state and all controlled write gates false;
- canonical public-read smoke success on the exact final SHA;
- durable staging readback for the prior bounded intake, deduplication, dismissal,
  reviewer attribution, and single audit event;
- confirmation-dialog implementation covered by the merged static/contract test
  suite, with backend confirmation phrases retained only as internal contracts.

Still outside this acceptance:

- completion of every row in `docs/next_parity_manual_staging_book.md`;
- same-SHA runtime acceptance of every protected write dialog;
- legal-copy approval, staging email delivery/inbox testing, and custom domains;
- any production deployment, production data, or production cutover.

## Operator and witness sign-off

The operator performs the protected workflow and owns the recovery boundary. The
witness must be a different person and independently checks the evidence; the
witness does not need staging secrets and must not perform a second mutation.

| Role | Identity | Decision | UTC time | Evidence/comment |
|---|---|---|---|---|
| Operator | Joe Baumann; GitHub `pickleballattrespalapas`; authenticated staging actor `baumannjoe@yahoo.com` | Accepted | `2026-07-22T15:50:12Z` | PR `#1019` comment `5048336382` |
| Witness | Not required for this limited staging public-read acceptance | N/A | — | A distinct witness remains required for the formal complete-book process or any later scope that explicitly requires separation of duties. |

This operator-only decision accepts the staging public-read scope above. It does not
complete the formal parity book, transfer the parent-candidate write evidence to the
final SHA, or approve legal copy, email delivery, custom domains, or production.

Operator acceptance statement:

```text
I operated the bounded staging checks, confirm the current release is write_wave=none,
and accept candidate 9a0975f18d5d43b3f25e53872a80b04737c3a29c for the staging public-read scope recorded here.
```

Witness acceptance statement:

```text
I independently verified the candidate SHA, Vercel deployment, Fly image and final-none
run, Supabase project, run ordering, and successful canonical Staging Smoke. I did not
perform an additional mutation. I accept the recorded staging public-read scope.
```

Do not merge an evidence-only commit and assume the existing deployment evidence
automatically transfers to the new commit SHA. If this record is later merged into
`staging`, either retain the immutable `9a0975f…` evidence as historical acceptance or
deploy the resulting documentation commit to both staging surfaces and rerun the
same-SHA final-`none` and canonical-smoke gates.
