# Tournament acceptance fix batch — CI alignment

The implementation intentionally renames routine Setup actions to **Save draft and continue** so operators understand that saved setup work remains private until **Publish reviewed setup** succeeds.

The builder validation fixture now includes the required per-day court count, labels, and operating hours introduced by this acceptance batch.

These are contract-alignment changes only. They do not expand production scope, and manual staging retesting remains required before acceptance.
