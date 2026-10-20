-- Round-Robin and Ladder Generator routes use the same private durable ledger.
-- Allow their existing surface without changing idempotency, version leases,
-- lifecycle states, RLS, or grants.
alter table public.live_ladder_admin_operations
  drop constraint if exists live_ladder_admin_operations_surface_check,
  add constraint live_ladder_admin_operations_surface_check
    check (surface in ('challenge_ladder', 'moneyball', 'jupr_live_admin', 'play_generator'));

notify pgrst, 'reload schema';
