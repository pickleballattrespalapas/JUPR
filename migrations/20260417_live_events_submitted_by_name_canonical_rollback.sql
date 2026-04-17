-- Rollback for 20260417_live_events_submitted_by_name_canonical.sql
-- Canonical field remains submitted_by_name in forward schema.
-- This rollback removes transition helpers and canonical column.

drop trigger if exists live_events_sync_submitted_by_name on public.live_events;
drop function if exists public.live_events_sync_submitted_by_name();
alter table if exists public.live_events drop column if exists submitted_by_name;
