begin;

-- Deleting an unused tournament draft is an existing admin operation. Remove
-- its history membership in the same transaction so the previous edition can
-- start a replacement. Keep the original action receipt for audit/idempotency.
create function public.pcs_event_deleted_tournament_draft()
returns trigger language plpgsql security invoker set search_path=public as $$
begin
 if upper(old.status)='DRAFT' then
  perform pg_advisory_xact_lock(hashtextextended('pcs-event-history:'||old.club_id||':tournament',0));
  delete from public.pcs_event_editions where club_id=old.club_id and event_kind='tournament' and source_id=old.id::text;
 end if;
 return old;
end $$;
revoke all on function public.pcs_event_deleted_tournament_draft() from public,anon,authenticated;
grant execute on function public.pcs_event_deleted_tournament_draft() to service_role;
create trigger pcs_event_deleted_tournament_draft after delete on public.tournaments
 for each row execute function public.pcs_event_deleted_tournament_draft();
notify pgrst,'reload schema';
commit;
