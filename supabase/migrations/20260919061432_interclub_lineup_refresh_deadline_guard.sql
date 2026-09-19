begin;
-- The audit is inserted inside the same transaction as the result update. A
-- meet beginning between the API check and its write must roll the update back.
create function public.pcs_guard_interclub_lineup_refresh_deadline() returns trigger
language plpgsql security invoker set search_path=public as $$
declare meet public.pcs_interclub_meets;
begin
 if new.action='refresh_lineups' and coalesce(new.before_document#>>'{document,weather}','normal')<>'rescheduled' then
  select m.* into meet from public.pcs_interclub_meets m join public.pcs_interclub_competition_batches b on b.meet_id=m.id
   where b.id=new.batch_id;
  if not found or meet.starts_at<=now() then
   raise exception 'The meet has begun; only injury changes between games are allowed' using errcode='40001';
  end if;
 end if;
 return new;
end $$;
revoke all on function public.pcs_guard_interclub_lineup_refresh_deadline() from public,anon,authenticated;
grant execute on function public.pcs_guard_interclub_lineup_refresh_deadline() to service_role;
create trigger pcs_interclub_refresh_deadline before insert on public.pcs_interclub_competition_audit
 for each row execute function public.pcs_guard_interclub_lineup_refresh_deadline();
commit;
