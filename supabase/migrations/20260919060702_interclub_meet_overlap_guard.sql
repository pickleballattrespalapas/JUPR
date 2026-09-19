begin;
-- Applies equally to post-invitation scheduling and weather replays. Serialize
-- against another meet write before checking courts/club availability.
create function public.pcs_guard_interclub_meet_overlap() returns trigger
language plpgsql security invoker set search_path=public as $$
begin
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||new.season_id::text,0));
 if exists(select 1 from public.pcs_interclub_meets other
  where other.season_id=new.season_id and other.id<>new.id
   and other.starts_at<new.starts_at+make_interval(mins=>new.duration_minutes)
   and new.starts_at<other.starts_at+make_interval(mins=>other.duration_minutes)
   and (other.host_club_id=new.host_club_id or
    exists(select 1 from jsonb_array_elements_text(new.club_ids) cid where other.club_ids ? cid))) then
  raise exception 'A participating club or host already has a meet at this time' using errcode='22023';
 end if;
 return new;
end $$;
revoke all on function public.pcs_guard_interclub_meet_overlap() from public,anon,authenticated;
grant execute on function public.pcs_guard_interclub_meet_overlap() to service_role;
create trigger pcs_interclub_meet_overlap before insert or update of starts_at,duration_minutes,club_ids,host_club_id
 on public.pcs_interclub_meets for each row execute function public.pcs_guard_interclub_meet_overlap();
commit;
