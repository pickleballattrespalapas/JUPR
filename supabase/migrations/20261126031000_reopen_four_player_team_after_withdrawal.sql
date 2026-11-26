-- Registration cancellation marks the registrant cancelled, then deletes their
-- membership. Keep that transaction from leaving a three-person team confirmed.
create or replace function public.reopen_four_player_team_on_registration_withdrawal()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
declare
  v_team public.tournament_four_player_teams%rowtype;
begin
  -- Team removal, roster replacement and cascading parent deletes have their
  -- own lifecycle. Only the withdrawal of a registered player opens a spot here.
  if not exists (
    select 1 from public.tournament_registrations registration
     where registration.id = old.registration_id
       and registration.tournament_id = old.tournament_id::text
       and lower(registration.status) = 'cancelled'
  ) then
    return old;
  end if;
  select team.* into v_team from public.tournament_four_player_teams team
   where team.id = old.team_id for update;
  if not found or v_team.status in ('CANCELLED', 'WITHDRAWN') then
    return old;
  end if;
  if v_team.draw_id is not null or exists (
    select 1 from public.tournament_team_matchups matchup
     where v_team.id in (matchup.team_a_id, matchup.team_b_id) and matchup.status <> 'VOID'
  ) then
    raise exception 'JUPR_CANCEL_IMPORTED: Remove the affected four-player team from its draw before cancelling this registration.';
  end if;
  if v_team.status in ('REVIEW_REQUIRED', 'CONFIRMED') then
    update public.tournament_four_player_teams
       set status = 'FORMING',
           eligibility_state = case when eligibility_state = 'ELIGIBLE' then 'REVIEW_REQUIRED' else eligibility_state end,
           version = version + 1
     where id = v_team.id;
  end if;
  return old;
end;
$$;

revoke all on function public.reopen_four_player_team_on_registration_withdrawal() from public, anon, authenticated;
grant execute on function public.reopen_four_player_team_on_registration_withdrawal() to service_role;
create trigger reopen_four_player_team_before_registration_withdrawal
before delete on public.tournament_four_player_team_members
for each row when (old.registration_id is not null and old.status in ('ACCEPTED', 'INVITED'))
execute function public.reopen_four_player_team_on_registration_withdrawal();

-- Repair previously saved vacancies only when the captain is still on the team
-- and the team has neither a draw nor a live or completed matchup.
update public.tournament_four_player_teams team
   set status = 'FORMING',
       eligibility_state = case when eligibility_state = 'ELIGIBLE' then 'REVIEW_REQUIRED' else eligibility_state end,
       version = version + 1
 where team.status in ('CONFIRMED', 'REVIEW_REQUIRED') and team.draw_id is null
   and exists (
     select 1 from public.tournament_four_player_team_members captain
      where captain.team_id = team.id and captain.registration_id = team.captain_registration_id
        and captain.status = 'ACCEPTED'
   )
   and (select count(*) from public.tournament_four_player_team_members member
         where member.team_id = team.id and member.status in ('ACCEPTED', 'INVITED')) < 4
   and not exists (
     select 1 from public.tournament_team_matchups matchup
      where team.id in (matchup.team_a_id, matchup.team_b_id) and matchup.status <> 'VOID'
   );
