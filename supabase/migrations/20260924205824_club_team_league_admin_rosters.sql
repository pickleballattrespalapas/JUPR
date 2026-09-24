-- Apply after team_league_normalized_rosters_substitute_pool, as ordered
-- by the production migration contract. Preserves private invoker execution.
create or replace function public.team_league_guard_roster_mutation_v1()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_club_id text := case when tg_op = 'DELETE' then old.club_id else new.club_id end;
  v_league_name text := case when tg_op = 'DELETE' then old.league_name else new.league_name end;
  v_settings public.team_league_settings%rowtype;
  v_becoming_confirmed boolean := (
    tg_op <> 'DELETE'
    and new.status = 'confirmed'
    and (tg_op = 'INSERT' or old.status is distinct from 'confirmed')
  );
  v_new_registration boolean := (
    tg_op = 'INSERT'
    and new.status in ('pending_partner', 'confirmed')
  );
  v_roster_changed boolean := (
    (tg_op = 'INSERT' and new.status = 'confirmed')
    or (tg_op = 'DELETE' and old.status = 'confirmed')
    or (
      tg_op = 'UPDATE'
      and (old.status = 'confirmed' or new.status = 'confirmed')
      and (
        old.status is distinct from new.status
        or old.captain_player_id is distinct from new.captain_player_id
        or old.partner_player_id is distinct from new.partner_player_id
      )
    )
  );
  v_normalized_completion boolean := false;
  v_admin_creation boolean := false;
begin
  select settings.*
    into v_settings
    from public.team_league_settings as settings
   where settings.club_id = v_club_id
     and settings.league_name = v_league_name
   for update;
  if not found then
    raise exception using
      errcode = '23503',
      message = 'TEAM_LEAGUE_SETTINGS_NOT_FOUND';
  end if;

  -- Only the audited admin-create RPC may assemble a roster while player
  -- signup is closed. Public registration and partner confirmation still
  -- require an open signup window; schedule locks apply to every creation.
  if tg_op = 'INSERT' and new.created_operation_id is not null then
    select exists (
      select 1 from public.team_league_operations as operation
       where operation.id = new.created_operation_id
         and operation.club_id = v_club_id
         and operation.league_name = v_league_name
         and operation.operation_type = 'admin_create_team'
         and operation.status = 'started'
         and operation.request_json ->> 'team_name' = new.team_name
         and (operation.request_json ->> 'captain_player_id')::bigint = new.captain_player_id
         and (operation.request_json ->> 'initial_primary_player_id')::bigint
             is not distinct from new.partner_player_id
    ) into v_admin_creation;
  end if;

  if tg_op = 'UPDATE'
     and old.status = 'pending_partner'
     and new.status = 'confirmed'
     and (
       pg_catalog.to_jsonb(old) - 'status' - 'updated_at'
     ) = (
       pg_catalog.to_jsonb(new) - 'status' - 'updated_at'
     ) then
    select count(*) = v_settings.team_size
      into v_normalized_completion
      from public.team_league_team_members as member
     where member.team_id = new.id
       and member.club_id = new.club_id
       and member.league_name = new.league_name
       and member.status = 'active'
       and member.role in ('captain', 'primary');
    if v_normalized_completion then
      perform public.team_league_assert_roster_policy_v1(
        v_club_id,
        v_league_name,
        v_settings.team_size,
        v_settings.team_category,
        v_settings.max_alternates,
        v_settings.mixed_required_men,
        v_settings.mixed_required_women
      );
    end if;
  end if;

  if (v_becoming_confirmed or v_new_registration)
     and not v_normalized_completion
     and (
       (not v_admin_creation and (
         not v_settings.registration_open
         or v_settings.status <> 'registration_open'
         or (
           v_settings.registration_closes_at is not null
           and v_settings.registration_closes_at <= pg_catalog.clock_timestamp()
         )
       ))
       or v_settings.schedule_version <> 0
       or exists (
         select 1
           from public.team_league_fixtures as fixture
          where fixture.club_id = v_club_id
            and fixture.league_name = v_league_name
       )
     ) then
    raise exception using
      errcode = '55000',
      message = 'TEAM_LEAGUE_REGISTRATION_CLOSED';
  end if;

  if v_roster_changed
     and v_settings.schedule_version > 0
     and not (
       tg_op = 'UPDATE'
       and old.status = new.status
       and old.captain_player_id is not distinct from new.captain_player_id
       and old.partner_player_id is not distinct from new.partner_player_id
     ) then
    raise exception using
      errcode = '55000',
      message = 'TEAM_LEAGUE_ROSTER_LOCKED_AFTER_SCHEDULE';
  end if;
  return case when tg_op = 'DELETE' then old else new end;
end
$function$;

revoke all on function public.team_league_guard_roster_mutation_v1()
  from public, anon, authenticated;
grant execute on function public.team_league_guard_roster_mutation_v1() to service_role;

notify pgrst, 'reload schema';
