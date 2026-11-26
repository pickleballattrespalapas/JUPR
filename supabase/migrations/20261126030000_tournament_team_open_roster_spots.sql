-- Keep the original creation RPC, grants, operation ledger, and invitation flow.
-- Only occupied spots are submitted. The captain and each named player still
-- need distinct slots and emails. Confirmation continues to require four.
do $migration$
declare
  definition text := pg_get_functiondef(
    'public.admin_create_tournament_four_player_team(text,text,text,text,text,jsonb,text,text,text,text)'::regprocedure
  );
begin
  if position('jsonb_array_length(p_members) <> 4' in definition) = 0
     or position(') <> 4 or exists (' in definition) = 0
     or position(') <> 4 then' in definition) = 0
     or position('upper(coalesce(selection.partner_mode, ''NONE'')) = ''NONE''' in definition) = 0 then
    raise exception 'Unexpected four-player creation contract';
  end if;
  definition := replace(definition,
    'jsonb_array_length(p_members) <> 4', 'jsonb_array_length(p_members) not between 2 and 4');
  definition := replace(definition,
    ') <> 4 or exists (', ') <> pg_catalog.jsonb_array_length(p_members) or exists (');
  definition := replace(definition,
    ') <> 4 then', ') <> pg_catalog.jsonb_array_length(p_members) then');
  definition := replace(definition,
    'upper(coalesce(selection.partner_mode, ''NONE'')) = ''NONE''',
    'upper(coalesce(selection.partner_mode, ''NONE'')) in (''NONE'', ''NEEDS_PARTNER'')');
  execute definition;
end;
$migration$;

-- Tournament-wide registration alone does not reserve a place in this
-- division. Verify the selected event under lock when accepting an invite.
-- The original replay and decline paths keep their existing semantics.
do $migration$
declare
  definition text := pg_get_functiondef(
    'public.server_respond_tournament_four_player_invite(text,text,text,text,text,integer,text,text,text,text,text)'::regprocedure
  );
  accept_start constant text := $fragment$  if v_action = 'ACCEPT' then
    v_gender := regexp_replace($fragment$;
begin
  if position(accept_start in definition) = 0 then
    raise exception 'Unexpected four-player invitation acceptance contract';
  end if;
  execute replace(definition, accept_start, $fragment$  if v_action = 'ACCEPT' then
    perform 1 from public.tournament_registration_selections selection
     where selection.tournament_id = p_tournament_id
       and selection.registration_id = v_registration.id
       and selection.event_option_id = v_team.event_option_id
     for share;
    if not found then
      raise exception using errcode = '22023',
        message = 'JUPR_TOURNAMENT_TEAM_INVITATION_EVENT_SELECTION_REQUIRED';
    end if;
    v_gender := regexp_replace($fragment$);
end;
$migration$;

-- Once a registered player joins any team, the selection must no longer
-- advertise a solo request. This covers captains, accepted invitations and
-- organizer assignments, in the same transaction that records membership.
create or replace function public.clear_four_player_team_request_on_placement()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
begin
  update public.tournament_registration_selections
     set partner_mode = 'NONE', show_on_partner_board = false
   where tournament_id::text = new.tournament_id::text
     and event_option_id = new.event_option_id
     and registration_id = new.registration_id
     and partner_mode = 'NEEDS_PARTNER';
  return new;
end;
$$;
revoke all on function public.clear_four_player_team_request_on_placement() from public, anon, authenticated;
grant execute on function public.clear_four_player_team_request_on_placement() to service_role;
create trigger clear_four_player_team_request_after_placement
after insert or update of status, registration_id on public.tournament_four_player_team_members
for each row when (new.status = 'ACCEPTED' and new.registration_id is not null)
execute function public.clear_four_player_team_request_on_placement();

-- A captain can add invitations to open spots without replacing existing
-- members. Concurrent captains' requests serialize on the team row.
create or replace function public.server_fill_tournament_four_player_team_spots(
  p_club_id text, p_tournament_id text, p_event_option_id text, p_team_id text,
  p_captain_registration_id text, p_members jsonb, p_operation_key text,
  p_request_fingerprint text, p_actor text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $$
declare
  v_team public.tournament_four_player_teams%rowtype;
  v_saved public.tournament_four_player_team_members%rowtype;
  v_member jsonb;
  v_slot text;
  v_email text;
  v_name text;
  v_invited jsonb := '[]'::jsonb;
  v_operation jsonb;
begin
  select team.* into v_team
    from public.tournament_four_player_teams team
    join public.tournaments tournament on tournament.id = team.tournament_id
    join public.tournament_registrations captain on captain.id = team.captain_registration_id
   where tournament.club_id::text = p_club_id and team.tournament_id::text = p_tournament_id
     and team.event_option_id = p_event_option_id and team.id::text = p_team_id
     and captain.tournament_id = p_tournament_id and captain.id = p_captain_registration_id
     and upper(captain.status) = 'CONFIRMED' and lower(btrim(captain.email)) = lower(btrim(p_actor))
     and exists (
       select 1 from public.tournament_registration_selections selection
        where selection.tournament_id = p_tournament_id
          and selection.registration_id = captain.id and selection.event_option_id = team.event_option_id
     )
   for update of team;
  if not found then
    raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_CAPTAIN_EVENT_SELECTION_REQUIRED';
  end if;
  v_operation := public.begin_tournament_team_operation(
    p_operation_key, p_request_fingerprint, p_club_id, p_tournament_id,
    'registration', 'four_player_team_fill', 'tournament_four_player_team', p_team_id, p_actor
  );
  if (v_operation->>'replay')::boolean then
    return v_operation->'result';
  end if;
  if v_team.status <> 'FORMING' or v_team.draw_id is not null or exists (
    select 1 from public.tournament_team_matchups matchup
    where v_team.id in (matchup.team_a_id, matchup.team_b_id) and matchup.status <> 'VOID'
  ) then
    raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_TEAM_ROSTER_LOCKED';
  end if;
  if pg_catalog.jsonb_typeof(coalesce(p_members, 'null'::jsonb)) <> 'array'
     or pg_catalog.jsonb_array_length(p_members) not between 1 and 3
     or (select count(distinct upper(btrim(member->>'slot'))) from pg_catalog.jsonb_array_elements(p_members) member)
        <> pg_catalog.jsonb_array_length(p_members)
     or (select count(distinct lower(btrim(member->>'email'))) from pg_catalog.jsonb_array_elements(p_members) member)
        <> pg_catalog.jsonb_array_length(p_members) then
    raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_FOUR_PLAYER_ROSTER_INVALID';
  end if;

  for v_member in select value from pg_catalog.jsonb_array_elements(p_members)
  loop
    v_slot := upper(btrim(v_member->>'slot'));
    v_email := lower(btrim(v_member->>'email'));
    v_name := btrim(v_member->>'display_name');
    if v_slot not in ('MAN_1','MAN_2','WOMAN_1','WOMAN_2') or nullif(v_name, '') is null
       or v_email !~ '^[^@[:space:]]+@[^@[:space:]]+\.[^@[:space:]]+$'
       or nullif(v_member->>'registration_id', '') is not null then
      raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_FOUR_PLAYER_ROSTER_INVALID';
    end if;
    if exists (
      select 1 from public.tournament_four_player_team_members member
       where member.team_id = v_team.id and member.status in ('INVITED','ACCEPTED')
         and (member.slot = v_slot or lower(btrim(member.invited_email)) = v_email)
    ) then
      raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_TEAM_SPOT_ALREADY_FILLED';
    end if;
    update public.tournament_four_player_team_members
       set status = 'REMOVED', invitation_token_hash = null, invitation_version = invitation_version + 1
     where team_id = v_team.id and slot = v_slot and status = 'DECLINED';
    insert into public.tournament_four_player_team_members (
      team_id, tournament_id, event_option_id, slot, invited_email, display_name_snapshot, gender_snapshot, status
    ) values (
      v_team.id, v_team.tournament_id, v_team.event_option_id, v_slot, v_email, v_name,
      case when v_slot like 'MAN_%' then 'Men' else 'Women' end, 'INVITED'
    ) returning * into v_saved;
    v_invited := v_invited || jsonb_build_array(to_jsonb(v_saved));
  end loop;
  update public.tournament_four_player_teams set version = version + 1
   where id = v_team.id returning * into v_team;
  insert into public.tournament_team_audit_events (
    club_id, tournament_id, event_option_id, entity_type, entity_id, action, actor, after_json, request_fingerprint
  ) values (
    p_club_id, v_team.tournament_id, v_team.event_option_id, 'tournament_four_player_team', p_team_id,
    'four_player_team_spots_filled', p_actor, v_invited, p_request_fingerprint
  );
  return public.complete_tournament_team_operation(p_operation_key, p_request_fingerprint,
    jsonb_build_object('ok', true, 'team', to_jsonb(v_team), 'members', v_invited));
end;
$$;

revoke all on function public.server_fill_tournament_four_player_team_spots(
  text,text,text,text,text,jsonb,text,text,text
) from public, anon, authenticated;
grant execute on function public.server_fill_tournament_four_player_team_spots(
  text,text,text,text,text,jsonb,text,text,text
) to service_role;
