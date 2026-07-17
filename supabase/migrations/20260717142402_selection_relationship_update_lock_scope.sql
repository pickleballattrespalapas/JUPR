-- Follow-up to 20260717141224_selection_update_transaction_guards.sql.
-- UPDATE triggers lock only the selections referenced by NEW. This preserves
-- ON DELETE SET NULL cascades, where the OLD selection may already be gone.

create or replace function private.guard_tournament_registration_relationship_change()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_selection_ids text[];
  v_requester public.tournament_registration_selections%rowtype;
  v_target public.tournament_registration_selections%rowtype;
  v_selection_one public.tournament_registration_selections%rowtype;
  v_selection_two public.tournament_registration_selections%rowtype;
  v_team_link public.tournament_registration_team_links%rowtype;
begin
  if tg_table_name = 'tournament_registration_partner_requests' then
    v_selection_ids := array[new.requester_selection_id, new.target_selection_id];
  elsif tg_table_name = 'tournament_registration_team_links' then
    v_selection_ids := array[new.selection1_id, new.selection2_id];
  elsif tg_table_name = 'tournament_registration_team_members' then
    v_selection_ids := array[new.selection_id];
  else
    raise exception using
      errcode = '0A000',
      message = 'JUPR_RELATION_INVALID: unsupported relationship trigger table.';
  end if;

  perform private.lock_tournament_registration_selection_scope(v_selection_ids);

  if tg_table_name = 'tournament_registration_partner_requests' then
    select selection.*
    into strict v_requester
    from public.tournament_registration_selections as selection
    where selection.id = new.requester_selection_id;

    if v_requester.tournament_id <> new.tournament_id::text
       or v_requester.event_option_id <> new.event_option_id
       or v_requester.registration_id <> new.requester_registration_id then
      raise exception using
        errcode = '23514',
        message = 'JUPR_RELATION_INVALID: requester selection no longer matches the partner request.';
    end if;

    if new.target_selection_id is not null then
      select selection.*
      into strict v_target
      from public.tournament_registration_selections as selection
      where selection.id = new.target_selection_id;

      if v_target.tournament_id <> new.tournament_id::text
         or v_target.event_option_id <> new.event_option_id
         or v_target.registration_id is distinct from new.target_registration_id
         or v_target.id = v_requester.id then
        raise exception using
          errcode = '23514',
          message = 'JUPR_RELATION_INVALID: target selection no longer matches the partner request.';
      end if;

      if new.status = 'PENDING'
         and new.source = 'PUBLIC_PARTNER_BOARD'
         and (
           v_target.partner_mode <> 'NEEDS_PARTNER'
           or not v_target.show_on_partner_board
         ) then
        raise exception using
          errcode = '23514',
          message = 'JUPR_RELATION_INVALID: public partner-board target is no longer available.';
      end if;
    end if;

    if new.status = 'PENDING' and exists (
      select 1
      from public.tournament_registration_team_links as existing_link
      where existing_link.status in ('CONFIRMED', 'ADMIN_CONFIRMED')
        and (
          existing_link.selection1_id = any(array_remove(v_selection_ids, null))
          or existing_link.selection2_id = any(array_remove(v_selection_ids, null))
        )
    ) then
      raise exception using
        errcode = '23514',
        message = 'JUPR_RELATION_INVALID: a selection is already on a confirmed partner team.';
    end if;

    if new.status = 'PENDING' and exists (
      select 1
      from public.tournament_registration_partner_requests as existing_request
      where existing_request.id <> new.id
        and existing_request.status = 'PENDING'
        and existing_request.requester_selection_id = new.requester_selection_id
        and existing_request.target_selection_id is not distinct from new.target_selection_id
        and existing_request.target_player_id is not distinct from new.target_player_id
    ) then
      raise exception using
        errcode = '23505',
        message = 'JUPR_RELATION_INVALID: the same pending partner request already exists.';
    end if;

  elsif tg_table_name = 'tournament_registration_team_links' then
    select selection.*
    into strict v_selection_one
    from public.tournament_registration_selections as selection
    where selection.id = new.selection1_id;

    select selection.*
    into strict v_selection_two
    from public.tournament_registration_selections as selection
    where selection.id = new.selection2_id;

    if v_selection_one.id = v_selection_two.id
       or v_selection_one.tournament_id <> new.tournament_id::text
       or v_selection_two.tournament_id <> new.tournament_id::text
       or v_selection_one.event_option_id <> new.event_option_id
       or v_selection_two.event_option_id <> new.event_option_id
       or v_selection_one.registration_id <> new.registration1_id
       or v_selection_two.registration_id <> new.registration2_id then
      raise exception using
        errcode = '23514',
        message = 'JUPR_RELATION_INVALID: partner team link no longer matches its selections.';
    end if;

    if new.status in ('CONFIRMED', 'ADMIN_CONFIRMED') and exists (
      select 1
      from public.tournament_registration_team_links as existing_link
      where existing_link.id <> new.id
        and existing_link.status in ('CONFIRMED', 'ADMIN_CONFIRMED')
        and (
          existing_link.selection1_id in (new.selection1_id, new.selection2_id)
          or existing_link.selection2_id in (new.selection1_id, new.selection2_id)
        )
    ) then
      raise exception using
        errcode = '23505',
        message = 'JUPR_RELATION_INVALID: a selection is already on another confirmed partner team.';
    end if;

    if new.accepted_request_id is not null and not exists (
      select 1
      from public.tournament_registration_partner_requests as accepted_request
      where accepted_request.id = new.accepted_request_id
        and accepted_request.tournament_id = new.tournament_id
        and accepted_request.event_option_id = new.event_option_id
        and accepted_request.requester_selection_id in (new.selection1_id, new.selection2_id)
        and accepted_request.target_selection_id in (new.selection1_id, new.selection2_id)
    ) then
      raise exception using
        errcode = '23514',
        message = 'JUPR_RELATION_INVALID: accepted request does not match the partner team link.';
    end if;

  elsif tg_table_name = 'tournament_registration_team_members' then
    select selection.*
    into strict v_selection_one
    from public.tournament_registration_selections as selection
    where selection.id = new.selection_id;

    select team_link.*
    into strict v_team_link
    from public.tournament_registration_team_links as team_link
    where team_link.id = new.team_link_id;

    if v_selection_one.tournament_id <> new.tournament_id::text
       or v_selection_one.event_option_id <> new.event_option_id
       or v_selection_one.registration_id <> new.registration_id
       or v_team_link.tournament_id <> new.tournament_id
       or v_team_link.event_option_id <> new.event_option_id
       or new.selection_id not in (v_team_link.selection1_id, v_team_link.selection2_id) then
      raise exception using
        errcode = '23514',
        message = 'JUPR_RELATION_INVALID: partner team member no longer matches its selection and link.';
    end if;

    if new.status = 'ACTIVE'
       and v_team_link.status not in ('CONFIRMED', 'ADMIN_CONFIRMED') then
      raise exception using
        errcode = '23514',
        message = 'JUPR_RELATION_INVALID: an active team member requires a confirmed partner link.';
    end if;
  end if;

  return new;
exception
  when no_data_found then
    raise exception using
      errcode = '23514',
      message = 'JUPR_RELATION_INVALID: a referenced tournament registration relationship row no longer exists.';
end
$function$;

revoke all on function private.guard_tournament_registration_relationship_change()
  from public, anon, authenticated;
grant execute on function private.guard_tournament_registration_relationship_change()
  to service_role;

notify pgrst, 'reload schema';
