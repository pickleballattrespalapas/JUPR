-- Staff-reviewed identity linking. The registration and its team projections
-- must change in one transaction; public intake never calls this function.
create or replace function public.admin_link_tournament_registration_player_v1(
  p_club_id text,
  p_tournament_id text,
  p_registration_id text,
  p_player_id integer,
  p_expected_updated_at timestamptz,
  p_actor_email text,
  p_actor_role text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_registration public.tournament_registrations%rowtype;
  v_player public.players%rowtype;
  v_scope text[];
  v_updated_at timestamptz;
begin
  if p_expected_updated_at is null or p_player_id is null or p_player_id < 1
     or nullif(pg_catalog.btrim(p_actor_email), '') is null then
    return jsonb_build_object('ok',false,'code','INVALID','message','Choose a profile and reload the registration before linking.');
  end if;
  if not exists (select 1 from public.tournaments where id::text=p_tournament_id and club_id=p_club_id) then
    return jsonb_build_object('ok',false,'code','SCOPE','message','Tournament not found in this club.');
  end if;

  -- Serialize two staff members assigning the same player in this tournament.
  perform pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended(
    'jupr:tournament-player-link:'||p_tournament_id||':'||p_player_id::text,0));
  select array_agg(distinct selection_id order by selection_id) into v_scope from (
    select id as selection_id from public.tournament_registration_selections
      where tournament_id=p_tournament_id and registration_id=p_registration_id
    union
    select selection1_id from public.tournament_registration_team_links
      where tournament_id::text=p_tournament_id and status in ('CONFIRMED','ADMIN_CONFIRMED')
        and p_registration_id in (registration1_id,registration2_id)
    union
    select selection2_id from public.tournament_registration_team_links
      where tournament_id::text=p_tournament_id and status in ('CONFIRMED','ADMIN_CONFIRMED')
        and p_registration_id in (registration1_id,registration2_id)
  ) scope;
  perform private.lock_tournament_registration_selection_scope(coalesce(v_scope,array[]::text[]));
  select * into v_registration from public.tournament_registrations
    where id=p_registration_id and tournament_id=p_tournament_id for update;
  if not found then
    return jsonb_build_object('ok',false,'code','SCOPE','message','Registration not found in this tournament.');
  end if;
  select * into v_player from public.players where id=p_player_id and club_id=p_club_id for share;
  if not found or lower(v_player.name) like '%(merged into %' then
    return jsonb_build_object('ok',false,'code','PLAYER','message','Choose an available player profile from this club.');
  end if;
  if v_registration.status not in ('confirmed','admin_confirmed') then
    return jsonb_build_object('ok',false,'code','STATUS','message','Only confirmed registrations can be linked here. Refresh the draw.');
  end if;

  -- A lost response can be retried safely without a second identity change.
  if v_registration.player_id=p_player_id then
    return jsonb_build_object('ok',true,'mode','registration_player_link','idempotent_replay',true,
      'registration_id',p_registration_id,'player_id',p_player_id,'updated_at',v_registration.updated_at);
  end if;
  if v_registration.updated_at is distinct from p_expected_updated_at or v_registration.player_id is not null then
    return jsonb_build_object('ok',false,'code','STALE','message','This registration changed after it was loaded. Reload the draw and review the link again.');
  end if;

  if exists (select 1 from public.tournament_rating_eligibility_reviews
      where selection_id::text=any(v_scope) and review_phase='REGISTRATION_CLOSE' and finalized_at is not null) then
    return jsonb_build_object('ok',false,'code','FINALIZED','message','This partnership has a finalized rating review. Resolve that review before changing its player identity.');
  end if;
  if exists (select 1 from public.tournament_teams
      where tournament_id::text=p_tournament_id and source_selection_id::text=any(v_scope)) then
    return jsonb_build_object('ok',false,'code','IMPORTED','message','This partnership is already imported into a draw. Resolve its draw membership before linking.');
  end if;
  -- A player may enter different divisions, but cannot be two entrants in one.
  if exists (
    select 1 from public.tournament_registrations other
    join public.tournament_registration_selections theirs on theirs.registration_id=other.id and theirs.tournament_id=p_tournament_id
    join public.tournament_registration_selections ours on ours.registration_id=p_registration_id and ours.tournament_id=p_tournament_id
      and ours.event_option_id=theirs.event_option_id and ours.registration_day_id=theirs.registration_day_id
    where other.tournament_id=p_tournament_id and other.id<>p_registration_id and other.player_id=p_player_id
      and coalesce(other.status,'') not in ('cancelled','canceled')
  ) then
    return jsonb_build_object('ok',false,'code','DUPLICATE','message','That player is already registered in one of these divisions. Choose the correct profile or resolve the duplicate registration.');
  end if;
  if exists (select 1 from public.tournament_registration_team_members
      where tournament_id::text=p_tournament_id and registration_id=p_registration_id and status='ACTIVE'
        and player_id is not null and player_id<>p_player_id)
     or exists (select 1 from public.tournament_registration_team_links
      where tournament_id::text=p_tournament_id and status in ('CONFIRMED','ADMIN_CONFIRMED')
        and ((registration1_id=p_registration_id and player1_id is not null and player1_id<>p_player_id)
          or (registration2_id=p_registration_id and player2_id is not null and player2_id<>p_player_id))) then
    return jsonb_build_object('ok',false,'code','TEAM_CONFLICT','message','The partnership already references a different player. Review its team details before linking.');
  end if;

  -- Preserve registration ratings, payments, entry choices, and partnership.
  update public.tournament_registrations set player_id=p_player_id,updated_at=pg_catalog.clock_timestamp()
    where id=p_registration_id returning updated_at into v_updated_at;
  update public.tournament_registration_team_members set player_id=p_player_id
    where tournament_id::text=p_tournament_id and registration_id=p_registration_id and status='ACTIVE' and player_id is null;
  update public.tournament_registration_team_links set player1_id=p_player_id,updated_at=pg_catalog.clock_timestamp()
    where tournament_id::text=p_tournament_id and registration1_id=p_registration_id
      and status in ('CONFIRMED','ADMIN_CONFIRMED') and player1_id is null;
  update public.tournament_registration_team_links set player2_id=p_player_id,updated_at=pg_catalog.clock_timestamp()
    where tournament_id::text=p_tournament_id and registration2_id=p_registration_id
      and status in ('CONFIRMED','ADMIN_CONFIRMED') and player2_id is null;
  update public.tournament_registration_partner_requests set requester_player_id=p_player_id,updated_at=pg_catalog.clock_timestamp()
    where tournament_id::text=p_tournament_id and requester_registration_id=p_registration_id and requester_player_id is null;
  update public.tournament_registration_partner_requests set target_player_id=p_player_id,updated_at=pg_catalog.clock_timestamp()
    where tournament_id::text=p_tournament_id and target_registration_id=p_registration_id and target_player_id is null;
  update public.tournament_primary_player_reconciliation
    set state='EXISTING_REGISTRATION',reason_code='ADMIN_LINKED_EXISTING_PLAYER',updated_at=pg_catalog.clock_timestamp()
    where registration_id=p_registration_id and tournament_id::text=p_tournament_id and club_id=p_club_id;
  update public.tournament_registration_player_projection_reconciliation
    set state='RESOLVED',registration_player_id=p_player_id,updated_at=pg_catalog.clock_timestamp()
    where registration_id=p_registration_id and tournament_id::text=p_tournament_id and state='PENDING';
  insert into public.admin_activity_log(club_id,actor_email,actor_role,action_type,entity_type,entity_id,before_json,after_json,source_page,flagged_for_review)
    values(p_club_id,p_actor_email,p_actor_role,'link_tournament_registration_player','tournament_registration',p_registration_id,
      jsonb_build_object('player_id',null,'updated_at',v_registration.updated_at),
      jsonb_build_object('player_id',p_player_id,'updated_at',v_updated_at,'tournament_id',p_tournament_id),
      'next_tournament_ops_player_link',true);
  return jsonb_build_object('ok',true,'mode','registration_player_link','registration_id',p_registration_id,
    'player_id',p_player_id,'updated_at',v_updated_at);
end
$function$;

revoke all on function public.admin_link_tournament_registration_player_v1(text,text,text,integer,timestamptz,text,text)
  from public,anon,authenticated;
grant execute on function public.admin_link_tournament_registration_player_v1(text,text,text,integer,timestamptz,text,text)
  to service_role;
notify pgrst,'reload schema';
