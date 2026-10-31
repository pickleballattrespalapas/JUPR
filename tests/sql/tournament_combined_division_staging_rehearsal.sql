-- Run only in the isolated staging project. All fixture rows and trigger side
-- effects are rolled back. Existing players are read, never changed.
begin;
do $rehearsal$
declare
  v_tournament uuid := gen_random_uuid();
  v_draw_a uuid := gen_random_uuid();
  v_draw_b uuid := gen_random_uuid();
  v_day text := gen_random_uuid()::text;
  v_event_a text := gen_random_uuid()::text;
  v_event_b text := gen_random_uuid()::text;
  v_club text;
  v_players integer[];
  v_result jsonb;
  v_draws jsonb;
  v_teams jsonb;
  v_events jsonb;
  v_original_team_ids uuid[];
begin
  select club_id into v_club from public.players where club_id is not null
    group by club_id having count(*) >= 8 order by club_id limit 1;
  if v_club is null then raise exception 'Staging rehearsal requires a club with eight players.'; end if;
  select array_agg(id order by id) into v_players from (
    select id from public.players where club_id=v_club order by id limit 8
  ) p;
  insert into public.tournaments(id,club_id,name,team_count,status)
    values(v_tournament,v_club,'Combined division staging rehearsal',4,'DRAFT');
  insert into public.tournament_registration_days(id,tournament_id,label,sort_order,enabled)
    values(v_day,v_tournament::text,'Rehearsal day',0,true);
  insert into public.tournament_event_options(id,tournament_id,registration_day_id,label,event_family_label,division_name,event_type,sort_order,enabled,status,partner_required)
    values(v_event_a,v_tournament::text,v_day,'Mixed 4.0','Mixed doubles','4.0','DOUBLES',0,true,'CONFIRMED',true),
          (v_event_b,v_tournament::text,v_day,'Mixed 4.5','Mixed doubles','4.5','DOUBLES',1,true,'CONFIRMED',true);
  insert into public.tournament_event_draws(id,tournament_id,registration_day_id,event_option_id,name,status)
    values(v_draw_a,v_tournament,v_day,v_event_a,'Mixed 4.0','DRAFT'),
          (v_draw_b,v_tournament,v_day,v_event_b,'Mixed 4.5','DRAFT');
  insert into public.tournament_teams(tournament_id,draw_id,registration_day_id,event_option_id,team_number,player1_id,player2_id,club_id)
    values(v_tournament,v_draw_a,v_day,v_event_a,1,v_players[1],v_players[2],v_club),
          (v_tournament,v_draw_a,v_day,v_event_a,2,v_players[3],v_players[4],v_club),
          (v_tournament,v_draw_b,v_day,v_event_b,1,v_players[5],v_players[6],v_club),
          (v_tournament,v_draw_b,v_day,v_event_b,2,v_players[7],v_players[8],v_club);
  select array_agg(id order by id),jsonb_agg(jsonb_build_object('id',id,'updated_at',updated_at))
    into v_original_team_ids,v_teams from public.tournament_teams where tournament_id=v_tournament;
  select jsonb_agg(jsonb_build_object('id',id,'updated_at',updated_at)) into v_draws
    from public.tournament_event_draws where tournament_id=v_tournament;
  select jsonb_agg(jsonb_build_object('id',id,'updated_at',updated_at)) into v_events
    from public.tournament_event_options where tournament_id=v_tournament::text;
  v_result := public.admin_combine_tournament_draw_divisions_cas(
    v_club,v_tournament::text,v_draw_a::text,v_draws,v_teams,v_events,array[v_event_a,v_event_b],'Mixed 4.0 / 4.5');
  if (v_result->>'moved_team_count')::integer<>2
    or (select count(*) from public.tournament_teams where draw_id=v_draw_a)<>4
    or (select array_agg(id order by id) from public.tournament_teams where draw_id=v_draw_a) is distinct from v_original_team_ids
    or (select count(*) from public.tournament_teams where draw_id=v_draw_a and source_event_option_id=v_event_b)<>2
    or not exists(select 1 from public.tournament_event_draws where id=v_draw_b and combined_into_draw_id=v_draw_a and hidden_from_primary_ops)
  then raise exception 'Combined division rehearsal did not preserve the expected roster and source coverage.'; end if;
end;
$rehearsal$;
rollback;
select 'PASS: full staging schema combined four teams; all fixture writes rolled back' as result;
