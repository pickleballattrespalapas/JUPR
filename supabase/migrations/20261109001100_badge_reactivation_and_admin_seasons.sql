-- Admin-defined badge seasons and repeatable community recognition.
-- Retains every existing award; no historical award rows are changed here.
create table if not exists public.badge_seasons (
  id uuid primary key default gen_random_uuid(),
  club_id text not null references public.clubs(id),
  name text not null check (length(btrim(name)) between 1 and 100),
  start_date date not null,
  end_date date not null check (end_date >= start_date and end_date < '9999-12-31'),
  timezone text not null default 'UTC',
  revision integer not null default 1,
  created_by text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  closed_queued_at timestamptz
);
create index if not exists badge_seasons_club_dates_idx on public.badge_seasons(club_id, start_date, end_date);
alter table public.badge_seasons enable row level security;
revoke all on public.badge_seasons from public, anon, authenticated;
grant select, insert, update on public.badge_seasons to service_role;

create table if not exists public.admin_badge_operations (
  club_id text not null references public.clubs(id),
  operation_id uuid not null,
  action text not null check (action in ('award_community','save_season')),
  actor_email text not null,
  request_json jsonb not null,
  result_json jsonb not null,
  created_at timestamptz not null default now(),
  primary key (club_id, operation_id)
);
alter table public.admin_badge_operations enable row level security;
revoke all on public.admin_badge_operations from public, anon, authenticated;
grant select, insert on public.admin_badge_operations to service_role;

create or replace function public.admin_manage_badges_v1(
  p_club_id text, p_actor_email text, p_actor_user_id uuid, p_actor_role text,
  p_operation_id uuid, p_action text, p_payload jsonb
) returns jsonb language plpgsql security invoker set search_path = '' as $$
declare
  prior public.admin_badge_operations%rowtype;
  old_season public.badge_seasons%rowtype;
  saved_season public.badge_seasons%rowtype;
  award public.player_badges%rowtype;
  season_id uuid;
  start_day date;
  end_day date;
  zone text;
  selected_badge_id text;
  selected_player_id bigint;
  allowed_criteria text[];
  result jsonb;
  request jsonb;
begin
  if p_actor_role not in ('administrator','super_admin','club_owner') or p_actor_user_id is null
     or not exists (
       select 1 from public.admin_role_assignments a
       where a.club_id = p_club_id and lower(a.email) = lower(p_actor_email)
         and a.role = p_actor_role and (a.user_id is null or a.user_id = p_actor_user_id)
         and a.revoked_at is null and (a.expires_at is null or a.expires_at > now())
     ) then
    raise exception using errcode='42501', message='A current club administrator assignment is required.';
  end if;
  if p_operation_id is null or jsonb_typeof(p_payload) is distinct from 'object' then
    raise exception using errcode='22023', message='A valid award request is required.';
  end if;
  perform pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended('admin-badges:' || p_club_id, 0));
  request := jsonb_build_object('action',p_action,'payload',p_payload,'actor_user_id',p_actor_user_id,'actor_role',p_actor_role);
  select * into prior from public.admin_badge_operations o where o.club_id=p_club_id and o.operation_id=p_operation_id;
  if found then
    if prior.request_json is distinct from request or lower(prior.actor_email) is distinct from lower(p_actor_email) then
      raise exception using errcode='22023', message='This save ID belongs to a different request. Reload before starting a new award.';
    end if;
    return prior.result_json || jsonb_build_object('replayed',true);
  end if;

  if p_action='award_community' then
    selected_badge_id := p_payload->>'badge_id';
    selected_player_id := (p_payload->>'player_id')::bigint;
    allowed_criteria := case selected_badge_id
      when 'good_sport' then array['honest_calls','respectful_resolution','encouragement']
      when 'community_builder' then array['welcome_newcomers','volunteer','inclusive_play','introduce_participants']
      when 'mentor' then array['ongoing_help','lead_learning','rules_and_etiquette']
      else null end;
    if allowed_criteria is null or not exists (select 1 from public.players p where p.id=selected_player_id and p.club_id=p_club_id)
       or not exists (select 1 from public.badges b where b.badge_id=selected_badge_id and b.is_active and coalesce(b.state,'live')='live') then
      raise exception using errcode='22023', message='Choose an available community badge and a player from this club.';
    end if;
    if jsonb_typeof(p_payload->'criteria') is distinct from 'array' or jsonb_array_length(p_payload->'criteria')=0
       or exists (select 1 from jsonb_array_elements_text(p_payload->'criteria') c where not c=any(allowed_criteria))
       or length(btrim(coalesce(p_payload->>'note',''))) not between 1 and 1000
       or (p_payload->>'contribution_date') is null
       or (p_payload->>'contribution_date')::date > (now() at time zone 'UTC')::date then
      raise exception using errcode='22023', message='Choose a qualifying action, a contribution date, and a recognition note.';
    end if;
    insert into public.player_badges(id,club_id,player_id,badge_id,earned_at,context_type,context_id,match_id,value_json,awarded_by,rule_version)
    values (p_operation_id,p_club_id,selected_player_id,selected_badge_id,now(),'overall','community-recognition:'||p_operation_id,null,
      jsonb_build_object('recognition_id',p_operation_id,'criteria',p_payload->'criteria','recognition_note',btrim(p_payload->>'note'),
                        'contribution_date',p_payload->>'contribution_date'),p_actor_email,'badge-reactivation-v1') returning * into award;
    result := jsonb_build_object('ok',true,'award',to_jsonb(award));
  elsif p_action='save_season' then
    season_id := (p_payload->>'id')::uuid;
    start_day := (p_payload->>'start_date')::date;
    end_day := (p_payload->>'end_date')::date;
    zone := p_payload->>'timezone';
    if season_id is null or start_day is null or end_day is null or end_day < start_day or end_day >= '9999-12-31'
       or length(btrim(coalesce(p_payload->>'name',''))) not between 1 and 100
       or not exists (select 1 from pg_catalog.pg_timezone_names z where z.name=zone) then
      raise exception using errcode='22023', message='Enter a season name, valid dates, and a timezone.';
    end if;
    select * into old_season from public.badge_seasons s where s.club_id=p_club_id and s.id=season_id for update;
    if coalesce(old_season.revision,0) is distinct from (p_payload->>'expected_revision')::integer then
      raise exception using errcode='40001', message='This season changed. Reload it before saving.';
    end if;
    if old_season.id is not null and (old_season.start_date,old_season.end_date,old_season.timezone) is distinct from (start_day,end_day,zone)
       and exists (select 1 from public.player_badges b where b.club_id=p_club_id and b.context_type='season' and b.context_id='badge-season:'||season_id) then
      raise exception using errcode='22023', message='Season dates cannot change after badges have been awarded. The name can still be updated.';
    end if;
    if exists (select 1 from public.badge_seasons s where s.club_id=p_club_id and s.id<>season_id
      and (start_day::timestamp at time zone zone) < ((s.end_date+1)::timestamp at time zone s.timezone)
      and ((end_day+1)::timestamp at time zone zone) > (s.start_date::timestamp at time zone s.timezone)) then
      raise exception using errcode='22023', message='Badge seasons for this club cannot overlap.';
    end if;
    if old_season.id is null then
      insert into public.badge_seasons(id,club_id,name,start_date,end_date,timezone,created_by)
      values(season_id,p_club_id,btrim(p_payload->>'name'),start_day,end_day,zone,p_actor_email) returning * into saved_season;
    else
      update public.badge_seasons s set name=btrim(p_payload->>'name'),start_date=start_day,end_date=end_day,timezone=zone,
        revision=s.revision+1,updated_at=now(),closed_queued_at=case when (s.start_date,s.end_date,s.timezone) is distinct from (start_day,end_day,zone) then null else s.closed_queued_at end
      where s.id=season_id and s.club_id=p_club_id returning * into saved_season;
    end if;
    result := jsonb_build_object('ok',true,'season',to_jsonb(saved_season));
    insert into public.badge_eval_queue(club_id,context_id,event_type,player_ids,payload_json,status)
      select p_club_id,'overall','season_changed',coalesce(array_agg(p.id),array[]::bigint[]),jsonb_build_object('season_id',season_id),'pending'
      from public.players p where p.club_id=p_club_id;
  else
    raise exception using errcode='22023', message='Unknown badge action.';
  end if;
  insert into public.admin_activity_log(club_id,actor_email,actor_role,action_type,entity_type,entity_id,before_json,after_json,source_page,flagged_for_review)
  values(p_club_id,p_actor_email,p_actor_role,p_action,'badges',p_operation_id::text,
    case when old_season.id is not null then to_jsonb(old_season) else null end,result,'badge_management',false);
  insert into public.admin_badge_operations(club_id,operation_id,action,actor_email,request_json,result_json)
  values(p_club_id,p_operation_id,p_action,p_actor_email,request,result);
  return result;
end;
$$;
revoke all on function public.admin_manage_badges_v1(text,text,uuid,text,uuid,text,jsonb) from public,anon,authenticated;
grant execute on function public.admin_manage_badges_v1(text,text,uuid,text,uuid,text,jsonb) to service_role;

create or replace function public.enqueue_completed_badge_seasons_v1(p_club_id text)
returns integer language plpgsql security invoker set search_path='' as $$
declare s public.badge_seasons%rowtype; queued integer := 0;
begin
  for s in select * from public.badge_seasons b where b.club_id=p_club_id and b.closed_queued_at is null
    and ((b.end_date+1)::timestamp at time zone b.timezone)<=now() for update skip locked loop
    insert into public.badge_eval_queue(club_id,context_id,event_type,player_ids,payload_json,status)
      select p_club_id,'overall','season_closed',coalesce(array_agg(p.id),array[]::bigint[]),jsonb_build_object('season_id',s.id),'pending'
      from public.players p where p.club_id=p_club_id;
    update public.badge_seasons set closed_queued_at=now() where id=s.id;
    queued := queued + 1;
  end loop;
  return queued;
end;
$$;
revoke all on function public.enqueue_completed_badge_seasons_v1(text) from public,anon,authenticated;
grant execute on function public.enqueue_completed_badge_seasons_v1(text) to service_role;

-- Some installations seeded only previously live badges. Add missing definitions.
insert into public.badges(badge_id,name,prestige,category,is_stackable,is_active,rarity,tier,icon_key,lore,hint,scope,state,eval_triggers) values
  ('breakthrough', 'Breakthrough', 55, 'Improvement', true, true, 'epic', null, 'breakthrough', 'In a league, play 10+ matches and move from outside the top 25 / top 10 (starting rank) into the current top 25 / top 10 (by JUPR rating). Earn each tier separately.', 'In a league, play 10+ matches and move from outside the top 25 / top 10 (starting rank) into the current top 25 / top 10 (by JUPR rating). Earn each tier separately.', 'league', 'live', '["match_recorded","match_updated"]'::jsonb),
  ('above_expectations', 'Above Expectations', 50, 'Match Achievements', true, true, 'rare', null, 'above_expectations', 'Win with expected win chance ≤ 40% and a 4+ point margin. If rating deltas are available, the match must be at or above the 75th percentile of absolute rating delta within that league.', 'Win with expected win chance ≤ 40% and a 4+ point margin. If rating deltas are available, the match must be at or above the 75th percentile of absolute rating delta within that league.', 'match', 'live', '["match_recorded","match_updated"]'::jsonb),
  ('clutch_performer', 'Clutch Performer', 60, 'Match Achievements', false, true, 'rare', null, 'clutch_performer', 'Record 5 clutch wins (wins by 2 points or fewer).', 'Record 5 clutch wins (wins by 2 points or fewer).', 'overall', 'live', '["match_recorded","match_updated"]'::jsonb),
  ('dominant_run', 'Dominant Run', 45, 'Match Achievements', true, true, 'rare', null, 'dominant_run', 'Across all leagues, reach a 10+ match win streak with average win margin ≥ 5 across the current streak. Earn it again for each additional win while the streak and average margin stay at or above the threshold.', 'Across all leagues, reach a 10+ match win streak with average win margin ≥ 5 across the current streak. Earn it again for each additional win while the streak and average margin stay at or above the threshold.', 'overall', 'live', '["match_recorded","match_updated"]'::jsonb),
  ('high_output', 'High Output', 40, 'Match Achievements', true, true, 'common', null, 'high_output', 'Across all leagues, in your last 25 matches, record 20+ wins with average margin ≥ 4.', 'Across all leagues, in your last 25 matches, record 20+ wins with average margin ≥ 4.', 'overall', 'live', '["match_recorded","match_updated"]'::jsonb),
  ('nemesis_found', 'Nemesis Found', 55, 'Match Achievements', false, true, 'rare', null, 'nemesis_found', 'Face an opponent at least 6 times with a head-to-head win rate of 40% or less. Earned once, when your first nemesis is identified.', 'Face an opponent at least 6 times with a head-to-head win rate of 40% or less. Earned once, when your first nemesis is identified.', 'overall', 'live', '["match_recorded","match_updated"]'::jsonb),
  ('rivalry_win', 'Rivalry Win', 60, 'Match Achievements', true, true, 'rare', null, 'rivalry_win', 'Beat an opponent previously identified as your nemesis. Earned for each win against each nemesis.', 'Beat an opponent previously identified as your nemesis. Earned for each win against each nemesis.', 'opponent', 'live', '["match_recorded","match_updated"]'::jsonb),
  ('rivalry_streak', 'Rivalry Streak', 70, 'Match Achievements', true, true, 'epic', null, 'rivalry_streak', 'After an opponent becomes your nemesis, win 3 consecutive head-to-head matches against them. Earned once per winning streak; a loss starts a new streak.', 'After an opponent becomes your nemesis, win 3 consecutive head-to-head matches against them. Earned once per winning streak; a loss starts a new streak.', 'opponent', 'live', '["match_recorded","match_updated"]'::jsonb),
  ('settled_the_score', 'Settled the Score', 65, 'Match Achievements', true, true, 'epic', null, 'settled_the_score', 'Beat a previously identified nemesis to bring your head-to-head wins and losses exactly level. The opponent remains tracked after your record improves.', 'Beat a previously identified nemesis to bring your head-to-head wins and losses exactly level. The opponent remains tracked after your record improves.', 'opponent', 'live', '["match_recorded","match_updated"]'::jsonb),
  ('battle_tested', 'Battle Tested', 50, 'Participation', true, true, 'rare', null, 'battle_tested', 'During a season set by a club admin, complete 50+ eligible recorded matches. Earned once per season; forfeits and invalid matches do not count.', 'During a season set by a club admin, complete 50+ eligible recorded matches. Earned once per season; forfeits and invalid matches do not count.', 'season', 'live', '["match_recorded","match_updated"]'::jsonb),
  ('consistency', 'Consistency', 60, 'Participation', true, true, 'epic', null, 'consistency', 'During a season set by a club admin, play at least one match in each of 6 consecutive ISO weeks (Monday–Sunday). Missing a week breaks the streak. Earned once per season.', 'During a season set by a club admin, play at least one match in each of 6 consecutive ISO weeks (Monday–Sunday). Missing a week breaks the streak. Earned once per season.', 'season', 'live', '["match_recorded","match_updated"]'::jsonb),
  ('mr_reliable', 'Mr. Reliable', 80, 'Match Achievements', true, true, 'epic', null, 'mr_reliable', 'In a season set by a club admin, play 30+ matches and finish with a 70%+ win rate. Awarded after the season ends when results are next processed or staff runs a badge check.', 'In a season set by a club admin, play 30+ matches and finish with a 70%+ win rate. Awarded after the season ends when results are next processed or staff runs a badge check.', 'season', 'live', '["match_recorded","match_updated"]'::jsonb),
  ('good_sport', 'Good Sport', 35, 'Partnerships', true, true, 'rare', null, 'good_sport', 'Awarded by a club admin for honest calls even when they cost a point, respectful handling of a difficult situation, or consistently encouraging and supporting other players. Can be awarded again for a separate contribution.', 'Awarded by a club admin for honest calls even when they cost a point, respectful handling of a difficult situation, or consistently encouraging and supporting other players. Can be awarded again for a separate contribution.', 'overall', 'live', '["match_recorded","match_updated"]'::jsonb),
  ('community_builder', 'Community Builder', 90, 'Partnerships', true, true, 'epic', null, 'community_builder', 'Awarded by a club admin for welcoming newcomers and helping them find games, volunteering at club events, organizing inclusive social play, or introducing new participants to the club. Can be awarded again for a separate contribution.', 'Awarded by a club admin for welcoming newcomers and helping them find games, volunteering at club events, organizing inclusive social play, or introducing new participants to the club. Can be awarded again for a separate contribution.', 'overall', 'live', '["match_recorded","match_updated"]'::jsonb),
  ('mentor', 'Mentor', 70, 'Partnerships', true, true, 'epic', null, 'mentor', 'Awarded by a club admin for helping a newer player improve over several visits, leading beginner lessons or practice sessions, or helping players learn rules and court etiquette. Can be awarded again for a separate contribution.', 'Awarded by a club admin for helping a newer player improve over several visits, leading beginner lessons or practice sessions, or helping players learn rules and court etiquette. Can be awarded again for a separate contribution.', 'overall', 'live', '["match_recorded","match_updated"]'::jsonb)
on conflict (badge_id) do nothing;

-- Catalog availability and plain earning requirements. Frozen definitions stay frozen.
with definitions(badge_id,category,requirement,is_active,is_stackable,scope) as (values
  ('breakthrough', 'Improvement', 'In a league, play 10+ matches and move from outside the top 25 / top 10 (starting rank) into the current top 25 / top 10 (by JUPR rating). Earn each tier separately.', true, true, 'league'),
  ('above_expectations', 'Match Achievements', 'Win with expected win chance ≤ 40% and a 4+ point margin. If rating deltas are available, the match must be at or above the 75th percentile of absolute rating delta within that league.', true, true, 'match'),
  ('clutch_performer', 'Match Achievements', 'Record 5 clutch wins (wins by 2 points or fewer).', true, false, 'overall'),
  ('dominant_run', 'Match Achievements', 'Across all leagues, reach a 10+ match win streak with average win margin ≥ 5 across the current streak. Earn it again for each additional win while the streak and average margin stay at or above the threshold.', true, true, 'overall'),
  ('high_output', 'Match Achievements', 'Across all leagues, in your last 25 matches, record 20+ wins with average margin ≥ 4.', true, true, 'overall'),
  ('nemesis_found', 'Match Achievements', 'Face an opponent at least 6 times with a head-to-head win rate of 40% or less. Earned once, when your first nemesis is identified.', true, false, 'overall'),
  ('rivalry_win', 'Match Achievements', 'Beat an opponent previously identified as your nemesis. Earned for each win against each nemesis.', true, true, 'opponent'),
  ('rivalry_streak', 'Match Achievements', 'After an opponent becomes your nemesis, win 3 consecutive head-to-head matches against them. Earned once per winning streak; a loss starts a new streak.', true, true, 'opponent'),
  ('settled_the_score', 'Match Achievements', 'Beat a previously identified nemesis to bring your head-to-head wins and losses exactly level. The opponent remains tracked after your record improves.', true, true, 'opponent'),
  ('battle_tested', 'Participation', 'During a season set by a club admin, complete 50+ eligible recorded matches. Earned once per season; forfeits and invalid matches do not count.', true, true, 'season'),
  ('consistency', 'Participation', 'During a season set by a club admin, play at least one match in each of 6 consecutive ISO weeks (Monday–Sunday). Missing a week breaks the streak. Earned once per season.', true, true, 'season'),
  ('steady_hand', 'Match Achievements', 'In a season set by a club admin, reach 20+ matches with a win rate of 60%+. Earned once per season; later losses do not remove it.', true, true, 'season'),
  ('mr_reliable', 'Match Achievements', 'In a season set by a club admin, play 30+ matches and finish with a 70%+ win rate. Awarded after the season ends when results are next processed or staff runs a badge check.', true, true, 'season'),
  ('league_champion', 'Trophies', 'Retired. Existing awards remain in player history; current league awards and tournament podiums provide trophies.', false, false, 'league'),
  ('league_runner_up', 'Trophies', 'Retired. Existing awards remain in player history; current league awards and tournament podiums provide trophies.', false, false, 'league'),
  ('league_third_place', 'Trophies', 'Retired. Existing awards remain in player history; current league awards and tournament podiums provide trophies.', false, false, 'league'),
  ('podium', 'Trophies', 'Retired. Existing awards remain in player history; current league awards and tournament podiums provide trophies.', false, false, 'league'),
  ('good_sport', 'Partnerships', 'Awarded by a club admin for honest calls even when they cost a point, respectful handling of a difficult situation, or consistently encouraging and supporting other players. Can be awarded again for a separate contribution.', true, true, 'overall'),
  ('community_builder', 'Partnerships', 'Awarded by a club admin for welcoming newcomers and helping them find games, volunteering at club events, organizing inclusive social play, or introducing new participants to the club. Can be awarded again for a separate contribution.', true, true, 'overall'),
  ('mentor', 'Partnerships', 'Awarded by a club admin for helping a newer player improve over several visits, leading beginner lessons or practice sessions, or helping players learn rules and court etiquette. Can be awarded again for a separate contribution.', true, true, 'overall')
)
update public.badges b set category=d.category,lore=d.requirement,hint=d.requirement,is_active=d.is_active,
  is_stackable=d.is_stackable,scope=d.scope,
  state=case when not d.is_active then 'deprecated' when b.state='frozen' then 'frozen' else 'live' end,
  state_change_reason='badge-reactivation-v1: reviewed badge criteria and legacy trophy retirement',state_changed_at=now()
from definitions d where b.badge_id=d.badge_id;

-- Mirror only the optional compatibility columns present in this installation.
do $$
declare col text;
begin
  foreach col in array array['name','prestige','category','lore','hint','is_active','is_stackable','scope'] loop
    if exists(select 1 from information_schema.columns where table_schema='public' and table_name='badges' and column_name=col||'_v2') then
      execute format('update public.badges set %I=%I where badge_id=any($1)',col||'_v2',col) using array['above_expectations','battle_tested','breakthrough','clutch_performer','community_builder','consistency','dominant_run','good_sport','high_output','league_champion','league_runner_up','league_third_place','mentor','mr_reliable','nemesis_found','podium','rivalry_streak','rivalry_win','settled_the_score','steady_hand'];
    end if;
  end loop;
end;
$$;

-- Coordinate date edits with concurrent award inserts; reject stale calculations.
create or replace function public.validate_badge_season_award_v1()
returns trigger language plpgsql security invoker set search_path='' as $$
declare s public.badge_seasons%rowtype;
begin
  if new.context_type='season' and new.context_id like 'badge-season:%' then
    select * into s from public.badge_seasons where id=substring(new.context_id from 14)::uuid and club_id=new.club_id for share;
    if not found or new.value_json->>'season_start' is distinct from s.start_date::text
      or new.value_json->>'season_end' is distinct from s.end_date::text
      or new.value_json->>'season_timezone' is distinct from s.timezone then
      raise exception using errcode='40001', message='Badge season dates changed; recompute this award before saving.';
    end if;
  end if;
  return jsonb_populate_record(new, jsonb_build_object(
    'badge_id_v2',new.badge_id,'club_id_v2',new.club_id,'context_type_v2',new.context_type,
    'earned_at_v2',new.earned_at,'player_id_v2',new.player_id));
end;
$$;
revoke all on function public.validate_badge_season_award_v1() from public,anon,authenticated;
grant execute on function public.validate_badge_season_award_v1() to service_role;
drop trigger if exists validate_badge_season_award on public.player_badges;
create trigger validate_badge_season_award before insert on public.player_badges for each row execute function public.validate_badge_season_award_v1();
