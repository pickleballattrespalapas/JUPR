-- Extends the already sequenced November badge migrations. Created with the
-- Supabase CLI and ordered after reactivation, on which the catalog depends.
-- All writes are service-role-only. Source triggers only record dirty revisions.
create table public.badge_program_state (
  club_id text primary key,
  revision bigint not null default 1,
  applied_revision bigint not null default 0,
  pending_ties jsonb not null default '[]',
  review jsonb not null default '[]',
  checked_at timestamptz,
  retry_after timestamptz not null default now(),
  last_error text
);
create index badge_program_pending_idx on public.badge_program_state(retry_after,club_id)
  where applied_revision < revision;
create table public.badge_program_finalizations (
  club_id text not null, source_type text not null check(source_type in ('league','tournament','round_robin')),
  source_id text not null, completed_at timestamptz not null,
  min_games integer check(min_games >= 0), evidence jsonb not null default '{}',
  primary key(club_id,source_type,source_id)
);
create table public.badge_round_robin_decisions (
  club_id text not null, source_key text not null, source_fingerprint text not null check(length(source_fingerprint)=64),
  winner_player_id bigint not null, actor_user_id uuid not null, actor_email text not null,
  decided_at timestamptz not null default now(), evidence jsonb not null,
  primary key(club_id,source_key)
);
create table public.badge_round_robin_operations (
  club_id text not null, operation_id uuid not null, actor_user_id uuid not null,
  payload jsonb not null, result jsonb not null, created_at timestamptz not null default now(),
  primary key(club_id,operation_id)
);

alter table public.badge_program_state enable row level security;
alter table public.badge_program_finalizations enable row level security;
alter table public.badge_round_robin_decisions enable row level security;
alter table public.badge_round_robin_operations enable row level security;
revoke all on public.badge_program_state, public.badge_program_finalizations,
  public.badge_round_robin_decisions, public.badge_round_robin_operations from public, anon, authenticated;
grant select,insert,update,delete on public.badge_program_state, public.badge_program_finalizations,
  public.badge_round_robin_decisions, public.badge_round_robin_operations to service_role;

-- Minimal privileged trigger: existing authorized source writes must not need
-- direct access to internal badge state. Fixed search_path, no public execution.
create function public.mark_program_badges_dirty_v1() returns trigger
language plpgsql security definer set search_path = '' as $$
declare r jsonb; old_r jsonb; club text; old_club text; source_kind text;
begin
  if tg_op <> 'DELETE' then r := to_jsonb(new); else r := to_jsonb(old); end if;
  if tg_op = 'UPDATE' then
    old_r := to_jsonb(old);
    if (r - array['updated_at','last_seen_at','expires_at']) = (old_r - array['updated_at','last_seen_at','expires_at']) then return new; end if;
  end if;
  club := r->>'club_id';
  old_club := old_r->>'club_id';
  if tg_table_name in ('live_event_participants','live_event_matches') then
    select e.club_id into club from public.live_events e where e.id::text=r->>'event_id';
    if old_r is not null then
      select e.club_id into old_club from public.live_events e where e.id::text=old_r->>'event_id';
    end if;
  elsif tg_table_name = 'tournament_event_draws' then
    select t.club_id into club from public.tournaments t where t.id::text=r->>'tournament_id';
    if old_r is not null then
      select t.club_id into old_club from public.tournaments t where t.id::text=old_r->>'tournament_id';
    end if;
  end if;
  if club is null then return coalesce(new,old); end if;
  insert into public.badge_program_state(club_id) values(club)
  on conflict(club_id) do update set revision=public.badge_program_state.revision+1,retry_after=now();
  if old_club is not null and old_club <> club then
    insert into public.badge_program_state(club_id) values(old_club)
    on conflict(club_id) do update set revision=public.badge_program_state.revision+1,retry_after=now();
  end if;
  -- Capture closure facts once. Cosmetic changes cannot move an earning date
  -- or substitute today's league minimum for the minimum at closure.
  if tg_op <> 'DELETE' then
    if tg_table_name='leagues_metadata' and r->>'status'='ended' then
      insert into public.badge_program_finalizations(club_id,source_type,source_id,completed_at,min_games,evidence)
      select club,'league',r->>'id',(r->>'ended_at')::timestamptz,(r->>'min_games')::integer,
             jsonb_build_object('league_name',r->>'league_name','source','closure')
      where r->>'ended_at' is not null and r->>'min_games' is not null
      on conflict do nothing;
    elsif tg_table_name='tournaments' and upper(r->>'status')='COMPLETED'
      and coalesce(upper(old_r->>'status'),'') not in ('COMPLETED','ARCHIVED') then
      insert into public.badge_program_finalizations(club_id,source_type,source_id,completed_at,evidence)
      values(club,'tournament',r->>'id',now(),jsonb_build_object('source','closure')) on conflict do nothing;
    elsif tg_table_name='live_sessions' and r->>'status'='completed'
      and coalesce(old_r->>'status','') <> 'completed' then
      insert into public.badge_program_finalizations(club_id,source_type,source_id,completed_at,evidence)
      values(club,'round_robin',r->>'id',coalesce((r->>'completed_at')::timestamptz,now()),jsonb_build_object('source','closure')) on conflict do nothing;
    end if;
  end if;
  return coalesce(new,old);
end $$;
revoke all on function public.mark_program_badges_dirty_v1() from public,anon,authenticated;

-- Every authoritative score, roster, moderation, exclusion or closure change
-- leaves durable work, including changes made outside this API process.
do $$ declare tbl text; begin
  foreach tbl in array array['matches','players','leagues_metadata','tournaments','tournament_games','tournament_teams',
     'tournament_podium','tournament_event_draws','live_events','live_sessions','live_event_participants','live_event_matches'] loop
    execute format('create trigger program_badges_dirty after insert or update or delete on public.%I for each row execute function public.mark_program_badges_dirty_v1()',tbl);
  end loop;
end $$;

-- Historical evidence must actually exist. Do not infer a tournament completion
-- time from an end-date or its last game, nor treat archived as completed alone.
insert into public.badge_program_finalizations(club_id,source_type,source_id,completed_at,min_games,evidence)
select club_id,'league',id::text,ended_at,min_games,jsonb_build_object('source','recorded_league_closure','league_name',league_name)
from public.leagues_metadata where status='ended' and ended_at is not null and min_games is not null;
insert into public.badge_program_finalizations(club_id,source_type,source_id,completed_at,evidence)
select club_id,'tournament',tournament_id::text,min(created_at),jsonb_build_object('source','tournament_lifecycle_receipt')
from public.tournament_lifecycle_receipts where action='complete' group by club_id,tournament_id;
insert into public.badge_program_finalizations(club_id,source_type,source_id,completed_at,evidence)
select club_id,'round_robin',id::text,completed_at,jsonb_build_object('source','recorded_session_closure')
from public.live_sessions where status='completed' and completed_at is not null;
insert into public.badge_program_state(club_id) select distinct club_id from public.players where club_id is not null on conflict do nothing;

create function public.pending_program_badge_clubs_v1(p_limit integer default 5)
returns table(club_id text) language sql stable security invoker set search_path='' as $$
  select s.club_id from public.badge_program_state s where s.applied_revision < s.revision and s.retry_after <= now()
  order by s.retry_after,s.club_id limit greatest(1,least(p_limit,20))
$$;
create function public.program_badge_failure_v1(p_club_id text) returns void
language sql security invoker set search_path='' as $$
  update public.badge_program_state set last_error='Evaluation failed; will retry. Check API logs.',retry_after=now()+interval '1 minute' where club_id=p_club_id
$$;

create function public.badge_program_snapshot_v1(p_club_id text) returns jsonb
language sql stable security invoker set search_path='' as $$
 select jsonb_build_object(
  'club_id',p_club_id,
  'revision',(select revision from public.badge_program_state where club_id=p_club_id),
  'as_of',now(),
  'players',coalesce((select jsonb_agg(jsonb_build_object('id',r.id,'name',r.name,'club_id',r.club_id) order by r.id) from public.players r where r.club_id=p_club_id),'[]'::jsonb),
  'matches',coalesce((select jsonb_agg(to_jsonb(r) order by r.id) from public.matches r where r.club_id=p_club_id),'[]'::jsonb),
  'leagues_metadata',coalesce((select jsonb_agg(to_jsonb(r) order by r.id) from public.leagues_metadata r where r.club_id=p_club_id),'[]'::jsonb),
  'tournaments',coalesce((select jsonb_agg(to_jsonb(r) order by r.id) from public.tournaments r where r.club_id=p_club_id),'[]'::jsonb),
  'tournament_games',coalesce((select jsonb_agg(to_jsonb(r) order by r.id) from public.tournament_games r where r.club_id=p_club_id),'[]'::jsonb),
  'tournament_teams',coalesce((select jsonb_agg(to_jsonb(r) order by r.id) from public.tournament_teams r where r.club_id=p_club_id),'[]'::jsonb),
  'tournament_podium',coalesce((select jsonb_agg(to_jsonb(r) order by r.id) from public.tournament_podium r where r.club_id=p_club_id),'[]'::jsonb),
  'tournament_event_draws',coalesce((select jsonb_agg(to_jsonb(r) order by r.id) from public.tournament_event_draws r where exists(select 1 from public.tournaments t where t.id=r.tournament_id and t.club_id=p_club_id)),'[]'::jsonb),
  'live_events',coalesce((select jsonb_agg(to_jsonb(r) order by r.id) from public.live_events r where r.club_id=p_club_id),'[]'::jsonb),
  'live_sessions',coalesce((select jsonb_agg(jsonb_build_object('id',r.id,'source',r.source,'status',r.status,'completed_at',r.completed_at,'state',jsonb_build_object('event',coalesce(r.state->'event',r.state->'page_state'->'event'),'official_publish',r.state->'official_publish') ) order by r.id) from public.live_sessions r where r.club_id=p_club_id),'[]'::jsonb),
  'live_event_participants',coalesce((select jsonb_agg(to_jsonb(r) order by r.id) from public.live_event_participants r where exists(select 1 from public.live_events e where e.id=r.event_id and e.club_id=p_club_id)),'[]'::jsonb),
  'live_event_matches',coalesce((select jsonb_agg(to_jsonb(r) order by r.id) from public.live_event_matches r where exists(select 1 from public.live_events e where e.id=r.event_id and e.club_id=p_club_id)),'[]'::jsonb),
  'finalizations',coalesce((select jsonb_agg(to_jsonb(r)) from public.badge_program_finalizations r where r.club_id=p_club_id),'[]'::jsonb),
  'decisions',coalesce((select jsonb_agg(to_jsonb(r)) from public.badge_round_robin_decisions r where r.club_id=p_club_id),'[]'::jsonb)
 )
$$;

create function public.apply_program_badges_v1(p_club_id text,p_revision bigint,p_awards jsonb,p_pending_ties jsonb,p_review jsonb)
returns jsonb language plpgsql security invoker set search_path='' as $$
declare current_revision bigint; v_award jsonb; inserted integer := 0; restored integer := 0; revoked integer := 0; changed integer;
        current_award public.player_badges%rowtype;
begin
  select revision into current_revision from public.badge_program_state where club_id=p_club_id for update;
  if current_revision is null or current_revision is distinct from p_revision then raise exception using errcode='40001',message='Program results changed. Read a fresh snapshot.'; end if;
  if jsonb_typeof(p_awards) is distinct from 'array' or jsonb_typeof(p_pending_ties) is distinct from 'array' or jsonb_typeof(p_review) is distinct from 'array' then
    raise exception using errcode='22023',message='Invalid program badge result.';
  end if;
  if exists(select 1 from jsonb_array_elements(p_awards) a where a->>'badge_id' not in ('leagues_completed_5','leagues_completed_10','leagues_completed_25','tournaments_completed_5','tournaments_completed_10','tournaments_completed_25','round_robins_completed_5','round_robins_completed_10','round_robins_completed_25','matches_together_10','matches_together_25','matches_together_50','wins_together_10','wins_together_25','wins_together_50','five_winning_partners','triple_crown','round_robin_wins_1','round_robin_wins_5','round_robin_wins_10','round_robin_wins_25','round_robin_wins_50')
      or a->>'context_type' <> 'overall' or nullif(a->>'context_id','') is null
      or (a->>'earned_at')::timestamptz > now() or (a->>'earned_at') is null
      or not exists(select 1 from public.players p where p.club_id=p_club_id and p.id=(a->>'player_id')::bigint)) then
    raise exception using errcode='22023',message='Invalid badge, player, date or context.';
  end if;
  if exists(select 1 from jsonb_array_elements(p_awards) a group by a->>'badge_id',a->>'player_id',a->>'context_id' having count(*)>1) then
    raise exception using errcode='22023',message='Duplicate program award identity.';
  end if;
  for v_award in select x from jsonb_array_elements(p_awards) x join public.badges b on b.badge_id=x->>'badge_id'
      where b.is_active and b.state='live' loop
    select * into current_award from public.player_badges where club_id=p_club_id and player_id=(v_award->>'player_id')::bigint
      and badge_id=v_award->>'badge_id' and context_type='overall' and context_id=v_award->>'context_id' for update;
    if not found then
      insert into public.player_badges(id,club_id,player_id,badge_id,context_type,context_id,earned_at,value_num,value_json,awarded_by,rule_version)
      values(gen_random_uuid(),p_club_id,(v_award->>'player_id')::bigint,v_award->>'badge_id','overall',v_award->>'context_id',
             (v_award->>'earned_at')::timestamptz,(v_award->>'value_num')::numeric,v_award->'value_json','engine','program-badges-v1');
      inserted := inserted+1;
    elsif current_award.rule_version='program-badges-v1' and (current_award.revoked_at is null or current_award.revoked_by='program_badge_engine') then
      if current_award.revoked_at is not null then restored := restored+1; end if;
      update public.player_badges set earned_at=(v_award->>'earned_at')::timestamptz,value_num=(v_award->>'value_num')::numeric,value_json=v_award->'value_json',
        revoked_at=null,revoked_by=null,revoked_reason=null,revoke_reason=null
      where id=current_award.id and (earned_at is distinct from (v_award->>'earned_at')::timestamptz or value_json is distinct from v_award->'value_json' or revoked_at is not null);
    end if;
  end loop;
  -- This reconciliation owns only the 22 new badge types. Manual revocations,
  -- the earlier historical review set and every legacy trophy remain untouched.
  update public.player_badges pb set revoked_at=now(),revoked_by='program_badge_engine',
      revoked_reason='Corrected program results no longer meet the requirement.',revoke_reason='Corrected program results no longer meet the requirement.'
  where pb.club_id=p_club_id and pb.badge_id in ('leagues_completed_5','leagues_completed_10','leagues_completed_25','tournaments_completed_5','tournaments_completed_10','tournaments_completed_25','round_robins_completed_5','round_robins_completed_10','round_robins_completed_25','matches_together_10','matches_together_25','matches_together_50','wins_together_10','wins_together_25','wins_together_50','five_winning_partners','triple_crown','round_robin_wins_1','round_robin_wins_5','round_robin_wins_10','round_robin_wins_25','round_robin_wins_50') and pb.rule_version='program-badges-v1' and pb.revoked_at is null
    and exists(select 1 from public.badges b where b.badge_id=pb.badge_id and b.is_active and b.state='live')
    and not exists(select 1 from jsonb_array_elements(p_awards) a where (a->>'player_id')::bigint=pb.player_id
                   and a->>'badge_id'=pb.badge_id and a->>'context_id'=pb.context_id and a->>'context_type'=pb.context_type);
  get diagnostics revoked=row_count;
  update public.badge_program_state set applied_revision=p_revision,pending_ties=p_pending_ties,review=p_review,checked_at=now(),last_error=null where club_id=p_club_id;
  return jsonb_build_object('ok',true,'inserted',inserted,'restored',restored,'revoked',revoked,'revision',p_revision);
end $$;

create function public.admin_resolve_round_robin_v1(p_club_id text,p_actor_email text,p_actor_user_id uuid,
 p_actor_role text,p_operation_id uuid,p_payload jsonb) returns jsonb
language plpgsql security invoker set search_path='' as $$
declare state public.badge_program_state%rowtype; op public.badge_round_robin_operations%rowtype;
        tie jsonb; result jsonb; display_decision jsonb; winner bigint := (p_payload->>'winner_player_id')::bigint;
begin
  if p_actor_role not in ('administrator','super_admin','club_owner') or not exists (
    select 1 from public.admin_role_assignments a where a.club_id=p_club_id and a.email=lower(trim(p_actor_email))
      and a.role=p_actor_role and a.revoked_at is null and (a.expires_at is null or a.expires_at>now())
      and (a.user_id is null or a.user_id=p_actor_user_id)) then
    raise exception using errcode='42501',message='A current club administrator assignment is required.';
  end if;
  select * into state from public.badge_program_state where club_id=p_club_id for update;
  select * into op from public.badge_round_robin_operations where club_id=p_club_id and operation_id=p_operation_id;
  if found then
    if op.actor_user_id is distinct from p_actor_user_id or op.payload is distinct from p_payload then
      raise exception using errcode='40001',message='This save request was already used for another decision.';
    end if;
    return op.result;
  end if;
  if state.revision is distinct from state.applied_revision then
    raise exception using errcode='40001',message='Round-robin results changed. Refresh after the badge check finishes.';
  end if;
  select x into tie from jsonb_array_elements(state.pending_ties) x where x->>'source_key'=p_payload->>'source_key'
     and x->>'source_fingerprint'=p_payload->>'source_fingerprint';
  if tie is null or winner is null or not exists(select 1 from jsonb_array_elements(tie->'leaders') l where (l->>'player_id')::bigint=winner)
     or not exists(select 1 from public.players p where p.id=winner and p.club_id=p_club_id) then
    raise exception using errcode='40001',message='Choose a linked player among the current tied leaders. Refresh to load the current result.';
  end if;
  insert into public.badge_round_robin_decisions(club_id,source_key,source_fingerprint,winner_player_id,actor_user_id,actor_email,evidence)
  values(p_club_id,p_payload->>'source_key',p_payload->>'source_fingerprint',winner,p_actor_user_id,lower(trim(p_actor_email)),tie)
  on conflict(club_id,source_key) do update set source_fingerprint=excluded.source_fingerprint,winner_player_id=excluded.winner_player_id,
    actor_user_id=excluded.actor_user_id,actor_email=excluded.actor_email,decided_at=now(),evidence=excluded.evidence;
  -- Store a signed-by-results display decision with both durable forms of the
  -- session. The scorer ignores it automatically after any score/roster change.
  select jsonb_build_object('participant_id',l->>'participant_id','display_fingerprint',tie->>'display_fingerprint')
    into display_decision from jsonb_array_elements(tie->'leaders') l where (l->>'player_id')::bigint=winner;
  update public.live_events set raw_event_json=jsonb_set(raw_event_json,'{round_robin_winner}',display_decision),updated_at=now()
    where club_id=p_club_id and 'live:'||source_event_uid=p_payload->>'source_key';
  update public.live_sessions set state=case when state ? 'event' then jsonb_set(state,'{event,round_robin_winner}',display_decision)
    else jsonb_set(state,'{page_state,event,round_robin_winner}',display_decision) end,updated_at=now(),version=version+1
    where club_id=p_club_id and 'live:'||coalesce(state->'event'->>'sourceEventUid',state->'page_state'->'event'->>'sourceEventUid')=p_payload->>'source_key';
  update public.badge_program_state set revision=revision+1,retry_after=now(),pending_ties=(select coalesce(jsonb_agg(t),'[]')
    from jsonb_array_elements(pending_ties) t where t->>'source_key'<>p_payload->>'source_key') where club_id=p_club_id;
  result:=jsonb_build_object('ok',true,'winner_player_id',winner,'source_key',p_payload->>'source_key');
  insert into public.badge_round_robin_operations(club_id,operation_id,actor_user_id,payload,result)
  values(p_club_id,p_operation_id,p_actor_user_id,p_payload,result);
  return result;
end $$;

revoke all on function public.badge_program_snapshot_v1(text), public.pending_program_badge_clubs_v1(integer),
 public.program_badge_failure_v1(text),public.apply_program_badges_v1(text,bigint,jsonb,jsonb,jsonb),
 public.admin_resolve_round_robin_v1(text,text,uuid,text,uuid,jsonb) from public,anon,authenticated;
grant execute on function public.badge_program_snapshot_v1(text), public.pending_program_badge_clubs_v1(integer),
 public.program_badge_failure_v1(text),public.apply_program_badges_v1(text,bigint,jsonb,jsonb,jsonb),
 public.admin_resolve_round_robin_v1(text,text,uuid,text,uuid,jsonb) to service_role;

-- Direct earning requirements, using existing artwork for consistent display.
insert into public.badges(badge_id,name,prestige,category,is_stackable,is_active,rarity,tier,icon_key,lore,hint,scope,state,eval_triggers)
select badge_id,name,prestige,category,is_stackable,is_active,rarity,tier,icon_key,lore,hint,scope,state,eval_triggers from jsonb_to_recordset($catalog$[{"badge_id": "leagues_completed_5", "name": "5 Leagues Completed", "prestige": 20, "category": "Participation", "is_stackable": false, "is_active": true, "rarity": "common", "tier": null, "icon_key": "participant", "lore": "Finish 5 leagues at this club, meeting each league's required minimum games.", "hint": "Finish 5 leagues at this club, meeting each league's required minimum games.", "scope": "overall", "state": "live", "eval_triggers": ["program_changed"]}, {"badge_id": "leagues_completed_10", "name": "10 Leagues Completed", "prestige": 35, "category": "Participation", "is_stackable": false, "is_active": true, "rarity": "rare", "tier": null, "icon_key": "participant", "lore": "Finish 10 leagues at this club, meeting each league's required minimum games.", "hint": "Finish 10 leagues at this club, meeting each league's required minimum games.", "scope": "overall", "state": "live", "eval_triggers": ["program_changed"]}, {"badge_id": "leagues_completed_25", "name": "25 Leagues Completed", "prestige": 60, "category": "Participation", "is_stackable": false, "is_active": true, "rarity": "epic", "tier": null, "icon_key": "participant", "lore": "Finish 25 leagues at this club, meeting each league's required minimum games.", "hint": "Finish 25 leagues at this club, meeting each league's required minimum games.", "scope": "overall", "state": "live", "eval_triggers": ["program_changed"]}, {"badge_id": "tournaments_completed_5", "name": "5 Tournaments Completed", "prestige": 20, "category": "Participation", "is_stackable": false, "is_active": true, "rarity": "common", "tier": null, "icon_key": "participant", "lore": "Complete play in 5 finished tournaments at this club.", "hint": "Complete play in 5 finished tournaments at this club.", "scope": "overall", "state": "live", "eval_triggers": ["program_changed"]}, {"badge_id": "tournaments_completed_10", "name": "10 Tournaments Completed", "prestige": 35, "category": "Participation", "is_stackable": false, "is_active": true, "rarity": "rare", "tier": null, "icon_key": "participant", "lore": "Complete play in 10 finished tournaments at this club.", "hint": "Complete play in 10 finished tournaments at this club.", "scope": "overall", "state": "live", "eval_triggers": ["program_changed"]}, {"badge_id": "tournaments_completed_25", "name": "25 Tournaments Completed", "prestige": 60, "category": "Participation", "is_stackable": false, "is_active": true, "rarity": "epic", "tier": null, "icon_key": "participant", "lore": "Complete play in 25 finished tournaments at this club.", "hint": "Complete play in 25 finished tournaments at this club.", "scope": "overall", "state": "live", "eval_triggers": ["program_changed"]}, {"badge_id": "round_robins_completed_5", "name": "5 Round Robins Completed", "prestige": 20, "category": "Participation", "is_stackable": false, "is_active": true, "rarity": "common", "tier": null, "icon_key": "participant", "lore": "Complete your assigned play in 5 finished round robins at this club.", "hint": "Complete your assigned play in 5 finished round robins at this club.", "scope": "overall", "state": "live", "eval_triggers": ["program_changed"]}, {"badge_id": "round_robins_completed_10", "name": "10 Round Robins Completed", "prestige": 35, "category": "Participation", "is_stackable": false, "is_active": true, "rarity": "rare", "tier": null, "icon_key": "participant", "lore": "Complete your assigned play in 10 finished round robins at this club.", "hint": "Complete your assigned play in 10 finished round robins at this club.", "scope": "overall", "state": "live", "eval_triggers": ["program_changed"]}, {"badge_id": "round_robins_completed_25", "name": "25 Round Robins Completed", "prestige": 60, "category": "Participation", "is_stackable": false, "is_active": true, "rarity": "epic", "tier": null, "icon_key": "participant", "lore": "Complete your assigned play in 25 finished round robins at this club.", "hint": "Complete your assigned play in 25 finished round robins at this club.", "scope": "overall", "state": "live", "eval_triggers": ["program_changed"]}, {"badge_id": "matches_together_10", "name": "10 Matches Together", "prestige": 20, "category": "Partnerships", "is_stackable": true, "is_active": true, "rarity": "common", "tier": null, "icon_key": "draft_master", "lore": "Play 10 doubles matches with the same partner at this club.", "hint": "Play 10 doubles matches with the same partner at this club.", "scope": "overall", "state": "live", "eval_triggers": ["program_changed"]}, {"badge_id": "matches_together_25", "name": "25 Matches Together", "prestige": 35, "category": "Partnerships", "is_stackable": true, "is_active": true, "rarity": "rare", "tier": null, "icon_key": "draft_master", "lore": "Play 25 doubles matches with the same partner at this club.", "hint": "Play 25 doubles matches with the same partner at this club.", "scope": "overall", "state": "live", "eval_triggers": ["program_changed"]}, {"badge_id": "matches_together_50", "name": "50 Matches Together", "prestige": 60, "category": "Partnerships", "is_stackable": true, "is_active": true, "rarity": "epic", "tier": null, "icon_key": "draft_master", "lore": "Play 50 doubles matches with the same partner at this club.", "hint": "Play 50 doubles matches with the same partner at this club.", "scope": "overall", "state": "live", "eval_triggers": ["program_changed"]}, {"badge_id": "wins_together_10", "name": "10 Wins Together", "prestige": 20, "category": "Partnerships", "is_stackable": true, "is_active": true, "rarity": "common", "tier": null, "icon_key": "draft_master", "lore": "Win 10 doubles matches with the same partner at this club.", "hint": "Win 10 doubles matches with the same partner at this club.", "scope": "overall", "state": "live", "eval_triggers": ["program_changed"]}, {"badge_id": "wins_together_25", "name": "25 Wins Together", "prestige": 35, "category": "Partnerships", "is_stackable": true, "is_active": true, "rarity": "rare", "tier": null, "icon_key": "draft_master", "lore": "Win 25 doubles matches with the same partner at this club.", "hint": "Win 25 doubles matches with the same partner at this club.", "scope": "overall", "state": "live", "eval_triggers": ["program_changed"]}, {"badge_id": "wins_together_50", "name": "50 Wins Together", "prestige": 60, "category": "Partnerships", "is_stackable": true, "is_active": true, "rarity": "epic", "tier": null, "icon_key": "draft_master", "lore": "Win 50 doubles matches with the same partner at this club.", "hint": "Win 50 doubles matches with the same partner at this club.", "scope": "overall", "state": "live", "eval_triggers": ["program_changed"]}, {"badge_id": "five_winning_partners", "name": "Five Winning Partners", "prestige": 40, "category": "Partnerships", "is_stackable": true, "is_active": true, "rarity": "rare", "tier": null, "icon_key": "draft_master", "lore": "Win with at least 5 different partners in one completed round robin.", "hint": "Win with at least 5 different partners in one completed round robin.", "scope": "overall", "state": "live", "eval_triggers": ["program_changed"]}, {"badge_id": "triple_crown", "name": "Triple Crown", "prestige": 75, "category": "Trophies", "is_stackable": true, "is_active": true, "rarity": "epic", "tier": null, "icon_key": "tournament_champion", "lore": "Earn a medal in at least 3 different events at the same completed tournament.", "hint": "Earn a medal in at least 3 different events at the same completed tournament.", "scope": "overall", "state": "live", "eval_triggers": ["program_changed"]}, {"badge_id": "round_robin_wins_1", "name": "First Round Robin Win", "prestige": 20, "category": "Trophies", "is_stackable": false, "is_active": true, "rarity": "common", "tier": null, "icon_key": "tournament_champion", "lore": "Win 1 completed round robin at this club. Winners are decided by wins, point differential, total points scored, then an admin decision if still tied.", "hint": "Win 1 completed round robin at this club. Winners are decided by wins, point differential, total points scored, then an admin decision if still tied.", "scope": "overall", "state": "live", "eval_triggers": ["program_changed"]}, {"badge_id": "round_robin_wins_5", "name": "5 Round Robin Wins", "prestige": 35, "category": "Trophies", "is_stackable": false, "is_active": true, "rarity": "rare", "tier": null, "icon_key": "tournament_champion", "lore": "Win 5 completed round robins at this club. Winners are decided by wins, point differential, total points scored, then an admin decision if still tied.", "hint": "Win 5 completed round robins at this club. Winners are decided by wins, point differential, total points scored, then an admin decision if still tied.", "scope": "overall", "state": "live", "eval_triggers": ["program_changed"]}, {"badge_id": "round_robin_wins_10", "name": "10 Round Robin Wins", "prestige": 50, "category": "Trophies", "is_stackable": false, "is_active": true, "rarity": "rare", "tier": null, "icon_key": "tournament_champion", "lore": "Win 10 completed round robins at this club. Winners are decided by wins, point differential, total points scored, then an admin decision if still tied.", "hint": "Win 10 completed round robins at this club. Winners are decided by wins, point differential, total points scored, then an admin decision if still tied.", "scope": "overall", "state": "live", "eval_triggers": ["program_changed"]}, {"badge_id": "round_robin_wins_25", "name": "25 Round Robin Wins", "prestige": 75, "category": "Trophies", "is_stackable": false, "is_active": true, "rarity": "epic", "tier": null, "icon_key": "tournament_champion", "lore": "Win 25 completed round robins at this club. Winners are decided by wins, point differential, total points scored, then an admin decision if still tied.", "hint": "Win 25 completed round robins at this club. Winners are decided by wins, point differential, total points scored, then an admin decision if still tied.", "scope": "overall", "state": "live", "eval_triggers": ["program_changed"]}, {"badge_id": "round_robin_wins_50", "name": "50 Round Robin Wins", "prestige": 100, "category": "Trophies", "is_stackable": false, "is_active": true, "rarity": "epic", "tier": null, "icon_key": "tournament_champion", "lore": "Win 50 completed round robins at this club. Winners are decided by wins, point differential, total points scored, then an admin decision if still tied.", "hint": "Win 50 completed round robins at this club. Winners are decided by wins, point differential, total points scored, then an admin decision if still tied.", "scope": "overall", "state": "live", "eval_triggers": ["program_changed"]}]$catalog$::jsonb) as x(badge_id text,name text,prestige int,category text,is_stackable bool,is_active bool,rarity text,tier int,icon_key text,lore text,hint text,scope text,state text,eval_triggers jsonb)
on conflict(badge_id) do nothing;
notify pgrst, 'reload schema';
