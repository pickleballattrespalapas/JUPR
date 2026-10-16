begin;

create table public.pcs_event_series (
 id uuid primary key default gen_random_uuid(),
 club_id text not null references public.clubs(id) on delete cascade,
 event_kind text not null check(event_kind in ('interclub','league','tournament')),
 name text not null check(length(trim(name)) between 1 and 180),
 created_at timestamptz not null default now(),
 unique(id,club_id,event_kind)
);
create index pcs_event_series_club on public.pcs_event_series(club_id,event_kind);
create table public.pcs_event_editions (
 id uuid primary key default gen_random_uuid(),
 series_id uuid not null,
 club_id text not null,
 event_kind text not null,
 source_id text not null check(length(source_id) between 1 and 180),
 label text not null check(length(trim(label)) between 1 and 120),
 position integer not null,
 created_at timestamptz not null default now(),
 foreign key(series_id,club_id,event_kind) references public.pcs_event_series(id,club_id,event_kind) on delete cascade,
 unique(club_id,event_kind,source_id), unique(series_id,position) deferrable initially deferred
);
create unique index pcs_event_editions_label on public.pcs_event_editions(series_id,lower(trim(label)));
create table public.pcs_event_season_actions (
 club_id text not null references public.clubs(id) on delete cascade,
 request_id uuid not null,
 actor_id uuid not null,
 request jsonb not null,
 result jsonb not null,
 created_at timestamptz not null default now(),
 primary key(club_id,request_id)
);
alter table public.pcs_event_series enable row level security;
alter table public.pcs_event_editions enable row level security;
alter table public.pcs_event_season_actions enable row level security;
revoke all on public.pcs_event_series,public.pcs_event_editions,public.pcs_event_season_actions from public,anon,authenticated;
grant all on public.pcs_event_series,public.pcs_event_editions,public.pcs_event_season_actions to service_role;

-- Server-only source projection. It also locks the reviewed configuration for
-- the duration of the atomic rollover transaction. Public routes project it
-- through the existing event publication/visibility rules.
create function public.pcs_event_season_source(p_club_id text,p_kind text,p_source_id text)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare e jsonb; config jsonb; settings jsonb; publication jsonb; payload jsonb; days jsonb; divisions jsonb;
 complete boolean:=false; visible boolean:=false; receipt boolean:=false; team jsonb;
begin
 if p_kind='interclub' then
  select to_jsonb(s) into e from public.pcs_interclub_seasons s where id=p_source_id::uuid and organizer_club_id=p_club_id for share;
  if e is null then
   select to_jsonb(d),d.draft into e,config from public.pcs_interclub_drafts d where id=p_source_id::uuid and organizer_club_id=p_club_id for share;
  else config:=e->'details'; end if;
  if e is null then return null; end if;
  select p.published into publication from public.pcs_interclub_publications p where season_id=p_source_id::uuid for share;
  complete:=coalesce(publication->>'season_complete'='true' and publication->'club_cup'->>'status'='complete',false);
  visible:=publication is not null;
 elsif p_kind='league' then
  select to_jsonb(l) into e from public.leagues_metadata l where club_id=p_club_id and league_name=p_source_id for share;
  if e is null then return null; end if;
  select to_jsonb(t) into team from public.team_league_settings t where club_id=p_club_id and league_name=p_source_id for share;
  complete:=lower(e->>'status') in ('ended','completed','complete','done','archived') and coalesce((e->>'is_active')::boolean,false)=false;
  visible:=(lower(e->>'status') in ('active','running','live') and (e->>'is_active')::boolean=true)
    or (lower(e->>'status') in ('ended','completed','complete','done') and (e->>'is_active')::boolean=false);
 elsif p_kind='tournament' then
  select to_jsonb(t) into e from public.tournaments t where id=p_source_id::uuid and club_id=p_club_id for share;
  if e is null then return null; end if;
  select to_jsonb(s) into settings from public.tournament_registration_settings s where tournament_id=p_source_id for share;
  select coalesce(jsonb_agg(to_jsonb(d)),'[]') into days from
   (select * from public.tournament_registration_days where tournament_id=p_source_id order by id for share) d;
  select coalesce(jsonb_agg(to_jsonb(d)),'[]') into divisions from
   (select * from public.tournament_event_options where tournament_id=p_source_id order by id for share) d;
  receipt:=exists(select 1 from public.tournament_lifecycle_receipts where tournament_id=p_source_id::uuid and club_id=p_club_id and action='complete');
  complete:=upper(e->>'status') in ('COMPLETED','ARCHIVED') and receipt;
  visible:=coalesce(nullif(settings->'builder_draft_json'->>'published_at','') is not null and
   (upper(e->>'status')='ACTIVE' or (upper(e->>'status')='COMPLETED' and receipt)),false);
 else raise exception 'Unsupported event type' using errcode='22023'; end if;
 payload:=jsonb_build_object('event',e,'setup',config,'settings',settings,'publication',publication,
   'team',team,'days',days,'divisions',divisions,'complete',complete,'public',visible);
 return payload||jsonb_build_object('fingerprint',md5(payload::text));
end $$;

create function public.pcs_start_event_season(p_actor_id uuid,p_actor_email text,p_club_id text,p_kind text,
 p_source_id text,p_request_id uuid,p_input jsonb,p_new_id uuid,p_template jsonb)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare source jsonb; e jsonb; team jsonb; existing public.pcs_event_season_actions;
 edition public.pcs_event_editions; sid uuid; next_position integer; target text; result jsonb; cfg jsonb;
begin
 perform public.pcs_require_interclub_admin(p_actor_id,p_actor_email,p_club_id);
 perform pg_advisory_xact_lock(hashtextextended('pcs-event-history:'||p_club_id||':'||p_kind,0));
 select * into existing from public.pcs_event_season_actions where club_id=p_club_id and request_id=p_request_id;
 if found then
  if existing.request<>jsonb_build_object('kind',p_kind,'source_id',p_source_id,'input',p_input,'action','start') then
   raise exception 'This request was already used for a different season' using errcode='PT409'; end if;
  return existing.result; end if;
 source:=public.pcs_event_season_source(p_club_id,p_kind,p_source_id);
 if source is null then raise exception 'Event not found' using errcode='P0002'; end if;
 if source->>'fingerprint' is distinct from p_input->>'fingerprint' then
  raise exception 'The event changed. Reload and review the new season again.' using errcode='PT409'; end if;
 if source->>'complete' is distinct from 'true' then
  raise exception 'Finish and publish the current season before starting the next one.' using errcode='PT409'; end if;
 if length(trim(p_input->>'name')) not between 1 and 120 or length(trim(p_input->>'label')) not between 1 and 120
  or length(trim(p_input->>'current_label')) not between 1 and 120 or length(trim(p_input->>'series_name')) not between 1 and 180
  or p_input->>'start_date' is null or p_input->>'end_date' is null
  or (p_input->>'end_date')::date<(p_input->>'start_date')::date then
  raise exception 'Check the season name, label and dates.' using errcode='22023'; end if;
 select * into edition from public.pcs_event_editions where club_id=p_club_id and event_kind=p_kind and source_id=p_source_id;
 if found then
  sid:=edition.series_id;
  if exists(select 1 from public.pcs_event_editions where series_id=sid and position>edition.position) then
   raise exception 'A later season already exists. Continue it from History.' using errcode='PT409'; end if;
 else
  insert into public.pcs_event_series(club_id,event_kind,name) values(p_club_id,p_kind,trim(p_input->>'series_name')) returning id into sid;
  insert into public.pcs_event_editions(series_id,club_id,event_kind,source_id,label,position)
   values(sid,p_club_id,p_kind,p_source_id,trim(p_input->>'current_label'),1);
 end if;
 select max(position)+1 into next_position from public.pcs_event_editions where series_id=sid;
 e:=source->'event';
 if p_kind='interclub' then
  target:=p_new_id::text;
  perform public.pcs_save_interclub_draft(p_actor_id,p_actor_email,p_club_id,p_new_id,0,p_template);
 elsif p_kind='league' then
  target:=trim(p_input->>'name');
  if exists(select 1 from public.leagues_metadata where club_id=p_club_id and lower(trim(league_name))=lower(target)) then
   raise exception 'A league with this name already exists.' using errcode='PT409'; end if;
  insert into public.leagues_metadata(club_id,league_name,league_type,match_format,min_weeks,min_games,description,k_factor,
   is_active,status,schedule_config,court_board_defaults,rules_config,awards_config,event_tags)
  values(p_club_id,target,e->>'league_type',e->>'match_format',(e->>'min_weeks')::integer,(e->>'min_games')::integer,e->>'description',
   (e->>'k_factor')::integer,false,'draft',p_template->'schedule_config',e->'court_board_defaults',e->'rules_config',e->'awards_config',
   coalesce(e->'event_tags','{}')||jsonb_build_object('date_tags','[]'::jsonb));
  team:=source->'team';
  if team is not null and team<>'null'::jsonb then
   team:=team||jsonb_build_object('league_name',target,'status','draft','registration_open',false,'start_date',p_input->>'start_date',
    'registration_closes_at',null,'settings_version',0,'schedule_version',0,'standings_version',0,'roster_version',0,
    'created_by',p_actor_email,'updated_by',p_actor_email,'created_at',now(),'updated_at',now());
   insert into public.team_league_settings select (jsonb_populate_record(null::public.team_league_settings,team)).*;
  end if;
 else
  target:=p_new_id::text;
  insert into public.tournaments(id,club_id,name,status,team_count,playoff_advance_count,playoff_best_of,start_date,end_date,created_by_admin_id)
   values(p_new_id,p_club_id,trim(p_input->>'name'),'DRAFT',coalesce((e->>'team_count')::integer,4),
    (e->>'playoff_advance_count')::integer,coalesce((e->>'playoff_best_of')::integer,1),
    (p_input->>'start_date')::date,(p_input->>'end_date')::date,p_actor_id::text);
  cfg:=p_template->'settings';
  insert into public.tournament_registration_settings(id,tournament_id,registration_slug,registration_status,locale,builder_draft_json,
   builder_draft_updated_at,location_name,timezone,rules_markdown,refund_policy_markdown,weather_policy_markdown,venue_address,venue_directions,venue_courts_json)
  values('regset-'||target,target,'season-'||target,'draft',coalesce(cfg->>'locale','en'),p_template,now(),cfg->>'location_name',
   coalesce(cfg->>'timezone','America/Mazatlan'),cfg->>'rules_markdown',cfg->>'refund_policy_markdown',cfg->>'weather_policy_markdown',
   cfg->>'venue_address',cfg->>'venue_directions',coalesce(cfg->'venue_courts_json','[]'));
 end if;
 insert into public.pcs_event_editions(series_id,club_id,event_kind,source_id,label,position)
  values(sid,p_club_id,p_kind,target,trim(p_input->>'label'),next_position);
 result:=jsonb_build_object('series_id',sid,'source_id',target,'kind',p_kind,'name',trim(p_input->>'name'),'label',trim(p_input->>'label'));
 insert into public.pcs_event_season_actions(club_id,request_id,actor_id,request,result)
  values(p_club_id,p_request_id,p_actor_id,jsonb_build_object('kind',p_kind,'source_id',p_source_id,'input',p_input,'action','start'),result);
 return result;
end $$;

create function public.pcs_link_event_season(p_actor_id uuid,p_actor_email text,p_club_id text,p_kind text,
 p_source_id text,p_request_id uuid,p_input jsonb)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare source jsonb; older jsonb; edition public.pcs_event_editions; sid uuid; next_position integer;
 existing public.pcs_event_season_actions; result jsonb;
begin
 perform public.pcs_require_interclub_admin(p_actor_id,p_actor_email,p_club_id);
 perform pg_advisory_xact_lock(hashtextextended('pcs-event-history:'||p_club_id||':'||p_kind,0));
 select * into existing from public.pcs_event_season_actions where club_id=p_club_id and request_id=p_request_id;
 if found then
  if existing.request<>jsonb_build_object('kind',p_kind,'source_id',p_source_id,'input',p_input,'action','link') then
   raise exception 'This request was already used for another link.' using errcode='PT409'; end if;
  return existing.result; end if;
 source:=public.pcs_event_season_source(p_club_id,p_kind,p_source_id);
 older:=public.pcs_event_season_source(p_club_id,p_kind,p_input->>'past_source_id');
 if source is null or older is null then raise exception 'Event not found' using errcode='P0002'; end if;
 if source->>'fingerprint' is distinct from p_input->>'fingerprint' or older->>'fingerprint' is distinct from p_input->>'past_fingerprint' then
  raise exception 'An event changed. Reload and review the history link.' using errcode='PT409'; end if;
 if older->>'complete' is distinct from 'true' or p_source_id=p_input->>'past_source_id' then
  raise exception 'Choose a different completed past season.' using errcode='22023'; end if;
 if exists(select 1 from public.pcs_event_editions where club_id=p_club_id and event_kind=p_kind and source_id=p_input->>'past_source_id') then
  raise exception 'That season already belongs to an event history.' using errcode='PT409'; end if;
 select * into edition from public.pcs_event_editions where club_id=p_club_id and event_kind=p_kind and source_id=p_source_id;
 if found then sid:=edition.series_id;
 else
  insert into public.pcs_event_series(club_id,event_kind,name) values(p_club_id,p_kind,trim(p_input->>'series_name')) returning id into sid;
  insert into public.pcs_event_editions(series_id,club_id,event_kind,source_id,label,position)
   values(sid,p_club_id,p_kind,p_source_id,trim(p_input->>'current_label'),1);
 end if;
 select position into next_position from public.pcs_event_editions where series_id=sid and source_id=p_input->>'before_source_id';
 if not found then raise exception 'Choose which season this earlier event precedes.' using errcode='22023'; end if;
 update public.pcs_event_editions set position=position+1 where series_id=sid and position>=next_position;
 insert into public.pcs_event_editions(series_id,club_id,event_kind,source_id,label,position)
  values(sid,p_club_id,p_kind,p_input->>'past_source_id',trim(p_input->>'label'),next_position);
 result:=jsonb_build_object('series_id',sid,'source_id',p_input->>'past_source_id','kind',p_kind);
 insert into public.pcs_event_season_actions(club_id,request_id,actor_id,request,result)
  values(p_club_id,p_request_id,p_actor_id,jsonb_build_object('kind',p_kind,'source_id',p_source_id,'input',p_input,'action','link'),result);
 return result;
end $$;

revoke all on function public.pcs_event_season_source(text,text,text),
 public.pcs_start_event_season(uuid,text,text,text,text,uuid,jsonb,uuid,jsonb),
 public.pcs_link_event_season(uuid,text,text,text,text,uuid,jsonb) from public,anon,authenticated;
grant execute on function public.pcs_event_season_source(text,text,text),
 public.pcs_start_event_season(uuid,text,text,text,text,uuid,jsonb,uuid,jsonb),
 public.pcs_link_event_season(uuid,text,text,text,text,uuid,jsonb) to service_role;
notify pgrst,'reload schema';
commit;
