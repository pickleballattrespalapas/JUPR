begin;

-- Request identity makes a lost create response safe to retry. Request payloads
-- stay private and are compared before returning an existing meet.
alter table public.pcs_interclub_meets add column creation_request_id uuid,
 add column creation_request jsonb,
 add constraint pcs_interclub_meet_creation_request_key unique(season_id,creation_request_id);

create function public.pcs_interclub_meet_schedule_lock_reason(p_meet_id uuid)
returns text language plpgsql security invoker set search_path=public as $$
declare meet public.pcs_interclub_meets;
begin
 select * into meet from public.pcs_interclub_meets where id=p_meet_id;
 if not found then return 'Meet unavailable.'; end if;
 if meet.starts_at<=clock_timestamp() then return 'This meet has started. Use the weather replay or results correction workflow.'; end if;
 if exists(select 1 from public.pcs_interclub_competition_batches b where b.meet_id=p_meet_id and
  (b.state<>'draft' or b.approved_document is not null or b.approved_revision is not null
   or coalesce(b.document->>'weather','normal')<>'normal'
   or exists(select 1 from jsonb_array_elements(b.document->'encounters') e where
    (e->'tiebreak' is not null and e->'tiebreak'<>'null'::jsonb and
     (e->'tiebreak'->>'status' is distinct from 'pending' or e->'tiebreak'->>'a' is not null
      or e->'tiebreak'->>'b' is not null or e->'tiebreak'->>'played_at' is not null))
    or exists(select 1 from jsonb_array_elements(e->'pairings') p,
     lateral jsonb_array_elements(p->'games') g where
      g->>'a' is not null or g->>'b' is not null or g->>'played_at' is not null
      or not(g->>'status'='pending' or (g->>'status' in ('forfeit','double_forfeit')
       and (jsonb_array_length(p->'players_a')=0 or jsonb_array_length(p->'players_b')=0))))))) then
  return 'Scores or a replay already exist. Use the weather replay or results correction workflow.';
 end if;
 return null;
end $$;

-- A submitted provisional lineup is not a frozen rating snapshot. Its players
-- can be retained in a new roster version when a future cutoff moves.
create function public.pcs_interclub_schedule_deadline_editable(p_meet_id uuid)
returns boolean language sql security invoker set search_path=public as $$
 select exists(select 1 from public.pcs_interclub_meets m where m.id=p_meet_id
  and m.starts_at>clock_timestamp() and m.roster_deadline>clock_timestamp()
  and not exists(select 1 from public.pcs_interclub_meet_eligibility_snapshots s where s.meet_id=m.id)
  and not exists(select 1 from public.pcs_interclub_competition_batches b where b.meet_id=m.id)
  and not exists(select 1 from public.pcs_interclub_current_rosters r,
   lateral jsonb_array_elements(r.roster) p where r.meet_id=m.id and (p->>'rating_locked')::boolean is true))
$$;

create function public.pcs_assert_interclub_schedule_dates(p_season_id uuid,p_starts_at timestamptz,p_deadline timestamptz,p_duration integer,p_courts integer)
returns void language plpgsql security invoker set search_path=public as $$
declare season public.pcs_interclub_seasons; first_day date; last_day date; zone text;
begin
 select * into season from public.pcs_interclub_seasons where id=p_season_id;
 if not found then raise exception 'Season unavailable' using errcode='P0002'; end if;
 zone:=season.details->>'timezone'; first_day:=(season.details->>'start_date')::date; last_day:=(season.details->>'end_date')::date;
 if p_starts_at is null or p_deadline is null or not isfinite(p_starts_at) or not isfinite(p_deadline)
  or p_starts_at<=clock_timestamp() or p_deadline>p_starts_at
  or p_duration is null or p_duration not between 30 and 180 or p_courts is null or p_courts not between 1 and 100
  or zone is null or first_day is null or last_day is null
  or (p_starts_at at time zone zone)::date not between first_day and last_day
  or ((p_starts_at+make_interval(mins=>p_duration)) at time zone zone)::date>last_day
  or season.registration_closes_at is null or p_starts_at<season.registration_closes_at then
  raise exception 'Choose a future meet within the season, after registration closes, with a deadline no later than its start' using errcode='22023';
 end if;
end $$;

-- The relational calendar is authoritative after season setup. Keep its legacy
-- details projection current, including weather replay changes, while public
-- publication snapshots continue to require explicit review and republishing.
create function public.pcs_sync_interclub_season_schedule() returns trigger
language plpgsql security invoker set search_path=public as $$
begin
 update public.pcs_interclub_seasons s set details=jsonb_set(s.details,'{meets}',
  (select coalesce(jsonb_agg(jsonb_build_object('host_club_id',m.host_club_id,'club_ids',m.club_ids,
   'starts_at',m.starts_at,'duration_minutes',m.duration_minutes,'courts',m.courts) order by m.plan_index),'[]'::jsonb)
   from public.pcs_interclub_meets m where m.season_id=new.season_id)) where s.id=new.season_id;
 return new;
end $$;
create trigger pcs_interclub_sync_season_schedule after insert or update of starts_at,duration_minutes,courts,club_ids,host_club_id
 on public.pcs_interclub_meets for each row execute function public.pcs_sync_interclub_season_schedule();

create or replace function public.pcs_create_interclub_competition_meet(p_actor_id uuid,p_actor_email text,p_club_id text,p_season_id uuid,p_meet jsonb)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare result public.pcs_interclub_meets; cid text; phase text; n integer; request_id uuid;
begin
 perform public.pcs_require_interclub_admin(p_actor_id,p_actor_email,p_club_id);
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||p_season_id::text,0));
 if not exists(select 1 from public.pcs_interclub_seasons where id=p_season_id and organizer_club_id=p_club_id) then raise exception 'Organizer required' using errcode='42501'; end if;
 perform public.pcs_require_interclub_registration_phase(p_season_id,'closed');
 request_id:=(p_meet->>'request_id')::uuid;
 if request_id is not null then
  select * into result from public.pcs_interclub_meets where season_id=p_season_id and creation_request_id=request_id;
  if found then
   if result.creation_request is distinct from p_meet then raise exception 'Scheduling request changed. Use a new request for different details.' using errcode='PT409'; end if;
   return to_jsonb(result)-'creation_request_id'-'creation_request';
  end if;
 end if;
 phase:=p_meet->>'competition_phase'; n:=jsonb_array_length(p_meet->'club_ids');
 if phase is null or phase not in ('regular','final','qualifier') or n is null or n<2 or n>(case when phase='regular' then 4 else 32 end)
  or (select count(distinct value) from jsonb_array_elements_text(p_meet->'club_ids'))<>n
  or not(p_meet->'club_ids' ? (p_meet->>'host_club_id'))
  or (p_meet->>'roster_deadline')::timestamptz<=clock_timestamp() then raise exception 'Check meet details' using errcode='22023'; end if;
 perform public.pcs_assert_interclub_schedule_dates(p_season_id,(p_meet->>'starts_at')::timestamptz,
  (p_meet->>'roster_deadline')::timestamptz,(p_meet->>'duration_minutes')::integer,(p_meet->>'courts')::integer);
 for cid in select jsonb_array_elements_text(p_meet->'club_ids') union select p_meet->>'host_club_id' loop
  if not exists(select 1 from public.pcs_interclub_participations where season_id=p_season_id and club_id=cid and status='accepted') then raise exception 'Accepted clubs required' using errcode='22023'; end if;
 end loop;
 insert into public.pcs_interclub_meets(season_id,plan_index,host_club_id,club_ids,starts_at,duration_minutes,courts,roster_deadline,competition_phase,creation_request_id,creation_request)
 values(p_season_id,(select coalesce(max(plan_index),-1)+1 from public.pcs_interclub_meets where season_id=p_season_id),
 p_meet->>'host_club_id',p_meet->'club_ids',(p_meet->>'starts_at')::timestamptz,(p_meet->>'duration_minutes')::integer,
 (p_meet->>'courts')::integer,(p_meet->>'roster_deadline')::timestamptz,phase,request_id,p_meet) returning * into result;
 insert into public.pcs_interclub_registration_audit(season_id,actor_id,actor_club_id,action,details)
 values(p_season_id,p_actor_id,p_club_id,'competition_meet_created',to_jsonb(result)-'creation_request');
 return to_jsonb(result)-'creation_request_id'-'creation_request';
end $$;

create function public.pcs_update_interclub_meet_schedule(p_actor_id uuid,p_actor_email text,p_club_id text,p_season_id uuid,
 p_meet_id uuid,p_revision integer,p_starts_at timestamptz,p_deadline timestamptz,p_duration integer,p_courts integer)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare meet public.pcs_interclub_meets; prior jsonb; locked text; cid text; reset_count integer:=0;
 availability_before jsonb; responses_before jsonb; roster_before jsonb; rosters_refreshed integer:=0;
 team public.pcs_interclub_teams; saved public.pcs_interclub_roster_versions; lineup jsonb;
begin
 perform public.pcs_require_interclub_admin(p_actor_id,p_actor_email,p_club_id);
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||p_season_id::text,0));
 if not exists(select 1 from public.pcs_interclub_seasons where id=p_season_id and organizer_club_id=p_club_id) then raise exception 'Organizer required' using errcode='42501'; end if;
 perform public.pcs_require_interclub_registration_phase(p_season_id,'closed');
 select * into meet from public.pcs_interclub_meets where id=p_meet_id and season_id=p_season_id for update;
 if not found then raise exception 'Meet unavailable' using errcode='P0002'; end if;
 if meet.revision is distinct from p_revision then raise exception 'Meet changed. Reload before saving.' using errcode='PT409'; end if;
 locked:=public.pcs_interclub_meet_schedule_lock_reason(p_meet_id);
 if locked is not null then raise exception '%',locked using errcode='PT409'; end if;
 perform public.pcs_assert_interclub_schedule_dates(p_season_id,p_starts_at,p_deadline,p_duration,p_courts);
 for cid in select jsonb_array_elements_text(meet.club_ids) union select meet.host_club_id loop
  if not exists(select 1 from public.pcs_interclub_participations where season_id=p_season_id and club_id=cid and status='accepted') then raise exception 'Accepted clubs required' using errcode='22023'; end if;
 end loop;
 if p_deadline is distinct from meet.roster_deadline and
  (p_deadline<=clock_timestamp() or not public.pcs_interclub_schedule_deadline_editable(p_meet_id)) then
  raise exception 'Keep the frozen eligibility deadline or use the weather replay workflow.' using errcode='PT409'; end if;
 if p_courts is distinct from meet.courts and exists(select 1 from public.pcs_interclub_competition_batches where meet_id=p_meet_id) then
  raise exception 'Keep the courts assigned to the generated match schedule.' using errcode='PT409'; end if;
 prior:=to_jsonb(meet)-'creation_request_id'-'creation_request';
 -- No score document, source lineup, or eligibility snapshot is rewritten.
 if p_deadline is distinct from meet.roster_deadline then
  select coalesce(jsonb_agg(to_jsonb(r)),'[]'::jsonb) into roster_before from public.pcs_interclub_current_rosters r where r.meet_id=p_meet_id;
  for team in select * from public.pcs_interclub_teams where meet_id=p_meet_id and not withdrawn order by id for update loop
   select * into saved from public.pcs_interclub_roster_versions where team_id=team.id and revision=team.revision;
   select jsonb_agg(p||jsonb_build_object('rating_deadline',p_deadline,'rating_locked',false) order by n)
    into lineup from jsonb_array_elements(saved.roster) with ordinality players(p,n);
   update public.pcs_interclub_teams set revision=revision+1,updated_at=now() where id=team.id returning * into team;
   insert into public.pcs_interclub_roster_versions(team_id,revision,name,roster,issues,status,late_change)
    values(team.id,team.revision,saved.name,lineup,saved.issues,
     case when saved.status in ('exception_approved','exception_denied') then 'needs_exception' else saved.status end,saved.late_change);
   rosters_refreshed:=rosters_refreshed+1;
  end loop;
 end if;
 if p_starts_at is distinct from meet.starts_at or p_duration is distinct from meet.duration_minutes then
  select coalesce(jsonb_agg(to_jsonb(a)),'[]'::jsonb) into availability_before from public.pcs_interclub_availability_settings a where a.meet_id=p_meet_id;
  select coalesce(jsonb_agg(to_jsonb(r)),'[]'::jsonb) into responses_before from public.pcs_interclub_availability_responses r where r.meet_id=p_meet_id;
  update public.pcs_interclub_availability_settings set open=false,deadline=least(deadline,p_starts_at),revision=revision+1,updated_at=now() where meet_id=p_meet_id;
  update public.pcs_interclub_availability_responses set status='invited',responded_at=null,token_nonce=gen_random_uuid(),revision=revision+1 where meet_id=p_meet_id;
  get diagnostics reset_count=row_count;
 end if;
 update public.pcs_interclub_meets set starts_at=p_starts_at,roster_deadline=p_deadline,duration_minutes=p_duration,courts=p_courts,revision=revision+1
  where id=p_meet_id returning * into meet;
 insert into public.pcs_interclub_registration_audit(season_id,actor_id,actor_club_id,action,details)
 values(p_season_id,p_actor_id,p_club_id,'meet_schedule_updated',jsonb_build_object('before',prior,
  'after',to_jsonb(meet)-'creation_request_id'-'creation_request','availability_before',availability_before,
  'responses_before',responses_before,'availability_reset_count',reset_count,'rosters_before',roster_before,'rosters_refreshed',rosters_refreshed));
 return jsonb_build_object('meet',to_jsonb(meet)-'creation_request_id'-'creation_request',
  'availability_reset_count',reset_count,'rosters_refreshed',rosters_refreshed,'publication_review_required',true);
end $$;

create or replace view public.pcs_interclub_meet_workspaces with(security_invoker=true) as
 select m.id,m.season_id,m.plan_index,m.host_club_id,m.club_ids,m.starts_at,m.duration_minutes,m.courts,m.roster_deadline,m.revision,
 m.starts_at>now() and public.pcs_interclub_registration_phase(s.registration_opens_at,s.registration_closes_at)='closed' as roster_open,
 public.pcs_interclub_registration_phase(s.registration_opens_at,s.registration_closes_at)='closed'
  and not exists(select 1 from public.pcs_interclub_teams t where t.meet_id=m.id)
  and public.pcs_interclub_schedule_deadline_editable(m.id) as deadline_editable,
 m.competition_phase,
 public.pcs_interclub_registration_phase(s.registration_opens_at,s.registration_closes_at)='closed'
  and public.pcs_interclub_meet_schedule_lock_reason(m.id) is null as schedule_editable,
 case when public.pcs_interclub_registration_phase(s.registration_opens_at,s.registration_closes_at)<>'closed'
  then 'Meet planning opens after season registration closes.' else public.pcs_interclub_meet_schedule_lock_reason(m.id) end as schedule_locked_reason,
 not exists(select 1 from public.pcs_interclub_competition_batches b where b.meet_id=m.id) as courts_editable,
 public.pcs_interclub_registration_phase(s.registration_opens_at,s.registration_closes_at)='closed'
  and public.pcs_interclub_schedule_deadline_editable(m.id) as schedule_deadline_editable
 from public.pcs_interclub_meets m join public.pcs_interclub_seasons s on s.id=m.season_id;

-- Legacy deadline-only edits still refuse submitted rosters and now also refuse
-- frozen ratings/batches; the audited schedule RPC handles provisional rosters.
do $patch$
declare definition text; anchor text:=E' or exists(select 1 from public.pcs_interclub_teams where meet_id=p_meet_id) then';
begin
 definition:=pg_get_functiondef('public.pcs_set_interclub_meet_deadline(uuid,text,text,uuid,uuid,integer,timestamptz)'::regprocedure);
 if position(anchor in definition)=0 then raise exception 'Meet deadline guard anchor missing'; end if;
 execute replace(definition,anchor,E' or exists(select 1 from public.pcs_interclub_teams where meet_id=p_meet_id)\n  or not public.pcs_interclub_schedule_deadline_editable(p_meet_id) then');
end $patch$;

revoke all on function public.pcs_interclub_meet_schedule_lock_reason(uuid),public.pcs_interclub_schedule_deadline_editable(uuid),
 public.pcs_assert_interclub_schedule_dates(uuid,timestamptz,timestamptz,integer,integer),public.pcs_sync_interclub_season_schedule(),
 public.pcs_update_interclub_meet_schedule(uuid,text,text,uuid,uuid,integer,timestamptz,timestamptz,integer,integer),
 public.pcs_create_interclub_competition_meet(uuid,text,text,uuid,jsonb) from public,anon,authenticated;
grant execute on function public.pcs_interclub_meet_schedule_lock_reason(uuid),public.pcs_interclub_schedule_deadline_editable(uuid),
 public.pcs_assert_interclub_schedule_dates(uuid,timestamptz,timestamptz,integer,integer),public.pcs_sync_interclub_season_schedule(),
 public.pcs_update_interclub_meet_schedule(uuid,text,text,uuid,uuid,integer,timestamptz,timestamptz,integer,integer),
 public.pcs_create_interclub_competition_meet(uuid,text,text,uuid,jsonb) to service_role;
revoke all on public.pcs_interclub_meet_workspaces from public,anon,authenticated;
grant select on public.pcs_interclub_meet_workspaces to service_role;
notify pgrst,'reload schema';
commit;
