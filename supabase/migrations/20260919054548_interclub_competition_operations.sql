begin;

alter table public.pcs_interclub_meets add column competition_phase text not null default 'regular'
 check(competition_phase in ('regular','final','qualifier'));
alter table public.pcs_interclub_meets drop constraint pcs_interclub_meets_club_ids_check;
alter table public.pcs_interclub_meets add constraint pcs_interclub_meets_club_ids_check
 check(jsonb_typeof(club_ids)='array' and jsonb_array_length(club_ids) between 2 and
  case when competition_phase='regular' then 4 else 32 end);
create or replace view public.pcs_interclub_meet_workspaces with(security_invoker=true) as
 select m.id,m.season_id,m.plan_index,m.host_club_id,m.club_ids,m.starts_at,m.duration_minutes,m.courts,m.roster_deadline,m.revision,
 m.starts_at>now() as roster_open,
 m.starts_at>now() and not exists(select 1 from public.pcs_interclub_teams t where t.meet_id=m.id) as deadline_editable,
 m.competition_phase from public.pcs_interclub_meets m;
revoke all on public.pcs_interclub_meet_workspaces from public,anon,authenticated;
grant select on public.pcs_interclub_meet_workspaces to service_role;

create function public.pcs_create_interclub_competition_meet(p_actor_id uuid,p_actor_email text,p_club_id text,p_season_id uuid,p_meet jsonb)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare result public.pcs_interclub_meets; cid text; phase text; n integer;
begin
 perform public.pcs_require_interclub_admin(p_actor_id,p_actor_email,p_club_id);
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||p_season_id::text,0));
 if not exists(select 1 from public.pcs_interclub_seasons where id=p_season_id and organizer_club_id=p_club_id) then raise exception 'Organizer required' using errcode='42501'; end if;
 phase:=p_meet->>'competition_phase'; n:=jsonb_array_length(p_meet->'club_ids');
 if phase not in ('regular','final','qualifier') or n<2 or n>(case when phase='regular' then 4 else 32 end)
  or (select count(distinct value) from jsonb_array_elements_text(p_meet->'club_ids'))<>n
  or (p_meet->>'roster_deadline')::timestamptz<=now()
  or (p_meet->>'roster_deadline')::timestamptz>(p_meet->>'starts_at')::timestamptz then raise exception 'Check meet details' using errcode='22023'; end if;
 for cid in select jsonb_array_elements_text(p_meet->'club_ids') union select p_meet->>'host_club_id' loop
  if not exists(select 1 from public.pcs_interclub_participations where season_id=p_season_id and club_id=cid and status='accepted') then raise exception 'Accepted clubs required' using errcode='22023'; end if;
 end loop;
 insert into public.pcs_interclub_meets(season_id,plan_index,host_club_id,club_ids,starts_at,duration_minutes,courts,roster_deadline,competition_phase)
 values(p_season_id,(select coalesce(max(plan_index),-1)+1 from public.pcs_interclub_meets where season_id=p_season_id),
 p_meet->>'host_club_id',p_meet->'club_ids',(p_meet->>'starts_at')::timestamptz,(p_meet->>'duration_minutes')::integer,
 (p_meet->>'courts')::integer,(p_meet->>'roster_deadline')::timestamptz,phase) returning * into result;
 insert into public.pcs_interclub_registration_audit(season_id,actor_id,actor_club_id,action,details)
 values(p_season_id,p_actor_id,p_club_id,'competition_meet_created',to_jsonb(result));
 return to_jsonb(result);
end $$;
revoke all on function public.pcs_create_interclub_competition_meet(uuid,text,text,uuid,jsonb) from public,anon,authenticated;
grant execute on function public.pcs_create_interclub_competition_meet(uuid,text,text,uuid,jsonb) to service_role;

-- Private working scores and immutable official snapshots. Club directories and
-- contact details never belong in this competition document.
create table public.pcs_interclub_competition_batches (
 id uuid primary key default gen_random_uuid(),
 season_id uuid not null references public.pcs_interclub_seasons(id),
 meet_id uuid not null,
 phase text not null check(phase in ('regular','final','qualifier')),
 revision integer not null default 1 check(revision>0),
 state text not null default 'draft' check(state in ('draft','submitted','approved')),
 document jsonb not null check(jsonb_typeof(document)='object'),
 roster_sources jsonb not null check(jsonb_typeof(roster_sources)='array'),
 approved_document jsonb,
 approved_revision integer,
 ratings_status text not null default 'not_requested' check(ratings_status in ('not_requested','pending','failed','completed')),
 ratings_error text,
 approved_at timestamptz,
 updated_at timestamptz not null default now(),
 unique(season_id,meet_id,phase),
 foreign key(meet_id,season_id) references public.pcs_interclub_meets(id,season_id),
 check((approved_document is null)=(approved_revision is null))
);
create index pcs_interclub_competition_season_idx on public.pcs_interclub_competition_batches(season_id,state);
create table public.pcs_interclub_competition_audit (
 id bigint generated always as identity primary key,
 batch_id uuid not null references public.pcs_interclub_competition_batches(id),
 revision integer not null,
 actor_id uuid not null,
 actor_club_id text not null references public.clubs(id),
 action text not null,
 reason text,
 before_document jsonb,
 after_document jsonb not null,
 created_at timestamptz not null default now(),
 unique(batch_id,revision)
);
alter table public.pcs_interclub_competition_batches enable row level security;
alter table public.pcs_interclub_competition_audit enable row level security;
revoke all on public.pcs_interclub_competition_batches,public.pcs_interclub_competition_audit from public,anon,authenticated;
grant all on public.pcs_interclub_competition_batches,public.pcs_interclub_competition_audit to service_role;
grant usage,select on sequence public.pcs_interclub_competition_audit_id_seq to service_role;

-- Same staff/season lock order as roster edits. Operators need an actual host
-- assignment scoped to leagues, this season/meet, or all club operations.
create function public.pcs_write_interclub_competition(
 p_actor_id uuid,p_actor_email text,p_club_id text,p_season_id uuid,p_meet_id uuid,p_phase text,
 p_action text,p_revision integer,p_document jsonb default null,p_roster_sources jsonb default null,
 p_reason text default null,p_starts_at timestamptz default null,p_deadline timestamptz default null,p_qualification_sources jsonb default null
) returns jsonb language plpgsql security invoker set search_path=public as $$
declare season public.pcs_interclub_seasons; meet public.pcs_interclub_meets;
 saved public.pcs_interclub_competition_batches; prior jsonb; old_document jsonb;
 source jsonb; encounter jsonb; pairing jsonb; game jsonb; entry_value text; side text;
 old_pairing jsonb; source_team public.pcs_interclub_current_rosters;
 is_admin boolean; is_operator boolean; organizer boolean; next_state text;
 game_index integer; lineup jsonb; current_qualification_sources jsonb; represented text; identity_row public.pcs_interclub_entries;
begin
 if p_actor_id is null or p_phase not in ('regular','final','qualifier')
  or p_action not in ('generate','save','submit','approve','reopen','reschedule','refresh_lineups') then
  raise exception 'Invalid competition action' using errcode='22023'; end if;
 perform pg_advisory_xact_lock(hashtextextended('pcs-staff:'||p_club_id,0));
 perform 1 from public.admin_role_assignments where club_id=p_club_id and email=lower(trim(p_actor_email))
  and (user_id is null or user_id=p_actor_id) and revoked_at is null and (expires_at is null or expires_at>now()) for share;
 select coalesce(bool_or(role in ('super_admin','administrator','club_owner')),false),
  coalesce(bool_or(role='operator' and exists(select 1 from jsonb_array_elements(coalesce(scopes,'[]'::jsonb)) sc
   where sc->>'kind'='club' or (sc->>'program_type'='leagues' and
    (sc->>'kind'='program_type' or (sc->>'kind'='resource' and sc->>'resource_id' in(p_season_id::text,p_meet_id::text)))))),false)
 into is_admin,is_operator from public.admin_role_assignments where club_id=p_club_id and email=lower(trim(p_actor_email))
  and (user_id is null or user_id=p_actor_id) and revoked_at is null and (expires_at is null or expires_at>now());
 if not(is_admin or is_operator) then raise exception 'Assigned meet staff required' using errcode='42501'; end if;
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||p_season_id::text,0));
 select * into season from public.pcs_interclub_seasons where id=p_season_id;
 select * into meet from public.pcs_interclub_meets where id=p_meet_id and season_id=p_season_id for update;
 if season.id is null or meet.id is null then raise exception 'Meet unavailable' using errcode='P0002'; end if;
 if meet.competition_phase<>p_phase then raise exception 'Use the scheduled competition phase' using errcode='22023'; end if;
 organizer:=season.organizer_club_id=p_club_id and is_admin;
 if p_club_id not in (season.organizer_club_id,meet.host_club_id) or
  (p_club_id<>season.organizer_club_id and not exists(select 1 from public.pcs_interclub_participations
    where season_id=p_season_id and club_id=p_club_id and status='accepted')) then
  raise exception 'Host or organizer required' using errcode='42501'; end if;
 if (p_action in ('approve','reopen','reschedule') or (p_action='generate' and p_phase<>'regular')) and not organizer then
  raise exception 'Organizer required' using errcode='42501'; end if;
 select * into saved from public.pcs_interclub_competition_batches
  where season_id=p_season_id and meet_id=p_meet_id and phase=p_phase for update;
 if (saved.id is null and (p_action<>'generate' or p_revision<>0)) or
  (saved.id is not null and saved.revision<>p_revision) then
  raise exception 'Competition changed' using errcode='40001'; end if;
 if saved.approved_document is not null and not organizer then
  raise exception 'Published corrections require organizer' using errcode='42501'; end if;
 if p_phase<>'regular' and p_action in ('generate','submit','approve') then
  select coalesce(jsonb_agg(jsonb_build_object('id',id,'revision',approved_revision) order by id),'[]'::jsonb)
   into current_qualification_sources from public.pcs_interclub_competition_batches where season_id=p_season_id and approved_document is not null;
  if p_qualification_sources is null or p_qualification_sources<>current_qualification_sources then
   raise exception 'Standings changed' using errcode='40001'; end if;
 end if;
 prior:=case when saved.id is null then null else to_jsonb(saved) end;
 old_document:=saved.document;
 if p_action in ('generate','save','refresh_lineups') then
  if saved.id is not null and saved.state<>'draft' then raise exception 'Reopen before editing' using errcode='40001'; end if;
  if p_document is null or jsonb_typeof(p_document)<>'object' or p_document->>'meet_id'<>p_meet_id::text
   or p_document->>'phase'<>p_phase or jsonb_typeof(p_document->'encounters')<>'array'
   or jsonb_array_length(p_document->'encounters')<1 then
   raise exception 'Invalid meet document' using errcode='22023'; end if;
  if p_roster_sources is null or jsonb_typeof(p_roster_sources)<>'array' or jsonb_array_length(p_roster_sources)<2 then
   raise exception 'Eligible meet rosters required' using errcode='22023'; end if;
  for source in select value from jsonb_array_elements(p_roster_sources) loop
   select * into source_team from public.pcs_interclub_current_rosters
    where id=(source->>'team_id')::uuid and season_id=p_season_id and meet_id=p_meet_id;
   if not found or source_team.revision<>(source->>'revision')::integer or source_team.withdrawn
    or source_team.status not in ('eligible','exception_approved') then
    raise exception 'Source lineup changed' using errcode='40001'; end if;
  end loop;
  for encounter in select value from jsonb_array_elements(p_document->'encounters') loop
   if encounter->>'club_a'=encounter->>'club_b' or not (season.details->'divisions' ? (encounter->>'division'))
    or not(meet.club_ids ? (encounter->>'club_a')) or not(meet.club_ids ? (encounter->>'club_b'))
    or (select count(*) from public.pcs_interclub_participations where season_id=p_season_id
      and club_id in(encounter->>'club_a',encounter->>'club_b') and status='accepted')<>2 then
    raise exception 'Results must use this meet and accepted clubs' using errcode='22023'; end if;
   if p_action<>'generate' and not exists(select 1 from jsonb_array_elements(old_document->'encounters') old
     where old->>'id'=encounter->>'id' and old->>'division'=encounter->>'division'
      and old->>'club_a'=encounter->>'club_a' and old->>'club_b'=encounter->>'club_b') then
    raise exception 'The generated schedule is fixed' using errcode='22023'; end if;
   for pairing in select value from jsonb_array_elements(encounter->'pairings') loop
    if p_action<>'generate' then
     select p into old_pairing from jsonb_array_elements(old_document->'encounters') e,
      lateral jsonb_array_elements(e->'pairings') p where e->>'id'=encounter->>'id' and p->>'kind'=pairing->>'kind';
     if old_pairing is null or old_pairing->>'id'<>pairing->>'id'
      or (select jsonb_agg(g->>'id' order by n) from jsonb_array_elements(old_pairing->'games') with ordinality gs(g,n))
       is distinct from (select jsonb_agg(g->>'id' order by n) from jsonb_array_elements(pairing->'games') with ordinality gs(g,n)) then
      raise exception 'Generated game identities cannot change' using errcode='22023'; end if;
    end if;
    if (pairing->>'eligibility_deadline')::timestamptz is distinct from meet.roster_deadline then
     select p into old_pairing from jsonb_array_elements(old_document->'encounters') e,
       lateral jsonb_array_elements(e->'pairings') p
       where e->>'id'=encounter->>'id' and p->>'kind'=pairing->>'kind';
     if old_pairing is distinct from pairing then raise exception 'Historical pairing cannot change' using errcode='22023'; end if;
    end if;
   end loop;
  end loop;
  if p_action<>'generate' and jsonb_array_length(p_document->'encounters')<>jsonb_array_length(old_document->'encounters') then
   raise exception 'Every scheduled matchup is required' using errcode='22023'; end if;
  if p_action='generate' and saved.id is not null and exists(select 1 from jsonb_array_elements(old_document->'encounters') e,
   lateral jsonb_array_elements(e->'pairings') p,lateral jsonb_array_elements(p->'games') g where g->>'status'<>'pending') then
   raise exception 'Scores already entered' using errcode='40001'; end if;
  next_state:='draft';
 elsif p_action in ('submit','approve') then
  if saved.state<>(case when p_action='submit' then 'draft' else 'submitted' end) then
   raise exception 'Result state changed' using errcode='40001'; end if;
  for source in select value from jsonb_array_elements(saved.roster_sources) loop
   if not exists(select 1 from public.pcs_interclub_current_rosters where id=(source->>'team_id')::uuid
    and revision=(source->>'revision')::integer and not withdrawn and status in ('eligible','exception_approved')) then
    raise exception 'Source lineup changed' using errcode='40001'; end if;
  end loop;
  if exists(select 1 from jsonb_array_elements(saved.document->'encounters') e,
    lateral jsonb_array_elements(e->'pairings') p,lateral jsonb_array_elements(p->'games') g where g->>'status'='pending') then
   raise exception 'Submit all official scores together' using errcode='22023'; end if;
  -- Defined by the eligibility migration; invoked only at runtime after all
  -- reviewed migrations are installed. It freezes historical deadline ratings.
  perform public.pcs_assert_interclub_competition_eligibility(p_season_id,p_meet_id,saved.document,p_phase);
  p_document:=saved.document; p_roster_sources:=saved.roster_sources;
  next_state:=case when p_action='approve' then 'approved' else 'submitted' end;
 elsif p_action='reopen' then
  if saved.state not in ('submitted','approved') or length(trim(coalesce(p_reason,'')))<3 then
   raise exception 'Choose a submitted result and correction reason' using errcode='22023'; end if;
  p_document:=saved.document; p_roster_sources:=saved.roster_sources; next_state:='draft';
 elsif p_action='reschedule' then
  if p_phase<>'regular' or saved.state='approved' or p_starts_at is null or p_deadline is null
   or p_deadline<=now() or p_deadline>p_starts_at or length(trim(coalesce(p_reason,'')))<3 then
   raise exception 'Choose the replay date and new deadline' using errcode='22023'; end if;
  if p_document->>'meet_id'<>p_meet_id::text or p_document->>'phase'<>'regular' or p_document->>'weather'<>'rescheduled' then
   raise exception 'Invalid replay document' using errcode='22023'; end if;
  for encounter in select value from jsonb_array_elements(saved.document->'encounters') loop
   for pairing in select value from jsonb_array_elements(encounter->'pairings') loop
    if not exists(select 1 from jsonb_array_elements(pairing->'games') g where g->>'status' in ('pending','unplayed')) then
     select p into old_pairing from jsonb_array_elements(p_document->'encounters') e,
      lateral jsonb_array_elements(e->'pairings') p where e->>'id'=encounter->>'id' and p->>'kind'=pairing->>'kind';
     if old_pairing is distinct from pairing then raise exception 'Completed pairings must stand' using errcode='22023'; end if;
    end if;
   end loop;
  end loop;
  perform public.pcs_lock_interclub_meet_eligibility(p_meet_id);
  update public.pcs_interclub_meets set starts_at=p_starts_at,roster_deadline=p_deadline,revision=revision+1 where id=p_meet_id;
  p_roster_sources:=saved.roster_sources; next_state:='draft';
 end if;
 if saved.id is null then
  insert into public.pcs_interclub_competition_batches(season_id,meet_id,phase,document,roster_sources)
   values(p_season_id,p_meet_id,p_phase,p_document,p_roster_sources) returning * into saved;
 else
  update public.pcs_interclub_competition_batches set revision=revision+1,state=next_state,
   document=p_document,roster_sources=p_roster_sources,updated_at=now(),
   approved_document=case when p_action='approve' then p_document else approved_document end,
   approved_revision=case when p_action='approve' then revision+1 else approved_revision end,
   approved_at=case when p_action='approve' then now() else approved_at end,
   ratings_status=case when p_action='approve' then 'pending' else ratings_status end,
   ratings_error=case when p_action='approve' then null else ratings_error end
   where id=saved.id returning * into saved;
 end if;
 if p_action='approve' then
  -- Corrections replace only this result batch's appearance facts. The identity
  -- lock trigger deliberately retains club representation once a player played.
  delete from public.pcs_interclub_appearances where batch_id=saved.id;
  for encounter in select value from jsonb_array_elements(saved.document->'encounters') loop
   for pairing in select value from jsonb_array_elements(encounter->'pairings') loop
    game_index:=0;
    for game in select value from jsonb_array_elements(pairing->'games') loop
     game_index:=game_index+1;
     if game->>'status' in ('completed','retired') then
      foreach side in array array['a','b'] loop
       represented:=encounter->>('club_'||side);
       lineup:=case when jsonb_array_length(coalesce(game->('players_'||side),'[]'))>0
        then game->('players_'||side) else pairing->('players_'||side) end;
       for entry_value in select jsonb_array_elements_text(lineup) loop
        select * into identity_row from public.pcs_interclub_entries
          where id=entry_value::uuid and season_id=p_season_id and club_id=represented;
        if not found then raise exception 'Player does not represent this club' using errcode='22023'; end if;
        insert into public.pcs_interclub_appearances(season_id,meet_id,entry_id,club_id,player_id,division,batch_id,revision,phase,game_id)
         values(p_season_id,p_meet_id,identity_row.id,represented,identity_row.player_id,encounter->>'division',saved.id,saved.revision,p_phase,
          game->>'id');
       end loop;
      end loop;
     end if;
    end loop;
   end loop;
  end loop;
 end if;
 insert into public.pcs_interclub_competition_audit(batch_id,revision,actor_id,actor_club_id,action,reason,before_document,after_document)
 values(saved.id,saved.revision,p_actor_id,p_club_id,p_action,p_reason,prior,to_jsonb(saved));
 return to_jsonb(saved);
end $$;
revoke all on function public.pcs_write_interclub_competition(uuid,text,text,uuid,uuid,text,text,integer,jsonb,jsonb,text,timestamptz,timestamptz,jsonb) from public,anon,authenticated;
grant execute on function public.pcs_write_interclub_competition(uuid,text,text,uuid,uuid,text,text,integer,jsonb,jsonb,text,timestamptz,timestamptz,jsonb) to service_role;
comment on table public.pcs_interclub_competition_batches is 'Private paper score batches; approved_document stays official while a correction is drafted. Exact revision approval is required for rating effects.';
commit;
