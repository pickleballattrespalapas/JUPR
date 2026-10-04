begin;

-- A season has one commissioner-controlled enrollment window. Existing seasons
-- deliberately remain unconfigured: neither dates nor permission are inferred.
alter table public.pcs_interclub_seasons
 add column registration_opens_at timestamptz,
 add column registration_closes_at timestamptz,
 add column registration_revision integer not null default 0 check(registration_revision>=0),
 add constraint pcs_interclub_registration_window_check check(
  (registration_opens_at is null and registration_closes_at is null) or
  (registration_opens_at is not null and registration_closes_at is not null
   and isfinite(registration_opens_at) and isfinite(registration_closes_at)
   and registration_opens_at<registration_closes_at));

create function public.pcs_interclub_registration_phase(p_opens_at timestamptz,p_closes_at timestamptz,p_at timestamptz default now())
returns text language sql immutable security invoker set search_path=public as $$
 select case when p_opens_at is null or p_closes_at is null then 'unconfigured'
  when p_at<p_opens_at then 'scheduled' when p_at<p_closes_at then 'open' else 'closed' end
$$;

-- Every protected RPC takes the same season lock before inspecting the window.
-- This prevents a simultaneous commissioner edit from racing a meet/pool write.
create function public.pcs_require_interclub_registration_phase(p_season_id uuid,p_expected text)
returns void language plpgsql security invoker set search_path=public as $$
declare season public.pcs_interclub_seasons;
begin
 if p_expected is null or p_expected not in ('open','closed') then raise exception 'Invalid registration phase requirement' using errcode='22023'; end if;
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||p_season_id::text,0));
 select * into season from public.pcs_interclub_seasons where id=p_season_id;
 if not found then raise exception 'Season unavailable' using errcode='P0002'; end if;
 if public.pcs_interclub_registration_phase(season.registration_opens_at,season.registration_closes_at,clock_timestamp())<>p_expected then
  if p_expected='open' then
   raise exception 'Season registration is not open. The commissioner sets the registration dates for every club.' using errcode='PT423';
  else
   raise exception 'Meet operations unlock after the commissioner-set season registration period closes.' using errcode='PT423';
  end if;
 end if;
end $$;

create function public.pcs_set_interclub_registration_window(p_actor_id uuid,p_actor_email text,p_club_id text,p_season_id uuid,
 p_revision integer,p_opens_at timestamptz,p_closes_at timestamptz)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare season public.pcs_interclub_seasons; earliest_meet timestamptz; prior jsonb;
begin
 perform public.pcs_require_interclub_admin(p_actor_id,p_actor_email,p_club_id);
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||p_season_id::text,0));
 select * into season from public.pcs_interclub_seasons where id=p_season_id for update;
 if not found then raise exception 'Season unavailable' using errcode='P0002'; end if;
 if season.organizer_club_id<>p_club_id then raise exception 'Only the season commissioner sets registration dates' using errcode='42501'; end if;
 if season.registration_revision is distinct from p_revision then raise exception 'Registration dates changed. Reload before saving.' using errcode='PT409'; end if;
 if p_opens_at is null or p_closes_at is null or not isfinite(p_opens_at) or not isfinite(p_closes_at) or p_opens_at>=p_closes_at then
  raise exception 'Choose a registration opening date before the closing date' using errcode='22023'; end if;
 select min(starts_at) into earliest_meet from public.pcs_interclub_meets where season_id=p_season_id;
 if earliest_meet is not null and p_closes_at>earliest_meet then
  raise exception 'Season registration must close by the first meet' using errcode='22023'; end if;
 prior:=jsonb_build_object('opens_at',season.registration_opens_at,'closes_at',season.registration_closes_at,'revision',season.registration_revision);
 update public.pcs_interclub_seasons set registration_opens_at=p_opens_at,registration_closes_at=p_closes_at,
  registration_revision=registration_revision+1 where id=p_season_id returning * into season;
 insert into public.pcs_interclub_registration_audit(season_id,actor_id,actor_club_id,action,details)
 values(p_season_id,p_actor_id,p_club_id,'season_registration_window',jsonb_build_object('before',prior,
  'after',jsonb_build_object('opens_at',season.registration_opens_at,'closes_at',season.registration_closes_at,'revision',season.registration_revision)));
 return jsonb_build_object('registration_opens_at',season.registration_opens_at,'registration_closes_at',season.registration_closes_at,
  'registration_revision',season.registration_revision,'registration_phase',public.pcs_interclub_registration_phase(season.registration_opens_at,season.registration_closes_at));
end $$;

-- A club's signup link exists as soon as its season invitation is accepted.
-- The legacy per-club open flag is retained only for storage compatibility.
create function public.pcs_ensure_interclub_pool_settings() returns trigger
language plpgsql security invoker set search_path=public as $$
begin
 if new.status='accepted' then
  insert into public.pcs_interclub_pool_settings(season_id,club_id) values(new.season_id,new.club_id)
  on conflict(season_id,club_id) do nothing;
 end if;
 return new;
end $$;
create trigger pcs_interclub_participation_pool_settings after insert or update of status on public.pcs_interclub_participations
 for each row execute function public.pcs_ensure_interclub_pool_settings();
insert into public.pcs_interclub_pool_settings(season_id,club_id)
 select season_id,club_id from public.pcs_interclub_participations where status='accepted'
 on conflict(season_id,club_id) do nothing;
comment on column public.pcs_interclub_pool_settings.open is 'Retired. Signup availability is governed only by the season registration window.';

-- Enforce additions and restoration at the row boundary as well as RPC entry.
-- Linking/correcting/approving an existing active member and withdrawals remain
-- possible after enrollment closes, preserving existing eligibility safeguards.
create function public.pcs_guard_interclub_pool_registration() returns trigger
language plpgsql security invoker set search_path=public as $$
begin
 if tg_op='INSERT' then
  perform public.pcs_require_interclub_registration_phase(new.season_id,'open');
 elsif old.status='withdrawn' and new.status='active' then
  perform public.pcs_require_interclub_registration_phase(new.season_id,'open');
 end if;
 return new;
end $$;
create trigger pcs_interclub_pool_registration before insert or update of status on public.pcs_interclub_pool_members
 for each row execute function public.pcs_guard_interclub_pool_registration();

-- Preserve every installed eligibility/authorization check and non-retryable
-- conflict code. Change only the exact, verified insertion anchors below.
-- Closed-phase entrypoint inventory: meet deadline; roster save/withdraw;
-- roster exception review; competition meet creation; every competition action;
-- direct meet rating/snapshot reads; competition eligibility/snapshot check.
do $patch$
declare signature text; definition text; anchor text; replacement text; routine regprocedure; touched integer:=0;
begin
 foreach signature in array array[
  'public.pcs_set_interclub_meet_deadline(uuid,text,text,uuid,uuid,integer,timestamptz)',
  'public.pcs_save_interclub_meet_roster(uuid,text,text,uuid,uuid,integer,uuid,integer,text,text,bigint[],boolean)',
  'public.pcs_review_interclub_meet_roster(uuid,text,text,uuid,uuid,integer,uuid,integer,boolean,text)',
  'public.pcs_create_interclub_competition_meet(uuid,text,text,uuid,jsonb)',
  'public.pcs_write_interclub_competition(uuid,text,text,uuid,uuid,text,text,integer,jsonb,jsonb,text,timestamptz,timestamptz,jsonb)'
 ] loop
  routine:=to_regprocedure(signature);
  if routine is null then raise exception 'Required registration gate routine missing: %',signature; end if;
  definition:=pg_get_functiondef(routine);
  anchor:=' perform pg_advisory_xact_lock(hashtextextended(''pcs-season:''||p_season_id::text,0));';
  if (length(definition)-length(replace(definition,anchor,'')))/length(anchor)<>1 then raise exception 'Expected one season lock in %',signature; end if;
  replacement:=anchor||E'\n perform public.pcs_require_interclub_registration_phase(p_season_id,''closed'');';
  execute replace(definition,anchor,replacement); touched:=touched+1;
 end loop;
 foreach signature in array array[
  'public.pcs_interclub_meet_player_ratings(uuid,uuid,text)',
  'public.pcs_assert_interclub_competition_eligibility(uuid,uuid,jsonb,text)'
 ] loop
  routine:=to_regprocedure(signature);
  if routine is null then raise exception 'Required meet snapshot routine missing: %',signature; end if;
  definition:=pg_get_functiondef(routine);
  anchor:=E'begin\n';
  if (length(definition)-length(replace(definition,anchor,'')))/length(anchor)<>1 then raise exception 'Expected one function body in %',signature; end if;
  execute replace(definition,anchor,anchor||E' perform public.pcs_require_interclub_registration_phase(p_season_id,''closed'');\n'); touched:=touched+1;
 end loop;
 if touched<>7 then raise exception 'Expected seven closed-phase RPC entrypoints, found %',touched; end if;
end $patch$;

-- Pool action retains administrator checks; per-club toggles are removed.
-- Meet invitations/availability require closed; share-link rotation is allowed
-- throughout the season and never changes commissioner registration dates.
do $patch$
declare definition text; anchor text; replacement text; first_pos integer; last_pos integer; routine regprocedure;
begin
 routine:=to_regprocedure('public.pcs_interclub_pool_action(uuid,text,text,uuid,text,jsonb)');
 if routine is null then raise exception 'Pool action routine missing'; end if;
 definition:=pg_get_functiondef(routine);
 anchor:=E' if p_action=''settings'' then\n';
 first_pos:=position(anchor in definition);
 last_pos:=position(E' elsif p_action=''member'' then\n' in definition);
 if first_pos=0 or last_pos<=first_pos or (length(definition)-length(replace(definition,anchor,'')))/length(anchor)<>1 then
  raise exception 'Pool settings branch changed'; end if;
 replacement:=$branch$ if p_action in ('availability','invite') then
  perform public.pcs_require_interclub_registration_phase(p_season_id,'closed');
 end if;
 if p_action='settings' then
  if p_payload ? 'open' then raise exception 'Only the commissioner controls season registration dates' using errcode='42501'; end if;
  select * into settings from public.pcs_interclub_pool_settings where season_id=p_season_id and club_id=p_club_id for update;
  if not found then raise exception 'Signup link unavailable' using errcode='P0002'; end if;
  if settings.revision is distinct from (p_payload->>'expected_revision')::integer then raise exception 'Settings changed' using errcode='PT409'; end if;
  if (p_payload->>'rotate_link')::boolean is not true then raise exception 'Choose to replace the signup link' using errcode='22023'; end if;
  update public.pcs_interclub_pool_settings set share_id=gen_random_uuid(),revision=revision+1,updated_at=now()
   where season_id=p_season_id and club_id=p_club_id returning * into settings;
  result:=to_jsonb(settings);
$branch$;
 execute substring(definition from 1 for first_pos-1)||replacement||substring(definition from last_pos);
end $patch$;

-- Public intake ignores retired settings.open; signed capability withdrawal
-- remains available outside the window. Meet RSVP is a meet operation.
do $patch$
declare definition text; anchor text; replacement text; routine regprocedure;
begin
 routine:=to_regprocedure('public.pcs_interclub_pool_public_action(text,jsonb,text)');
 if routine is null then raise exception 'Public pool routine missing'; end if;
 definition:=pg_get_functiondef(routine);
 anchor:=' perform pg_advisory_xact_lock(hashtextextended(''pcs-season:''||sid::text,0));';
 if (length(definition)-length(replace(definition,anchor,'')))/length(anchor)<>1 then raise exception 'Public pool season lock changed'; end if;
 replacement:=anchor||$branch$
 if p_action='signup' or (p_action='update_season' and p_payload->>'status'<>'withdrawn') then
  perform public.pcs_require_interclub_registration_phase(sid,'open');
 elsif p_action='respond_meet' then
  perform public.pcs_require_interclub_registration_phase(sid,'closed');
 end if;$branch$;
 definition:=replace(definition,anchor,replacement);
 anchor:='if settings.open is not true or settings.share_id<>(p_payload->>''share_id'')::uuid then';
 if (length(definition)-length(replace(definition,anchor,'')))/length(anchor)<>1 then raise exception 'Public signup availability check changed'; end if;
 definition:=replace(definition,anchor,'if settings.share_id is distinct from (p_payload->>''share_id'')::uuid then');
 anchor:='  if settings.open is not true and p_payload->>''status''<>''withdrawn'' then raise exception ''Season signup is closed'' using errcode=''PT409''; end if;';
 if (length(definition)-length(replace(definition,anchor,'')))/length(anchor)<>1 then raise exception 'Public edit availability check changed'; end if;
 definition:=replace(definition,anchor,'  -- Active edits were checked against the central window above.');
 anchor:='  if member.revision is distinct from (p_payload->>''expected_revision'')::integer then raise exception ''Your response changed'' using errcode=''PT409''; end if;';
 if (length(definition)-length(replace(definition,anchor,'')))/length(anchor)<>1 then raise exception 'Public member revision check changed'; end if;
 replacement:=anchor||$branch$
  if p_payload->>'status'='withdrawn' and public.pcs_interclub_registration_phase(season.registration_opens_at,season.registration_closes_at,clock_timestamp())<>'open'
   and (clean_name is distinct from trim(regexp_replace(member.name,'\s+',' ','g')) or clean_email is distinct from member.email
    or clean_divisions is distinct from member.divisions or coalesce(p_payload->>'notes','') is distinct from member.notes) then
   raise exception 'Season registration is not open. Only withdrawal is available; saved signup details cannot change.' using errcode='PT423';
  end if;$branch$;
 definition:=replace(definition,anchor,replacement);
 execute definition;
 routine:=to_regprocedure('public.pcs_interclub_pool_bulk_add(uuid,text,text,uuid,jsonb)');
 if routine is null then raise exception 'Bulk pool routine missing'; end if;
 definition:=pg_get_functiondef(routine);
 anchor:=' perform pg_advisory_xact_lock(hashtextextended(''pcs-season:''||p_season_id::text,0));';
 if (length(definition)-length(replace(definition,anchor,'')))/length(anchor)<>1 then raise exception 'Bulk season lock changed'; end if;
 execute replace(definition,anchor,anchor||E'\n perform public.pcs_require_interclub_registration_phase(p_season_id,''open'');');
end $patch$;

-- Explicit meet rating approval/retry is gated, while the shared no-batch
-- local club replay path remains available. The gate precedes rating table
-- locks, retaining the season-before-table lock order of meet writers.
do $patch$
declare definition text; anchor text; replacement text; routine regprocedure;
begin
 routine:=to_regprocedure('public.pcs_apply_interclub_rating_projection(text[],text,jsonb,uuid,integer)');
 if routine is null then raise exception 'Interclub rating projection routine missing'; end if;
 definition:=pg_get_functiondef(routine);
 anchor:=E'begin\n';
 if (length(definition)-length(replace(definition,anchor,'')))/length(anchor)<>1 then raise exception 'Rating projection body changed'; end if;
 replacement:=anchor||$branch$ if p_expected_batch is not null then
  perform public.pcs_require_interclub_registration_phase(
   (select season_id from public.pcs_interclub_competition_batches where id=p_expected_batch),'closed');
 end if;
$branch$;
 execute replace(definition,anchor,replacement);
 routine:=to_regprocedure('public.pcs_fail_interclub_ratings(uuid,integer,text)');
 if routine is null then raise exception 'Interclub rating status routine missing'; end if;
 definition:=pg_get_functiondef(routine);
 if (length(definition)-length(replace(definition,anchor,'')))/length(anchor)<>1 then raise exception 'Rating status body changed'; end if;
 execute replace(definition,anchor,anchor||$branch$ perform public.pcs_require_interclub_registration_phase(
  (select season_id from public.pcs_interclub_competition_batches where id=p_batch_id),'closed');
$branch$);
end $patch$;

-- Schedule-only publication is part of planning and remains available. Any
-- result-bearing write (including removal/unpublish of saved results) requires
-- closed enrollment. Reviewed publication calls this same protected writer.
do $patch$
declare definition text; anchor text; replacement text; routine regprocedure;
begin
 routine:=to_regprocedure('public.pcs_write_interclub_publication(uuid,text,text,uuid,integer,text,jsonb)');
 if routine is null then raise exception 'Interclub publication writer missing'; end if;
 definition:=pg_get_functiondef(routine);
 anchor:=' perform pg_advisory_xact_lock(hashtextextended(''pcs-season:''||p_season_id::text,0));';
 if (length(definition)-length(replace(definition,anchor,'')))/length(anchor)<>1 then raise exception 'Publication season lock changed'; end if;
 replacement:=anchor||$branch$
 if coalesce(p_document->'results','[]'::jsonb)<>'[]'::jsonb
  or coalesce(p_document->'competition_results','[]'::jsonb)<>'[]'::jsonb
  or exists(select 1 from public.pcs_interclub_competition_batches where season_id=p_season_id and approved_document is not null)
  or exists(select 1 from public.pcs_interclub_publications where season_id=p_season_id and (
   coalesce(draft->'results','[]'::jsonb)<>'[]'::jsonb or coalesce(draft->'competition_results','[]'::jsonb)<>'[]'::jsonb
   or coalesce(published->'results','[]'::jsonb)<>'[]'::jsonb or coalesce(published->'competition_results','[]'::jsonb)<>'[]'::jsonb)) then
  perform public.pcs_require_interclub_registration_phase(p_season_id,'closed');
 end if;$branch$;
 execute replace(definition,anchor,replacement);
end $patch$;

-- Workspace permission hints use the same central clock as actual writes.
create or replace view public.pcs_interclub_meet_workspaces with(security_invoker=true) as
 select m.id,m.season_id,m.plan_index,m.host_club_id,m.club_ids,m.starts_at,m.duration_minutes,m.courts,m.roster_deadline,m.revision,
 m.starts_at>now() and public.pcs_interclub_registration_phase(s.registration_opens_at,s.registration_closes_at)='closed' as roster_open,
 m.starts_at>now() and public.pcs_interclub_registration_phase(s.registration_opens_at,s.registration_closes_at)='closed'
  and not exists(select 1 from public.pcs_interclub_teams t where t.meet_id=m.id) as deadline_editable,
 m.competition_phase from public.pcs_interclub_meets m join public.pcs_interclub_seasons s on s.id=m.season_id;
revoke all on public.pcs_interclub_meet_workspaces from public,anon,authenticated;
grant select on public.pcs_interclub_meet_workspaces to service_role;

revoke all on function public.pcs_interclub_registration_phase(timestamptz,timestamptz,timestamptz),
 public.pcs_require_interclub_registration_phase(uuid,text),
 public.pcs_set_interclub_registration_window(uuid,text,text,uuid,integer,timestamptz,timestamptz),
 public.pcs_ensure_interclub_pool_settings(),public.pcs_guard_interclub_pool_registration() from public,anon,authenticated;
grant execute on function public.pcs_interclub_registration_phase(timestamptz,timestamptz,timestamptz),
 public.pcs_require_interclub_registration_phase(uuid,text),
 public.pcs_set_interclub_registration_window(uuid,text,text,uuid,integer,timestamptz,timestamptz),
 public.pcs_ensure_interclub_pool_settings(),public.pcs_guard_interclub_pool_registration() to service_role;
notify pgrst,'reload schema';
commit;
