begin;
create table public.pcs_interclub_drafts (
 id uuid primary key, organizer_club_id text not null references public.clubs(id),
 revision integer not null default 1 check(revision>0),
 draft jsonb not null check(jsonb_typeof(draft)='object'),
 updated_at timestamptz not null default now()
);
create index pcs_interclub_drafts_organizer on public.pcs_interclub_drafts(organizer_club_id,updated_at desc);
alter table public.pcs_interclub_drafts enable row level security;
revoke all on public.pcs_interclub_drafts from public,anon,authenticated;
grant all on public.pcs_interclub_drafts to service_role;
create table public.pcs_interclub_draft_audit (
 id bigint generated always as identity primary key, season_id uuid not null references public.pcs_interclub_drafts(id),
 actor_id uuid not null, revision integer not null, draft jsonb not null, created_at timestamptz not null default now()
);
alter table public.pcs_interclub_draft_audit enable row level security;
revoke all on public.pcs_interclub_draft_audit from public,anon,authenticated;
grant all on public.pcs_interclub_draft_audit to service_role;
grant usage,select on sequence public.pcs_interclub_draft_audit_id_seq to service_role;
create function public.pcs_save_interclub_draft(p_actor_id uuid,p_actor_email text,p_club_id text,p_id uuid,p_revision integer,p_draft jsonb)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare saved public.pcs_interclub_drafts; selected_club text;
begin
 if not exists(select 1 from public.admin_role_assignments where club_id=p_club_id and email=lower(trim(p_actor_email)) and (user_id is null or user_id=p_actor_id) and role in ('super_admin','administrator','club_owner') and revoked_at is null and (expires_at is null or expires_at>now())) then
  raise exception 'Organizer administrator required' using errcode='42501'; end if;
 if jsonb_typeof(p_draft)<>'object' or jsonb_typeof(p_draft->'club_ids')<>'array' then raise exception 'Invalid draft' using errcode='22023'; end if;
 for selected_club in select jsonb_array_elements_text(p_draft->'club_ids') loop
  perform 1 from public.clubs where id=selected_club for key share;
  if not found then raise exception 'Unknown club' using errcode='22023'; end if;
 end loop;
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||p_id::text,0));
 select * into saved from public.pcs_interclub_drafts where id=p_id;
 if found then
  if saved.organizer_club_id<>p_club_id then raise exception 'Organizer mismatch' using errcode='42501'; end if;
  if saved.revision<>p_revision then raise exception 'Stale draft' using errcode='40001'; end if;
  update public.pcs_interclub_drafts set draft=p_draft,revision=revision+1,updated_at=now() where id=p_id returning * into saved;
 else
  if p_revision<>0 then raise exception 'Stale draft' using errcode='40001'; end if;
  insert into public.pcs_interclub_drafts(id,organizer_club_id,draft) values(p_id,p_club_id,p_draft) returning * into saved;
 end if;
 insert into public.pcs_interclub_draft_audit(season_id,actor_id,revision,draft) values(saved.id,p_actor_id,saved.revision,saved.draft);
 return to_jsonb(saved);
end $$;
revoke all on function public.pcs_save_interclub_draft(uuid,text,text,uuid,integer,jsonb) from public,anon,authenticated;
grant execute on function public.pcs_save_interclub_draft(uuid,text,text,uuid,integer,jsonb) to service_role;
create function public.pcs_update_club_profile(p_actor_id uuid,p_club_id text,p_name text,p_support_email text)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare c public.clubs; old_details jsonb;
begin
 if not exists(select 1 from public.pcs_platform_admins where user_id=p_actor_id and revoked_at is null) then raise exception 'Platform access required' using errcode='42501'; end if;
 if length(trim(p_name)) not between 1 and 120 or p_support_email !~ '^[^\s@]+@[^\s@]+\.[^\s@]+$' then raise exception 'Invalid profile' using errcode='22023'; end if;
 select * into c from public.clubs where id=p_club_id for update;
 if not found then raise exception 'Club not found' using errcode='P0002'; end if;
 old_details=jsonb_build_object('name',c.name,'support_email',c.support_email);
 update public.clubs set name=trim(p_name),support_email=lower(trim(p_support_email)),updated_at=now() where id=p_club_id returning * into c;
 insert into public.pcs_platform_audit(actor_id,club_id,action,details) values(p_actor_id,p_club_id,'profile',jsonb_build_object('before',old_details,'after',jsonb_build_object('name',c.name,'support_email',c.support_email)));
 return to_jsonb(c);
end $$;
revoke all on function public.pcs_update_club_profile(uuid,text,text,text) from public,anon,authenticated;
grant execute on function public.pcs_update_club_profile(uuid,text,text,text) to service_role;
commit;
