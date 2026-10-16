begin;
create table public.pcs_interclub_publications (
 season_id uuid primary key references public.pcs_interclub_seasons(id),
 revision integer not null default 1,
 draft jsonb not null default '{"results":[]}',
 published jsonb,
 published_at timestamptz,
 updated_at timestamptz not null default now()
);
alter table public.pcs_interclub_publications enable row level security;
revoke all on public.pcs_interclub_publications from public,anon,authenticated;
grant all on public.pcs_interclub_publications to service_role;
create function public.pcs_write_interclub_publication(p_actor_id uuid,p_actor_email text,p_club_id text,
 p_season_id uuid,p_revision integer,p_action text,p_document jsonb) returns jsonb
language plpgsql security invoker set search_path=public as $$
declare publication public.pcs_interclub_publications; season public.pcs_interclub_seasons;
begin
 perform public.pcs_require_interclub_admin(p_actor_id,p_actor_email,p_club_id);
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||p_season_id::text,0));
 select * into season from public.pcs_interclub_seasons where id=p_season_id for share;
 if not found then raise exception 'Season unavailable' using errcode='P0002'; end if;
 if season.organizer_club_id<>p_club_id then raise exception 'Organizer required' using errcode='42501'; end if;
 select * into publication from public.pcs_interclub_publications where season_id=p_season_id for update;
 if coalesce(publication.revision,0)<>p_revision then raise exception 'Draft changed' using errcode='40001'; end if;
 if p_action='save' then
  insert into public.pcs_interclub_publications(season_id,draft) values(p_season_id,p_document)
   on conflict(season_id) do update set draft=excluded.draft,revision=pcs_interclub_publications.revision+1,updated_at=now();
 elsif p_action='publish' then
  if publication.season_id is null then raise exception 'Save draft first' using errcode='22023'; end if;
  -- Snapshot carries public schedule/results only; no roster/contact records.
  update public.pcs_interclub_publications set published=p_document,published_at=now(),revision=revision+1,updated_at=now() where season_id=p_season_id;
 elsif p_action='unpublish' then
  update public.pcs_interclub_publications set published=null,published_at=null,revision=revision+1,updated_at=now() where season_id=p_season_id;
 else raise exception 'Invalid action' using errcode='22023'; end if;
 insert into public.pcs_platform_audit(actor_id,club_id,action,details)
 values(p_actor_id,p_club_id,'interclub_site_'||p_action,jsonb_build_object('season_id',p_season_id));
 select * into publication from public.pcs_interclub_publications where season_id=p_season_id;
 return to_jsonb(publication);
end $$;
revoke all on function public.pcs_write_interclub_publication(uuid,text,text,uuid,integer,text,jsonb) from public,anon,authenticated;
grant execute on function public.pcs_write_interclub_publication(uuid,text,text,uuid,integer,text,jsonb) to service_role;
commit;
