begin;

create table public.pcs_interclub_award_sets (
 season_id uuid primary key references public.pcs_interclub_seasons(id),
 revision integer not null check(revision>0),
 preview_fingerprint text not null check(preview_fingerprint ~ '^[a-f0-9]{64}$'),
 publication_digest text not null,
 awards jsonb not null check(jsonb_typeof(awards)='array'),
 issued_at timestamptz not null default now(),
 updated_at timestamptz not null default now(),
 issued_by uuid not null
);
create table public.pcs_interclub_awards (
 id uuid primary key,
 season_id uuid not null references public.pcs_interclub_award_sets(season_id),
 club_id text not null references public.clubs(id),
 entry_id uuid references public.pcs_interclub_entries(id),
 player_id bigint,
 recipient_type text not null check(recipient_type in ('club','player')),
 recipient_key text generated always as (coalesce(entry_id::text,'club:'||club_id)) stored,
 recipient_name text not null check(length(recipient_name) between 1 and 300),
 award_key text not null check(award_key in ('participation','division_champion','club_cup_champion')),
 division text not null default '',
 title text not null check(length(title) between 1 and 200),
 earned_at timestamptz not null default now(),
 foreign key(club_id,player_id) references public.players(club_id,id),
 unique(season_id,recipient_key,award_key,division),
 check((recipient_type='club' and entry_id is null and player_id is null) or
       (recipient_type='player' and entry_id is not null and player_id is not null)),
 check((award_key='division_champion' and division<>'') or (award_key<>'division_champion' and division=''))
);
create index pcs_interclub_awards_player_idx on public.pcs_interclub_awards(club_id,player_id,earned_at desc) where player_id is not null;
create index pcs_interclub_awards_club_idx on public.pcs_interclub_awards(club_id,earned_at desc) where recipient_type='club';
create index pcs_interclub_awards_entry_idx on public.pcs_interclub_awards(entry_id) where entry_id is not null;
alter table public.pcs_interclub_award_sets enable row level security;
alter table public.pcs_interclub_awards enable row level security;
revoke all on public.pcs_interclub_award_sets,public.pcs_interclub_awards from public,anon,authenticated;
grant all on public.pcs_interclub_award_sets,public.pcs_interclub_awards to service_role;

-- A corrected/unpublished public snapshot hides old honors until the organizer
-- reviews the replacements. A private correction draft retains public honors.
create view public.pcs_public_interclub_awards with(security_invoker=true) as
 select a.id,a.season_id,a.club_id,a.entry_id,a.player_id,a.recipient_type,a.recipient_name,
 a.award_key,a.division,a.title,a.earned_at,p.published->>'name' as season_name
 from public.pcs_interclub_awards a
 join public.pcs_interclub_award_sets s on s.season_id=a.season_id
 join public.pcs_interclub_publications p on p.season_id=a.season_id
 where p.published is not null and md5(p.published::text)=s.publication_digest;
revoke all on public.pcs_public_interclub_awards from public,anon,authenticated;
grant select on public.pcs_public_interclub_awards to service_role;

create function public.pcs_award_interclub_season(
 p_actor_id uuid,p_actor_email text,p_club_id text,p_season_id uuid,p_revision integer,p_publication_revision integer,
 p_preview_fingerprint text,p_document jsonb,p_sources jsonb,p_awards jsonb
) returns jsonb language plpgsql security invoker set search_path=public as $$
declare saved public.pcs_interclub_award_sets; publication public.pcs_interclub_publications;
 next_revision integer; publication_revision integer; result jsonb; item jsonb; entry public.pcs_interclub_entries;
begin
 perform public.pcs_require_interclub_admin(p_actor_id,p_actor_email,p_club_id);
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||p_season_id::text,0));
 if not exists(select 1 from public.pcs_interclub_seasons where id=p_season_id and organizer_club_id=p_club_id) then
  raise exception 'Season organizer required' using errcode='42501'; end if;
 if p_document->>'season_complete' is distinct from 'true' or p_document->'club_cup'->>'status' is distinct from 'complete'
  or p_preview_fingerprint is null or p_preview_fingerprint !~ '^[a-f0-9]{64}$'
  or jsonb_typeof(p_awards) is distinct from 'array' or jsonb_array_length(p_awards) not between 1 and 10000
  or not exists(select 1 from public.pcs_interclub_meets where season_id=p_season_id)
  or exists(select 1 from public.pcs_interclub_meets m left join public.pcs_interclub_competition_batches b
    on b.meet_id=m.id and b.season_id=m.season_id and b.phase=m.competition_phase where m.season_id=p_season_id and
    (b.id is null or b.state<>'approved' or b.approved_revision is distinct from b.revision or b.ratings_status<>'completed'
     or b.document->>'weather'='rescheduled')) then
  raise exception 'Finish and approve the whole season before awarding trophies' using errcode='22023'; end if;
 select * into saved from public.pcs_interclub_award_sets where season_id=p_season_id for update;
 select * into publication from public.pcs_interclub_publications where season_id=p_season_id for update;
 if saved.season_id is not null and saved.preview_fingerprint=p_preview_fingerprint and saved.awards=p_awards
  and publication.published=p_document then
  return jsonb_build_object('revision',saved.revision,'issued_at',saved.issued_at,'awards',jsonb_array_length(saved.awards),'unchanged',true);
 end if;
 if coalesce(saved.revision,0)<>p_revision or coalesce(publication.revision,0)<>p_publication_revision then
  raise exception 'Awards or publication changed; reload the preview' using errcode='PT409'; end if;
 if (select count(distinct value->>'id') from jsonb_array_elements(p_awards))<>jsonb_array_length(p_awards) then
  raise exception 'Duplicate trophy identity' using errcode='22023'; end if;
 for item in select value from jsonb_array_elements(p_awards) loop
  if not exists(select 1 from public.pcs_interclub_participations where season_id=p_season_id
    and club_id=item->>'club_id' and status='accepted') then
   raise exception 'Recipient club is outside this season' using errcode='22023'; end if;
  if item->>'recipient_type'='player' then
   select * into entry from public.pcs_interclub_entries where id=(item->>'entry_id')::uuid
    and season_id=p_season_id and club_id=item->>'club_id';
   if not found then raise exception 'Recipient player is outside this club or season' using errcode='22023'; end if;
  elsif item->>'recipient_type'<>'club' or item->>'entry_id' is not null then
   raise exception 'Invalid trophy recipient' using errcode='22023'; end if;
  if exists(select 1 from public.pcs_interclub_awards where id=(item->>'id')::uuid and season_id<>p_season_id) then
   raise exception 'Trophy identity belongs to another season' using errcode='22023'; end if;
 end loop;
 publication_revision:=p_publication_revision;
 if publication.season_id is null then
  result:=public.pcs_write_interclub_publication(p_actor_id,p_actor_email,p_club_id,p_season_id,0,'save','{"results":[]}'::jsonb);
  publication_revision:=(result->>'revision')::integer;
 end if;
 -- This call rechecks the exact approved revisions, season, clubs and calendar
 -- under the same season lock. Publication and every trophy commit together.
 perform public.pcs_publish_reviewed_interclub_publication(p_actor_id,p_actor_email,p_club_id,p_season_id,
  publication_revision,p_document,p_sources);
 next_revision:=coalesce(saved.revision,0)+1;
 insert into public.pcs_interclub_award_sets(season_id,revision,preview_fingerprint,publication_digest,awards,issued_by)
 values(p_season_id,next_revision,p_preview_fingerprint,md5(p_document::text),p_awards,p_actor_id)
 on conflict(season_id) do update set revision=excluded.revision,preview_fingerprint=excluded.preview_fingerprint,
  publication_digest=excluded.publication_digest,awards=excluded.awards,issued_by=excluded.issued_by,updated_at=now();
 delete from public.pcs_interclub_awards where season_id=p_season_id and id not in
  (select (value->>'id')::uuid from jsonb_array_elements(p_awards));
 insert into public.pcs_interclub_awards(id,season_id,club_id,entry_id,player_id,recipient_type,recipient_name,award_key,division,title)
 select (a->>'id')::uuid,p_season_id,a->>'club_id',(a->>'entry_id')::uuid,e.player_id,
  a->>'recipient_type',a->>'recipient_name',a->>'award_key',a->>'division',a->>'title'
 from jsonb_array_elements(p_awards) a left join public.pcs_interclub_entries e on e.id=(a->>'entry_id')::uuid
 on conflict(id) do update set recipient_name=excluded.recipient_name,title=excluded.title;
 insert into public.pcs_interclub_registration_audit(season_id,actor_id,actor_club_id,action,details)
 values(p_season_id,p_actor_id,p_club_id,'season_trophies_awarded',jsonb_build_object('revision',next_revision,
  'preview_fingerprint',p_preview_fingerprint,'before_awards',coalesce(saved.awards,'[]'::jsonb),'after_awards',p_awards));
 return jsonb_build_object('revision',next_revision,'awards',jsonb_array_length(p_awards),'unchanged',false);
end $$;
revoke all on function public.pcs_award_interclub_season(uuid,text,text,uuid,integer,integer,text,jsonb,jsonb,jsonb) from public,anon,authenticated;
grant execute on function public.pcs_award_interclub_season(uuid,text,text,uuid,integer,integer,text,jsonb,jsonb,jsonb) to service_role;

commit;
