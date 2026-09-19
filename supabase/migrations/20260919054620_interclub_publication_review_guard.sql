begin;

-- The preview's exact official revisions must still be current when the public
-- snapshot commits. Use the same season lock as score approval and meet edits.
create function public.pcs_publish_reviewed_interclub_publication(
 p_actor_id uuid,p_actor_email text,p_club_id text,p_season_id uuid,
 p_revision integer,p_document jsonb,p_sources jsonb
) returns jsonb language plpgsql security invoker set search_path=public as $$
declare current_sources jsonb;
begin
 perform public.pcs_require_interclub_admin(p_actor_id,p_actor_email,p_club_id);
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||p_season_id::text,0));
 if not exists(select 1 from public.pcs_interclub_seasons where id=p_season_id and organizer_club_id=p_club_id) then
  raise exception 'Organizer required' using errcode='42501'; end if;
 perform 1 from public.pcs_interclub_seasons where id=p_season_id for share;
 perform 1 from public.pcs_interclub_meets where season_id=p_season_id for share;
 perform 1 from public.pcs_interclub_participations where season_id=p_season_id for share;
 perform 1 from public.clubs c where exists(select 1 from public.pcs_interclub_participations p
  where p.season_id=p_season_id and p.club_id=c.id and p.status='accepted') order by c.id for share;
 select jsonb_build_object(
  'approved',coalesce((select jsonb_agg(jsonb_build_object('id',id,'revision',approved_revision) order by id)
   from public.pcs_interclub_competition_batches where season_id=p_season_id and approved_document is not null),'[]'::jsonb),
  'season',(select details from public.pcs_interclub_seasons where id=p_season_id),
  'clubs',coalesce((select jsonb_agg(jsonb_build_object('id',c.id,'name',c.name) order by c.id)
   from public.clubs c where exists(select 1 from public.pcs_interclub_participations p
    where p.season_id=p_season_id and p.club_id=c.id and p.status='accepted')),'[]'::jsonb),
  'meets',coalesce((select jsonb_agg(jsonb_build_object('id',id,'revision',revision) order by id)
   from public.pcs_interclub_meets where season_id=p_season_id),'[]'::jsonb)
 ) into current_sources;
 if p_sources is null or p_sources is distinct from current_sources then
  raise exception 'League preview changed; reload and review before publishing' using errcode='40001'; end if;
 if p_document->>'scoring_version' is distinct from '1' then
  raise exception 'Reviewed competition publication required' using errcode='22023'; end if;
 return public.pcs_write_interclub_publication(p_actor_id,p_actor_email,p_club_id,p_season_id,p_revision,'publish',p_document);
end $$;
revoke all on function public.pcs_publish_reviewed_interclub_publication(uuid,text,text,uuid,integer,jsonb,jsonb) from public,anon,authenticated;
grant execute on function public.pcs_publish_reviewed_interclub_publication(uuid,text,text,uuid,integer,jsonb,jsonb) to service_role;

commit;
