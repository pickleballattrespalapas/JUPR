begin;
-- JSON drafts now retain partial wizard steps and eligibility rules.
-- Saving a draft after invitations open must never imply that the live setup changed.
create or replace function public.pcs_save_interclub_draft(p_actor_id uuid,p_actor_email text,p_club_id text,p_id uuid,p_revision integer,p_draft jsonb)
returns jsonb language plpgsql security invoker set search_path=public as $$
declare saved public.pcs_interclub_drafts; selected_club text;
begin
 perform public.pcs_require_interclub_admin(p_actor_id,p_actor_email,p_club_id);
 perform pg_advisory_xact_lock(hashtextextended('pcs-season:'||p_id::text,0));
 if exists(select 1 from public.pcs_interclub_seasons where id=p_id) then
  raise exception 'Invitations are open. Manage club responses and meet rosters from the season workspace.' using errcode='40001';
 end if;
 if jsonb_typeof(p_draft)<>'object' or jsonb_typeof(p_draft->'club_ids')<>'array' then raise exception 'Invalid draft' using errcode='22023'; end if;
 for selected_club in select value from jsonb_array_elements_text(p_draft->'club_ids') order by value loop
  perform 1 from public.clubs where id=selected_club for key share;
  if not found then raise exception 'Unknown club' using errcode='22023'; end if;
 end loop;
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
commit;
