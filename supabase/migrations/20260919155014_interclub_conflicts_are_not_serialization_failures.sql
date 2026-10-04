begin;

-- A stale application revision cannot succeed by replaying the same request.
-- SQLSTATE 40001 means retry the transaction; PostgREST's transaction runner
-- retries it instead of returning the application's conflict to FastAPI.
-- PT409 is PostgREST's explicit, non-retryable HTTP conflict SQLSTATE.
-- Preserve the installed definitions, signatures, grants and search paths.
do $$
declare routine record; definition text; changed integer := 0;
begin
 for routine in
  select p.oid from pg_proc p join pg_namespace n on n.oid=p.pronamespace
  where n.nspname='public' and p.prokind='f' and p.proname=any(array[
   'pcs_apply_interclub_rating_projection','pcs_assert_interclub_competition_eligibility',
   'pcs_create_interclub_club_invitation','pcs_guard_interclub_lineup_refresh_deadline',
   'pcs_interclub_club_invitation','pcs_interclub_participation',
   'pcs_interclub_pool_action','pcs_interclub_pool_public_action',
   'pcs_open_interclub_meet_registration','pcs_open_interclub_registration',
   'pcs_publish_reviewed_interclub_publication','pcs_review_interclub_meet_roster',
   'pcs_review_interclub_pool_member','pcs_review_interclub_roster','pcs_save_interclub_draft',
   'pcs_save_interclub_meet_roster','pcs_save_interclub_roster',
   'pcs_set_interclub_meet_deadline','pcs_write_interclub_competition','pcs_write_interclub_publication'])
 loop
  definition := pg_get_functiondef(routine.oid);
  if definition like '%''40001''%' then
   execute replace(definition,'''40001''','''PT409''');
   changed := changed+1;
  end if;
 end loop;
 if changed<>20 then raise exception 'Expected 20 interclub conflict routines; found %',changed; end if;
end $$;
notify pgrst,'reload schema';
commit;
