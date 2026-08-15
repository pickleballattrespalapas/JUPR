-- Keep tournament-team eligibility enforcement compatible with the legacy
-- tournament_event_options.tournament_id text column. tournament_teams stores
-- the same identifier as uuid, so compare through the canonical text form.

create or replace function public.enforce_combined_rating_draw_eligibility()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
declare
  v_event public.tournament_event_options%rowtype;
  v_review public.tournament_rating_eligibility_reviews%rowtype;
begin
  if new.event_option_id is null then
    return new;
  end if;
  select event.* into v_event
    from public.tournament_event_options event
   where event.id::text = new.event_option_id::text
     and event.tournament_id = new.tournament_id::text
   for share;
  if not found or v_event.eligibility_mode <> 'COMBINED_RATING_CAP' then
    return new;
  end if;
  if new.source_selection_id is null then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_COMBINED_RATING_SELECTION_REQUIRED';
  end if;
  select review.* into v_review
    from public.tournament_rating_eligibility_reviews review
   where review.tournament_id = new.tournament_id
     and review.event_option_id::text = new.event_option_id::text
     and review.selection_id = new.source_selection_id
     and review.review_phase = 'REGISTRATION_CLOSE'
   for share;
  if not found or v_review.finalized_at is null
     or coalesce(v_review.override_state, v_review.state) <> 'ELIGIBLE'
     or (
       v_review.override_state is not null
       and nullif(btrim(v_review.override_reason), '') is null
     ) then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_COMBINED_RATING_DRAW_BLOCKED';
  end if;
  new.eligibility_review_id := v_review.id;
  new.eligibility_reviewed_at := v_review.rating_as_of;
  new.eligibility_state_snapshot := coalesce(
    v_review.override_state, v_review.state
  );
  new.combined_rating_snapshot := v_review.combined_rating;
  new.combined_rating_cap_snapshot := v_review.combined_rating_cap;
  return new;
end;
$$;

revoke all on function public.enforce_combined_rating_draw_eligibility()
  from public, anon, authenticated;
grant execute on function public.enforce_combined_rating_draw_eligibility()
  to service_role;
