-- Registration tables retain text tournament IDs; rating-review and audit
-- tables use UUIDs. Resolve that boundary explicitly so combined-cap entries
-- can be saved, including the reciprocal selection created for a named partner.
-- Registration ordering uses submitted_at, the existing registration timestamp.

create or replace function public.refresh_initial_combined_rating_review_v1(
  p_selection_id text,
  p_actor text
)
returns void
language plpgsql
security invoker
set search_path = ''
as $$
declare
  v_selection public.tournament_registration_selections%rowtype;
  v_event public.tournament_event_options%rowtype;
  v_registration public.tournament_registrations%rowtype;
  v_partner_registration public.tournament_registrations%rowtype;
  v_player public.players%rowtype;
  v_partner public.players%rowtype;
  v_player_verification public.tournament_rating_verifications%rowtype;
  v_partner_verification public.tournament_rating_verifications%rowtype;
  v_saved public.tournament_rating_eligibility_reviews%rowtype;
  v_before jsonb;
  v_club_id text;
  v_player_rating numeric(4,2);
  v_partner_rating numeric(4,2);
  v_combined numeric(5,2);
  v_player_source text := 'MISSING';
  v_partner_source text := 'MISSING';
  v_state text;
begin
  select selection.* into v_selection
    from public.tournament_registration_selections selection
   where selection.id::text = p_selection_id
   for share;
  if not found then
    return;
  end if;
  select event.* into v_event
    from public.tournament_event_options event
   where event.id::text = v_selection.event_option_id::text
     and event.tournament_id = v_selection.tournament_id
   for share;
  if not found or v_event.eligibility_mode <> 'COMBINED_RATING_CAP'
     or v_event.combined_rating_cap is null then
    return;
  end if;
  select registration.* into v_registration
    from public.tournament_registrations registration
   where registration.id::text = v_selection.registration_id::text
     and registration.tournament_id = v_selection.tournament_id
   for share;
  if not found then
    raise exception using errcode = 'P0002',
      message = 'JUPR_TOURNAMENT_INITIAL_RATING_REGISTRATION_NOT_FOUND';
  end if;
  select tournament.club_id::text into v_club_id
    from public.tournaments tournament
   where tournament.id = v_selection.tournament_id::uuid;

  select partner_registration.* into v_partner_registration
    from public.tournament_registration_team_members selected_member
    join public.tournament_registration_team_members partner_member
      on partner_member.team_link_id = selected_member.team_link_id
     and partner_member.selection_id::text <> selected_member.selection_id::text
     and upper(coalesce(partner_member.status, '')) = 'ACTIVE'
    join public.tournament_registrations partner_registration
      on partner_registration.id::text = partner_member.registration_id::text
     and partner_registration.tournament_id = v_selection.tournament_id
   where selected_member.selection_id::text = v_selection.id::text
     and selected_member.event_option_id::text = v_selection.event_option_id::text
     and upper(coalesce(selected_member.status, '')) = 'ACTIVE'
   order by partner_member.player_order, partner_member.id
   limit 1
     for share of partner_registration;
  if not found and nullif(lower(btrim(v_selection.partner_email)), '') is not null then
    select registration.* into v_partner_registration
      from public.tournament_registrations registration
     where registration.tournament_id = v_selection.tournament_id
       and lower(btrim(registration.email)) = lower(btrim(v_selection.partner_email))
       and upper(coalesce(registration.status, '')) not in ('CANCELLED', 'WITHDRAWN')
     order by registration.submitted_at, registration.id
     limit 1
       for share of registration;
  end if;

  if v_registration.player_id is not null then
    select player.* into v_player
      from public.players player
     where player.id = v_registration.player_id
       for share of player;
    v_player_rating := public.normalize_tournament_combined_rating(
      coalesce(
        to_jsonb(v_player)->>'doubles_rating',
        to_jsonb(v_player)->>'doubles_skill',
        to_jsonb(v_player)->>'rating'
      )
    );
    if v_player_rating is not null then
      v_player_source := 'PCS_LINKED';
    end if;
  end if;
  if v_player_rating is null then
    select verification.* into v_player_verification
      from public.tournament_rating_verifications verification
     where verification.tournament_id = v_selection.tournament_id::uuid
       and verification.event_option_id::text = v_selection.event_option_id::text
       and verification.registration_id::text = v_registration.id::text
       and verification.status = 'ACTIVE'
     order by verification.verified_at desc, verification.id
     limit 1
       for share of verification;
    if found then
      v_player_rating := v_player_verification.rating;
      v_player_source := 'ORGANIZER_VERIFIED';
    end if;
  end if;

  if v_partner_registration.id is not null
     and v_partner_registration.player_id is not null then
    select player.* into v_partner
      from public.players player
     where player.id = v_partner_registration.player_id
       for share of player;
    v_partner_rating := public.normalize_tournament_combined_rating(
      coalesce(
        to_jsonb(v_partner)->>'doubles_rating',
        to_jsonb(v_partner)->>'doubles_skill',
        to_jsonb(v_partner)->>'rating'
      )
    );
    if v_partner_rating is not null then
      v_partner_source := 'PCS_LINKED';
    end if;
  end if;
  if v_partner_rating is null and v_partner_registration.id is not null then
    select verification.* into v_partner_verification
      from public.tournament_rating_verifications verification
     where verification.tournament_id = v_selection.tournament_id::uuid
       and verification.event_option_id::text = v_selection.event_option_id::text
       and verification.registration_id::text = v_partner_registration.id::text
       and verification.status = 'ACTIVE'
     order by verification.verified_at desc, verification.id
     limit 1
       for share of verification;
    if found then
      v_partner_rating := v_partner_verification.rating;
      v_partner_source := 'ORGANIZER_VERIFIED';
    end if;
  end if;

  if upper(coalesce(v_selection.partner_mode, '')) = 'NEEDS_PARTNER' then
    v_state := 'PROVISIONAL_NEEDS_PARTNER';
    v_combined := null;
  elsif v_player_rating is null or v_partner_rating is null then
    v_state := 'REVIEW_REQUIRED';
    v_combined := null;
  else
    v_combined := round(v_player_rating + v_partner_rating, 2);
    v_state := case when v_combined < v_event.combined_rating_cap
      then 'ELIGIBLE' else 'INELIGIBLE' end;
  end if;

  select to_jsonb(review) into v_before
    from public.tournament_rating_eligibility_reviews review
   where review.event_option_id::text = v_selection.event_option_id::text
     and review.selection_id::text = v_selection.id::text
     and review.review_phase = 'INITIAL';
  insert into public.tournament_rating_eligibility_reviews (
    tournament_id, event_option_id, selection_id, registration_id,
    partner_registration_id, player_id_snapshot,
    partner_player_id_snapshot, review_phase, state, player_rating,
    partner_rating, combined_rating, combined_rating_cap,
    player_rating_source, partner_rating_source,
    player_verification_id, partner_verification_id, rating_as_of,
    finalized_at, override_state, override_reason, reviewed_by
  ) values (
    v_selection.tournament_id::uuid, v_selection.event_option_id, v_selection.id,
    v_registration.id,
    case when v_partner_registration.id is null
      then null else v_partner_registration.id end,
    v_registration.player_id,
    v_partner_registration.player_id,
    'INITIAL', v_state, v_player_rating, v_partner_rating, v_combined,
    v_event.combined_rating_cap, v_player_source, v_partner_source,
    case when v_player_source = 'ORGANIZER_VERIFIED'
      then v_player_verification.id else null end,
    case when v_partner_source = 'ORGANIZER_VERIFIED'
      then v_partner_verification.id else null end,
    clock_timestamp(), null, null, null,
    coalesce(nullif(p_actor, ''), 'registration-trigger')
  )
  on conflict (event_option_id, selection_id, review_phase)
    where selection_id is not null
  do update set
    registration_id = excluded.registration_id,
    partner_registration_id = excluded.partner_registration_id,
    player_id_snapshot = excluded.player_id_snapshot,
    partner_player_id_snapshot = excluded.partner_player_id_snapshot,
    state = excluded.state,
    player_rating = excluded.player_rating,
    partner_rating = excluded.partner_rating,
    combined_rating = excluded.combined_rating,
    combined_rating_cap = excluded.combined_rating_cap,
    player_rating_source = excluded.player_rating_source,
    partner_rating_source = excluded.partner_rating_source,
    player_verification_id = excluded.player_verification_id,
    partner_verification_id = excluded.partner_verification_id,
    rating_as_of = excluded.rating_as_of,
    finalized_at = null,
    override_state = null,
    override_reason = null,
    reviewed_by = excluded.reviewed_by
  returning * into v_saved;

  insert into public.tournament_team_audit_events (
    club_id, tournament_id, event_option_id, entity_type, entity_id,
    action, actor, before_json, after_json
  ) values (
    v_club_id, v_selection.tournament_id::uuid, v_selection.event_option_id,
    'tournament_rating_eligibility_review', v_saved.id::text,
    'rating_eligibility_initial_refreshed',
    coalesce(nullif(p_actor, ''), 'registration-trigger'),
    v_before, to_jsonb(v_saved)
  );
  -- No exception is swallowed: review/audit failure rolls the selection
  -- statement back, while ON CONFLICT makes a retried statement converge.
end;
$$;

revoke all on function public.refresh_initial_combined_rating_review_v1(text, text)
  from public, anon, authenticated;
grant execute on function public.refresh_initial_combined_rating_review_v1(text, text)
  to service_role;
