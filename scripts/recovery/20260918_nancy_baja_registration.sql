-- PREPARED ONLY: requires Joe's explicit production approval before execution.
-- Correct an unused 4.00 starting rating to the organizer-confirmed 3.71,
-- preserve the existing 3.5 team, hydrate its reciprocal partner fields,
-- and remove the superseded unpaired 4.0 entry. No emails are sent.
-- This is a one-off data repair, not a deploy-time schema migration.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '20s';
DO $repair$
DECLARE
  target_tournament_id constant uuid := '563b7922-ae92-41d4-8286-75fe9846e944';
  marker constant text := 'nancy_baja_registration_correction_20260918';
  nancy public.tournament_registrations%ROWTYPE;
  lisa public.tournament_registrations%ROWTYPE;
  profile public.players%ROWTYPE;
  keep_entry public.tournament_registration_selections%ROWTYPE;
  old_entry public.tournament_registration_selections%ROWTYPE;
  team public.tournament_registration_team_links%ROWTYPE;
  before_state jsonb;
  after_state jsonb;
  affected integer;
BEGIN
  -- Match the existing registration/partner writers' advisory locks.
  PERFORM pg_advisory_xact_lock(hashtextextended('manual-tournament-partner:' || target_tournament_id::text, 0));
  PERFORM private.lock_tournament_registration_selection_scope(
    ARRAY['sel_00df5e4994', 'sel_aa628c5696e7', 'sel_39bad8e0a33646dd9b10d241e1b6c1cc']
  );
  IF EXISTS (SELECT 1 FROM public.admin_activity_log WHERE source_page=marker
      AND entity_id='reg_268cf62b84c8') THEN
    RAISE EXCEPTION 'This correction was already applied; inspect its audit instead of replaying it.';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM public.tournaments t
      WHERE t.id=target_tournament_id AND t.club_id='tres_palapas') THEN
    RAISE EXCEPTION 'Tournament scope changed.';
  END IF;
  PERFORM 1 FROM public.tournament_registrations r
    WHERE r.id IN ('reg_268cf62b84c8', 'reg_6f8d9b259563') ORDER BY r.id FOR UPDATE;
  SELECT * INTO STRICT nancy FROM public.tournament_registrations WHERE id='reg_268cf62b84c8';
  SELECT * INTO STRICT lisa FROM public.tournament_registrations WHERE id='reg_6f8d9b259563';
  SELECT * INTO STRICT profile FROM public.players WHERE id=596 AND club_id='tres_palapas' FOR UPDATE;
  IF nancy.tournament_id IS DISTINCT FROM target_tournament_id::text
      OR lisa.tournament_id IS DISTINCT FROM target_tournament_id::text
      OR nancy.player_id IS DISTINCT FROM 596 OR lisa.player_id IS DISTINCT FROM 605
      OR nancy.status IS DISTINCT FROM 'confirmed' OR lisa.status IS DISTINCT FROM 'confirmed'
      OR nancy.updated_at IS DISTINCT FROM '2026-09-13T16:08:34.879833Z'::timestamptz
      OR lisa.updated_at IS DISTINCT FROM '2026-09-14T21:20:02.948178Z'::timestamptz
      OR nancy.doubles_skill IS DISTINCT FROM 4.00
      OR profile.rating IS DISTINCT FROM 1600 OR profile.starting_rating IS DISTINCT FROM 1600
      OR profile.matches_played IS DISTINCT FROM 0 OR profile.wins IS DISTINCT FROM 0
      OR profile.losses IS DISTINCT FROM 0 OR profile.last_game_at IS NOT NULL
  THEN RAISE EXCEPTION 'Registration or unused starting-rating evidence changed.'; END IF;
  IF EXISTS (SELECT 1 FROM public.tournament_registrations r
      WHERE r.player_id=596 AND r.id<>nancy.id)
      OR EXISTS (SELECT 1 FROM public.tournament_commerce_orders o
        WHERE o.registration_id IN (nancy.id,lisa.id))
      OR EXISTS (SELECT 1 FROM public.tournament_teams t
        WHERE t.tournament_id=target_tournament_id AND
          (t.player1_id IN (596,605) OR t.player2_id IN (596,605)))
  THEN RAISE EXCEPTION 'New registration, commerce order, or draw requires review.'; END IF;
  PERFORM 1 FROM public.tournament_registration_selections s
    WHERE s.registration_id IN (nancy.id,lisa.id) ORDER BY s.id FOR UPDATE;
  SELECT * INTO STRICT old_entry FROM public.tournament_registration_selections WHERE id='sel_00df5e4994';
  SELECT * INTO STRICT keep_entry FROM public.tournament_registration_selections WHERE id='sel_39bad8e0a33646dd9b10d241e1b6c1cc';
  SELECT * INTO STRICT team FROM public.tournament_registration_team_links
    WHERE id='tlink_56c967fd029042c3886eb7b5eaddfaf9' FOR UPDATE;
  IF old_entry.registration_id IS DISTINCT FROM nancy.id
      OR old_entry.tournament_id IS DISTINCT FROM target_tournament_id::text
      OR old_entry.event_option_id IS DISTINCT FROM 'div_6c2575038e'
      OR old_entry.partner_mode IS DISTINCT FROM 'NEEDS_PARTNER'
      OR old_entry.updated_at IS DISTINCT FROM '2026-09-13T16:08:34.875492Z'::timestamptz
      OR keep_entry.registration_id IS DISTINCT FROM nancy.id
      OR keep_entry.tournament_id IS DISTINCT FROM target_tournament_id::text
      OR keep_entry.event_option_id IS DISTINCT FROM 'div_ef932f08e6'
      OR keep_entry.partner_mode IS DISTINCT FROM 'HAS_PARTNER'
      OR keep_entry.updated_at IS DISTINCT FROM '2026-09-14T21:20:03.139824Z'::timestamptz
      OR team.tournament_id IS DISTINCT FROM target_tournament_id
      OR team.event_option_id IS DISTINCT FROM keep_entry.event_option_id
      OR team.registration1_id IS DISTINCT FROM lisa.id OR team.registration2_id IS DISTINCT FROM nancy.id
      OR team.selection1_id IS DISTINCT FROM 'sel_aa628c5696e7' OR team.selection2_id IS DISTINCT FROM keep_entry.id
      OR team.status IS DISTINCT FROM 'ADMIN_CONFIRMED'
      OR (SELECT count(*) FROM public.tournament_registration_selections s WHERE s.registration_id=nancy.id)<>2
  THEN RAISE EXCEPTION 'Exact entry or confirmed team evidence changed.'; END IF;
  IF EXISTS (SELECT 1 FROM public.tournament_registration_partner_requests p
        WHERE p.requester_selection_id=old_entry.id OR p.target_selection_id=old_entry.id)
      OR EXISTS (SELECT 1 FROM public.tournament_registration_team_links t
        WHERE t.selection1_id=old_entry.id OR t.selection2_id=old_entry.id)
      OR EXISTS (SELECT 1 FROM public.tournament_registration_team_members m WHERE m.selection_id=old_entry.id)
      OR (SELECT count(*) FROM public.tournament_registration_team_members m
        WHERE m.team_link_id=team.id AND m.status='ACTIVE'
          AND m.selection_id IN (team.selection1_id,team.selection2_id))<>2
  THEN RAISE EXCEPTION 'Partner relationships changed.'; END IF;

  before_state := jsonb_build_object('profile',to_jsonb(profile),'registration',to_jsonb(nancy),
    'removed_entry',to_jsonb(old_entry),'retained_entry',to_jsonb(keep_entry),'team',to_jsonb(team));
  UPDATE public.players SET rating=1484,starting_rating=1484 WHERE id=596 AND club_id='tres_palapas';
  UPDATE public.tournament_registrations SET doubles_skill=3.71,updated_at=clock_timestamp() WHERE id=nancy.id;
  UPDATE public.tournament_registration_selections
    SET partner_name=lisa.display_name,partner_email=lisa.email,partner_skill=lisa.doubles_skill,
        partner_age=lisa.age,partner_gender=lisa.gender,show_on_partner_board=false,updated_at=clock_timestamp()
    WHERE id=keep_entry.id;
  DELETE FROM public.tournament_registration_selections WHERE id=old_entry.id;
  GET DIAGNOSTICS affected = ROW_COUNT;
  IF affected<>1 THEN RAISE EXCEPTION 'Expected exactly one obsolete entry.'; END IF;

  SELECT jsonb_build_object('profile',to_jsonb(p),'registration',to_jsonb(r),'retained_entry',to_jsonb(s),'team',to_jsonb(t))
    INTO after_state FROM public.players p
    JOIN public.tournament_registrations r ON r.player_id=p.id AND r.id=nancy.id
    JOIN public.tournament_registration_selections s ON s.registration_id=r.id AND s.id=keep_entry.id
    JOIN public.tournament_registration_team_links t ON t.id=team.id WHERE p.id=596;
  IF after_state IS NULL OR after_state->'team' IS DISTINCT FROM before_state->'team'
      OR (SELECT count(*) FROM public.tournament_registration_selections s WHERE s.registration_id=nancy.id)<>1
  THEN RAISE EXCEPTION 'Correction did not preserve the existing team and single intended entry.'; END IF;
  INSERT INTO public.admin_activity_log
    (club_id,actor_email,actor_role,action_type,entity_type,entity_id,before_json,after_json,note,source_page,flagged_for_review)
    VALUES ('tres_palapas','codex','system','tournament_registration_correction','tournament_registration',nancy.id,
      before_state,after_state,
      'Organizer confirmed doubles rating 3.71. Corrected unused starting rating, retained Lisa partnership in women''s doubles 3.5, and removed extra unpaired 4.0 entry.',
      marker,true);
END
$repair$;
COMMIT;
