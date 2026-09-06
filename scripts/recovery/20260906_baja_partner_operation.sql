-- Exact, audited recovery of the settled September 4 partner assignment.
-- The confirmed link and both active members already exist. Both selections
-- already have HAS_PARTNER; only the second partner-board flag is unfinished.
-- No partner identity, team, sponsor, draw, or score is created or replaced.
-- Run in a transaction; all evidence checks and the audit must succeed.
BEGIN;
DO $recovery$
DECLARE
  op public.tournament_admin_operations%ROWTYPE;
  team public.tournament_registration_team_links%ROWTYPE;
  entry public.tournament_registration_selections%ROWTYPE;
  result jsonb;
  evidence jsonb;
  affected integer;
BEGIN
  SELECT * INTO STRICT op FROM public.tournament_admin_operations
  WHERE operation_key = 'd6969f134f8dad189ad34574e744507e64ba6415cced75ab6cd4b491decbcadb'
  FOR UPDATE;
  IF op.status = 'completed' AND op.result_json->>'recovery_source' = 'sponsor_lock_recovery_20260906' THEN
    RETURN;
  END IF;
  IF op.club_id <> 'tres_palapas'
    OR op.lock_scope <> '563b7922-ae92-41d4-8286-75fe9846e944'
    OR op.surface <> 'registration'
    OR op.action <> 'tournament_registration_partner_update'
    OR op.entity_id <> 'sel_5ece34f254'
    OR op.status <> 'recovery_required'
    OR op.result_json <> '{}'::jsonb
    OR op.updated_at <> '2026-09-04 05:07:03.828371+00'::timestamptz
    OR op.request_json->'payload' <> '{"unpaired_mode":"NONE","partner_selection_id":"sel_59e57dea88"}'::jsonb
    OR op.error_text NOT LIKE '%JUPR_MANUAL_PARTNER_EMAIL_REQUIRED%'
  THEN RAISE EXCEPTION 'Recovery operation evidence changed'; END IF;
  IF NOT EXISTS (SELECT 1 FROM public.tournaments
    WHERE id::text=op.lock_scope AND club_id::text=op.club_id)
  THEN RAISE EXCEPTION 'Tournament scope does not match'; END IF;

  SELECT * INTO STRICT team FROM public.tournament_registration_team_links
  WHERE id='tlink_f53c4160b856' FOR UPDATE;
  IF team.tournament_id::text <> op.lock_scope
    OR team.event_option_id <> 'div_ef932f08e6'
    OR team.selection1_id <> op.entity_id
    OR team.selection2_id <> 'sel_59e57dea88'
    OR team.status <> 'ADMIN_CONFIRMED'
    OR team.accepted_request_id <> 'preq_ca000812fa02'
    OR team.created_at < op.created_at OR team.created_at > op.updated_at
  THEN RAISE EXCEPTION 'Confirmed team does not prove the requested assignment'; END IF;
  PERFORM 1 FROM public.tournament_registration_team_members
    WHERE team_link_id=team.id FOR UPDATE;
  IF (SELECT count(*) FROM public.tournament_registration_team_members
      WHERE team_link_id=team.id AND status='ACTIVE'
        AND tournament_id::text=op.lock_scope AND event_option_id=team.event_option_id
        AND selection_id IN (team.selection1_id,team.selection2_id)) <> 2
    OR (SELECT count(*) FROM public.tournament_registration_team_members
      WHERE status='ACTIVE' AND selection_id IN (team.selection1_id,team.selection2_id)) <> 2
    OR (SELECT count(DISTINCT selection_id) FROM public.tournament_registration_team_members
      WHERE team_link_id=team.id AND status='ACTIVE') <> 2
  THEN RAISE EXCEPTION 'Active team membership is not exact'; END IF;
  PERFORM 1 FROM public.tournament_registration_partner_requests
    WHERE id=team.accepted_request_id FOR UPDATE;
  IF NOT EXISTS (SELECT 1 FROM public.tournament_registration_partner_requests
      WHERE id=team.accepted_request_id AND tournament_id::text=op.lock_scope
      AND event_option_id=team.event_option_id AND status='ADMIN_CONFIRMED'
      AND requester_selection_id=team.selection1_id AND target_selection_id=team.selection2_id)
    OR EXISTS (SELECT 1 FROM public.tournament_registration_partner_requests
      WHERE tournament_id::text=op.lock_scope AND event_option_id=team.event_option_id
      AND status='PENDING' AND (requester_selection_id IN (team.selection1_id,team.selection2_id)
      OR target_selection_id IN (team.selection1_id,team.selection2_id)))
  THEN RAISE EXCEPTION 'Partner request evidence changed'; END IF;
  PERFORM 1 FROM public.tournament_registration_selections
    WHERE id IN (team.selection1_id,team.selection2_id) FOR UPDATE;
  IF (SELECT count(*) FROM public.tournament_registration_selections
      WHERE id IN (team.selection1_id,team.selection2_id)
      AND tournament_id::text=op.lock_scope AND event_option_id=team.event_option_id
      AND partner_mode='HAS_PARTNER') <> 2
    OR NOT EXISTS (SELECT 1 FROM public.tournament_registration_selections
      WHERE id=team.selection1_id AND show_on_partner_board=false
      AND updated_at='2026-09-04 05:07:03.753419+00'::timestamptz)
    OR NOT EXISTS (SELECT 1 FROM public.tournament_registration_selections
      WHERE id=team.selection2_id AND show_on_partner_board=true
      AND updated_at='2026-08-03 17:39:35.521684+00'::timestamptz)
  THEN RAISE EXCEPTION 'Selection evidence changed'; END IF;

  evidence := jsonb_build_object('team_link_id',team.id,
    'accepted_request_id',team.accepted_request_id,'active_member_count',2,
    'partner_modes_already_complete',true,'pending_requests',0,
    'finished_selection_id',team.selection2_id,'show_on_partner_board_before',true,
    'show_on_partner_board_after',false);
  -- Changing only the board flag does not resubmit the already-set partner
  -- mode or invoke manual partner creation from stale legacy free-text fields.
  UPDATE public.tournament_registration_selections
    SET show_on_partner_board=false
    WHERE id=team.selection2_id AND show_on_partner_board=true;
  GET DIAGNOSTICS affected = ROW_COUNT;
  IF affected <> 1 THEN RAISE EXCEPTION 'Expected one unfinished board flag'; END IF;
  SELECT * INTO STRICT entry FROM public.tournament_registration_selections WHERE id=op.entity_id;
  result := jsonb_build_object('ok',true,'mode','tournament_registration_partner_update',
    'selection',to_jsonb(entry),'partner_result',jsonb_build_object('outcome','paired',
    'team_link',to_jsonb(team),'partner_selection_id',team.selection2_id),
    'warnings','[]'::jsonb,'recovery_source','sponsor_lock_recovery_20260906');
  INSERT INTO public.admin_activity_log
    (club_id,actor_email,actor_role,action_type,entity_type,entity_id,before_json,after_json,note,source_page,flagged_for_review)
    VALUES (op.club_id,'codex','system',op.action||'_reconciliation',op.entity_type,op.entity_id,
      jsonb_build_object('status',op.status,'error',op.error_text),
      jsonb_build_object('audit_marker',jsonb_build_object('operation_key',op.operation_key,
        'request_fingerprint',op.request_fingerprint,'operation_status','reconciliation'),
        'value',jsonb_build_object('disposition','completed','evidence',evidence)),
      'User-requested production incident recovery: verified the requested pair and finished its remaining board flag; no assignment replay.',
      'sponsor_lock_recovery_20260906',true);
  UPDATE public.tournament_admin_operations
    SET status='completed',result_json=result,error_text=NULL,attempt_count=attempt_count+1,
      updated_by='codex',updated_at=now(),completion_audited_at=now()
    WHERE operation_key=op.operation_key AND status='recovery_required';
  GET DIAGNOSTICS affected = ROW_COUNT;
  IF affected <> 1 THEN RAISE EXCEPTION 'Recovery status did not settle'; END IF;
END
$recovery$;
COMMIT;
