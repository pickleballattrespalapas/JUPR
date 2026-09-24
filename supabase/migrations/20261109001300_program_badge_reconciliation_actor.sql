-- Automatic corrections have no human UUID actor. Identify engine-owned
-- revocations by their rule-specific reason; preserve every manual revocation.
create or replace function public.apply_program_badges_v1(p_club_id text,p_revision bigint,p_awards jsonb,p_pending_ties jsonb,p_review jsonb)
returns jsonb language plpgsql security invoker set search_path='' as $$
declare current_revision bigint; v_award jsonb; inserted integer := 0; restored integer := 0; revoked integer := 0; changed integer;
        current_award public.player_badges%rowtype;
begin
  select revision into current_revision from public.badge_program_state where club_id=p_club_id for update;
  if current_revision is null or current_revision is distinct from p_revision then raise exception using errcode='40001',message='Program results changed. Read a fresh snapshot.'; end if;
  if jsonb_typeof(p_awards) is distinct from 'array' or jsonb_typeof(p_pending_ties) is distinct from 'array' or jsonb_typeof(p_review) is distinct from 'array' then
    raise exception using errcode='22023',message='Invalid program badge result.';
  end if;
  if exists(select 1 from jsonb_array_elements(p_awards) a where a->>'badge_id' not in ('leagues_completed_5','leagues_completed_10','leagues_completed_25','tournaments_completed_5','tournaments_completed_10','tournaments_completed_25','round_robins_completed_5','round_robins_completed_10','round_robins_completed_25','matches_together_10','matches_together_25','matches_together_50','wins_together_10','wins_together_25','wins_together_50','five_winning_partners','triple_crown','round_robin_wins_1','round_robin_wins_5','round_robin_wins_10','round_robin_wins_25','round_robin_wins_50')
      or a->>'context_type' <> 'overall' or nullif(a->>'context_id','') is null
      or (a->>'earned_at')::timestamptz > now() or (a->>'earned_at') is null
      or not exists(select 1 from public.players p where p.club_id=p_club_id and p.id=(a->>'player_id')::bigint)) then
    raise exception using errcode='22023',message='Invalid badge, player, date or context.';
  end if;
  if exists(select 1 from jsonb_array_elements(p_awards) a group by a->>'badge_id',a->>'player_id',a->>'context_id' having count(*)>1) then
    raise exception using errcode='22023',message='Duplicate program award identity.';
  end if;
  for v_award in select x from jsonb_array_elements(p_awards) x join public.badges b on b.badge_id=x->>'badge_id'
      where b.is_active and b.state='live' loop
    select * into current_award from public.player_badges where club_id=p_club_id and player_id=(v_award->>'player_id')::bigint
      and badge_id=v_award->>'badge_id' and context_type='overall' and context_id=v_award->>'context_id' for update;
    if not found then
      insert into public.player_badges(id,club_id,player_id,badge_id,context_type,context_id,earned_at,value_num,value_json,awarded_by,rule_version)
      values(gen_random_uuid(),p_club_id,(v_award->>'player_id')::bigint,v_award->>'badge_id','overall',v_award->>'context_id',
             (v_award->>'earned_at')::timestamptz,(v_award->>'value_num')::numeric,v_award->'value_json','engine','program-badges-v1');
      inserted := inserted+1;
    elsif current_award.rule_version='program-badges-v1' and (current_award.revoked_at is null or (current_award.revoked_by is null and current_award.revoked_reason='program-badges-v1: Corrected program results no longer meet the requirement.')) then
      if current_award.revoked_at is not null then restored := restored+1; end if;
      update public.player_badges set earned_at=(v_award->>'earned_at')::timestamptz,value_num=(v_award->>'value_num')::numeric,value_json=v_award->'value_json',
        revoked_at=null,revoked_by=null,revoked_reason=null,revoke_reason=null
      where id=current_award.id and (earned_at is distinct from (v_award->>'earned_at')::timestamptz or value_json is distinct from v_award->'value_json' or revoked_at is not null);
    end if;
  end loop;
  -- This reconciliation owns only the 22 new badge types. Manual revocations,
  -- the earlier historical review set and every legacy trophy remain untouched.
  update public.player_badges pb set revoked_at=now(),revoked_by=null,
      revoked_reason='program-badges-v1: Corrected program results no longer meet the requirement.',revoke_reason='Corrected program results no longer meet the requirement.'
  where pb.club_id=p_club_id and pb.badge_id in ('leagues_completed_5','leagues_completed_10','leagues_completed_25','tournaments_completed_5','tournaments_completed_10','tournaments_completed_25','round_robins_completed_5','round_robins_completed_10','round_robins_completed_25','matches_together_10','matches_together_25','matches_together_50','wins_together_10','wins_together_25','wins_together_50','five_winning_partners','triple_crown','round_robin_wins_1','round_robin_wins_5','round_robin_wins_10','round_robin_wins_25','round_robin_wins_50') and pb.rule_version='program-badges-v1' and pb.revoked_at is null
    and exists(select 1 from public.badges b where b.badge_id=pb.badge_id and b.is_active and b.state='live')
    and not exists(select 1 from jsonb_array_elements(p_awards) a where (a->>'player_id')::bigint=pb.player_id
                   and a->>'badge_id'=pb.badge_id and a->>'context_id'=pb.context_id and a->>'context_type'=pb.context_type);
  get diagnostics revoked=row_count;
  update public.badge_program_state set applied_revision=p_revision,pending_ties=p_pending_ties,review=p_review,checked_at=now(),last_error=null where club_id=p_club_id;
  return jsonb_build_object('ok',true,'inserted',inserted,'restored',restored,'revoked',revoked,'revision',p_revision);
end $$;
