grant select (
    id,
    club_id,
    player_id,
    badge_id,
    earned_at,
    context_type,
    context_id,
    match_id,
    value_num,
    value_json,
    awarded_by,
    rule_version,
    eval_run_id,
    revoked_at,
    revoked_by,
    revoke_reason
) on public.player_badges to anon, authenticated;
