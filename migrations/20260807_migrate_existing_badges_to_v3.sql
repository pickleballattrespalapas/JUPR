-- Migrate active legacy badge_rules_v2 rules into Badge Engine V3 conditions.
-- Steps performed:
--   1) Publish matching badges_v2 rows as system badges.
--   2) Translate enabled badge_rules_v2 rows into badge_rule_conditions.
--   3) Queue a recompute event for every club.
--   4) Snapshot legacy-vs-v3 award counts into an audit table.
--   5) Deactivate legacy badge_rules_v2 rules.

create table if not exists public.badge_v3_migration_audit (
    migration_key text not null,
    migrated_at timestamptz not null default now(),
    badge_id text not null,
    legacy_award_count bigint not null,
    v3_award_count bigint not null,
    counts_match boolean not null,
    primary key (migration_key, badge_id)
);

with active_rule_badges as (
    select distinct r.badge_id
    from public.badge_rules_v2 r
    where coalesce(r.enabled, true) = true
      and nullif(trim(coalesce(r.badge_id, '')), '') is not null
),
matched_badges as (
    select b.id, b.badge_id
    from public.badges_v2 b
    inner join active_rule_badges arb on arb.badge_id = b.badge_id
),
published as (
    update public.badges_v2 b
    set status = 'published',
        published_at = coalesce(b.published_at, now()),
        archived_at = null,
        is_system_badge = true,
        club_id = null
    where b.id in (select id from matched_badges)
    returning b.id, b.badge_id
),
clear_conditions as (
    delete from public.badge_rule_conditions c
    where c.badge_id in (select id from published)
    returning c.badge_id
),
legacy_rules as (
    select
        p.id as v3_badge_pk,
        p.badge_id,
        to_jsonb(r) as rule_json
    from published p
    inner join public.badge_rules_v2 r
        on r.badge_id = p.badge_id
    where coalesce(r.enabled, true) = true
),
normalized as (
    select
        lr.v3_badge_pk,
        coalesce(
            nullif(trim(lr.rule_json ->> 'fact_key'), ''),
            nullif(trim(lr.rule_json ->> 'metric_key'), ''),
            nullif(trim(lr.rule_json ->> 'metric'), ''),
            nullif(trim(lr.rule_json ->> 'rule_key'), ''),
            nullif(trim(lr.rule_json ->> 'stat_key'), '')
        ) as fact_key,
        case lower(coalesce(
            lr.rule_json ->> 'operator',
            lr.rule_json ->> 'op',
            lr.rule_json ->> 'comparator',
            ''
        ))
            when '>=' then '>='
            when 'gte' then '>='
            when '>' then '>'
            when 'gt' then '>'
            when '=' then '='
            when '==' then '='
            when 'eq' then '='
            when '<=' then '<='
            when 'lte' then '<='
            when '<' then '<'
            when 'lt' then '<'
            when 'is' then 'is'
            else null
        end as operator,
        case
            when coalesce(
                nullif(trim(lr.rule_json ->> 'value_numeric'), ''),
                nullif(trim(lr.rule_json ->> 'value_num'), ''),
                nullif(trim(lr.rule_json ->> 'threshold'), ''),
                nullif(trim(lr.rule_json ->> 'target'), ''),
                nullif(trim(lr.rule_json ->> 'value'), '')
            ) ~ '^-?[0-9]+(\.[0-9]+)?$'
            then coalesce(
                nullif(trim(lr.rule_json ->> 'value_numeric'), ''),
                nullif(trim(lr.rule_json ->> 'value_num'), ''),
                nullif(trim(lr.rule_json ->> 'threshold'), ''),
                nullif(trim(lr.rule_json ->> 'target'), ''),
                nullif(trim(lr.rule_json ->> 'value'), '')
            )::numeric
            else null
        end as value_numeric,
        case lower(coalesce(
            nullif(trim(lr.rule_json ->> 'value_boolean'), ''),
            nullif(trim(lr.rule_json ->> 'value_bool'), ''),
            nullif(trim(lr.rule_json ->> 'expected_bool'), ''),
            nullif(trim(lr.rule_json ->> 'expected'), '')
        ))
            when 'true' then true
            when 'false' then false
            else null
        end as value_boolean
    from legacy_rules lr
),
inserted_conditions as (
    insert into public.badge_rule_conditions (
        badge_id,
        fact_key,
        operator,
        value_numeric,
        value_boolean
    )
    select
        n.v3_badge_pk,
        n.fact_key,
        n.operator,
        n.value_numeric,
        n.value_boolean
    from normalized n
    where n.fact_key is not null
      and n.operator is not null
      and (
        (n.operator = 'is' and n.value_boolean is not null)
        or
        (n.operator <> 'is' and n.value_numeric is not null)
      )
    returning badge_id
),
award_counts as (
    select
        b.badge_id,
        coalesce(legacy.cnt, 0)::bigint as legacy_award_count,
        coalesce(v3.cnt, 0)::bigint as v3_award_count,
        (coalesce(legacy.cnt, 0) = coalesce(v3.cnt, 0)) as counts_match
    from published b
    left join (
        select badge_id, count(*) as cnt
        from public.user_badges_v2
        group by badge_id
    ) legacy on legacy.badge_id = b.badge_id
    left join (
        select badge_id, count(*) as cnt
        from public.player_badges
        group by badge_id
    ) v3 on v3.badge_id = b.badge_id
),
audit_upsert as (
    insert into public.badge_v3_migration_audit (
        migration_key,
        badge_id,
        legacy_award_count,
        v3_award_count,
        counts_match
    )
    select
        '20260807_migrate_existing_badges_to_v3',
        ac.badge_id,
        ac.legacy_award_count,
        ac.v3_award_count,
        ac.counts_match
    from award_counts ac
    on conflict (migration_key, badge_id)
    do update
    set migrated_at = now(),
        legacy_award_count = excluded.legacy_award_count,
        v3_award_count = excluded.v3_award_count,
        counts_match = excluded.counts_match
    returning badge_id
),
queued_recompute as (
    insert into public.badge_eval_queue (
        club_id,
        event_type,
        player_ids,
        payload_json,
        status
    )
    select
        c.id,
        'badge_recompute_v3_migration',
        '{}'::bigint[],
        jsonb_build_object(
            'migration_key', '20260807_migrate_existing_badges_to_v3',
            'badge_ids', (
                select coalesce(jsonb_agg(p.badge_id order by p.badge_id), '[]'::jsonb)
                from published p
            )
        ),
        'pending'
    from public.clubs c
    where not exists (
        select 1
        from public.badge_eval_queue q
        where q.club_id = c.id
          and q.event_type = 'badge_recompute_v3_migration'
          and q.status in ('pending', 'processing')
          and q.payload_json ->> 'migration_key' = '20260807_migrate_existing_badges_to_v3'
    )
    returning club_id
)
update public.badge_rules_v2 r
set enabled = false
where coalesce(r.enabled, true) = true
  and r.badge_id in (select badge_id from published)
;
