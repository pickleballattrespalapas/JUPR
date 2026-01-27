with ranked as (
    select
        id,
        row_number() over (
            partition by club_id, player_id, badge_id, context_id
            order by earned_at asc, id asc
        ) as rn
    from player_badges
)
delete from player_badges
where id in (select id from ranked where rn > 1);

alter table player_badges
    drop constraint if exists player_badges_unique_context,
    drop constraint if exists player_badges_club_id_player_id_badge_id_context_id_key,
    drop constraint if exists player_badges_unique_context_type,
    add constraint player_badges_unique_context unique (club_id, player_id, badge_id, context_id);
