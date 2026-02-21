-- Seed initial badge facts used by rule conditions.
insert into badge_fact_registry (fact_key, description, data_type, allowed_scope)
values
    ('best_win_streak', 'Longest win streak', 'numeric', 'overall'),
    ('total_matches', 'Total matches played', 'numeric', 'overall'),
    ('rating_delta', 'Delta between starting and current rating', 'numeric', 'overall'),
    ('is_league_champion', 'Player is league champion', 'boolean', 'league'),
    ('is_event_winner', 'Player won an event', 'boolean', 'event'),
    ('is_undefeated_event', 'Player undefeated in an event', 'boolean', 'event')
on conflict (fact_key) do update set
    description = excluded.description,
    data_type = excluded.data_type,
    allowed_scope = excluded.allowed_scope
;
