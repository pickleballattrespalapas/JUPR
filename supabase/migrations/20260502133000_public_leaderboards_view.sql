do $$
begin
    if to_regclass('public.league_ratings') is not null
       and to_regclass('public.players') is not null then
        execute $view$
            create or replace view public.public_leaderboards as
            select
                lr.club_id,
                lr.league_name,
                lr.player_id,
                p.name as player_name,
                lr.rating,
                lr.rating as rating_jupr,
                coalesce(lr.wins, 0) as wins,
                coalesce(lr.losses, 0) as losses,
                coalesce(lr.matches_played, coalesce(lr.wins, 0) + coalesce(lr.losses, 0)) as matches_played,
                coalesce(lr.is_active, (p.inactive_at is null)) as is_active,
                row_number() over (
                    partition by lr.club_id, lr.league_name
                    order by lr.rating desc nulls last, p.name asc
                ) as rank_position,
                null::timestamptz as updated_at
            from public.league_ratings lr
            join public.players p
                on p.club_id = lr.club_id
               and p.id = lr.player_id
        $view$;
    else
        raise notice 'Skipping public.public_leaderboards view creation: required tables public.league_ratings and/or public.players do not exist.';
    end if;
end
$$;
