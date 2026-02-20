-- Ensure ON CONFLICT (club_id, normalized_name) can be inferred by Postgres.
-- The previous partial unique index (where normalized_name is not null) cannot
-- be inferred by a plain ON CONFLICT (club_id, normalized_name) clause.
-- Guard for environments where legacy core tables are provisioned outside this
-- migration chain and public.players may not yet exist.

do $$
begin
  if to_regclass('public.players') is not null then
    execute 'drop index if exists public.players_club_id_normalized_name_uq';

    execute $sql$
      create unique index if not exists players_club_id_normalized_name_uq
        on public.players (club_id, normalized_name)
    $sql$;
  end if;
end;
$$;
