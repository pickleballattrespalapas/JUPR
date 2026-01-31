create unique index if not exists leagues_metadata_club_id_league_name_unique
    on leagues_metadata (club_id, lower(league_name));
