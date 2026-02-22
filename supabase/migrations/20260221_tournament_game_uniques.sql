alter table tournament_games
add constraint tournament_games_id_unique unique (id);

create unique index if not exists
tournament_games_unique_per_slot
on tournament_games (club_id, tournament_id, stage, rr_round_number, rr_slot_number);
