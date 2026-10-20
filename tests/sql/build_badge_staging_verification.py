"""Emit rollback-only badge SQL; this script never opens a database connection.

Run from any directory:
  python tests/sql/build_badge_staging_verification.py > /tmp/badge-staging.sql
Review the output and execute the whole script only on sijpxjxvdtrehmqvirfi.
"""
from pathlib import Path
from uuid import UUID, uuid4


ROOT = Path(__file__).resolve().parents[2]


def build(fixture_uuid: UUID | None = None) -> str:
    fixture_uuid = fixture_uuid or uuid4()
    club = f'audit-f01-rollback-{fixture_uuid.hex}'
    # Explicit negative IDs avoid consuming player/admin identity sequences and
    # stay within int32 for compatibility with installed badge triggers.
    player_id = -(1_000_000_000 + fixture_uuid.int % 1_000_000_000)
    parts = [f"""-- Authorized only for staging sijpxjxvdtrehmqvirfi. Synthetic fixture; always rollback.
begin isolation level repeatable read;
set local statement_timeout='15s';
set local lock_timeout='2s';
set local idle_in_transaction_session_timeout='30s';
select set_config('jupr.audit_badge_fixture_club','{club}',true);
insert into public.clubs(id,name,slug,status,onboarding_status)
values('{club}','Audit F01 rollback fixture','{club}','active','ready');
insert into public.players(id,club_id,name,normalized_name)
values({player_id},'{club}','Audit F01 synthetic player','audit f01 synthetic player');
"""]
    for filename, variables, admin_id in (
        ('program_badge_transactions.sql', 'club,player', player_id - 1),
        ('badge_admin_transactions.sql', 'target_club,target_player', player_id - 2),
    ):
        sql = (ROOT / 'tests/sql' / filename).read_text()
        selector = f'select club_id,id into {variables} from public.players order by id limit 1;'
        # The production branch already isolates this regression. Reuse the
        # harness's explicit fixture instead of creating a second club/player.
        if filename == 'program_badge_transactions.sql' and selector not in sql:
            isolated_fixture = """club := 'badge-transaction-' || gen_random_uuid();
 insert into public.clubs(id,name,slug) values(club,'Badge transaction test',club);
 insert into public.players(club_id,name,rating,starting_rating,wins,losses,matches_played,active)
 values(club,'Badge transaction player',1200,1200,0,0,0,true) returning id into player;"""
            if sql.count(isolated_fixture) != 1:
                raise RuntimeError(f'{filename} changed; review fixture isolation before executing it.')
            sql = sql.replace(isolated_fixture, selector)
        admin_insert = 'insert into public.admin_role_assignments(club_id,user_id,email,role) values('
        if sql.count(selector) != 1 or sql.count(admin_insert) != 1:
            raise RuntimeError(f'{filename} changed; review fixture isolation before executing it.')
        sql = sql.replace(selector, f"select club_id,id into {variables} from public.players "
                          "where club_id=current_setting('jupr.audit_badge_fixture_club') order by id limit 1;")
        sql = sql.replace(admin_insert, 'insert into public.admin_role_assignments(id,club_id,user_id,email,role) '
                                       f'values({admin_id},')
        parts.append(sql)
    parts.append("rollback;\nselect 'F01 program and admin badge transactions passed; rolled back' as result,")
    relations = (
        ('clubs', 'id', 'clubs'),
        ('players', 'club_id', 'players'),
        ('player_badges', 'club_id', 'awards'),
        ('badge_program_state', 'club_id', 'program_states'),
        ('badge_seasons', 'club_id', 'seasons'),
        ('admin_role_assignments', 'club_id', 'admin_assignments'),
        ('badge_round_robin_operations', 'club_id', 'round_robin_receipts'),
        ('badge_eval_queue', 'club_id', 'queue_rows'),
        ('admin_activity_log', 'club_id', 'activity_rows'),
        ('live_sessions', 'club_id', 'live_sessions'),
    )
    parts.append(',\n'.join(f" (select count(*) from public.{table} where {key}='{club}') as remaining_{label}"
                            for table, key, label in relations) + ';\n')
    return '\n'.join(parts)


if __name__ == '__main__':
    print(build(), end='')
