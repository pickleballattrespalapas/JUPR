import assert from 'node:assert/strict';
import fs from 'node:fs';
import { PGlite } from '@electric-sql/pglite';

const db = new PGlite();
const migration = new URL('../../supabase/migrations/20261111200814_immediate_player_game_reactivation.sql', import.meta.url);
await db.exec(`begin;
  create role anon; create role authenticated; create role service_role;
  create table public.players (
    id bigint primary key, name text, active boolean default true,
    inactive_at timestamptz, last_game_at timestamptz,
    rating numeric default 1600, wins integer default 7,
    losses integer default 1, matches_played integer default 8
  );`);
await db.exec(fs.readFileSync(new URL('../../supabase/migrations/20261004035233_player_game_activity_only.sql', import.meta.url), 'utf8'));

// A date-only game predates the automatic marker assigned on profile creation.
await db.exec(`insert into players(id,name) values (1,'Same-day signup');
  update players set last_game_at=date_trunc('day',now()) where id=1;
  insert into players(id,name,active,inactive_at,last_game_at) values
    (2,'Old automatic flag',false,now()-interval '3 days',now()-interval '2 days'),
    (3,'Explicit exclusion',false,now()-interval '1 hour',now()-interval '2 days'),
    (4,'Manual hidden flag',false,null,now()-interval '2 days'),
    (5,'Boundary',true,null,now()-interval '14 days'),
    (6,'Recent boundary',true,null,now()-interval '14 days'+interval '1 second'),
    (7,'Future game',true,null,now()+interval '1 day'),
    (8,'Never played',true,null,null),
    (9,'Old (MERGED into Remaining #1)',true,null,now()-interval '2 days');
  create temp table before_results as
    select id,rating,wins,losses,matches_played,last_game_at from players;`);
assert.ok((await db.query('select inactive_at from players where id=1')).rows[0].inactive_at,
  'fixture reproduces the original same-day signup bug');

await db.exec(fs.readFileSync(migration, 'utf8'));
const players = Object.fromEntries((await db.query('select * from players order by id')).rows.map(row => [row.id, row]));
for (const id of [1,2,6]) {
  assert.equal(players[id].active, true);
  assert.equal(players[id].inactive_at, null, `player ${id} is immediately visible`);
}
for (const id of [3,4]) assert.equal(players[id].active, false, 'manual visibility is preserved');
assert.ok(players[3].inactive_at);
assert.equal(players[4].inactive_at, null);
for (const id of [5,7,8,9]) assert.ok(players[id].inactive_at, `ineligible player ${id} stays inactive`);
assert.equal((await db.query(`select count(*)::integer n from (
  (select id,rating,wins,losses,matches_played,last_game_at from players) except select * from before_results
) changed`)).rows[0].n, 0, 'repair leaves ratings, records, and game dates unchanged');
await db.exec('create temp table after_activity as select id,active,inactive_at from players; select mark_inactive_players();');
assert.equal((await db.query(`select count(*)::integer n from (
  (select id,active,inactive_at from players) except select * from after_activity
) changed`)).rows[0].n, 0, 'daily repair is idempotent');

// A normal writer need only record recent activity; no worker or explicit marker reset.
await db.exec(`insert into players(id,name) values (20,'New scored participant');
  savepoint scored_game;
  update players set last_game_at=date_trunc('day',now()) where id=20;`);
assert.equal((await db.query('select inactive_at from players where id=20')).rows[0].inactive_at, null);
await db.exec('rollback to scored_game');
assert.ok((await db.query('select inactive_at from players where id=20')).rows[0].inactive_at,
  'failed match transaction rolls activity back too');
await db.exec(`update players set last_game_at=date_trunc('day',now()) where id=20;
  update players set inactive_at=now() where id=20;`);
assert.equal((await db.query('select inactive_at from players where id=20')).rows[0].inactive_at, null,
  'a legacy writer cannot reintroduce an automatic marker on a recent active profile');
await db.exec('update players set last_game_at=null where id=20');
assert.ok((await db.query('select inactive_at from players where id=20')).rows[0].inactive_at,
  'removing the final recorded game restores inactivity');
await db.exec(`update players set last_game_at=now()-interval '30 days',inactive_at=null where id=20;`);
assert.ok((await db.query('select inactive_at from players where id=20')).rows[0].inactive_at,
  'historical imports do not establish current activity');

for (const fn of ['enforce_player_game_activity()', 'mark_inactive_players()']) {
  for (const role of ['anon', 'authenticated', 'service_role']) {
    assert.equal((await db.query('select has_function_privilege($1,$2,$3) allowed', [role,fn,'execute'])).rows[0].allowed, false);
  }
}
assert.equal((await db.query(`select count(*)::integer n from pg_proc
  where oid in ('enforce_player_game_activity()'::regprocedure,'mark_inactive_players()'::regprocedure)
  and prosecdef`)).rows[0].n, 0, 'functions retain invoker security');
await db.exec('rollback');
await db.close();
console.log('Player reactivation SQL: immediate same-day activity, legacy writers, repair, manual/merged exclusions, boundaries, rollback, unchanged results, idempotency, and privileges passed.');
