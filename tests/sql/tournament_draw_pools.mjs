// Real PostgreSQL/WASM routines on an isolated minimal schema. This checks
// transactional semantics, not PostgREST, the full schema, or concurrent clients.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { fileURLToPath } from 'node:url';
const { PGlite } = await import(process.env.PGLITE_MODULE || '@electric-sql/pglite');
const root = fileURLToPath(new URL('../../', import.meta.url));
const read = path => readFile(`${root}/${path}`, 'utf8');
const db = new PGlite();
await db.exec(`
create role anon; create role authenticated; create role service_role;
create schema private;
create table public.tournaments(id uuid primary key, club_id text, status text);
create table public.tournament_event_draws(id uuid primary key,tournament_id uuid,registration_day_id text,event_option_id text,name text,status text,updated_at timestamptz default now());
create table public.tournament_teams(id uuid primary key,tournament_id uuid,draw_id uuid,team_number integer,player1_id integer,updated_at timestamptz default now());
create table public.tournament_games(id uuid primary key,tournament_id uuid,draw_id uuid,registration_day_id text,event_option_id text,stage text,rr_round_number integer,rr_slot_number integer,team_a_id uuid,team_b_id uuid,score_a integer,score_b integer,winner_team_id uuid,loser_team_id uuid,finalized_at timestamptz,created_at timestamptz,updated_at timestamptz default now(),series_parent_game_id uuid);
create table public.tournament_podium(id uuid,tournament_id uuid,draw_id uuid);
create table public.tournament_day_live_draws(id uuid,tournament_id uuid,draw_id uuid);
create table public.tournament_day_live_queue(id uuid,tournament_id uuid,draw_id uuid);
create table public.matches(id uuid,club_id text,tournament_id uuid,tournament_game_id uuid);
create table public.player_badges(id uuid,club_id text,context_type text,context_id text);
`);
function routine(source, name) {
  const start = source.indexOf(`create or replace function ${name}(`);
  assert.ok(start >= 0, name);
  const end = source.indexOf('$function$;', start) + '$function$;'.length;
  return source.slice(start, end);
}
const oldRecovery = await read('supabase/migrations/20261108003000_tournament_terminal_completion_and_schedule_recovery.sql');
const oldSeries = await read('supabase/migrations/20261108026000_tournament_best_of_three_game_scores.sql');
await db.exec(routine(oldRecovery, 'private.assert_tournament_draw_recovery_dependencies_clear'));
await db.exec(routine(oldSeries, 'private.assert_tournament_draw_recovery_snapshot'));
const migrations = [
  await read('supabase/migrations/20261109010000_tournament_draw_pools.sql'),
  await read('supabase/migrations/20261109010100_tournament_pool_schedule_recovery.sql'),
];
for (const migration of migrations) await db.exec(migration);
for (const migration of migrations) await db.exec(migration); // idempotent forward DDL
const acl = (await db.query(`select has_function_privilege('anon','public.admin_save_tournament_draw_pools_cas(text,text,text,timestamptz,jsonb,jsonb)','EXECUTE') as anon, has_function_privilege('authenticated','public.admin_save_tournament_draw_pools_cas(text,text,text,timestamptz,jsonb,jsonb)','EXECUTE') as authenticated, has_function_privilege('service_role','public.admin_save_tournament_draw_pools_cas(text,text,text,timestamptz,jsonb,jsonb)','EXECUTE') as service`)).rows[0];
assert.deepEqual(acl, {anon:false,authenticated:false,service:true});
const state = async f => ({
  draw: (await db.query('select *,updated_at::text as updated_at from tournament_event_draws where id=$1',[f.draw])).rows[0],
  teams: (await db.query('select id,updated_at::text as updated_at from tournament_teams where draw_id=$1 order by id',[f.draw])).rows,
  games: (await db.query('select *,updated_at::text as updated_at from tournament_games where draw_id=$1 order by id',[f.draw])).rows,
});
async function fixture(sizes) {
  const f = {tour:randomUUID(),draw:randomUUID(),teamIds:Array.from({length:sizes.reduce((a,b)=>a+b,0)},randomUUID)};
  await db.query("insert into tournaments values($1,'pool-test','PUBLISHED')",[f.tour]);
  await db.query("insert into tournament_event_draws(id,tournament_id,name,status) values($1,$2,'Pool test','draft')",[f.draw,f.tour]);
  for (const [i,id] of f.teamIds.entries()) await db.query('insert into tournament_teams(id,tournament_id,draw_id,team_number) values($1,$2,$3,$4)',[id,f.tour,f.draw,i+1]);
  let offset=0;
  f.config={version:1,pools:sizes.map((n,i)=>({id:String.fromCharCode(65+i),name:`Pool ${String.fromCharCode(65+i)}`,team_ids:f.teamIds.slice(offset,offset+=n),advance_count:2}))};
  return f;
}
async function save(f, config=f.config, overrides={}) {
  const s=await state(f);
  return db.query('select admin_save_tournament_draw_pools_cas($1,$2,$3,$4,$5::jsonb,$6::jsonb) as result',[
    overrides.club||'pool-test', f.tour,f.draw,overrides.version||s.draw.updated_at,
    JSON.stringify(overrides.teams||s.teams),JSON.stringify(config),
  ]);
}
async function rejected(operation, marker) {
  await assert.rejects(operation, error => error.message.includes(marker));
}
function schedule(f) {
  const all=[];
  const nextSlot={};
  for(const pool of f.config.pools) {
    const n=pool.team_ids.length;
    const rounds=n===3?[[[0,1]],[[1,2]],[[0,2]]]:[[[0,1],[2,3]],[[0,2],[1,3]],[[0,3],[1,2]]];
    rounds.forEach((pairs,index)=>pairs.forEach(([a,b])=>{
      nextSlot[index]=(nextSlot[index]||0)+1;
      all.push({id:randomUUID(),stage:'ROUND_ROBIN',rr_round_number:index+1,rr_slot_number:nextSlot[index],team_a_id:pool.team_ids[a],team_b_id:pool.team_ids[b]});
    }));
  }
  return all;
}
async function insert(f,game) {
  await db.query(`insert into tournament_games(id,tournament_id,draw_id,stage,rr_round_number,rr_slot_number,team_a_id,team_b_id,score_a,score_b,winner_team_id,loser_team_id,finalized_at) values($1,$2,$3,'ROUND_ROBIN',$4,$5,$6,$7,$8,$9,$10,$11,$12)`,[game.id,f.tour,f.draw,game.rr_round_number,game.rr_slot_number,game.team_a_id,game.team_b_id,game.score_a??null,game.score_b??null,game.winner_team_id??null,game.loser_team_id??null,game.finalized_at??null]);
}
async function recover(f,mode,games,overrides={}) {
  const s=await state(f);
  return db.query(`select public.admin_${mode}_tournament_round_robin_games_cas($1,$2,$3,$4,$5::jsonb,$6::jsonb,$7::jsonb) as result`,['pool-test',f.tour,f.draw,overrides.version||s.draw.updated_at,JSON.stringify(s.teams),JSON.stringify(s.games.map(({id,updated_at})=>({id,updated_at}))),JSON.stringify(games)]);
}
const f=await fixture([3,3]);
await rejected(()=>save(f,f.config,{club:'other'}),'DRAW_STALE');
await rejected(()=>save(f,f.config,{version:'2000-01-01'}),'DRAW_STALE');
await rejected(()=>save(f,f.config,{teams:(f.teamIds.slice(1).map(id=>({id,updated_at:'2000-01-01'})))}),'TEAM_SNAPSHOT_STALE');
for(const mutate of [
 c=>c.pools[1].team_ids[0]=c.pools[0].team_ids[0],
 c=>c.pools[1].team_ids.pop(),
 c=>c.pools[1].team_ids[0]=randomUUID(),
 c=>c.pools[1].name='pool a',
 c=>c.pools[1].advance_count=1.5,
 c=>c.pools.forEach(p=>p.advance_count=0),
]) {const invalid=structuredClone(f.config);mutate(invalid);await rejected(()=>save(f,invalid),'POOL_CONFIG_INVALID');}
await save(f);
assert.equal((await db.query('select tournament_draw_expected_rr_game_count($1) as n',[f.draw])).rows[0].n,6);
await rejected(()=>db.query('delete from tournament_teams where id=$1',[f.teamIds[0]]),'POOL_ROSTER_LOCKED');
await rejected(()=>db.query('insert into tournament_teams(id,tournament_id,draw_id) values($1,$2,$3)',[randomUUID(),f.tour,f.draw]),'POOL_ROSTER_LOCKED');
await save(f,null);
await save(f);
await rejected(()=>insert(f,{...schedule(f)[0],team_b_id:f.config.pools[1].team_ids[0]}),'POOL_PAIR_INVALID');
await insert(f,schedule(f)[0]);
await rejected(()=>save(f,null),'POOLS_LOCKED');
await recover(f,'rebuild',schedule(f));
assert.equal((await state(f)).games.length,6);
console.log('PASS: strict3+3 assignment, ownership/CAS, roster locks, within-pool games and6-game rebuild');

const g=await fixture([4,4]);await save(g);
const complete=schedule(g);const played={...complete[0],score_a:11,score_b:8,winner_team_id:complete[0].team_a_id,loser_team_id:complete[0].team_b_id,finalized_at:'2026-10-31T10:00:00Z'};
await insert(g,played);
const bad=complete.slice(1);bad[0]={...bad[0],team_b_id:g.config.pools[1].team_ids[0]};
await rejected(()=>recover(g,'reconcile',bad),'RECONCILE_BLOCKED');
await rejected(()=>recover(g,'rebuild',complete),'REBUILD_BLOCKED');
await recover(g,'reconcile',complete.slice(1));
const gs=await state(g);assert.equal(gs.games.length,12);
assert.equal(gs.games.find(row=>row.id===played.id).score_a,11);
await rejected(()=>recover(g,'rebuild',complete,{version:'2000-01-01'}),'DRAW_STALE');
console.log('PASS:4+4 reconciliation keeps finalized result and requires exactly12 valid pool games');

const h=await fixture([4,3]);await save(h);
assert.equal((await db.query('select tournament_draw_expected_rr_game_count($1) as n',[h.draw])).rows[0].n,9);
await insert(h,schedule(h)[0]);
await rejected(()=>recover(h,'rebuild',schedule(h).slice(1)),'REBUILD_BLOCKED');
await recover(h,'rebuild',schedule(h));
assert.equal((await state(h)).games.length,9);
console.log('PASS: uneven4+3 needs exactly9 pairings; incomplete rebuild rejected');
await db.close();
