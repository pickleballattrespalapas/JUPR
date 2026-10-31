// Execute the pool progression SQL against real PostgreSQL/WASM. Synthetic
// schema exercises complete pool schedules and real small-bracket podiums;
// it does not claim Supabase RLS or multi-connection deployment verification.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
const {PGlite}=await import(process.env.PGLITE_MODULE || '@electric-sql/pglite');
const root=fileURLToPath(new URL('../../',import.meta.url));
const read=path=>readFile(`${root}/${path}`,'utf8');
const db=new PGlite();
function routine(source,name){
 const match=source.match(new RegExp(`create or replace function public\\.${name}\\([\\s\\S]*?as \\$function\\$[\\s\\S]*?\\$function\\$;`,'i'));
 assert.ok(match,`Missing ${name}`);return match[0];
}
await db.exec(`
 create role anon; create role authenticated; create role service_role;
 create table public.tournament_event_draws(id uuid primary key,tournament_id uuid,pool_config jsonb,updated_at timestamptz);
 create table public.tournament_teams(id uuid primary key,tournament_id uuid,draw_id uuid,competition_status text default 'ACTIVE');
 create table public.tournament_games(id uuid primary key default gen_random_uuid(),tournament_id uuid,draw_id uuid,stage text,
   series_parent_game_id uuid,playoff_round text,team_a_id uuid,team_b_id uuid,score_a integer,score_b integer,
   winner_team_id uuid,loser_team_id uuid,finalized_at timestamptz);
 create table public.tournament_registration_days(id text);
 create table public.tournament_event_options(id text);
 create table public.tournament_day_live_runs(id uuid);
 create table public.tournament_day_live_draws(id uuid);
`);
const config=await read('supabase/migrations/20261109010000_tournament_draw_pools.sql');
const recovery=await read('supabase/migrations/20261109010100_tournament_pool_schedule_recovery.sql');
await db.exec(routine(config,'assert_tournament_draw_pool_config'));
await db.exec(routine(config,'tournament_pool_pair_allowed'));
await db.exec(routine(recovery,'tournament_draw_expected_rr_game_count'));
const migration=await read('supabase/migrations/20261109010200_tournament_pool_playoff_progression.sql');
await db.exec(migration);
const firstCatalog=await db.query("select proname,proacl,prosecdef,proconfig from pg_proc where pronamespace='public'::regnamespace order by proname");
assert.ok(firstCatalog.rows.every(row=>row.prosecdef===false),'No progression routine may acquire definer privileges');
for(const signature of [
 'public.assert_tournament_pool_schedule_complete(uuid)',
 'public.assert_tournament_pool_playoff_selection(uuid,integer,jsonb)',
 'public.tournament_draw_expected_playoff_podium(uuid,uuid)',
 'public.admin_generate_tournament_day_playoffs_cas(text,text,text,text,integer,bigint,bigint,bigint,timestamptz,jsonb,jsonb,jsonb,text,text,text)',
 'public.admin_close_tournament_day_live_cas(text,text,text,bigint,bigint,text,text,text)',
 'public.admin_award_tournament_draw_podium_cas(text,text,text,timestamptz,jsonb,jsonb,jsonb,jsonb)',
]) {
 const row=(await db.query(`select has_function_privilege($1,$2,'execute') as anon,has_function_privilege($3,$2,'execute') as authenticated,has_function_privilege($4,$2,'execute') as service_role`,['anon',signature,'authenticated','service_role'])).rows[0];
 assert.deepEqual(row,{anon:false,authenticated:false,service_role:true},`Incorrect execute grants for ${signature}`);
}
await db.exec(migration);
assert.deepEqual(await db.query("select proname,proacl,prosecdef,proconfig from pg_proc where pronamespace='public'::regnamespace order by proname"),firstCatalog);
console.log('PASS: full progression migration applies and reapplies with preserved invoker security and ACLs');
const tour='10000000-0000-4000-8000-000000000000';
const draw='20000000-0000-4000-8000-000000000000';
const ids=Array.from({length:6},(_,i)=>`30000000-0000-4000-8000-00000000000${i+1}`);
const poolConfig=(a,b)=>({version:1,pools:[{id:'A',name:'Pool A',team_ids:ids.slice(0,3),advance_count:a},{id:'B',name:'Pool B',team_ids:ids.slice(3),advance_count:b}]});
await db.query('insert into tournament_event_draws values($1,$2,$3,now())',[draw,tour,JSON.stringify(poolConfig(1,1))]);
for(const id of ids)await db.query('insert into tournament_teams(id,tournament_id,draw_id) values($1,$2,$3)',[id,tour,draw]);
const addGame=async(a,b,stage='ROUND_ROBIN',round=null,complete=true)=>await db.query(`insert into tournament_games(tournament_id,draw_id,stage,playoff_round,team_a_id,team_b_id,score_a,score_b,winner_team_id,loser_team_id,finalized_at) values($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11) returning id`,[tour,draw,stage,round,a,b,complete?11:null,complete?5:null,complete?a:null,complete?b:null,complete?new Date().toISOString():null]);
const reject=async(promise,code)=>{let error;try{await promise;}catch(e){error=e;}assert.ok(error,'Unsafe operation was accepted');assert.equal(error.code,code);};
await reject(db.query('select assert_tournament_pool_schedule_complete($1)',[draw]),'22023');
for(const group of [ids.slice(0,3),ids.slice(3)])for(let i=0;i<3;i++)for(let j=i+1;j<3;j++)await addGame(group[i],group[j]);
await db.query('select assert_tournament_pool_schedule_complete($1)',[draw]);
const bad=(await addGame(ids[0],ids[3])).rows[0].id;
await reject(db.query('select assert_tournament_pool_schedule_complete($1)',[draw]),'22023');
await db.query('delete from tournament_games where id=$1',[bad]);
const duplicate=(await addGame(ids[0],ids[1])).rows[0].id;
await reject(db.query('select assert_tournament_pool_schedule_complete($1)',[draw]),'22023');
await db.query('delete from tournament_games where id=$1',[duplicate]);
console.log('PASS: 3+3 pools require all six unique in-pool pairs; missing, cross-pool and duplicate games reject');
const selection=async(count,seeds,reason)=>db.query('select assert_tournament_pool_playoff_selection($1,$2,$3)',[draw,count,JSON.stringify({template_code:`SINGLE_ELIMINATION_${count}`,seed_team_ids:seeds,...(reason===undefined?{}:{override_reason:reason})})]);
await selection(2,[ids[0],ids[3]]);
await reject(selection(3,[ids[0],ids[1],ids[3]]),'22023');
await reject(selection(3,[ids[0],ids[1],ids[3]],'   '),'22023');
await selection(3,[ids[0],ids[1],ids[3]],'Director approved an additional qualifier.');
await reject(selection(2,[ids[0],ids[0]]),'22023');
await reject(selection(2,[ids[0],'99999999-0000-4000-8000-000000000000']),'22023');
await db.query("update tournament_teams set competition_status='RETIRED' where id=$1",[ids[1]]);
await reject(selection(3,[ids[0],ids[1],ids[3]],'Withdrawals reviewed.'),'22023');
await db.query("update tournament_teams set competition_status='ACTIVE' where id=$1",[ids[1]]);
await reject(selection(1,[ids[0]],'Unsupported count.'),'22023');
await reject(selection(2,[ids[0],ids[3]],'x'.repeat(501)),'22023');
console.log('PASS: qualifier totals, duplicates, foreign/retired teams and unsupported counts reject; explicit explained total override succeeds');
const podium=async()=> (await db.query('select * from tournament_draw_expected_playoff_podium($1,$2) order by placement',[tour,draw])).rows;
await addGame(ids[0],ids[3],'PLAYOFF','Final');
assert.deepEqual(await podium(),[{placement:1,team_id:ids[0]},{placement:2,team_id:ids[3]}]);
await db.query('update tournament_event_draws set pool_config=null where id=$1',[draw]);
assert.deepEqual(await podium(),[],'Legacy incomplete bronze bracket must not become a two-team podium');
await db.query('update tournament_event_draws set pool_config=$2 where id=$1',[draw,JSON.stringify(poolConfig(2,1))]);
assert.deepEqual(await podium(),[{placement:1,team_id:ids[0]},{placement:2,team_id:ids[3]}],'A reviewed two-qualifier override uses its real bracket rather than the original quotas');
await addGame(ids[3],ids[1],'PLAYOFF','SF');
assert.deepEqual(await podium(),[{placement:1,team_id:ids[0]},{placement:2,team_id:ids[3]},{placement:3,team_id:ids[1]}]);
await db.query("update tournament_games set finalized_at=null where playoff_round='SF'");
assert.deepEqual(await podium(),[],'Unfinished semifinal must block podium');
await db.query("update tournament_games set finalized_at=now() where playoff_round='SF'");
await db.query('update tournament_event_draws set pool_config=$2 where id=$1',[draw,JSON.stringify(poolConfig(2,2))]);
await addGame(ids[0],ids[4],'PLAYOFF','SF');
assert.deepEqual(await podium(),[],'Four qualifiers require a bronze match');
await addGame(ids[1],ids[4],'PLAYOFF','Bronze');
assert.deepEqual(await podium(),[{placement:1,team_id:ids[0]},{placement:2,team_id:ids[3]},{placement:3,team_id:ids[1]}]);
await db.query("update tournament_games set winner_team_id=team_b_id where playoff_round='Final'");
assert.deepEqual(await podium(),[],'Score/winner disagreement must block podium');
console.log('PASS: final-only two-place, three-qualifier semifinal-loser bronze, and explicit bronze podiums; incomplete or inconsistent evidence rejects');
await db.close();
