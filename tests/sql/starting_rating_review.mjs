import assert from 'node:assert/strict';
import fs from 'node:fs';
import {PGlite} from '@electric-sql/pglite';

const db = new PGlite();
await db.exec(`
create role anon; create role authenticated; create role service_role bypassrls;
create table players(id bigint primary key,club_id text not null,name text,rating numeric(10,4),starting_rating numeric(10,4),matches_played int default 0,wins int default 0,losses int default 0,singles_rating double precision,singles_matches_played int default 0,singles_wins int default 0,singles_losses int default 0,singles_replay_baseline jsonb,unique(club_id,id));
create table league_ratings(id bigint primary key,club_id text,player_id bigint,league_name text,rating numeric(10,4),starting_rating numeric(10,4),matches_played int default 0,wins int default 0,losses int default 0);
create table leagues_metadata(club_id text,league_name text,match_format text);
create table matches(id bigint primary key,club_id text,league text,t1_p1 bigint,t1_p2 bigint,t2_p1 bigint,t2_p2 bigint,score_t1 int,score_t2 int,deleted_at timestamptz,rating_scope text,match_format text);
create table replay_jobs(club_id text,status text);
create table admin_activity_log(id bigserial primary key,club_id text,actor_email text,actor_role text,action_type text,entity_type text,entity_id text,before_json jsonb,after_json jsonb,note text,source_page text,flagged_for_review boolean);
grant select,insert,update,delete on all tables in schema public to service_role;
grant usage on all sequences in schema public to service_role;
insert into players(id,club_id,name,rating,starting_rating,singles_rating,singles_replay_baseline) values
(1,'club','New player',1200,1200,1400,'{"rating":1400,"matches_played":0,"wins":0,"losses":0}'),
(2,'club','History with stale counters',1200,1200,1400,'{"rating":1400,"matches_played":0,"wins":0,"losses":0}'),
(3,'club','Legacy history',1350,1200,1400,'{"rating":1400,"matches_played":0,"wins":0,"losses":0}'),
(4,'club','Singles history',1200,1200,1450,'{"rating":1400,"matches_played":0,"wins":0,"losses":0}'),
(5,'other','Other club',1200,1200,1400,'{"rating":1400,"matches_played":0,"wins":0,"losses":0}'),
(6,'club','New player two',1200,1200,1400,'{"rating":1400,"matches_played":0,"wins":0,"losses":0}');
update players set matches_played=6,wins=4,losses=2 where id=3;
insert into leagues_metadata values('club','Ladder','doubles'),('club','Singles','singles');
insert into league_ratings values(1,'club',1,'Ladder',1200,1200,0,0,0),(2,'club',1,'Singles',1400,1400,0,0,0);
insert into matches values(1,'club','Ladder',2,10,11,12,11,8,null,'','doubles');
insert into matches values(2,'club','Singles',4,null,10,null,11,8,null,'','singles');
insert into matches values(3,'club','Ladder',1,10,11,12,11,8,null,'unrated','doubles');
insert into matches values(4,'club','Ladder',1,10,11,12,11,8,now(),'','doubles');
`);
await db.exec(fs.readFileSync(new URL('../../supabase/migrations/20261030183226_starting_rating_review.sql',import.meta.url),'utf8'));
await db.exec('set role service_role');
const preview=async(ids=[1],format='doubles',club='club')=>(await db.query('select preview_starting_rating_review_v1($1,$2::bigint[],$3) data',[club,ids,format])).rows[0].data;
const confirm=async(reviews,format='doubles',role='administrator')=>(await db.query('select confirm_starting_rating_reviews_v1($1,$2,$3::jsonb,$4,$5,$6) data',['club',format,JSON.stringify(reviews),'admin@example.test',role,'test'])).rows[0].data;
const input=(raw,elo=1300,requested='3.5')=>({player_id:raw.player.id,rating_elo:elo,expected_state_fingerprint:raw.state_fingerprint,requested_rating:requested});
let raw=(await preview()).players[0];
assert.equal(raw.rated_matches,0,'unrated and deleted games do not establish rated history');
const saved=await confirm([input(raw)]);
assert.equal(saved.ok,true);
assert.equal(saved.players[0].player.starting_rating,1300);
assert.equal(saved.players[0].confirmation.rating_elo,1300,'receipt reads committed confirmation');
assert.equal(saved.players[0].confirmation.requested_rating,'3.5');
assert.notEqual(saved.players[0].state_fingerprint,raw.state_fingerprint);
const league=(await db.query('select league_name,rating,starting_rating from league_ratings order by id')).rows;
assert.deepEqual(league,[{league_name:'Ladder',rating:'1300.0000',starting_rating:'1300.0000'},{league_name:'Singles',rating:'1400.0000',starting_rating:'1400.0000'}]);
await assert.rejects(()=>confirm([input(raw)]),/STARTING_REVIEW_CONFLICT/,'stale retry cannot mutate again');
raw=(await preview()).players[0];
await confirm([input(raw,1300)],'doubles');
assert.equal((await db.query('select count(*) n from player_starting_rating_reviews')).rows[0].n,1,'unchanged confirmation updates the same record');
assert.equal((await preview([2])).players[0].rated_matches,1,'actual evidence wins over zero counters');
await assert.rejects(()=>preview([5]),/STARTING_REVIEW_PLAYER_NOT_FOUND/,'cross-club IDs fail');
await assert.rejects(async()=>confirm([input((await preview([2])).players[0])]),/STARTING_REVIEW_HISTORY_EXISTS/);
await assert.rejects(async()=>confirm([input((await preview([3])).players[0])]),/STARTING_REVIEW_HISTORY_EXISTS/,'legacy counters protect existing ratings');
assert.equal((await preview([4],'doubles')).players[0].rated_matches,0,'singles games do not establish doubles history');
assert.equal((await preview([4],'singles')).players[0].rated_matches,1);
const singles=(await preview([1],'singles')).players[0];
await assert.rejects(()=>confirm([input(singles,1500)],'singles'),/STARTING_REVIEW_SINGLES_IMMUTABLE/);
await confirm([input(singles,1400)],'singles');
const before=(await preview([6])).players[0];
await db.exec("insert into replay_jobs values('club','running')");
await assert.rejects(()=>confirm([input(before)]),/STARTING_REVIEW_REPLAY_ACTIVE/);
await db.exec('delete from replay_jobs');
await assert.rejects(()=>confirm([input(before)],'doubles','organizer'),/STARTING_REVIEW_PERMISSION_DENIED/);
await db.exec("insert into matches values(5,'club','Ladder',6,10,11,12,11,7,null,'','doubles')");
await assert.rejects(()=>confirm([input(before)]),/STARTING_REVIEW_HISTORY_EXISTS/,'game inserted after preview refuses seed reset');
const preAtomic=(await preview()).players[0];
await assert.rejects(async()=>confirm([input(preAtomic,1350),input((await preview([2])).players[0],1350)]),/STARTING_REVIEW_HISTORY_EXISTS/);
assert.equal((await preview()).players[0].player.rating,1300,'batch failure rolls back earlier player change');
await db.exec('reset role; revoke insert on admin_activity_log from service_role; set role service_role');
await assert.rejects(async()=>confirm([input((await preview()).players[0],1350)]),/permission denied/);
assert.equal((await preview()).players[0].player.rating,1300,'audit failure rolls back seed change');
await db.exec('reset role');
const perms=(await db.query(`select has_function_privilege('anon','preview_starting_rating_review_v1(text,bigint[],text)','execute') anon_preview,
has_function_privilege('authenticated','confirm_starting_rating_reviews_v1(text,text,jsonb,text,text,text)','execute') auth_confirm,
has_table_privilege('authenticated','player_starting_rating_reviews','select') auth_read,
(select relrowsecurity from pg_class where relname='player_starting_rating_reviews') rls`)).rows[0];
assert.deepEqual(perms,{anon_preview:false,auth_confirm:false,auth_read:false,rls:true});
await db.close();
console.log('Starting-rating SQL: evidence, atomic seeds, league isolation, history and stale guards, singles immutability, audit rollback, replay exclusion, and service-role permissions passed.');
