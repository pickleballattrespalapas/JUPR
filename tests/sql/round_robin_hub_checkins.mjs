import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const { PGlite } = await import(process.env.PGLITE_MODULE || '@electric-sql/pglite');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const db = new PGlite();
await db.exec(`create role anon; create role authenticated; create role service_role bypassrls;
 create table live_sessions(id uuid default gen_random_uuid(), club_id text, session_key text, title text,
 state jsonb default '{}', status text default 'active', version bigint default 1,
 created_at timestamptz default now(), updated_at timestamptz default now(), last_seen_at timestamptz,
 expires_at timestamptz, completed_at timestamptz, last_operation_key text, last_request_fingerprint text,
 pending_operation_key text, unique(club_id,session_key));
 grant all on live_sessions to service_role;`);
for (const name of ['20261020033713_round_robin_hub_atomic_operations.sql', '20261021144253_round_robin_hub_public_checkin.sql']) {
  await db.exec(fs.readFileSync(path.join(root, 'supabase/migrations', name), 'utf8'));
}
const hub = '10000000-0000-4000-8000-000000000001';
const other = '10000000-0000-4000-8000-000000000002';
await db.query(`insert into round_robin_hubs(id,club_id,title,state) values ($1,'club','Event',$2),($3,'elsewhere','Other','{}')`,[hub,{registration_open:true},other]);
await db.query(`insert into live_sessions(club_id,session_key,state) values ('club','rr',$1),('club','unlinked',$1)`,[{mode:'admin_play_generator',page_state:{event:{status:'active',generatorKind:'round_robin'}}}]);
await db.query(`insert into round_robin_hub_sessions(club_id,hub_id,session_key) values('club',$1,'rr')`,[hub]);
await db.exec('set role service_role');
const submit = (name='New Player',key='new player',group='rr',id=hub) => db.query('select submit_round_robin_hub_checkin($1,$2,$3,$4) result',[id,group,name,key]);
assert.equal((await submit()).rows[0].result.ok,true);
await submit('new PLAYER');
assert.equal((await db.query('select count(*) n from round_robin_hub_checkins')).rows[0].n,1);
assert.equal((await db.query('select version from round_robin_hubs where id=$1',[hub])).rows[0].version,1);
await assert.rejects(()=>submit('Other','other','unlinked'),/Choose a round robin/);
await assert.rejects(()=>submit('Other','other','rr',other),/Check-in is closed/);
await assert.rejects(()=>submit('Other','other','rr','10000000-0000-4000-8000-000000000099'),/not found/);
await assert.rejects(()=>submit(' ',' '),/Invalid check-in/);
await db.query(`update round_robin_hubs set state='{"registration_open":false}' where id=$1`,[hub]);
await assert.rejects(()=>submit('Other','other'),/closed/);
assert.equal((await submit()).rows[0].result.ok,true,'lost-response replay succeeds after closure');
await db.query(`update round_robin_hubs set state='{"registration_open":true}' where id=$1`,[hub]);
await db.exec(`update live_sessions set state=jsonb_set(state,'{generator_submission}','{"id":"submitted"}') where session_key='rr'`);
await assert.rejects(()=>submit('Other','other'),/not accepting/);
await db.exec(`update live_sessions set state=(state-'generator_submission') || '{"page_state":{"event":{"status":"active","generatorKind":"round_robin","hubFinishAfterRound":1}}}' where session_key='rr'`);
await assert.rejects(()=>submit('Other','other'),/not accepting/);
await db.exec(`update live_sessions set state=state #- '{page_state,event,hubFinishAfterRound}' where session_key='rr'`);
for(let i=1;i<120;i++) await submit(`Player ${i}`,`player ${i}`);
await assert.rejects(()=>submit('Over limit','over limit'),/limit reached/);
await db.exec('reset role');
const permissions = (await db.query(`select
 has_function_privilege('anon','submit_round_robin_hub_checkin(uuid,text,text,text)','execute') anon_call,
 has_function_privilege('authenticated','submit_round_robin_hub_checkin(uuid,text,text,text)','execute') auth_call,
 has_table_privilege('anon','round_robin_hub_checkins','select') anon_read,
 has_table_privilege('authenticated','round_robin_hub_checkins','insert') auth_write,
 (select relrowsecurity from pg_class where relname='round_robin_hub_checkins') rls`)).rows[0];
assert.deepEqual(permissions,{anon_call:false,auth_call:false,anon_read:false,auth_write:false,rls:true});
await db.exec('set role anon');
await assert.rejects(()=>submit(),/permission denied/);
await assert.rejects(()=>db.query('select * from round_robin_hub_checkins'),/permission denied/);
await db.close();
console.log('Check-in SQL passed: intake, dedupe, closure, group scoping, submitted/finishing rejection, limits, and RLS/privileges.');
