import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const { PGlite } = await import(process.env.PGLITE_MODULE || '@electric-sql/pglite');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const migration = fs.readFileSync(path.join(root, 'supabase/migrations/20261020033713_round_robin_hub_atomic_operations.sql'), 'utf8');
const db = new PGlite();
await db.exec(`
  create role anon; create role authenticated; create role service_role bypassrls;
  create table public.live_sessions (
    id uuid primary key default gen_random_uuid(), club_id text not null, session_key text not null,
    state jsonb not null default '{}', title text, status text not null default 'active',
    version bigint not null default 1, created_at timestamptz not null default clock_timestamp(),
    updated_at timestamptz not null default clock_timestamp(), last_seen_at timestamptz,
    expires_at timestamptz, completed_at timestamptz, last_operation_key text,
    last_request_fingerprint text, pending_operation_key text,
    unique(club_id,session_key)
  );
  grant select,insert,update,delete on public.live_sessions to service_role;
`);
await db.exec(migration);
await db.exec(migration);
const hubId = '10000000-0000-4000-8000-000000000001';
const otherHubId = '10000000-0000-4000-8000-000000000002';
const assignment = (session,match='m1') => ({session_key:session,match_id:match,round_number:1,assigned_at:'2026-10-20T04:00:00Z'});
const court = (id,label,dedicated=null,current=null,next=null) => ({id,label,dedicated_session_key:dedicated,open:true,current,next});
const state = {courts:[court('c1','Court 1'),court('c2','Court 2')],groups:[],players:{}};
let op = 0;
const clone = x => structuredClone(x);
async function commit({id=hubId,version=0,hubState=state,sessions=[],key,fp='0123456789abcdef0123456789abcdef',status='active'}={}) {
  const result=await db.query(`select public.commit_round_robin_hub($1,$2,$3,$4,$5,$6,$7,$8,$9,$10) result`,
    ['club',id,version,key || `operation-${++op}`,fp,'Friday Night',status,hubState,sessions,'joe@example.test']);
  return result.rows[0].result;
}
async function row(key) { return (await db.query('select * from live_sessions where club_id=$1 and session_key=$2',['club',key])).rows[0]; }
async function hub(id=hubId) { return (await db.query('select * from round_robin_hubs where id=$1',[id])).rows[0]; }
async function readSet(keys,patches={}) { return Promise.all(keys.map(async key => {const r=await row(key);return {session_key:key,expected_version:r.updated_at,expected_numeric_version:Number(r.version),patch:patches[key] || {}};})); }
async function rejects(fn,pattern) { await assert.rejects(fn,pattern); }
async function insertSession(key,playerIds) {
 const participants=playerIds.map((id,i)=>({id:`p${i+1}`,name:`Player ${id}`,player_id:id}));
 const event={generatorKind:'round_robin',currentRoundNumber:1,participants,rounds:[{number:1,status:'active',matches:[{id:'m1',sideA:['p1'],sideB:['p2']}]}]};
 await db.query('insert into live_sessions(club_id,session_key,state) values($1,$2,$3)',['club',key,{mode:'admin_play_generator',page_state:{event}}]);
}
await db.exec('set role service_role');
let created=await commit({key:'create-hub-0001'});
assert.equal(created.hub.version,1);
assert.equal((await commit({key:'create-hub-0001'})).replayed,true);
assert.equal((await hub()).version,1);
await rejects(async()=>commit({key:'create-hub-0001',fp:'ffffffffffffffffffffffffffffffff'}),/different hub command/);
await rejects(async()=>commit({version:0}),/Hub changed/);
await insertSession('a',[1,2]); await insertSession('b',[3,4]); await insertSession('c',[1,5]);
let h=clone(state); h.groups=[{session_key:'a'},{session_key:'b'}];
const patches={}; for(const k of ['a','b']) {const r=await row(k);patches[k]={state:{...r.state,round_robin_hub_id:hubId}};}
let attached=await commit({version:1,hubState:h,sessions:await readSet(['a','b'],patches)});
assert.equal(attached.hub.version,2);
assert.equal((await db.query('select count(*) n from round_robin_hub_sessions')).rows[0].n,2);
await rejects(async()=>commit({version:2,hubState:h,sessions:await readSet(['a'])}),/Every hub session/);
await rejects(async()=>db.query(`update live_sessions set state=jsonb_set(state,'{page_state,event,currentRoundNumber}','2') where session_key='a'`),/shared court authority/);
await rejects(async()=>db.query(`delete from live_sessions where session_key='a'`),/Remove this session/);
await db.query(`update live_sessions set state=jsonb_set(state,'{sharing}','{"enabled":true}') where session_key='a'`);
// A stale generator read must reject the entire court allocation.
let stale=await readSet(['a','b']);
await db.query(`update live_sessions set updated_at=clock_timestamp(),version=version+1 where session_key='b'`);
let assigned=clone(h);assigned.courts[0].current=assignment('a');
await rejects(async()=>commit({version:2,hubState:assigned,sessions:stale}),/round robin changed/);
assert.equal((await hub()).state.courts[0].current,null);
let allocated=await commit({version:2,hubState:assigned,sessions:await readSet(['a','b'])});
assert.equal(allocated.hub.version,3);
assert.equal((await db.query('select count(*) n from round_robin_hub_player_leases')).rows[0].n,2);
// Score and free the court in the same commit; standings still exclude active round.
let freed=clone(assigned);freed.courts[0].current=null;
const a=await row('a');const scoredState=clone(a.state);Object.assign(scoredState.page_state.event.rounds[0].matches[0],{scoreA:11,scoreB:8});
let saved=await commit({version:3,hubState:freed,sessions:await readSet(['a','b'],{a:{state:scoredState}}),key:'save-score-0001'});
assert.equal(saved.hub.version,4);
assert.equal((await row('a')).state.page_state.event.rounds[0].status,'active');
assert.equal((await db.query('select count(*) n from round_robin_hub_player_leases')).rows[0].n,0);
assert.equal((await commit({version:3,hubState:freed,sessions:[],key:'save-score-0001'})).replayed,true);
// Dedicated allocation enforced even if a caller bypasses domain logic.
let dedicated=clone(freed);dedicated.courts[0].dedicated_session_key='a';dedicated.courts[1].current=assignment('a');
await rejects(async()=>commit({version:4,hubState:dedicated,sessions:await readSet(['a','b'])}),/outside its group allocation/);
assert.equal((await hub()).version,4);
// Another hub cannot claim the same physical name with case/spacing changes.
await rejects(async()=>commit({id:otherHubId,hubState:{courts:[court('other',' court   1 ')],groups:[]}}),/duplicate key/);
assert.equal(await hub(otherHubId),undefined);
// Attach a separate session containing the same known player to another hub.
const c=await row('c');let otherState={courts:[court('other','Court 3')],groups:[{session_key:'c'}],players:{}};
await commit({id:otherHubId,hubState:otherState,sessions:await readSet(['c'],{c:{state:{...c.state,round_robin_hub_id:otherHubId}}})});
let reallocated=clone(freed);reallocated.courts[0].current=assignment('a');
await commit({version:4,hubState:reallocated,sessions:await readSet(['a','b'])});
otherState.courts[0].current=assignment('c');
const cBeforeConflict=await row('c');
const cEdited=clone(cBeforeConflict.state);cEdited.page_state.event.name='Must roll back';
await rejects(async()=>commit({id:otherHubId,version:1,hubState:otherState,sessions:await readSet(['c'],{c:{state:cEdited}})}),/duplicate key/);
assert.deepEqual(await row('c'),cBeforeConflict);
assert.equal((await hub(otherHubId)).state.courts[0].current,null);
// A guest alias cannot evade a known profile's occupied-player lease in another hub.
await insertSession('d',[6,7]);
const d=await row('d');const dState=clone(d.state);delete dState.page_state.event.participants[0].player_id;
dState.page_state.event.participants[0].name='  Player   1  ';
const thirdHubId='10000000-0000-4000-8000-000000000003';
dState.round_robin_hub_id=thirdHubId;
const thirdState={courts:[court('third','Court 4',null,null,assignment('d'))],groups:[{session_key:'d'}],players:{}};
await rejects(async()=>commit({id:thirdHubId,hubState:thirdState,sessions:await readSet(['d'],{d:{state:dState}})}),/same club profile/);
assert.equal(await hub(thirdHubId),undefined);
assert.equal((await row('d')).state.round_robin_hub_id,undefined);
// Simulated stale two-operator assignments: only one expected-version winner.
let operator1=clone(reallocated);operator1.courts[1].current=assignment('b');
const read=await readSet(['a','b']);
await commit({version:5,hubState:operator1,sessions:read});
await rejects(async()=>commit({version:5,hubState:reallocated,sessions:read}),/Hub changed/);
// Completing a hub with an occupied court is rejected without touching scores.
await rejects(async()=>commit({version:6,hubState:operator1,sessions:await readSet(['a','b']),status:'completed'}),/Release every court/);
await rejects(async()=>commit({version:6,hubState:operator1,sessions:await readSet(['a','b'],{a:{edit_token_hash:'bad'}})}),/Unsupported session field/);
await db.exec('reset role');
const perms=await db.query(`select
 has_function_privilege('anon','public.commit_round_robin_hub(text,uuid,bigint,text,text,text,text,jsonb,jsonb,text)','execute') anon_execute,
 has_function_privilege('authenticated','public.commit_round_robin_hub(text,uuid,bigint,text,text,text,text,jsonb,jsonb,text)','execute') authenticated_execute,
 has_table_privilege('anon','public.round_robin_hubs','select') anon_read,
 (select bool_and(relrowsecurity) from pg_class where relname in ('round_robin_hubs','round_robin_hub_sessions','round_robin_hub_court_leases','round_robin_hub_player_leases','round_robin_hub_operations')) rls`);
assert.deepEqual(perms.rows[0],{anon_execute:false,authenticated_execute:false,anon_read:false,rls:true});
console.log('Round Robin Hub SQL: atomic score/release, read-set CAS, duplicate replay, stale operators, exclusive courts, player leases, bypass guards, RLS and privileges passed.');
await db.close();
