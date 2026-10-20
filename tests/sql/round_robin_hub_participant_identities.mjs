import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const { PGlite } = await import(process.env.PGLITE_MODULE || '@electric-sql/pglite');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const migrations = path.join(root, 'supabase/migrations');
const identityMigration = fs.readFileSync(path.join(migrations, fs.readdirSync(migrations).find(name => /^\d+_round_robin_hub_participant_identities\.sql$/.test(name))), 'utf8');
const db = new PGlite();
await db.exec(`
  create role anon; create role authenticated; create role service_role bypassrls;
  create table public.live_sessions (
    id uuid primary key default gen_random_uuid(), club_id text not null, session_key text not null,
    state jsonb not null default '{}', title text, status text not null default 'active',
    version bigint not null default 1, created_at timestamptz not null default clock_timestamp(),
    updated_at timestamptz not null default clock_timestamp(), last_seen_at timestamptz,
    expires_at timestamptz, completed_at timestamptz, last_operation_key text,
    last_request_fingerprint text, pending_operation_key text, unique(club_id,session_key)
  );
  grant select,insert,update,delete on public.live_sessions to service_role;
`);
await db.exec(fs.readFileSync(path.join(migrations, '20261020033713_round_robin_hub_atomic_operations.sql'), 'utf8'));
const clone = x => structuredClone(x);
const hubId = '10000000-0000-4000-8000-000000000001';
const hub2 = '10000000-0000-4000-8000-000000000002';
const ref = (session_key, participant_id='p1') => ({session_key,participant_id});
const decision = (left,right,resolution) => ({left:ref(left),right:ref(right),resolution});
const assignment = session_key => ({session_key,match_id:'m1',round_number:1,assigned_at:'2026-10-20T04:00:00Z'});
const court = (id,label) => ({id,label,open:true,current:null,next:null});
const guest = (session,pid='p1') => `guest:${session}:${pid}`;
const initial = {groups:['a','b','c'].map(session_key=>({session_key})),courts:[court('c1','One'),court('c2','Two'),court('c3','Three')],players:{}};
let op=0;
async function row(key) { return (await db.query('select * from live_sessions where session_key=$1',[key])).rows[0]; }
async function hub(id=hubId) { return (await db.query('select * from round_robin_hubs where id=$1',[id])).rows[0]; }
async function addSession(key,names=['Alex','Other'],profiles=[null,null]) {
  const participants=names.map((name,i)=>({id:`p${i+1}`,name,player_id:profiles[i]}));
  const event={generatorKind:'round_robin',currentRoundNumber:1,participants,rounds:[{number:1,status:'active',matches:[{id:'m1',sideA:['p1'],sideB:['p2']}]}]};
  await db.query('insert into live_sessions(club_id,session_key,state) values($1,$2,$3)',['club',key,{mode:'admin_play_generator',page_state:{event}}]);
}
async function readSet(keys,id=hubId,patches={}) {
  return Promise.all(keys.map(async key=>{const r=await row(key);return {session_key:key,expected_version:r.updated_at,expected_numeric_version:Number(r.version),patch:patches[key] || (r.state.round_robin_hub_id ? {} : {state:{...r.state,round_robin_hub_id:id}})};}));
}
async function commit(state,{id=hubId,version,key,fp='0123456789abcdef0123456789abcdef',patches={},sessions}={}) {
  if(version===undefined) version=(await hub(id))?.version || 0;
  const keys=[...new Set([...(await hub(id))?.state.groups?.map(g=>g.session_key) || [],...state.groups.map(g=>g.session_key)])];
  const result=await db.query('select public.commit_round_robin_hub($1,$2,$3,$4,$5,$6,$7,$8,$9,$10) result',
    ['club',id,version,key || `identity-operation-${++op}`,fp,'Identity event','active',state,sessions || await readSet(keys,id,patches),'staff@example.test']);
  return result.rows[0].result;
}
async function version2(state) {
  const next={...clone(state),identity_version:2,identity_decisions:[],participant_identities:{}};
  for(const {session_key:key} of state.groups) {
    next.participant_identities[key]={};
    for(const p of (await row(key)).state.page_state.event.participants) next.participant_identities[key][p.id]=p.player_id===null ? guest(key,p.id) : `player:${p.player_id}`;
  }
  return next;
}
await db.exec('set role service_role');
await addSession('a',['Alex','A second']); await addSession('b',[' Alex ','B second']); await addSession('c',['Chris','C second']);
// An occupied pre-upgrade hub must remain usable before and after new API rollout.
let legacy=clone(initial);legacy.courts[0].current=assignment('a');
await commit(legacy);
await db.exec('reset role'); await db.exec(identityMigration); await db.exec(identityMigration); await db.exec('set role service_role');
for (const [name, expected] of [['\u00a0JOSÉ\u00a0','josé'],['\u001cAlex\u3000 Smith\u001c','alex smith'],['ΟΣ','ΟΣ'],['İ','İ'],['ẞ','ẞ']]) {
  assert.equal((await db.query('select public.round_robin_hub_name_key($1) value',[name])).rows[0].value,expected);
}
// Pending review follows the canonical person through a differently named alias.
const aliasEvents={x:{participants:[{id:'p1',name:'Alex'}]},y:{participants:[{id:'p1',name:'Other label'}]},z:{participants:[{id:'p1',name:'Alex'}]}};
const aliasState={identity_version:2,participant_identities:{x:{p1:guest('x')},y:{p1:guest('x')},z:{p1:guest('z')}},identity_decisions:[decision('x','y','same')]};
const aliases=(await db.query('select public.round_robin_hub_identity_snapshot($1,$2) value',[aliasState,aliasEvents])).rows[0].value;
assert.deepEqual(aliases.pending.sort(),[guest('x'),guest('y'),guest('z')]);
await commit(legacy);
let state=await version2(legacy);
await commit(state);
assert.equal((await hub()).state.identity_version,2);
assert.equal((await db.query("select player_key from round_robin_hub_player_leases where session_key='a' order by player_key")).rows[0].player_key,guest('a'));
await assert.rejects(()=>commit(legacy),/cannot be downgraded/);
// Unresolved names block only affected new assignments, including reservations.
for(const slot of ['current','next']) {
  const bad=clone(state);bad.courts[1][slot]=assignment('b');
  await assert.rejects(()=>commit(bad),/identity review/);
}
let unrelated=clone(state);unrelated.courts[2].current=assignment('c');
await commit(unrelated);state=unrelated;
// A fake materialized identity cannot bypass decisions or profile authority.
const forged=clone(state);forged.participant_identities.b.p1='player:999';
await assert.rejects(()=>commit(forged),/do not match/);
// Explicit different people can share a display name within the same hub.
state.identity_decisions=[decision('a','b','different')];state.courts[1].next=assignment('b');
const read=await readSet(['a','b','c']); const expected=(await hub()).version;
await commit(state,{version:expected,key:'resolve-different-once',sessions:read});
assert.equal((await commit(state,{version:expected,key:'resolve-different-once',sessions:[]})).replayed,true);
await assert.rejects(()=>commit(state,{version:expected,key:'resolve-different-once',fp:'fedcba9876543210fedcba9876543210'}),/different hub command/);
await assert.rejects(()=>commit(state,{version:expected,sessions:read}),/Hub changed/);
assert.equal((await db.query('select count(*) n from round_robin_hub_player_leases')).rows[0].n,6);
// Same-person merge cannot create two occupied/reserved claims and rolls back.
const merged=clone(state);merged.identity_decisions=[decision('a','b','same')];merged.participant_identities.b.p1=guest('a');
const before=await hub();
await assert.rejects(()=>commit(merged),/duplicate key|assigned|reserved|Release/);
assert.deepEqual(await hub(),before);
// Existing scores can release a legacy-upgraded current court while reviews remain.
let released=clone(state);released.courts[0].current=null;released.courts[1].next=null;released.identity_decisions=[];
const a=await row('a'); const scored=clone(a.state);Object.assign(scored.page_state.event.rounds[0].matches[0],{scoreA:11,scoreB:8});
await commit(released,{patches:{a:{state:scored}}});
assert.equal((await row('a')).state.page_state.event.rounds[0].status,'active');
state=released;
// A guest linked by an explicit decision takes the profile lease, never player_id.
await addSession('d',['Alex','D second'],[42,43]);
state.groups.push({session_key:'d'});state.participant_identities.d={p1:'player:42',p2:'player:43'};
state.identity_decisions=[decision('a','b','different'),decision('b','d','same'),decision('a','d','different')];
state.participant_identities.b.p1='player:42';state.courts[1].current=assignment('b');
await commit(state);
assert.equal((await row('b')).state.page_state.event.participants[0].player_id,null);
assert.equal((await db.query("select player_key from round_robin_hub_player_leases where session_key='b' and profile_id='42'")).rows[0].player_key,'player:42');
// Detaching a source profile cannot silently remove a guest's established identity.
const detached=clone(state);detached.groups=detached.groups.filter(g=>g.session_key!=='d');
delete detached.participant_identities.d;
detached.identity_decisions=detached.identity_decisions.filter(d=>d.left.session_key!=='d' && d.right.session_key!=='d');
detached.participant_identities.b.p1=guest('b');
await assert.rejects(()=>commit(detached),/linked player identities/);
const profileBypass=clone(state);profileBypass.participant_identities.d.p1=guest('d');
await assert.rejects(()=>commit(profileBypass),/do not match/);
const profileConflict=clone(state);profileConflict.identity_decisions.push({left:ref('d','p1'),right:ref('d','p2'),resolution:'same'});
await assert.rejects(()=>commit(profileConflict),/authoritative club profile/);
// Legacy workers preserve their name guard; v2 guest decisions are event-scoped.
await addSession('e',['Alex','E second']);
const external={groups:[{session_key:'e'}],courts:[{...court('x','Elsewhere'),current:assignment('e')}],players:{}};
await assert.rejects(()=>commit(external,{id:hub2}),/same club profile/);
const externalV2=await version2(external);
await commit(externalV2,{id:hub2});
assert.equal((await hub(hub2)).state.courts[0].current.session_key,'e');
// The same authoritative profile still has one club-wide occupied/reserved lease.
await addSession('external-profile',['A renamed profile','Another'],[42,90]);
const profileHubId='10000000-0000-4000-8000-000000000003';
const externalProfile=await version2({groups:[{session_key:'external-profile'}],courts:[{...court('p','Profile court'),next:assignment('external-profile')}],players:{}});
await assert.rejects(()=>commit(externalProfile,{id:profileHubId}),/duplicate key/);
assert.equal(await hub(profileHubId),undefined);
// Known profiles cannot be split using different decisions.
await addSession('f',['Alex renamed','F second'],[42,44]);
const sameProfile=clone(state);sameProfile.groups.push({session_key:'f'});sameProfile.participant_identities.f={p1:'player:42',p2:'player:44'};
sameProfile.identity_decisions.push(decision('d','f','different'));
await assert.rejects(()=>commit(sameProfile),/authoritative club profile/);
// A full-component correction can safely split distinct people while one plays.
await addSession('g',['Pat','G second']); await addSession('h',['Pat','H second']); await addSession('i',['Pat','I second']);
const correctionHub='10000000-0000-4000-8000-000000000004';
const correction=await version2({groups:[{session_key:'g'},{session_key:'h'},{session_key:'i'}],courts:[court('g','Correction 1'),court('h','Correction 2')],players:{}});
correction.identity_decisions=[decision('g','h','same'),decision('g','i','same')];correction.participant_identities.h.p1=guest('g');correction.participant_identities.i.p1=guest('g');
await commit(correction,{id:correctionHub});
correction.courts[0].current=assignment('g');await commit(correction,{id:correctionHub});
// Reconfirming an already unified component is a harmless no-op, even occupied.
correction.identity_decisions.push(decision('h','i','same'));await commit(correction,{id:correctionHub});
correction.identity_decisions=[decision('g','h','different'),decision('g','i','different'),decision('h','i','different')];correction.participant_identities.h.p1=guest('h');correction.participant_identities.i.p1=guest('i');
correction.courts[1].next=assignment('h');await commit(correction,{id:correctionHub});
assert.equal((await db.query('select count(*) n from round_robin_hub_player_leases where hub_id=$1',[correctionHub])).rows[0].n,4);
// The helper and transaction remain private invoker functions, with existing RLS.
await db.exec('reset role');
const permissions=(await db.query(`select
 has_function_privilege('anon','public.round_robin_hub_identity_snapshot(jsonb,jsonb)','execute') anon,
 has_function_privilege('authenticated','public.round_robin_hub_identity_snapshot(jsonb,jsonb)','execute') authenticated,
 has_function_privilege('anon','public.round_robin_hub_name_key(text)','execute') anon_name_key,
 (select bool_or(prosecdef) from pg_proc where proname in ('round_robin_hub_identity_snapshot','commit_round_robin_hub')) definer,
 (select bool_and(relrowsecurity) from pg_class where relname like 'round_robin_hub%' and relkind='r') rls`)).rows[0];
assert.deepEqual(permissions,{anon:false,authenticated:false,anon_name_key:false,definer:false,rls:true});
console.log('Round Robin Hub identities: scoped guests, authoritative profiles, decisions, partial blocking, current/reserved protection, old-worker compatibility, CAS/replay, rollback, cross-hub leases and private permissions passed.');
await db.close();
