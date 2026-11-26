import assert from 'node:assert/strict';
import fs from 'node:fs';
const { PGlite } = await import(process.env.PGLITE_MODULE || '@electric-sql/pglite');
const db = new PGlite();
await db.exec(`create role anon; create role authenticated; create role service_role bypassrls;
 create table clubs(id text primary key);
 insert into clubs values('tres_palapas'),('other');
 create table admin_role_assignments(club_id text,email text,user_id uuid,role text,revoked_at timestamptz,expires_at timestamptz);
 insert into admin_role_assignments values('tres_palapas','joe@example.invalid','10000000-0000-4000-8000-000000000001','administrator',null,null);
 create table pcs_interclub_seasons(id uuid primary key,organizer_club_id text);
 grant select on admin_role_assignments,clubs,pcs_interclub_seasons to service_role;
`);
await db.exec(fs.readFileSync(new URL('../../supabase/migrations/20261114223140_interclub_organizer_managed.sql',import.meta.url),'utf8'));
await db.exec(fs.readFileSync(new URL('../../supabase/migrations/20261126120000_managed_interclub_club_entries.sql',import.meta.url),'utf8'));
const actor='10000000-0000-4000-8000-000000000001', sid='20000000-0000-4000-8000-000000000001',mid='30000000-0000-4000-8000-000000000001';
const details={name:'Southern BCS',clubs:[{id:'label-tres',name:'Tres Palapas'},{id:'label-away',name:'Away club'}],divisions:['3.5']};
const plan={host_club_id:'label-away',club_ids:['label-tres','label-away'],starts_at:'2099-01-16T17:00:00Z',courts:2,teams:[]};
const doc={schema_version:1,meet_id:mid,phase:'regular',format:'gender',encounters:[{id:'match',pairings:[{id:'pair',games:[{id:'g',status:'pending',a:null,b:null}]}]}]};
const read=async()=> (await db.query('select * from pcs_managed_interclub_meets where id=$1',[mid])).rows[0];
const season=async()=> (await db.query('select * from pcs_managed_interclub_seasons where id=$1',[sid])).rows[0];
async function write(action,revision,payload={},opts={}) {
 const result=await db.query('select pcs_managed_interclub_write($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9) as result',
 [opts.token?null:actor,opts.token?null:'joe@example.invalid',opts.club||'tres_palapas',opts.sid||sid,opts.mid===null?null:(opts.mid||mid),action,revision,JSON.stringify(payload),opts.token||null]);
 return result.rows[0].result;
}
await assert.rejects(write('season',0,{details},{club:'other',mid:null}),/administrator/);
await db.exec('set role service_role');
await write('season',0,{details},{mid:null});
assert.equal((await season()).revision,1);
await assert.rejects(write('season',0,{details},{mid:null}),/changed/);
await write('plan',0,{details:plan,season_revision:1});
await write('generate',1,{document:doc});
await assert.rejects(write('share',2,{}),/Invalid host access/);
const digest='a'.repeat(64),replacement='b'.repeat(64);
await write('share',2,{token_hash:digest,expires_at:'2099-04-01T00:00:00Z'});
assert.equal((await read()).revision,3);
await assert.rejects(write('approve',3,{}, {token:digest}),/scores and submission only/);
await assert.rejects(write('scores',3,{document:doc},{token:replacement}),/link unavailable/);
await assert.rejects(write('scores',3,{document:doc},{token:digest,mid:'30000000-0000-4000-8000-000000000099'}),/link unavailable/);
await write('scores',3,{document:doc},{token:digest});
assert.equal((await read()).approved_document,null);
await assert.rejects(write('scores',3,{document:doc},{token:digest}),/changed/);
await write('share',4,{token_hash:replacement,expires_at:'2099-04-01T00:00:00Z'});
await assert.rejects(write('scores',5,{document:doc},{token:digest}),/link unavailable/);
await write('submit',5,{}, {token:replacement});
await assert.rejects(write('scores',6,{document:doc},{token:replacement}),/not open/);
await write('approve',6);
assert.deepEqual((await read()).approved_document,doc);
await assert.rejects(write('reopen',7,{reason:'x'}),/reason/);
await write('reopen',7,{reason:'Correct court 1 score'});
const correction=structuredClone(doc);correction.encounters[0].pairings[0].games[0].a=5;
await write('scores',8,{document:correction},{token:replacement});
assert.deepEqual((await read()).approved_document,doc,'Previously approved revision stays official during correction');
await assert.rejects(write('plan',9,{details:plan,season_revision:(await season()).revision}),/Scored meets/);
await write('revoke',9);
await assert.rejects(write('scores',10,{document:doc},{token:replacement}),/link unavailable/);
assert.equal((await db.query('select count(*)::int as n from clubs')).rows[0].n,2,'No participating club accounts created');
const audit=(await db.query('select * from pcs_managed_interclub_audit')).rows;
assert.ok(audit.some(row=>row.actor_kind==='host' && row.actor_id===null));
assert.ok(!JSON.stringify(audit).includes(digest) && !JSON.stringify(audit).includes(replacement));
await db.exec('reset role');
const privileges=(await db.query("select has_table_privilege('anon','pcs_managed_interclub_meets','select') as can_read,has_function_privilege('authenticated','pcs_managed_interclub_write(uuid,text,text,uuid,uuid,text,integer,jsonb,text)','execute') as can_write")).rows[0];
assert.equal(privileges.can_read,false);assert.equal(privileges.can_write,false);
const rls=(await db.query("select relrowsecurity from pg_class where relname in ('pcs_managed_interclub_seasons','pcs_managed_interclub_meets','pcs_managed_interclub_audit')")).rows;
assert.ok(rls.every(row=>row.relrowsecurity));
const importedSid='20000000-0000-4000-8000-000000000002', importedMid='30000000-0000-4000-8000-000000000002';
await db.query("insert into pcs_interclub_seasons values($1,'tres_palapas')",[importedSid]);
await db.exec('set role service_role');
const snapshot={details,meets:[{id:importedMid,details:plan,state:'approved',document:doc,approved_document:doc}]};
await write('import',0,snapshot,{sid:importedSid,mid:null});
await write('import',0,{details:{name:'Must not overwrite'}},{sid:importedSid,mid:null});
assert.deepEqual((await db.query('select details from pcs_managed_interclub_seasons where id=$1',[importedSid])).rows[0].details,details);
assert.deepEqual((await db.query('select approved_document from pcs_managed_interclub_meets where id=$1',[importedMid])).rows[0].approved_document,doc);
assert.equal((await db.query('select count(*)::int as n from pcs_interclub_seasons where id=$1',[importedSid])).rows[0].n,1);
// A captain can register and offer dates before anyone has created a meet.
const signupSeason = '20000000-0000-4000-8000-000000000003';
const signupDetails = {...details, start_date:'2027-01-01', end_date:'2027-03-31'};
await write('season', 0, {details:signupDetails}, {sid:signupSeason, mid:null});
const linkSQL = 'select pcs_managed_interclub_club_link($1,$2,$3,$4,$5,$6)';
const captainSQL = 'select pcs_managed_interclub_club_entry($1,$2,$3::jsonb) as entry';
const first = 'c'.repeat(64), second = 'd'.repeat(64), other = 'e'.repeat(64);
const issue = (token, club='label-away', organizer='tres_palapas') =>
 db.query(linkSQL, [actor,'joe@example.invalid',organizer,signupSeason,club,token]);
const entry = async (token, revision=null, data=null) =>
 (await db.query(captainSQL, [token, revision, data === null ? null : JSON.stringify(data)])).rows[0].entry;
const registration = {
 captain:'Captain', contact_method:'email', contact:'private-captain@example.invalid', divisions:['3.5'],
 participation:'host_travel', travel_notes:'Contact us first',
 host_offers:[{date:'2027-02-05', start_time:null, end_time:null, courts:2}]
};
await assert.rejects(issue(first, 'unknown'), /Club unavailable/);
await assert.rejects(issue(first, 'label-away', 'other'), /Organizer access required/);
await issue(first);
await issue(other, 'label-tres');
assert.deepEqual(await entry(first), {club_name:'Away club', season_name:'Southern BCS', divisions:['3.5'],
 start_date:'2027-01-01', end_date:'2027-03-31', revision:0, registration:null});
await assert.rejects(entry(first, 0, {...registration, divisions:['4.5']}), /offered divisions/);
await assert.rejects(entry(first, 0, {...registration, host_offers:[{date:'2027-04-01'}]}), /outside season/);
assert.equal((await entry(first, 0, registration)).revision, 1);
assert.deepEqual((await entry(first)).registration, registration);
assert.equal((await entry(other)).registration, null);
await assert.rejects(entry(first, 0, {...registration, captain:'Stale'}), /entry changed/);
await issue(second);
await assert.rejects(entry(first), /link unavailable/);
await assert.rejects(entry(first, 1, registration), /link unavailable/);
assert.deepEqual((await entry(second)).registration, registration);
const revised = {...registration, participation:'travel', host_offers:[]};
assert.equal((await entry(second, 1, revised)).revision, 2);
assert.equal((await db.query('select count(*)::int as n from pcs_managed_interclub_meets where season_id=$1', [signupSeason])).rows[0].n, 0);
const signupAudit = (await db.query('select * from pcs_managed_interclub_audit where season_id=$1', [signupSeason])).rows;
assert.equal(signupAudit.filter(r=>r.actor_kind==='captain').length, 2);
assert.ok(!JSON.stringify(signupAudit).includes(registration.contact) && !JSON.stringify(signupAudit).includes(second));
await write('season', 1, {details:{...signupDetails, clubs:[signupDetails.clubs[0]]}}, {sid:signupSeason, mid:null});
await write('season', 2, {details:signupDetails}, {sid:signupSeason, mid:null});
await assert.rejects(entry(second), /link unavailable/);
await issue('f'.repeat(64));
assert.equal((await entry('f'.repeat(64))).registration, null);
for (const role of ['anon','authenticated']) {
 const access = (await db.query(`select
  has_table_privilege($1, 'pcs_managed_interclub_club_entries', 'select') as read,
  has_function_privilege($1, 'pcs_managed_interclub_club_entry(text,integer,jsonb)', 'execute') as register,
  has_function_privilege($1, 'pcs_managed_interclub_club_link(uuid,text,text,uuid,text,text)', 'execute') as invite`,
 [role])).rows[0];
 assert.deepEqual(access, {read:false, register:false, invite:false});
}
await db.close();
console.log('Managed interclub SQL: ownership, host scope/revocation, revisions, approval, audit and RLS passed.');
