// Execute the production SQL on real PostgreSQL/WASM with an isolated schema.
// This checks transaction rollback and guards, not full-schema or concurrency.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { fileURLToPath } from 'node:url';
const { PGlite } = await import(process.env.PGLITE_MODULE || '@electric-sql/pglite');
const root = fileURLToPath(new URL('../../', import.meta.url));
const db = new PGlite();
await db.exec(`
create role anon; create role authenticated; create role service_role; create schema private;
create table tournaments(id uuid primary key,club_id text,status text);
create table tournament_registration_days(id text primary key,tournament_id text,enabled boolean default true);
create table tournament_event_options(id text primary key,tournament_id text,registration_day_id text,event_family_label text,event_type text,gender_restriction text,partner_required boolean,competition_format text default 'STANDARD',eligibility_mode text default 'STANDARD',enabled boolean default true,status text default 'CONFIRMED',scheduled_day_ids jsonb default '[]',updated_at timestamptz default now());
create table tournament_event_draws(id uuid primary key,tournament_id uuid,registration_day_id text,event_option_id text,name text,status text default 'DRAFT',draw_kind text default 'STANDARD',hidden_from_primary_ops boolean default false,pool_config jsonb,updated_at timestamptz default now());
create table players(id integer primary key,club_id text);
create table tournament_registrations(id text primary key,tournament_id text,player_id integer);
create table tournament_registration_team_links(id text primary key,tournament_id uuid,event_option_id text,status text,selection1_id text,selection2_id text,player1_id integer,player2_id integer);
create table tournament_registration_selections(id text primary key,tournament_id text,registration_id text,event_option_id text,registration_day_id text);
create table tournament_teams(id uuid primary key,tournament_id uuid,draw_id uuid,registration_day_id text,event_option_id text,team_number integer,player1_id integer,player2_id integer,seed integer,source text,notes text,source_selection_id text references tournament_registration_selections(id),created_at timestamptz default now(),updated_at timestamptz default now(),unique(tournament_id,draw_id,team_number));
create table tournament_games(id uuid primary key,draw_id uuid,tournament_id uuid);
create table tournament_podium(id uuid primary key,draw_id uuid);
create table tournament_day_live_draws(id uuid primary key,draw_id uuid);
create table tournament_day_live_queue(id uuid primary key,draw_id uuid);
create function private.public_edit_before_division_transfer(text,text,timestamptz,jsonb,jsonb,jsonb) returns jsonb language plpgsql as $body$
declare v_registration tournament_registrations%rowtype;
begin
select * into v_registration from tournament_registrations where id=$2;
return jsonb_build_object('locked',exists(select 1 from tournament_registration_selections selection
join tournament_teams team on team.tournament_id::text=selection.tournament_id
left join tournament_event_draws draw on draw.id=team.draw_id
where selection.registration_id=$2 and v_registration.player_id in(team.player1_id,team.player2_id)
and coalesce(
        draw.event_option_id::text,
        team.event_option_id::text
      ) = selection.event_option_id));
end $body$;
create function private.transfer_tournament_registration_team_division(text,text,text,timestamptz,jsonb) returns jsonb language sql as $body$
select jsonb_build_object('locked',exists(select 1 from tournament_teams t join tournament_event_draws d on d.id=t.draw_id where coalesce(d.event_option_id::text,t.event_option_id::text)=$3)) $body$;
create function server_cancel_tournament_registrations(text,jsonb,text) returns jsonb language sql as $body$
select jsonb_build_object('locked',exists(select 1 from tournament_teams t join tournament_event_draws d on d.id=t.draw_id where coalesce(d.event_option_id::text,t.event_option_id::text)=$3)) $body$;
`);
const guardSource=await readFile(`${root}/supabase/migrations/20260717141224_selection_update_transaction_guards.sql`,'utf8');
const guardStart=guardSource.indexOf('create or replace function private.lock_tournament_registration_selection_scope(');
const guardEnd=guardSource.indexOf('$function$;',guardStart)+'$function$;'.length;
await db.exec(guardSource.slice(guardStart,guardEnd));
const migration=await readFile(`${root}/supabase/migrations/20261109011000_tournament_combined_division_draws.sql`,'utf8');
await db.exec(migration); await db.exec(migration);
const acl=(await db.query(`select has_function_privilege('anon','admin_combine_tournament_draw_divisions_cas(text,text,text,jsonb,jsonb,jsonb,text[],text)','EXECUTE') anon,has_function_privilege('authenticated','admin_combine_tournament_draw_divisions_cas(text,text,text,jsonb,jsonb,jsonb,text[],text)','EXECUTE') authenticated,has_function_privilege('service_role','admin_combine_tournament_draw_divisions_cas(text,text,text,jsonb,jsonb,jsonb,text[],text)','EXECUTE') service`)).rows[0];
assert.deepEqual(acl,{anon:false,authenticated:false,service:true});
const snapshot=async f=>({
  draws:(await db.query('select id,updated_at::text from tournament_event_draws where tournament_id=$1 and combined_into_draw_id is null and not hidden_from_primary_ops order by id',[f.tour])).rows,
  teams:(await db.query('select id,updated_at::text from tournament_teams where tournament_id=$1 order by id',[f.tour])).rows,
  events:(await db.query('select id,updated_at::text from tournament_event_options where tournament_id=$1 order by id',[f.tour])).rows,
});
let nextPlayer=1;
async function fixture(counts=[2,2]) {
  const f={tour:randomUUID(),draws:[randomUUID(),randomUUID()],events:[randomUUID(),randomUUID()],day:randomUUID(),teams:[]};
  await db.query("insert into tournaments values($1,'combine-test','PUBLISHED')",[f.tour]);
  await db.query('insert into tournament_registration_days(id,tournament_id) values($1,$2)',[f.day,f.tour]);
  for(let e=0;e<2;e++) {
    await db.query("insert into tournament_event_options(id,tournament_id,registration_day_id,event_family_label,event_type,gender_restriction,partner_required) values($1,$2,$3,'Mixed doubles','DOUBLES','MIXED',true)",[f.events[e],f.tour,f.day]);
    await db.query("insert into tournament_event_draws(id,tournament_id,registration_day_id,event_option_id,name) values($1,$2,$3,$4,$5)",[f.draws[e],f.tour,f.day,f.events[e],e?'4.5':'4.0']);
    for(let n=0;n<counts[e];n++) {
      const players=[nextPlayer++,nextPlayer++]; const selections=[];
      for(const player of players) {
        const reg=randomUUID(),sel=randomUUID();selections.push(sel);
        await db.query("insert into players values($1,'combine-test')",[player]);
        await db.query('insert into tournament_registrations values($1,$2,$3)',[reg,f.tour,player]);
        await db.query('insert into tournament_registration_selections values($1,$2,$3,$4,$5)',[sel,f.tour,reg,f.events[e],f.day]);
      }
      const id=randomUUID();f.teams.push(id);
      await db.query("insert into tournament_teams(id,tournament_id,draw_id,registration_day_id,event_option_id,team_number,player1_id,player2_id,source,source_selection_id) values($1,$2,$3,$4,$5,$6,$7,$8,'REGISTRATION',$9)",[id,f.tour,f.draws[e],f.day,f.events[e],n+1,...players,selections[0]]);
    }
  }
  return f;
}
async function combine(f,overrides={}) {
  const s=await snapshot(f);
  return db.query('select admin_combine_tournament_draw_divisions_cas($1,$2,$3,$4::jsonb,$5::jsonb,$6::jsonb,$7::text[],$8) result',[
    overrides.club||'combine-test',f.tour,f.draws[0],JSON.stringify(overrides.draws||s.draws),JSON.stringify(overrides.teams||s.teams),JSON.stringify(overrides.events||s.events),overrides.sources||f.events,overrides.name||'Mixed 4.0 / 4.5',
  ]);
}
const rejected=async(fn,marker)=>assert.rejects(fn,error=>error.message.includes(marker));
const f=await fixture();
const registrationsBefore=(await db.query('select * from tournament_registration_selections where tournament_id=$1 order by id',[f.tour])).rows;
await rejected(()=>combine(f,{club:'another-club'}),'DRAW_STALE');
await rejected(()=>combine(f,{teams:[]}),'TEAMS_STALE');
const before=await snapshot(f);
await rejected(()=>combine(f,{events:before.events.map(e=>({...e,updated_at:'2001-01-01'}))}),'EVENTS_STALE');
await rejected(()=>combine(f,{draws:before.draws.slice(1)}),'DRAW_STALE');
const result=(await combine(f)).rows[0].result;
assert.equal(result.moved_team_count,2);
assert.deepEqual(new Set(result.combined_source_draw_ids),new Set([f.draws[1]]));
assert.equal(result.teams.length,4);
assert.deepEqual(new Set(result.teams.map(t=>t.id)),new Set(f.teams));
assert.ok(result.teams.every(t=>t.draw_id===f.draws[0]&&t.event_option_id===f.events[0]&&t.registration_day_id===f.day));
assert.deepEqual(result.teams.map(t=>t.team_number),[1,2,3,4]);
assert.deepEqual(result.teams.map(t=>t.source_event_option_id),[f.events[0],f.events[0],f.events[1],f.events[1]]);
assert.deepEqual((await db.query('select * from tournament_registration_selections where tournament_id=$1 order by id',[f.tour])).rows,registrationsBefore);
const retired=(await db.query('select * from tournament_event_draws where id=$1',[f.draws[1]])).rows[0];
assert.equal(retired.combined_into_draw_id,f.draws[0]);assert.equal(retired.hidden_from_primary_ops,true);assert.equal(retired.status,'CANCELLED');
await rejected(()=>db.query("insert into tournament_event_draws(id,tournament_id,registration_day_id,event_option_id,name) values($1,$2,$3,$4,'new')",[randomUUID(),f.tour,f.day,f.events[1]]),'SOURCE_ALREADY_COVERED');
await rejected(()=>db.query('update tournament_event_draws set source_event_option_ids=$1 where id=$2',[[f.events[0]],f.draws[0]]),'REQUIRES_RPC');
const sourceRegistrations=(await db.query('select r.id from tournament_registrations r join tournament_registration_selections s on s.registration_id=r.id where s.event_option_id=$1',[f.events[1]])).rows;
for(const {id} of sourceRegistrations) assert.equal((await db.query("select private.public_edit_before_division_transfer($1,$2,now(),'[]','{}','[]') result",[f.tour,id])).rows[0].result.locked,true);
assert.equal((await db.query("select server_cancel_tournament_registrations($1,'[]',$2) result",[f.tour,f.events[1]])).rows[0].result.locked,true);
console.log('PASS: combined roster preserves team IDs, original registrations and source divisions; source draws retire; both partners remain edit-locked');

for(const dependency of ['tournament_games','tournament_podium','tournament_day_live_draws','tournament_day_live_queue']) {
  const g=await fixture();const original=await snapshot(g);
  await db.query(`insert into ${dependency}(id,draw_id) values($1,$2)`,[randomUUID(),g.draws[1]]);
  await rejected(()=>combine(g),'DRAW_IN_USE');
  assert.deepEqual(await snapshot(g),original);
}
const pool=await fixture();await db.query("update tournament_event_draws set pool_config='{}' where id=$1",[pool.draws[1]]);await rejected(()=>combine(pool),'DRAW_STALE');
const wrong=await fixture();await db.query("update tournament_event_options set gender_restriction='MEN' where id=$1",[wrong.events[1]]);await rejected(()=>combine(wrong),'EVENTS_INCOMPATIBLE');
const wrongDay=await fixture();await db.query('update tournament_event_options set registration_day_id=$1 where id=$2',[randomUUID(),wrongDay.events[1]]);await rejected(()=>combine(wrongDay),'EVENTS_INCOMPATIBLE');
const disabledDay=await fixture();await db.query('update tournament_registration_days set enabled=false where id=$1',[disabledDay.day]);await rejected(()=>combine(disabledDay),'DAY_UNAVAILABLE');
const scheduled=await fixture();await db.query("update tournament_event_options set scheduled_day_ids='{}' where id=$1",[scheduled.events[1]]);await rejected(()=>combine(scheduled),'EVENTS_INCOMPATIBLE');
const foreignDraw=await fixture();await db.query('update tournament_event_draws set registration_day_id=$1 where id=$2',[randomUUID(),foreignDraw.draws[1]]);await rejected(()=>combine(foreignDraw),'EVENTS_INCOMPATIBLE');
const duplicates=await fixture();await db.query('update tournament_teams set player1_id=(select player1_id from tournament_teams where id=$1) where id=$2',[duplicates.teams[0],duplicates.teams[2]]);await rejected(()=>combine(duplicates),'DUPLICATE_PLAYER');
console.log('PASS: incompatible divisions, duplicate players, pools, games, podium and live dependencies reject atomically');

// Appending a newly confirmed secondary-source team preserves its provenance.
const empty=await fixture([0,0]);await combine(empty);
const player=nextPlayer++;await db.query("insert into players values($1,'combine-test')",[player]);
const registration=randomUUID(),selection=randomUUID();
await db.query('insert into tournament_registrations values($1,$2,$3)',[registration,empty.tour,player]);
await db.query('insert into tournament_registration_selections values($1,$2,$3,$4,$5)',[selection,empty.tour,registration,empty.events[1],empty.day]);
const team={id:randomUUID(),team_number:1,player1_id:player,source:'REGISTRATION',source_selection_id:selection,source_event_option_id:empty.events[1]};
async function writeTeams(draw,rows) {
 const current=(await db.query('select updated_at::text from tournament_event_draws where id=$1',[draw])).rows[0];
 return db.query('select admin_write_tournament_draw_teams_cas($1,$2,$3,$4,false,$5::jsonb) result',['combine-test',empty.tour,draw,current.updated_at,JSON.stringify(rows)]);
}
const imported=(await writeTeams(empty.draws[0],[team])).rows[0].result.teams[0];
assert.equal(imported.source_selection_id,selection);assert.equal(imported.source_event_option_id,empty.events[1]);assert.equal(imported.event_option_id,empty.events[0]);
await rejected(()=>writeTeams(empty.draws[1],[{...team,id:randomUUID()}]),'DRAW_STALE');
await rejected(()=>writeTeams(empty.draws[0],[{...team,id:randomUUID(),source_event_option_id:randomUUID()}]),'SOURCE_SCOPE_INVALID');
await rejected(()=>writeTeams(empty.draws[0],[{...team,id:randomUUID(),team_number:2}]),'DUPLICATE_PLAYER');
console.log('PASS: standard registration writer keeps source provenance and refuses retired, foreign-source and duplicate-player writes');

const pairPlayers=[nextPlayer++,nextPlayer++],pairSelections=[];
for(const player of pairPlayers) {
  const registration=randomUUID(),selection=randomUUID();pairSelections.push(selection);
  await db.query("insert into players values($1,'combine-test')",[player]);
  await db.query('insert into tournament_registrations values($1,$2,$3)',[registration,empty.tour,player]);
  await db.query('insert into tournament_registration_selections values($1,$2,$3,$4,$5)',[selection,empty.tour,registration,empty.events[1],empty.day]);
}
const pair={id:randomUUID(),team_number:2,player1_id:pairPlayers[1],player2_id:pairPlayers[0],source:'REGISTRATION',source_selection_id:pairSelections[0],source_event_option_id:empty.events[1]};
await rejected(()=>writeTeams(empty.draws[0],[pair]),'SOURCE_SCOPE_INVALID');
const link=randomUUID();
await db.query("insert into tournament_registration_team_links values($1,$2,$3,'CONFIRMED',$4,$5,$6,$7)",[link,empty.tour,empty.events[1],...pairSelections,...pairPlayers]);
assert.equal((await writeTeams(empty.draws[0],[pair])).rows[0].result.teams[0].source_selection_id,pairSelections[0]);
await db.query("update tournament_registration_team_links set status='CANCELLED' where id=$1",[link]);
await rejected(()=>writeTeams(empty.draws[0],[{...pair,id:randomUUID(),team_number:3}]),'SOURCE_SCOPE_INVALID');
assert.deepEqual((await db.query('select tournament_draw_source_event_ids($1::jsonb) sources',[JSON.stringify({event_option_id:'primary',source_event_option_ids:['secondary','secondary']})])).rows[0].sources,['primary','secondary']);
console.log('PASS: doubles imports require current confirmed partners and accept reversed player order; source helper always includes primary');
await db.close();
