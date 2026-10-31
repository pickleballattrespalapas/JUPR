// Actual PostgreSQL/WASM trigger behavior on an isolated minimal schema.
// This verifies mutation rollback and cleanup paths, not concurrent clients.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import { fileURLToPath } from "node:url";

const { PGlite } = await import(process.env.PGLITE_MODULE || "@electric-sql/pglite");
const root = fileURLToPath(new URL("../../", import.meta.url));
const db = new PGlite();
await db.exec(`
  create role anon; create role authenticated; create role service_role;
  create schema private;
  create table public.tournament_event_draws (
    id uuid primary key, tournament_id uuid, registration_day_id text,
    event_option_id text, source_event_option_ids text[] not null default '{}'
  );
  create table public.tournament_event_options (
    id text primary key, tournament_id text, registration_day_id text,
    scheduled_day_ids jsonb, enabled boolean, status text, event_family_key text,
    event_family_label text, event_type text, gender_restriction text,
    partner_required boolean, competition_format text, eligibility_mode text, skill_mode text
  );
  create table public.tournament_day_live_draws (
    id uuid primary key, draw_id uuid, registration_day_id text, state text,
    run_id uuid, version integer default 1, last_assigned_at timestamptz,
    last_operation_key text, updated_by text, updated_at timestamptz
  );
  create table public.tournament_day_live_queue (
    id uuid primary key, draw_id uuid, registration_day_id text, state text,
    court_id uuid, reserved_court_id uuid, released_at timestamptz,
    run_id uuid, day_draw_id uuid, tournament_id uuid, game_id uuid,
    team_a_id uuid, team_b_id uuid, version integer default 1,
    blocker_code text, blocker_detail text, eligible_since timestamptz,
    reserved_at timestamptz, held_at timestamptz, called_at timestamptz,
    started_at timestamptz, completed_at timestamptz,
    last_operation_key text, updated_by text, updated_at timestamptz
  );
  create table public.tournament_day_live_runs (
    id uuid primary key, club_id text, state text, version integer default 1,
    queue_version integer default 1, last_operation_key text,
    updated_by text, updated_at timestamptz
  );
  create table public.tournament_day_live_courts (
    id uuid primary key, run_id uuid, state text, version integer default 1,
    last_operation_key text, updated_by text, updated_at timestamptz
  );
  create table public.tournament_games (
    id uuid primary key, tournament_id uuid, draw_id uuid,
    registration_day_id text, team_a_id uuid, team_b_id uuid, finalized_at timestamptz
  );
  create table public.tournament_day_live_participant_claims (
    id uuid primary key, queue_id uuid, game_id uuid, player_id integer, state text,
    released_at timestamptz, released_by text, version integer default 1,
    last_operation_key text, claimed_at timestamptz, updated_at timestamptz
  );
  create function public.tournament_day_live_game_player_ids(text,uuid,uuid)
    returns integer[] language sql as 'select array[1,2]';
  create function public.tournament_day_live_players_ready(text,text,text,integer[])
    returns boolean language sql as 'select true';
`);
const combine = await readFile(`${root}/supabase/migrations/20261109011000_tournament_combined_division_draws.sql`, "utf8");
for (const routine of [
  "public.tournament_draw_source_event_ids",
  "private.tournament_combined_draw_family",
]) {
  const start = combine.indexOf(`create or replace function ${routine}(`);
  assert.ok(start >= 0, `${routine} exists in the preceding migration`);
  const end = combine.indexOf("$function$;", start) + "$function$;".length;
  await db.exec(combine.slice(start, end));
}
const reservations = await readFile(`${root}/supabase/migrations/20261108021000_tournament_day_court_reservations.sql`, "utf8");
const promotionStart = reservations.indexOf("create or replace function public.promote_tournament_day_court_reservation()");
const promotionEnd = reservations.indexOf("$function$;", promotionStart) + "$function$;".length;
await db.exec(reservations.slice(promotionStart,promotionEnd));
await db.exec(`create trigger trg_tournament_day_live_promote_reserved_court
  after update of state,court_id,released_at on public.tournament_day_live_queue
  for each row execute function public.promote_tournament_day_court_reservation()`);
const guards = await readFile(`${root}/supabase/migrations/20261109011100_combined_division_live_guards.sql`, "utf8");
await db.exec(guards);
await db.exec(guards); // The guarded promotion patch is safe to replay.

async function fixture() {
  const draw = randomUUID();
  const tournament = randomUUID();
  const source = [randomUUID(), randomUUID()];
  await db.query(`insert into public.tournament_event_options
    (id,tournament_id,registration_day_id,scheduled_day_ids,enabled,status,event_family_key,event_type,gender_restriction,partner_required,competition_format)
    values ($1,$3,'day-1','["day-1"]',true,'ACTIVE','mixed','DOUBLES','MIXED',true,'STANDARD'),
           ($2,$3,'day-1','["day-1"]',true,'ACTIVE','mixed','DOUBLES','MIXED',true,'STANDARD')`, [...source, tournament]);
  await db.query("insert into public.tournament_event_draws values ($1,$2,'day-1',$3,$4::text[])", [draw,tournament,source[0],source]);
  return { draw, tournament, source };
}
async function reject(action) {
  await assert.rejects(action, error => error.message.includes("JUPR_TOURNAMENT_COMBINED_SOURCE_INVALID"));
}
async function queue(f, state = "WAITING") {
  const id = randomUUID();
  await db.query(`insert into public.tournament_day_live_queue
    (id,draw_id,registration_day_id,state,court_id,reserved_court_id)
    values ($1,$2,'day-1',$3,$4,$5)`, [id,f.draw,state,state === "CALLED" ? randomUUID() : null,state === "RESERVED" ? randomUUID() : null]);
  return id;
}

// Every source is checked at activation and at the actual assignment write.
for (const change of ["disabled", "rescheduled", "family", "foreign", "missing", "eligibility_cap", "skill_cap"]) {
  const f = await fixture();
  const dayDraw = randomUUID();
  await db.query("insert into public.tournament_day_live_draws (id,draw_id,registration_day_id,state) values ($1,$2,'day-1','ACTIVE')", [dayDraw,f.draw]);
  const waiting = await queue(f);
  const called = await queue(f,"CALLED");
  const reserved = await queue(f,"RESERVED");
  if (change === "disabled") await db.query("update public.tournament_event_options set enabled=false where id=$1", [f.source[1]]);
  if (change === "rescheduled") await db.query("update public.tournament_event_options set scheduled_day_ids='[\"day-2\"]' where id=$1", [f.source[1]]);
  if (change === "family") await db.query("update public.tournament_event_options set event_family_key='mens' where id=$1", [f.source[1]]);
  if (change === "foreign") await db.query("update public.tournament_event_options set tournament_id=$2 where id=$1", [f.source[1],randomUUID()]);
  if (change === "missing") await db.query("delete from public.tournament_event_options where id=$1", [f.source[1]]);
  if (change === "eligibility_cap") await db.query("update public.tournament_event_options set eligibility_mode='COMBINED_RATING_CAP' where id=$1", [f.source[1]]);
  if (change === "skill_cap") await db.query("update public.tournament_event_options set skill_mode='COMBINED_RATING_CAP' where id=$1", [f.source[1]]);

  await db.query("update public.tournament_day_live_draws set state='PAUSED' where id=$1", [dayDraw]);
  await reject(() => db.query("update public.tournament_day_live_draws set state='ACTIVE' where id=$1", [dayDraw]));
  await reject(() => db.query("insert into public.tournament_day_live_draws (id,draw_id,registration_day_id,state) values ($1,$2,'day-1','ACTIVE')", [randomUUID(),f.draw]));
  await reject(() => db.query("update public.tournament_day_live_queue set state='CALLED',court_id=$2 where id=$1", [waiting,randomUUID()]));
  await reject(() => db.query("update public.tournament_day_live_queue set state='RESERVED',reserved_court_id=$2 where id=$1", [waiting,randomUUID()]));
  assert.equal((await db.query("select state from public.tournament_day_live_queue where id=$1",[waiting])).rows[0].state,"WAITING");

  // Finishing an existing game and cancelling a reservation remain possible.
  await db.query("update public.tournament_day_live_queue set state='COMPLETED',court_id=null,released_at=now() where id=$1", [called]);
  await db.query("update public.tournament_day_live_queue set state='WAITING',reserved_court_id=null where id=$1", [reserved]);
}

const wrongDay = await fixture();
await reject(() => db.query("insert into public.tournament_day_live_draws (id,draw_id,registration_day_id,state) values ($1,$2,'day-2','ACTIVE')", [randomUUID(),wrongDay.draw]));
const legacy = await fixture();
await db.query("update public.tournament_event_draws set source_event_option_ids='{}' where id=$1", [legacy.draw]);
await db.query("update public.tournament_event_options set enabled=false where id=$1", [legacy.source[1]]);
await db.query("insert into public.tournament_day_live_draws (id,draw_id,registration_day_id,state) values ($1,$2,'day-1','ACTIVE')", [randomUUID(),legacy.draw]);
await queue(legacy,"CALLED");

// A scored court release invokes the real reservation-promotion trigger. A
// now-invalid combined source must unwind the reservation without undoing that
// completed game or retaining the reserved players' claims.
const promotion = await fixture();
const run = randomUUID(), court = randomUUID(), dayDraw = randomUUID();
const reservedGame = randomUUID(), teamA = randomUUID(), teamB = randomUUID();
const current = await queue(promotion,"CALLED"), next = await queue(promotion,"RESERVED");
await db.query("insert into public.tournament_day_live_runs (id,club_id,state) values ($1,'club','ACTIVE')",[run]);
await db.query("insert into public.tournament_day_live_courts (id,run_id,state) values ($1,$2,'OPEN')",[court,run]);
await db.query("insert into public.tournament_day_live_draws (id,draw_id,registration_day_id,state,run_id) values ($1,$2,'day-1','ACTIVE',$3)",[dayDraw,promotion.draw,run]);
await db.query("insert into public.tournament_games values ($1,$2,$3,'day-1',$4,$5,null)",[reservedGame,promotion.tournament,promotion.draw,teamA,teamB]);
await db.query("update public.tournament_day_live_queue set run_id=$2,court_id=$3,updated_by='director',last_operation_key='score' where id=$1",[current,run,court]);
await db.query(`update public.tournament_day_live_queue set run_id=$2,reserved_court_id=$3,
  day_draw_id=$4,tournament_id=$5,game_id=$6,team_a_id=$7,team_b_id=$8,
  updated_by='director',last_operation_key='reserve' where id=$1`,[next,run,court,dayDraw,promotion.tournament,reservedGame,teamA,teamB]);
for (const player of [1,2]) await db.query("insert into public.tournament_day_live_participant_claims (id,queue_id,game_id,player_id,state) values ($1,$2,$3,$4,'RESERVED')",[randomUUID(),next,reservedGame,player]);
await db.query("update public.tournament_event_options set eligibility_mode='COMBINED_RATING_CAP' where id=$1",[promotion.source[1]]);
await db.query("update public.tournament_day_live_queue set state='COMPLETED',court_id=null,released_at=now() where id=$1",[current]);
assert.equal((await db.query("select state from public.tournament_day_live_queue where id=$1",[current])).rows[0].state,"COMPLETED");
assert.deepEqual((await db.query("select state,reserved_court_id from public.tournament_day_live_queue where id=$1",[next])).rows[0],{state:"WAITING",reserved_court_id:null});
assert.equal((await db.query("select count(*)::int as total from public.tournament_day_live_participant_claims where queue_id=$1 and released_at is null",[next])).rows[0].total,0);

const permissions = (await db.query(`select
  has_function_privilege('anon','private.assert_tournament_combined_draw_sources(uuid,text)','execute') as anon,
  has_function_privilege('authenticated','private.assert_tournament_combined_draw_sources(uuid,text)','execute') as authenticated,
  has_function_privilege('service_role','private.assert_tournament_combined_draw_sources(uuid,text)','execute') as service_role`)).rows[0];
assert.deepEqual(permissions,{anon:false,authenticated:false,service_role:true});
await db.close();
console.log("Combined division live guards passed: activation, resume, assignment, reservation, source drift, cleanup, single-division compatibility, privileges.");
