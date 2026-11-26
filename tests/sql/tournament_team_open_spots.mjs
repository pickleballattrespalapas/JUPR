// Apply the migration to the original team RPCs and exercise them on a disposable database.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import { PGlite } from "@electric-sql/pglite";

const migrations = new URL("../../supabase/migrations/", import.meta.url);
const source = await readFile(new URL("20260728020000_combined_rating_team_tournaments.sql", migrations), "utf8");
const db = new PGlite();

function original(start, end) {
  const first = source.indexOf(start);
  const last = source.indexOf(end, first + start.length);
  assert.ok(first >= 0 && last > first, "The original migration contract must exist");
  return source.slice(first, last);
}

await db.exec(`
  create role anon; create role authenticated; create role service_role;
  create table tournaments(id uuid primary key, club_id text not null);
  create table tournament_event_options(id text primary key, tournament_id text, competition_format text, eligibility_mode text);
  create table tournament_event_draws(id uuid primary key, updated_at timestamptz);
  create table players(id integer primary key, club_id text);
  create table tournament_registrations(id text primary key, tournament_id text, email text, status text, player_id integer, display_name text, first_name text, last_name text, gender text);
  create table tournament_registration_selections(id text primary key, tournament_id text, registration_id text, event_option_id text, partner_mode text, show_on_partner_board boolean);
  create table tournament_team_matchups(id uuid primary key, team_a_id uuid, team_b_id uuid, status text);
`);
await db.exec(original("create table if not exists public.tournament_four_player_teams", "create table if not exists public.tournament_rating_verifications"));
await db.exec(original("create table if not exists public.tournament_team_audit_events", "create or replace function public.normalize_tournament_combined_rating"));
await db.exec(original("create table if not exists public.tournament_team_operations", "create table if not exists public.tournament_team_invitation_deliveries"));
await db.exec(original("create or replace function public.begin_tournament_team_operation", "create or replace function public.reject_tournament_team_audit_mutation"));
await db.exec(original("create or replace function public.admin_create_tournament_four_player_team", "create or replace function public.admin_reissue_tournament_four_player_invite_cas"));
await db.exec(await readFile(new URL("20261126030000_tournament_team_open_roster_spots.sql", migrations), "utf8"));

const tournament = randomUUID();
const slots = ["MAN_1", "MAN_2", "WOMAN_1", "WOMAN_2"];
const members = slots.map((slot, index) => ({
  slot, email: `player${index}@example.invalid`, display_name: `Player ${index}`,
  gender: index < 2 ? "Men" : "Women", ...(index === 0 ? { registration_id: "r0" } : {})
}));
await db.query("insert into tournaments values ($1,'club')", [tournament]);
for (const event of ["team", "full", "pair"]) {
  await db.query("insert into tournament_event_options values ($1,$2,'FOUR_PLAYER_TEAM','STANDARD')", [event, tournament]);
}
for (const [index, member] of members.entries()) {
  await db.query("insert into players values ($1,'club')", [index+1]);
  await db.query("insert into tournament_registrations(id,tournament_id,email,status,player_id,display_name,gender) values ($1,$2,$3,'CONFIRMED',$4,$5,$6)", [`r${index}`, tournament, member.email, index+1, member.display_name, member.gender]);
}
for (const event of ["team", "full", "pair"]) {
  await db.query("insert into tournament_registration_selections values ($1,$2,'r0',$3,'NEEDS_PARTNER',true)", [event, tournament, event]);
}

// An invitee can be an existing solo entrant in this same division.
await db.query("insert into tournament_registration_selections values ('solo1',$1,'r1','team','NEEDS_PARTNER',true)", [tournament]);
async function soloStillLooking() {
  const result = await db.query("select partner_mode,show_on_partner_board from tournament_registration_selections where id='solo1'");
  return result.rows[0];
}

async function create(entrants, event = "team", key = randomUUID()) {
  const result = await db.query("select admin_create_tournament_four_player_team($1,$2,$3,$4,$5,$6,$7,$8,$9,$10) as result", ["club", tournament, event, "Test team", "r0", JSON.stringify(entrants), "a".repeat(64), key, key, members[0].email]);
  return result.rows[0].result;
}
const full = await create(members, "full");
assert.equal(full.members.length, 4);
const pair = await create(members.slice(0,2), "pair");
assert.equal(pair.members.length, 2);
await assert.rejects(create(members.slice(0,1)), /PAYLOAD_INVALID/);
await assert.rejects(create([members[0], members[1], members[1]]), /ROSTER_INVALID/);
await assert.rejects(create([members[0], {...members[1], email:members[0].email}]), /ROSTER_INVALID/);
const creationKey = randomUUID();
const partial = await create(members.slice(0,3), "team", creationKey);
assert.deepEqual(await create(members.slice(0,3), "team", creationKey), partial);
assert.equal(partial.team.status, "FORMING");
assert.equal(partial.members.length, 3);
assert.equal(partial.members.filter(row => row.status === "ACCEPTED").length, 1);
assert.deepEqual((await db.query("select partner_mode,show_on_partner_board from tournament_registration_selections where id='team'")).rows, [{partner_mode:"NONE",show_on_partner_board:false}]);

const fillSignature = "public.server_fill_tournament_four_player_team_spots(text,text,text,text,text,jsonb,text,text,text)";
assert.deepEqual((await db.query("select has_function_privilege('anon',$1,'EXECUTE') as anon, has_function_privilege('authenticated',$1,'EXECUTE') as authenticated, has_function_privilege('service_role',$1,'EXECUTE') as service", [fillSignature])).rows[0], {anon:false,authenticated:false,service:true});
async function fill(entrants, options = {}) {
  const { key = randomUUID(), actor = members[0].email, event = "team", team = partial.team.id, captain = "r0", club = "club" } = options;
  const result = await db.query("select server_fill_tournament_four_player_team_spots($1,$2,$3,$4,$5,$6,$7,$8,$9) as result", [club, tournament, event, team, captain, JSON.stringify(entrants), key, key, actor]);
  return result.rows[0].result;
}
const newMember = members[3];
for (const fields of [{actor:"wrong@example.invalid"}, {captain:"r1"}, {event:"full"}, {club:"other"}]) {
  await assert.rejects(fill([newMember], fields), /CAPTAIN_EVENT_SELECTION_REQUIRED/);
}
await assert.rejects(fill([{...newMember, slot:"MAN_2"}]), /SPOT_ALREADY_FILLED/);
await assert.rejects(fill([{...newMember, email:members[1].email}]), /SPOT_ALREADY_FILLED/);
await assert.rejects(fill([{...newMember, email:"invalid"}]), /ROSTER_INVALID/);
await assert.rejects(fill([newMember, {...members[1], email:"retry@example.invalid"}]), /SPOT_ALREADY_FILLED/);
assert.equal((await db.query("select count(*)::int as n from tournament_four_player_team_members where team_id=$1", [partial.team.id])).rows[0].n, 3);

const fillKey = randomUUID();
const filled = await fill([newMember], {key:fillKey});
assert.deepEqual(filled.members.map(member => member.slot), ["WOMAN_2"]);
assert.equal(filled.team.status, "FORMING");
assert.deepEqual(await fill([newMember], {key:fillKey}), filled);
await assert.rejects(fill([newMember]), /SPOT_ALREADY_FILLED/);

async function respond(member, index, action, key = randomUUID()) {
  // Mail delivery normally writes the token digest. Model only that boundary.
  const digest = "test-token-digest";
  await db.query("update tournament_four_player_team_members set invitation_token_hash=$1 where id=$2 and status='INVITED'", [digest, member.id]);
  const result = await db.query("select server_respond_tournament_four_player_invite($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11) as result", ["club", tournament, partial.team.id, member.id, `r${index}`, member.invitation_version, digest, action, key, key, members[index].email]);
  return result.rows[0].result;
}
await db.query("insert into tournament_registration_selections values ('wrong-division',$1,'r3','full','NONE',false)", [tournament]);
await assert.rejects(respond(filled.members[0],3,"ACCEPT"), /INVITATION_EVENT_SELECTION_REQUIRED/);
await respond(filled.members[0],3,"DECLINE");
for (const index of [2,3]) {
  await db.query("insert into tournament_registration_selections values ($1,$2,$3,'team','NONE',false)", [`teammate${index}`,tournament,`r${index}`]);
}
const replacement = await fill([newMember]);
assert.notEqual(replacement.members[0].id, filled.members[0].id);
assert.deepEqual(await soloStillLooking(), {partner_mode:"NEEDS_PARTNER",show_on_partner_board:true});
for (const index of [1,2]) {
  await respond(partial.members.find(member => member.slot === slots[index]),index,"ACCEPT");
}
assert.deepEqual(await soloStillLooking(), {partner_mode:"NONE",show_on_partner_board:false});
const lastKey = randomUUID();
const final = await respond(replacement.members[0],3,"ACCEPT",lastKey);
assert.deepEqual(await respond(replacement.members[0],3,"ACCEPT",lastKey), final);
assert.equal((await db.query("select status from tournament_four_player_teams where id=$1", [partial.team.id])).rows[0].status, "CONFIRMED");
await assert.rejects(fill([newMember]), /ROSTER_LOCKED/);
await db.close();
console.log("Team signup SQL passed: partial/full creation, solo conversion, routing and permissions, rollback, replay, declined replacement and confirmation after four acceptances.");
