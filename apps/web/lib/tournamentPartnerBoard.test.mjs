import assert from "node:assert/strict";
import { resolve } from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";

const compiledModulePath = process.env.JUPR_TOURNAMENT_PARTNER_BOARD_MODULE;
const moduleUrl = compiledModulePath
  ? pathToFileURL(resolve(compiledModulePath)).href
  : new URL("./tournamentPartnerBoard.ts", import.meta.url).href;

const { groupPartnerEntries } = await import(moduleUrl);

test("groups one player's division listings in schedule order without merging matching names", () => {
  const groups = groupPartnerEntries([
    {
      player_name: "Alex Smith",
      player_entry_key: "player-a",
      board_entry_key: "selection-a-mixed",
      event_day_label: "Saturday",
      event_family: "Mixed Doubles",
      division: "4.0",
      note: "Mixed note"
    },
    {
      player_name: "Alex Smith",
      player_entry_key: "player-b",
      board_entry_key: "selection-b-womens",
      event_day_label: "Thursday",
      event_family: "Women's Doubles",
      division: "4.0",
      note: "Different player"
    },
    {
      player_name: "Alex Smith",
      player_entry_key: "player-a",
      board_entry_key: "selection-a-womens",
      event_day_label: "Thursday",
      event_family: "Women's Doubles",
      division: "4.0",
      note: "Women's note"
    }
  ]);

  assert.equal(groups.length, 2);
  assert.equal(groups[0].playerName, "Alex Smith");
  const firstPlayer = groups.find((group) => group.playerKey === "player-a");
  assert.deepEqual(
    firstPlayer.entries.map((entry) => entry.board_entry_key),
    ["selection-a-mixed", "selection-a-womens"]
  );
  assert.deepEqual(
    firstPlayer.entries.map((entry) => entry.note),
    ["Mixed note", "Women's note"]
  );
  assert.equal(
    groups.find((group) => group.playerKey === "player-b").entries.length,
    1
  );
});

test("does not sort day labels lexicographically", () => {
  const groups = groupPartnerEntries([
    {
      player_name: "Jamie",
      player_entry_key: "player-jamie",
      board_entry_key: "day-2",
      event_day_label: "Day 2"
    },
    {
      player_name: "Jamie",
      player_entry_key: "player-jamie",
      board_entry_key: "day-10",
      event_day_label: "Day 10"
    }
  ]);

  assert.deepEqual(
    groups[0].entries.map((entry) => entry.board_entry_key),
    ["day-2", "day-10"]
  );
});

test("falls back to one group per division listing when player identity is absent", () => {
  const groups = groupPartnerEntries([
    { player_name: "Player", board_entry_key: "selection-a" },
    { player_name: "Player", board_entry_key: "selection-b" },
    { player_name: "Player" }
  ]);

  assert.deepEqual(
    groups.map((group) => group.playerKey),
    ["selection-a", "selection-b", "listing-2"]
  );
});
