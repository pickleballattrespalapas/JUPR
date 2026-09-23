const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const ts = require("typescript");

const root = path.resolve(__dirname, "..");

function loadTypeScript(relativePath, requireOverrides = {}) {
  const filename = path.join(root, relativePath);
  const output = ts.transpileModule(fs.readFileSync(filename, "utf8"), {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2022,
      esModuleInterop: true
    },
    fileName: filename
  }).outputText;
  const loaded = { exports: {} };
  const localRequire = (request) =>
    Object.prototype.hasOwnProperty.call(requireOverrides, request)
      ? requireOverrides[request]
      : require(request);
  new Function("exports", "require", "module", "__filename", "__dirname", output)(
    loaded.exports,
    localRequire,
    loaded,
    filename,
    path.dirname(filename)
  );
  return loaded.exports;
}

const eligibility = loadTypeScript("lib/tournamentRegistrationEligibility.ts", {
  "@/lib/tournamentSkillEligibility": {
    skillEligibilityPolicy: () => ({
      mode: "STANDARD",
      minimum: null,
      maximumExclusive: 4.0,
      combinedCap: null
    })
  }
});

const singlesEvent = {
  event_type: "SINGLES",
  partner_required: false,
  gender_restriction: "ANY"
};
const doublesEvent = {
  event_type: "DOUBLES",
  partner_required: true,
  gender_restriction: "ANY"
};

for (const [restriction, player, partner, warning] of [
  ["MIXED", "Women", "Women", /one man and one woman/],
  ["MIXED", "Male", "Men", /one man and one woman/],
  ["MEN", "Women", "Men", /listed for men/],
  ["WOMEN", "Women", "Male", /listed for women/],
  ["MIXED", "Other", "Women", /one man and one woman/],
  ["MIXED", "Women", "Non-binary", null],
  ["MIXED", "Non-binary", "Women", null],
  ["MIXED", "Women", "Men", null],
  ["ANY", "Women", "Women", null],
  ["MIXED", "Women", "", null]
]) {
  const event = { ...doublesEvent, gender_restriction: restriction };
  assert.equal(eligibility.publicEventEligibilityReason(event, { gender: player, doublesSkill: 3 }), null, "Gender exceptions must not hide or block the event");
  const notice = eligibility.publicEventGenderNotice(event, { gender: player }, partner);
  if (warning) {
    assert.match(notice, warning);
    assert.match(notice, /can still submit/);
    assert.doesNotMatch(notice, /admin|approval/i);
  } else assert.equal(notice, null);
}

for (const [stored, expected] of [
  ["Female", "Women"], [" f ", "Women"], ["WOMAN", "Women"],
  ["Male", "Men"], ["m", "Men"], ["MEN", "Men"],
  ["nonbinary", "Non-binary"], ["OTHER", "Other"],
  ["prefer not to say", "Prefer not to say"], [null, ""],
  ["Self-described", "Self-described"]
]) {
  const value = eligibility.normalizeRegistrationGender(stored);
  assert.equal(value, expected);
  if (value) assert.ok(eligibility.registrationGenderOptions(stored).includes(value));
}

assert.equal(
  eligibility.publicEventEligibilityReason(singlesEvent, {
    doublesSkill: 4.0,
    singlesSkill: null
  }),
  null,
  "a missing singles rating must not fall back to doubles"
);
assert.equal(
  eligibility.publicEventEligibilityReason(singlesEvent, {
    doublesSkill: 4.0,
    singlesSkill: 3.5
  }),
  null,
  "the entered singles rating must control singles eligibility"
);
assert.match(
  eligibility.publicEventEligibilityReason(singlesEvent, {
    doublesSkill: 3.5,
    singlesSkill: 4.0
  }),
  /above this division cap/i,
  "an ineligible singles rating must still be rejected"
);
assert.match(
  eligibility.publicEventEligibilityReason(doublesEvent, {
    doublesSkill: 4.0,
    singlesSkill: 3.5
  }),
  /above this division cap/i,
  "doubles eligibility must continue to use doubles"
);
assert.equal(
  eligibility.publicEventEligibilityReason(doublesEvent, {
    doublesSkill: null,
    singlesSkill: 4.0
  }),
  null,
  "a missing doubles rating must not fall back to singles"
);
