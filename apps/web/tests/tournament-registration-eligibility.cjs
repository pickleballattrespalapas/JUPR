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
