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

function jsonResponse(status, payload) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" }
  });
}

async function withFetch(response, action) {
  const original = global.fetch;
  global.fetch = async () => response;
  try {
    return await action();
  } finally {
    global.fetch = original;
  }
}

async function main() {
  process.env.JUPR_API_BASE_URL = "https://api.example.test";
  const support = loadTypeScript("lib/supportIntakeApi.ts");
  const email = loadTypeScript("lib/emailPreferencesApi.ts");
  const commerce = loadTypeScript("lib/tournamentCommerceApi.ts");
  const registration = loadTypeScript("lib/tournamentRegistrationApi.ts");
  const teams = loadTypeScript("lib/tournamentTeamCompetitionApi.ts");
  const eligibility = loadTypeScript("lib/tournamentRegistrationEligibility.ts", {
    "@/lib/tournamentSkillEligibility": {
      skillEligibilityPolicy: () => ({ mode: "OPEN" })
    }
  });

  const supportPayload = {
    request_type: "general_support",
    requester_name: "Sam",
    requester_email: "sam@example.com",
    subject: "Help",
    description: "Please help",
    consent_to_contact: true
  };
  const supportValidation = await withFetch(
    jsonResponse(400, { detail: "A valid email is required." }),
    () => support.submitPublicSupportRequest("club", supportPayload)
  );
  assert.equal(supportValidation.status, 400);
  assert.equal(supportValidation.error, "Enter a valid email address.");
  const supportLimit = await withFetch(
    jsonResponse(429, { detail: "Too many support requests were submitted." }),
    () => support.submitPublicSupportRequest("club", supportPayload)
  );
  assert.equal(supportLimit.status, 429);
  assert.match(supportLimit.error, /wait an hour/i);
  const supportSuccess = await withFetch(
    jsonResponse(200, { ok: true, message: "Thanks — we received your request." }),
    () => support.submitPublicSupportRequest("club", supportPayload)
  );
  assert.equal(supportSuccess.data.message, "Thanks — we received your request.");

  const oldPreferenceLink = await withFetch(
    jsonResponse(400, { detail: "Legacy subscription-id preference links are no longer accepted." }),
    () => email.getEmailPreferences({ subscriptionId: "old" })
  );
  assert.equal(oldPreferenceLink.status, 400);
  assert.match(oldPreferenceLink.error, /out of date/i);
  const badScope = await withFetch(
    jsonResponse(400, { detail: "Unsupported email preference scope." }),
    () => email.unsubscribeEmailPreferences({ token: "token", scope: "bad" })
  );
  assert.match(badScope.error, /choose which emails/i);

  const invalidQuantity = await withFetch(
    jsonResponse(400, { detail: "Tournament shirt — Large is no longer available in that quantity." }),
    () => commerce.quoteTournamentCommerce("club", {
      tournament_id: "tournament",
      event_option_ids: [],
      item_selections: []
    })
  );
  assert.equal(invalidQuantity.status, 400);
  assert.match(invalidQuantity.error, /Tournament shirt.*quantity/i);
  const changedTotal = await withFetch(
    jsonResponse(409, { detail: "Tournament extras or pricing changed." }),
    () => commerce.quoteTournamentCommerce("club", {
      tournament_id: "tournament",
      event_option_ids: [],
      item_selections: []
    })
  );
  assert.match(changedTotal.error, /total changed/i);
  const commerceAdminDiagnostic = await withFetch(
    jsonResponse(500, { detail: "Private commerce operator diagnostic" }),
    () => commerce.mutateAdminTournamentCommerce("/admin/example", "POST", {}, "access")
  );
  assert.equal(commerceAdminDiagnostic.error, "Private commerce operator diagnostic");

  const duplicate = await withFetch(
    jsonResponse(409, { detail: "You’re already registered with this email." }),
    () => registration.submitClubTournamentRegistration("club", {})
  );
  assert.equal(duplicate.status, 409);
  assert.match(duplicate.error, /already registered/i);
  const imported = await withFetch(
    jsonResponse(409, { detail: "This registration is already imported into a draw and can no longer be edited publicly." }),
    () => registration.submitClubTournamentRegistrationEdit("club", {})
  );
  assert.match(imported.error, /tournament draw/i);
  const locked = await withFetch(
    jsonResponse(409, { detail: "This event entry has an active partner relationship." }),
    () => registration.submitClubTournamentRegistrationEdit("club", {})
  );
  assert.match(locked.error, /partner connection/i);
  const expiredEdit = await withFetch(
    jsonResponse(400, { detail: "Registration edit link has expired." }),
    () => registration.getClubTournamentRegistrationEdit("club", { editToken: "old" })
  );
  assert.match(expiredEdit.error, /expired.*request a new one/i);
  const quoteRefresh = await withFetch(
    jsonResponse(409, {
      detail: {
        message: "Tournament extras or pricing changed.",
        current_quote: { quote_fingerprint: "new" }
      }
    }),
    () => registration.submitClubTournamentRegistration("club", {})
  );
  assert.equal(quoteRefresh.current_quote.quote_fingerprint, "new");

  const expiredInvite = await withFetch(
    jsonResponse(400, { detail: "Team invitation has expired." }),
    () => teams.resolvePublicTeamInvitation("club", "old")
  );
  assert.equal(expiredInvite.error, "This invitation has expired.");
  const answeredInvite = await withFetch(
    jsonResponse(400, { detail: "Team invitation has already been resolved." }),
    () => teams.resolvePublicTeamInvitation("club", "used")
  );
  assert.match(answeredInvite.error, /already been answered/i);
  const staleInvite = await withFetch(
    jsonResponse(400, { detail: "Team invitation was replaced by a newer invitation." }),
    () => teams.resolvePublicTeamInvitation("club", "stale")
  );
  assert.match(staleInvite.error, /newest invitation email/i);
  const identityMismatch = await withFetch(
    jsonResponse(400, { detail: "Invitation identity does not match a confirmed registration." }),
    () => teams.respondPublicTeamInvitation("club", {})
  );
  assert.match(identityMismatch.error, /registered player/i);
  const invitationPermission = await withFetch(
    jsonResponse(403, { detail: "You do not have permission to use this invitation." }),
    () => teams.respondPublicTeamInvitation("club", {})
  );
  assert.match(invitationPermission.error, /don’t have permission/i);
  const invitationWritesClosed = await withFetch(
    jsonResponse(403, { detail: "Four-player team writes are disabled." }),
    () => teams.respondPublicTeamInvitation("club", {})
  );
  assert.match(invitationWritesClosed.error, /can’t be changed right now/i);
  const invitationService = await withFetch(
    jsonResponse(503, { detail: "private database failure" }),
    () => teams.respondPublicTeamInvitation("club", {})
  );
  assert.match(invitationService.error, /temporarily unavailable/i);
  assert.doesNotMatch(invitationService.error, /private|database/i);
  const teamService = await withFetch(
    jsonResponse(503, { detail: "private database failure" }),
    () => teams.createPublicFourPlayerTeam("club", {})
  );
  assert.equal(teamService.status, 503);
  assert.doesNotMatch(teamService.error, /private|database/i);
  const recoveryService = await withFetch(
    jsonResponse(503, { detail: "private database failure" }),
    () => teams.recoverPublicFourPlayerTeamSetup("club", "token")
  );
  assert.doesNotMatch(recoveryService.error, /private|database/i);
  const adminDiagnostic = await withFetch(
    jsonResponse(500, { detail: "Private operator diagnostic" }),
    () => teams.mutateAdminTeamCompetition("/admin/example", {}, "access")
  );
  assert.equal(adminDiagnostic.error, "Private operator diagnostic");

  for (const result of [
    await withFetch(jsonResponse(500, { detail: "private database failure" }), () =>
      support.submitPublicSupportRequest("club", supportPayload)
    ),
    await withFetch(jsonResponse(500, { detail: "private database failure" }), () =>
      email.getEmailPreferences({ token: "token" })
    ),
    await withFetch(jsonResponse(500, { detail: "private database failure" }), () =>
      commerce.quoteTournamentCommerce("club", {
        tournament_id: "tournament",
        event_option_ids: [],
        item_selections: []
      })
    ),
    await withFetch(jsonResponse(500, { detail: "private database failure" }), () =>
      registration.getClubTournamentRegistration("club")
    )
  ]) {
    assert.equal(result.status, 500);
    assert.doesNotMatch(result.error, /private|database/i);
  }

  assert.equal(eligibility.publicEventFormatLabel("POOL_PLAY"), "Pool play");
  assert.equal(eligibility.publicScoringLabel("RALLY_SCORING"), "Rally scoring");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
