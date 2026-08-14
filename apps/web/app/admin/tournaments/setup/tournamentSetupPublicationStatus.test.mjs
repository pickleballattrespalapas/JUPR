import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const modulePath = process.env.JUPR_TOURNAMENT_SETUP_PUBLICATION_STATUS_MODULE;

if (!modulePath) {
  throw new Error("JUPR_TOURNAMENT_SETUP_PUBLICATION_STATUS_MODULE is required");
}

const { setupPublicationStatus } = require(modulePath);

assert.equal(
  setupPublicationStatus({
    detailLoadState: "idle",
    hasAuthoritativeDetail: false,
    hasUnpublishedChanges: false
  }),
  "checking",
  "the first render must not claim that the published setup is current"
);

assert.equal(
  setupPublicationStatus({
    detailLoadState: "loading",
    hasAuthoritativeDetail: false,
    hasUnpublishedChanges: false
  }),
  "checking",
  "a pending detail request must remain neutral"
);

assert.equal(
  setupPublicationStatus({
    detailLoadState: "loading",
    hasAuthoritativeDetail: true,
    hasUnpublishedChanges: false
  }),
  "checking",
  "a refresh must not present stale detail as an authoritative current result"
);

assert.equal(
  setupPublicationStatus({
    detailLoadState: "loaded",
    hasAuthoritativeDetail: true,
    hasUnpublishedChanges: false
  }),
  "current"
);

assert.equal(
  setupPublicationStatus({
    detailLoadState: "loaded",
    hasAuthoritativeDetail: true,
    hasUnpublishedChanges: true
  }),
  "unpublished"
);

assert.equal(
  setupPublicationStatus({
    detailLoadState: "failed",
    hasAuthoritativeDetail: false,
    hasUnpublishedChanges: false
  }),
  "unavailable",
  "a failed load must never fall back to the green current state"
);

assert.equal(
  setupPublicationStatus({
    detailLoadState: "loaded",
    hasAuthoritativeDetail: false,
    hasUnpublishedChanges: false
  }),
  "unavailable",
  "green requires an authoritative detail payload"
);

console.log("tournament setup publication status: 7/7 passed");
