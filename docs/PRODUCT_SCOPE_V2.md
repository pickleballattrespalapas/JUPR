# JUPR v2 Product Scope (Binding)

## Purpose

This document defines what is INCLUDED and EXCLUDED in JUPR v2.

It prevents feature creep and ensures alignment with ARCHITECTURE_V2.md.

If a feature is not listed as "Must Have" or "Allowed", it is
out-of-scope for v2.

------------------------------------------------------------------------

# Product Definition

JUPR v2 is a multi-tenant SaaS ratings engine for clubs.

It is NOT a full club management system.

Primary User: Club Admin\
Secondary User: Club Member (public viewer)

Primary Goal: Operational simplicity\
Secondary Goal: Member engagement through visible standings

------------------------------------------------------------------------

# MUST HAVE (Required for v2 Launch)

## Multi-Tenant Foundation

-   Path-based routing: /club/`<slug>`{=html}
-   Club isolation via club_id
-   Unique club slug enforcement
-   Separate admin accounts per club
-   Optional coordinator role per club

## Admin Core Workflow

-   Create event (League / Ladder / Round Robin / Tournament)
-   Enter match results quickly
-   Clear validation errors
-   Immediate rating update
-   Ability to undo recent match
-   Stable event grouping

## Ratings Engine

-   Deterministic rating updates
-   Snapshot storage for match ratings
-   No cross-club rating logic
-   Idempotent match submission

## Public Experience

-   Clean leaderboard page
-   Rating display per player
-   Stable deep links
-   Public mode (no login required)

## Badge System (Simplified)

-   5--10 core badges only
-   Deterministic trigger rules
-   Idempotent award logic
-   Background job processing only
-   No badge evaluation in render()

## Background Processing

-   Queue-based badge evaluation
-   Weekly recap generation outside render
-   No heavy compute in Streamlit render()

------------------------------------------------------------------------

# NICE TO HAVE (If Stable)

-   Simple admin dashboard overview
-   Basic event archive view
-   Minimal club branding (logo + color)
-   Lightweight recap presentation (non-PDF fallback allowed)

------------------------------------------------------------------------

# EXPLICITLY OUT OF SCOPE (v2)

-   Payment processing
-   Court reservations
-   Full CRM / member accounts
-   Cross-club rating portability
-   Global leaderboard
-   Advanced badge trees
-   Custom domain mapping
-   Automated billing infrastructure
-   Mobile app
-   Complex analytics dashboards

------------------------------------------------------------------------

# Launch Definition

JUPR v2 is ready when:

1.  Admin can enter 8 matches in under 2 minutes.
2.  No known crash paths in match entry flow.
3.  Badge engine does not block UI.
4.  Public leaderboard renders instantly.
5.  Multi-club isolation is verified.

Stability \> Feature Quantity.

------------------------------------------------------------------------

# Scope Discipline Rule

Before building any feature, ask:

1.  Does this improve admin workflow?
2.  Does this improve stability?
3.  Is this required for v2?
4.  Does it comply with ARCHITECTURE_V2.md?

If the answer is no, defer it.

This document is binding for all v2 product decisions.
