# JUPR v2 Architecture Direction (Binding)

## Purpose

JUPR is a multi-tenant SaaS platform for club ratings management.

It operates: - Leagues - Ladders - Round Robins - Tournaments

It generates player ratings from match results and presents standings in
a clean, engaging public-facing site.

Primary user: Club Admin\
Secondary user: Club Member (public viewer)

Primary goal: Operational simplicity\
Secondary goal: Club engagement & growth

------------------------------------------------------------------------

# Core Product Principles

1.  Admin-first design.
2.  Score entry must be frictionless and stable.
3.  No heavy compute during Streamlit render.
4.  Public UX must feel polished and official.
5.  Clubs operate as isolated islands.
6.  Architecture must support future SaaS growth.

------------------------------------------------------------------------

# Multi-Tenant Strategy

## URL Structure (Phase 1)

Path-based routing:

    jupr.app/club/<slug>

Each club has: - Unique slug - Isolated data - Separate admin accounts -
Separate coordinators

Subdomain support may be added in future phases.

Custom domains are out-of-scope for v2.

------------------------------------------------------------------------

# Data Isolation

All domain tables are scoped by:

    club_id

Ratings are strictly club-isolated.

There is NO global identity layer. There is NO cross-club rating
portability.

JUPR is not a universal ratings system.

------------------------------------------------------------------------

# Auth Model

Each club has: - Admin accounts - Optional coordinator accounts

Public mode requires no login.

Self-onboarding should be supported, but manual onboarding is acceptable
in early stages.

------------------------------------------------------------------------

# Core Loop

Admin runs event\
→ Admin enters match results\
→ Ratings update immediately\
→ Leaderboards shift\
→ Members engage\
→ Repeat

Badges and recaps enhance the loop but do not power it.

The engine is reliable match entry + deterministic rating updates.

------------------------------------------------------------------------

# Badge System (v2 Scope)

Badges remain in v2 but simplified.

Rules: - Deterministic triggers only - Limited initial badge set
(5--10) - Idempotent award logic - Fully processed in background job -
Never executed in render()

Badge evaluation must use a worker model (queue-based).

------------------------------------------------------------------------

# Background Work

Heavy processes must run outside Streamlit render: - Badge evaluation -
Weekly recap generation - PDF generation - Story generation

System must evolve toward:

    badge_eval_queue + worker process

Render must remain fast and safe.

------------------------------------------------------------------------

# Stability Requirements

-   No raw HTML rendering
-   Guard against NoneType crashes
-   Idempotent match submission
-   Deterministic routing
-   Additive DB migrations only
-   No destructive operations without explicit confirmation
-   No secrets logged

JUPR v2 must feel finished and reliable.

------------------------------------------------------------------------

# Admin UX Priorities

1.  Fast match entry (keyboard optimized)
2.  Clear validation errors
3.  Immediate rating update feedback
4.  Ability to undo recent match
5.  Clean event grouping (League / Ladder / Tournament)
6.  No confusing UI states

Admin time-to-enter 8 matches target: Under 2 minutes.

------------------------------------------------------------------------

# Public UX Priorities

-   Clean leaderboard presentation
-   Clear rating explanations
-   Stable deep links
-   Weekly recap reliability
-   Welcoming, inclusive tone

Public site must feel official and polished.

------------------------------------------------------------------------

# Out of Scope (v2)

-   Payment processing
-   Court booking
-   Full club management CRM
-   Cross-club rating portability
-   Advanced billing automation
-   Custom domain mapping

------------------------------------------------------------------------

# Long-Term Vision (Not v2 Commitment)

-   Subscription-based multi-club SaaS
-   Strong club branding options
-   Optional subdomain model
-   Potential rating reference portability
-   Name rebrand when scaling

------------------------------------------------------------------------

# Non-Negotiables

1.  Stability over features.
2.  Admin workflow over novelty.
3.  Determinism over magic.
4.  Performance over cleverness.
5.  Small, safe PRs only.

This document is binding for all architectural decisions in JUPR v2.

Badge Engine Subsystem (V3)
The Badge Engine is defined in ARCHITECTURE_V3.md and must fully comply with all V2 system constraints, including deterministic routing, background job execution for heavy computation, additive migrations only, strict club_id scoping, and idempotent write guarantees. V3 is a subsystem specification layered on top of the V2 platform architecture and does not replace V2 principles.
