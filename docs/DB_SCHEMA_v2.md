# DB_SCHEMA_v2.md

JUPR v2 -- Canonical Production Schema

This document reflects the actual PROD database structure and is the
authoritative reference for all future migrations.

------------------------------------------------------------------------

## Core Principle

All club-operational data must be isolated via:

club_id (text) → references clubs(id)

There is: - No global rating portability - No cross-club joins - No
shared player identity across clubs

Clubs operate as isolated islands.

------------------------------------------------------------------------

# ROOT ENTITY

## clubs (GLOBAL ROOT)

Primary tenant entity.

-   id (text, PK)
-   name (text)
-   status (text)
-   logo_url (text)
-   primary_color (text)
-   created_at (timestamptz)

All club-scoped tables reference: FOREIGN KEY (club_id) → clubs(id)

------------------------------------------------------------------------

# PLAYERS DOMAIN (Club Scoped)

## players

-   id (bigint, PK)
-   club_id (text, FK → clubs.id)
-   name (text)
-   rating (numeric)
-   starting_rating (numeric)
-   matches_played (int)
-   wins (int)
-   losses (int)
-   active (boolean)
-   created_at (timestamptz)

FK enforced.

## player_profiles

Composite PK: (club_id, player_id)

-   club_id (text)
-   player_id (bigint, FK → players.id)
-   preferred_channel (text)
-   email (text)
-   phone (text)
-   whatsapp (text)

Missing FK on club_id → clubs(id).

------------------------------------------------------------------------

# MATCHES DOMAIN (Club Scoped)

## matches

-   id (bigint, PK)
-   club_id (text, FK → clubs.id)
-   league (text)
-   player references → players(id)
-   rating snapshot fields

FK enforced.

------------------------------------------------------------------------

# LEAGUE SYSTEM (Club Scoped)

## league_ratings

-   id (bigint, PK)
-   club_id (text, FK → clubs.id)
-   player_id (bigint, FK → players.id)
-   league_name (text)
-   rating (numeric)
-   wins (int)
-   losses (int)

FK enforced.

## leagues_metadata

-   id (bigint, PK)
-   club_id (text)
-   league_name (text)
-   configuration fields

Missing FK on club_id.

------------------------------------------------------------------------

# LADDER SYSTEM (Club Scoped)

All ladder tables properly reference club_id → clubs(id).

Includes: - ladder_roster - ladder_challenges -
ladder_challenge_matches - ladder_settings - ladder_pass_usage -
ladder_player_flags - ladder_audit_log

Ladder system structurally strong.

------------------------------------------------------------------------

# TOURNAMENT SYSTEM (Club Scoped)

## tournaments

-   id (uuid, PK)
-   club_id (text)
-   name (text)
-   status (text)

Missing FK on club_id.

Related tables: - tournament_games - tournament_teams -
tournament_podium

------------------------------------------------------------------------

# BADGE SYSTEM

## Legacy

### badges (global catalog)

-   badge_id (text, PK)
-   name
-   category
-   prestige
-   rarity
-   metadata fields

### player_badges (club scoped)

-   club_id (text)
-   player_id (bigint)
-   badge_id (text)

Missing FK on club_id → clubs(id).

------------------------------------------------------------------------

## Badge Engine Tables

### badge_eval_queue (club scoped)

-   club_id (text)

Missing FK on club_id.

### badge_eval_runs (global)

No club_id (engine-level execution).

### badge_events_v2 (global)

No club_id.

------------------------------------------------------------------------

## v2 Badge Catalog (Global)

-   badges_v2
-   user_badges_v2

------------------------------------------------------------------------

# WEEKLY RECAPS (Club Scoped)

## weekly_recaps

-   club_id (text)
-   week_start (date)

Missing FK on club_id.

------------------------------------------------------------------------

# EVENTS (Club Scoped)

## events

-   club_id (text)

Missing FK on club_id.

------------------------------------------------------------------------

# Global vs Club-Scoped Summary

Properly Enforced Club Isolation: - players - matches - league_ratings -
ladder tables

Club-Scoped But Missing FK: - player_profiles - player_badges -
badge_eval_queue - weekly_recaps - events - tournaments -
leagues_metadata

Global Tables: - badges_v2 - user_badges_v2 - badge_events_v2 -
badge_eval_runs

------------------------------------------------------------------------

# Governance Rules

1.  All club-scoped tables must include club_id text NOT NULL.
2.  All club-scoped tables must enforce FK → clubs(id).
3.  All migrations applied to STAGING first, then PROD.
4.  DB_SCHEMA_v2.md must be updated after structural changes.
