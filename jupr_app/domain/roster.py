import re
import pandas as pd


def normalize_slots(roster_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure slots are contiguous 1..N within each court, then sort by (court, slot).
    Required columns: court, slot (slot will be created if missing)
    """
    df = roster_df.copy()

    if df.empty:
        # Return a consistent empty frame with expected columns if possible
        if "court" not in df.columns:
            df["court"] = pd.Series(dtype=int)
        if "slot" not in df.columns:
            df["slot"] = pd.Series(dtype=int)
        return df

    if "slot" not in df.columns:
        df["slot"] = 1

    df["court"] = df["court"].astype(int)
    df["slot"] = df["slot"].astype(int)

    for c in sorted(df["court"].unique()):
        idx = df[df["court"] == c].sort_values("slot").index
        df.loc[idx, "slot"] = list(range(1, len(idx) + 1))

    return df.sort_values(["court", "slot"]).reset_index(drop=True)


def swap_players(roster_df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """
    Swap the (court, slot) assignments of two players by name.
    Note: Long-term, prefer swapping by player_id to avoid duplicate-name issues.
    """
    df = roster_df.copy()

    ia = df.index[df["name"] == a]
    ib = df.index[df["name"] == b]
    if len(ia) != 1 or len(ib) != 1:
        return df

    ia, ib = int(ia[0]), int(ib[0])

    ca, sa = int(df.at[ia, "court"]), int(df.at[ia, "slot"])
    cb, sb = int(df.at[ib, "court"]), int(df.at[ib, "slot"])

    df.at[ia, "court"], df.at[ia, "slot"] = cb, sb
    df.at[ib, "court"], df.at[ib, "slot"] = ca, sa

    return normalize_slots(df)


def move_within_court(roster_df: pd.DataFrame, player: str, new_slot: int) -> pd.DataFrame:
    """
    Reorder a player within their current court.
    """
    df = roster_df.copy()

    if df.empty or "name" not in df.columns or "court" not in df.columns:
        return df

    if player not in df["name"].astype(str).tolist():
        return df

    row = df[df["name"] == player].iloc[0]
    c = int(row["court"])

    grp = df[df["court"] == c].sort_values("slot").copy()
    names = grp["name"].astype(str).tolist()
    if player not in names:
        return df

    names.remove(player)
    new_slot = max(1, min(int(new_slot), len(names) + 1))
    names.insert(new_slot - 1, player)

    for i, nm in enumerate(names, start=1):
        df.loc[(df["court"] == c) & (df["name"] == nm), "slot"] = i

    return normalize_slots(df)


def compress_courts(roster_df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-map court numbers to be contiguous 1..N (fixes gaps if a court becomes empty).
    """
    df = roster_df.copy()
    if df.empty or "court" not in df.columns:
        return df

    courts = sorted(df["court"].astype(int).unique().tolist())
    mapping = {old: i + 1 for i, old in enumerate(courts)}
    df["court"] = df["court"].astype(int).map(mapping)

    return normalize_slots(df)


def move_player_to_court(
    roster_df: pd.DataFrame,
    player: str,
    target_court: int,
    target_slot: int = 1
) -> pd.DataFrame:
    """
    Move a player to a different court and insert them at target_slot within that court.
    Court sizes will change accordingly. Slots are normalized after.

    Note: This version identifies the player by 'name'. Long-term, prefer player_id.
    """
    df = roster_df.copy()

    if df.empty or "name" not in df.columns:
        return df

    if player not in df["name"].astype(str).tolist():
        return df

    target_court = int(target_court)
    target_slot = int(target_slot)

    # Remove player row
    row = df[df["name"] == player].iloc[0].copy()
    df = df[df["name"] != player].copy()

    # Ensure courts/slots are stable before insert
    df = normalize_slots(df)

    # Build target court ordering and insert player name
    target_names = df[df["court"] == target_court].sort_values("slot")["name"].tolist()
    target_slot = max(1, min(target_slot, len(target_names) + 1))
    target_names.insert(target_slot - 1, player)

    # Apply ordering to existing rows on that court
    for i, nm in enumerate(target_names, start=1):
        df.loc[(df["court"] == target_court) & (df["name"] == nm), "slot"] = i

    # Add player back if they aren't in df yet (they won't be)
    if player not in df["name"].tolist():
        df = pd.concat([df, pd.DataFrame([{
            "player_id": int(row.get("player_id")) if "player_id" in row else None,
            "name": str(player),
            "rating": float(row.get("rating", 1200.0)),
            "court": int(target_court),
            "slot": int(target_slot),
        }])], ignore_index=True)

    # Normalize + compress (handles empty courts)
    df = normalize_slots(df)
    df = compress_courts(df)

    return df


def roster_df_to_courts(roster_df: pd.DataFrame, ladder_court_sizes: list[int] | None = None) -> list[dict]:
    """
    Convert roster dataframe into the payload expected by your React court_board component.

    Expects roster_df columns: player_id, name, rating, court, slot
    rating is stored as ELO in DB; we convert to JUPR for display (rating/400.0).

    ladder_court_sizes is optional; pass in st.session_state.get("ladder_court_sizes")
    """
    df = roster_df.copy()

    if df.empty:
        return [{"court_id": "Bench", "players": []}]

    # Normalize types
    if "court" in df.columns:
        df["court"] = df["court"].astype(int)
    if "player_id" in df.columns:
        df["player_id"] = df["player_id"].astype(int)

    # Stable ordering within court
    if "slot" in df.columns:
        df["slot"] = df["slot"].astype(int)
        df = df.sort_values(["court", "slot"], ascending=[True, True])
    else:
        df = df.sort_values(["court", "rating"], ascending=[True, False])

    # Prefer configured court structure if provided
    sizes = ladder_court_sizes if isinstance(ladder_court_sizes, list) and ladder_court_sizes else None
    if sizes:
        court_nums = list(range(1, len(sizes) + 1))
    else:
        court_nums = sorted(df["court"].unique().tolist()) if "court" in df.columns else [1]

    courts: list[dict] = []
    for c in court_nums:
        cdf = df[df["court"] == int(c)].copy() if "court" in df.columns else pd.DataFrame()
        players = []
        if not cdf.empty:
            for _, r in cdf.iterrows():
                players.append(
                    {
                        "player_id": str(int(r["player_id"])),  # draggableId must be string
                        "name": str(r["name"]),
                        "rating": float(r.get("rating", 1200.0)) / 400.0,  # display JUPR
                    }
                )

        target_size = None
        if sizes and 0 <= (int(c) - 1) < len(sizes):
            target_size = int(sizes[int(c) - 1])

        courts.append(
            {
                "court_id": f"Court {int(c)}",
                "players": players,
                "target_size": target_size,  # frontend may ignore; harmless
            }
        )

    # Always include Bench (frontend may use it)
    courts.append({"court_id": "Bench", "players": []})
    return courts


def courts_to_roster_df(courts: list[dict], prev_roster_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the court_board output payload back into a roster dataframe.

    Bench is ignored here (current behavior in your app). If you later want bench persistence,
    we can store Bench as court=0 instead of dropping them.
    """
    df_prev = prev_roster_df.copy()
    if df_prev.empty:
        return prev_roster_df

    df_prev["player_id"] = df_prev["player_id"].astype(int)

    elo_map = dict(zip(df_prev["player_id"], df_prev["rating"]))
    name_map = dict(zip(df_prev["player_id"], df_prev["name"]))

    rows = []
    for c in courts:
        cid = str(c.get("court_id", ""))
        if cid == "Bench":
            continue

        m = re.findall(r"\d+", cid)
        if not m:
            continue
        cnum = int(m[0])

        players = c.get("players", []) or []
        for i, p in enumerate(players, start=1):
            pid = int(p["player_id"])
            rows.append(
                {
                    "player_id": pid,
                    "name": name_map.get(pid, str(p.get("name", pid))),
                    "rating": float(elo_map.get(pid, 1200.0)),
                    "court": int(cnum),
                    "slot": int(i),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return prev_roster_df

    out = out.sort_values(["court", "slot"], ascending=[True, True]).reset_index(drop=True)
    out = compress_courts(normalize_slots(out))
    return out
