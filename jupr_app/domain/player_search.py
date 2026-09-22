"""Name fragments for discovery only; never used as proof of player identity."""

import unicodedata


def player_search_key(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    return " ".join("".join(char for char in text if not unicodedata.combining(char)).split())


def matches_player_search(name: object, query: object) -> bool:
    haystack = player_search_key(name)
    return all(part in haystack for part in player_search_key(query).split())
