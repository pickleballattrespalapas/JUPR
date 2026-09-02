from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts import deployment_verifier


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config" / "recovered_staging_migration_sources.json"
MIGRATIONS = ROOT / "supabase" / "migrations"


def test_recovered_staging_migration_sources_match_ledger_digests() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = payload["migrations"]

    assert payload["schema_version"] == 1
    assert payload["source_ledger_table"] == (
        "supabase_migrations.schema_migrations"
    )
    assert len(entries) == 13
    assert entries == sorted(entries, key=lambda entry: entry["version"])

    seen_versions: set[str] = set()
    seen_names: set[str] = set()
    for entry in entries:
        version = entry["version"]
        name = entry["name"]
        assert version not in seen_versions
        assert name not in seen_names
        seen_versions.add(version)
        seen_names.add(name)

        source_path = MIGRATIONS / f"{version}_{name}.sql"
        repository_bytes = source_path.read_bytes()
        assert repository_bytes.endswith(b"\n")

        source_bytes = repository_bytes
        if not entry["source_statement_ended_with_newline"]:
            source_bytes = repository_bytes[:-1]

        assert len(source_bytes.decode("utf-8")) == entry[
            "source_statement_chars"
        ]
        assert hashlib.sha256(source_bytes).hexdigest() == entry[
            "source_statement_sha256"
        ]


def test_recovered_staging_migrations_keep_repository_inventory_unique() -> None:
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))["migrations"]
    inventory = deployment_verifier.expected_migration_inventory(MIGRATIONS)
    inventory_versions = [version for version, _ in inventory]
    inventory_names = [name for _, name in inventory]

    assert len(inventory_versions) == len(set(inventory_versions))
    assert len(inventory_names) == len(set(inventory_names))
    for entry in entries:
        assert (entry["version"], entry["name"]) in inventory
