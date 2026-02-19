#!/usr/bin/env python3
from __future__ import annotations

import glob
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_GLOB = str(ROOT / "docs" / "schema" / "live_schema_snapshot_*.sql")
MIGRATIONS_GLOB = str(ROOT / "migrations" / "*.sql")


@dataclass(frozen=True)
class ColumnDef:
    type_name: str
    nullable: bool


SchemaState = dict[str, dict[str, ColumnDef]]


def _normalize_identifier(value: str) -> str:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    if "." in value:
        value = value.split(".")[-1]
    return value.strip('"').lower()


def _strip_comments(sql_text: str) -> str:
    return re.sub(r"--.*?$", "", sql_text, flags=re.MULTILINE)


def _split_sql_statements(sql_text: str) -> list[str]:
    statements: list[str] = []
    buffer: list[str] = []
    depth = 0
    in_single_quote = False
    in_double_quote = False

    for char in sql_text:
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote

        if not in_single_quote and not in_double_quote:
            if char == "(":
                depth += 1
            elif char == ")" and depth > 0:
                depth -= 1

        if char == ";" and depth == 0 and not in_single_quote and not in_double_quote:
            statement = "".join(buffer).strip()
            if statement:
                statements.append(statement)
            buffer = []
            continue

        buffer.append(char)

    tail = "".join(buffer).strip()
    if tail:
        statements.append(tail)
    return statements


def _split_top_level_csv(chunk: str) -> list[str]:
    parts: list[str] = []
    token: list[str] = []
    depth = 0
    in_single_quote = False
    in_double_quote = False

    for char in chunk:
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote

        if not in_single_quote and not in_double_quote:
            if char == "(":
                depth += 1
            elif char == ")" and depth > 0:
                depth -= 1

        if char == "," and depth == 0 and not in_single_quote and not in_double_quote:
            part = "".join(token).strip()
            if part:
                parts.append(part)
            token = []
            continue

        token.append(char)

    tail = "".join(token).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_column_definition(column_sql: str) -> tuple[str, ColumnDef] | None:
    definition = column_sql.strip().rstrip(",")
    if not definition:
        return None

    upper = definition.upper()
    if upper.startswith(("PRIMARY KEY", "FOREIGN KEY", "UNIQUE", "CONSTRAINT", "CHECK")):
        return None

    match = re.match(r'^("?[a-zA-Z_][\w$]*"?)\s+(.+)$', definition, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None

    column_name = _normalize_identifier(match.group(1))
    tail = match.group(2).strip()

    nullable = True
    if re.search(r"\bNOT\s+NULL\b", tail, flags=re.IGNORECASE):
        nullable = False
    elif re.search(r"\bNULL\b", tail, flags=re.IGNORECASE):
        nullable = True

    type_part = re.split(
        r"\b(DEFAULT|NOT\s+NULL|NULL|PRIMARY\s+KEY|REFERENCES|CHECK|UNIQUE|CONSTRAINT|GENERATED)\b",
        tail,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()

    if not type_part:
        return None

    return column_name, ColumnDef(type_name=" ".join(type_part.split()).lower(), nullable=nullable)


def _parse_create_table(statement: str, state: SchemaState) -> None:
    match = re.match(
        r'^CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([\w\."]+)\s*\((.*)\)$',
        statement.strip(),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return

    table = _normalize_identifier(match.group(1))
    body = match.group(2)
    columns: dict[str, ColumnDef] = {}
    for chunk in _split_top_level_csv(body):
        parsed = _parse_column_definition(chunk)
        if parsed is None:
            continue
        col_name, col_def = parsed
        columns[col_name] = col_def

    state[table] = columns


def _apply_alter_action(table_columns: dict[str, ColumnDef], action: str) -> None:
    action = " ".join(action.split())

    add_match = re.match(
        r"^ADD\s+COLUMN\s+(?:IF\s+NOT\s+EXISTS\s+)?(.+)$",
        action,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if add_match:
        parsed = _parse_column_definition(add_match.group(1))
        if parsed:
            col_name, col_def = parsed
            table_columns[col_name] = col_def
        return

    drop_match = re.match(
        r'^DROP\s+COLUMN\s+(?:IF\s+EXISTS\s+)?("?[a-zA-Z_][\w$]*"?).*$',
        action,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if drop_match:
        col_name = _normalize_identifier(drop_match.group(1))
        table_columns.pop(col_name, None)
        return

    type_match = re.match(
        r'^ALTER\s+COLUMN\s+("?[a-zA-Z_][\w$]*"?)\s+TYPE\s+(.+)$',
        action,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if type_match:
        col_name = _normalize_identifier(type_match.group(1))
        type_part = re.split(r"\bUSING\b", type_match.group(2), maxsplit=1, flags=re.IGNORECASE)[0].strip()
        if col_name in table_columns and type_part:
            existing = table_columns[col_name]
            table_columns[col_name] = ColumnDef(
                type_name=" ".join(type_part.split()).lower(),
                nullable=existing.nullable,
            )
        return

    not_null_match = re.match(
        r'^ALTER\s+COLUMN\s+("?[a-zA-Z_][\w$]*"?)\s+SET\s+NOT\s+NULL$',
        action,
        flags=re.IGNORECASE,
    )
    if not_null_match:
        col_name = _normalize_identifier(not_null_match.group(1))
        if col_name in table_columns:
            existing = table_columns[col_name]
            table_columns[col_name] = ColumnDef(type_name=existing.type_name, nullable=False)
        return

    drop_not_null_match = re.match(
        r'^ALTER\s+COLUMN\s+("?[a-zA-Z_][\w$]*"?)\s+DROP\s+NOT\s+NULL$',
        action,
        flags=re.IGNORECASE,
    )
    if drop_not_null_match:
        col_name = _normalize_identifier(drop_not_null_match.group(1))
        if col_name in table_columns:
            existing = table_columns[col_name]
            table_columns[col_name] = ColumnDef(type_name=existing.type_name, nullable=True)


def _parse_alter_table(statement: str, state: SchemaState) -> None:
    match = re.match(
        r'^ALTER\s+TABLE\s+(?:ONLY\s+)?([\w\."]+)\s+(.+)$',
        statement.strip(),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return

    table = _normalize_identifier(match.group(1))
    actions_blob = match.group(2).strip()
    state.setdefault(table, {})

    for action in _split_top_level_csv(actions_blob):
        _apply_alter_action(state[table], action)


def _build_state(sql_files: list[Path]) -> SchemaState:
    state: SchemaState = {}
    for sql_path in sql_files:
        sql_text = _strip_comments(sql_path.read_text(encoding="utf-8"))
        for statement in _split_sql_statements(sql_text):
            stmt = statement.strip()
            if re.match(r"^CREATE\s+TABLE\b", stmt, flags=re.IGNORECASE):
                _parse_create_table(stmt, state)
            elif re.match(r"^ALTER\s+TABLE\b", stmt, flags=re.IGNORECASE):
                _parse_alter_table(stmt, state)
    return state


def _latest_live_snapshot() -> Path:
    matches = sorted(Path(p) for p in glob.glob(SCHEMA_GLOB))
    if not matches:
        raise FileNotFoundError(f"No live schema snapshot found for pattern: {SCHEMA_GLOB}")
    return matches[-1]


def _migration_files() -> list[Path]:
    files = sorted(Path(p) for p in glob.glob(MIGRATIONS_GLOB) if p.endswith(".sql"))
    if not files:
        raise FileNotFoundError(f"No migration SQL files found for pattern: {MIGRATIONS_GLOB}")
    return files


def _compare_schemas(live: SchemaState, expected: SchemaState) -> list[dict[str, str]]:
    diffs: list[dict[str, str]] = []
    for table in sorted(expected):
        expected_cols = expected[table]
        live_cols = live.get(table, {})

        for column in sorted(expected_cols):
            if column not in live_cols:
                diffs.append({"kind": "MISSING_COLUMN", "table": table, "column": column})
                continue

            expected_def = expected_cols[column]
            live_def = live_cols[column]
            if expected_def.type_name != live_def.type_name or expected_def.nullable != live_def.nullable:
                diffs.append(
                    {
                        "kind": "TYPE_MISMATCH",
                        "table": table,
                        "column": column,
                        "expected": f"{expected_def.type_name} {'NULL' if expected_def.nullable else 'NOT NULL'}",
                        "actual": f"{live_def.type_name} {'NULL' if live_def.nullable else 'NOT NULL'}",
                    }
                )

        for column in sorted(live_cols):
            if column not in expected_cols:
                diffs.append({"kind": "EXTRA_COLUMN", "table": table, "column": column})

    return diffs


def _print_diffs(diffs: list[dict[str, str]]) -> None:
    if not diffs:
        print("SCHEMA_DIFF_OK")
        return

    for diff in diffs:
        line = f"{diff['kind']} table={diff['table']} column={diff['column']}"
        if diff["kind"] == "TYPE_MISMATCH":
            line += f" expected=\"{diff['expected']}\" actual=\"{diff['actual']}\""
        print(line)


def main() -> int:
    live_snapshot = _latest_live_snapshot()
    migration_files = _migration_files()

    live_state = _build_state([live_snapshot])
    expected_state = _build_state(migration_files)
    diffs = _compare_schemas(live_state, expected_state)
    _print_diffs(diffs)

    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
