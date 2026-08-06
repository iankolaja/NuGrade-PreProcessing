"""Remove duplicated measurement rows from a database.

    python deduplicate_measurements.py output/nugrade_data.db --dry-run
    python deduplicate_measurements.py output/nugrade_data.db

A resume run of stage 1 appends any reaction channel not listed in ingest_progress. If the
measurements table was built by something that did not populate that table — the original
notebook, or a pipeline run from before it existed — every channel looks unprocessed and a
second full copy is appended. The result is an exact doubling: 2,611,611 rows became
5,223,226.

This removes rows that are identical across every column, keeping the first of each. That is
safe here because a genuine EXFOR corpus does not contain two rows agreeing on entry,
subentry, energy, cross section, uncertainty and both evaluation comparisons at once; the
duplicates come from the append, not from the data.

The database is modified in place, so it is copied first unless --in-place is given.
"""
import argparse
import shutil
import sqlite3
import sys
import time
from pathlib import Path


def table_columns(con, table):
    return [r[1] for r in con.execute(f"PRAGMA table_info({table})")]


def count_rows(con, table="measurements"):
    return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def count_distinct(con, columns, table="measurements"):
    quoted = ", ".join(f'"{c}"' for c in columns)
    return con.execute(
        f"SELECT COUNT(*) FROM (SELECT DISTINCT {quoted} FROM {table})").fetchone()[0]


def deduplicate(con, table="measurements", emit=print):
    """Delete rows identical across every column, keeping the lowest rowid of each group."""
    columns = [c for c in table_columns(con, table) if c != "rowid"]
    quoted = ", ".join(f'"{c}"' for c in columns)

    before = count_rows(con, table)
    emit(f"  {before:,} rows, {len(columns)} columns")

    started = time.monotonic()
    con.execute(f"""
        DELETE FROM {table}
        WHERE rowid NOT IN (SELECT MIN(rowid) FROM {table} GROUP BY {quoted})
    """)
    con.commit()
    after = count_rows(con, table)
    emit(f"  removed {before - after:,} duplicate rows in "
         f"{time.monotonic() - started:.0f}s; {after:,} remain")
    return before, after


def reset_progress(con, emit=print):
    """Clear ingest_progress so a later resume does not trust counts that no longer hold.

    After a dedup the per-channel n_rows in that table describe the doubled corpus. Leaving
    it would make a subsequent resume skip channels on the strength of stale bookkeeping.
    """
    tables = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    if "ingest_progress" not in tables:
        return 0
    rows = count_rows(con, "ingest_progress")
    if rows:
        con.execute("DELETE FROM ingest_progress")
        con.commit()
        emit(f"  cleared {rows:,} stale ingest_progress rows")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("database")
    parser.add_argument("--out", default=None,
                        help="write the repaired copy here (default: <database>.deduped)")
    parser.add_argument("--in-place", action="store_true",
                        help="modify the database directly instead of copying it first")
    parser.add_argument("--dry-run", action="store_true",
                        help="report how many rows are duplicated, change nothing")
    parser.add_argument("--keep-progress", action="store_true",
                        help="do not clear ingest_progress after deduplicating")
    args = parser.parse_args()

    source = Path(args.database)
    if not source.is_file():
        print(f"no database at {source}", file=sys.stderr)
        return 2

    if args.dry_run:
        con = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
        try:
            columns = [c for c in table_columns(con, "measurements") if c != "rowid"]
            total = count_rows(con)
            distinct = count_distinct(con, columns)
        finally:
            con.close()
        print(f"{source}")
        print(f"  rows            {total:,}")
        print(f"  distinct rows   {distinct:,}")
        print(f"  duplicates      {total - distinct:,}")
        if total and distinct:
            print(f"  ratio           {total / distinct:.4f}x")
        print("\nnothing was changed (--dry-run)")
        return 0

    target = source if args.in_place else Path(args.out or f"{source}.deduped")
    if not args.in_place:
        print(f"copying {source} -> {target}")
        shutil.copyfile(source, target)

    con = sqlite3.connect(target)
    try:
        print(f"deduplicating {target}")
        before, after = deduplicate(con)
        if not args.keep_progress:
            reset_progress(con)
        print("  VACUUM ...")
        con.isolation_level = None
        con.execute("VACUUM")
    finally:
        con.close()

    print(f"\ndone: {before:,} -> {after:,} rows")
    print(f"check it with:  python validate_output_db.py {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
