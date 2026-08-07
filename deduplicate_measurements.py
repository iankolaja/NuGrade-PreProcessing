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


# What identifies one physical measurement, independent of anything the pipeline computes
# from it. Two rows agreeing on all of these are the same measurement ingested twice.
#
# Dataset_Number and dEnergy are in the key because EXFOR distinguishes on them: the corpus
# holds 11 groups that agree on subentry, energy, cross section and uncertainty but belong
# to different datasets, or carry a different energy uncertainty. Leaving them out collapsed
# 13 genuine measurements — small, but silent data loss is exactly what this tool must not
# do.
NATURAL_KEY = ["EXFOR_Subentry", "Dataset_Number", "Energy", "dEnergy",
               "Data", "dData", "MT", "Projectile"]

# Columns produced by the evaluation comparison. A re-ingest that could not find its ACE
# files writes these as NULL, so they measure how complete a row is.
EVALUATION_COLUMNS = ["endf8", "endf7-1"]


def deduplicate(con, table="measurements", emit=print, natural=True):
    """Remove rows ingested more than once, keeping the most complete copy of each.

    Exact-match deduplication is not enough on its own. A second ingest run computes the
    derived columns again, and if it used different code or could not reach the evaluation
    files, its rows differ from the first copy in exactly those columns — so every row looks
    distinct while the corpus is plainly doubled. That is what happened on the cluster: a
    re-run with no ACE paths appended 2,611,193 rows whose endf8 and endf7-1 are all NULL.

    Rows are therefore grouped by the raw EXFOR identity in NATURAL_KEY, and the survivor of
    each group is the one with evaluation data — falling back to the lowest rowid when the
    copies are equally complete. Pass ``natural=False`` for the stricter identical-in-every-
    column rule.
    """
    columns = [c for c in table_columns(con, table) if c != "rowid"]
    before = count_rows(con, table)
    emit(f"  {before:,} rows, {len(columns)} columns")

    if not natural:
        group = ", ".join(f'"{c}"' for c in columns)
        order = "MIN(rowid)"
    else:
        present = [c for c in NATURAL_KEY if c in columns]
        if len(present) < 3:
            raise ValueError(
                f"cannot identify measurements: {table} lacks {set(NATURAL_KEY) - set(columns)}")
        group = ", ".join(f'"{c}"' for c in present)
        # Prefer a row that carries evaluation data; break ties by rowid so the choice is
        # deterministic rather than whatever sqlite happens to return.
        completeness = " + ".join(
            f'(CASE WHEN "{c}" IS NOT NULL THEN 1 ELSE 0 END)'
            for c in EVALUATION_COLUMNS if c in columns) or "0"
        order = f"(SELECT rowid FROM {table} t2 WHERE " + " AND ".join(
            f't2."{c}" IS {table}."{c}" ' for c in present) + \
            f"ORDER BY ({completeness}) DESC, rowid ASC LIMIT 1)"
        emit(f"  grouping on {', '.join(present)}; keeping the copy with evaluation data")

    started = time.monotonic()
    if natural:
        # Materialise the keepers first: a correlated subquery per row is far too slow on
        # a table of this size.
        con.execute("DROP TABLE IF EXISTS _dedup_keep")
        con.execute(f"""
            CREATE TEMP TABLE _dedup_keep AS
            SELECT rowid AS keep FROM (
                SELECT rowid, ROW_NUMBER() OVER (
                    PARTITION BY {group}
                    ORDER BY ({completeness}) DESC, rowid ASC) AS rank
                FROM {table}
            ) WHERE rank = 1
        """)
        con.execute(f"DELETE FROM {table} WHERE rowid NOT IN (SELECT keep FROM _dedup_keep)")
        con.execute("DROP TABLE IF EXISTS _dedup_keep")
    else:
        con.execute(f"""
            DELETE FROM {table}
            WHERE rowid NOT IN (SELECT MIN(rowid) FROM {table} GROUP BY {group})
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
    parser.add_argument("--exact", action="store_true",
                        help="only remove rows identical in every column, rather than "
                             "grouping on the raw EXFOR identity")
    args = parser.parse_args()

    source = Path(args.database)
    if not source.is_file():
        print(f"no database at {source}", file=sys.stderr)
        return 2

    if args.dry_run:
        con = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
        try:
            columns = [c for c in table_columns(con, "measurements") if c != "rowid"]
            present = [c for c in NATURAL_KEY if c in columns]
            total = count_rows(con)
            identical = count_distinct(con, columns)
            physical = count_distinct(con, present)
            incomplete = con.execute(
                'SELECT COUNT(*) FROM measurements WHERE "endf8" IS NULL').fetchone()[0]
        finally:
            con.close()
        print(f"{source}")
        print(f"  rows                       {total:,}")
        print(f"  identical in every column  {identical:,}"
              f"   ({total - identical:,} exact duplicates)")
        print(f"  distinct measurements      {physical:,}"
              f"   ({total - physical:,} repeat ingests)")
        if total and physical:
            print(f"  ratio                      {total / physical:.4f}x")
        print(f"  rows with no endf8         {incomplete:,}")
        if total > physical > identical - 1:
            print("\n  The repeats differ in their derived columns, so an exact-match pass")
            print("  would remove almost none of them. The natural-key pass keeps the copy")
            print("  carrying evaluation data.")
        print("\nnothing was changed (--dry-run)")
        return 0

    target = source if args.in_place else Path(args.out or f"{source}.deduped")
    if not args.in_place:
        print(f"copying {source} -> {target}")
        shutil.copyfile(source, target)

    con = sqlite3.connect(target)
    try:
        print(f"deduplicating {target}")
        before, after = deduplicate(con, natural=not args.exact)
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
