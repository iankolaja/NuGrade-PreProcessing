"""Copy the report/sentence embedding tables from one database into another.

    python graft_embedding_tables.py <source.db> <destination.db>

Re-running 1_raw_data_ingestion.ipynb rebuilds `measurements`, `subentries` and `entries`
in a fresh output database, which therefore has no embedding tables. Regenerating them
means re-running 2_report_embedding.ipynb, which needs the report PDFs.

That is avoidable: `report_embeddings` and `sentence_embeddings` are keyed only on
EXFOR_Entry and never reference individual measurement rows, so they are unchanged by an
ingestion re-run and can simply be copied across. Grafting them lets you go straight from
a rebuilt database to 3_knn_imputation.ipynb.

Only the destination is modified; existing tables of the same name there are replaced.
"""
import sqlite3
import sys

TABLES = ["report_embeddings", "sentence_embeddings"]


def graft(src_path, dst_path):
    src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
    dst = sqlite3.connect(dst_path)

    src_tables = {r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing = [t for t in TABLES if t not in src_tables]
    if missing:
        print(f"Source has no {', '.join(missing)} — nothing to graft.")
        src.close()
        dst.close()
        sys.exit(1)

    for table in TABLES:
        schema = src.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name = ?", (table,)
        ).fetchone()[0]

        dst.execute(f"DROP TABLE IF EXISTS {table}")
        dst.execute(schema)

        rows = src.execute(f"SELECT * FROM {table}").fetchall()
        if rows:
            placeholders = ",".join("?" * len(rows[0]))
            dst.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)
        print(f"  {table}: copied {len(rows):,} rows")

    dst.commit()

    print("\nVerifying:")
    for table in TABLES:
        n = dst.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {n:,} rows in destination")

    # The embeddings are only useful for entries that survive in the new measurements.
    dst_tables = {r[0] for r in dst.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "measurements" in dst_tables:
        orphans = dst.execute(
            """SELECT COUNT(DISTINCT EXFOR_Entry) FROM sentence_embeddings
               WHERE EXFOR_Entry NOT IN (SELECT DISTINCT EXFOR_Entry FROM measurements)"""
        ).fetchone()[0]
        if orphans:
            print(f"  note: {orphans} embedded entries have no measurements in the "
                  "destination (harmless, but they cannot be used as KNN candidates)")

    src.close()
    dst.close()


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        print("usage: python graft_embedding_tables.py <source.db> <destination.db>")
        sys.exit(2)
    if sys.argv[1] == sys.argv[2]:
        print("Source and destination are the same file.")
        sys.exit(2)
    graft(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
