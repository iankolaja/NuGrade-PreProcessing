"""Validate output/nugrade_data.db against the contract the NuGrade Flask app expects.

The database is the interface between this repo (producer) and the NuGrade web app
(consumer, which enforces the same contract at startup in nugrade/db_contract.py).
Run this after the notebooks finish, before shipping the DB:

    python validate_output_db.py [path/to/nugrade_data.db]

Read-only: this script never modifies the database. The source ENDF data lives on
the group cluster and is not casually reproducible, so validation failing loudly
here is much cheaper than discovering a broken DB after deployment.

Keep REQUIRED_SCHEMA and the embedding constants in sync with
NuGrade/nugrade/db_contract.py — if you change the schema intentionally,
update both repos in the same change.
"""
import sqlite3
import sys

import numpy as np

REQUIRED_SCHEMA = {
    "measurements": {
        "Z", "A", "MT", "Projectile", "Reaction", "Element",
        "Energy", "dEnergy", "Data", "dData", "dData_assumed", "dData_adopted",
        "EXFOR_Entry", "EXFOR_Subentry", "Dataset_Number", "Year", "Author",
        "endf8", "endf8_chi_squared", "endf8_relative_error",
        "endf7-1", "endf7-1_chi_squared", "endf7-1_relative_error",
    },
    "subentries": {
        "Z", "A", "MT", "Reaction", "Element",
        "EXFOR_Entry", "EXFOR_Subentry", "E_min", "E_max",
    },
    "sentence_embeddings": {
        "EXFOR_Entry", "Sentence_Number", "Text", "Embedding",
    },
}

# Embedding contract: SciBERT, attention-mask-weighted MEAN pooling (get_embeddings_batch
# in 2_report_embedding.ipynb). The NuGrade app embeds queries the same way; if the
# pooling here ever changes, its NuclearDataAgent._embed must change with it.
EMBEDDING_DIM = 768
EMBEDDING_DTYPE = np.float32


def validate(db_path):
    problems, warnings = [], []
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    present = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}

    for table, required in REQUIRED_SCHEMA.items():
        if table not in present:
            problems.append(f"missing table '{table}'")
            continue
        columns = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        for column in sorted(required - columns):
            problems.append(f"table '{table}' is missing column '{column}'")
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if n == 0:
            problems.append(f"table '{table}' is empty")

    if "sentence_embeddings" in present:
        row = con.execute(
            "SELECT Embedding FROM sentence_embeddings WHERE Embedding IS NOT NULL LIMIT 1"
        ).fetchone()
        if row is None:
            problems.append("sentence_embeddings has no non-null embeddings")
        else:
            vec = np.frombuffer(row[0], dtype=EMBEDDING_DTYPE)
            if vec.shape != (EMBEDDING_DIM,):
                problems.append(
                    f"embeddings are {vec.shape[0]}-dim under "
                    f"{EMBEDDING_DTYPE.__name__}, expected {EMBEDDING_DIM}"
                )

    if "measurements" in present:
        problems.extend(_verify_physical_ranges(con))

    if "reports" in present:
        warnings.append(
            "stale 'reports' table present (current notebooks write "
            "'report_embeddings'); consider dropping it before shipping"
        )

    con.close()
    return problems, warnings


def _verify_physical_ranges(con):
    """Check quantities that are non-negative by definition.

    An uncertainty is a width and a chi-squared is a sum of squares, so neither can be
    negative. EXFOR does contain genuinely negative cross sections (background-subtraction
    artifacts), so `Data` is deliberately not checked — but anything *derived* from it
    must take the magnitude, or the sign leaks into uncertainties and chi-squared.
    """
    problems = []
    non_negative_columns = [
        "dData", "dData_assumed", "dData_adopted",
        "endf8_chi_squared", "endf7-1_chi_squared",
    ]
    columns = {r[1] for r in con.execute("PRAGMA table_info(measurements)")}
    for column in non_negative_columns:
        if column not in columns:
            continue
        n = con.execute(
            f'SELECT COUNT(*) FROM measurements WHERE "{column}" < 0'
        ).fetchone()[0]
        if n:
            problems.append(
                f"'{column}' has {n} negative values; this quantity is non-negative "
                "by definition (likely a negative cross section propagated without abs())"
            )
    return problems


def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else "output/nugrade_data.db"
    problems, warnings = validate(db_path)

    for w in warnings:
        print(f"WARNING: {w}")
    if problems:
        print(f"FAILED: {db_path} does not satisfy the NuGrade app contract:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print(f"OK: {db_path} satisfies the NuGrade app contract.")


if __name__ == "__main__":
    main()
