"""Repair derived uncertainty/chi-squared columns in an existing nugrade_data.db.

Two bugs in 1_raw_data_ingestion.ipynb left the shipped database with unphysical values:

  1. Chi-squared was computed as (Data - eval)^2 / sigma instead of
     ((Data - eval) / sigma)^2. The stored column divides by the standard deviation
     rather than the variance, so it is not a chi-squared and is not dimensionless.

  2. Uncertainties were derived from Data without taking the magnitude. EXFOR contains
     genuinely negative cross sections (background-subtraction artifacts), so ~52k rows
     produced negative uncertainties, which then propagated into negative chi-squared.

Both are recomputable **from the database alone** — the interpolated evaluation cross
sections are already stored in the `endf8` / `endf7-1` columns — so this does NOT require
re-running the ingestion notebook against the group cluster's ENDF files.

This script never modifies its input. It copies the database and repairs the copy:

    python repair_derived_columns.py output/nugrade_data.db output/nugrade_data_fixed.db

Verify the result with `python validate_output_db.py output/nugrade_data_fixed.db`, then
re-run 3_knn_imputation.ipynb against the repaired file to redo the KNN imputation on
corrected inputs.
"""
import shutil
import sqlite3
import sys

EVALUATIONS = ["endf8", "endf7-1"]


def repair(src_path, dst_path):
    shutil.copyfile(src_path, dst_path)
    con = sqlite3.connect(dst_path)

    columns = {r[1] for r in con.execute("PRAGMA table_info(measurements)")}
    changes = []

    # 1. Uncertainties are magnitudes.
    for column in ["dData", "dData_assumed", "dData_adopted"]:
        if column not in columns:
            continue
        n = con.execute(f'SELECT COUNT(*) FROM measurements WHERE "{column}" < 0').fetchone()[0]
        if n:
            con.execute(f'UPDATE measurements SET "{column}" = ABS("{column}") WHERE "{column}" < 0')
            changes.append(f"{column}: took magnitude of {n} negative values")

    # 2. Chi-squared is ((observed - expected) / sigma)^2.
    #    dData_assumed is the per-point sigma actually used (dData with the quantile
    #    fallback filled in), matching what the ingestion notebook intended.
    sigma = "dData_assumed" if "dData_assumed" in columns else "dData"
    for evaluation in EVALUATIONS:
        chi_column = f"{evaluation}_chi_squared"
        if chi_column not in columns or evaluation not in columns:
            continue
        con.execute(
            f'''UPDATE measurements
                SET "{chi_column}" = CASE
                    WHEN "{evaluation}" IS NULL OR "{sigma}" IS NULL OR "{sigma}" = 0 THEN NULL
                    ELSE ((Data - "{evaluation}") / "{sigma}") * ((Data - "{evaluation}") / "{sigma}")
                END'''
        )
        changes.append(f"{chi_column}: recomputed as ((Data - {evaluation}) / {sigma})^2")

    con.commit()

    # Report what the repaired file looks like.
    print(f"Repaired copy written to {dst_path}")
    for change in changes:
        print(f"  - {change}")

    print("\nRemaining negative values (should all be 0):")
    for column in ["dData", "dData_assumed", "dData_adopted",
                   "endf8_chi_squared", "endf7-1_chi_squared"]:
        if column not in columns:
            continue
        n = con.execute(f'SELECT COUNT(*) FROM measurements WHERE "{column}" < 0').fetchone()[0]
        print(f"  {column}: {n}")
    con.close()


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        print("usage: python repair_derived_columns.py <source.db> <destination.db>")
        sys.exit(2)
    src, dst = sys.argv[1], sys.argv[2]
    if src == dst:
        print("Refusing to write over the source database; choose a different destination.")
        sys.exit(2)
    repair(src, dst)


if __name__ == "__main__":
    main()
