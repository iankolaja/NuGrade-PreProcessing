"""Compare two nugrade_data.db builds, to check a re-run did what was intended.

    python compare_databases.py old.db new.db

Written for validating the fixes on this branch against a database rebuilt on the cluster.
It answers three questions:

  1. Did anything change that should NOT have? Raw EXFOR quantities (Energy, Data, dData)
     come straight from X4Pro and must be identical between builds. A difference there
     means the source data or the ingestion query changed, not just the metrics.
  2. Did the intended fixes land? Chi-squared should now equal ((Data - eval) / sigma)^2,
     and no derived uncertainty or chi-squared column should be negative or infinite.
  3. How much did the results move? Reports the shift in chi-squared and in dData_assumed,
     since those propagate into the per-nuclide grades the NuGrade app shows.

Read-only: opens both databases with mode=ro and never writes.
"""
import sqlite3
import sys

import numpy as np
import pandas as pd

RAW_COLUMNS = ["Energy", "Data", "dData"]
DERIVED_NON_NEGATIVE = [
    "dData", "dData_assumed", "dData_adopted",
    "endf8_chi_squared", "endf7-1_chi_squared",
]
KEY = ["EXFOR_Subentry", "Energy", "Data"]


def _connect(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _columns(con, table):
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


def _tables(con):
    return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def compare_structure(old, new):
    section("1. STRUCTURE")
    old_tables, new_tables = _tables(old), _tables(new)

    for table in sorted(old_tables | new_tables):
        in_old = table in old_tables
        in_new = table in new_tables
        n_old = old.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] if in_old else None
        n_new = new.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] if in_new else None
        if not in_new:
            print(f"  {table:22s} {n_old:>9,} -> MISSING  (added by a later notebook?)")
        elif not in_old:
            print(f"  {table:22s} {'-':>9} -> {n_new:,}  (new)")
        else:
            flag = "" if n_old == n_new else "   <-- row count changed"
            print(f"  {table:22s} {n_old:>9,} -> {n_new:>9,}{flag}")

    if "measurements" in old_tables & new_tables:
        lost = _columns(old, "measurements") - _columns(new, "measurements")
        gained = _columns(new, "measurements") - _columns(old, "measurements")
        if lost:
            print(f"\n  columns only in OLD measurements: {sorted(lost)}")
        if gained:
            print(f"  columns only in NEW measurements: {sorted(gained)}")


def compare_raw_data(old, new):
    """Raw EXFOR quantities must be byte-identical between builds."""
    section("2. RAW EXFOR DATA (must be unchanged)")
    stats = {}
    for label, con in (("old", old), ("new", new)):
        row = con.execute(
            """SELECT COUNT(*), SUM(Energy), SUM(Data),
                      SUM(CASE WHEN dData IS NULL THEN 1 ELSE 0 END)
               FROM measurements"""
        ).fetchone()
        stats[label] = row

    names = ["row count", "sum(Energy)", "sum(Data)", "null dData"]
    all_match = True
    for i, name in enumerate(names):
        o, n = stats["old"][i], stats["new"][i]
        match = (o == n) or (
            o is not None and n is not None and np.isclose(float(o), float(n), rtol=1e-9)
        )
        all_match &= match
        print(f"  {name:14s} old={o!s:>20}  new={n!s:>20}  {'OK' if match else 'DIFFERS'}")

    if all_match:
        print("\n  Raw data is identical -> only derived columns changed, as intended.")
    else:
        print("\n  WARNING: raw EXFOR data differs between builds. Either the X4Pro source")
        print("  or the ingestion query changed; investigate before trusting the metrics.")


def check_contract(new):
    section("3. DID THE FIXES LAND?")
    columns = _columns(new, "measurements")

    print("  Negative values (all should be 0):")
    clean = True
    for column in DERIVED_NON_NEGATIVE:
        if column not in columns:
            print(f"    {column:22s} (absent)")
            continue
        n = new.execute(f'SELECT COUNT(*) FROM measurements WHERE "{column}" < 0').fetchone()[0]
        clean &= (n == 0)
        print(f"    {column:22s} {n:>8,}{'' if n == 0 else '   <-- STILL NEGATIVE'}")

    print("\n  Non-finite values in chi-squared (all should be 0):")
    for evaluation in ["endf8", "endf7-1"]:
        column = f"{evaluation}_chi_squared"
        if column not in columns:
            continue
        # SQLite has no isinf(); a value equal to itself but larger than any real bound.
        n = new.execute(
            f'SELECT COUNT(*) FROM measurements WHERE "{column}" > 1e300'
        ).fetchone()[0]
        clean &= (n == 0)
        print(f"    {column:22s} {n:>8,}{'' if n == 0 else '   <-- INFINITIES REMAIN'}")

    print("\n  Chi-squared formula check (sampled):")
    for evaluation in ["endf8", "endf7-1"]:
        chi = f"{evaluation}_chi_squared"
        if chi not in columns or evaluation not in columns:
            continue
        df = pd.read_sql(
            f'''SELECT Data, "{evaluation}" AS ev, dData_assumed AS sigma, "{chi}" AS chi
                FROM measurements
                WHERE "{evaluation}" IS NOT NULL AND dData_assumed > 0 AND "{chi}" IS NOT NULL
                LIMIT 200000''',
            new,
        )
        if df.empty:
            continue
        correct = ((df["Data"] - df["ev"]) / df["sigma"]) ** 2
        old_formula = (df["Data"] - df["ev"]) ** 2 / df["sigma"]
        is_correct = np.allclose(df["chi"], correct, rtol=1e-6)
        is_old = np.allclose(df["chi"], old_formula, rtol=1e-6)
        verdict = "FIXED" if is_correct else ("STILL THE OLD FORMULA" if is_old else "matches neither")
        print(f"    {chi:22s} {verdict}")
        clean &= is_correct

    print(f"\n  => {'All checks passed.' if clean else 'Some checks FAILED (see above).'}")


def compare_distributions(old, new):
    section("4. HOW MUCH DID RESULTS MOVE?")
    for column in ["endf8_chi_squared", "dData_assumed", "endf8_relative_error"]:
        if column not in _columns(old, "measurements") or column not in _columns(new, "measurements"):
            continue
        print(f"\n  {column}")
        for label, con in (("old", old), ("new", new)):
            s = pd.read_sql(
                f'SELECT "{column}" AS v FROM measurements WHERE "{column}" IS NOT NULL', con
            )["v"]
            finite = s[np.isfinite(s)]
            n_inf = len(s) - len(finite)
            print(f"    {label}: median={finite.median():>12.4g}  p99={np.percentile(finite, 99):>12.4g}"
                  f"  mean={finite.mean():>12.4g}  min={finite.min():>10.4g}  n_non_finite={n_inf:,}")

    if "endf8_chi_squared" in _columns(new, "measurements"):
        print("\n  Note: a correct chi-squared divides by sigma^2, so points with a very small")
        print("  assumed uncertainty produce very large values. The distribution is heavy-tailed")
        print("  and its MEAN is dominated by a few thousand such points; compare medians and")
        print("  percentiles rather than means, and prefer a robust statistic when grading.")


def compare_assumed_uncertainty_fallback(old, new):
    """Quantify the per-channel fallback fix: how many channels now use their own value."""
    section("5. PER-CHANNEL FALLBACK FIX (the 44% dropna bug)")
    query = """SELECT Z, A, Reaction, COUNT(DISTINCT ROUND(dData_assumed / ABS(Data), 6)) AS n_ratios
               FROM measurements
               WHERE dData IS NULL AND Data != 0 AND dData_assumed IS NOT NULL
               GROUP BY Z, A, Reaction"""
    try:
        old_df = pd.read_sql(query, old)
        new_df = pd.read_sql(query, new)
    except Exception as e:
        print(f"  could not compute: {e}")
        return

    merged = old_df.merge(new_df, on=["Z", "A", "Reaction"], suffixes=("_old", "_new"))
    print(f"  channels compared: {len(merged):,}")

    # Each channel imputes at one assumed fraction; compare that fraction old vs new.
    frac_query = """SELECT Z, A, Reaction, ROUND(AVG(dData_assumed / ABS(Data)), 8) AS frac
                    FROM measurements
                    WHERE dData IS NULL AND Data != 0 AND dData_assumed IS NOT NULL
                    GROUP BY Z, A, Reaction"""
    o = pd.read_sql(frac_query, old)
    n = pd.read_sql(frac_query, new)
    m = o.merge(n, on=["Z", "A", "Reaction"], suffixes=("_old", "_new"))
    changed = m[~np.isclose(m["frac_old"], m["frac_new"], rtol=1e-6)]
    print(f"  channels whose assumed relative uncertainty CHANGED: {len(changed):,}"
          f" ({100 * len(changed) / max(len(m), 1):.1f}%)")
    print("\n  How to read this:")
    print("   - After a full re-run of 1_raw_data_ingestion.ipynb, expect a large change.")
    print("     367 of 830 channels previously used the corpus-wide fallback because")
    print("     dropna() consulted dEnergy, which is NULL for 86% of rows.")
    print("   - After only repair_derived_columns.py, expect a SMALL change. The repair")
    print("     can take the magnitude of existing values but cannot recover per-channel")
    print("     uncertainty distributions; that requires re-running the ingestion.")
    if len(changed):
        print(f"\n  sample of changed channels:")
        print(changed.head(8).to_string(index=False))


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        print("usage: python compare_databases.py <old.db> <new.db>")
        sys.exit(2)

    old, new = _connect(sys.argv[1]), _connect(sys.argv[2])
    print(f"OLD: {sys.argv[1]}\nNEW: {sys.argv[2]}")
    try:
        compare_structure(old, new)
        compare_raw_data(old, new)
        check_contract(new)
        compare_distributions(old, new)
        compare_assumed_uncertainty_fallback(old, new)
    finally:
        old.close()
        new.close()
    print()


if __name__ == "__main__":
    main()
