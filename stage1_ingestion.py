"""Stage 1: ingest X4Pro measurements and compare them against evaluated cross sections.

    python stage1_ingestion.py                    # both phases
    python stage1_ingestion.py --phase aggregates # summary tables only

Two phases, separated by a database round-trip:

  measurements — read X4Pro, and for each (nuclide, reaction channel) interpolate the
                 evaluation cross sections onto the measured energies and compute the
                 comparison metrics. Needs cluster ACE files. Runs for hours.
  aggregates   — re-read the measurements table and build the per-subentry and per-entry
                 summaries. Needs only the database. Runs in minutes.

Keeping the round-trip is deliberate: it is what makes the aggregates independently
re-runnable, which matters after a repair pass over the derived columns.

The arithmetic lives in `ingestion.py` and is tested there. This module is the
orchestration: the loop, the evaluation lookups, resume, and persistence.
"""
import argparse
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from evaluations import readers_from_config
from helper_functions import get_A, get_element, get_z
from ingestion import compute_channel_metrics, relative_uncertainties
from pipeline_config import Config, ConfigError, add_config_arguments, check_or_raise
from stage_result import ProgressTracker, StageResult, null_printer, printer

X4_QUERY = """
    SELECT Reaction, Projectile, En, dEn, Sig, dSig, MT, DatasetID, Entry, Subent,
           YearRef1, Author1Ini, Author1, Target, fullCode
    FROM sig1
"""

X4_DTYPES = {
    "Reaction": str, "Projectile": str, "MT": np.int16, "Target": str,
    "En": np.float64, "dEn": np.float64, "Sig": np.float64, "dSig": np.float64,
    "fullCode": str, "YearRef1": np.int16, "Author1Ini": str, "Author1": str,
    "DatasetID": str, "Subent": str, "Entry": str,
}

X4_RENAMES = {
    "En": "Energy", "dEn": "dEnergy", "Subent": "EXFOR_Subentry", "Entry": "EXFOR_Entry",
    "DatasetID": "Dataset_Number", "YearRef1": "Year", "Sig": "Data", "dSig": "dData",
}

# Doubles as the reaction summary. Column names come from this schema, so the header can no
# longer drift from the values it labels — which is how MT and Reaction ended up swapped in
# the notebook's all_reactions.csv.
PROGRESS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS ingest_progress (
        label TEXT PRIMARY KEY,
        Z INTEGER, A INTEGER, Symbol TEXT, Projectile TEXT,
        MT INTEGER, Reaction TEXT,
        "has_endf7-1" INTEGER, has_endf8 INTEGER,
        n_rows INTEGER, completed_at TEXT
    )
"""

SUMMARY_COLUMNS = ["Z", "A", "Symbol", "Projectile", "MT", "Reaction",
                   "has_endf7-1", "has_endf8"]


def load_exfor_frame(x4_db, emit):
    """Read X4Pro and derive the columns the rest of the pipeline expects."""
    con = sqlite3.connect(f"file:{x4_db}?mode=ro", uri=True)
    try:
        frame = pd.read_sql_query(X4_QUERY, con, dtype=X4_DTYPES)
    finally:
        con.close()
    emit(f"read {len(frame):,} rows from {x4_db}")

    frame["A"] = frame["Target"].map(get_A)
    frame["Element"] = frame["Target"].map(get_element)
    frame["Z"] = frame["Element"].map(get_z)
    frame["Author"] = frame["Author1Ini"] + frame["Author1"]
    return frame.rename(columns=X4_RENAMES)


def corpus_fallback_uncertainty(frame, quantile=0.90):
    """Corpus-wide relative uncertainty, used only where a channel has none of its own.

    The notebook held this in a global that `compute_channel_data` closed over, which is
    what made that function impossible to test. It is now passed explicitly.
    """
    return float(np.quantile(relative_uncertainties(frame["Data"], frame["dData"]), quantile))


def channel_label(projectile, Z, A, reaction):
    """Stable identifier for one (nuclide, channel), used for resume and eval filenames."""
    return f"{projectile}_{Z}_{A}_{reaction}"


def completed_labels(db_path):
    """Channels already ingested, so an interrupted run resumes where it stopped."""
    if not Path(db_path).is_file():
        return set()
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "ingest_progress" not in tables:
            return set()
        return {r[0] for r in con.execute("SELECT label FROM ingest_progress")}
    finally:
        con.close()


def write_channel(con, label, channel_data, summary):
    """Persist one channel's rows and its progress marker in a single transaction.

    Atomicity here is what guarantees `measurements` and `ingest_progress` can never
    disagree — so on resume, a channel is either fully present or absent.
    """
    channel_data.to_sql("measurements", con, if_exists="append", index=False)
    # The progress schema has a column per evaluation, but the reader list is configurable
    # (a run may use one library, or none). An evaluation nobody looked for is recorded as
    # absent rather than raising.
    con.execute(
        'INSERT OR REPLACE INTO ingest_progress '
        '(label, Z, A, Symbol, Projectile, MT, Reaction, "has_endf7-1", has_endf8, '
        ' n_rows, completed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
        (label, int(summary["Z"]), int(summary["A"]), summary["Symbol"],
         summary["Projectile"], int(summary["MT"]), summary["Reaction"],
         int(bool(summary.get("has_endf7-1", False))),
         int(bool(summary.get("has_endf8", False))),
         len(channel_data), datetime.now(timezone.utc).isoformat(timespec="seconds")))
    con.commit()


def write_reaction_summary(db_path, output_dir):
    """Write all_reactions.csv from the progress table.

    Sourcing it from the database rather than an in-memory list means a crashed run still
    leaves a summary of the work that completed, and the column order is fixed by the
    schema.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        summary = pd.read_sql(
            f'SELECT {", ".join(chr(34) + c + chr(34) for c in SUMMARY_COLUMNS)} '
            'FROM ingest_progress ORDER BY Z, A, MT', con)
    finally:
        con.close()
    path = Path(output_dir) / "all_reactions.csv"
    summary.to_csv(path, index=False)
    return path, len(summary)


def run_measurements(config, *, readers=None, exfor_df=None, progress=None):
    """Ingest X4Pro and compute evaluation-comparison metrics per channel."""
    emit = progress or printer("1")
    started = time.monotonic()
    check_or_raise(config, "1")
    config.ensure_directories()

    readers = readers if readers is not None else readers_from_config(config)
    frame = exfor_df if exfor_df is not None else load_exfor_frame(config.x4_db, emit)

    fallback = corpus_fallback_uncertainty(frame)
    emit(f"corpus fallback relative uncertainty: {fallback:.1%}")

    nuclides = frame.drop_duplicates(subset=["Z", "A"])[["Z", "A", "Element"]]
    nuclides = nuclides[(nuclides["Z"] >= 1) & (nuclides["A"] >= 1)]

    con = sqlite3.connect(config.db_path)
    con.execute(PROGRESS_SCHEMA)
    con.commit()

    done = completed_labels(config.db_path) if config.resume else set()
    if not config.resume:
        con.execute("DROP TABLE IF EXISTS measurements")
        con.execute("DELETE FROM ingest_progress")
        con.commit()
    elif done:
        emit(f"resuming: {len(done)} channels already ingested")

    work = [(row.Z, row.A, row.Element, reaction, mt)
            for row in nuclides.itertuples()
            for reaction, mt in config.reaction_channels.items()]
    if config.limit:
        work = work[:config.limit]

    warnings = []
    counts = {"channels": 0, "measurements": 0, "skipped_existing": 0, "no_data": 0}
    tracker = ProgressTracker(len(work), emit, every=config.log_every, unit="channels")

    try:
        for Z, A, symbol, reaction, mt in work:
            projectile = reaction[0].lower()
            label = channel_label(projectile, Z, A, reaction)
            if label in done:
                counts["skipped_existing"] += 1
                tracker.advance()
                continue

            channel_data = frame[(frame["Z"] == Z) & (frame["A"] == A)
                                 & (frame["Reaction"] == reaction)]
            if channel_data.empty:
                counts["no_data"] += 1
                tracker.advance()
                continue

            channel_data = channel_data.sort_values("Energy")
            energies = channel_data["Energy"].to_numpy()
            zaid = f"{Z}{str(A).zfill(3)}"
            summary = {"Z": Z, "A": A, "Symbol": symbol, "Projectile": projectile,
                       "MT": mt, "Reaction": reaction}

            for reader in readers:
                try:
                    evaluation = reader.read(symbol, zaid, mt, energies)
                except Exception as error:      # one bad nuclide must not end a long run
                    warnings.append(f"{label}: {reader.name} read failed ({error})")
                    evaluation = None
                summary[f"has_{reader.name}"] = evaluation is not None
                channel_data = compute_channel_metrics(
                    channel_data, reader.name,
                    evaluation.interpolated if evaluation else None, fallback)
                if evaluation is not None:
                    grid_path = (Path(config.output_dir) / "evals" / reader.name
                                 / f"{label}_grid_data.csv")
                    pd.DataFrame({"grid_energy(eV)": evaluation.grid_energy,
                                  "grid_xs(b)": evaluation.grid_xs}).to_csv(
                        grid_path, index=False, na_rep="nan")

            write_channel(con, label, channel_data, summary)
            counts["channels"] += 1
            counts["measurements"] += len(channel_data)
            tracker.advance()
    finally:
        con.close()

    tracker.finish()
    summary_path, n_channels = write_reaction_summary(config.db_path, config.output_dir)
    emit(f"wrote {n_channels} channels to {summary_path}")

    return StageResult(
        stage="1", ok=True, counts=counts, warnings=warnings,
        outputs=[Path(config.db_path), summary_path],
        elapsed_s=time.monotonic() - started)


def summarise_subentries(measurements):
    """One row per EXFOR subentry: energy span and uncertainty completeness."""
    grouped = measurements.groupby("EXFOR_Subentry", sort=False)
    return pd.DataFrame({
        "Z": grouped["Z"].first(), "A": grouped["A"].first(), "MT": grouped["MT"].first(),
        "Reaction": grouped["Reaction"].first(), "Element": grouped["Element"].first(),
        "EXFOR_Entry": grouped["EXFOR_Entry"].first(),
        "Dataset_Number": grouped["Dataset_Number"].first(),
        "E_min": grouped["Energy"].min(), "E_max": grouped["Energy"].max(),
        "E_median": grouped["Energy"].median(),
        "num_measurements": grouped.size(),
        "num_measurements_uncertainty": grouped["dData"].count(),
    }).reset_index()


def summarise_entries(measurements):
    """One row per EXFOR entry, aggregating its subentries."""
    def joined(series):
        return ",".join(series.astype(str).unique())

    grouped = measurements.groupby("EXFOR_Entry", sort=False)
    frame = pd.DataFrame({
        "Z_Values": grouped["Z"].apply(joined),
        "Elements": grouped["Element"].apply(joined),
        "A_Values": grouped["A"].apply(joined),
        "MT_Codes": grouped["MT"].apply(joined),
        "Reactions": grouped["Reaction"].apply(lambda s: "-".join(s.astype(str).unique())),
        "Num_Subentries": grouped["EXFOR_Subentry"].nunique(),
        "Num_Measurements": grouped.size(),
        "Num_Missing_Uncertainty": grouped["dData"].apply(lambda s: int(s.isna().sum())),
    }).reset_index()
    frame["Uncertainty_Complete"] = (frame["Num_Missing_Uncertainty"] == 0).astype(int)
    return frame


def run_aggregates(config, *, progress=None):
    """Build the subentry and entry summary tables from the measurements table."""
    emit = progress or printer("1b")
    started = time.monotonic()
    check_or_raise(config, "1b")

    con = sqlite3.connect(config.db_path)
    try:
        measurements = pd.read_sql("SELECT * FROM measurements", con)
        emit(f"read {len(measurements):,} measurements")

        subentries = summarise_subentries(measurements)
        subentries.to_sql("subentries", con, if_exists="replace", index=False)
        emit(f"wrote {len(subentries):,} subentries")

        entries = summarise_entries(measurements)
        entries.to_sql("entries", con, if_exists="replace", index=False)
        emit(f"wrote {len(entries):,} entries")
        con.commit()
    finally:
        con.close()

    return StageResult(
        stage="1b", ok=True,
        counts={"subentries": len(subentries), "entries": len(entries),
                "entries_complete": int(entries["Uncertainty_Complete"].sum())},
        outputs=[Path(config.db_path)], elapsed_s=time.monotonic() - started)


def run(config, *, readers=None, progress=None):
    """Both phases, as the notebook ran them."""
    first = run_measurements(config, readers=readers, progress=progress)
    if not first.ok:
        return first
    return first.merge(run_aggregates(config, progress=progress))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arguments(parser)
    parser.add_argument("--phase", choices=["all", "measurements", "aggregates"],
                        default="all", help="which phase to run (default: all)")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    args = parser.parse_args()

    config = Config.from_args(args)
    emit = null_printer if args.quiet else None
    try:
        if args.phase == "measurements":
            result = run_measurements(config, progress=emit)
        elif args.phase == "aggregates":
            result = run_aggregates(config, progress=emit)
        else:
            result = run(config, progress=emit)
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 2
    print(result.render())
    return result.exit_code()


if __name__ == "__main__":
    sys.exit(main())
