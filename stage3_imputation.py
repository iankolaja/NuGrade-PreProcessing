"""Stage 3: impute missing experimental uncertainties by KNN over report similarity.

    python stage3_imputation.py --db output/nugrade_data.db

For each measurement with no reported uncertainty, finds the most similar complete
measurements from other experiments — by report-text similarity features plus standardised
Z, A and energy — and adopts an inverse-distance-weighted mean of their relative
uncertainties, scaled by the measurement's own cross section.

The arithmetic lives in `imputation.py` and is tested there. This module is the
orchestration: loading, candidate selection, the loop, and persistence.

Reads `measurements`, `entries`, `report_embeddings`; rewrites `measurements` with
`dData_adopted` and `uncertainty_source` filled in.
"""
import argparse
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from imputation import (
    SCALAR_FEATURES,
    SIMILARITY_LABELS,
    absolute_uncertainty,
    composite_distance,
    representative_candidate,
    standardize_features,
    weighted_mean_relative_uncertainty,
)
from pipeline_config import Config, add_config_arguments, check_or_raise, ConfigError
from stage_result import ProgressTracker, StageResult, null_printer, printer, step

# pandas writes these when a frame is saved with its index; the notebook did exactly that,
# so re-running against its own output accumulated a new one each time.
INDEX_ARTEFACT_COLUMNS = ("index", "level_0")

REACTION_CHANNEL_INDEX = """
    CREATE INDEX IF NOT EXISTS idx_reaction_channel
    ON measurements (Z, A, MT, Projectile)
"""


def default_feature_weights():
    """Equal weight over similarity features, standardised scalars, and the embedding."""
    return pd.Series(
        {label: 1.0 for label in SIMILARITY_LABELS}
        | {feature: 1.0 for feature in SCALAR_FEATURES}
        | {"mean_embedding": 1.0}
    )


def load_inputs(db_path):
    """Read the tables stage 3 needs, and drop any accumulated index artefacts."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        entries = pd.read_sql("SELECT * FROM entries", con)
        reports = pd.read_sql("SELECT * FROM report_embeddings", con)
        # 2.6 M rows; the slowest silent step in this stage.
        measurements = pd.read_sql("SELECT * FROM measurements", con)
    finally:
        con.close()

    measurements = measurements.drop(
        columns=[c for c in INDEX_ARTEFACT_COLUMNS if c in measurements.columns])
    reports["mean_embedding"] = reports["mean_embedding"].apply(
        lambda blob: np.frombuffer(blob, dtype=np.float32))
    return measurements, reports, entries


def prepare(measurements):
    """Filter to physical energies, seed the adopted columns, and standardise features."""
    measurements = measurements[measurements["Energy"] > 0].copy()
    measurements["dData_adopted"] = measurements["dData_assumed"]
    measurements["uncertainty_source"] = "N/A"
    measurements.loc[~measurements["dData"].isna(), "uncertainty_source"] = "included"
    return standardize_features(measurements)


def select_pools(measurements, reports, entries):
    """Split reports into KNN candidates and imputation targets.

    A candidate report must have complete uncertainties (so its relative uncertainties are
    real) *and* an embedding (so it can be compared). Targets are the reverse.
    """
    complete = entries[entries["Uncertainty_Complete"] == 1]["EXFOR_Entry"]
    incomplete = entries[entries["Uncertainty_Complete"] == 0]["EXFOR_Entry"]
    candidates = reports[reports["EXFOR_Entry"].isin(complete)]
    targets = reports[reports["EXFOR_Entry"].isin(incomplete)]

    impute_index = measurements[
        measurements["EXFOR_Entry"].isin(targets["EXFOR_Entry"])
        & measurements["dData"].isna()
    ].index
    return candidates, targets, impute_index


def impute(measurements, reports, candidates, impute_index, config, emit):
    """Fill dData_adopted for each target row. Returns per-source counts.

    Indexing is by label throughout (``.loc``). The notebook mixed ``.iloc`` and ``.loc``,
    which is only safe while the frame's index has no gaps — true today only because the
    Energy > 0 filter happens to drop nothing.
    """
    with step(emit, "grouping measurements by entry and channel"):
        grouped = {key: group for key, group in measurements.groupby(["EXFOR_Entry", "MT"])}
    reports_by_entry = reports.set_index("EXFOR_Entry")
    candidate_entries = list(candidates["EXFOR_Entry"])
    weights = config.feature_weights or default_feature_weights()

    counts = {"nlp_imputed": 0, "quantile_imputed": 0}
    tracker = ProgressTracker(len(impute_index), emit, every=config.log_every,
                              unit="measurements", min_interval=60.0)

    for label in impute_index:
        query = measurements.loc[label]
        entry = query["EXFOR_Entry"]

        neighbours = []
        if entry in reports_by_entry.index:
            query_features = pd.concat([query, reports_by_entry.loc[entry]])
            for candidate_entry in candidate_entries:
                candidate_rows = grouped.get((candidate_entry, query["MT"]))
                if candidate_rows is None:
                    continue
                representative = representative_candidate(
                    candidate_rows, query, config.inner_weights)
                neighbours.append(pd.concat(
                    [representative, reports_by_entry.loc[candidate_entry]]))

        if not neighbours:
            # No embedding for this entry, or no same-channel candidate: keep the
            # per-channel quantile assumption computed in stage 1.
            measurements.loc[label, "uncertainty_source"] = "quantile_imputed"
            measurements.loc[label, "dData_adopted"] = measurements.loc[
                label, "dData_assumed"]
            counts["quantile_imputed"] += 1
            tracker.advance()
            continue

        knn = pd.concat(neighbours, axis=1).T
        distances = composite_distance(query_features, knn, weights)
        relative = knn["dData"] / knn["Data"]
        mean_relative = weighted_mean_relative_uncertainty(
            distances, relative, config.k_neighbors)

        measurements.loc[label, "dData_adopted"] = absolute_uncertainty(
            mean_relative, measurements.loc[label, "Data"])
        measurements.loc[label, "uncertainty_source"] = "nlp_imputed"
        counts["nlp_imputed"] += 1
        tracker.advance()

    tracker.finish()
    return counts


def persist(measurements, db_path, emit):
    """Write measurements back atomically, without accumulating an index column.

    Writes to a temporary table and swaps by rename, so an interrupted write cannot leave a
    half-replaced `measurements`. `index=False` is the fix for the re-run bug: the notebook
    used `if_exists='replace'` with `index=True`, so each run added another index column.
    """
    measurements = measurements.drop(columns=list(SCALAR_FEATURES), errors="ignore")
    con = sqlite3.connect(db_path)
    try:
        measurements.to_sql("measurements_new", con, if_exists="replace", index=False)
        con.execute("BEGIN")
        con.execute("DROP TABLE IF EXISTS measurements")
        con.execute("ALTER TABLE measurements_new RENAME TO measurements")
        con.execute(REACTION_CHANNEL_INDEX)
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    emit(f"wrote {len(measurements):,} measurements to {db_path}")
    return measurements


def write_test_subset(measurements, test_db_path, emit):
    """Write the Li-7 n,total slice used by the Flask repo's tests."""
    subset = measurements[(measurements["Z"] == 3) & (measurements["A"] == 7)
                          & (measurements["MT"] == 1) & (measurements["Projectile"] == "n")]
    if subset.empty:
        return 0
    con = sqlite3.connect(test_db_path)
    try:
        subset.to_sql("measurements", con, if_exists="replace", index=False)
        con.execute(REACTION_CHANNEL_INDEX)
        con.commit()
    finally:
        con.close()
    emit(f"wrote {len(subset):,} rows to {test_db_path}")
    return len(subset)


def run(config, *, progress=None, write_test_db=True):
    """Impute uncertainties across the corpus. Returns a StageResult."""
    emit = progress or printer("3")
    started = time.monotonic()
    check_or_raise(config, "3")

    with step(emit, f"loading tables from {config.db_path}"):
        measurements, reports, entries = load_inputs(config.db_path)
    emit(f"loaded {len(measurements):,} measurements, {len(reports)} embedded reports")

    with step(emit, "standardising KNN features"):
        measurements = prepare(measurements)
    with step(emit, "selecting candidate and target pools"):
        candidates, targets, impute_index = select_pools(measurements, reports, entries)
    if config.limit:
        impute_index = impute_index[:config.limit]

    emit(f"{len(candidates)} candidate reports (complete + embedded); "
         f"{len(impute_index):,} measurements to impute")

    warnings = []
    if candidates.empty:
        warnings.append("no candidate reports have both complete uncertainties and an "
                        "embedding; every target falls back to the quantile assumption")

    counts = impute(measurements, reports, candidates, impute_index, config, emit)

    # Anything still unset had no uncertainty and was not a KNN target.
    remaining = measurements["uncertainty_source"] == "N/A"
    measurements.loc[remaining, "uncertainty_source"] = "quantile_imputed"
    counts["quantile_imputed"] += int(remaining.sum())

    with step(emit, "writing measurements back"):
        written = persist(measurements, config.db_path, emit)
    outputs = [Path(config.db_path)]
    if write_test_db:
        if write_test_subset(written, config.test_db_path, emit):
            outputs.append(Path(config.test_db_path))

    return StageResult(
        stage="3",
        ok=True,
        counts={
            "measurements": len(written),
            "candidate_reports": len(candidates),
            "target_reports": len(targets),
            "nlp_imputed": counts["nlp_imputed"],
            "quantile_imputed": counts["quantile_imputed"],
            "already_reported": int((written["uncertainty_source"] == "included").sum()),
        },
        warnings=warnings,
        outputs=outputs,
        elapsed_s=time.monotonic() - started,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arguments(parser)
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    parser.add_argument("--no-test-db", action="store_true",
                        help="skip writing the Li-7 test subset")
    args = parser.parse_args()

    config = Config.from_args(args)
    try:
        result = run(config,
                     progress=null_printer if args.quiet else None,
                     write_test_db=not args.no_test_db)
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 2
    print(result.render())
    return result.exit_code()


if __name__ == "__main__":
    sys.exit(main())
