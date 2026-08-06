"""Tests for the X4Pro ingestion stage.

Runs without cluster ACE files or OpenMC: evaluation readers are injected, and the X4Pro
frame is passed in directly. Assertions are about orchestration — resume, atomicity, the
summary contract, error containment — not about the metric arithmetic, which is covered by
test_ingestion.py.
"""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from conftest import read_table
from evaluations import (
    ExplodingEvaluationReader,
    NullEvaluationReader,
    SyntheticEvaluationReader,
)
from pipeline_config import Config, ConfigError
from stage_result import null_printer
import stage1_ingestion


def make_exfor_frame():
    """Two nuclides across two channels, shaped like the real post-rename frame."""
    rows = []
    for (Z, A, element), reaction, mt in [
        ((3, 7, "Li"), "N,TOT", 1), ((3, 7, "Li"), "N,G", 102),
        ((26, 56, "Fe"), "N,TOT", 1),
    ]:
        for i, energy in enumerate([1e3, 1e4, 1e5]):
            rows.append({
                "Z": Z, "A": A, "Element": element, "Reaction": reaction, "MT": mt,
                "Projectile": reaction[0].lower(),
                "Energy": energy, "dEnergy": np.nan,
                "Data": 2.0 + i, "dData": 0.2 if i else np.nan,
                "EXFOR_Entry": f"1000{Z}", "EXFOR_Subentry": f"1000{Z}00{mt}",
                "Dataset_Number": "1", "Year": 1970, "Author": "Alpha",
                "Target": f"{element}-{A}", "fullCode": "x",
            })
    return pd.DataFrame(rows)


@pytest.fixture
def stage1_config(tmp_path):
    """allow_missing_evals keeps static validation happy without cluster directories."""
    x4 = tmp_path / "x4sqlite1.db"
    x4.touch()
    return Config.resolved(output_dir=tmp_path, x4_db=x4, allow_missing_evals=True,
                           log_every=1000)


def run_measurements(config, readers=None, frame=None):
    return stage1_ingestion.run_measurements(
        config,
        readers=readers if readers is not None else [SyntheticEvaluationReader("endf7-1"),
                                                     SyntheticEvaluationReader("endf8")],
        exfor_df=frame if frame is not None else make_exfor_frame(),
        progress=null_printer)


class TestRunMeasurements:
    def test_ingests_every_channel_with_data(self, stage1_config):
        result = run_measurements(stage1_config)

        assert result.counts["channels"] == 3
        assert result.counts["measurements"] == 9

    def test_skips_channels_with_no_measurements(self, stage1_config):
        """The reaction channel grid is a cross product; most cells are empty."""
        result = run_measurements(stage1_config)

        assert result.counts["no_data"] > 0

    def test_writes_the_metric_columns(self, stage1_config):
        run_measurements(stage1_config)

        measurements = read_table(stage1_config.db_path, "measurements")
        for column in ("endf8", "endf8_chi_squared", "endf8_relative_error",
                       "dData_assumed"):
            assert column in measurements.columns

    def test_fills_missing_uncertainties(self, stage1_config):
        """dData_assumed is the fallback stage 3 adopts when KNN cannot run."""
        run_measurements(stage1_config)

        measurements = read_table(stage1_config.db_path, "measurements")
        assert measurements["dData_assumed"].notna().all()

    def test_chi_squared_is_never_negative(self, stage1_config):
        run_measurements(stage1_config)

        measurements = read_table(stage1_config.db_path, "measurements")
        assert (measurements["endf8_chi_squared"].dropna() >= 0).all()

    def test_writes_evaluation_grid_files(self, stage1_config):
        run_measurements(stage1_config)

        grids = list((stage1_config.output_dir / "evals" / "endf8").glob("*_grid_data.csv"))
        assert grids

    def test_missing_evaluation_yields_null_columns_not_a_crash(self, stage1_config):
        result = run_measurements(stage1_config,
                                  readers=[NullEvaluationReader("endf7-1"),
                                           NullEvaluationReader("endf8")])

        measurements = read_table(stage1_config.db_path, "measurements")
        assert result.counts["channels"] == 3
        assert measurements["endf8"].isna().all()

    def test_runs_with_a_single_evaluation_library(self, stage1_config):
        """The progress table has a column per evaluation, but the reader list is
        configurable — an evaluation nobody looked for must record as absent, not raise."""
        result = run_measurements(stage1_config,
                                  readers=[SyntheticEvaluationReader("endf8")])

        assert result.counts["channels"] == 3
        summary = pd.read_csv(stage1_config.output_dir / "all_reactions.csv")
        assert summary["has_endf8"].all()
        assert not summary["has_endf7-1"].any()

    def test_runs_with_no_evaluation_libraries(self, stage1_config):
        """--allow-missing-evals on a laptop: measurements still ingest, metrics are NaN."""
        result = run_measurements(stage1_config, readers=[])

        assert result.counts["channels"] == 3

    def test_a_failing_reader_does_not_abort_the_run(self, stage1_config):
        """One bad nuclide must not cost a six-hour job."""
        result = run_measurements(stage1_config,
                                  readers=[ExplodingEvaluationReader("endf8")])

        assert result.counts["channels"] == 3
        assert len(result.warnings) == 3
        assert "read failed" in result.warnings[0]


class TestReactionSummary:
    def test_mt_and_reaction_are_not_swapped(self, stage1_config):
        """Regression: the notebook wrote [Z,A,symbol,proj,mt,reaction] under headers
        [...,'Reaction','MT',...], so all_reactions.csv had MT holding 'N,TOT' and
        Reaction holding 1."""
        run_measurements(stage1_config)

        summary = pd.read_csv(stage1_config.output_dir / "all_reactions.csv")
        row = summary[(summary["Z"] == 3) & (summary["A"] == 7)
                      & (summary["Reaction"] == "N,TOT")].iloc[0]
        assert row["MT"] == 1
        assert row["Reaction"] == "N,TOT"

    def test_records_evaluation_availability(self, stage1_config):
        """The has_* flags drive which channels the app can score."""
        run_measurements(stage1_config,
                         readers=[NullEvaluationReader("endf7-1"),
                                  SyntheticEvaluationReader("endf8")])

        summary = pd.read_csv(stage1_config.output_dir / "all_reactions.csv")
        assert not summary["has_endf7-1"].any()
        assert summary["has_endf8"].all()

    def test_columns_the_flask_app_reads_are_present(self, stage1_config):
        """grading_functions.load_nuclide_index reads exactly these three."""
        run_measurements(stage1_config)

        summary = pd.read_csv(stage1_config.output_dir / "all_reactions.csv")
        for column in ("Z", "A", "Symbol"):
            assert column in summary.columns

    def test_summary_survives_a_partial_run(self, stage1_config):
        """Sourced from the progress table, so a crash still leaves what completed."""
        run_measurements(stage1_config, readers=[SyntheticEvaluationReader("endf8")])
        limited = Config.resolved(output_dir=stage1_config.output_dir,
                                  x4_db=stage1_config.x4_db, allow_missing_evals=True)

        summary = pd.read_csv(limited.output_dir / "all_reactions.csv")
        assert len(summary) == 3


class TestResume:
    def test_skips_channels_already_ingested(self, stage1_config):
        run_measurements(stage1_config)

        second = run_measurements(stage1_config)

        assert second.counts["skipped_existing"] == 3
        assert second.counts["channels"] == 0

    def test_does_not_duplicate_measurements(self, stage1_config):
        run_measurements(stage1_config)
        before = len(read_table(stage1_config.db_path, "measurements"))

        run_measurements(stage1_config)

        assert len(read_table(stage1_config.db_path, "measurements")) == before

    def test_no_resume_starts_over(self, stage1_config):
        run_measurements(stage1_config)
        fresh = Config.resolved(output_dir=stage1_config.output_dir,
                                x4_db=stage1_config.x4_db, allow_missing_evals=True,
                                resume=False)

        result = run_measurements(fresh)

        assert result.counts["channels"] == 3
        assert len(read_table(fresh.db_path, "measurements")) == 9

    def test_progress_and_measurements_stay_consistent(self, stage1_config):
        """Written in one transaction, so a channel is either fully present or absent."""
        run_measurements(stage1_config)

        measurements = read_table(stage1_config.db_path, "measurements")
        progress = read_table(stage1_config.db_path, "ingest_progress")
        assert progress["n_rows"].sum() == len(measurements)

    def test_resume_after_an_interruption_completes_the_work(self, stage1_config):
        """The scenario resume exists for: a killed job restarts at the next channel."""
        partial = Config.resolved(output_dir=stage1_config.output_dir,
                                  x4_db=stage1_config.x4_db, allow_missing_evals=True,
                                  limit=1)
        run_measurements(partial)
        assert len(read_table(partial.db_path, "ingest_progress")) == 1

        result = run_measurements(stage1_config)

        assert result.counts["skipped_existing"] == 1
        assert len(read_table(stage1_config.db_path, "ingest_progress")) == 3


class TestAggregates:
    def test_builds_both_summary_tables(self, stage1_config):
        run_measurements(stage1_config)

        result = stage1_ingestion.run_aggregates(stage1_config, progress=null_printer)

        assert result.counts["subentries"] == 3
        assert result.counts["entries"] == 2

    def test_entry_completeness_reflects_missing_uncertainties(self, stage1_config):
        """Uncertainty_Complete is what stage 3 uses to pick KNN candidates."""
        run_measurements(stage1_config)

        stage1_ingestion.run_aggregates(stage1_config, progress=null_printer)

        entries = read_table(stage1_config.db_path, "entries")
        assert (entries["Uncertainty_Complete"] == 0).all()   # each channel has one NaN
        assert (entries["Num_Missing_Uncertainty"] > 0).all()

    def test_subentry_energy_span_is_correct(self, stage1_config):
        run_measurements(stage1_config)

        stage1_ingestion.run_aggregates(stage1_config, progress=null_printer)

        subentries = read_table(stage1_config.db_path, "subentries")
        assert subentries["E_min"].min() == 1e3
        assert subentries["E_max"].max() == 1e5

    def test_can_run_without_cluster_files(self, stage1_config):
        """The reason the phases are separate: aggregates need only the database."""
        run_measurements(stage1_config)
        no_cluster = Config.resolved(output_dir=stage1_config.output_dir,
                                     x4_db=stage1_config.output_dir / "absent.db")

        result = stage1_ingestion.run_aggregates(no_cluster, progress=null_printer)

        assert result.ok

    def test_rerunning_replaces_rather_than_appends(self, stage1_config):
        run_measurements(stage1_config)
        stage1_ingestion.run_aggregates(stage1_config, progress=null_printer)
        first = len(read_table(stage1_config.db_path, "entries"))

        stage1_ingestion.run_aggregates(stage1_config, progress=null_printer)

        assert len(read_table(stage1_config.db_path, "entries")) == first

    def test_requires_measurements(self, tmp_path):
        config = Config.resolved(output_dir=tmp_path)

        with pytest.raises(ConfigError, match="measurements"):
            stage1_ingestion.run_aggregates(config, progress=null_printer)


class TestFallbackUncertainty:
    def test_is_the_corpus_quantile(self):
        frame = pd.DataFrame({"Data": [1.0] * 10,
                              "dData": [0.1 * i for i in range(1, 11)]})

        value = stage1_ingestion.corpus_fallback_uncertainty(frame)

        assert value == pytest.approx(np.quantile([0.1 * i for i in range(1, 11)], 0.90))

    def test_ignores_undefined_ratios(self):
        frame = pd.DataFrame({"Data": [1.0, 0.0, 2.0], "dData": [0.5, 0.5, np.nan]})

        assert np.isfinite(stage1_ingestion.corpus_fallback_uncertainty(frame))


class TestChannelLabel:
    def test_is_stable_and_readable(self):
        assert stage1_ingestion.channel_label("n", 3, 7, "N,TOT") == "n_3_7_N,TOT"


class TestDuplicationGuard:
    """Resume trusts ingest_progress to describe what is already in measurements. If that
    table was written by something else, appending doubles the corpus — which is exactly
    what happened on the cluster: 2,611,611 rows became 5,223,226."""

    def _unmanaged_measurements(self, config):
        """A measurements table with no ingest_progress rows, as the notebook left it."""
        con = sqlite3.connect(config.db_path)
        make_exfor_frame().to_sql("measurements", con, index=False)
        con.commit()
        con.close()

    def test_refuses_to_append_to_an_unmanaged_table(self, stage1_config):
        self._unmanaged_measurements(stage1_config)

        with pytest.raises(ConfigError, match="no ingest_progress"):
            run_measurements(stage1_config)

    def test_the_message_names_both_ways_out(self, stage1_config):
        self._unmanaged_measurements(stage1_config)

        with pytest.raises(ConfigError) as excinfo:
            run_measurements(stage1_config)

        message = str(excinfo.value)
        assert "--no-resume" in message
        assert "DROP TABLE" in message

    def test_no_resume_rebuilds_without_complaint(self, stage1_config):
        """The documented way out must actually work."""
        self._unmanaged_measurements(stage1_config)
        fresh = Config.resolved(output_dir=stage1_config.output_dir,
                                x4_db=stage1_config.x4_db, allow_missing_evals=True,
                                resume=False)

        result = run_measurements(fresh)

        assert result.counts["channels"] == 3
        assert len(read_table(fresh.db_path, "measurements")) == 9

    def test_a_normal_resume_is_unaffected(self, stage1_config):
        """The guard must not fire on a database this pipeline built itself."""
        run_measurements(stage1_config)

        second = run_measurements(stage1_config)

        assert second.counts["skipped_existing"] == 3

    def test_an_empty_database_is_not_treated_as_unmanaged(self, stage1_config):
        result = run_measurements(stage1_config)

        assert result.counts["channels"] == 3
