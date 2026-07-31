"""Tests for the KNN imputation stage.

These assert orchestration — persistence, schema, fallback routing, idempotence — not
arithmetic. The distance and weighting maths is already covered by test_imputation.py.
"""
import sqlite3

import numpy as np
import pytest

from conftest import read_table, table_columns
from pipeline_config import Config, ConfigError
from stage_result import null_printer
import stage3_imputation


class TestRun:
    def test_every_measurement_ends_with_an_adopted_uncertainty(self, tiny_config):
        """The point of the stage: no row may be left without a usable uncertainty."""
        stage3_imputation.run(tiny_config, progress=null_printer, write_test_db=False)

        measurements = read_table(tiny_config.db_path, "measurements")
        assert measurements["dData_adopted"].notna().all()

    def test_no_row_keeps_the_placeholder_source(self, tiny_config):
        stage3_imputation.run(tiny_config, progress=null_printer, write_test_db=False)

        sources = set(read_table(tiny_config.db_path, "measurements")["uncertainty_source"])
        assert "N/A" not in sources
        assert sources <= {"included", "nlp_imputed", "quantile_imputed"}

    def test_reported_uncertainties_are_left_alone(self, tiny_config):
        """Entry 10001 has real uncertainties; imputation must not overwrite them."""
        stage3_imputation.run(tiny_config, progress=null_printer, write_test_db=False)

        measurements = read_table(tiny_config.db_path, "measurements")
        complete = measurements[measurements["EXFOR_Entry"] == "10001"]
        assert (complete["uncertainty_source"] == "included").all()

    def test_an_entry_without_an_embedding_falls_back(self, tiny_config):
        """Entry 10003 has no report embedding, so KNN cannot run for it."""
        result = stage3_imputation.run(tiny_config, progress=null_printer,
                                       write_test_db=False)

        measurements = read_table(tiny_config.db_path, "measurements")
        unembedded = measurements[measurements["EXFOR_Entry"] == "10003"]
        assert (unembedded["uncertainty_source"] == "quantile_imputed").all()
        assert result.counts["quantile_imputed"] >= 1

    def test_a_target_with_candidates_is_imputed_by_knn(self, tiny_config):
        result = stage3_imputation.run(tiny_config, progress=null_printer,
                                       write_test_db=False)

        assert result.counts["nlp_imputed"] == 2   # entry 10002's two rows

    def test_adopted_uncertainty_is_never_negative(self, tiny_config):
        """Entry 10003 has Data = -1.5; an uncertainty is a width, never negative."""
        stage3_imputation.run(tiny_config, progress=null_printer, write_test_db=False)

        measurements = read_table(tiny_config.db_path, "measurements")
        assert (measurements["dData_adopted"] >= 0).all()

    def test_scratch_feature_columns_are_not_persisted(self, tiny_config):
        stage3_imputation.run(tiny_config, progress=null_printer, write_test_db=False)

        columns = table_columns(tiny_config.db_path, "measurements")
        for scratch in ("Energy_Logstd", "Z_std", "A_std"):
            assert scratch not in columns

    def test_reaction_channel_index_exists_afterwards(self, tiny_config):
        stage3_imputation.run(tiny_config, progress=null_printer, write_test_db=False)

        con = sqlite3.connect(tiny_config.db_path)
        indexes = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='index'")}
        con.close()
        assert "idx_reaction_channel" in indexes

    def test_limit_restricts_the_work(self, tiny_config):
        limited = Config.resolved(output_dir=tiny_config.output_dir,
                                  db_path=tiny_config.db_path, limit=1)

        result = stage3_imputation.run(limited, progress=null_printer, write_test_db=False)

        assert result.counts["nlp_imputed"] == 1


class TestIdempotence:
    def test_rerunning_does_not_add_an_index_column(self, tiny_config):
        """Regression: the notebook wrote with index=True and if_exists='replace', so each
        re-run added another index column and eventually broke the schema contract."""
        stage3_imputation.run(tiny_config, progress=null_printer, write_test_db=False)
        first = table_columns(tiny_config.db_path, "measurements")

        stage3_imputation.run(tiny_config, progress=null_printer, write_test_db=False)
        second = table_columns(tiny_config.db_path, "measurements")

        assert first == second
        assert "index" not in second
        assert "level_0" not in second

    def test_a_damaged_database_heals_on_the_next_run(self, tiny_config):
        """A database already carrying an index column from the old code must recover."""
        con = sqlite3.connect(tiny_config.db_path)
        con.execute("ALTER TABLE measurements ADD COLUMN 'index' INTEGER")
        con.commit()
        con.close()

        stage3_imputation.run(tiny_config, progress=null_printer, write_test_db=False)

        assert "index" not in table_columns(tiny_config.db_path, "measurements")

    def test_row_count_is_stable_across_runs(self, tiny_config):
        stage3_imputation.run(tiny_config, progress=null_printer, write_test_db=False)
        first = len(read_table(tiny_config.db_path, "measurements"))

        stage3_imputation.run(tiny_config, progress=null_printer, write_test_db=False)

        assert len(read_table(tiny_config.db_path, "measurements")) == first


class TestEmbeddingRoundTrip:
    def test_stage2_blobs_decode_to_the_original_vectors(self, tiny_db):
        """The most fragile cross-stage contract: mean_embedding is an untyped blob, so a
        dtype change in stage 2 would silently corrupt stage 3's distances."""
        from conftest import _embedding

        _, reports, _ = stage3_imputation.load_inputs(tiny_db)
        recovered = reports.set_index("EXFOR_Entry").loc["10001", "mean_embedding"]

        assert recovered.dtype == np.float32
        assert recovered == pytest.approx(_embedding(0))


class TestTestSubset:
    def test_writes_the_li7_slice(self, tiny_config):
        result = stage3_imputation.run(tiny_config, progress=null_printer,
                                       write_test_db=True)

        subset = read_table(tiny_config.test_db_path, "measurements")
        assert len(subset) == 6                     # entries 10001 and 10002 are Li-7 n,tot
        assert set(subset["Z"]) == {3}

    def test_subset_has_no_index_column_either(self, tiny_config):
        stage3_imputation.run(tiny_config, progress=null_printer, write_test_db=True)

        assert "index" not in table_columns(tiny_config.test_db_path, "measurements")


class TestPrerequisites:
    def test_missing_table_raises_before_any_work(self, tmp_path):
        db = tmp_path / "nugrade_data.db"
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE measurements (x INT)")
        con.commit()
        con.close()
        config = Config.resolved(output_dir=tmp_path)

        with pytest.raises(ConfigError, match="report_embeddings"):
            stage3_imputation.run(config, progress=null_printer)

    def test_missing_database_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="missing database"):
            stage3_imputation.run(Config.resolved(output_dir=tmp_path),
                                  progress=null_printer)


class TestResult:
    def test_reports_counts_that_add_up(self, tiny_config):
        result = stage3_imputation.run(tiny_config, progress=null_printer,
                                       write_test_db=False)

        total = (result.counts["nlp_imputed"] + result.counts["quantile_imputed"]
                 + result.counts["already_reported"])
        assert total == result.counts["measurements"]

    def test_lists_the_files_it_wrote(self, tiny_config):
        result = stage3_imputation.run(tiny_config, progress=null_printer,
                                       write_test_db=True)

        assert tiny_config.db_path in result.outputs
        assert tiny_config.test_db_path in result.outputs

    def test_warns_when_no_candidates_exist(self, tmp_path):
        """Everything falls back to the quantile assumption — quietly useless, so say so."""
        import conftest

        db = tmp_path / "nugrade_data.db"
        con = sqlite3.connect(db)
        conftest.make_measurements().to_sql("measurements", con, index=False)
        conftest.make_reports(entries=("10002",)).to_sql("report_embeddings", con,
                                                         index=False)
        entries = conftest.make_entries()
        entries["Uncertainty_Complete"] = 0        # no complete entries at all
        entries.to_sql("entries", con, index=False)
        conftest.make_subentries().to_sql("subentries", con, index=False)
        con.commit()
        con.close()

        result = stage3_imputation.run(Config.resolved(output_dir=tmp_path),
                                       progress=null_printer, write_test_db=False)

        assert any("no candidate reports" in w for w in result.warnings)
