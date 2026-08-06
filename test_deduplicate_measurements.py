"""Tests for the duplicate-row repair."""
import sqlite3

import pandas as pd
import pytest

from deduplicate_measurements import (
    count_distinct,
    count_rows,
    deduplicate,
    reset_progress,
    table_columns,
)


def doubled_db(path, rows=4):
    frame = pd.DataFrame([{"EXFOR_Entry": f"1000{i}", "Energy": 1e3 * i, "Data": 2.0 + i,
                           "dData": 0.1} for i in range(rows)])
    con = sqlite3.connect(path)
    frame.to_sql("measurements", con, index=False)
    frame.to_sql("measurements", con, if_exists="append", index=False)
    con.commit()
    return con


class TestDeduplicate:
    def test_halves_an_exactly_doubled_table(self, tmp_path):
        con = doubled_db(tmp_path / "db.sqlite")

        before, after = deduplicate(con, emit=lambda m: None)

        assert (before, after) == (8, 4)
        con.close()

    def test_keeps_every_distinct_row(self, tmp_path):
        con = doubled_db(tmp_path / "db.sqlite")

        deduplicate(con, emit=lambda m: None)

        entries = {r[0] for r in con.execute("SELECT EXFOR_Entry FROM measurements")}
        assert entries == {"10000", "10001", "10002", "10003"}
        con.close()

    def test_is_idempotent(self, tmp_path):
        con = doubled_db(tmp_path / "db.sqlite")
        deduplicate(con, emit=lambda m: None)

        before, after = deduplicate(con, emit=lambda m: None)

        assert before == after
        con.close()

    def test_leaves_a_clean_table_alone(self, tmp_path):
        con = sqlite3.connect(tmp_path / "db.sqlite")
        pd.DataFrame([{"a": 1}, {"a": 2}]).to_sql("measurements", con, index=False)
        con.commit()

        before, after = deduplicate(con, emit=lambda m: None)

        assert before == after == 2
        con.close()

    def test_does_not_merge_rows_differing_in_one_column(self, tmp_path):
        """Two measurements at the same energy with different cross sections are distinct."""
        con = sqlite3.connect(tmp_path / "db.sqlite")
        pd.DataFrame([{"Energy": 1e3, "Data": 2.0}, {"Energy": 1e3, "Data": 2.1}]).to_sql(
            "measurements", con, index=False)
        con.commit()

        before, after = deduplicate(con, emit=lambda m: None)

        assert after == 2
        con.close()


class TestResetProgress:
    def test_clears_stale_counts(self, tmp_path):
        """After a dedup the per-channel n_rows describe the doubled corpus, so a later
        resume would skip channels on the strength of bookkeeping that no longer holds."""
        con = sqlite3.connect(tmp_path / "db.sqlite")
        pd.DataFrame([{"label": "n_3_7_N,TOT", "n_rows": 18}]).to_sql(
            "ingest_progress", con, index=False)
        con.commit()

        removed = reset_progress(con, emit=lambda m: None)

        assert removed == 1
        assert count_rows(con, "ingest_progress") == 0
        con.close()

    def test_harmless_when_the_table_is_absent(self, tmp_path):
        con = sqlite3.connect(tmp_path / "db.sqlite")

        assert reset_progress(con, emit=lambda m: None) == 0
        con.close()


class TestCounting:
    def test_distinct_ignores_duplicates(self, tmp_path):
        con = doubled_db(tmp_path / "db.sqlite")
        columns = table_columns(con, "measurements")

        assert count_rows(con) == 8
        assert count_distinct(con, columns) == 4
        con.close()
