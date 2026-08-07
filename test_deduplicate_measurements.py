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
    frame = pd.DataFrame([{"EXFOR_Entry": f"1000{i}", "EXFOR_Subentry": f"1000{i}002",
                           "Energy": 1e3 * (i + 1), "Data": 2.0 + i, "dData": 0.1,
                           "MT": 1, "Projectile": "n", "endf8": 2.0, "endf7-1": 2.0}
                          for i in range(rows)])
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
        con = doubled_db(tmp_path / "db.sqlite", rows=3)
        con.execute("DELETE FROM measurements WHERE rowid > 3")   # remove the second copy
        con.commit()

        before, after = deduplicate(con, emit=lambda m: None)

        assert before == after == 3
        con.close()

    def test_does_not_merge_rows_differing_in_one_column(self, tmp_path):
        """Two measurements at the same energy with different cross sections are distinct:
        Data is part of the identity, not something derived from it."""
        con = sqlite3.connect(tmp_path / "db.sqlite")
        pd.DataFrame([
            {"EXFOR_Subentry": "1", "Energy": 1e3, "Data": 2.0, "dData": 0.1,
             "MT": 1, "Projectile": "n", "endf8": 2.0, "endf7-1": 2.0},
            {"EXFOR_Subentry": "1", "Energy": 1e3, "Data": 2.1, "dData": 0.1,
             "MT": 1, "Projectile": "n", "endf8": 2.0, "endf7-1": 2.0},
        ]).to_sql("measurements", con, index=False)
        con.commit()

        before, after = deduplicate(con, emit=lambda m: None)

        assert after == 2
        con.close()

    def test_refuses_a_table_it_cannot_identify_measurements_in(self, tmp_path):
        """Guessing an identity would silently delete real data."""
        con = sqlite3.connect(tmp_path / "db.sqlite")
        pd.DataFrame([{"a": 1}, {"a": 2}]).to_sql("measurements", con, index=False)
        con.commit()

        with pytest.raises(ValueError, match="cannot identify measurements"):
            deduplicate(con, emit=lambda m: None)
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


class TestNaturalKeyDeduplication:
    """The case exact matching cannot see: a re-ingest whose derived columns differ."""

    def _mixed_db(self, path):
        """The cluster's actual state — a good copy plus one with no evaluation data."""
        con = sqlite3.connect(path)
        good = {"EXFOR_Subentry": "10004002", "Energy": 494600.0, "Data": 4.596,
                "dData": 0.1, "MT": 1, "Projectile": "n",
                "endf8": 5.42, "endf7-1": 5.42}
        empty = dict(good, endf8=None)
        empty["endf7-1"] = None
        pd.DataFrame([good, empty]).to_sql("measurements", con, index=False)
        con.commit()
        return con

    def test_exact_matching_cannot_see_them(self, tmp_path):
        con = self._mixed_db(tmp_path / "db.sqlite")

        before, after = deduplicate(con, emit=lambda m: None, natural=False)

        assert before == after == 2, "the rows differ, so exact matching keeps both"
        con.close()

    def test_natural_key_removes_the_repeat(self, tmp_path):
        con = self._mixed_db(tmp_path / "db.sqlite")

        before, after = deduplicate(con, emit=lambda m: None)

        assert (before, after) == (2, 1)
        con.close()

    def test_keeps_the_copy_carrying_evaluation_data(self, tmp_path):
        """Keeping the empty copy would discard the whole point of the ingest."""
        con = self._mixed_db(tmp_path / "db.sqlite")

        deduplicate(con, emit=lambda m: None)

        endf8 = con.execute('SELECT "endf8" FROM measurements').fetchone()[0]
        assert endf8 is not None
        con.close()

    def test_is_deterministic_when_copies_are_equally_complete(self, tmp_path):
        con = sqlite3.connect(tmp_path / "db.sqlite")
        row = {"EXFOR_Subentry": "1", "Energy": 1.0, "Data": 2.0, "dData": 0.1,
               "MT": 1, "Projectile": "n", "endf8": 2.0, "endf7-1": 2.0}
        pd.DataFrame([dict(row, Year=1970), dict(row, Year=1971)]).to_sql(
            "measurements", con, index=False)
        con.commit()

        deduplicate(con, emit=lambda m: None)

        assert con.execute("SELECT Year FROM measurements").fetchone()[0] == 1970
        con.close()


class TestNaturalKeyCompleteness:
    """The key must not collapse measurements EXFOR treats as distinct."""

    def _rows(self, **overrides):
        base = {"EXFOR_Subentry": "31736003", "Dataset_Number": "317360032",
                "Energy": 0.0253, "dEnergy": None, "Data": 2.0, "dData": 0.1,
                "MT": 1, "Projectile": "n", "endf8": 2.0, "endf7-1": 2.0}
        return [base, dict(base, **overrides)]

    def test_keeps_rows_differing_only_by_dataset(self, tmp_path):
        """Real case in the corpus: subentry 31736003 at 0.0253 eV appears in datasets
        317360032 and 317360034."""
        con = sqlite3.connect(tmp_path / "db.sqlite")
        pd.DataFrame(self._rows(Dataset_Number="317360034")).to_sql(
            "measurements", con, index=False)
        con.commit()

        before, after = deduplicate(con, emit=lambda m: None)

        assert after == 2, "different datasets are different measurements"
        con.close()

    def test_keeps_rows_differing_only_by_energy_uncertainty(self, tmp_path):
        """Eight groups in the corpus differ only in dEnergy."""
        con = sqlite3.connect(tmp_path / "db.sqlite")
        pd.DataFrame(self._rows(dEnergy=0.001)).to_sql("measurements", con, index=False)
        con.commit()

        before, after = deduplicate(con, emit=lambda m: None)

        assert after == 2
        con.close()

    def test_still_removes_a_true_repeat_ingest(self, tmp_path):
        con = sqlite3.connect(tmp_path / "db.sqlite")
        rows = self._rows()
        rows[1]["endf8"] = None          # the re-ingest that found no ACE files
        rows[1]["endf7-1"] = None
        pd.DataFrame(rows).to_sql("measurements", con, index=False)
        con.commit()

        before, after = deduplicate(con, emit=lambda m: None)

        assert (before, after) == (2, 1)
        assert con.execute('SELECT "endf8" FROM measurements').fetchone()[0] is not None
        con.close()
