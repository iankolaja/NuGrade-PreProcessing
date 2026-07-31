"""Tests for pipeline sequencing, selection and failure handling."""
import sqlite3

import pytest

import run_pipeline
from pipeline_config import Config, ConfigError
from stage_result import StageResult, null_printer


class TestParseStages:
    def test_all_by_default(self):
        assert run_pipeline.parse_stages("all") == ["1", "1b", "2", "3"]
        assert run_pipeline.parse_stages("") == ["1", "1b", "2", "3"]

    def test_naming_stage_one_includes_its_aggregate_phase(self):
        """They are one logical stage; asking for '1' and getting no summary tables would
        surprise anyone who had run the notebook."""
        assert run_pipeline.parse_stages("1") == ["1", "1b"]

    def test_subset(self):
        assert run_pipeline.parse_stages("2,3") == ["2", "3"]

    def test_user_order_is_ignored_in_favour_of_dependency_order(self):
        """Running 3 before 2 is a mistake, not a preference."""
        assert run_pipeline.parse_stages("3,2") == ["2", "3"]

    def test_tolerates_whitespace(self):
        assert run_pipeline.parse_stages(" 2 , 3 ") == ["2", "3"]

    def test_unknown_stage_is_rejected(self):
        with pytest.raises(ConfigError, match="unknown stage"):
            run_pipeline.parse_stages("7")


class TestPreflight:
    def test_reports_missing_prerequisites(self, tmp_path):
        config = Config.resolved(output_dir=tmp_path)

        problems = run_pipeline.preflight(config, ["3"])

        assert any("missing database" in p for p in problems)

    def test_skips_input_checks_for_a_stage_produced_in_this_run(self, tmp_path):
        """--stages 1,2,3 must not fail on a table stage 2 is about to create."""
        x4 = tmp_path / "x4.db"
        x4.touch()
        (tmp_path / "pdfs").mkdir()
        (tmp_path / "pdfs" / "1.pdf").touch()
        (tmp_path / "t.json").touch()
        config = Config.resolved(output_dir=tmp_path, x4_db=x4, allow_missing_evals=True,
                                 pdf_dir=tmp_path / "pdfs", template_file=tmp_path / "t.json")

        problems = run_pipeline.preflight(config, ["1", "1b", "2", "3"])

        assert problems == []

    def test_still_checks_inputs_no_stage_will_produce(self, tmp_path):
        """Asking for 3 alone must fail fast rather than after loading a huge table."""
        config = Config.resolved(output_dir=tmp_path)

        assert run_pipeline.preflight(config, ["3"]) != []


class TestRunPipeline:
    def test_runs_stages_in_order(self, monkeypatch, tmp_path):
        calls = []

        def fake(name):
            def run(config, progress=None):
                calls.append(name)
                return StageResult(stage=name, ok=True)
            return run

        for name, stage in run_pipeline.STAGES.items():
            monkeypatch.setattr(stage, "run", fake(name))

        run_pipeline.run_pipeline(Config(), ["1", "1b", "2", "3"],
                                  progress_factory=lambda n: null_printer)

        assert calls == ["1", "1b", "2", "3"]

    def test_stops_at_the_first_failure(self, monkeypatch):
        calls = []

        def make(name, ok):
            def run(config, progress=None):
                calls.append(name)
                return StageResult(stage=name, ok=ok)
            return run

        monkeypatch.setattr(run_pipeline.STAGES["1"], "run", make("1", False))
        monkeypatch.setattr(run_pipeline.STAGES["1b"], "run", make("1b", True))

        run_pipeline.run_pipeline(Config(), ["1", "1b"],
                                  progress_factory=lambda n: null_printer)

        assert calls == ["1"]

    def test_keep_going_skips_dependents_rather_than_attempting_them(self, monkeypatch):
        """Running the aggregates over measurements that failed would produce nonsense."""
        calls = []

        def make(name, ok):
            def run(config, progress=None):
                calls.append(name)
                return StageResult(stage=name, ok=ok)
            return run

        monkeypatch.setattr(run_pipeline.STAGES["1"], "run", make("1", False))
        monkeypatch.setattr(run_pipeline.STAGES["1b"], "run", make("1b", True))
        monkeypatch.setattr(run_pipeline.STAGES["2"], "run", make("2", True))

        results, skipped = run_pipeline.run_pipeline(
            Config(), ["1", "1b", "2"], keep_going=True,
            progress_factory=lambda n: null_printer)

        assert "1b" in skipped                 # depends on measurements
        assert "2" in results                  # independent, so it still runs
        assert calls == ["1", "2"]

    def test_a_config_error_becomes_a_failed_result(self, monkeypatch):
        """A prerequisite discovered mid-run must not surface as a traceback."""
        def boom(config, progress=None):
            raise ConfigError("nope")

        monkeypatch.setattr(run_pipeline.STAGES["3"], "run", boom)

        results, _ = run_pipeline.run_pipeline(Config(), ["3"],
                                               progress_factory=lambda n: null_printer)

        assert not results["3"].ok
        assert "nope" in results["3"].warnings[0]


class TestSummary:
    def test_lists_every_stage_with_its_status(self):
        results = {"1": StageResult("1", ok=True, counts={"channels": 3}, elapsed_s=61)}
        skipped = {"1b": "depends on measurements"}

        text = run_pipeline.render_summary(results, skipped, 61)

        assert "ok" in text
        assert "skipped" in text
        assert "channels=3" in text
        assert "1:01" in text


class TestEndToEnd:
    def test_dry_run_validates_without_executing(self, tmp_path, monkeypatch, capsys):
        """The check that turns a three-hour failure into a one-second one."""
        called = []
        monkeypatch.setattr(run_pipeline.STAGES["3"], "run",
                            lambda c, progress=None: called.append("ran"))

        db = tmp_path / "nugrade_data.db"
        con = sqlite3.connect(db)
        for table in ("measurements", "entries", "subentries", "report_embeddings"):
            con.execute(f"CREATE TABLE {table} (x INT)")
        con.commit()
        con.close()

        monkeypatch.setattr("sys.argv",
                            ["run_pipeline.py", "--stages", "3",
                             "--output-dir", str(tmp_path), "--dry-run"])

        code = run_pipeline.main()

        assert code == 0
        assert called == []
        assert "dry run" in capsys.readouterr().out

    def test_missing_prerequisite_exits_two(self, tmp_path, monkeypatch):
        """Exit 2 distinguishes a setup mistake from a stage that ran and failed."""
        monkeypatch.setattr("sys.argv",
                            ["run_pipeline.py", "--stages", "3",
                             "--output-dir", str(tmp_path)])

        assert run_pipeline.main() == 2

    def test_unknown_stage_exits_two(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["run_pipeline.py", "--stages", "9"])

        assert run_pipeline.main() == 2


class TestContractCheck:
    """A database the app will reject must not be reported as a successful run."""

    def _database(self, tmp_path, negative):
        import pandas as pd

        db = tmp_path / "nugrade_data.db"
        con = sqlite3.connect(db)
        frame = pd.DataFrame([{
            "Z": 3, "A": 7, "MT": 1, "Projectile": "n", "Reaction": "N,TOT",
            "Element": "Li", "Energy": 1e3, "dEnergy": None, "Data": 2.0,
            "dData": -0.1 if negative else 0.1, "dData_assumed": 0.1,
            "dData_adopted": 0.1, "EXFOR_Entry": "1", "EXFOR_Subentry": "1",
            "Dataset_Number": "1", "Year": 1970, "Author": "A",
            "endf8": 2.0, "endf8_chi_squared": 1.0, "endf8_relative_error": 1.0,
            "endf7-1": 2.0, "endf7-1_chi_squared": 1.0, "endf7-1_relative_error": 1.0,
        }])
        frame.to_sql("measurements", con, index=False)
        pd.DataFrame([{"Z": 3, "A": 7, "MT": 1, "Reaction": "N,TOT", "Element": "Li",
                       "EXFOR_Entry": "1", "EXFOR_Subentry": "1",
                       "E_min": 1e3, "E_max": 1e3}]).to_sql("subentries", con, index=False)
        pd.DataFrame([{"EXFOR_Entry": "1", "Sentence_Number": 1, "Text": "x",
                       "Embedding": (b"\x00" * 4 * 768)}]).to_sql(
            "sentence_embeddings", con, index=False)
        con.commit()
        con.close()
        return db

    def test_a_valid_database_exits_zero(self, tmp_path, monkeypatch):
        self._database(tmp_path, negative=False)
        monkeypatch.setattr(run_pipeline.STAGES["3"], "run",
                            lambda c, progress=None: StageResult("3", ok=True))
        monkeypatch.setattr("sys.argv", ["run_pipeline.py", "--stages", "3",
                                         "--output-dir", str(tmp_path), "--quiet"])
        monkeypatch.setattr(run_pipeline, "preflight", lambda c, s: [])

        assert run_pipeline.main() == 0

    def test_an_invalid_database_exits_one(self, tmp_path, monkeypatch, capsys):
        """Otherwise a SLURM job silently ships a database the app cannot load."""
        self._database(tmp_path, negative=True)
        monkeypatch.setattr(run_pipeline.STAGES["3"], "run",
                            lambda c, progress=None: StageResult("3", ok=True))
        monkeypatch.setattr("sys.argv", ["run_pipeline.py", "--stages", "3",
                                         "--output-dir", str(tmp_path), "--quiet"])
        monkeypatch.setattr(run_pipeline, "preflight", lambda c, s: [])

        code = run_pipeline.main()

        assert code == 1
        assert "contract check FAILED" in capsys.readouterr().out

    def test_the_check_can_be_waived(self, tmp_path, monkeypatch):
        """While an earlier stage is still pending, the failure is expected."""
        self._database(tmp_path, negative=True)
        monkeypatch.setattr(run_pipeline.STAGES["3"], "run",
                            lambda c, progress=None: StageResult("3", ok=True))
        monkeypatch.setattr("sys.argv", ["run_pipeline.py", "--stages", "3",
                                         "--output-dir", str(tmp_path), "--quiet",
                                         "--skip-contract-check"])
        monkeypatch.setattr(run_pipeline, "preflight", lambda c, s: [])

        assert run_pipeline.main() == 0
