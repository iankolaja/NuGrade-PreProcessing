"""Tests for pipeline configuration and prerequisite validation.

Precedence bugs are the kind that only show up on the cluster, at 3am, inside a SLURM job.
`Config.from_args` takes `env` as a parameter specifically so they can be tested here.
"""
import argparse
import sqlite3
from pathlib import Path

import pytest

from pipeline_config import (
    ENV_PREFIX,
    Config,
    ConfigError,
    add_config_arguments,
    check_or_raise,
    template_root,
    validate_inputs,
    validate_static,
)


def parse(argv):
    parser = add_config_arguments(argparse.ArgumentParser())
    return parser.parse_args(argv)


class TestPrecedence:
    def test_default_when_nothing_is_set(self):
        config = Config.from_args(parse([]), env={})

        assert config.k_neighbors == 5
        assert config.x4_db == Path("sources/x4sqlite1.db")

    def test_environment_overrides_default(self):
        config = Config.from_args(parse([]), env={f"{ENV_PREFIX}K_NEIGHBORS": "9"})

        assert config.k_neighbors == 9

    def test_flag_overrides_environment(self):
        """The whole reason flags default to None rather than the real default value."""
        config = Config.from_args(parse(["--k-neighbors", "3"]),
                                  env={f"{ENV_PREFIX}K_NEIGHBORS": "9"})

        assert config.k_neighbors == 3

    def test_flag_set_to_the_default_value_still_wins(self):
        """If flags defaulted to 5, this could not be distinguished from 'flag absent'."""
        config = Config.from_args(parse(["--k-neighbors", "5"]),
                                  env={f"{ENV_PREFIX}K_NEIGHBORS": "9"})

        assert config.k_neighbors == 5

    def test_empty_environment_variable_is_ignored(self):
        config = Config.from_args(parse([]), env={f"{ENV_PREFIX}K_NEIGHBORS": ""})

        assert config.k_neighbors == 5

    def test_paths_are_converted_from_strings(self):
        config = Config.from_args(parse([]), env={f"{ENV_PREFIX}X4_DB": "/data/x4.db"})

        assert config.x4_db == Path("/data/x4.db")
        assert isinstance(config.x4_db, Path)

    def test_cluster_style_environment(self):
        """The shape a SLURM script actually uses."""
        config = Config.from_args(parse([]), env={
            f"{ENV_PREFIX}X4_DB": "/global/scratch/x4sqlite1.db",
            f"{ENV_PREFIX}ENDF8_TEMPLATE": "/global/lib/<symbol>/<ZAID>.800nc",
            f"{ENV_PREFIX}OUTPUT_DIR": "/scratch/run7",
        })

        assert config.x4_db == Path("/global/scratch/x4sqlite1.db")
        assert config.endf8_template == "/global/lib/<symbol>/<ZAID>.800nc"
        assert config.db_path == Path("/scratch/run7/nugrade_data.db")


class TestDerivedPaths:
    def test_database_follows_the_output_directory(self):
        """The notebooks hardcoded 'output/nugrade_data.db' separately from output_dir, so
        redirecting the output only moved half the artefacts."""
        config = Config.resolved(output_dir="/scratch/run7")

        assert config.db_path == Path("/scratch/run7/nugrade_data.db")
        assert config.test_db_path == Path("/scratch/run7/test_data.db")

    def test_explicit_database_path_wins(self):
        config = Config.resolved(output_dir="/scratch/run7", db_path="/elsewhere/my.db")

        assert config.db_path == Path("/elsewhere/my.db")
        assert config.test_db_path == Path("/scratch/run7/test_data.db")

    def test_ensure_directories_creates_the_eval_tree(self, tmp_path):
        config = Config.resolved(output_dir=tmp_path / "out")

        config.ensure_directories()

        assert (tmp_path / "out" / "evals" / "endf8").is_dir()
        assert (tmp_path / "out" / "evals" / "endf7-1").is_dir()

    def test_ensure_directories_is_idempotent(self, tmp_path):
        config = Config.resolved(output_dir=tmp_path / "out")
        config.ensure_directories()
        config.ensure_directories()  # must not raise

        assert (tmp_path / "out").is_dir()

    def test_config_is_immutable(self):
        """A stage must not be able to quietly reconfigure itself mid-run."""
        config = Config()

        with pytest.raises(Exception):
            config.k_neighbors = 99


class TestTemplateRoot:
    def test_extracts_the_directory_before_the_placeholder(self):
        assert template_root("/data/endf8/<symbol>/<ZAID>.800nc") == Path("/data/endf8")

    def test_handles_a_template_with_no_placeholder(self):
        assert template_root("/data/endf8/file.ace") == Path("/data/endf8")


class TestValidateStatic:
    def test_stage1_reports_a_missing_x4pro_database(self, tmp_path):
        config = Config.resolved(x4_db=tmp_path / "nope.db", allow_missing_evals=True)

        problems = validate_static(config, "1")

        assert len(problems) == 1
        assert "missing X4Pro database" in problems[0]

    def test_stage1_message_names_both_the_flag_and_the_variable(self, tmp_path):
        """Half these runs are SLURM scripts where only the variable is in play."""
        config = Config.resolved(x4_db=tmp_path / "nope.db", allow_missing_evals=True)

        message = validate_static(config, "1")[0]

        assert "--x4-db" in message
        assert f"{ENV_PREFIX}X4_DB" in message

    def test_stage1_reports_missing_evaluation_libraries(self, tmp_path):
        x4 = tmp_path / "x4.db"
        x4.touch()
        config = Config.resolved(x4_db=x4,
                                 endf71_template=str(tmp_path / "no71" / "<symbol>" / "<ZAID>"),
                                 endf8_template=str(tmp_path / "no8" / "<symbol>" / "<ZAID>"))

        problems = validate_static(config, "1")

        assert len(problems) == 2
        assert any("endf7-1" in p for p in problems)
        assert any("endf8" in p for p in problems)

    def test_allow_missing_evals_downgrades_the_evaluation_check(self, tmp_path):
        """Makes a laptop dry-run possible: compute_channel_metrics already handles None."""
        x4 = tmp_path / "x4.db"
        x4.touch()
        config = Config.resolved(x4_db=x4, allow_missing_evals=True,
                                 endf71_template=str(tmp_path / "gone" / "<symbol>"),
                                 endf8_template=str(tmp_path / "gone" / "<symbol>"))

        assert validate_static(config, "1") == []

    def test_stage1b_does_not_require_cluster_files(self, tmp_path):
        """The aggregate phase reads the database only, so it must run without ACE files."""
        config = Config.resolved(x4_db=tmp_path / "nope.db")

        assert validate_static(config, "1b") == []

    def test_stage2_reports_a_missing_pdf_directory(self, tmp_path):
        config = Config.resolved(pdf_dir=tmp_path / "nope",
                                 template_file=tmp_path / "t.json")
        (tmp_path / "t.json").touch()

        problems = validate_static(config, "2")

        assert any("missing PDF directory" in p for p in problems)

    def test_stage2_reports_an_empty_pdf_directory(self, tmp_path):
        """Present but empty is a different mistake from absent, and worth saying so."""
        (tmp_path / "pdfs").mkdir()
        (tmp_path / "t.json").touch()
        config = Config.resolved(pdf_dir=tmp_path / "pdfs", template_file=tmp_path / "t.json")

        problems = validate_static(config, "2")

        assert any("no PDFs in" in p for p in problems)

    def test_stage2_passes_with_a_pdf_present(self, tmp_path):
        (tmp_path / "pdfs").mkdir()
        (tmp_path / "pdfs" / "10283.pdf").touch()
        (tmp_path / "t.json").touch()
        config = Config.resolved(pdf_dir=tmp_path / "pdfs", template_file=tmp_path / "t.json")

        assert validate_static(config, "2") == []

    def test_stage3_has_no_static_prerequisites(self):
        """All of stage 3's inputs come from earlier stages, so nothing to check up front."""
        assert validate_static(Config(), "3") == []


def make_db(path, tables):
    con = sqlite3.connect(path)
    for table in tables:
        con.execute(f"CREATE TABLE {table} (x INT)")
    con.commit()
    con.close()


class TestValidateInputs:
    def test_stage3_reports_a_missing_database(self, tmp_path):
        config = Config.resolved(output_dir=tmp_path)

        problems = validate_inputs(config, "3")

        assert any("missing database" in p for p in problems)

    def test_stage3_names_the_missing_tables(self, tmp_path):
        db = tmp_path / "nugrade_data.db"
        make_db(db, ["measurements", "entries", "subentries"])
        config = Config.resolved(output_dir=tmp_path)

        problems = validate_inputs(config, "3")

        assert any("report_embeddings" in p for p in problems)

    def test_stage3_points_at_the_producing_stage(self, tmp_path):
        """'run stage 2 first' is the actionable part; a table name alone is not."""
        db = tmp_path / "nugrade_data.db"
        make_db(db, ["measurements", "entries", "subentries"])
        config = Config.resolved(output_dir=tmp_path)

        assert "stage 2" in validate_inputs(config, "3")[0]

    def test_stage3_passes_when_every_table_exists(self, tmp_path):
        db = tmp_path / "nugrade_data.db"
        make_db(db, ["measurements", "entries", "subentries", "report_embeddings"])
        config = Config.resolved(output_dir=tmp_path)

        assert validate_inputs(config, "3") == []

    def test_stage1_has_no_input_prerequisites(self, tmp_path):
        assert validate_inputs(Config.resolved(output_dir=tmp_path), "1") == []

    def test_stage1b_requires_measurements(self, tmp_path):
        db = tmp_path / "nugrade_data.db"
        make_db(db, ["something_else"])
        config = Config.resolved(output_dir=tmp_path)

        assert any("measurements" in p for p in validate_inputs(config, "1b"))

    def test_does_not_modify_the_database(self, tmp_path):
        """Validation opens read-only; a check must never create or alter the file."""
        db = tmp_path / "nugrade_data.db"
        make_db(db, ["measurements"])
        before = db.stat().st_mtime_ns

        validate_inputs(Config.resolved(output_dir=tmp_path), "1b")

        assert db.stat().st_mtime_ns == before


class TestCheckOrRaise:
    def test_reports_every_problem_at_once(self, tmp_path):
        """One problem per run would mean one cluster round-trip per problem."""
        config = Config.resolved(pdf_dir=tmp_path / "nope",
                                 template_file=tmp_path / "nope.json")

        with pytest.raises(ConfigError) as excinfo:
            check_or_raise(config, "2")

        message = str(excinfo.value)
        assert "missing PDF directory" in message
        assert "missing template sentences" in message

    def test_passes_silently_when_satisfied(self, tmp_path):
        (tmp_path / "pdfs").mkdir()
        (tmp_path / "pdfs" / "1.pdf").touch()
        (tmp_path / "t.json").touch()
        config = Config.resolved(pdf_dir=tmp_path / "pdfs", template_file=tmp_path / "t.json")

        check_or_raise(config, "2")  # must not raise

    def test_input_checks_can_be_skipped(self, tmp_path):
        """run_pipeline skips these for a stage whose producer is in the same selection."""
        config = Config.resolved(output_dir=tmp_path)

        check_or_raise(config, "3", inputs=False)  # must not raise
