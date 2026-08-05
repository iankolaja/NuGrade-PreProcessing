"""Tests for StageResult, progress reporting, and the evaluation reader seam."""
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from evaluations import (
    AceEvaluationReader,
    NullEvaluationReader,
    SyntheticEvaluationReader,
    readers_from_config,
)
from pipeline_config import Config
from stage_result import (
    ProgressTracker,
    StageResult,
    format_duration,
    printer,
    step,
)


class TestStageResult:
    def test_exit_code_reflects_success(self):
        assert StageResult("1", ok=True).exit_code() == 0
        assert StageResult("1", ok=False).exit_code() == 1

    def test_render_includes_counts_and_outputs(self):
        result = StageResult("1", counts={"channels": 2314}, outputs=[Path("out/x.csv")],
                             elapsed_s=125)

        text = result.render()

        assert "2,314" in text          # thousands separator, for six-digit corpora
        assert "out/x.csv" in text
        assert "2:05" in text

    def test_render_truncates_a_long_warning_list(self):
        result = StageResult("2", warnings=[f"w{i}" for i in range(30)])

        text = result.render()

        assert "and 10 more" in text

    def test_merge_combines_two_phases(self):
        first = StageResult("1", counts={"a": 1}, outputs=[Path("x")], elapsed_s=10)
        second = StageResult("1b", counts={"b": 2}, warnings=["w"], elapsed_s=5)

        merged = first.merge(second)

        assert merged.counts == {"a": 1, "b": 2}
        assert merged.elapsed_s == 15
        assert merged.warnings == ["w"]

    def test_merge_fails_if_either_phase_failed(self):
        assert not StageResult("1", ok=True).merge(StageResult("1b", ok=False)).ok

    def test_to_frame_is_one_row(self):
        frame = StageResult("3", counts={"imputed": 5}).to_frame()

        assert len(frame) == 1
        assert frame.iloc[0]["imputed"] == 5


class TestFormatDuration:
    @pytest.mark.parametrize("seconds,expected", [
        (0, "0:00"), (65, "1:05"), (3600, "1:00:00"), (3725, "1:02:05"),
    ])
    def test_formats(self, seconds, expected):
        assert format_duration(seconds) == expected


class TestPrinter:
    def test_flushes_every_line(self):
        """SLURM block-buffers redirected stdout: without flush a run looks hung."""
        flushes = []

        class Recording(io.StringIO):
            def flush(self):
                flushes.append(True)
                super().flush()

        stream = Recording()
        emit = printer("1", stream=stream)
        emit("hello")

        assert flushes, "progress must be flushed"
        assert "hello" in stream.getvalue()

    def test_line_carries_stage_and_elapsed(self):
        """Format is [HH:MM:SS stage +elapsed]; the wall clock answers "when did it stop",
        which elapsed time alone cannot."""
        stream = io.StringIO()
        printer("2", stream=stream)("working")

        line = stream.getvalue()
        assert " 2 +0:00]" in line
        assert "working" in line

    def test_wall_clock_can_be_suppressed(self):
        stream = io.StringIO()
        printer("2", stream=stream, clock=False)("working")

        assert stream.getvalue().startswith("[2 +0:00]")


class TestProgressTracker:
    def test_emits_on_the_cadence(self):
        messages = []
        tracker = ProgressTracker(10, messages.append, every=5, unit="channels")

        for _ in range(10):
            tracker.advance()

        assert len(messages) == 2

    def test_always_emits_the_final_item(self):
        messages = []
        tracker = ProgressTracker(7, messages.append, every=5)

        for _ in range(7):
            tracker.advance()

        assert "7/7" in messages[-1]

    def test_reports_rate_and_share(self):
        messages = []
        tracker = ProgressTracker(100, messages.append, every=1)
        tracker.advance()

        assert "1/100" in messages[0]
        assert "%" in messages[0]
        assert "/s" in messages[0]

    def test_finish_summarises(self):
        messages = []
        tracker = ProgressTracker(2, messages.append, every=100)
        tracker.advance()
        tracker.finish()

        assert "done:" in messages[-1]


class TestAceEvaluationReader:
    def test_substitutes_symbol_and_zaid(self):
        reader = AceEvaluationReader("endf8", "/lib/<symbol>/<ZAID>.800nc")

        assert reader.path_for("Li", "3007") == Path("/lib/Li/3007.800nc")

    def test_capitalises_the_element_directory(self):
        """ACE libraries use Li/, not li/ — normalising here avoids per-call-site fixes."""
        reader = AceEvaluationReader("endf8", "/lib/<symbol>/<ZAID>.800nc")

        assert reader.path_for("li", "3007") == Path("/lib/Li/3007.800nc")

    def test_returns_none_for_a_missing_file(self, tmp_path):
        """A corpus gap must not raise: it would abort a multi-hour run."""
        reader = AceEvaluationReader("endf8", str(tmp_path / "<symbol>" / "<ZAID>.ace"))

        assert reader.read("Li", "3007", 1, np.array([1.0])) is None

    def test_real_cluster_templates_substitute_correctly(self):
        from pipeline_config import DEFAULT_ENDF8

        path = AceEvaluationReader("endf8", DEFAULT_ENDF8).path_for("U", "92235")

        assert path.name == "92235.800nc"
        assert path.parent.name == "U"


class TestNullEvaluationReader:
    def test_always_returns_none(self):
        assert NullEvaluationReader("endf8").read("Li", "3007", 1, np.array([1.0])) is None


class TestSyntheticEvaluationReader:
    def test_returns_a_predictable_cross_section(self):
        """Analytic, so downstream chi-squared has a value that can be asserted."""
        result = SyntheticEvaluationReader(scale=2.0).read(
            "Li", "3007", 1, np.array([4.0, 16.0]))

        assert result.interpolated == pytest.approx([1.0, 0.5])

    def test_matches_the_query_length(self):
        energies = np.array([1.0, 2.0, 3.0])

        result = SyntheticEvaluationReader().read("Li", "3007", 1, energies)

        assert len(result.interpolated) == len(energies)


class TestReadersFromConfig:
    def test_builds_both_evaluations_in_column_order(self, tmp_path):
        for name in ("e71", "e8"):
            (tmp_path / name).mkdir()
        config = Config.resolved(
            endf71_template=str(tmp_path / "e71" / "<symbol>" / "<ZAID>"),
            endf8_template=str(tmp_path / "e8" / "<symbol>" / "<ZAID>"))

        readers = readers_from_config(config)

        assert [r.name for r in readers] == ["endf7-1", "endf8"]
        assert all(isinstance(r, AceEvaluationReader) for r in readers)

    def test_allow_missing_evals_substitutes_null_readers(self, tmp_path):
        config = Config.resolved(allow_missing_evals=True,
                                 endf71_template=str(tmp_path / "gone" / "<symbol>"),
                                 endf8_template=str(tmp_path / "gone" / "<symbol>"))

        readers = readers_from_config(config)

        assert all(isinstance(r, NullEvaluationReader) for r in readers)

    def test_mixed_availability_is_representable(self, tmp_path):
        """The branch the has_endf7-1 / has_endf8 summary flags depend on."""
        (tmp_path / "e8").mkdir()
        config = Config.resolved(allow_missing_evals=True,
                                 endf71_template=str(tmp_path / "gone" / "<symbol>"),
                                 endf8_template=str(tmp_path / "e8" / "<symbol>"))

        readers = readers_from_config(config)

        assert isinstance(readers[0], NullEvaluationReader)
        assert isinstance(readers[1], AceEvaluationReader)


class TestNoClusterImports:
    def test_evaluations_imports_without_openmc(self):
        """OpenMC is cluster-only; importing the seam on a laptop must not need it."""
        code = (
            "import sys;"
            "sys.modules['openmc'] = None;"          # poison the import
            "import evaluations, stage_result, pipeline_config;"
            "print('ok')"
        )
        result = subprocess.run([sys.executable, "-c", code],
                                cwd=Path(__file__).parent,
                                capture_output=True, text=True)

        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout


class TestTimeBasedProgress:
    """A count-only cadence goes silent for as long as the work takes, so a merely slow job
    becomes indistinguishable from a hung one."""

    def test_emits_when_the_interval_elapses_before_the_count(self):
        messages = []
        tracker = ProgressTracker(1000, messages.append, every=500, min_interval=0.0)

        tracker.advance()

        assert len(messages) == 1, "a long-quiet log must report before the count is due"

    def test_stays_quiet_when_neither_trigger_is_due(self):
        messages = []
        tracker = ProgressTracker(1000, messages.append, every=500, min_interval=3600)

        for _ in range(10):
            tracker.advance()

        assert messages == []

    def test_heartbeat_reports_only_after_the_interval(self):
        messages = []
        tracker = ProgressTracker(10, messages.append, every=1000, min_interval=3600)

        tracker.heartbeat()

        assert messages == []

    def test_heartbeat_reports_when_the_log_has_gone_quiet(self):
        messages = []
        tracker = ProgressTracker(10, messages.append, every=1000, min_interval=0.0)

        tracker.heartbeat("still working")

        assert len(messages) == 1
        assert "still working" in messages[0]


class TestStep:
    def test_announces_before_and_after(self):
        messages = []

        with step(messages.append, "reading X4Pro"):
            pass

        assert messages[0].endswith("...")
        assert "done" in messages[1]

    def test_reports_a_failure_with_its_duration(self):
        """A stage that dies mid-operation should say which one."""
        messages = []

        with pytest.raises(ValueError):
            with step(messages.append, "reading X4Pro"):
                raise ValueError("disk gone")

        assert "FAILED" in messages[1]
        assert "disk gone" in messages[1]


class TestPrinterClock:
    def test_line_carries_wall_clock_and_elapsed(self):
        stream = io.StringIO()
        printer("1", stream=stream)("hello")

        line = stream.getvalue()
        assert ":" in line.split("]")[0]      # HH:MM:SS
        assert "+0:00" in line
