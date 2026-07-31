"""Tests for the OCR pass.

The decision rule is what matters: a bad OCR pass produces confident-looking garbage, which
is worse than a document that is honestly empty, because garbage gets embedded and
retrieved. No subprocess is ever launched here.
"""
import json
from unittest.mock import patch

import pytest

import ocr_reports
from ocr_reports import OcrError, adopt, assess, clear_overrides, needs_ocr, process

GOOD_TEXT = " ".join([
    "The neutron activation cross sections were measured with a natural lithium sample.",
    "The efficiency of the detector was determined with a calibrated cobalt source.",
    "Corrections were applied for gamma ray self absorption within the sample material.",
    "The dead time of the counting system remained below one percent throughout the run.",
    "Background contributions were measured with the sample removed from the beam line.",
    "The neutron flux was monitored continuously using a fission chamber downstream.",
    "Time of flight techniques resolved the incident neutron energy to two percent.",
    "Statistical uncertainties were propagated through the full analysis chain here.",
    "The sample was irradiated for twenty four hours at a constant reactor power.",
    "Gamma ray spectra were recorded with a high purity germanium detector system.",
    "The energy resolution was measured at several convenient reference energies.",
    "Coincidence counting resolved the complex decay scheme of the reaction product.",
    "Normalization was checked against a well known standard cross section value.",
    "Scattering corrections were computed with a Monte Carlo transport calculation.",
    "Measured yields were corrected for decay during the counting interval itself.",
    "Neutron energies followed from the flight path and the timing resolution used.",
    "The uncertainty budget includes counting statistics and normalization terms.",
    "Repeated measurements agreed within the quoted statistical uncertainties given.",
    "Target thickness was measured by weighing before and after the irradiation.",
    "Systematic effects from beam drift were monitored throughout the experiment.",
    "Comparison with earlier measurements shows agreement within two deviations.",
    "The final results are tabulated together with their estimated uncertainties.",
])

GARBAGE_TEXT = " ".join(["z 3 u. 4 p u 5 (is . .- . ._. ~~ |||"] * 40)


class TestAssess:
    def test_real_prose_is_usable(self):
        kept, usable = assess(GOOD_TEXT)

        assert usable
        assert kept >= 20

    def test_ocr_garbage_is_not(self):
        kept, usable = assess(GARBAGE_TEXT)

        assert not usable

    def test_empty_text(self):
        assert assess("") == (0, False)


class TestNeedsOcr:
    def test_detects_a_document_with_no_text(self, tmp_path):
        path = tmp_path / "x.pdf"
        path.write_bytes(b"not a pdf")

        assert needs_ocr(path)

    def test_skips_a_document_that_already_has_text(self, tmp_path):
        path = tmp_path / "x.pdf"
        path.write_bytes(b"%PDF-")

        with patch("ocr_reports.extract_text", return_value=GOOD_TEXT):
            assert not needs_ocr(path)


class TestProcess:
    def _pdf(self, tmp_path, name="10104.pdf"):
        path = tmp_path / name
        path.write_bytes(b"%PDF-1.4")
        return path

    def test_adopts_an_improvement(self, tmp_path):
        source = self._pdf(tmp_path)
        work = tmp_path / "work"
        work.mkdir()

        with patch("ocr_reports.run_ocrmypdf", return_value=None), \
             patch("ocr_reports.extract_text", side_effect=["", GOOD_TEXT]), \
             patch("ocr_reports.page_count", return_value=69):
            row = process(source, work, emit=lambda m: None)

        assert row["status"] == "improved"
        assert row["sentences_before"] == 0
        assert row["sentences_after"] >= 20

    def test_rejects_output_that_is_still_garbage(self, tmp_path):
        """The decision this tool exists to make. Keeping this would be worse than the
        honestly empty document it replaces."""
        source = self._pdf(tmp_path)
        work = tmp_path / "work"
        work.mkdir()

        with patch("ocr_reports.run_ocrmypdf", return_value=None), \
             patch("ocr_reports.extract_text", side_effect=["", GARBAGE_TEXT]), \
             patch("ocr_reports.page_count", return_value=69):
            row = process(source, work, emit=lambda m: None)

        assert row["status"] == "no_improvement"

    def test_rejects_output_with_no_more_sentences(self, tmp_path):
        """Re-OCR that changes nothing is not an improvement."""
        source = self._pdf(tmp_path)
        work = tmp_path / "work"
        work.mkdir()

        with patch("ocr_reports.run_ocrmypdf", return_value=None), \
             patch("ocr_reports.extract_text", side_effect=[GOOD_TEXT, GOOD_TEXT]), \
             patch("ocr_reports.page_count", return_value=10):
            row = process(source, work, emit=lambda m: None)

        assert row["status"] == "no_improvement"

    def test_a_failure_is_recorded_not_raised(self, tmp_path):
        """One unreadable scan must not stop a batch that runs for an hour."""
        source = self._pdf(tmp_path)
        work = tmp_path / "work"
        work.mkdir()

        with patch("ocr_reports.run_ocrmypdf", side_effect=OcrError("timed out")), \
             patch("ocr_reports.extract_text", return_value=""), \
             patch("ocr_reports.page_count", return_value=200):
            row = process(source, work, emit=lambda m: None)

        assert row["status"] == "failed"
        assert "timed out" in row["detail"]


class TestAdopt:
    def test_keeps_the_original_alongside(self, tmp_path):
        """Never overwrite a source document irrecoverably."""
        pdfs = tmp_path / "pdfs"
        work = tmp_path / "work"
        pdfs.mkdir()
        work.mkdir()
        (pdfs / "10104.pdf").write_bytes(b"original scan")
        (work / "10104.pdf").write_bytes(b"ocr'd version")

        adopt("10104", work, pdfs)

        assert (pdfs / "10104.pdf").read_bytes() == b"ocr'd version"
        assert (work / "10104.original.pdf").read_bytes() == b"original scan"

    def test_does_not_clobber_an_existing_backup(self, tmp_path):
        pdfs = tmp_path / "pdfs"
        work = tmp_path / "work"
        pdfs.mkdir()
        work.mkdir()
        (pdfs / "1.pdf").write_bytes(b"second pass")
        (work / "1.pdf").write_bytes(b"ocr")
        (work / "1.original.pdf").write_bytes(b"true original")

        adopt("1", work, pdfs)

        assert (work / "1.original.pdf").read_bytes() == b"true original"


class TestClearOverrides:
    def test_prunes_rules_that_no_longer_apply(self, tmp_path):
        """A stale skip rule silently excludes a document that is no longer broken — the
        failure an override file invites if nobody prunes it."""
        path = tmp_path / "o.json"
        path.write_text(json.dumps({"reports": {
            "10104": {"reason": "image-only scan", "skip": True, "action": "ocr"},
            "99999": {"reason": "cited chapter", "pages": "1-5"},
        }}))

        removed = clear_overrides(["10104"], path=path)

        reports = json.loads(path.read_text())["reports"]
        assert removed == 1
        assert "10104" not in reports
        assert "99999" in reports, "unrelated rules must survive"

    def test_leaves_non_ocr_rules_alone(self, tmp_path):
        path = tmp_path / "o.json"
        path.write_text(json.dumps({"reports": {
            "1": {"reason": "non-English", "skip": True, "language": "fr"}}}))

        removed = clear_overrides(["1"], path=path)

        assert removed == 0
        assert "1" in json.loads(path.read_text())["reports"]

    def test_missing_file_is_harmless(self, tmp_path):
        assert clear_overrides(["1"], path=tmp_path / "absent.json") == 0


class TestGroupByContent:
    """Several EXFOR entries routinely cite the same report; OCR is the slow step."""

    def test_groups_identical_files(self, tmp_path):
        for name in ("20183.pdf", "20184.pdf", "20185.pdf"):
            (tmp_path / name).write_bytes(b"same 208-page scan")
        (tmp_path / "20187.pdf").write_bytes(b"a different document")

        groups = ocr_reports.group_by_content(sorted(tmp_path.glob("*.pdf")))

        assert len(groups) == 2
        sizes = sorted(len(members) for _, members in groups)
        assert sizes == [1, 3]

    def test_representative_comes_first(self, tmp_path):
        (tmp_path / "a.pdf").write_bytes(b"x")
        (tmp_path / "b.pdf").write_bytes(b"x")

        (representative, members) = ocr_reports.group_by_content(
            sorted(tmp_path.glob("*.pdf")))[0]

        assert representative == members[0]

    def test_distinct_files_are_not_merged(self, tmp_path):
        (tmp_path / "a.pdf").write_bytes(b"one")
        (tmp_path / "b.pdf").write_bytes(b"two")

        assert len(ocr_reports.group_by_content(sorted(tmp_path.glob("*.pdf")))) == 2
