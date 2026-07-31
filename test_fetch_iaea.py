"""Tests for the IAEA repository fetcher.

The matching rule is what matters here: a wrong report is worse than no report, because it
passes every quality gate and gets embedded as that entry's experiment. No network access.
"""
import io
import json
from unittest.mock import patch

import pytest

from fetch_iaea import (
    IaeaError,
    download_file,
    is_searchable,
    matching_file,
    normalise_code,
    report_code,
)


def hit(record_id, *, identifiers=(), title="", files=()):
    return {
        "id": record_id,
        "metadata": {"title": title,
                     "identifiers": [{"identifier": i} for i in identifiers]},
        "files": {"entries": {f: {} for f in files}},
    }


class TestNormaliseCode:
    def test_ignores_punctuation_differences(self):
        """EXFOR writes INDC(NOR)-1; the repository names the file indc-nor-0001G.pdf."""
        assert normalise_code("INDC(NOR)-1") == normalise_code("INDC-NOR-1")

    def test_ignores_leading_zeros(self):
        assert normalise_code("INDC(EUR)-001") == normalise_code("INDC(EUR)-1")

    def test_is_case_insensitive(self):
        assert normalise_code("EANDC(E)-89") == normalise_code("eandc(e)-89")

    def test_distinguishes_different_numbers(self):
        """The failure that matters: reports in a series differ only by number."""
        assert normalise_code("EANDC(E)-89") != normalise_code("EANDC(E)-98")

    def test_handles_empty(self):
        assert normalise_code(None) == ""


class TestReportCode:
    @pytest.mark.parametrize("citation,expected", [
        ("INDC(NOR)-1 p 2 (1972)", "INDC(NOR)-1"),
        ("EANDC(E)-89 (1968)", "EANDC(E)-89"),
        ("EANDC-33", "EANDC-33"),
        ("ANL-7710 vol 9 (1971)", "ANL-7710"),
    ])
    def test_strips_page_and_year_decoration(self, citation, expected):
        assert report_code(citation) == expected

    def test_handles_empty(self):
        assert report_code("") == ""


class TestIsSearchable:
    @pytest.mark.parametrize("citation", [
        "INDC(NOR)-1 p 2 (1972)", "EANDC(E)-89 (1968)", "IAEA-TECDOC-123",
    ])
    def test_recognises_series_iaea_holds(self, citation):
        assert is_searchable(citation)

    @pytest.mark.parametrize("citation", [
        "ORNL-4805 (1973)", "ANL-7710 vol 9 (1971)", "Physical Review vol 76 p 1750",
    ])
    def test_skips_series_it_does_not(self, citation):
        """Guessing wastes requests on a shared service and invites bad matches."""
        assert not is_searchable(citation)


class TestMatchingFile:
    def test_accepts_an_exact_identifier(self):
        hits = [hit("rec1", identifiers=["EANDC(E)-89"], files=["indc-eur-0001.pdf"])]

        assert matching_file(hits, "EANDC(E)-89") == ("rec1", "indc-eur-0001.pdf",
                                                      "identifier EANDC(E)-89")

    def test_resolves_a_renamed_report(self):
        """IAEA renamed EANDC to INDC and records both, so an EXFOR citation of the old
        code still resolves. Verified against four reports whose first pages name the code
        EXFOR cites."""
        hits = [hit("rec1", identifiers=["EANDC(US)-62", "INDC(US)*012"],
                    files=["indc-usa-asterisk012.pdf"])]

        record, name, evidence = matching_file(hits, "EANDC(US)-62")

        assert record == "rec1"
        assert "EANDC(US)-62" in evidence

    def test_rejects_a_neighbouring_report_in_the_same_series(self):
        """The central risk: searching for one report returns its neighbours, and they are
        all plausible-looking hits."""
        hits = [hit("rec1", identifiers=["EANDC(E)-98"], title="Progress Report",
                    files=["indc-eur-0098.pdf"])]

        assert matching_file(hits, "EANDC(E)-89") is None

    def test_prefers_an_identifier_over_a_weaker_match(self):
        hits = [
            hit("weak", title="EANDC(E)-89 mentioned in passing",
                files=["something-else.pdf"]),
            hit("strong", identifiers=["EANDC(E)-89"], files=["indc-eur-0001.pdf"]),
        ]

        record, _, evidence = matching_file(hits, "EANDC(E)-89")

        assert record == "strong"
        assert evidence.startswith("identifier")

    def test_falls_back_to_a_filename_match(self):
        hits = [hit("rec1", files=["indc-nor-0001G.pdf"])]

        record, name, evidence = matching_file(hits, "INDC(NOR)-1")

        assert record == "rec1"
        assert evidence == "filename"

    def test_ignores_a_record_with_no_pdf(self):
        hits = [hit("rec1", identifiers=["EANDC(E)-89"], files=["data.zip"])]

        assert matching_file(hits, "EANDC(E)-89") is None

    def test_no_hits_is_no_match(self):
        assert matching_file([], "EANDC(E)-89") is None

    def test_empty_code_never_matches(self):
        """An unparsed citation must not match the first thing returned."""
        hits = [hit("rec1", identifiers=["ANYTHING"], files=["x.pdf"])]

        assert matching_file(hits, "") is None


class TestDownloadFile:
    def test_saves_a_pdf(self, tmp_path):
        response = io.BytesIO(b"%PDF-1.4 body")
        response.__enter__ = lambda self=response: self
        response.__exit__ = lambda *a: None
        destination = tmp_path / "20151.pdf"

        with patch("urllib.request.urlopen", return_value=response):
            size = download_file("rec1", "indc-eur-0001.pdf", destination, delay=0)

        assert size > 0
        assert destination.read_bytes().startswith(b"%PDF-")

    def test_rejects_a_non_pdf_response(self, tmp_path):
        response = io.BytesIO(b"<!DOCTYPE html>")
        response.__enter__ = lambda self=response: self
        response.__exit__ = lambda *a: None

        with patch("urllib.request.urlopen", return_value=response):
            with pytest.raises(IaeaError, match="not a PDF"):
                download_file("rec1", "x.pdf", tmp_path / "x.pdf", delay=0)

    def test_quotes_a_filename_with_special_characters(self, tmp_path):
        """Real filenames contain asterisks, e.g. indc-usa-asterisk012.pdf."""
        captured = {}

        def capture(request, timeout=None):
            captured["url"] = request.full_url
            response = io.BytesIO(b"%PDF-1.4")
            response.__enter__ = lambda self=response: self
            response.__exit__ = lambda *a: None
            return response

        with patch("urllib.request.urlopen", side_effect=capture):
            download_file("rec1", "indc eur*003.pdf", tmp_path / "x.pdf", delay=0)

        assert " " not in captured["url"]
