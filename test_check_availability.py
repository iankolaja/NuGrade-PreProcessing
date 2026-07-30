"""Tests for the availability checker. No network: HTTP responses are stubbed."""
import io
import json
from unittest.mock import patch

import pytest

from check_availability import (
    check_entry,
    check_osti,
    check_unpaywall,
    normalise_report_number,
)


def stub(payload):
    return patch("urllib.request.urlopen",
                 return_value=io.BytesIO(json.dumps(payload).encode()))


OPEN_WITH_PDF = {
    "is_oa": True, "oa_status": "green", "journal_name": "Physical Review",
    "best_oa_location": {"url_for_pdf": "https://repo.example/paper.pdf",
                         "url": "https://repo.example/paper", "host_type": "repository"},
}
OPEN_NO_PDF = {
    "is_oa": True, "oa_status": "bronze", "journal_name": "Nuclear Physics A",
    "best_oa_location": {"url_for_pdf": None, "url": "https://publisher.example/paper",
                         "host_type": "publisher"},
}
CLOSED = {"is_oa": False, "oa_status": "closed", "journal_name": "Physical Review",
          "best_oa_location": None}

# Verbatim shape from OSTI for ORNL-4805, which EXFOR cites as entry 10283.
OSTI_FULLTEXT = [{
    "osti_id": "4567", "report_number": "ORNL--4805",
    "title": "Nitrogen neutron elastic and inelastic scattering cross sections",
    "links": [{"rel": "citation", "href": "https://www.osti.gov/biblio/4567"},
              {"rel": "fulltext", "href": "https://www.osti.gov/servlets/purl/4567"}],
}]
OSTI_RECORD_ONLY = [{
    "osti_id": "99", "report_number": "WAPD-TM--837", "title": "Integral measurements",
    "links": [{"rel": "citation", "href": "https://www.osti.gov/biblio/99"}],
}]
OSTI_FUZZY_MISS = [{
    "osti_id": "1", "report_number": "DOE_Jung-Liu_DE-SC0023700",
    "title": "Final Scientific and Technical Report", "links": [],
}]


class TestNormaliseReportNumber:
    def test_collapses_doubled_dashes(self):
        """OSTI writes ORNL--4805 where EXFOR writes ORNL-4805."""
        assert normalise_report_number("ORNL--4805") == normalise_report_number("ORNL-4805")

    def test_upper_cases(self):
        assert normalise_report_number("ornl-4805") == "ORNL-4805"

    def test_handles_none(self):
        assert normalise_report_number(None) == ""


class TestUnpaywall:
    def test_open_with_a_pdf(self):
        with stub(OPEN_WITH_PDF):
            result = check_unpaywall("10.1/x", "a@b.c", delay=0)

        assert result["availability"] == "open_pdf"
        assert result["url"].endswith(".pdf")
        assert result["oa_status"] == "green"

    def test_open_without_a_direct_pdf(self):
        """Still useful, but needs a human to fetch it — counted separately."""
        with stub(OPEN_NO_PDF):
            result = check_unpaywall("10.1/x", "a@b.c", delay=0)

        assert result["availability"] == "open_landing_page"

    def test_closed(self):
        with stub(CLOSED):
            result = check_unpaywall("10.1/x", "a@b.c", delay=0)

        assert result["availability"] == "closed"
        assert result["url"] == ""

    def test_unknown_doi(self):
        with stub({"error": True, "message": "not found"}):
            result = check_unpaywall("10.1/nope", "a@b.c", delay=0)

        assert result["availability"] == "not_in_unpaywall"

    def test_network_failure_returns_none(self):
        with patch("urllib.request.urlopen", side_effect=TimeoutError):
            assert check_unpaywall("10.1/x", "a@b.c", delay=0) is None


class TestOsti:
    def test_exact_match_with_fulltext(self):
        with stub(OSTI_FULLTEXT):
            result = check_osti("ORNL-4805", delay=0)

        assert result["availability"] == "osti_fulltext"
        assert "purl" in result["url"]

    def test_exact_match_without_fulltext(self):
        with stub(OSTI_RECORD_ONLY):
            result = check_osti("WAPD-TM-837", delay=0)

        assert result["availability"] == "osti_record_only"

    def test_rejects_a_fuzzy_hit_on_an_unrelated_report(self):
        """OSTI's search is loose: a query for ORNL-4805 returns unrelated documents.

        Counting those as available would claim we can obtain papers we cannot.
        """
        with stub(OSTI_FUZZY_MISS):
            result = check_osti("ORNL-4805", delay=0)

        assert result["availability"] == "not_in_osti"
        assert "no exact report-number match" in result["detail"]

    def test_no_results(self):
        with stub([]):
            assert check_osti("NOPE-1")["availability"] == "not_in_osti"

    def test_missing_code_returns_none(self):
        assert check_osti("") is None
        assert check_osti(None) is None


class TestCheckEntry:
    def test_routes_a_resolved_doi_to_unpaywall(self):
        row = {"entry": "1", "status": "doi_resolved", "doi": "10.1/x", "ref_type": "journal",
               "code": "PR"}

        with stub(OPEN_WITH_PDF):
            result = check_entry(row, "a@b.c", delay=0)

        assert result["route"] == "unpaywall"
        assert result["availability"] == "open_pdf"

    def test_routes_a_report_to_osti(self):
        row = {"entry": "2", "status": "grey_literature", "doi": "", "ref_type": "report",
               "code": "ORNL-4805"}

        with stub(OSTI_FULLTEXT):
            result = check_entry(row, "a@b.c", delay=0)

        assert result["route"] == "osti"
        assert result["availability"] == "osti_fulltext"

    @pytest.mark.parametrize("ref_type", ["private_communication", "thesis", "conference"])
    def test_types_with_nowhere_to_look_are_marked_not_attempted(self, ref_type):
        """A private communication has no service to query; that is a real answer."""
        row = {"entry": "3", "status": "grey_literature", "doi": "", "ref_type": ref_type,
               "code": "X"}

        result = check_entry(row, "a@b.c", delay=0)

        assert result["availability"] == "no_route"
        assert result["detail"] == ref_type

    def test_lookup_failure_is_distinguished_from_unavailability(self):
        """A timeout must not be recorded as 'closed' — it is unknown, and retryable."""
        row = {"entry": "4", "status": "doi_resolved", "doi": "10.1/x", "ref_type": "journal",
               "code": "PR"}

        with patch("urllib.request.urlopen", side_effect=TimeoutError):
            result = check_entry(row, "a@b.c", delay=0)

        assert result["availability"] == "lookup_failed"
