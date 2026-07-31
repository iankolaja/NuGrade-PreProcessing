"""Tests for the report fetcher.

The allowlist is the safety-critical part: fetching from a publisher platform risks the
campus IP range, not just this project. Those tests are the reason this file exists.
No network access — HTTP is stubbed.
"""
import io
import json
from unittest.mock import patch

import pytest

from fetch_reports import (
    ALLOWED_HOSTS,
    PUBLISHER_HOSTS,
    FetchError,
    classify_host,
    download,
    fetch_all,
    has_text_layer,
    write_manual_queue,
)

PDF_BYTES = b"%PDF-1.4\n%fake pdf body\n"


def stub_response(payload, content_type="application/pdf"):
    response = io.BytesIO(payload)
    response.headers = {"Content-Type": content_type}
    response.__enter__ = lambda self=response: self
    response.__exit__ = lambda *args: None
    return response


def always(payload, content_type="application/pdf"):
    """A urlopen side effect yielding a fresh response per call.

    ``return_value`` cannot be used for multi-call tests: the first ``with`` block closes
    the BytesIO, so the second call raises "I/O operation on closed file".
    """
    return lambda *args, **kwargs: stub_response(payload, content_type)


class TestClassifyHost:
    @pytest.mark.parametrize("url", [
        "https://www.osti.gov/servlets/purl/4008345",
        "https://arxiv.org/pdf/1234.5678",
        "https://escholarship.org/content/qt123/qt123.pdf",
    ])
    def test_repositories_are_allowed(self, url):
        assert classify_host(url)[1] == "allowed"

    @pytest.mark.parametrize("url", [
        "https://www.tandfonline.com/doi/pdf/10.1080/x",
        "https://link.aps.org/pdf/10.1103/PhysRev.76.1750",
        "https://www.degruyter.com/document/doi/x/pdf",
        "https://link.springer.com/content/pdf/10.1007/x.pdf",
    ])
    def test_publisher_platforms_are_not_fetched(self, url):
        """Open access describes the licence on the content, not permission to script the
        platform. Taylor & Francis and APS actively block automated traffic, and a block
        lands on the whole campus IP range."""
        assert classify_host(url)[1] == "publisher"

    def test_doi_resolver_counts_as_a_publisher(self):
        """doi.org redirects to a publisher, so following it would defeat the allowlist."""
        assert classify_host("https://doi.org/10.1103/PhysRev.76.1750")[1] == "publisher"

    def test_unknown_hosts_are_not_fetched(self):
        """Default deny: a host nobody has vetted is treated as manual."""
        assert classify_host("https://example.com/paper.pdf")[1] == "unknown"

    def test_allow_and_publisher_lists_do_not_overlap(self):
        assert not (set(ALLOWED_HOSTS) & set(PUBLISHER_HOSTS))


class TestDownload:
    def test_saves_a_valid_pdf(self, tmp_path):
        destination = tmp_path / "10009.pdf"

        with patch("urllib.request.urlopen", return_value=stub_response(PDF_BYTES)):
            size = download("https://www.osti.gov/x", destination, email="a@b.c")

        assert size == len(PDF_BYTES)
        assert destination.read_bytes().startswith(b"%PDF-")

    def test_rejects_html_served_with_a_200(self, tmp_path):
        """Repositories answer a blocked document with a 200 and a login page, so the
        status code alone cannot be trusted — the magic bytes decide."""
        destination = tmp_path / "x.pdf"
        html = stub_response(b"<!DOCTYPE html><html>Access denied", "text/html")

        with patch("urllib.request.urlopen", return_value=html):
            with pytest.raises(FetchError, match="not a PDF"):
                download("https://hdl.handle.net/x", destination, email="a@b.c")

        assert not destination.exists()

    def test_reports_an_http_error(self, tmp_path):
        import urllib.error

        error = urllib.error.HTTPError("u", 403, "Forbidden", {}, None)
        with patch("urllib.request.urlopen", side_effect=error):
            with pytest.raises(FetchError, match="HTTP 403"):
                download("https://x/y", tmp_path / "x.pdf", email="a@b.c")

    def test_identifies_itself(self, tmp_path):
        """A shared research service should be able to contact us rather than block us."""
        captured = {}

        def capture(request, timeout=None):
            captured["ua"] = request.get_header("User-agent")
            return stub_response(PDF_BYTES)

        with patch("urllib.request.urlopen", side_effect=capture):
            download("https://www.osti.gov/x", tmp_path / "x.pdf", email="me@berkeley.edu")

        assert "me@berkeley.edu" in captured["ua"]


class TestHasTextLayer:
    def test_missing_pymupdf_returns_unknown_not_false(self, tmp_path):
        """Unknown must not be reported as 'ready to embed'; that overstates the corpus."""
        path = tmp_path / "x.pdf"
        path.write_bytes(PDF_BYTES)

        with patch.dict("sys.modules", {"fitz": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                assert has_text_layer(path) is None

    def test_unreadable_file_is_false(self, tmp_path):
        path = tmp_path / "x.pdf"
        path.write_bytes(b"not really a pdf")

        assert has_text_layer(path) is False


class TestFetchAll:
    def _queue(self):
        return [
            {"entry": "10009", "url": "https://www.osti.gov/servlets/purl/1",
             "citation": "ANL-7710", "title": "T", "doi": ""},
            {"entry": "20001", "url": "https://www.tandfonline.com/doi/pdf/10.1080/x",
             "citation": "J", "title": "T", "doi": "10.1080/x"},
            {"entry": "30001", "url": "https://arxiv.org/pdf/1234.5678",
             "citation": "arXiv", "title": "T", "doi": ""},
        ]

    def test_fetches_only_allowed_hosts(self, tmp_path):
        with patch("urllib.request.urlopen", side_effect=always(PDF_BYTES)):
            results, manual = fetch_all(self._queue(), tmp_path, email="a@b.c", delay=0,
                                        emit=lambda m: None)

        assert {r["entry"] for r in results} == {"10009", "30001"}
        assert [r["entry"] for r in manual] == ["20001"]

    def test_names_files_by_exfor_entry(self, tmp_path):
        """Stage 2 discovers reports by entry number, so the filename is the join key."""
        with patch("urllib.request.urlopen", side_effect=always(PDF_BYTES)):
            fetch_all(self._queue(), tmp_path, email="a@b.c", delay=0, emit=lambda m: None)

        assert (tmp_path / "10009.pdf").is_file()

    def test_resume_skips_documents_already_present(self, tmp_path):
        (tmp_path / "10009.pdf").write_bytes(PDF_BYTES)

        with patch("urllib.request.urlopen", side_effect=always(PDF_BYTES)):
            results, _ = fetch_all(self._queue(), tmp_path, email="a@b.c", delay=0,
                                   emit=lambda m: None)

        assert {r["entry"] for r in results} == {"30001"}

    def test_a_failure_does_not_stop_the_run(self, tmp_path):
        import urllib.error

        responses = [urllib.error.HTTPError("u", 403, "no", {}, None),
                     stub_response(PDF_BYTES)]

        with patch("urllib.request.urlopen", side_effect=responses):
            results, _ = fetch_all(self._queue(), tmp_path, email="a@b.c", delay=0,
                                   emit=lambda m: None)

        assert {r["status"] for r in results} == {"failed", "fetched"}

    def test_unknown_hosts_join_the_manual_queue(self, tmp_path):
        queue = [{"entry": "1", "url": "https://unvetted.example/x.pdf",
                  "citation": "", "title": "", "doi": ""}]

        with patch("urllib.request.urlopen", side_effect=always(PDF_BYTES)):
            results, manual = fetch_all(queue, tmp_path, email="a@b.c", delay=0,
                                        emit=lambda m: None)

        assert results == []
        assert len(manual) == 1


class TestManualQueue:
    def test_carries_what_a_person_needs_to_fetch_it(self, tmp_path):
        rows = [{"entry": "20001", "url": "https://www.tandfonline.com/doi/pdf/x",
                 "citation": "J. Nucl. Energy vol 14 p 186 (1961)",
                 "title": "Neutron cross sections", "doi": "10.1080/x"}]
        path = tmp_path / "manual.csv"

        write_manual_queue(rows, path)

        import csv
        row = list(csv.DictReader(path.open()))[0]
        assert row["publisher"] == "Taylor & Francis"
        assert row["doi"] == "10.1080/x"
        assert row["citation"].startswith("J. Nucl. Energy")


class TestAlternateUrls:
    """Unpaywall sometimes reports a landing page rather than a file."""

    def test_derives_the_osti_full_text_url(self):
        """OSTI shares an identifier between the record and the full text, so the second
        can be derived rather than scraped."""
        from fetch_reports import alternate_urls

        assert alternate_urls("https://www.osti.gov/biblio/1237548") == [
            "https://www.osti.gov/servlets/purl/1237548"]

    def test_handles_the_bare_domain(self):
        from fetch_reports import alternate_urls

        assert alternate_urls("https://osti.gov/biblio/99")[0].endswith("/purl/99")

    def test_offers_nothing_for_a_purl_url(self):
        """Already the full-text form; retrying it would just repeat the same request."""
        from fetch_reports import alternate_urls

        assert alternate_urls("https://www.osti.gov/servlets/purl/123") == []

    def test_does_not_guess_for_other_hosts(self):
        """Deriving a documented URL is fine; inventing one for an arbitrary host is not."""
        from fetch_reports import alternate_urls

        assert alternate_urls("https://hdl.handle.net/2027.42/32799") == []
        assert alternate_urls("https://escholarship.org/uc/item/abc") == []

    def test_falls_back_when_the_first_url_is_html(self, tmp_path):
        import urllib.error
        from fetch_reports import fetch_all

        queue = [{"entry": "14364", "url": "https://www.osti.gov/biblio/1237548",
                  "citation": "", "title": "", "doi": ""}]
        responses = [stub_response(b"<html>landing page", "text/html"),
                     stub_response(PDF_BYTES)]

        with patch("urllib.request.urlopen", side_effect=responses):
            results, _ = fetch_all(queue, tmp_path, email="a@b.c", delay=0,
                                   emit=lambda m: None)

        assert results[0]["status"] == "fetched"
        assert "purl" in results[0]["url"]
        assert results[0]["detail"] == "via derived full-text URL"

    def test_reports_failure_when_neither_url_works(self, tmp_path):
        """A citation-only OSTI record 404s on the purl form; that is a real answer."""
        import urllib.error
        from fetch_reports import fetch_all

        queue = [{"entry": "14149", "url": "https://www.osti.gov/biblio/1101778",
                  "citation": "", "title": "", "doi": ""}]
        errors = [urllib.error.HTTPError("u", 404, "no", {}, None)] * 2

        with patch("urllib.request.urlopen", side_effect=errors):
            results, _ = fetch_all(queue, tmp_path, email="a@b.c", delay=0,
                                   emit=lambda m: None)

        assert results[0]["status"] == "failed"
