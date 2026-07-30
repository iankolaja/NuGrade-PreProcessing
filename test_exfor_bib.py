"""Tests for EXFOR bibliographic parsing.

Every fixture is a verbatim record fetched from IAEA, so the parser is tested against the
real format rather than an idealised version of it. No network access is used here.
"""
import pytest

from exfor_bib import (
    fetch_bib,
    is_resolvable_to_doi,
    parse_bib,
    parse_reference,
)

# Verbatim record for entry 11150 (Melkonian, Physical Review 76, 1750).
MELKONIAN = """ENTRY            11150     800814              20050926       0000
SUBENT        11150001     800814              20050926       0000
BIB                  6          8
INSTITUTE  (1USACOL)
REFERENCE  (J,PR,76,1750,4912)
AUTHOR     (E.MELKONIAN)
TITLE      SLOW NEUTRON VELOCITY SPECTROMETER STUDIES OF O2, N2,
           A, H2 ,H2O AND SEVEN HYDROCARBONS.
STATUS     (SCSRS)
HISTORY    (760628T) TRANSLATED FROM SCISRS
           (800814A) CONVERTED TO REACTION FORMALISM
ENDBIB               8
NOCOMMON             0          0
ENDSUBENT           11
ENDENTRY             1
"""

MULTI_AUTHOR = """ENTRY            22217
SUBENT        22217001
BIB                  6          8
INSTITUTE  (2GERMUN)
REFERENCE  (J,ZP/A,337,341,1990)
AUTHOR     (L.Koester,W.Waschkowski,J.Meier)
TITLE      Cross sections for neutrons of 1970 eV and contributions
           to the mean free path.
ENDBIB               8
"""


class TestParseReference:
    def test_plain_journal(self):
        result = parse_reference("(J,PR,74,364,1948)")

        assert result["type"] == "journal"
        assert result["journal"] == "Physical Review"
        assert result["volume"] == "74"
        assert result["page"] == "364"
        assert result["year"] == 1948

    def test_journal_with_section(self):
        result = parse_reference("(J,PR/C,3,1886,197105)")

        assert result["journal"] == "Physical Review C"
        assert result["volume"] == "3"
        assert result["page"] == "1886"
        assert result["year"] == 1971

    def test_journal_with_issue(self):
        """(J,NP/A,258,(1),1,7602) — the parenthesised field is the issue, not the page."""
        result = parse_reference("(J,NP/A,258,(1),1,7602)")

        assert result["journal"] == "Nuclear Physics A"
        assert result["volume"] == "258"
        assert result["issue"] == "1"
        assert result["page"] == "1"
        assert result["year"] == 1976

    def test_issue_attached_to_page_without_a_comma(self):
        """(J,PR,59,917(19),1941) — real record for entry 11178; page and issue are joined.

        Parsing this as page='917(19)' makes volume+page matching against Crossref fail,
        which understates how much of the corpus is resolvable.
        """
        result = parse_reference("(J,PR,59,917(19),1941)")

        assert result["volume"] == "59"
        assert result["page"] == "917"
        assert result["issue"] == "19"

    def test_issue_label_containing_a_slash(self):
        """(J,FCY/L,11,(2/186),186,2014) — real record for entry 41602."""
        result = parse_reference("(J,FCY/L,11,(2/186),186,2014)")

        assert result["volume"] == "11"
        assert result["issue"] == "2/186"
        assert result["page"] == "186"
        assert result["year"] == 2014

    def test_two_digit_year_with_month(self):
        """4912 means December 1949, not the year 4912."""
        assert parse_reference("(J,PR,76,1750,4912)")["year"] == 1949

    def test_progress_report_with_nested_parentheses(self):
        """(P,EANDC(E)-66,52,196602) — the code itself contains parentheses."""
        result = parse_reference("(P,EANDC(E)-66,52,196602)")

        assert result["type"] == "progress_report"
        assert result["code"] == "EANDC(E)-66"
        assert result["year"] == 1966

    def test_conference(self):
        result = parse_reference("(C,87KIEV,2,298,1987)")

        assert result["type"] == "conference"
        assert result["code"] == "87KIEV"
        assert result["year"] == 1987

    def test_ignores_trailing_free_text(self):
        """Real records append commentary after the closing parenthesis."""
        result = parse_reference("(J,NC/B,58,402,196812) table of cross sections given")

        assert result["journal"] == "Nuovo Cimento B"
        assert result["page"] == "402"
        assert result["year"] == 1968

    def test_unknown_journal_code_is_preserved(self):
        result = parse_reference("(J,ZZZQQ,1,2,1980)")

        assert result["journal"] == "ZZZQQ", "unknown codes must not be dropped"

    def test_malformed_reference_does_not_raise(self):
        result = parse_reference("not a reference at all")

        assert result["type"] is None
        assert result["raw"] == "not a reference at all"


class TestParseBib:
    def test_extracts_all_fields(self):
        bib = parse_bib(MELKONIAN)

        assert bib["authors"] == ["E.MELKONIAN"]
        assert bib["institute"] == "1USACOL"
        assert bib["reference"]["volume"] == "76"
        assert bib["reference"]["page"] == "1750"

    def test_joins_multi_line_title(self):
        bib = parse_bib(MELKONIAN)

        assert bib["title"] == (
            "SLOW NEUTRON VELOCITY SPECTROMETER STUDIES OF O2, N2, A, H2 ,H2O "
            "AND SEVEN HYDROCARBONS"
        )

    def test_splits_multiple_authors(self):
        bib = parse_bib(MULTI_AUTHOR)

        assert bib["authors"] == ["L.Koester", "W.Waschkowski", "J.Meier"]

    def test_stops_at_endbib(self):
        """HISTORY lines after ENDBIB must not leak into the title."""
        bib = parse_bib(MELKONIAN)

        assert "CONVERTED" not in (bib["title"] or "")

    def test_missing_fields_are_none_not_errors(self):
        bib = parse_bib("ENTRY            99999\nSUBENT        99999001\nENDBIB\n")

        assert bib["title"] is None
        assert bib["reference"] is None
        assert bib["authors"] == []


class TestIsResolvableToDoi:
    def test_journal_article_is_resolvable(self):
        assert is_resolvable_to_doi(parse_bib(MELKONIAN))

    @pytest.mark.parametrize("reference", [
        "(P,EANDC(E)-66,52,196602)",
        "(C,87KIEV,2,298,1987)",
        "(W,PRIVATE.COMM,1975)",
        "(T,SMITH,1980)",
    ])
    def test_grey_literature_is_not_resolvable(self, reference):
        """Reports, conferences, theses and private communications have no DOI."""
        bib = {"reference": parse_reference(reference)}

        assert not is_resolvable_to_doi(bib)

    def test_handles_none(self):
        assert not is_resolvable_to_doi(None)
        assert not is_resolvable_to_doi({})


class TestFetchBibCaching:
    def test_reads_from_cache_without_network(self, tmp_path):
        """A cached record must be served from disk; this test would fail on a live call."""
        (tmp_path / "11150.x4").write_text(MELKONIAN)

        bib = fetch_bib("11150", cache_dir=tmp_path, delay=0)

        assert bib["reference"]["page"] == "1750"

    def test_cache_lookup_uses_the_entry_number(self, tmp_path):
        (tmp_path / "22217.x4").write_text(MULTI_AUTHOR)

        bib = fetch_bib(22217, cache_dir=tmp_path, delay=0)  # int, not str

        assert bib["authors"][0] == "L.Koester"
