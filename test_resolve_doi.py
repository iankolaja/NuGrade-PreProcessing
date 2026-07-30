"""Tests for DOI resolution.

No network access: the verification logic is what matters, so the HTTP layer is stubbed with
real response shapes captured from ADS and Crossref.
"""
import json
import io
from unittest.mock import patch

import pytest

from exfor_bib import parse_reference
from resolve_doi import (
    BIBSTEMS,
    UNRESOLVABLE_BY_STRUCTURE,
    _strip_section_letter,
    _titles_agree,
    load_ads_token,
    resolve,
    resolve_via_ads,
    resolve_via_crossref,
)

MELKONIAN_REF = parse_reference("(J,PR,76,1750,4912)")
MELKONIAN_TITLE = "Slow Neutron Velocity Spectrometer Studies of O2, N2, A, H2, H2O"

# Real ADS response shape: most fields arrive as single-element lists.
ADS_HIT = {"response": {"numFound": 1, "docs": [{
    "bibcode": "1949PhRv...76.1750M",
    "doi": ["10.1103/PhysRev.76.1750"],
    "title": ["Slow Neutron Velocity Spectrometer Studies of O2, N2, A, H2"],
    "volume": "76", "page": ["1750"], "year": "1949",
}]}}

# The wrong article Crossref actually returned for this paper: same volume, different page.
ADS_WRONG_PAGE = {"response": {"numFound": 1, "docs": [{
    "bibcode": "1949PhRv...76.1744X", "doi": ["10.1103/PhysRev.76.1744"],
    "title": ["Some other paper"], "volume": "76", "page": ["1744"],
}]}}

CROSSREF_HIT = {"message": {"items": [
    {"DOI": "10.1103/physrev.76.1750", "title": ["Slow Neutron Velocity Spectrometer"],
     "volume": "76", "page": "1750-1760"},
]}}

CROSSREF_WRONG = {"message": {"items": [
    {"DOI": "10.1103/physrev.76.1744", "title": ["Unrelated"], "volume": "76", "page": "1744"},
]}}


def stub(payload):
    """Patch urlopen to return ``payload`` as JSON."""
    return patch("urllib.request.urlopen",
                 return_value=io.BytesIO(json.dumps(payload).encode()))


class TestResolveViaAds:
    def test_accepts_a_volume_and_page_match(self):
        with stub(ADS_HIT):
            match = resolve_via_ads(MELKONIAN_REF, token="x", delay=0)

        assert match["doi"] == "10.1103/PhysRev.76.1750"
        assert match["bibcode"] == "1949PhRv...76.1750M"
        assert match["source"] == "ads"
        assert "page:1750" in match["evidence"]

    def test_rejects_a_wrong_page_in_the_same_volume(self):
        """The concrete failure mode: Crossref returned 76.1744 for a paper on 76.1750."""
        with stub(ADS_WRONG_PAGE):
            match = resolve_via_ads(MELKONIAN_REF, token="x", delay=0)

        assert match is None, "a same-volume different-page article must be rejected"

    def test_returns_none_without_a_bibstem_mapping(self):
        """Atomnaya Energiya has no mapping; guessing one risks a wrong match."""
        reference = parse_reference("(J,AE,15,416,1963)")

        with stub(ADS_HIT):
            assert resolve_via_ads(reference, token="x", delay=0) is None

    def test_returns_none_without_volume_or_page(self):
        reference = parse_reference("(J,PR,,,1949)")

        with stub(ADS_HIT):
            assert resolve_via_ads(reference, token="x", delay=0) is None


class TestResolveViaCrossref:
    def test_accepts_when_volume_and_first_page_agree(self):
        """Crossref gives page ranges; only the first page is compared."""
        with stub(CROSSREF_HIT):
            match = resolve_via_crossref(MELKONIAN_REF, MELKONIAN_TITLE, delay=0)

        assert match["doi"] == "10.1103/physrev.76.1750"
        assert match["source"] == "crossref"

    def test_rejects_a_mismatched_page(self):
        with stub(CROSSREF_WRONG):
            assert resolve_via_crossref(MELKONIAN_REF, MELKONIAN_TITLE, delay=0) is None

    def test_requires_a_title(self):
        with stub(CROSSREF_HIT):
            assert resolve_via_crossref(MELKONIAN_REF, None, delay=0) is None


class TestResolve:
    def test_prefers_ads_when_a_token_is_present(self):
        bib = {"reference": MELKONIAN_REF, "title": MELKONIAN_TITLE}

        with stub(ADS_HIT):
            match = resolve(bib, token="x", delay=0)

        assert match["source"] == "ads"

    def test_falls_back_to_crossref_without_a_token(self):
        bib = {"reference": MELKONIAN_REF, "title": MELKONIAN_TITLE}

        with stub(CROSSREF_HIT):
            match = resolve(bib, token=None, delay=0)

        assert match["source"] == "crossref"

    @pytest.mark.parametrize("reference", [
        "(R,INDC(GER)-12,1975)",
        "(C,87KIEV,2,298,1987)",
        "(W,PRIVATE.COMM,1975)",
    ])
    def test_skips_grey_literature(self, reference):
        """Reports, conferences and private communications have no DOI to find."""
        bib = {"reference": parse_reference(reference), "title": "something"}

        assert resolve(bib, token="x", delay=0) is None

    def test_handles_missing_reference(self):
        assert resolve({"reference": None, "title": "t"}, token="x", delay=0) is None
        assert resolve(None, token="x", delay=0) is None


class TestLoadAdsToken:
    def test_returns_none_when_absent(self, tmp_path):
        assert load_ads_token(tmp_path / "nope.txt") is None

    def test_strips_trailing_newline(self, tmp_path):
        path = tmp_path / "ads_token.txt"
        path.write_text("ABC123\n")

        assert load_ads_token(path) == "ABC123"

    def test_treats_empty_file_as_absent(self, tmp_path):
        path = tmp_path / "ads_token.txt"
        path.write_text("   \n")

        assert load_ads_token(path) is None


class TestBibstems:
    def test_maps_the_journals_that_dominate_the_corpus(self):
        for code in ["PR", "PR/C", "NP/A", "ZP/A", "NC/B", "NIM"]:
            assert code in BIBSTEMS

    def test_no_mapping_is_empty(self):
        assert all(v for v in BIBSTEMS.values())

    def test_does_not_reintroduce_the_zero_record_guesses(self):
        """JNucE and NucSE look right but have no records in ADS.

        Both were verified against `bibstem:<value>` on 2026-07-29: JNucE and NucSE return
        0, while JNuE returns 1,064 and NSE returns 10,418. Guessing cost two resolutions
        in the sample, so pin the verified values.
        """
        assert BIBSTEMS["JNE"] == "JNuE"
        assert BIBSTEMS["NSE"] == "NSE"
        assert "JNucE" not in BIBSTEMS.values()
        assert "NucSE" not in BIBSTEMS.values()

    def test_survey_derived_mappings_are_present(self):
        """Added after the 300-entry survey; each verified against a real volume/page."""
        for code, bibstem in [("AP", "AnPhy"), ("JRN", "JRNC"), ("ZN/A", "ZNatA"),
                              ("PRS/A", "RSPSA"), ("NST", "JNST"), ("JP/A", "JPhA"),
                              ("FBS", "FBS"), ("EPJ/A", "EPJA")]:
            assert BIBSTEMS[code] == bibstem

    def test_soviet_journals_are_not_mapped(self):
        """Not an oversight: translations renumber volumes, so structure cannot match.

        AtEne, SvAtE, AtEn, JETP, ZhETF, SvJNP and BASUP were all tested against real
        volume/page pairs from this corpus and none matched. Adding a plausible bibstem
        here would produce wrong articles, which is worse than leaving them unresolved.
        """
        for code in UNRESOLVABLE_BY_STRUCTURE:
            assert code not in BIBSTEMS, f"{code} needs title matching, not a bibstem"

    def test_journal_sections_share_a_bibstem(self):
        """EXFOR splits J.Nucl.Energy into parts A/B; ADS indexes them under one bibstem."""
        assert BIBSTEMS["JNE/AB"] == BIBSTEMS["JNE"]


class TestSectionPrefixedPages:
    """1960s Physical Review split volumes into sections A and B with separate page runs.

    EXFOR cites "B353"; ADS stores "353". Verified against entry 11122: querying
    page:B353 returns nothing, page:353 returns 1964PhRv..133..353H.
    """

    def test_strips_a_section_letter(self):
        assert _strip_section_letter("B353") == "353"
        assert _strip_section_letter("A1277") == "1277"

    def test_leaves_a_plain_page_alone(self):
        assert _strip_section_letter("1750") is None
        assert _strip_section_letter("917(19)") is None

    def test_retries_with_the_letter_stripped_when_titles_agree(self):
        reference = parse_reference("(J,PR,133,B353,1964)")
        no_hit = {"response": {"docs": []}}
        hit = {"response": {"docs": [{
            "bibcode": "1964PhRv..133..353H", "doi": ["10.1103/PhysRev.133.B353"],
            "title": ["Nuclear Levels in F20"], "volume": "133", "page": ["353"],
        }]}}
        responses = [io.BytesIO(json.dumps(p).encode()) for p in (no_hit, hit)]

        with patch("urllib.request.urlopen", side_effect=responses):
            match = resolve_via_ads(reference, token="x",
                                    title="Nuclear Levels in F20", delay=0)

        assert match["doi"] == "10.1103/PhysRev.133.B353"
        assert "section letter stripped" in match["evidence"]

    def test_rejects_the_stripped_match_when_titles_disagree(self):
        """Sections A and B can both have a page 353, so the title must confirm it."""
        reference = parse_reference("(J,PR,133,B353,1964)")
        no_hit = {"response": {"docs": []}}
        wrong = {"response": {"docs": [{
            "bibcode": "1964PhRv..133..353X", "doi": ["10.1103/PhysRev.133.A353"],
            "title": ["An entirely different paper about something else"],
            "volume": "133", "page": ["353"],
        }]}}
        responses = [io.BytesIO(json.dumps(p).encode()) for p in (no_hit, wrong)]

        with patch("urllib.request.urlopen", side_effect=responses):
            match = resolve_via_ads(reference, token="x",
                                    title="Nuclear Levels in F20", delay=0)

        assert match is None

    def test_does_not_retry_without_a_title(self):
        """No title means no way to confirm, so the relaxed match is not attempted."""
        reference = parse_reference("(J,PR,133,B353,1964)")
        no_hit = {"response": {"docs": []}}

        with patch("urllib.request.urlopen",
                   side_effect=[io.BytesIO(json.dumps(no_hit).encode())]) as mock:
            assert resolve_via_ads(reference, token="x", title=None, delay=0) is None
            assert mock.call_count == 1, "must not make a second request"


class TestTitlesAgree:
    def test_matches_despite_markup_and_case(self):
        assert _titles_agree("Nuclear Levels in F<SUB>20</SUB>", "NUCLEAR LEVELS IN F20")

    def test_rejects_unrelated_titles(self):
        assert not _titles_agree("Nuclear Levels in F20", "Detector efficiency calibration")

    def test_handles_empty(self):
        assert not _titles_agree("", "something")
        assert not _titles_agree(None, "something")
