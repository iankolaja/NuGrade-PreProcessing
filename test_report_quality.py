"""Tests for the PDF-text quality gates.

The rejection cases are verbatim strings taken from `sentence_embeddings` in the existing
database — real sentences that the current pipeline embedded. The acceptance cases are
real methodology sentences from the same corpus, which must survive the gate.
"""
import pytest

from report_quality import (
    clean_sentences,
    document_quality_report,
    has_ocr_damage,
    is_bad_sentence,
    looks_like_boilerplate,
    looks_like_citation,
    looks_non_english,
    ocr_damage_ratio,
    strip_reference_list,
)

# --- verbatim garbage currently in the database -------------------------------
OCR_DAMAGED = [
    "s. s. hasa~-, a. k. ci~aube¥ and m. l. sehgal physics department, aligarh muslim university",
    "all these cross-sections have been measured relative to that of 12vi(n, y)~2si whose cress-section was",
    "the (n, y) cross-sections at 24kev have been measured by many work- ers (1-~).",
]

CITATION_RESIDUE = [
    "(1) i. k. cflaubey and m. l. s]~hgal: nucl.",
    "phys., 66, 267 (1965). (~) a. k. chaubey and m. l. sxhgal: phys.",
    "n. h. lazor and w. s. lyon:",
]

NON_ENGLISH = [
    "riassunt0 si sono misurate le sezioni d'urto d'attivazione neutronica per 19 easi a 24 kev.",
    "lviii b, n. 2 11 dicembre 1968 study of the average level spacing from neutron-capture",
    "traduzione a cura della redazione.",
]

FRAGMENTS = ["high and low th", "the detector was", "table 2"]

# --- real methodology prose that MUST survive --------------------------------
GOOD_SENTENCES = [
    "in the cases of na, al, v and 82br the y-rays are in coincidence with a single b-ray "
    "and they were counted in the conventional manner using a 3 x 3 nai crystal.",
    "the efficiency of the detector was measured using the coincidence counting technique "
    "and corrected for the decay scheme of each isotope.",
    "neutron-activation cross-sections have been measured for nineteen nuclides at an "
    "effective neutron energy of 24 kev using a sb-be photoneutron source.",
    "corrections were applied for self-absorption of the gamma rays in the sample and for "
    "the finite geometry of the counting arrangement.",
    "the dead time of the counting system was determined to be less than one percent over "
    "the range of counting rates encountered in these measurements.",
]


class TestRejectsRealGarbage:
    @pytest.mark.parametrize("sentence", OCR_DAMAGED)
    def test_rejects_ocr_damaged(self, sentence):
        assert is_bad_sentence(sentence), "OCR-mangled text must not be embedded"

    @pytest.mark.parametrize("sentence", CITATION_RESIDUE)
    def test_rejects_citation_residue(self, sentence):
        assert is_bad_sentence(sentence), "bibliography entries are not methodology prose"

    @pytest.mark.parametrize("sentence", NON_ENGLISH)
    def test_rejects_non_english(self, sentence):
        assert is_bad_sentence(sentence), "SciBERT is an English model"

    @pytest.mark.parametrize("sentence", FRAGMENTS)
    def test_rejects_fragments(self, sentence):
        assert is_bad_sentence(sentence)

    def test_rejects_page_furniture(self):
        assert is_bad_sentence("received 12 december 1968; revised 3 march 1969")
        assert is_bad_sentence("vol. 66, no. 2, pp. 267-273, printed in great britain")


class TestKeepsRealProse:
    @pytest.mark.parametrize("sentence", GOOD_SENTENCES)
    def test_keeps_methodology_sentences(self, sentence):
        assert not is_bad_sentence(sentence), f"false positive on real prose: {sentence[:60]}"

    def test_keeps_sentence_with_a_few_numbers(self):
        """Methodology prose legitimately contains numbers; only heavy digit density is bad."""
        sentence = ("the sample was irradiated for 24 hours at a flux of 3 times 10 to the 12 "
                    "neutrons per square centimetre per second.")
        assert not is_bad_sentence(sentence)

    def test_keeps_sentence_mentioning_a_journal_word_in_prose(self):
        """'physics' or 'nuclear' in prose must not trip the citation heuristic."""
        sentence = ("this technique is standard in nuclear physics and has been applied to "
                    "many activation measurements of this kind.")
        assert not is_bad_sentence(sentence)


class TestOcrDamage:
    def test_clean_text_scores_zero(self):
        assert ocr_damage_ratio("the detector was calibrated with a standard source.") == 0.0

    def test_mangled_text_scores_above_zero(self):
        assert ocr_damage_ratio("ci~aube¥ and sehgal") > 0.0

    def test_empty_text_is_maximally_damaged(self):
        assert ocr_damage_ratio("") == 1.0

    def test_threshold_is_configurable(self):
        sentence = "a" * 1000 + "~"
        assert not has_ocr_damage(sentence, threshold=0.01)
        assert has_ocr_damage(sentence, threshold=0.0001)


class TestNonEnglish:
    def test_needs_multiple_markers(self):
        """A single ambiguous word must not condemn an English sentence.

        'le' and 'die' appear in English text ('le' in names, 'die' as a verb), so one
        marker is not enough.
        """
        assert not looks_non_english("the samples die out after a long counting period")

    def test_detects_italian(self):
        assert looks_non_english("si sono misurate le sezioni d'urto")

    def test_detects_german(self):
        assert looks_non_english("die messungen wurden mit der methode von der gruppe")


class TestCitationDetection:
    def test_detects_numbered_entry(self):
        assert looks_like_citation("(3) j. r. smith and a. b. jones: phys. rev., 80, 34")

    def test_detects_journal_tail(self):
        assert looks_like_citation("nucl. phys., 66, 267 (1965)")

    def test_does_not_flag_prose_with_one_initial(self):
        assert not looks_like_citation(
            "the method of r. smith was used to calibrate the detector efficiency "
            "against a known standard source of cobalt"
        )


class TestStripReferenceList:
    def test_cuts_at_heading(self):
        text = "methods were standard.\nReferences\n1. smith, phys rev 80 34"
        assert "smith" not in strip_reference_list(text)

    def test_cuts_at_acknowledgments(self):
        text = "we conclude the value is correct.\nAcknowledgments\nwe thank the operators"
        assert "operators" not in strip_reference_list(text)

    def test_returns_text_unchanged_when_no_heading(self):
        text = "methods were standard and the detector was calibrated."
        assert strip_reference_list(text) == text


class TestDocumentQualityReport:
    def test_good_document_is_usable(self):
        sentences = GOOD_SENTENCES * 5  # 25 usable sentences
        report = document_quality_report(sentences)

        assert report["usable"]
        assert report["recommendation"] == "embed"
        assert report["kept_sentences"] == len(sentences)

    def test_mostly_garbage_document_is_flagged_for_reocr(self):
        sentences = OCR_DAMAGED * 20 + GOOD_SENTENCES
        report = document_quality_report(sentences)

        assert not report["usable"]
        assert report["recommendation"] == "re-OCR or exclude"
        assert any("rejected" in r for r in report["reasons"])

    def test_too_few_sentences_is_flagged(self):
        report = document_quality_report(GOOD_SENTENCES)  # only 5

        assert not report["usable"]
        assert any("usable sentences" in r for r in report["reasons"])

    def test_empty_document_is_flagged_not_crashed(self):
        report = document_quality_report([])

        assert not report["usable"]
        assert report["kept_sentences"] == 0

    def test_reports_the_numbers_needed_to_triage(self):
        report = document_quality_report(OCR_DAMAGED * 20 + GOOD_SENTENCES * 5)

        for key in ["total_sentences", "kept_sentences", "rejection_rate", "ocr_damage_ratio"]:
            assert key in report


class TestCleanSentences:
    def test_normalises_and_lowercases(self):
        result = clean_sentences(["The   Detector  Was Calibrated With A Standard Source Today."])
        assert result == ["the detector was calibrated with a standard source today."]

    def test_drops_garbage_and_keeps_prose(self):
        result = clean_sentences(GOOD_SENTENCES + OCR_DAMAGED + NON_ENGLISH)
        assert len(result) == len(GOOD_SENTENCES)
