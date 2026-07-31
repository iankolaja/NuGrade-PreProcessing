"""Quality gates for text extracted from EXFOR experimental report PDFs.

Motivation, measured against the 7,193 sentences currently in the database: 19.3% are
unusable. They are not harmless. Every stored sentence is embedded, so garbage is
retrievable by the RAG tools, and the per-category `{category}_max_sim` features are a
*max* over sentences — a single mangled line that happens to embed near a template
inflates that report's similarity score and corrupts its KNN neighbours.

Measured failure modes and what handles each:

  11.1%  OCR-mangled characters      -> has_ocr_damage / ocr_damage_ratio
   4.4%  reference-list residue      -> strip_reference_list, looks_like_citation
   4.1%  fragments (<= 6 words)      -> is_bad_sentence (word-count floor)
   1.0%  journal/page boilerplate    -> looks_like_boilerplate
   0.4%  non-English text            -> looks_non_english

These reports are largely scanned 1950s-1970s papers, so a poor text layer is the norm
rather than the exception. `document_quality_report` scores a whole document so a bad
scan can be routed to re-OCR instead of silently embedded.
"""
import re
import string
import unicodedata

# Characters that essentially never appear in clean scientific prose but are common
# artifacts of a bad OCR pass (mojibake, stray diacritics, box-drawing).
OCR_NOISE_CHARACTERS = set("~¥§¤®©°¬×\\|†‡¶∂□■▪�")

# EXFOR includes French, German and Italian reports, which SciBERT (an English scientific
# model) should not embed. Two tiers, because the evidence differs in strength:
#
# Strong markers are words with no English reading at all — foreign month names and
# section headings. One is conclusive, which matters for short journal-header lines that
# contain only a date plus an English title fragment.
STRONG_NON_ENGLISH_MARKERS = {
    "janvier", "fevrier", "mars", "avril", "juin", "juillet", "septembre", "octobre",
    "novembre", "decembre",
    "januar", "februar", "marz", "juni", "juli", "oktober", "dezember",
    "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio",
    "settembre", "ottobre", "novembre", "dicembre",
    "riassunto", "zusammenfassung", "traduzione", "uebersetzung", "resume",
}

# Weak markers are function words that also occur in English ("le" in names, "die" as a
# verb, "sono"/"mit" in transliterations), so two or more are required.
NON_ENGLISH_MARKERS = {
    "le", "les", "des", "une", "nous", "pour", "avec", "cette", "resultats",
    "und", "der", "die", "das", "von", "mit", "wurde", "wurden", "nicht", "ergebnisse",
    "nel", "della", "sono", "sezioni", "urto", "misurate", "cura", "redazione",
}

BOILERPLATE_PATTERNS = [
    re.compile(r"\b(vol|no)\.\s*\d", re.I),
    re.compile(r"\bissn\b|\bdoi:", re.I),
    re.compile(r"\b(received|revised|accepted)\b\s*:?\s*\d?", re.I),
    re.compile(r"\ball rights reserved\b|\bcopyright\b|\(c\)\s*\d{4}", re.I),
    re.compile(r"\bprinted in\b", re.I),
    re.compile(r"\btraduzione\b|\btranslated by\b", re.I),
]

# A citation looks like "(1) i. k. chaubey and m. l. sehgal: nucl. phys., 66, 267 (1965)."
# The giveaway is a run of single-letter initials, or a journal-abbreviation tail.
INITIALS_RUN = re.compile(r"(?:\b[a-z]\.\s*){2,}", re.I)
CITATION_TAIL = re.compile(
    r"\b(phys|nucl|rev|lett|j|z|ann|proc|rep|sci|instrum|methods)\b\.?\s*,?\s*"
    r"(?:[a-z]?\s*)?\d+\s*,\s*\d+",
    re.I,
)
NUMBERED_CITATION_START = re.compile(r"^\s*[\(\[]?\s*(?:\d{1,3}|[~*])\s*[\)\]]\s*[a-z]\.", re.I)

REFERENCE_HEADING = re.compile(
    r"\n\s*(references|bibliography|acknowledgment?s|literatur|literature cited)\s*\.?\s*\n",
    re.I,
)


def ocr_damage_ratio(text):
    """Fraction of characters that are OCR noise or unassigned Unicode."""
    if not text:
        return 1.0
    damaged = sum(
        1 for c in text
        if c in OCR_NOISE_CHARACTERS or unicodedata.category(c) in ("Co", "Cn")
    )
    return damaged / len(text)


def has_ocr_damage(sentence, threshold=0.005):
    """True if the sentence carries OCR noise characters above ``threshold``.

    Deliberately strict: a single stray '¥' inside a word means the surrounding tokens
    are unreliable, and an unreliable sentence is worse than a missing one.
    """
    return ocr_damage_ratio(sentence) > threshold


def looks_non_english(sentence, min_markers=2):
    """True if the sentence is not English.

    A single strong marker (a foreign month name or section heading) is conclusive; weak
    function words need ``min_markers`` of them, since they also occur in English.
    """
    words = set(re.findall(r"[a-z']+", sentence.lower()))
    if words & STRONG_NON_ENGLISH_MARKERS:
        return True
    return len(words & NON_ENGLISH_MARKERS) >= min_markers


def looks_like_boilerplate(sentence):
    """True for journal headers, copyright lines, submission dates and similar furniture."""
    return any(pattern.search(sentence) for pattern in BOILERPLATE_PATTERNS)


def looks_like_citation(sentence):
    """True if the sentence is a bibliography entry rather than prose.

    Catches the residue left when a report has no 'References' heading to cut at — common
    in scanned papers that use numbered footnote-style citation lists.
    """
    if NUMBERED_CITATION_START.match(sentence):
        return True
    if CITATION_TAIL.search(sentence):
        return True
    # Several runs of initials and little else: "n. h. lazor and w. s. lyon:"
    initials = len(INITIALS_RUN.findall(sentence))
    words = sentence.split()
    return initials >= 2 and len(words) <= 14


def strip_reference_list(text, max_sections=2, tail_fraction=0.4):
    """Remove a trailing reference list, without truncating a multi-chapter report.

    A single article has one reference list at the end, and cutting there is right. A
    compiled laboratory report has one per chapter, and cutting at the first destroys the
    document: ANL-7710 carries 71 reference headings, the earliest 1.4% of the way in, so
    truncating there discarded 98.6% of the text — every measurement description in it.

    So truncation applies only when the headings look like a single trailing list: at most
    ``max_sections`` of them, and the cut point inside the last ``tail_fraction`` of the
    document. Otherwise the text is returned whole and the citation lines are left to
    ``looks_like_citation``, which removes them sentence by sentence and is what catches
    reports with no heading at all.
    """
    matches = list(REFERENCE_HEADING.finditer(text))
    if not matches or len(matches) > max_sections:
        return text

    cut = matches[0].start()
    if not text or cut < len(text) * (1 - tail_fraction):
        return text
    return text[:cut]


def is_bad_sentence(sentence, min_words=7):
    """True if ``sentence`` should be excluded from embedding.

    Combines the original notebook heuristics (length, alpha/digit/punctuation balance,
    equation density) with the failure modes measured in the existing corpus.
    """
    s = sentence.strip()
    if len(s) < 25:
        return True

    words = s.split()
    if len(words) < min_words:
        return True

    characters = [c for c in s if not c.isspace()]
    if not characters:
        return True

    alpha_fraction = sum(c.isalpha() for c in characters) / len(characters)
    digit_fraction = sum(c.isdigit() for c in characters) / len(characters)
    punctuation_fraction = sum(c in string.punctuation for c in characters) / len(characters)

    if alpha_fraction < 0.35 or digit_fraction > 0.25 or punctuation_fraction > 0.45:
        return True
    if len(re.findall(r"[=<>±∑√∫≈×]", s)) >= 3:
        return True

    return (
        has_ocr_damage(s)
        or looks_non_english(s)
        or looks_like_boilerplate(s)
        or looks_like_citation(s)
    )


def clean_sentences(sentences, min_words=7):
    """Normalise whitespace, lowercase, and drop sentences that fail the quality gate."""
    kept = []
    for sentence in sentences:
        normalised = re.sub(r"\s+", " ", sentence).strip().lower()
        if not is_bad_sentence(normalised, min_words=min_words):
            kept.append(normalised)
    return kept


def document_quality_report(sentences, min_usable_sentences=20, max_rejection_rate=0.6,
                            healthy_yield=100):
    """Decide whether a document's extracted text is good enough to embed.

    Two different failures are being distinguished, and only one of them is fatal:

      * too little usable text — the text layer failed, so there is nothing to embed
      * a high rejection rate — most extracted "sentences" were discarded

    A high rejection rate alone does not condemn a document. Scanned laboratory reports are
    full of tables, figure captions and column headers, so most fragments are legitimately
    discarded while the surviving prose is perfectly good. Measured on the OSTI reports:
    ORNL-4805 rejects 81% of its 567 fragments and still yields 106 clean methodology
    sentences ("a peak was stripped by drawing a background beneath it, subtracting the
    background..."), while what it discarded was noise like "z 3 u." and ".-.". That is the
    sentence filter succeeding, not the document failing.

    So the rate only matters when the absolute yield is *also* poor: below ``healthy_yield``
    a high rejection rate is evidence the OCR is untrustworthy, and above it the filter has
    demonstrably done its job. The original 60% limit was calibrated on hand-picked journal
    articles and rejected 29 of 59 fetched lab reports outright.

    Returns a dict with ``usable``, the kept sentences, and the reason if unusable.
    """
    total = len(sentences)
    kept = clean_sentences(sentences)
    rejected = total - len(kept)
    rejection_rate = (rejected / total) if total else 1.0
    damage = ocr_damage_ratio(" ".join(sentences)) if sentences else 1.0

    reasons = []
    if len(kept) < min_usable_sentences:
        reasons.append(
            f"only {len(kept)} usable sentences (need {min_usable_sentences}); "
            "likely a scan with no usable text layer"
        )
    elif rejection_rate > max_rejection_rate and len(kept) < healthy_yield:
        reasons.append(
            f"rejected {rejection_rate:.0%} of sentences (limit {max_rejection_rate:.0%}) "
            f"and kept only {len(kept)} (need {healthy_yield} to trust a rate that high); "
            "likely OCR damage or a non-English report"
        )

    return {
        "usable": not reasons,
        "reasons": reasons,
        "sentences": kept,
        "total_sentences": total,
        "kept_sentences": len(kept),
        "rejection_rate": rejection_rate,
        "ocr_damage_ratio": damage,
        "recommendation": "embed" if not reasons else "re-OCR or exclude",
    }
