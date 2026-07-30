"""Resolve EXFOR journal references to DOIs, preferring NASA ADS over Crossref.

Why ADS first: ADS indexes older physics literature by *structured* citation — journal
bibstem, volume, page — which is exactly the shape of an EXFOR REFERENCE field. Crossref
only offers fuzzy bibliographic search, which on this corpus returns confidently wrong
articles. Measured on entry 11150 (Melkonian, Phys. Rev. 76, 1750):

    ADS       bibstem:PhRv volume:76 page:1750  -> 10.1103/PhysRev.76.1750   correct
    Crossref  title search                      -> 10.1103/physrev.76.1744   wrong article

Both are still verified the same way before a DOI is accepted: the candidate must agree on
volume and first page. A wrong DOI is worse than none, because it fetches a real paper about
something else which then passes every quality gate and is embedded as that experiment's
report.

Requires an ADS token in keys/ads_token.txt (free, from ui.adsabs.harvard.edu). Falls back
to Crossref when no token is present or the journal has no bibstem mapping.
"""
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ADS_URL = "https://api.adsabs.harvard.edu/v1/search/query"
CROSSREF_URL = "https://api.crossref.org/works"
TOKEN_FILE = Path("keys/ads_token.txt")

# EXFOR journal code -> ADS bibstem. Every value below was verified against ADS on
# 2026-07-29 by querying `bibstem:<value>` and confirming a non-zero record count — two
# plausible-looking guesses (JNucE, NucSE) returned zero and silently cost resolutions, so
# do not add an entry here without checking it the same way.
#
# An unmapped code deliberately falls through to a Crossref title search rather than
# guessing a bibstem, since a wrong bibstem can match a real but unrelated article.
BIBSTEMS = {
    "PR": "PhRv",          # Physical Review (pre-1970)
    "PR/C": "PhRvC",
    "PR/B": "PhRvB",
    "PR/D": "PhRvD",
    "PRL": "PhRvL",
    "NP": "NucPh",
    "NP/A": "NuPhA",
    "NP/B": "NuPhB",
    "PL": "PhL",
    "PL/B": "PhLB",
    "ZP": "ZPhy",
    "ZP/A": "ZPhyA",
    "NC": "NCim",
    "NC/A": "NCimA",
    "NC/B": "NCimB",
    "NIM": "NucIM",
    "NIM/A": "NIMPA",
    "JNE": "JNuE",         # NOT JNucE, which has zero records
    "JNE/A": "JNuE",       # Journal of Nuclear Energy parts A and B share the bibstem
    "JNE/B": "JNuE",
    "JNE/AB": "JNuE",
    "NSE": "NSE",          # NOT NucSE, which has zero records
    "ANE": "AnNuE",
    "CJP": "CaJPh",
    "JP/G": "JPhG",
    "JPJ": "JPSJ",
    "PRAM": "Prama",

    # Added 2026-07-29 from the 300-entry survey's unresolved list. Each was verified by
    # querying a real (volume, page) from this corpus and confirming ADS returns that
    # article — a stronger check than confirming the bibstem merely exists.
    "AP": "AnPhy",         # Annals of Physics
    "JRN": "JRNC",         # Journal of Radioanalytical and Nuclear Chemistry
    "ZN/A": "ZNatA",       # Zeitschrift fuer Naturforschung A
    "PRS/A": "RSPSA",      # Proceedings of the Royal Society A
    "NST": "JNST",         # Journal of Nuclear Science and Technology
    "JP/A": "JPhA",        # Journal of Physics A
    "FBS": "FBS",          # Few-Body Systems
    "EPJ/A": "EPJA",       # European Physical Journal A
}

# Soviet and Russian journals are deliberately absent. AE (Atomnaya Energiya), SJA, JET,
# SNP, ZET and BAS were all tested against AtEne, SvAtE, AtEn, JETP, ZhETF, SvJNP and
# BASUP using real volume/page pairs from this corpus, and none matched. The cause is not a
# missing bibstem: the English translations renumber volumes and pages relative to the
# Russian originals that EXFOR cites, so structured matching cannot bridge them at all.
# These need title-based matching against the translation, or IAEA's INDC series.
UNRESOLVABLE_BY_STRUCTURE = {"AE", "SJA", "JET", "SNP", "ZET", "BAS", "YF", "UFZ"}


class ResolutionError(RuntimeError):
    """A lookup service could not be reached."""


def load_ads_token(path=TOKEN_FILE):
    """Return the ADS token, or None if absent. Never log or echo the value."""
    path = Path(path)
    if not path.is_file():
        return None
    token = path.read_text().strip()
    return token or None


def _pages_agree(candidate_page, reference_page):
    """True if a candidate page matches the EXFOR page.

    Candidates give ranges ("1750-1760") or lists; EXFOR gives the first page only.
    """
    if candidate_page is None or reference_page is None:
        return False
    if isinstance(candidate_page, list):
        candidate_page = candidate_page[0] if candidate_page else ""
    first = str(candidate_page).split("-")[0].strip()
    return first == str(reference_page).strip()


def _volumes_agree(candidate_volume, reference_volume):
    if candidate_volume is None or reference_volume is None:
        return False
    if isinstance(candidate_volume, list):
        candidate_volume = candidate_volume[0] if candidate_volume else ""
    return str(candidate_volume).strip() == str(reference_volume).strip()


def _first(value):
    """ADS returns most fields as single-element lists."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def resolve_via_ads(reference, token, delay=0.4, timeout=30):
    """Look up a DOI in ADS using bibstem + volume + page.

    Returns a dict with ``doi``, ``bibcode``, ``source`` and ``evidence``, or None if the
    reference has no bibstem mapping or nothing matches on volume and page.
    """
    bibstem = BIBSTEMS.get((reference.get("code") or "").upper())
    volume, page = reference.get("volume"), reference.get("page")
    if not (bibstem and volume and page):
        return None

    query = urllib.parse.urlencode({
        "q": f"bibstem:{bibstem} volume:{volume} page:{page}",
        "fl": "bibcode,doi,title,volume,page,year",
        "rows": 5,
    })
    request = urllib.request.Request(
        f"{ADS_URL}?{query}",
        headers={"Authorization": f"Bearer {token}",
                 "User-Agent": "NuGrade-research/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError) as e:
        raise ResolutionError(f"ADS lookup failed: {e}") from e
    finally:
        time.sleep(delay)

    for doc in payload.get("response", {}).get("docs", []):
        if _volumes_agree(doc.get("volume"), volume) and _pages_agree(doc.get("page"), page):
            doi = _first(doc.get("doi"))
            return {
                "doi": doi,
                "bibcode": doc.get("bibcode"),
                "source": "ads",
                "evidence": f"bibstem:{bibstem} volume:{volume} page:{page}",
                "title": _first(doc.get("title")),
            }
    return None


def resolve_via_crossref(reference, title, email=None, delay=0.4, timeout=30):
    """Look up a DOI in Crossref by title, verified against volume and page.

    Used only as a fallback: Crossref has no structured citation query, so a title search
    is the best available and must be verified before the DOI is trusted.
    """
    if not title or not reference.get("volume") or not reference.get("page"):
        return None

    query = urllib.parse.urlencode({
        "query.bibliographic": title[:180],
        "rows": 5,
        "select": "DOI,title,volume,page",
    })
    contact = f" (mailto:{email})" if email else ""
    request = urllib.request.Request(
        f"{CROSSREF_URL}?{query}",
        headers={"User-Agent": f"NuGrade-research/0.1{contact}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError) as e:
        raise ResolutionError(f"Crossref lookup failed: {e}") from e
    finally:
        time.sleep(delay)

    for item in payload.get("message", {}).get("items", []):
        if (_volumes_agree(item.get("volume"), reference["volume"])
                and _pages_agree(item.get("page"), reference["page"])):
            return {
                "doi": item.get("DOI"),
                "bibcode": None,
                "source": "crossref",
                "evidence": f"volume:{reference['volume']} page:{reference['page']}",
                "title": _first(item.get("title")),
            }
    return None


def resolve(bib, token=None, email=None, delay=0.4):
    """Resolve one parsed EXFOR BIB record to a DOI, trying ADS then Crossref.

    Returns the match dict, or None if neither service could confirm one. Never returns an
    unverified guess: every result agreed on volume and first page.
    """
    reference = (bib or {}).get("reference")
    if not reference or reference.get("type") != "journal":
        return None

    if token:
        try:
            match = resolve_via_ads(reference, token, delay=delay)
        except ResolutionError:
            match = None
        if match:
            return match

    try:
        return resolve_via_crossref(reference, bib.get("title"), email=email, delay=delay)
    except ResolutionError:
        return None
