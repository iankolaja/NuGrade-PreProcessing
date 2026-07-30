"""Fetch and parse EXFOR bibliographic records from the IAEA nuclear data service.

The database built by 1_raw_data_ingestion.ipynb carries only first author and year, which
is not enough to identify a paper: matching on author+year against Crossref returns
confidently wrong articles (see ACQUISITION.md for the measurement). Journal, volume, page
and — most usefully — the title are needed.

Those all live in the BIB section of EXFOR subentry 001, which IAEA serves directly:

    https://nds.iaea.org/exfor/servlet/X4sGetSubent?subID=<entry>001

So this does NOT require X4Pro or cluster access. Responses are cached on disk, so a second
run costs nothing and the service is only ever asked for an entry once.

Be a good citizen: IAEA NDS is a shared public service run by a small team. Keep the delay,
keep the cache, and identify yourself in the User-Agent.
"""
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://nds.iaea.org/exfor/servlet/X4sGetSubent?subID={sub_id}"
DEFAULT_CACHE = Path("cache/exfor_bib")
DEFAULT_DELAY = 0.5  # seconds between live requests

# EXFOR journal codes -> printable names. Not exhaustive; unknown codes are returned as-is
# so a caller can still match on volume/page/title.
JOURNAL_CODES = {
    "PR": "Physical Review",
    "PR/C": "Physical Review C",
    "PR/B": "Physical Review B",
    "PRL": "Physical Review Letters",
    "NP": "Nuclear Physics",
    "NP/A": "Nuclear Physics A",
    "NP/B": "Nuclear Physics B",
    "PL": "Physics Letters",
    "PL/B": "Physics Letters B",
    "ZP": "Zeitschrift fuer Physik",
    "ZP/A": "Zeitschrift fuer Physik A",
    "NC": "Nuovo Cimento",
    "NC/B": "Nuovo Cimento B",
    "NC/A": "Nuovo Cimento A",
    "JNE": "Journal of Nuclear Energy",
    "JNE/A": "Journal of Nuclear Energy A",
    "NSE": "Nuclear Science and Engineering",
    "NIM": "Nuclear Instruments and Methods",
    "NIM/A": "Nuclear Instruments and Methods A",
    "AE": "Atomnaya Energiya",
    "YF": "Yadernaya Fizika",
    "SNP": "Soviet Journal of Nuclear Physics",
    "JETP": "Journal of Experimental and Theoretical Physics",
    "CN": "Chinese Journal of Nuclear Physics",
    "ANE": "Annals of Nuclear Energy",
    "JP/G": "Journal of Physics G",
    "CJP": "Canadian Journal of Physics",
    "PRAM": "Pramana",
    "IJPA": "Indian Journal of Physics A",
}

# Reference type prefixes. Only J (journal) is reliably resolvable to a DOI; the rest are
# grey literature that needs OSTI / IAEA / library lookup instead.
REFERENCE_TYPES = {
    "J": "journal",
    "R": "report",
    "P": "progress_report",
    "C": "conference",
    "S": "proceedings",
    "B": "book",
    "T": "thesis",
    "W": "private_communication",
    "3": "other",
}


class ExforFetchError(RuntimeError):
    """The IAEA service could not be reached, or returned no BIB record."""


def _parse_year(token):
    """Interpret an EXFOR date token: 1948, 194103, 196602 or a 2-digit-year 7602.

    Returns a 4-digit year, or None if the token is not a date.
    """
    token = token.strip()
    if not token.isdigit():
        return None
    if len(token) >= 6:                     # YYYYMM
        year = int(token[:4])
        if 1900 <= year <= 2100:
            return year
        # 2-digit year plus month, e.g. 7602 -> 1976
    if len(token) == 4:
        value = int(token)
        if 1900 <= value <= 2100:           # plain YYYY
            return value
        # YYMM, e.g. 4912 -> 1949, 7602 -> 1976
        yy = int(token[:2])
        return 1900 + yy if yy >= 20 else 2000 + yy
    if len(token) == 2:
        yy = int(token)
        return 1900 + yy if yy >= 20 else 2000 + yy
    return None


def parse_reference(reference):
    """Parse an EXFOR REFERENCE field into its components.

    Handles the forms seen in the corpus, e.g.::

        (J,PR,74,364,1948)              -> journal, Physical Review, vol 74, page 364, 1948
        (J,PR/C,3,1886,197105)          -> Physical Review C, vol 3, page 1886, 1971
        (J,NP/A,258,(1),1,7602)         -> Nuclear Physics A, vol 258, issue 1, page 1, 1976
        (P,EANDC(E)-66,52,196602)       -> progress report EANDC(E)-66, page 52, 1966
        (C,87KIEV,2,298,1987)           -> conference 87KIEV, vol 2, page 298, 1987

    Trailing free text after the closing parenthesis is ignored. Returns a dict with
    ``type``, ``code``, ``journal``, ``volume``, ``issue``, ``page`` and ``year``; any
    field that is absent or unparseable is None.
    """
    result = {"type": None, "code": None, "journal": None,
              "volume": None, "issue": None, "page": None, "year": None,
              "raw": reference.strip()}

    match = re.search(r"\(([^()]*(?:\([^()]*\)[^()]*)*)\)", reference)
    if not match:
        return result

    # EXFOR wraps alternative references for the same document in an outer parenthesis
    # joined by '=', e.g. ((S,ISINN-7,269,199905)=(S,JINR-E3-99-212,269,199905)) or
    # ((R,JU-RR-1/1976,1976)=(T,VALKONEN,197603)). Parse the first alternative and keep the
    # rest, rather than returning the whole string as the reference type.
    if match.group(1).lstrip().startswith("("):
        alternatives = re.findall(r"\(([^()]*)\)", match.group(1))
        if alternatives:
            primary = parse_reference(f"({alternatives[0]})")
            primary["raw"] = reference.strip()
            primary["alternatives"] = [f"({a})" for a in alternatives[1:]]
            return primary

    # Split on commas that are not inside nested parentheses, e.g. EANDC(E)-66 or (1).
    body, depth, field = match.group(1), 0, ""
    fields = []
    for char in body:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            fields.append(field)
            field = ""
        else:
            field += char
    fields.append(field)
    fields = [f.strip() for f in fields]

    if not fields:
        return result

    result["type"] = REFERENCE_TYPES.get(fields[0].upper(), fields[0].upper() or None)
    if len(fields) > 1:
        result["code"] = fields[1]
        result["journal"] = JOURNAL_CODES.get(fields[1].upper(), fields[1])

    # The last numeric-looking field is the date; whatever precedes it is volume/issue/page.
    remaining = fields[2:]
    if remaining:
        year = _parse_year(remaining[-1])
        if year is not None:
            result["year"] = year
            remaining = remaining[:-1]

    # An issue appears as a parenthesised field, either standalone between volume and page
    # — (J,PR,59,917,(19),4105) — or jammed onto the page with no comma, as in
    # (J,PR,59,917(19),1941). Issue labels are not always alphanumeric: (J,FCY/L,11,(2/186),186,2014).
    positional = []
    for token in remaining:
        standalone = re.fullmatch(r"\(([^)]+)\)", token)
        attached = re.fullmatch(r"(\d+)\s*\(([^)]+)\)", token)
        if standalone:
            result["issue"] = standalone.group(1)
        elif attached:
            positional.append(attached.group(1))
            result["issue"] = attached.group(2)
        else:
            positional.append(token)

    # Journals carry both a volume and a page — (J,PR,74,364,1948). Grey literature is
    # identified by its report or series code, so a lone number is a page within that
    # document, not a volume: (P,EANDC(E)-66,52,196602), (S,ISINN-7,269,199905).
    if len(positional) == 1 and result["type"] != "journal":
        result["page"] = positional[0] or None
    else:
        if len(positional) >= 1:
            result["volume"] = positional[0] or None
        if len(positional) >= 2:
            result["page"] = positional[1] or None

    return result


def parse_bib(text):
    """Extract AUTHOR, TITLE, REFERENCE and INSTITUTE from a raw EXFOR subentry record.

    EXFOR is a fixed-column format: a keyword starts in column 1 and its value is indented
    on continuation lines. Multi-line titles are joined with a single space.
    """
    fields, current = {}, None
    for line in text.splitlines():
        if not line.strip() or line.startswith(("ENTRY", "SUBENT", "BIB", "ENDBIB",
                                                "NOCOMMON", "ENDSUBENT", "ENDENTRY")):
            if line.startswith("ENDBIB"):
                break
            continue
        keyword = line[:11].strip()
        value = line[11:].strip()
        if keyword:
            current = keyword
            fields[current] = value
        elif current:
            fields[current] += " " + value

    def clean(key):
        value = fields.get(key, "").strip()
        if value.startswith("(") and value.endswith(")"):
            value = value[1:-1]
        return re.sub(r"\s+", " ", value) or None

    authors = clean("AUTHOR")
    return {
        "authors": [a.strip() for a in authors.split(",")] if authors else [],
        "title": (re.sub(r"\s+", " ", fields.get("TITLE", "")).strip().rstrip(".") or None),
        "reference": parse_reference(fields["REFERENCE"]) if "REFERENCE" in fields else None,
        "institute": clean("INSTITUTE"),
    }


def fetch_bib(entry, cache_dir=DEFAULT_CACHE, delay=DEFAULT_DELAY, email=None, timeout=30):
    """Return the parsed BIB record for an EXFOR entry, using an on-disk cache.

    ``email`` is placed in the User-Agent. IAEA NDS is a shared public service; identifying
    yourself is the polite minimum and lets them contact you instead of blocking you.
    """
    entry = str(entry).strip()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{entry}.x4"

    if cache_file.is_file():
        return parse_bib(cache_file.read_text(errors="replace"))

    contact = f" (mailto:{email})" if email else ""
    request = urllib.request.Request(
        BASE_URL.format(sub_id=f"{entry}001"),
        headers={"User-Agent": f"NuGrade-research/0.1{contact}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError) as e:
        raise ExforFetchError(f"could not fetch entry {entry}: {e}") from e

    if "ENTRY" not in text:
        raise ExforFetchError(f"no BIB record returned for entry {entry}")

    cache_file.write_text(text)
    time.sleep(delay)  # only after a live request, never on a cache hit
    return parse_bib(text)


def is_resolvable_to_doi(bib):
    """True if this reference is a journal article, i.e. worth sending to Crossref.

    Reports, conference papers, theses and private communications will not have DOIs and
    should be routed to OSTI / IAEA / interlibrary loan instead.
    """
    reference = (bib or {}).get("reference")
    return bool(reference and reference.get("type") == "journal" and reference.get("volume"))
