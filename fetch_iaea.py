"""Find report PDFs in the IAEA Nuclear Data Section repository.

    python fetch_iaea.py --dry-run          # what would match, download nothing
    python fetch_iaea.py --limit 20
    python fetch_iaea.py

Many EXFOR entries cite grey literature that has no DOI and is not held by OSTI: INDC
progress reports, EANDC reports, laboratory series, conference proceedings. A large amount
of it is in IAEA NDS, which runs InvenioRDM and exposes a documented JSON API.

The matching rule is the same one used for DOIs: a candidate is accepted only when the
report code from the EXFOR reference appears in the record, after normalising the
punctuation the two sources spell differently — EXFOR writes ``INDC(NOR)-1`` and the
repository names the file ``indc-nor-0001G.pdf``. A search hit alone is not enough, because
a query for one report readily returns a different report from the same series.
"""
import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://nds.iaea.org/api/records"
DEFAULT_PDF_DIR = Path("pdfs")
DEFAULT_REPORT = Path("output/iaea_fetch_log.csv")

FIELDS = ["entry", "status", "code", "record", "filename", "match_evidence",
          "bytes", "has_text_layer", "detail"]

# Report series worth asking IAEA about. Anything else falls through to the manual queue;
# guessing wastes requests on a shared service and invites bad matches.
SEARCHABLE_SERIES = re.compile(
    r"\b(INDC|EANDC|IAEA|NDS|CINDA|INDSWG)\b", re.I)


class IaeaError(RuntimeError):
    pass


def normalise_code(text):
    """Reduce a report code to comparable form: lowercase alphanumerics only.

    EXFOR writes INDC(NOR)-1; the repository names the file indc-nor-0001G.pdf. Stripping
    punctuation and leading zeros lets the two be compared without guessing either format.
    """
    parts = re.findall(r"[A-Za-z]+|\d+", str(text or ""))
    return "".join(p.lower().lstrip("0") or "0" for p in parts)


def report_code(citation):
    """Pull the report code out of an EXFOR citation such as 'INDC(NOR)-1 p 2 (1972)'."""
    if not citation:
        return ""
    # Cut trailing page and year decoration, which is not part of the identifier.
    trimmed = re.split(r"\s+(?:p|vol)\s", citation)[0]
    trimmed = re.sub(r"\s*\(\s*(19|20)\d\d\s*\)\s*$", "", trimmed)
    return trimmed.strip()


def is_searchable(citation):
    """True if this citation names a series IAEA is likely to hold."""
    return bool(SEARCHABLE_SERIES.search(report_code(citation)))


def _get(url, timeout=30, accept_json=True):
    headers = {"User-Agent": "NuGrade-research/0.1 (mailto:ikolaja@berkeley.edu)"}
    if accept_json:
        headers["Accept"] = "application/json"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except (urllib.error.URLError, TimeoutError) as error:
        raise IaeaError(str(error)) from error


def search(code, size=8, delay=1.0, timeout=30):
    """Search the repository for a report code. Returns the raw hit list."""
    query = urllib.parse.urlencode({"q": f'"{code}"', "size": size})
    try:
        payload = json.loads(_get(f"{API}?{query}", timeout=timeout))
    except json.JSONDecodeError as error:
        raise IaeaError(f"malformed response for {code!r}") from error
    finally:
        time.sleep(delay)
    return payload.get("hits", {}).get("hits", [])


def matching_file(hits, code):
    """Return (record id, filename, evidence) for the record that really is this report.

    A search hit alone is not evidence: querying one report in a series readily returns its
    neighbours. Two kinds of match are accepted, strongest first.

    An exact entry in the record's ``identifiers`` is decisive, and covers the rename these
    reports went through — IAEA records EANDC(US)-62 and INDC(US)*012 as identifiers of the
    same document, so an EXFOR citation of the old code still resolves. Verified against
    four reports whose PDF first pages name the code EXFOR cites.

    Otherwise the code must appear in the filename or title, which is weaker but still a
    statement about that document rather than about the search ranking.
    """
    wanted = normalise_code(code)
    if not wanted:
        return None

    def pdf_of(hit):
        entries = list((hit.get("files", {}).get("entries") or {}))
        pdfs = [f for f in entries if f.lower().endswith(".pdf")]
        return pdfs[0] if pdfs else None

    for hit in hits:                      # identifier match: authoritative
        for identifier in hit.get("metadata", {}).get("identifiers", []) or []:
            value = identifier.get("identifier", "")
            if value and normalise_code(value) == wanted:
                name = pdf_of(hit)
                if name:
                    return hit["id"], name, f"identifier {value}"

    for hit in hits:                      # filename or title mention: weaker
        name = pdf_of(hit)
        if not name:
            continue
        if wanted in normalise_code(name):
            return hit["id"], name, "filename"
        if wanted in normalise_code(hit.get("metadata", {}).get("title", "")):
            return hit["id"], name, "title"
    return None


def download_file(record, filename, destination, delay=1.0, timeout=120):
    """Fetch one file from a record, verifying it is a PDF before keeping it."""
    url = (f"https://nds.iaea.org/api/records/{record}/files/"
           f"{urllib.parse.quote(filename)}/content")
    try:
        payload = _get(url, timeout=timeout, accept_json=False)
    finally:
        time.sleep(delay)

    if not payload.startswith(b"%PDF-"):
        raise IaeaError("response is not a PDF")
    destination.write_bytes(payload)
    return len(payload)


def has_text_layer(path, min_characters=400):
    """Whether the PDF carries extractable text; None if PyMuPDF is unavailable.

    Many of these are scans. A scan downloads fine and then embeds to nothing, so this is
    what separates "obtained" from "usable".
    """
    try:
        import fitz
    except ImportError:
        return None
    try:
        document = fitz.open(str(path))
    except Exception:
        return False
    try:
        text = "".join(page.get_text("text") for page in document)
    finally:
        document.close()
    return len(text.strip()) >= min_characters


def candidates(unreachable_path):
    """Entries whose citation names a series IAEA plausibly holds."""
    rows = list(csv.DictReader(open(unreachable_path, newline="")))
    return [r for r in rows if is_searchable(r.get("citation", ""))]


def fetch_all(rows, pdf_dir, *, delay=1.0, limit=None, resume=True, emit=print):
    pdf_dir = Path(pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in pdf_dir.glob("*.pdf")} if resume else set()

    todo = [r for r in rows if r["entry"] not in have]
    if limit:
        todo = todo[:limit]
    emit(f"{len(rows)} candidates, {len(rows) - len(todo)} already present; "
         f"searching {len(todo)}")

    results = []
    for index, row in enumerate(todo, 1):
        entry = row["entry"]
        code = report_code(row.get("citation", ""))
        record = {"entry": entry, "code": code, "status": "", "record": "",
                  "filename": "", "match_evidence": "", "bytes": "",
                  "has_text_layer": "", "detail": ""}
        try:
            hits = search(code, delay=delay)
        except IaeaError as error:
            record.update(status="search_failed", detail=str(error)[:80])
            results.append(record)
            emit(f"  [{index}/{len(todo)}] {entry}: search failed")
            continue

        match = matching_file(hits, code)
        if not match:
            record.update(status="no_match",
                          detail=f"{len(hits)} hits, none matching {code}")
            results.append(record)
            emit(f"  [{index}/{len(todo)}] {entry}: no match for {code}")
            continue

        record_id, filename, evidence = match
        destination = pdf_dir / f"{entry}.pdf"
        try:
            size = download_file(record_id, filename, destination, delay=delay)
        except IaeaError as error:
            record.update(status="download_failed", record=record_id,
                          filename=filename, match_evidence=evidence,
                          detail=str(error)[:80])
            destination.unlink(missing_ok=True)
            results.append(record)
            emit(f"  [{index}/{len(todo)}] {entry}: download failed")
            continue

        text_layer = has_text_layer(destination)
        record.update(status="fetched", record=record_id, filename=filename,
                      match_evidence=evidence, bytes=size,
                      has_text_layer="" if text_layer is None else text_layer)
        results.append(record)
        emit(f"  [{index}/{len(todo)}] {entry}: {filename} {size / 1e6:.1f} MB"
             f"{'' if text_layer is None else (' with text' if text_layer else ' NO TEXT LAYER')}")

    return results


def summarise(results, emit=print):
    from collections import Counter

    status = Counter(r["status"] for r in results)
    fetched = [r for r in results if r["status"] == "fetched"]
    with_text = [r for r in fetched if r["has_text_layer"] is True]
    no_text = [r for r in fetched if r["has_text_layer"] is False]

    emit("")
    emit("=" * 66)
    emit("IAEA FETCH SUMMARY")
    emit("=" * 66)
    for key, count in status.most_common():
        emit(f"  {key:20s} {count:>5}")
    if fetched:
        emit(f"\n  of {len(fetched)} fetched:")
        emit(f"    with a text layer  {len(with_text):>5}   ready to embed")
        emit(f"    needing OCR        {len(no_text):>5}   these are unOCR'd scans")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unreachable", default="output/unreachable.csv")
    parser.add_argument("--pdf-dir", default=str(DEFAULT_PDF_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    path = Path(args.unreachable)
    if not path.is_file():
        print(f"no unreachable list at {path}; run make_work_queues.py first",
              file=sys.stderr)
        return 2

    rows = candidates(path)
    if args.dry_run:
        print(f"{len(rows)} entries cite a series IAEA plausibly holds:\n")
        for row in rows[:40]:
            print(f"   {row['entry']:8s} {report_code(row['citation'])}")
        if len(rows) > 40:
            print(f"   ... and {len(rows) - 40} more")
        return 0

    results = fetch_all(rows, args.pdf_dir, delay=args.delay, limit=args.limit,
                        resume=not args.no_resume)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)
    summarise(results)
    print(f"\nwrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
