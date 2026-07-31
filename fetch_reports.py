"""Download the report PDFs that are genuinely free to retrieve automatically.

    python fetch_reports.py --dry-run        # show what would be fetched, and what is not
    python fetch_reports.py --limit 10       # try a few first
    python fetch_reports.py                  # the lot

Reads output/fetch_queue.csv (produced by make_work_queues.py) and saves each document to
the PDF directory as <EXFOR entry>.pdf, which is the naming stage 2 expects.

Only hosts on ALLOWED_HOSTS are fetched. That list is deliberately narrow: repositories and
government services that publish machine-readable full text and expect programmatic access.

Publisher platforms are excluded even when Unpaywall reports the article as open access.
Open access describes the licence on the *content*; it does not grant permission to script
against the *platform*, and several of these (Taylor & Francis, APS) actively block
automated traffic. Being blocked would fall on the whole campus IP range, not just this
project. Those documents are written to a manual-fetch queue instead — one click each in a
browser, which is entirely fine and needs no permission from anyone.

Every download is verified to be a real PDF rather than an HTML error page, and reported
with whether it carries a usable text layer, since a scan without one cannot be embedded
until it has been through OCR.
"""
import argparse
import csv
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_QUEUE = Path("output/fetch_queue.csv")
DEFAULT_PDF_DIR = Path("pdfs")
DEFAULT_REPORT = Path("output/fetch_log.csv")
MANUAL_QUEUE = Path("output/manual_fetch_queue.csv")

# Hosts that publish full text for programmatic retrieval. Anything not listed here is
# routed to the manual queue rather than fetched.
ALLOWED_HOSTS = {
    "www.osti.gov": "OSTI — DOE laboratory reports, openly licensed",
    "osti.gov": "OSTI — DOE laboratory reports, openly licensed",
    "arxiv.org": "arXiv — open preprint server",
    "export.arxiv.org": "arXiv — open preprint server",
    "escholarship.org": "eScholarship — University of California repository",
    "hdl.handle.net": "Handle.net — resolves to institutional repositories",
    "repo.qst.go.jp": "QST institutional repository",
    "ir.lzu.edu.cn": "Lanzhou University institutional repository",
    "www.epj-conferences.org": "EPJ Web of Conferences — open access proceedings",
    "epj-conferences.org": "EPJ Web of Conferences — open access proceedings",
    # Added after review: institutional and national repositories following the same model
    # as eScholarship — they exist to distribute their institution's output.
    "digitalcommons.uri.edu": "University of Rhode Island repository",
    "dspace.library.uu.nl": "Utrecht University repository",
    "www.dora.lib4ri.ch": "DORA — Swiss federal research institutes repository",
    "www.jstage.jst.go.jp": "J-STAGE — Japan Science and Technology Agency platform",
}

# Named so the report can explain the exclusion rather than just listing a host.
PUBLISHER_HOSTS = {
    "www.tandfonline.com": "Taylor & Francis",
    "link.aps.org": "American Physical Society",
    "journals.aps.org": "American Physical Society",
    "www.degruyter.com": "De Gruyter",
    "link.springer.com": "Springer",
    "www.sciencedirect.com": "Elsevier",
    "doi.org": "DOI resolver — redirects to a publisher platform",
    "dx.doi.org": "DOI resolver — redirects to a publisher platform",
    "iopscience.iop.org": "IOP Publishing",
    "royalsocietypublishing.org": "The Royal Society",
    "www.publish.csiro.au": "CSIRO Publishing",
}

PDF_MAGIC = b"%PDF-"
MAX_BYTES = 200 * 1024 * 1024        # a scanned report can legitimately be tens of MB

FIELDS = ["entry", "status", "host", "url", "bytes", "has_text_layer", "detail"]


class FetchError(RuntimeError):
    pass


def classify_host(url):
    """Return (host, 'allowed' | 'publisher' | 'unknown')."""
    host = urlparse(url).netloc.lower()
    if host in ALLOWED_HOSTS:
        return host, "allowed"
    if host in PUBLISHER_HOSTS:
        return host, "publisher"
    return host, "unknown"


def alternate_urls(url):
    """Other URLs worth trying for the same document, in order.

    Unpaywall sometimes reports a landing page rather than a file. For OSTI the two forms
    share an identifier — /biblio/<id> is the record, /servlets/purl/<id> is the full text —
    so the second can be derived rather than scraped from the page. It 404s cleanly when
    the record is citation-only, which is the common case.

    No HTML is parsed for links here. Deriving a documented URL is one thing; scraping a
    page for whatever looks like a PDF is fragile and unwelcome.
    """
    import re

    match = re.match(r"(https?://(?:www\.)?osti\.gov)/biblio/(\d+)", url)
    if match:
        return [f"{match.group(1)}/servlets/purl/{match.group(2)}"]
    return []


def download(url, destination, *, email, timeout=120):
    """Fetch one document, verifying it is a PDF before keeping it.

    Repositories commonly answer a blocked or missing document with a 200 and an HTML
    page, so the magic bytes are checked rather than the status code alone.
    """
    request = urllib.request.Request(
        url, headers={"User-Agent": f"NuGrade-research/0.1 (mailto:{email})",
                      "Accept": "application/pdf,*/*"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read(MAX_BYTES)
            content_type = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as error:
        raise FetchError(f"HTTP {error.code}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise FetchError(f"unreachable: {error}") from error

    if not payload.startswith(PDF_MAGIC):
        raise FetchError(f"not a PDF (content-type {content_type or 'unknown'})")

    destination.write_bytes(payload)
    return len(payload)


def has_text_layer(path, min_characters=400):
    """True if the PDF carries extractable text.

    A scan without a text layer downloads fine and then produces nothing to embed, so this
    distinction decides whether the document is usable now or needs OCR first.
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


def load_queue(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def already_fetched(pdf_dir):
    return {p.stem for p in Path(pdf_dir).glob("*.pdf")}


def write_manual_queue(rows, path):
    """Documents a person should fetch by hand, with everything needed to do so."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["entry", "publisher", "citation", "title", "doi", "url"])
        writer.writeheader()
        for row in rows:
            host, _ = classify_host(row["url"])
            writer.writerow({
                "entry": row["entry"],
                "publisher": PUBLISHER_HOSTS.get(host, host),
                "citation": row.get("citation", ""),
                "title": row.get("title", ""),
                "doi": row.get("doi", ""),
                "url": row["url"],
            })
    return len(rows)


def fetch_all(queue, pdf_dir, *, email, delay=2.0, limit=None, resume=True, emit=print):
    """Download every allowed document. Returns (results, skipped_publisher)."""
    pdf_dir = Path(pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    allowed, publisher, unknown = [], [], []
    for row in queue:
        _, kind = classify_host(row["url"])
        {"allowed": allowed, "publisher": publisher, "unknown": unknown}[kind].append(row)

    if unknown:
        emit(f"{len(unknown)} documents on unrecognised hosts are treated as manual: "
             f"{sorted({classify_host(r['url'])[0] for r in unknown})}")
        publisher.extend(unknown)

    have = already_fetched(pdf_dir) if resume else set()
    todo = [r for r in allowed if r["entry"] not in have]
    if limit:
        todo = todo[:limit]

    present = len([r for r in allowed if r["entry"] in have])
    deferred = len(allowed) - present - len(todo)
    emit(f"{len(allowed)} fetchable, {len(publisher)} manual, {present} already present"
         + (f", {deferred} deferred by --limit" if deferred else "")
         + f"; fetching {len(todo)}")

    results = []
    last_request = defaultdict(float)
    for index, row in enumerate(todo, 1):
        entry, url = row["entry"], row["url"]
        host, _ = classify_host(url)

        # Space out requests per host rather than globally: one slow host should not make
        # every other host wait, and no host should be hit faster than the delay.
        wait = delay - (time.monotonic() - last_request[host])
        if wait > 0:
            time.sleep(wait)
        last_request[host] = time.monotonic()

        destination = pdf_dir / f"{entry}.pdf"
        record = {"entry": entry, "host": host, "url": url, "bytes": "",
                  "has_text_layer": "", "detail": ""}

        size, error, used_url = None, None, url
        for candidate in [url] + alternate_urls(url):
            try:
                size = download(candidate, destination, email=email)
                used_url = candidate
                break
            except FetchError as failure:
                error = failure
                if candidate != url:
                    time.sleep(delay)   # a retry is still a request to the same host

        if size is not None:
            text_layer = has_text_layer(destination)
            record.update(status="fetched", bytes=size, url=used_url,
                          has_text_layer="" if text_layer is None else text_layer,
                          detail="via derived full-text URL" if used_url != url else "")
            emit(f"  [{index}/{len(todo)}] {entry}: {size / 1e6:.1f} MB"
                 f"{'' if text_layer is None else (' with text' if text_layer else ' NO TEXT LAYER')}")
        else:
            record.update(status="failed", detail=str(error))
            emit(f"  [{index}/{len(todo)}] {entry}: FAILED — {error}")
            destination.unlink(missing_ok=True)
        results.append(record)

    return results, publisher


def summarise(results, manual, emit=print):
    fetched = [r for r in results if r["status"] == "fetched"]
    failed = [r for r in results if r["status"] == "failed"]
    # Three states, not two: PyMuPDF may be absent, in which case the text layer is
    # unknown rather than present. Folding unknown into "ready to embed" would overstate
    # how much of the corpus is actually usable.
    with_text = [r for r in fetched if r["has_text_layer"] is True]
    no_text = [r for r in fetched if r["has_text_layer"] is False]
    unknown_text = [r for r in fetched if r["has_text_layer"] not in (True, False)]

    emit("")
    emit("=" * 66)
    emit("FETCH SUMMARY")
    emit("=" * 66)
    emit(f"  fetched                {len(fetched):>5}")
    emit(f"    with a text layer    {len(with_text):>5}   ready to embed")
    emit(f"    needing OCR          {len(no_text):>5}   run ocrmypdf --redo-ocr first")
    if unknown_text:
        emit(f"    text layer unknown   {len(unknown_text):>5}   install pymupdf to check")
    emit(f"  failed                 {len(failed):>5}")
    emit(f"  left for manual fetch  {len(manual):>5}   publisher platforms")

    if failed:
        reasons = defaultdict(int)
        for row in failed:
            reasons[row["detail"]] += 1
        emit("\n  failures by reason:")
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            emit(f"    {count:>4}  {reason}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queue", default=str(DEFAULT_QUEUE))
    parser.add_argument("--pdf-dir", default=str(DEFAULT_PDF_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--manual-queue", default=str(MANUAL_QUEUE))
    parser.add_argument("--email", default="ikolaja@berkeley.edu")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="minimum seconds between requests to the same host")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true",
                        help="re-fetch documents already present")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be fetched, download nothing")
    args = parser.parse_args()

    queue_path = Path(args.queue)
    if not queue_path.is_file():
        print(f"no fetch queue at {queue_path}; run make_work_queues.py first",
              file=sys.stderr)
        return 2

    queue = load_queue(queue_path)

    if args.dry_run:
        buckets = defaultdict(list)
        for row in queue:
            host, kind = classify_host(row["url"])
            buckets[kind].append(host)
        print(f"queue: {len(queue)} documents\n")
        for kind, label in (("allowed", "would fetch"),
                            ("publisher", "manual only — publisher platform"),
                            ("unknown", "manual only — unrecognised host")):
            hosts = buckets.get(kind, [])
            if not hosts:
                continue
            print(f"{label}: {len(hosts)}")
            counts = defaultdict(int)
            for host in hosts:
                counts[host] += 1
            for host, count in sorted(counts.items(), key=lambda kv: -kv[1]):
                note = ALLOWED_HOSTS.get(host) or PUBLISHER_HOSTS.get(host) or ""
                print(f"    {count:>4}  {host:28s} {note}")
            print()
        return 0

    results, manual = fetch_all(queue, args.pdf_dir, email=args.email, delay=args.delay,
                                limit=args.limit, resume=not args.no_resume)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)

    n_manual = write_manual_queue(manual, Path(args.manual_queue))
    summarise(results, manual)
    print(f"\nwrote {report_path}")
    print(f"wrote {args.manual_queue} ({n_manual} documents to fetch by hand)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
