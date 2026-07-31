"""Add a text layer to scanned report PDFs, and check whether it was worth it.

    python ocr_reports.py --dry-run       # which documents would be processed
    python ocr_reports.py --limit 3       # try a few first
    python ocr_reports.py

Roughly one report in eight arrives as a pure image scan: it downloads fine, extracts to
nothing, and would embed as silence. ``ocrmypdf`` can give those a text layer.

OCR is not assumed to succeed. Each document is measured before and after against the same
quality gate stage 2 uses, and the result is only kept when it actually yields usable
sentences — a bad OCR pass produces confident-looking garbage, which is worse than a
document that is honestly empty, because garbage gets embedded and retrieved.

Originals are never overwritten. Output goes to a sibling directory, and only documents
that improve are copied back over the scan.
"""
import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from report_quality import document_quality_report

DEFAULT_PDF_DIR = Path("pdfs")
DEFAULT_WORK_DIR = Path("output/ocr")
DEFAULT_REPORT = Path("output/ocr_log.csv")

FIELDS = ["entry", "status", "pages", "chars_before", "chars_after",
          "sentences_before", "sentences_after", "usable_after", "seconds", "detail"]

# Enough extractable text to be worth embedding; matches the threshold used elsewhere.
TEXT_THRESHOLD = 400


class OcrError(RuntimeError):
    pass


def ocrmypdf_available():
    return shutil.which("ocrmypdf") is not None


def extract_text(path):
    """All text in a PDF, or '' if it cannot be read."""
    import fitz

    try:
        document = fitz.open(str(path))
    except Exception:
        return ""
    try:
        return "".join(page.get_text("text") for page in document)
    finally:
        document.close()


def page_count(path):
    import fitz

    try:
        document = fitz.open(str(path))
    except Exception:
        return 0
    try:
        return document.page_count
    finally:
        document.close()


def assess(text):
    """Sentence yield and usability for some extracted text, via stage 2's own gate."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return 0, False
    report = document_quality_report(sentences)
    return report["kept_sentences"], report["usable"]


def needs_ocr(path):
    """True if this document has no usable text layer."""
    return len(extract_text(path).strip()) < TEXT_THRESHOLD


def group_by_content(paths):
    """Group identical files, so a shared document is OCR'd once rather than per entry.

    Several EXFOR entries routinely cite the same report — EANDC(E)-76 is cited by three,
    and one 215-page scan by another three. OCR is the slow step here, so processing each
    copy separately would triple the work and, worse, could yield slightly different text
    for entries that cite the same document.

    Returns [(representative path, [all paths with that content])].
    """
    import hashlib
    from collections import OrderedDict

    groups = OrderedDict()
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        groups.setdefault(digest, []).append(path)
    return [(members[0], members) for members in groups.values()]


def run_ocrmypdf(source, destination, *, language="eng", timeout=1800, jobs=2):
    """Run ocrmypdf over one document.

    ``--redo-ocr`` re-does any existing partial layer, which these scans often have from a
    poor earlier pass. ``--optimize 0`` keeps the run fast: these files are already large
    and shrinking them is not the point.
    """
    command = [
        "ocrmypdf", "--redo-ocr", "--optimize", "0", "--jobs", str(jobs),
        "--language", language, "--quiet", str(source), str(destination),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=timeout)
    except FileNotFoundError as error:
        raise OcrError("ocrmypdf is not installed") from error
    except subprocess.TimeoutExpired as error:
        raise OcrError(f"timed out after {timeout}s") from error

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        raise OcrError(detail[-1][:120] if detail else
                       f"exit {completed.returncode}")
    return destination


def process(path, work_dir, *, language="eng", timeout=1800, emit=print):
    """OCR one document and report whether it improved. Returns a log row."""
    entry = path.stem
    started = time.monotonic()
    before = extract_text(path)
    sentences_before, _ = assess(before)

    row = {"entry": entry, "status": "", "pages": page_count(path),
           "chars_before": len(before.strip()), "chars_after": "",
           "sentences_before": sentences_before, "sentences_after": "",
           "usable_after": "", "seconds": "", "detail": ""}

    destination = Path(work_dir) / f"{entry}.pdf"
    try:
        run_ocrmypdf(path, destination, language=language, timeout=timeout)
    except OcrError as error:
        row.update(status="failed", detail=str(error),
                   seconds=round(time.monotonic() - started, 1))
        emit(f"  {entry}: FAILED — {error}")
        return row

    after = extract_text(destination)
    sentences_after, usable = assess(after)
    row.update(chars_after=len(after.strip()), sentences_after=sentences_after,
               usable_after=usable, seconds=round(time.monotonic() - started, 1))

    if usable and sentences_after > sentences_before:
        row["status"] = "improved"
        emit(f"  {entry}: {row['pages']}p  {sentences_before} -> {sentences_after} "
             f"sentences ({row['seconds']}s)")
    else:
        # Keeping this would be worse than the honest empty document it replaces.
        row["status"] = "no_improvement"
        emit(f"  {entry}: no usable text after OCR "
             f"({sentences_after} sentences, {row['seconds']}s)")
    return row


def adopt(entry, work_dir, pdf_dir, emit=print):
    """Replace the scan with its OCR'd version, keeping the original alongside."""
    source = Path(work_dir) / f"{entry}.pdf"
    target = Path(pdf_dir) / f"{entry}.pdf"
    backup = Path(work_dir) / f"{entry}.original.pdf"
    if not backup.exists():
        shutil.copy2(target, backup)
    shutil.copy2(source, target)
    return target


def clear_overrides(entries, path=Path("data/report_overrides.json"), emit=print):
    """Remove the skip/ocr rules for documents that now have a text layer.

    Leaving them would silently exclude documents that are no longer broken — the failure
    mode an override file invites if nobody prunes it.
    """
    path = Path(path)
    if not path.is_file() or not entries:
        return 0
    document = json.loads(path.read_text())
    reports = document.get("reports", {})
    removed = 0
    for entry in entries:
        rule = reports.get(entry)
        if rule and rule.get("action") == "ocr" and rule.get("skip"):
            reports.pop(entry)
            removed += 1
    if removed:
        path.write_text(json.dumps(document, indent=2))
    return removed


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf-dir", default=str(DEFAULT_PDF_DIR))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--language", default="eng")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="seconds per document (these scans are long)")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-overrides", action="store_true",
                        help="do not prune skip/ocr rules for documents that improved")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    targets = sorted(p for p in pdf_dir.glob("*.pdf") if needs_ocr(p))
    if args.limit:
        targets = targets[:args.limit]

    if args.dry_run:
        print(f"{len(targets)} documents have no usable text layer:\n")
        for path in targets:
            print(f"   {path.stem}  {page_count(path):>4} pages  "
                  f"{path.stat().st_size / 1e6:>6.1f} MB")
        return 0

    if not ocrmypdf_available():
        print("ocrmypdf is not installed.\n"
              "    macOS:  brew install ocrmypdf\n"
              "    Linux:  apt install ocrmypdf  (or pip install ocrmypdf + tesseract)",
              file=sys.stderr)
        return 2

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(targets)} documents to OCR (originals preserved in {work_dir})\n")

    groups = group_by_content(targets)
    shared = sum(len(m) - 1 for _, m in groups)
    if shared:
        print(f"  {shared} of these are duplicate copies; OCR runs once per distinct "
              f"document ({len(groups)} runs)\n")

    rows = []
    improved = []
    for representative, members in groups:
        row = process(representative, work_dir, language=args.language,
                      timeout=args.timeout)
        rows.append(row)
        if row["status"] != "improved":
            # Record the same outcome for the other entries sharing this document.
            for other in members[1:]:
                rows.append(dict(row, entry=other.stem, detail="same document as "
                                 f"{representative.stem}"))
            continue

        improved.append(representative.stem)
        adopt(representative.stem, work_dir, pdf_dir)
        for other in members[1:]:
            # Copy the OCR'd text to every entry citing this document, so they agree.
            shutil.copy2(work_dir / f"{representative.stem}.pdf", work_dir / f"{other.stem}.pdf")
            adopt(other.stem, work_dir, pdf_dir)
            improved.append(other.stem)
            rows.append(dict(row, entry=other.stem,
                             detail=f"same document as {representative.stem}"))

    removed = 0
    if improved and not args.keep_overrides:
        removed = clear_overrides(improved)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    gained = sum(r["sentences_after"] - r["sentences_before"]
                 for r in rows if r["status"] == "improved")
    print(f"\n{'=' * 60}")
    print(f"  improved         {len(improved):>4}   adopted into {pdf_dir}")
    print(f"  no improvement   {sum(1 for r in rows if r['status'] == 'no_improvement'):>4}")
    print(f"  failed           {sum(1 for r in rows if r['status'] == 'failed'):>4}")
    print(f"  sentences gained {gained:>4,}")
    if removed:
        print(f"  pruned {removed} skip/ocr overrides that no longer apply")
    print(f"\nwrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
