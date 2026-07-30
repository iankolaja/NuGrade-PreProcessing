"""Check whether a legal, free full text exists for each surveyed EXFOR entry.

    python check_availability.py --limit 300      # sample first
    python check_availability.py                  # everything

Reads output/coverage_survey.csv and asks, per entry:

  * journal articles with a DOI  -> Unpaywall: is there an open-access copy?
  * reports and progress reports -> OSTI: is the document there, and is full text attached?

Nothing is downloaded. This only records what exists and where, which is the number needed
to decide whether the corpus can realistically grow past its current 39 reports.

Two normalisations matter, both found by checking why specific lookups failed:

  * OSTI writes lab report numbers with a doubled dash before the numeric part —
    "ORNL--4805" for EXFOR's "ORNL-4805". A candidate is only accepted when the report
    number matches after normalising dashes, so a fuzzy hit on an unrelated document is
    rejected. OSTI's `report_number=` parameter is ignored by the API; `identifier=` works.

  * Paywalled publishers are never contacted. Unpaywall and OSTI are metadata services; the
    entries they cannot help with become an interlibrary-loan queue, not a scraping target.

Resumable and cached in the same way as survey_coverage.py.
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

UNPAYWALL_URL = "https://api.unpaywall.org/v2/{doi}"
OSTI_URL = "https://www.osti.gov/api/v1/records"
DEFAULT_IN = Path("output/coverage_survey.csv")
DEFAULT_OUT = Path("output/availability.csv")

FIELDS = ["entry", "route", "availability", "oa_status", "host", "url", "detail"]

# Reference types worth asking OSTI about: DOE laboratory and progress reports.
OSTI_TYPES = {"report", "progress_report"}


def _get_json(url, timeout=30):
    request = urllib.request.Request(
        url, headers={"User-Agent": "NuGrade-research/0.1 (mailto:ikolaja@berkeley.edu)"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def normalise_report_number(number):
    """Collapse dash runs and upper-case, so 'ORNL--4805' and 'ORNL-4805' compare equal."""
    return re.sub(r"-+", "-", str(number or "")).strip().upper()


def check_unpaywall(doi, email, delay=0.4, timeout=30):
    """Return an availability dict for a DOI, or None on lookup failure."""
    url = UNPAYWALL_URL.format(doi=urllib.parse.quote(doi, safe="/")) + \
        "?" + urllib.parse.urlencode({"email": email})
    try:
        payload = _get_json(url, timeout=timeout)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    finally:
        time.sleep(delay)

    if payload.get("error"):
        return {"availability": "not_in_unpaywall", "oa_status": "", "host": "",
                "url": "", "detail": str(payload.get("message", ""))[:80]}

    best = payload.get("best_oa_location") or {}
    if payload.get("is_oa"):
        pdf = best.get("url_for_pdf")
        return {
            "availability": "open_pdf" if pdf else "open_landing_page",
            "oa_status": payload.get("oa_status") or "",
            "host": best.get("host_type") or "",
            "url": pdf or best.get("url") or "",
            "detail": payload.get("journal_name") or "",
        }
    return {"availability": "closed", "oa_status": payload.get("oa_status") or "closed",
            "host": "", "url": "", "detail": payload.get("journal_name") or ""}


def check_osti(report_code, delay=0.5, timeout=30):
    """Look up a laboratory report in OSTI and report whether full text is attached.

    Verifies the returned report number matches after dash normalisation, so a fuzzy hit on
    an unrelated document is not counted as availability.
    """
    if not report_code:
        return None
    query = urllib.parse.urlencode({"identifier": report_code, "rows": 5})
    try:
        payload = _get_json(f"{OSTI_URL}?{query}", timeout=timeout)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    finally:
        time.sleep(delay)

    if not isinstance(payload, list) or not payload:
        return {"availability": "not_in_osti", "oa_status": "", "host": "",
                "url": "", "detail": ""}

    wanted = normalise_report_number(report_code)
    for record in payload:
        numbers = [normalise_report_number(n)
                   for n in re.split(r"[;,]", str(record.get("report_number") or ""))]
        if wanted not in numbers:
            continue
        links = record.get("links") or []
        fulltext = next((l for l in links if l.get("rel") == "fulltext"), None)
        citation = next((l for l in links if l.get("rel") == "citation"), None)
        return {
            "availability": "osti_fulltext" if fulltext else "osti_record_only",
            "oa_status": "osti",
            "host": "osti",
            "url": (fulltext or citation or {}).get("href", ""),
            "detail": str(record.get("title") or "")[:80],
        }
    return {"availability": "not_in_osti", "oa_status": "", "host": "", "url": "",
            "detail": f"{len(payload)} results, no exact report-number match"}


def check_entry(row, email, delay):
    """Route one surveyed entry to the right service and return an availability row."""
    out = {k: "" for k in FIELDS}
    out["entry"] = row["entry"]

    if row["status"] == "doi_resolved" and row.get("doi"):
        out["route"] = "unpaywall"
        result = check_unpaywall(row["doi"], email, delay=delay)
    elif row["ref_type"] in OSTI_TYPES:
        out["route"] = "osti"
        result = check_osti(row.get("code"), delay=delay)
    else:
        out["route"] = "none"
        out["availability"] = "no_route"
        out["detail"] = row["ref_type"] or row["status"]
        return out

    if result is None:
        out["availability"] = "lookup_failed"
        return out
    out.update(result)
    return out


def summarise(rows, surveyed):
    total = len(rows)
    counts = {}
    for r in rows:
        counts[r["availability"]] = counts.get(r["availability"], 0) + 1

    free = sum(counts.get(k, 0) for k in ("open_pdf", "open_landing_page", "osti_fulltext"))
    print(f"\n{'=' * 66}\nAVAILABILITY — {total} ENTRIES CHECKED\n{'=' * 66}")
    for status, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {status:24s} {n:>5}  {100 * n / total:5.1f}%")
    print(f"\n  FREELY OBTAINABLE        {free:>5}  {100 * free / total:5.1f}% of checked")
    if surveyed:
        print(f"  as a share of the corpus {free:>5}  {100 * free / surveyed:5.1f}% of {surveyed} entries")

    by_route = {}
    for r in rows:
        key = (r["route"], r["availability"])
        by_route[key] = by_route.get(key, 0) + 1
    print("\n  by route:")
    for (route, status), n in sorted(by_route.items(), key=lambda kv: (kv[0][0], -kv[1])):
        print(f"    {route:10s} {status:24s} {n:>5}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey", default=str(DEFAULT_IN))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.4)
    parser.add_argument("--email", default="ikolaja@berkeley.edu")
    args = parser.parse_args()

    survey_path, out_path = Path(args.survey), Path(args.out)
    if not survey_path.is_file():
        print(f"no survey at {survey_path}; run survey_coverage.py first")
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)

    surveyed = [r for r in csv.DictReader(survey_path.open(newline=""))]
    # Only entries with somewhere to look.
    candidates = [r for r in surveyed
                  if (r["status"] == "doi_resolved" and r.get("doi"))
                  or r["ref_type"] in OSTI_TYPES]
    if args.limit:
        candidates = candidates[:args.limit]

    done = set()
    if out_path.is_file():
        done = {r["entry"] for r in csv.DictReader(out_path.open(newline=""))}
    todo = [r for r in candidates if r["entry"] not in done]

    print(f"checkable entries: {len(candidates)} | already done: {len(done)} | to do: {len(todo)}")

    is_new = not out_path.is_file()
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        for i, row in enumerate(todo, 1):
            result = check_entry(row, args.email, args.delay)
            writer.writerow(result)
            f.flush()
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} … {result['entry']} -> {result['availability']}",
                      flush=True)

    checked = {r["entry"] for r in candidates}
    rows = [r for r in csv.DictReader(out_path.open(newline="")) if r["entry"] in checked]
    summarise(rows, len(surveyed))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
