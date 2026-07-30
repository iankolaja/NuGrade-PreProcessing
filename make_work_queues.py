"""Turn the survey and availability results into work queues someone can act on.

    python make_work_queues.py

Writes three files to output/:

  fetch_queue.csv       — free full text exists; a script can retrieve these
  ill_queue.csv         — identified but paywalled; request through interlibrary loan
  unreachable.csv       — nothing to try, with the reason recorded

The point of the third file is that "we cannot get this, because X" is a legitimate
research output. Recording it stops the same dead ends being re-investigated later, and
makes the accessible fraction of EXFOR reportable in the paper.

Pure data transform over the two CSVs — no network access.
"""
import argparse
import csv
import sys
from pathlib import Path

FREE = {"open_pdf", "open_landing_page", "osti_fulltext"}
PAYWALLED = {"closed"}
# OSTI has a catalogue record but no attached document. Not free, but a librarian has
# something concrete to work from, so these join the ILL queue rather than being written off.
RECORD_ONLY = {"osti_record_only"}

QUEUE_FIELDS = ["entry", "citation", "title", "doi", "url", "availability", "route"]
UNREACHABLE_FIELDS = ["entry", "citation", "title", "ref_type", "reason"]


def citation_of(row):
    """A human-readable citation from the EXFOR reference fields."""
    parts = [p for p in [row.get("journal") or row.get("code"),
                         f"vol {row['volume']}" if row.get("volume") else "",
                         f"p {row['page']}" if row.get("page") else "",
                         f"({row['year']})" if row.get("year") else ""] if p]
    return " ".join(parts)


def reason_for(survey_row, availability):
    """Why this entry cannot be obtained, in words a person can act on or accept."""
    ref_type = survey_row["ref_type"]
    if availability == "not_in_osti":
        return "laboratory report not held by OSTI; try the issuing laboratory or IAEA INDC"
    if ref_type == "private_communication":
        return "private communication; no published document exists"
    if ref_type == "thesis":
        return "thesis; try the awarding institution's repository"
    if ref_type in ("conference", "proceedings"):
        return "conference paper; try the proceedings volume or IAEA INDC"
    if survey_row["status"] == "doi_unresolved":
        code = (survey_row.get("code") or "").upper()
        if code in {"AE", "YK", "YF", "SJA", "SNP", "UFZ", "ZET", "BAS"}:
            return ("Soviet/Russian journal; the translation renumbers volumes, so needs "
                    "title matching against the translated edition or IAEA INDC")
        if code == "BAP":
            return "APS meeting abstract; generally has no published full text"
        return f"no DOI found for {code or 'this journal'}"
    if survey_row["status"] == "no_reference":
        return "EXFOR entry has no REFERENCE field"
    return "no route available"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey", default="output/coverage_survey.csv")
    parser.add_argument("--availability", default="output/availability.csv")
    parser.add_argument("--outdir", default="output")
    args = parser.parse_args()

    survey_path, avail_path = Path(args.survey), Path(args.availability)
    for path in (survey_path, avail_path):
        if not path.is_file():
            print(f"missing {path}; run survey_coverage.py and check_availability.py first")
            return 2

    survey = {r["entry"]: r for r in csv.DictReader(survey_path.open(newline=""))}
    availability = {r["entry"]: r for r in csv.DictReader(avail_path.open(newline=""))}

    fetch, ill, unreachable = [], [], []
    for entry, row in survey.items():
        avail = availability.get(entry, {})
        status = avail.get("availability", "")
        common = {"entry": entry, "citation": citation_of(row),
                  "title": row.get("title", ""), "doi": row.get("doi", ""),
                  "url": avail.get("url", ""), "availability": status,
                  "route": avail.get("route", "")}

        if status in FREE:
            fetch.append(common)
        elif status in PAYWALLED or status in RECORD_ONLY:
            ill.append(common)
        else:
            unreachable.append({"entry": entry, "citation": citation_of(row),
                                "title": row.get("title", ""),
                                "ref_type": row.get("ref_type", ""),
                                "reason": reason_for(row, status)})

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def write(name, rows, fields):
        path = outdir / name
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: r["entry"]))
        print(f"  {path}  {len(rows)} rows")

    print("wrote:")
    write("fetch_queue.csv", fetch, QUEUE_FIELDS)
    write("ill_queue.csv", ill, QUEUE_FIELDS)
    write("unreachable.csv", unreachable, UNREACHABLE_FIELDS)

    total = len(survey)
    print(f"\n  free to fetch      {len(fetch):>5}  {100*len(fetch)/total:5.1f}% of {total} entries")
    print(f"  interlibrary loan  {len(ill):>5}  {100*len(ill)/total:5.1f}%")
    print(f"  unreachable        {len(unreachable):>5}  {100*len(unreachable)/total:5.1f}%")

    reasons = {}
    for r in unreachable:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    print("\n  unreachable, by reason:")
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"    {n:>5}  {reason[:74]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
