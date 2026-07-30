"""Survey how much of the EXFOR corpus is reachable: bibliography, then DOI.

    python survey_coverage.py --limit 300          # sanity-check batch
    python survey_coverage.py                      # all entries

Answers the question that should drive the project: not "how many papers can we scrape",
but what fraction of EXFOR can be identified and located at all.

Resumable and idempotent by design, because this will be interrupted and because most runs
should do nothing. Progress is appended to a CSV; entries already in it are skipped, and
IAEA responses are cached on disk, so a second run makes no network requests at all.

Every DOI recorded here agreed with the EXFOR reference on volume and first page. Nothing
is guessed: an unresolved entry is left unresolved rather than filled with a likely match.
"""
import argparse
import csv
import random
import sqlite3
import sys
from pathlib import Path

from exfor_bib import ExforFetchError, fetch_bib
from resolve_doi import load_ads_token, resolve

DEFAULT_DB = "/Users/iankolaja/NuGrade/data/nugrade_data.db"
DEFAULT_OUT = Path("output/coverage_survey.csv")

FIELDS = ["entry", "status", "ref_type", "journal", "code", "volume", "page", "year",
          "doi", "doi_source", "evidence", "title"]


def load_entries(db_path):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return [r[0] for r in con.execute(
            "SELECT DISTINCT EXFOR_Entry FROM measurements ORDER BY EXFOR_Entry")]
    finally:
        con.close()


def already_done(out_path):
    if not out_path.is_file():
        return set()
    with out_path.open(newline="") as f:
        return {row["entry"] for row in csv.DictReader(f)}


def survey_entry(entry, token, email, delay):
    """Return one CSV row for an entry: bibliography, then DOI if it is a journal article."""
    row = {k: "" for k in FIELDS}
    row["entry"] = entry

    try:
        bib = fetch_bib(entry, delay=delay, email=email)
    except ExforFetchError:
        row["status"] = "bib_unavailable"
        return row

    reference = bib.get("reference") or {}
    row.update({
        "ref_type": reference.get("type") or "",
        "journal": reference.get("journal") or "",
        "code": reference.get("code") or "",
        "volume": reference.get("volume") or "",
        "page": reference.get("page") or "",
        "year": reference.get("year") or "",
        "title": (bib.get("title") or "")[:200],
    })

    if not reference:
        row["status"] = "no_reference"
        return row

    if reference.get("type") != "journal":
        # Grey literature: no DOI exists to find. Routed to OSTI / IAEA / ILL instead.
        row["status"] = "grey_literature"
        return row

    match = resolve(bib, token=token, email=email, delay=delay)
    if match:
        row.update({"status": "doi_resolved", "doi": match["doi"] or "",
                    "doi_source": match["source"], "evidence": match["evidence"]})
    else:
        row["status"] = "doi_unresolved"
    return row


def summarise(rows):
    total = len(rows)
    if not total:
        print("no rows")
        return

    def count(**criteria):
        return sum(1 for r in rows if all(r.get(k) == v for k, v in criteria.items()))

    def pct(n):
        return f"{100 * n / total:5.1f}%"

    bib_ok = total - count(status="bib_unavailable")
    titled = sum(1 for r in rows if r["title"])
    journal = sum(1 for r in rows if r["ref_type"] == "journal")
    resolved = count(status="doi_resolved")
    unresolved = count(status="doi_unresolved")
    grey = count(status="grey_literature")

    print(f"\n{'=' * 66}\nSURVEYED {total} ENTRIES\n{'=' * 66}")
    print(f"  BIB record retrieved      {bib_ok:>5} / {total}  {pct(bib_ok)}")
    print(f"  has a title               {titled:>5} / {total}  {pct(titled)}")
    print(f"  journal article           {journal:>5} / {total}  {pct(journal)}")
    print(f"  grey literature           {grey:>5} / {total}  {pct(grey)}")
    print(f"\n  DOI resolved              {resolved:>5} / {total}  {pct(resolved)}")
    print(f"  DOI unresolved            {unresolved:>5} / {total}  {pct(unresolved)}")
    if journal:
        print(f"  resolution rate among journal articles: "
              f"{100 * resolved / journal:.1f}%")

    by_source = {}
    for r in rows:
        if r["doi_source"]:
            by_source[r["doi_source"]] = by_source.get(r["doi_source"], 0) + 1
    if by_source:
        print("\n  resolved by service:")
        for source, n in sorted(by_source.items(), key=lambda kv: -kv[1]):
            print(f"    {source:12s} {n:>5}")

    types = {}
    for r in rows:
        if r["ref_type"]:
            types[r["ref_type"]] = types.get(r["ref_type"], 0) + 1
    print("\n  reference types:")
    for kind, n in sorted(types.items(), key=lambda kv: -kv[1]):
        print(f"    {kind:22s} {n:>5}  {pct(n)}")

    # Journals that most often fail to resolve — where the next effort should go.
    failures = {}
    for r in rows:
        if r["status"] == "doi_unresolved" and r["journal"]:
            failures[r["journal"]] = failures.get(r["journal"], 0) + 1
    if failures:
        print("\n  unresolved journal articles, by journal (top 10):")
        for journal_name, n in sorted(failures.items(), key=lambda kv: -kv[1])[:10]:
            print(f"    {journal_name[:44]:44s} {n:>4}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--limit", type=int, default=None,
                        help="survey a random sample of this many entries")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=0.4,
                        help="seconds between live requests to each service")
    parser.add_argument("--email", default="ikolaja@berkeley.edu")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    entries = load_entries(args.db)
    if args.limit:
        random.seed(args.seed)
        entries = sorted(random.sample(entries, min(args.limit, len(entries))))

    done = already_done(out_path)
    todo = [e for e in entries if e not in done]

    token = load_ads_token()
    print(f"entries selected: {len(entries)} | already surveyed: {len(done)} | to do: {len(todo)}")
    print(f"ADS token: {'present' if token else 'ABSENT (Crossref only)'}")
    if not todo:
        print("nothing to do — reading existing results")

    is_new_file = not out_path.is_file()
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new_file:
            writer.writeheader()
        for i, entry in enumerate(todo, 1):
            row = survey_entry(entry, token, args.email, args.delay)
            writer.writerow(row)
            f.flush()  # commit each row: a crash costs one entry, not the run
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} … {entry} -> {row['status']}", flush=True)

    with out_path.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["entry"] in set(entries)]
    summarise(rows)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
