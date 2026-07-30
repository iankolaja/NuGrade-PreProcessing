# Scaling report acquisition and tokenization

## The problem

39 of 2,193 EXFOR entries have processed reports (1.8%). The KNN candidate pool — entries
that have both complete uncertainties *and* an embedding — is 33 reports. Everything the
project can conclude is bounded by that number, and the 39 were collected by hand.

Three separate bottlenecks, worth attacking in this order:

| Stage | Current state | Bottleneck |
|---|---|---|
| Identify what to fetch | manual | **no bibliographic data ingested** |
| Obtain the PDF | manual search | genuinely hard; many are inaccessible |
| Extract clean text | automated, unverified | 16.5% of sentences are garbage |
| Embed | automated | fine |

## Stage 1: get the bibliography — solved, no cluster needed

**IAEA serves the EXFOR bibliographic record directly**, so this stage needs neither X4Pro
nor cluster access:

    https://nds.iaea.org/exfor/servlet/X4sGetSubent?subID=<entry>001

returns the BIB section of subentry 001, which carries exactly what is missing:

    INSTITUTE  (1USACOL)
    REFERENCE  (J,PR,76,1750,4912)      <- journal, volume, page, year
    AUTHOR     (E.MELKONIAN)
    TITLE      SLOW NEUTRON VELOCITY SPECTROMETER STUDIES OF O2, N2, A, H2, H2O ...

`exfor_bib.py` fetches and parses this, caching every response on disk so the service is
asked for an entry at most once. `test_exfor_bib.py` covers the parser against verbatim
records, including the awkward real-world forms: nested parentheses in report codes
(`EANDC(E)-66`), an issue attached to the page with no comma (`917(19)`), issue labels
containing slashes (`(2/186)`), and two-digit years with a month (`4912` = Dec 1949).

### Measured coverage, 60-entry random sample

| | count | share |
|---|---|---|
| BIB record retrieved | 60/60 | **100%** |
| has a title | 59/60 | 98% |
| journal article with volume | 42/60 | 70% |

Reference types: 73% journal, 13% report, 5% private communication, 5% conference,
2% thesis, 2% progress report. Extrapolated to the full corpus: **~1,535 journal articles
and ~657 items of grey literature.**

### Measured DOI resolution, using title + volume + page

Strict matching — the candidate must agree on **both** volume and first page:

| | resolved |
|---|---|
| all journal articles | 6/18 (33%) |
| Western-indexed journals only | 6/10 (**60%**) |
| non-Western (Atomnaya Energiya, Chinese journals, ADP, FCY/L) | 0/8 |

**Zero false positives**, because a wrong article cannot agree on volume and page. The
unresolved Western cases are mostly pre-1960 papers for which Crossref simply has no
metadata — absent, not mismatched. Non-Western journals are not indexed by Crossref at all
and need a different route: Russian titles usually have a translated counterpart
(*Soviet Journal of Nuclear Physics*, *Soviet Atomic Energy*), and IAEA's INDC series
carries translations of many others.

Compare this with the author+year approach measured below: 60% correct with no false
positives, versus 85% "matches" that were mostly wrong.

---

## Superseded: ingesting X4Pro reference fields

The section below was the original plan, written before the IAEA endpoint was tested. It is
kept because the X4Pro route is still worth doing if you want the reference data offline and
in bulk without 2,193 HTTP requests — but it is no longer a blocker for anything.

`1_raw_data_ingestion.ipynb` queries X4Pro for physics columns only:

```sql
SELECT Reaction, Projectile, En, dEn, Sig, dSig, MT, DatasetID,
       Entry, Subent, YearRef1, Author1Ini, Author1, Target, fullCode FROM sig1
```

So the database has first author and year, but no journal, volume, page, report number, or
DOI — exactly the fields needed to look a paper up automatically. X4Pro carries EXFOR's
full REFERENCE field (codes like `(J,PR,80,34,1950)` = Physical Review vol. 80 p. 34, and
`(R,INDC(GER)-12,1975)` for laboratory reports).

Action: inspect the X4Pro schema on the cluster and add the reference columns to the
ingestion query, then to a `references` table. Concretely:

```sql
SELECT name FROM sqlite_master WHERE type='table';   -- find the reference/bib table
PRAGMA table_info(sig1);                             -- check for unused reference columns
```

Until this exists, every later stage is guesswork. After it exists, most of Stage 2 is
mechanical.

### This is a hard prerequisite, not an optimisation — measured

Tested against Crossref on 20 real entries using only the author and year the database
currently has:

- **Matching on surname + year: 17/20 "confident" matches, almost all wrong.** Houk 1971
  resolved to *Review of Social Economy*, Priesmeyer 1985 to *Journal of Counseling
  Psychology*, Frisch 1946 to *Econometrica*, Allen 1955 to *BMJ*, Bailey 1946 to *Design*.
  Common surnames match papers across all of science, and the match looks confident.
- **Adding a physics-journal constraint: 9/20, still with false positives.** Broecker 1966
  resolved to *Journal of Geophysical Research* — that is Wallace Broecker the geochemist,
  not the B. Broecker who measured hydrogen cross sections. Kirilyuk landed in solid-state
  physics. Realistically about 6 of 20 are correct, and nothing in the response
  distinguishes them.

A wrong DOI is worse than no DOI: it fetches a real paper about something else, which then
passes every quality gate and gets embedded as if it were that experiment's report. That
silently poisons both the RAG corpus and the KNN similarity features, and it would be very
hard to notice afterwards.

So author + year is not a usable fallback. Journal, volume and page from X4Pro are required
to disambiguate. Any resolution step should additionally:

1. Require agreement on journal **and** volume **and** first page, not just author/year.
2. Record the match evidence per entry, so a wrong match can be traced later.
3. Be spot-checked by hand on a sample before any bulk fetch — 20 entries is enough to
   catch a systematically broken matcher.

## Stage 2: obtain the PDFs

Split the corpus by what is actually obtainable, and do not treat it as one problem.

**Openly available, safe to automate.** These have real APIs and permissive terms:

- **OSTI** (`osti.gov`) — DOE laboratory reports (ORNL, LANL, ANL, KAPL). A large share of
  US measurements from the 1950s–70s are here, full text, openly licensed. Has a documented
  search API.
- **IAEA NDS** (`nds.iaea.org`) — INDC reports and the nuclear data documentation series,
  which EXFOR references heavily.
- **Unpaywall** (`api.unpaywall.org`) — takes a DOI, returns a legal open-access copy if one
  exists. Purpose-built for exactly this question. Requires only an email in the request.
- **Crossref** (`api.crossref.org`) — resolves author/year/journal/volume/page to a DOI.
  This is what Stage 1's reference data feeds.
- **NASA ADS** (`api.adsabs.harvard.edu`) — excellent coverage of older physics literature
  and often links scanned full text for pre-1990 articles. Free API key.
- **arXiv** — a small slice, but free.

Rate-limit politely, cache every response, and identify yourself in the User-Agent. These
are shared research services.

**Paywalled.** Do not build a bulk downloader for publisher sites (Elsevier, Springer, APS
and similar). It violates their terms, gets the campus IP range blocked — which harms the
whole group, not just this project — and is the single most likely way to turn a research
tool into an incident. Your Berkeley access is for reading papers, not for automated bulk
retrieval.

Instead: have the pipeline emit a **work queue** — a CSV of entries it could not obtain,
with the resolved citation and a DOI link. Then use interlibrary loan, which handles bulk
requests properly and is free to you. This converts an open-ended search problem into a
list someone can work through, which is the actual win over what you did by hand.

**Genuinely inaccessible.** Private communications, untranslated foreign-language theses,
lab reports that were never digitized. Record them as permanently unavailable with a reason
so nobody re-investigates them later. Being able to say "this entry is unobtainable because
X" is a legitimate output.

## Stage 3: extract clean text, verifiably

`report_quality.py` handles this, with thresholds derived from measuring the existing
corpus rather than guessed. Applied to the 7,193 sentences currently stored:

- 1,184 sentences (16.5%) are rejected — OCR damage, bibliography residue, journal
  furniture, fragments, non-English text
- 3 of 39 documents are flagged for re-OCR rather than embedded: entries 30077 (70% of
  sentences unusable), 30390 (68%), and 40061 (too few usable sentences)

Why this matters beyond tidiness: every stored sentence is embedded and therefore
retrievable by the agent's RAG tools, and the `{category}_max_sim` features are a **max**
over a report's sentences. One mangled line that happens to embed near a template inflates
that report's similarity score and corrupts the KNN neighbours chosen for it. Garbage in
this pipeline is not inert.

Most of these reports are scanned mid-century papers, so a poor text layer is the norm.
For documents the gate rejects, re-OCR with `ocrmypdf --redo-ocr` (Tesseract) before
giving up; that recovers a meaningful fraction and costs nothing but CPU.

## Stage 4: orchestrate

Make the pipeline resumable and idempotent, because it will be interrupted and because
most runs should do nothing:

- One row per entry in an `acquisition_status` table: `pending`, `resolved`, `fetched`,
  `extracted`, `embedded`, `unavailable`, with a reason and a timestamp.
- Cache every HTTP response on disk keyed by URL. Re-running should hit cache, not APIs.
- Never re-fetch or re-OCR a document that is already `embedded`.
- Process in small batches and commit after each, so a crash costs one batch.

This is also what makes the work parallelizable across a cluster job array later.

## Suggested order

1. ~~Ingest the X4Pro reference fields.~~ **Done differently:** `exfor_bib.py` fetches the
   same data from IAEA, no cluster needed. Run it over all 2,193 entries (about 20 minutes
   at the polite delay, then cached forever).
2. **Resolve journal articles to DOIs** via Crossref on title + volume + page. Measured at
   60% for Western-indexed journals with zero false positives. Route Russian and Chinese
   journals to their translated counterparts instead.
3. **Query the open sources** (OSTI, IAEA INDC, Unpaywall, ADS) for those DOIs and for the
   ~657 grey-literature items. Produces the real answer to "how many can we get for free" —
   still the most decision-relevant unknown.
4. **Wire in the quality gate + re-OCR fallback**, then embed.
5. **Emit the ILL work queue** for the paywalled remainder.

Steps 1–2 are now measured; step 3 is the remaining unknown. Together they produce the
number that should drive the rest of the project: not "how many papers can we scrape", but "what fraction of
EXFOR is reachable at all". If that is 60%, the imputation method is broadly applicable; if
it is 5%, the paper's framing needs to be about the accessible subset. That is worth
knowing before writing more pipeline code.
