# NuGrade-PreProcessing

Builds the SQLite database the NuGrade app consumes. For every EXFOR measurement a cross
section from an evaluation such as ENDF/B-VIII is interpolated and error metrics are
precomputed; experimental reports are turned into sentence embeddings; and missing
uncertainties are imputed by KNN over report similarity.

Everything runs as a plain Python script — no Jupyter required. The notebooks are thin
wrappers around the same modules, kept for interactive exploration.

## Running

```bash
python run_pipeline.py --stages 1,2,3 --dry-run   # validate configuration, run nothing
python run_pipeline.py --stages 1,2,3             # everything
python run_pipeline.py --stages 1                 # ingestion only (the usual cluster job)
python run_pipeline.py --stages 3                 # re-impute an existing database
```

Individual stages also run on their own:

```bash
python stage1_ingestion.py --phase measurements
python stage2_embedding.py --pdf-dir pdfs
python stage3_imputation.py
```

Exit codes: `0` success, `1` a stage failed or the finished database violates the app
contract, `2` a prerequisite is missing or the configuration is unusable.

See `slurm_example.sh` for a cluster submission template.

### Configuration

Settings resolve **command-line flag > environment variable > default**. Environment
variables suit SLURM scripts; `python run_pipeline.py --help` lists every flag with its
variable.

| Variable | Purpose |
|---|---|
| `NUGRADE_X4_DB` | X4Pro sqlite database |
| `NUGRADE_ENDF71_TEMPLATE` | ACE path template, with `<symbol>` and `<ZAID>` |
| `NUGRADE_ENDF8_TEMPLATE` | as above, for ENDF/B-VIII |
| `NUGRADE_PDF_DIR` | report PDFs, named by EXFOR entry (`10283.pdf`) |
| `NUGRADE_OUTPUT_DIR` | output directory; the database path derives from it |
| `NUGRADE_K_NEIGHBORS` | KNN neighbours (default 5) |
| `NUGRADE_LOG_EVERY` | progress line cadence |

`--dry-run` validates every path before any work starts, and names the flag or variable to
set for each problem. Worth doing before spending a cluster allocation.

### Reading the log

Every run opens with a header giving the host, PID, SLURM job id, and each resolved path
marked `[dir ok]`, `[N MB]` or `[MISSING]` — so a job that picked up the wrong inputs is
obvious from the first ten lines rather than from a failure an hour later.

Progress lines carry wall-clock time as well as elapsed (`[14:32:07 1 +1:04:12]`), because
the first question about a stalled job is *when* it stopped. Long single operations announce
themselves before starting and report their duration on completion, so a fifteen-minute
table read is visible as work rather than as silence.

Progress emits on whichever comes first: `--log-every` items, or sixty seconds. The time
trigger is what keeps a slow stage distinguishable from a hung one; without it a
count-based cadence goes quiet for exactly as long as the work takes.

### Resuming

Stage 1 runs for hours and is resumable per reaction channel: each channel's rows and its
progress marker are written in one transaction, so an interrupted job continues at the next
channel rather than starting over. Stage 2 resumes per report. Pass `--no-resume` to
recompute from scratch.

## Stages

| Stage | Needs | Produces |
|---|---|---|
| `1` ingestion | X4Pro + cluster ACE files | `measurements`, `all_reactions.csv` |
| `1b` aggregates | the database only | `subentries`, `entries` |
| `2` embedding | report PDFs, SciBERT | `report_embeddings`, `sentence_embeddings` |
| `3` imputation | the database only | `dData_adopted`, `uncertainty_source` |

Naming stage `1` runs `1b` too. Stages always execute in dependency order regardless of the
order given.

## Pre-requisites

```bash
pip install -r requirements.txt              # numpy + pandas; that is all stages 1 and 3 need
conda install -c conda-forge openmc          # stage 1 only; not a pip package

# stage 2 only — skip unless you are embedding reports
pip install -r requirements-embedding.txt
python -m spacy download en_core_web_sm
```

The split matters on a cluster: `requirements-embedding.txt` pulls torch, transformers and
spaCy, which is a multi-gigabyte resolve and long enough on a slow link to look like a hang.
Stage 1 needs none of it. If pip must fetch torch, the CPU-only build is far smaller and is
all this pipeline uses — SciBERT over a few hundred documents is not GPU-bound:

    pip install torch --index-url https://download.pytorch.org/whl/cpu

Data, which is not in the repository:

- X4Pro database (https://nds.iaea.org/cdroms/#x4pro1) — stage 1 only
- ACE files for the desired evaluations — stage 1 only
- **Report PDFs — stage 2 only.** These are gitignored (662 MB) and do **not** arrive with a
  `git pull`. Either copy `pdfs/` across:

      rsync -avz --progress pdfs/ user@cluster:/path/to/NuGrade-PreProcessing/pdfs/

  or rebuild them there with `fetch_reports.py`, `fetch_iaea.py` and `ocr_reports.py`, which
  takes about an hour and needs `ocrmypdf` installed.

Every heavy dependency is imported lazily, so stages 1 and 3 and the entire test suite run
without OpenMC, torch, spaCy or PyMuPDF installed.

## Testing

```bash
pytest
```

The full suite runs on a laptop with no cluster, no ACE files, no X4Pro database, and
without torch, spaCy or PyMuPDF installed — heavy dependencies sit behind injectable
interfaces (`evaluations.py`, `embedding_backend.py`) that tests replace with analytic
stubs.

| Module | Tests |
|---|---|
| `pipeline_config.py` | flag/environment precedence, prerequisite validation |
| `ingestion.py` | assumed uncertainties, chi-squared, relative error |
| `imputation.py` | composite distance, neighbour weighting, standardisation |
| `report_quality.py` | PDF-text quality gates, calibrated on real corpus garbage |
| `helper_functions.py` | EXFOR target parsing, element-to-proton-number map |
| `exfor_bib.py` / `resolve_doi.py` | bibliographic parsing, DOI resolution |
| `stage1/2/3`, `run_pipeline.py` | orchestration: resume, ordering, persistence, schema |

## Per-report handling

A few documents need individual treatment — an image-only scan, a compiled report cited at
one chapter, a non-English report. Those rules live in `data/report_overrides.json`:

```json
"10104": {"reason": "69-page image-only scan; no text layer", "skip": true, "action": "ocr"},
"10374": {"reason": "cites one chapter of a 477-page annual report", "pages": "20-40"}
```

They are kept in a file rather than in the database because they are curation, not derived
data: stage 1 rebuilds the database wholesale and would erase them. Every entry needs a
`reason`, and loading fails on an unknown key — a silently ignored typo would mean the
override never applies. Supported keys: `skip`, `action`, `pages`, `strip_references`,
`language`, `duplicate_of`, and the three quality-gate thresholds.

Whatever stage 2 applies is written to a `report_overrides_applied` table, so a surprising
embedding can be traced back to the rule responsible.

## The database contract

The NuGrade app enforces a schema contract at startup (`nugrade/db_contract.py` in that
repo). Check a build against it:

```bash
python validate_output_db.py output/nugrade_data.db
```

`run_pipeline.py` runs this automatically after stage 3 and fails if the database would be
rejected — so a cluster job cannot quietly ship an unusable file. Keep the two in sync when
changing the schema.

Sentence embeddings are attention-mask-weighted **mean-pooled** SciBERT vectors. The app
embeds search queries with identical pooling; if the two diverge, queries and documents land
in different vector spaces and retrieval silently degrades.

## Repairing an existing database

`repair_derived_columns.py` fixes the chi-squared formula and negative derived
uncertainties in an already-built database without cluster access, since the interpolated
evaluation cross sections are already stored:

```bash
python repair_derived_columns.py output/nugrade_data.db output/nugrade_data_fixed.db
python validate_output_db.py output/nugrade_data_fixed.db
python run_pipeline.py --stages 3 --db output/nugrade_data_fixed.db
```

It writes a repaired copy and never modifies its input.

## Rebuilding and comparing

Only stage 1 needs the cluster. If the report PDFs are not where you are rebuilding, the
embedding tables can be carried across instead of regenerated — they key on EXFOR entry and
are unaffected by an ingestion re-run:

```bash
python graft_embedding_tables.py old/nugrade_data.db new/nugrade_data.db
python run_pipeline.py --stages 3 --output-dir new/
python compare_databases.py old/nugrade_data.db new/nugrade_data.db
```

`compare_databases.py` checks that raw EXFOR quantities are unchanged — only derived columns
should move — that no derived uncertainty or chi-squared is negative or infinite, and that
chi-squared matches `((Data - eval) / sigma)^2`.

## Corpus acquisition

`ACQUISITION.md` covers scaling the report corpus past hand-collection: fetching EXFOR
bibliographic records, resolving DOIs, and checking open-access availability. The supporting
tools are `exfor_bib.py`, `resolve_doi.py`, `survey_coverage.py`, `check_availability.py`
and `make_work_queues.py`.
