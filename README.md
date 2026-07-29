# NuGrade-PreProcessing
A collection of notebooks for processing raw nuclear data for use in NuGrade. For every EXFOR measurement, a cross section value from evaluations like ENDF is interpolated, and error metrics are precomputed. The experimental reports from EXFOR are also tokenized and turned into sentence-wise embeddings. Missing uncertainty values are imputed using KNN. 

### Pre-requisites 
The following packages are needed to run NuGrade locally:
- X4Pro database (https://nds.iaea.org/cdroms/#x4pro1)
- ACE files for desired evaluations such as ENDF8
- Pandas
- PyTorch 
- NumPy
- OpenMC (needed only to read ACE/HDF5 evaluation files, i.e. only where
  `1_raw_data_ingestion.ipynb` runs. It is imported lazily, so the rest of the
  repo — including the test suite — works without it.)
- Spacy
- PyMuPDF
- Transformers
- Sklearn

### Testing

The numerical work is extracted out of the notebooks into modules so it can be
unit-tested without a cluster, ENDF files, or the X4Pro database:

    pytest

- `ingestion.py` / `test_ingestion.py` — per-channel assumed uncertainties, chi-squared,
  relative error (used by `1_raw_data_ingestion.ipynb`)
- `imputation.py` / `test_imputation.py` — composite distance, neighbour weighting,
  feature standardization (used by `3_knn_imputation.ipynb`)
- `helper_functions.py` / `test_helper_functions.py` — EXFOR target parsing and the
  element-to-proton-number map

### Running
1. Run 1_raw_data_ingestion.ipynb to start nugrade_data.db.
2. Place EXFOR experiment reports in pdfs with the EXFOR Entry as the file name.
3. Run 2_report_embedding.ipynb to generate tokens, sentence-wise embeddings, and similarity features.
4. Run 3_knn_imputation.ipynb to fill in missing uncertainty values using KNN.
5. Run `python validate_output_db.py` to check the database against the schema contract the NuGrade app enforces at startup (`nugrade/db_contract.py` in the NuGrade repo — keep the two in sync when changing the schema).
6. Place nugrade_data.db in the /data directory of your NuGrade installation.

Note: the sentence embeddings are attention-mask-weighted **mean-pooled** SciBERT vectors (see `get_embeddings_batch` in 2_report_embedding.ipynb). The NuGrade app embeds search queries with the identical pooling; the two must never diverge.

### Repairing an existing database

`repair_derived_columns.py` fixes two ingestion bugs in an already-built database without
re-running the notebooks against the cluster (the interpolated evaluation cross sections
are already stored in the `endf8` / `endf7-1` columns):

    python repair_derived_columns.py output/nugrade_data.db output/nugrade_data_fixed.db
    python validate_output_db.py output/nugrade_data_fixed.db

It writes a repaired copy and never modifies its input. Afterwards, re-run
`3_knn_imputation.ipynb` against the repaired file so the KNN imputation uses corrected
inputs.

### Rebuilding on the cluster and comparing

`1_raw_data_ingestion.ipynb` is the only notebook that needs cluster access (ENDF/ACE
files and the X4Pro database). Notebook 2 needs the report PDFs; notebook 3 needs only
the database. So a rebuild does not require re-running everything:

1. On the cluster, run `1_raw_data_ingestion.ipynb` into a fresh `output/`.
2. Copy the resulting `nugrade_data.db` down.
3. Graft the embedding tables from your existing database, so notebook 2 and the PDFs are
   not needed — `report_embeddings` and `sentence_embeddings` are keyed on EXFOR_Entry and
   are unaffected by an ingestion re-run:

       python graft_embedding_tables.py old/nugrade_data.db new/nugrade_data.db

4. Run `3_knn_imputation.ipynb` against the new database to redo the KNN imputation.
5. Compare against the previous build and confirm the fixes landed:

       python compare_databases.py old/nugrade_data.db new/nugrade_data.db
       python validate_output_db.py new/nugrade_data.db

`compare_databases.py` checks that raw EXFOR quantities are unchanged (only derived
columns should move), that no derived uncertainty or chi-squared is negative or infinite,
and that chi-squared matches `((Data - eval) / sigma)^2`. Both scripts are read-only with
respect to the databases they are given, except for `graft_embedding_tables.py`, which
writes only to its destination.
