# NuGrade-PreProcessing
A collection of notebooks for processing raw nuclear data for use in NuGrade. For every EXFOR measurement, a cross section value from evaluations like ENDF is interpolated, and error metrics are precomputed. The experimental reports from EXFOR are also tokenized and turned into sentence-wise embeddings. Missing uncertainty values are imputed using KNN. 

### Pre-requisites 
The following packages are needed to run NuGrade locally:
- X4Pro database (https://nds.iaea.org/cdroms/#x4pro1)
- ACE files for desired evaluations such as ENDF8
- Pandas
- PyTorch 
- NumPy
- OpenMC
- Spacy
- PyMuPDF
- Transformers
- Sklearn

### Running
1. Run 1_raw_data_ingestion.ipynb to start nugrade_data.db.
2. Place EXFOR experiment reports in pdfs with the EXFOR Entry as the file name.
3. Run 2_report_embedding.ipynb to generate tokens, sentence-wise embeddings, and similarity features.
4. Run 3_knn_imputation.ipynb to fill in missing uncertainty values using KNN.
5. Run `python validate_output_db.py` to check the database against the schema contract the NuGrade app enforces at startup (`nugrade/db_contract.py` in the NuGrade repo — keep the two in sync when changing the schema).
6. Place nugrade_data.db in the /data directory of your NuGrade installation.

Note: the sentence embeddings are attention-mask-weighted **mean-pooled** SciBERT vectors (see `get_embeddings_batch` in 2_report_embedding.ipynb). The NuGrade app embeds search queries with the identical pooling; the two must never diverge.
