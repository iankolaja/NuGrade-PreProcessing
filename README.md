# NuGrade-PreProcessing
A collection of notebooks for processing raw nuclear data for use in NuGrade. For every EXFOR measurement, a cross section value from evaluations like ENDF is interpolated, and error metrics are precomputed. The experimental reports from EXFOR are also tokenized and turned into sentence-wise embeddings. Missing uncertainty values are imputed using KNN. 

### Pre-requisites 
The following packages are needed to run NuGrade locally:
- ACE files for desired evaluations such as ENDF7
- Pandas
- PyTorch 
- NumPy
- OpenMC
- Spacy
- PyMuPDF
- Transformers
- Sklearn
