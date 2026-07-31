#!/bin/bash
# Example SLURM submission for the NuGrade preprocessing pipeline.
#
#   sbatch slurm_example.sh
#
# Copy and edit for your account and partition. Nothing here needs Jupyter.

#SBATCH --job-name=nugrade-ingest
#SBATCH --output=logs/nugrade-%j.out
#SBATCH --error=logs/nugrade-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# #SBATCH --account=<your account>
# #SBATCH --partition=<your partition>

set -euo pipefail

mkdir -p logs

# --- environment -----------------------------------------------------------
# module load python/3.11
# source activate nuclear_data

# --- data locations --------------------------------------------------------
# These override the defaults in pipeline_config.py. A command-line flag beats an
# environment variable, which beats the default, so anything set here can still be
# overridden per-invocation.
export NUGRADE_X4_DB="/global/scratch/users/${USER}/x4pro/x4sqlite1.db"
export NUGRADE_ENDF71_TEMPLATE="/global/scratch/users/co_nuclear/endf71x/<symbol>/<ZAID>.710nc"
export NUGRADE_ENDF8_TEMPLATE="/global/home/groups/co_nuclear/serpent/xsdata/endf8/Lib80x/<symbol>/<ZAID>.800nc"
export NUGRADE_PDF_DIR="/global/scratch/users/${USER}/nugrade/pdfs"

# Write to scratch: the database is large and the run is long.
export NUGRADE_OUTPUT_DIR="/global/scratch/users/${USER}/nugrade/output"

# A progress line every 50 channels. Lines are flushed, so the log updates live.
export NUGRADE_LOG_EVERY=50

# --- validate before spending the allocation -------------------------------
# Exits non-zero if anything is missing, naming the variable to set. Cheap insurance
# against discovering a bad path three hours in.
python run_pipeline.py --stages 1,2,3 --dry-run

# --- run -------------------------------------------------------------------
# Stage 1 is resumable per reaction channel: if this job hits its time limit, resubmitting
# picks up at the next channel rather than starting over.
python run_pipeline.py --stages 1,2,3

# Exit codes: 0 all stages succeeded and the database satisfies the app contract,
#             1 a stage failed, or the finished database violates the contract,
#             2 a prerequisite is missing or the configuration is unusable.
