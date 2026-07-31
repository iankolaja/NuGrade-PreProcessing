"""Shared fixtures: small synthetic databases that mirror the real schema.

Built to be readable rather than realistic — every count in a test should be traceable to a
row written here.
"""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from imputation import SIMILARITY_LABELS

EMBEDDING_DIM = 768


def _embedding(seed):
    """A deterministic float32 vector, matching what stage 2 stores."""
    rng = np.random.default_rng(seed)
    return rng.random(EMBEDDING_DIM, dtype=np.float32)


def make_measurements():
    """Three entries: one complete, one missing every uncertainty, one unembedded.

    Entry 10001 is the only viable KNN candidate; 10002 is the imputation target; 10003 has
    no embedding, so it must fall back to the quantile assumption.
    """
    rows = []
    for i, energy in enumerate([1e3, 1e4, 1e5, 1e6]):
        rows.append({"EXFOR_Entry": "10001", "EXFOR_Subentry": "10001002",
                     "Z": 3, "A": 7, "MT": 1, "Projectile": "n", "Reaction": "N,TOT",
                     "Element": "Li", "Energy": energy, "Data": 2.0 + i,
                     "dData": 0.2 + 0.1 * i, "dData_assumed": 0.2 + 0.1 * i,
                     "Dataset_Number": "1", "Year": 1970, "Author": "Alpha",
                     "dEnergy": np.nan, "endf8": 2.0, "endf8_chi_squared": 1.0,
                     "endf8_relative_error": 1.0, "endf7-1": 2.0,
                     "endf7-1_chi_squared": 1.0, "endf7-1_relative_error": 1.0})
    for i, energy in enumerate([2e3, 2e4]):
        rows.append({"EXFOR_Entry": "10002", "EXFOR_Subentry": "10002002",
                     "Z": 3, "A": 7, "MT": 1, "Projectile": "n", "Reaction": "N,TOT",
                     "Element": "Li", "Energy": energy, "Data": 3.0 + i,
                     "dData": np.nan, "dData_assumed": 0.9,
                     "Dataset_Number": "2", "Year": 1980, "Author": "Beta",
                     "dEnergy": np.nan, "endf8": 3.0, "endf8_chi_squared": 1.0,
                     "endf8_relative_error": 1.0, "endf7-1": 3.0,
                     "endf7-1_chi_squared": 1.0, "endf7-1_relative_error": 1.0})
    # Negative cross section: EXFOR contains these, and the adopted uncertainty must
    # still come out positive.
    rows.append({"EXFOR_Entry": "10003", "EXFOR_Subentry": "10003002",
                 "Z": 26, "A": 56, "MT": 102, "Projectile": "n", "Reaction": "N,G",
                 "Element": "Fe", "Energy": 5e3, "Data": -1.5,
                 "dData": np.nan, "dData_assumed": 0.4,
                 "Dataset_Number": "3", "Year": 1990, "Author": "Gamma",
                 "dEnergy": np.nan, "endf8": 1.0, "endf8_chi_squared": 1.0,
                 "endf8_relative_error": 1.0, "endf7-1": 1.0,
                 "endf7-1_chi_squared": 1.0, "endf7-1_relative_error": 1.0})
    return pd.DataFrame(rows)


def make_reports(entries=("10001", "10002")):
    """One row per embedded report, with the nine similarity features stage 2 produces."""
    rows = []
    for seed, entry in enumerate(entries):
        row = {"EXFOR_Entry": entry}
        row.update({label: 0.5 + 0.01 * seed for label in SIMILARITY_LABELS})
        row["mean_embedding"] = _embedding(seed).tobytes()
        rows.append(row)
    return pd.DataFrame(rows)


def make_entries():
    return pd.DataFrame([
        {"EXFOR_Entry": "10001", "Uncertainty_Complete": 1, "Num_Measurements": 4,
         "Num_Missing_Uncertainty": 0, "Num_Subentries": 1},
        {"EXFOR_Entry": "10002", "Uncertainty_Complete": 0, "Num_Measurements": 2,
         "Num_Missing_Uncertainty": 2, "Num_Subentries": 1},
        {"EXFOR_Entry": "10003", "Uncertainty_Complete": 0, "Num_Measurements": 1,
         "Num_Missing_Uncertainty": 1, "Num_Subentries": 1},
    ])


def make_subentries():
    return pd.DataFrame([
        {"EXFOR_Entry": "10001", "EXFOR_Subentry": "10001002", "Z": 3, "A": 7, "MT": 1,
         "Reaction": "N,TOT", "Element": "Li", "E_min": 1e3, "E_max": 1e6},
        {"EXFOR_Entry": "10002", "EXFOR_Subentry": "10002002", "Z": 3, "A": 7, "MT": 1,
         "Reaction": "N,TOT", "Element": "Li", "E_min": 2e3, "E_max": 2e4},
        {"EXFOR_Entry": "10003", "EXFOR_Subentry": "10003002", "Z": 26, "A": 56, "MT": 102,
         "Reaction": "N,G", "Element": "Fe", "E_min": 5e3, "E_max": 5e3},
    ])


@pytest.fixture
def tiny_db(tmp_path):
    """A complete stage-1 + stage-2 output, small enough to reason about exactly."""
    path = tmp_path / "nugrade_data.db"
    con = sqlite3.connect(path)
    try:
        make_measurements().to_sql("measurements", con, index=False)
        make_reports().to_sql("report_embeddings", con, index=False)
        make_entries().to_sql("entries", con, index=False)
        make_subentries().to_sql("subentries", con, index=False)
        con.commit()
    finally:
        con.close()
    return path


@pytest.fixture
def tiny_config(tiny_db):
    from pipeline_config import Config

    return Config.resolved(output_dir=tiny_db.parent, db_path=tiny_db, log_every=1000)


def table_columns(db_path, table):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
    finally:
        con.close()


def read_table(db_path, table):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return pd.read_sql(f"SELECT * FROM {table}", con)
    finally:
        con.close()
