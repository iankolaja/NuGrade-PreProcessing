"""KNN uncertainty imputation, extracted from 3_knn_imputation.ipynb so it can be tested.

For each measurement missing an experimental uncertainty, we find the most similar
complete measurements from other experiments (by report-text similarity features plus
standardized Z / A / energy) and adopt an inverse-distance-weighted mean of their
*relative* uncertainties, scaled by the measurement's own cross section.

The functions here are pure (no database, no notebook globals) so the imputation math
can be unit-tested with synthetic data. Three bugs from the original inline notebook
version are fixed here and pinned by tests in test_imputation.py:

  1. The adopted absolute uncertainty was scaled by a hardcoded row (`.loc[192948]`)
     instead of the row being imputed.
  2. The composite distance applied the square root per term and then summed
     (sqrt(w*d^2) summed = a weighted Manhattan distance), rather than summing the
     weighted squares and taking one root (the documented Euclidean distance).
  3. Row selection mixed positional (.iloc) and label (.loc) indexing, which is only
     safe while the frame's index is gap-free.
"""
import numpy as np
import pandas as pd

SIMILARITY_LABELS = [
    "background_treatment_max_sim", "detector_efficiency_max_sim",
    "normalization_treatment_max_sim", "sample_attenuation_scattering_corrections_max_sim",
    "dead_time_max_sim", "coincidence_max_sim", "statistical_uncertainty_max_sim",
    "uncertainty_propagation_max_sim", "time_of_flight_max_sim",
]

# Features that enter the composite distance, in addition to the similarity labels and
# the mean report embedding (which is compared by cosine distance, not squared diff).
SCALAR_FEATURES = ["Energy_Logstd", "Z_std", "A_std"]


def standardize_features(measurement_df):
    """Add Energy_Logstd, Z_std, A_std as corpus-standardized columns (z-scores).

    Energy is standardized in log10 space because cross-section energies span decades.
    Returns a new frame; the input is not mutated.
    """
    df = measurement_df.copy()
    log_energy = np.log10(df["Energy"])
    df["Energy_Logstd"] = (log_energy - log_energy.mean()) / log_energy.std()
    df["Z_std"] = (df["Z"] - df["Z"].mean()) / df["Z"].std()
    df["A_std"] = (df["A"] - df["A"].mean()) / df["A"].std()
    return df


def composite_distance(query, candidates, weights, sim_labels=SIMILARITY_LABELS):
    """Weighted Euclidean distance from one query row to each candidate row.

    Parameters
    ----------
    query : pd.Series
        Feature values for the point being imputed. Must contain every entry in
        ``sim_labels`` and ``SCALAR_FEATURES``, plus a ``mean_embedding`` vector.
    candidates : pd.DataFrame
        One row per candidate neighbour, same columns as ``query``.
    weights : pd.Series
        Per-feature weight, indexed by feature name (including ``mean_embedding``).

    Returns
    -------
    pd.Series
        Distance to each candidate, indexed like ``candidates``.

    The distance is ``sqrt(sum_f w_f * d_f^2)`` over similarity + scalar features, with
    the mean-embedding contribution taken as its cosine distance (1 - cos sim). The root
    is applied once, to the weighted sum — not per term.
    """
    feature_cols = list(sim_labels) + list(SCALAR_FEATURES)

    # Candidate frames are usually built with pd.concat(rows, axis=1).T, which yields
    # object dtype; arithmetic then produces object columns that numpy ufuncs reject.
    # Coerce explicitly rather than relying on the caller's dtypes.
    numeric_candidates = candidates[feature_cols].apply(pd.to_numeric)

    squared = pd.DataFrame(index=candidates.index)
    for col in feature_cols:
        squared[col] = (pd.to_numeric(query[col]) - numeric_candidates[col]) ** 2

    candidate_embeddings = np.vstack(candidates["mean_embedding"].values).astype(float)
    query_embedding = np.asarray(query["mean_embedding"], dtype=float).reshape(1, -1)
    cos_sim = _cosine_similarity(query_embedding, candidate_embeddings)[0]
    squared["mean_embedding"] = (1.0 - cos_sim) ** 2

    weighted = squared.mul(weights, axis=1)
    return np.sqrt(weighted.sum(axis=1))


def weighted_mean_relative_uncertainty(distances, relative_uncertainties, k_neighbors):
    """Inverse-distance-weighted mean of the k nearest neighbours' relative uncertainties.

    A distance of exactly zero (an identical candidate) would give infinite weight, so it
    is treated as an exact match and returned directly.
    """
    order = distances.sort_values().index[:k_neighbors]
    d = distances.loc[order]
    rel_unc = relative_uncertainties.loc[order]

    exact = d[d == 0]
    if len(exact) > 0:
        return float(rel_unc.loc[exact.index].mean())

    weights = 1.0 / d
    return float(np.average(rel_unc, weights=weights))


def absolute_uncertainty(relative_uncertainty, data):
    """Convert a relative uncertainty to an absolute one for a cross section ``data``.

    EXFOR contains genuinely negative cross sections (background-subtraction artifacts),
    so the magnitude is taken: an uncertainty is a width and is never negative. Without
    this, ~52k negative-cross-section rows produce negative adopted uncertainties, which
    then poison any chi-squared that divides by them.
    """
    return relative_uncertainty * np.abs(data)


def _cosine_similarity(a, b):
    """Rows-of-a against rows-of-b cosine similarity, without a sklearn dependency."""
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


def representative_candidate(candidates, query, inner_weights):
    """Pick the single candidate point closest to the query in (log-energy, Z, A) space.

    One representative point per candidate report avoids letting high-resolution
    experiments (many measurements) dominate the neighbour pool by sheer count.
    """
    d2 = (
        inner_weights["Energy_Logstd"] * (candidates["Energy_Logstd"] - query["Energy_Logstd"]) ** 2
        + inner_weights["Z_std"] * (candidates["Z_std"] - query["Z_std"]) ** 2
        + inner_weights["A_std"] * (candidates["A_std"] - query["A_std"]) ** 2
    )
    return candidates.loc[d2.idxmin()]
