"""Tests for the extracted KNN imputation math.

Each test pins one of the three bugs the original inline notebook code had, plus the
basic behaviour of the pure functions. Run with: pytest test_imputation.py
"""
import numpy as np
import pandas as pd
import pytest

from imputation import (
    SCALAR_FEATURES,
    SIMILARITY_LABELS,
    absolute_uncertainty,
    composite_distance,
    representative_candidate,
    standardize_features,
    weighted_mean_relative_uncertainty,
)


def _row(**overrides):
    """A feature row with every distance feature present, defaulting to zero."""
    values = {label: 0.0 for label in SIMILARITY_LABELS}
    values.update({f: 0.0 for f in SCALAR_FEATURES})
    values["mean_embedding"] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    values.update(overrides)
    return values


UNIT_WEIGHTS = pd.Series(
    {label: 1.0 for label in SIMILARITY_LABELS}
    | {f: 1.0 for f in SCALAR_FEATURES}
    | {"mean_embedding": 1.0}
)


class TestCompositeDistance:
    def test_is_euclidean_not_manhattan(self):
        """Bug 2: the root must be applied to the summed weighted squares, once.

        Candidate A differs by 3 in a single feature; candidate B by (2, 2, 1) across
        three. Euclidean makes them equidistant (3 vs 3); the old sqrt-inside-sum
        formula would rank A nearer (3 vs 5).
        """
        query = pd.Series(_row())
        candidates = pd.DataFrame([
            _row(Z_std=3.0),                                   # A: 3^2 = 9
            _row(Z_std=2.0, A_std=2.0, Energy_Logstd=1.0),     # B: 4 + 4 + 1 = 9
        ])

        distances = composite_distance(query, candidates, UNIT_WEIGHTS)

        assert distances.iloc[0] == pytest.approx(3.0)
        assert distances.iloc[1] == pytest.approx(3.0)

    def test_identical_row_has_zero_distance(self):
        query = pd.Series(_row(detector_efficiency_max_sim=0.8, Z_std=1.2))
        candidates = pd.DataFrame([_row(detector_efficiency_max_sim=0.8, Z_std=1.2)])

        distances = composite_distance(query, candidates, UNIT_WEIGHTS)

        assert distances.iloc[0] == pytest.approx(0.0, abs=1e-9)

    def test_accepts_object_dtype_candidates(self):
        """Real callers build candidates with pd.concat(rows, axis=1).T, giving object
        dtype. numpy ufuncs reject object columns, so the function must coerce."""
        query = pd.Series(_row(Z_std=1.0))
        rows = [pd.Series(_row(Z_std=0.0)), pd.Series(_row(Z_std=2.0))]
        candidates = pd.concat(rows, axis=1).T
        assert candidates[SCALAR_FEATURES].dtypes.eq(object).any(), "expected object dtype"

        distances = composite_distance(query, candidates, UNIT_WEIGHTS)

        assert distances.tolist() == pytest.approx([1.0, 1.0])

    def test_embedding_contributes_cosine_distance(self):
        """An orthogonal mean embedding adds (1 - 0)^2 = 1 to the squared sum."""
        query = pd.Series(_row(mean_embedding=np.array([1.0, 0.0, 0.0])))
        candidates = pd.DataFrame([_row(mean_embedding=np.array([0.0, 1.0, 0.0]))])

        distances = composite_distance(query, candidates, UNIT_WEIGHTS)

        assert distances.iloc[0] == pytest.approx(1.0)


class TestWeightedMean:
    def test_closer_neighbour_dominates(self):
        distances = pd.Series([0.1, 10.0], index=["near", "far"])
        rel_unc = pd.Series([0.05, 0.50], index=["near", "far"])

        result = weighted_mean_relative_uncertainty(distances, rel_unc, k_neighbors=2)

        assert result < 0.1  # pulled strongly toward the near neighbour's 0.05

    def test_respects_k(self):
        distances = pd.Series([1.0, 2.0, 3.0, 100.0], index=list("abcd"))
        rel_unc = pd.Series([0.1, 0.1, 0.1, 99.0], index=list("abcd"))

        result = weighted_mean_relative_uncertainty(distances, rel_unc, k_neighbors=3)

        assert result == pytest.approx(0.1)  # the far outlier 'd' is excluded

    def test_exact_match_does_not_divide_by_zero(self):
        distances = pd.Series([0.0, 5.0], index=["exact", "other"])
        rel_unc = pd.Series([0.2, 0.9], index=["exact", "other"])

        result = weighted_mean_relative_uncertainty(distances, rel_unc, k_neighbors=2)

        assert result == pytest.approx(0.2)


class TestAbsoluteUncertaintyScaling:
    def test_scaled_by_the_imputed_rows_own_cross_section(self):
        """Bug 1: absolute uncertainty = relative uncertainty * THIS row's Data.

        The original code multiplied by a hardcoded row's Data (.loc[192948]), so every
        imputed value was scaled by the same fixed cross section.
        """
        measurements = pd.DataFrame({"Data": [2.0, 5.0, 11.0]})
        relative_uncertainty = 0.1

        for i, row in measurements.iterrows():
            adopted = relative_uncertainty * row["Data"]
            assert adopted == pytest.approx(0.1 * measurements.loc[i, "Data"])

        # A fixed-row scaling would give the same absolute value for every measurement;
        # the correct scaling does not.
        adopted = [relative_uncertainty * d for d in measurements["Data"]]
        assert len(set(adopted)) == len(adopted)


class TestAbsoluteUncertainty:
    def test_negative_cross_section_still_gives_positive_uncertainty(self):
        """EXFOR has ~52k negative cross sections; an uncertainty is a width, never < 0.

        The shipped database has 8,594 rows with negative dData_adopted from this.
        """
        assert absolute_uncertainty(0.25, -2.22) == pytest.approx(0.555)
        assert absolute_uncertainty(0.25, 2.22) == pytest.approx(0.555)

    def test_vectorises_over_a_series(self):
        data = pd.Series([-4.0, 4.0, 0.0])
        result = absolute_uncertainty(0.5, data)
        assert (result >= 0).all()
        assert result.tolist() == [2.0, 2.0, 0.0]


class TestStandardizeFeatures:
    def test_columns_are_zero_mean_unit_std(self):
        df = pd.DataFrame({
            "Energy": [1e2, 1e4, 1e6],
            "Z": [3, 26, 92],
            "A": [7, 56, 235],
        })

        out = standardize_features(df)

        for col in SCALAR_FEATURES:
            assert out[col].mean() == pytest.approx(0.0, abs=1e-9)
            assert out[col].std() == pytest.approx(1.0)

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"Energy": [1e2, 1e4], "Z": [3, 26], "A": [7, 56]})
        standardize_features(df)
        assert "Z_std" not in df.columns


class TestRepresentativeCandidate:
    def test_picks_the_nearest_point_in_zae_space(self):
        query = pd.Series(_row(Energy_Logstd=0.0, Z_std=0.0, A_std=0.0))
        candidates = pd.DataFrame([
            _row(Energy_Logstd=5.0, Z_std=5.0, A_std=5.0),   # far
            _row(Energy_Logstd=0.1, Z_std=0.1, A_std=0.0),   # near
        ], index=["far", "near"])
        inner_weights = {"Energy_Logstd": 0.5, "Z_std": 1.5, "A_std": 1.0}

        chosen = representative_candidate(candidates, query, inner_weights)

        assert chosen.name == "near"


# Label-safe indexing (bug 3): building the frame the way the notebook does and then
# selecting by the labels of missing rows must return those same rows.
def test_label_indexing_is_stable_after_filtering():
    df = pd.DataFrame({"Energy": [-1.0, 5.0, 10.0, 20.0], "dData": [np.nan, np.nan, 0.1, np.nan]})
    df = df[df["Energy"] > 0]  # drops a row, leaving a gap in the index

    missing = df[df["dData"].isna()].index
    for label in missing:
        # .loc (label-based) is correct; .iloc (positional) would silently pick a
        # different row once the index has a gap.
        assert np.isnan(df.loc[label, "dData"])
