"""Tests for the extracted ingestion metric computation.

Each class pins one of the bugs the inline notebook version had. Run with:
    pytest test_ingestion.py
"""
import numpy as np
import pandas as pd
import pytest

from ingestion import (
    MAX_PLAUSIBLE_RELATIVE_UNCERTAINTY,
    assumed_relative_uncertainty,
    chi_squared,
    compute_channel_metrics,
    fill_assumed_uncertainty,
    relative_error_percent,
    relative_uncertainties,
)


class TestChiSquared:
    def test_divides_by_variance_not_standard_deviation(self):
        """Bug 1: the notebook computed (o - e)^2 / sigma, not ((o - e) / sigma)^2.

        With a 2-barn discrepancy and sigma = 0.5, the correct chi-squared is
        (2 / 0.5)^2 = 16. The old formula gave 4 / 0.5 = 8.
        """
        result = chi_squared([5.0], [3.0], [0.5])

        assert result.iloc[0] == pytest.approx(16.0)

    def test_is_dimensionless_under_unit_rescaling(self):
        """A real chi-squared is invariant if data, evaluation and sigma all rescale."""
        base = chi_squared([5.0], [3.0], [0.5])
        scaled = chi_squared([5000.0], [3000.0], [500.0])

        assert base.iloc[0] == pytest.approx(scaled.iloc[0])

    def test_perfect_agreement_is_zero(self):
        assert chi_squared([3.0], [3.0], [0.5]).iloc[0] == pytest.approx(0.0)

    def test_zero_sigma_gives_nan_not_infinity(self):
        """Bug 4: dividing by sigma = 0 previously produced inf, which poisons means."""
        result = chi_squared([5.0, 5.0], [3.0, 3.0], [0.0, 0.5])

        assert np.isnan(result.iloc[0])
        assert np.isfinite(result.iloc[1])

    def test_is_never_negative(self):
        result = chi_squared([1.0, -4.0, 7.0], [3.0, 2.0, 2.0], [0.5, 1.0, 2.0])

        assert (result.dropna() >= 0).all()


class TestAssumedRelativeUncertainty:
    def test_uses_channel_data_despite_nan_in_other_columns(self):
        """Bug 2: gating on a whole-row dropna() consulted dEnergy, which is NULL for
        86% of EXFOR rows, pushing 44% of channels onto the global fallback."""
        channel = pd.DataFrame({
            "Energy":  [1.0, 2.0, 3.0],
            "dEnergy": [np.nan, np.nan, np.nan],   # absent, as usual
            "Data":    [1.0, 1.0, 1.0],
            "dData":   [0.1, 0.2, 0.3],            # present and usable
        })
        assert len(channel.dropna()) == 0, "precondition: whole-row dropna is empty"

        result = assumed_relative_uncertainty(
            channel["Data"], channel["dData"], fallback=0.99
        )

        assert result != pytest.approx(0.99), "must not fall back to the global value"
        assert result == pytest.approx(np.quantile([0.1, 0.2, 0.3], 0.90))

    def test_falls_back_only_when_no_uncertainty_exists(self):
        result = assumed_relative_uncertainty([1.0, 2.0], [np.nan, np.nan], fallback=0.42)

        assert result == pytest.approx(0.42)

    def test_caps_implausible_values(self):
        """A 5000% relative uncertainty is a data-entry error, not a measurement."""
        result = assumed_relative_uncertainty([1.0, 1.0], [50.0, 60.0], fallback=0.1)

        assert result == pytest.approx(MAX_PLAUSIBLE_RELATIVE_UNCERTAINTY)

    def test_zero_cross_section_does_not_produce_infinity(self):
        result = assumed_relative_uncertainty([0.0, 1.0], [0.1, 0.2], fallback=0.5)

        assert np.isfinite(result)
        assert result == pytest.approx(0.2)


class TestFillAssumedUncertainty:
    def test_negative_cross_section_gives_positive_uncertainty(self):
        """Bug 3: ~52k EXFOR rows have negative cross sections."""
        result = fill_assumed_uncertainty([-2.0, 2.0], [np.nan, np.nan], 0.25)

        assert (result > 0).all()
        assert result.tolist() == pytest.approx([0.5, 0.5])

    def test_existing_uncertainties_are_preserved(self):
        result = fill_assumed_uncertainty([10.0, 10.0], [0.7, np.nan], 0.25)

        assert result.iloc[0] == pytest.approx(0.7)
        assert result.iloc[1] == pytest.approx(2.5)


class TestRelativeError:
    def test_zero_evaluation_gives_nan_not_infinity(self):
        result = relative_error_percent([1.0, 1.0], [0.0, 2.0])

        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx(-50.0)

    def test_sign_indicates_direction(self):
        assert relative_error_percent([3.0], [2.0]).iloc[0] == pytest.approx(50.0)


class TestRelativeUncertainties:
    def test_drops_undefined_ratios(self):
        result = relative_uncertainties([1.0, 0.0, 2.0], [0.5, 0.5, np.nan])

        assert len(result) == 1
        assert result.iloc[0] == pytest.approx(0.5)


class TestComputeChannelMetrics:
    def _channel(self):
        return pd.DataFrame({
            "Energy":  [1.0, 2.0, 3.0],
            "dEnergy": [np.nan] * 3,
            "Data":    [1.0, 2.0, -3.0],   # includes a negative cross section
            "dData":   [0.1, np.nan, 0.3],
        })

    def test_produces_all_expected_columns(self):
        result = compute_channel_metrics(
            self._channel(), "endf8", [1.1, 2.2, -2.7], fallback_uncertainty=0.2
        )

        for column in ["dData_assumed", "endf8", "endf8_chi_squared", "endf8_relative_error"]:
            assert column in result.columns

    def test_derived_uncertainties_and_chi_squared_are_non_negative(self):
        result = compute_channel_metrics(
            self._channel(), "endf8", [1.1, 2.2, -2.7], fallback_uncertainty=0.2
        )

        assert (result["dData_assumed"] >= 0).all()
        assert (result["endf8_chi_squared"].dropna() >= 0).all()

    def test_missing_evaluation_yields_nan_columns(self):
        result = compute_channel_metrics(
            self._channel(), "endf8", None, fallback_uncertainty=0.2
        )

        assert result["endf8"].isna().all()
        assert result["endf8_chi_squared"].isna().all()
        # The assumed uncertainty is still computed — it does not depend on the evaluation.
        assert result["dData_assumed"].notna().all()

    def test_does_not_mutate_its_input(self):
        channel = self._channel()
        compute_channel_metrics(channel, "endf8", [1.0, 2.0, 3.0], fallback_uncertainty=0.2)

        assert "dData_assumed" not in channel.columns
