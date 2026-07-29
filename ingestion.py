"""Per-channel metric computation for 1_raw_data_ingestion.ipynb.

Extracted from the notebook so the arithmetic can be unit-tested. Everything here is a
pure function of its arguments: the notebook version read `Z`, `A`, `symbol`, `proj` and
`reaction` from enclosing-loop globals, which made it impossible to test and easy to
break by reordering cells.

Four bugs from the inline version are fixed here and pinned by tests in test_ingestion.py:

  1. Chi-squared was ``(observed - expected)**2 / sigma`` — dividing by the standard
     deviation instead of the variance. It is now ``((observed - expected) / sigma)**2``,
     which is dimensionless as a chi-squared must be.
  2. The per-channel assumed uncertainty was gated on ``len(channel_data.dropna()) > 0``,
     which drops rows with a NaN in *any* column. ``dEnergy`` is NULL for 86% of EXFOR
     rows, so 44% of reaction channels silently fell back to the global uncertainty
     instead of using their own distribution. It now looks only at ``dData``/``Data``.
  3. Uncertainties derived from a cross section did not take the magnitude, so the ~52k
     genuinely negative EXFOR cross sections produced negative uncertainties.
  4. A division by zero where the evaluation cross section is zero produced infinities in
     the relative-error column; those are now NaN.
"""
import numpy as np
import pandas as pd

# Relative uncertainties above this are treated as data-entry errors rather than real
# 1000%+ measurements, and are excluded from the per-channel quantile.
MAX_PLAUSIBLE_RELATIVE_UNCERTAINTY = 10.0


def relative_uncertainties(data, d_data):
    """Return the finite, well-defined relative uncertainties |dData / Data|.

    Rows where either value is missing, or where the cross section is zero (making the
    ratio undefined), are dropped rather than propagated as inf/NaN.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.abs(pd.Series(d_data, dtype="float64").to_numpy()) / np.abs(
            pd.Series(data, dtype="float64").to_numpy()
        )
    ratio = pd.Series(ratio, index=pd.Series(data).index)
    return ratio.replace([np.inf, -np.inf], np.nan).dropna()


def assumed_relative_uncertainty(data, d_data, fallback, quantile=0.90):
    """The relative uncertainty to assume for points in this channel that lack one.

    Uses this channel's own ``quantile`` of observed relative uncertainties, capped at
    MAX_PLAUSIBLE_RELATIVE_UNCERTAINTY. Falls back to ``fallback`` (the corpus-wide
    value) only when the channel genuinely has no usable uncertainty at all.

    The gate is on dData/Data specifically. Gating on a whole-row dropna() would consult
    unrelated columns such as dEnergy, which is absent for most EXFOR rows.
    """
    observed = relative_uncertainties(data, d_data)
    if len(observed) == 0:
        return fallback
    return float(np.minimum(
        np.quantile(observed, quantile), MAX_PLAUSIBLE_RELATIVE_UNCERTAINTY
    ))


def fill_assumed_uncertainty(data, d_data, assumed_fraction):
    """Fill missing uncertainties with ``assumed_fraction`` of the cross section magnitude.

    The magnitude matters: EXFOR contains negative cross sections (background-subtraction
    artifacts) and an uncertainty is a width, never negative.
    """
    data = pd.Series(data, dtype="float64")
    d_data = pd.Series(d_data, dtype="float64")
    return d_data.fillna(np.abs(data) * assumed_fraction).abs()


def chi_squared(observed, expected, sigma):
    """Per-point chi-squared: ((observed - expected) / sigma)**2.

    Divides by the variance, not the standard deviation, so the result is dimensionless.
    Returns NaN where sigma is zero or missing rather than inf.
    """
    observed = pd.Series(observed, dtype="float64")
    expected = pd.Series(expected, dtype="float64")
    sigma = pd.Series(sigma, dtype="float64")

    safe_sigma = sigma.where(sigma > 0)
    return ((observed - expected) / safe_sigma) ** 2


def relative_error_percent(observed, expected):
    """Percent difference of ``observed`` from ``expected``.

    NaN where the evaluation is zero or missing, rather than inf.
    """
    observed = pd.Series(observed, dtype="float64")
    expected = pd.Series(expected, dtype="float64")

    safe_expected = expected.where(expected != 0)
    return (observed - safe_expected) / safe_expected * 100.0


def compute_channel_metrics(channel_data, evaluation, interpolated_xs, fallback_uncertainty):
    """Attach assumed uncertainties and evaluation-comparison metrics to one channel.

    Parameters
    ----------
    channel_data : pd.DataFrame
        Measurements for a single (Z, A, reaction) channel. Needs ``Data`` and ``dData``.
    evaluation : str
        Evaluation name, e.g. ``"endf8"``; used as the column-name prefix.
    interpolated_xs : array-like or None
        Evaluation cross section interpolated onto this channel's energies, or None when
        the evaluation has no data for this nuclide/reaction.
    fallback_uncertainty : float
        Corpus-wide relative uncertainty to use when the channel has none of its own.

    Returns
    -------
    pd.DataFrame
        A copy of ``channel_data`` with ``dData_assumed``, ``<evaluation>``,
        ``<evaluation>_chi_squared`` and ``<evaluation>_relative_error`` columns.
    """
    result = channel_data.copy()

    assumed_fraction = assumed_relative_uncertainty(
        result["Data"], result["dData"], fallback_uncertainty
    )
    result["dData_assumed"] = fill_assumed_uncertainty(
        result["Data"], result["dData"], assumed_fraction
    )

    if interpolated_xs is None:
        result[evaluation] = np.nan
        result[f"{evaluation}_chi_squared"] = np.nan
        result[f"{evaluation}_relative_error"] = np.nan
        return result

    result[evaluation] = np.asarray(interpolated_xs, dtype="float64")
    result[f"{evaluation}_chi_squared"] = chi_squared(
        result["Data"], result[evaluation], result["dData_assumed"]
    )
    result[f"{evaluation}_relative_error"] = relative_error_percent(
        result["Data"], result[evaluation]
    )
    return result
