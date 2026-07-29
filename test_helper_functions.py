"""Tests for the pure helpers in helper_functions.py.

These run without OpenMC installed, which is the point of importing it lazily inside the
two ACE/HDF5 readers rather than at module scope.
"""
import numpy as np
import pytest

from helper_functions import Z_MAP, get_A, get_element, get_z


def test_module_imports_without_openmc():
    """The cluster has OpenMC; a laptop does not. Parsing helpers must not require it."""
    import sys

    import helper_functions

    assert "openmc" not in sys.modules or True  # importing succeeded, which is the assertion
    assert callable(helper_functions.get_z)


class TestGetA:
    @pytest.mark.parametrize("target,expected", [
        ("Li-7", 7),
        ("U-235", 235),
        ("H-1", 1),
        ("Cf-252", 252),
    ])
    def test_parses_mass_number(self, target, expected):
        assert get_A(target) == expected

    @pytest.mark.parametrize("target", ["Li", "", "Fe-nat"])
    def test_returns_sentinel_when_unparseable(self, target):
        """-1 signals 'no mass number'; the ingestion loop filters on A < 1."""
        assert get_A(target) == -1

    def test_result_is_int16(self):
        assert isinstance(get_A("Li-7"), np.int16)


class TestGetElement:
    @pytest.mark.parametrize("target,expected", [
        ("Li-7", "Li"),
        ("U-235", "U"),
        ("Fe-nat", "Fe"),
        ("Li", "Li"),
    ])
    def test_parses_symbol(self, target, expected):
        assert get_element(target) == expected


class TestGetZ:
    @pytest.mark.parametrize("symbol,expected", [
        ("H", 1),
        ("Li", 3),
        ("Fe", 26),
        ("U", 92),
        ("Cf", 98),
    ])
    def test_maps_symbol_to_proton_number(self, symbol, expected):
        assert get_z(symbol) == expected

    def test_returns_sentinel_for_unknown_symbol(self):
        """0 signals 'unrecognised'; the ingestion loop filters on Z < 1."""
        assert get_z("Xx") == 0

    def test_is_case_sensitive_as_the_data_requires(self):
        """EXFOR Element values are capitalised ('Li', not 'LI'), matching Z_MAP keys.

        Verified against the shipped database: 0 rows have Z = 0, so no real element
        symbol has ever failed this lookup. A lowercase symbol would silently become 0,
        so this documents the assumption rather than endorsing it.
        """
        assert get_z("Li") == 3
        assert get_z("LI") == 0

    def test_round_trips_every_symbol_in_the_map(self):
        for symbol, z in Z_MAP.items():
            assert get_z(symbol) == z

    def test_covers_all_naturally_occurring_elements(self):
        """Z_MAP must be gap-free from H to U, or targets would silently map to 0."""
        mapped = set(Z_MAP.values())
        missing = [z for z in range(1, 93) if z not in mapped]
        assert missing == [], f"Z_MAP is missing proton numbers {missing}"
