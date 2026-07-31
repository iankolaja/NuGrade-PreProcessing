"""Reading evaluated cross sections, behind an interface stage 1 can be tested against.

Stage 1 needs ACE files and OpenMC, both of which exist only on the cluster. Rather than
monkeypatching `helper_functions.extract_XS_openMC_ace` in tests — which is invisible in the
call signature and cannot express "endf8 is present but endf7-1 is not", the exact branch
the reaction summary flags depend on — the reader is an object passed into `run()`.

OpenMC stays lazily imported inside `helper_functions`, so importing this module (or stage 1)
on a laptop pulls in no cluster packages.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional, Protocol

import numpy as np


class EvaluationResult(NamedTuple):
    """Cross sections interpolated onto the query energies, plus a plotting grid."""

    interpolated: np.ndarray
    grid_energy: np.ndarray
    grid_xs: np.ndarray


class EvaluationReader(Protocol):
    """Reads one evaluation library. ``name`` becomes the database column prefix."""

    name: str

    def read(self, symbol: str, zaid: str, mt: int,
             energies: np.ndarray) -> Optional[EvaluationResult]:
        ...


@dataclass
class AceEvaluationReader:
    """Production reader: OpenMC over ACE files on the cluster.

    ``read`` returns None — never raises — when the evaluation has no data for this
    nuclide or reaction. Corpus gaps are expected and must not abort a multi-hour run.
    """

    name: str
    path_template: str
    temp: str = "294K"

    def path_for(self, symbol, zaid):
        """Substitute <symbol> and <ZAID> into the template.

        ACE libraries capitalise the element directory (Li, not li), so the symbol is
        normalised here rather than at each call site.
        """
        symbol = str(symbol)
        capitalised = symbol[:1].upper() + symbol[1:]
        return Path(self.path_template
                    .replace("<symbol>", capitalised)
                    .replace("<ZAID>", str(zaid)))

    def read(self, symbol, zaid, mt, energies):
        from helper_functions import extract_XS_openMC_ace  # openmc imported lazily inside

        path = self.path_for(symbol, zaid)
        if not path.is_file():
            return None
        try:
            interpolated, grid_energy, grid_xs = extract_XS_openMC_ace(
                str(path), mt, energies, temp=self.temp)
        except KeyError:
            return None          # evaluation exists but has no data for this MT
        except AssertionError:
            return None          # OpenMC's internal consistency checks; a corpus gap
        return EvaluationResult(interpolated, grid_energy, grid_xs)


@dataclass
class NullEvaluationReader:
    """Always reports no data. Used by --allow-missing-evals and by tests.

    `ingestion.compute_channel_metrics` already handles ``interpolated_xs=None`` by writing
    NaN metric columns, so this is a supported path rather than a degraded one.
    """

    name: str

    def path_for(self, symbol, zaid):
        return None

    def read(self, symbol, zaid, mt, energies):
        return None


def readers_from_config(config):
    """Build the readers stage 1 will use, in the order their columns are written.

    The only place AceEvaluationReader is constructed, so --allow-missing-evals is honoured
    in exactly one spot.
    """
    from pipeline_config import template_root

    readers = []
    for name, template in config.eval_templates():
        root = template_root(template)
        if config.allow_missing_evals and not root.is_dir():
            readers.append(NullEvaluationReader(name))
        else:
            readers.append(AceEvaluationReader(name, template))
    return readers


# ---------------------------------------------------------------------------
# Test doubles — importable from tests, and deliberately not "mocks"
# ---------------------------------------------------------------------------

@dataclass
class SyntheticEvaluationReader:
    """Analytic 1/sqrt(E) cross section, so metric values are predictable.

    Chi-squared and relative error can then be asserted against arithmetic rather than
    against a recorded blob.
    """

    name: str = "endf8"
    scale: float = 1.0

    def path_for(self, symbol, zaid):
        return Path(f"/synthetic/{symbol}/{zaid}")

    def read(self, symbol, zaid, mt, energies):
        energies = np.asarray(energies, dtype=float)
        interpolated = self.scale / np.sqrt(np.maximum(energies, 1e-12))
        grid_energy = np.logspace(-5, 7, 100)
        return EvaluationResult(interpolated, grid_energy,
                                self.scale / np.sqrt(grid_energy))


@dataclass
class ExplodingEvaluationReader:
    """Raises on every read, to pin that one bad nuclide cannot abort a long run."""

    name: str = "endf8"

    def path_for(self, symbol, zaid):
        return Path("/exploding")

    def read(self, symbol, zaid, mt, energies):
        raise RuntimeError(f"synthetic reader failure for {symbol}-{zaid}")
