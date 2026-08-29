"""Native UppASD convention helpers.

The public input convention is the scalar-Heisenberg convention used by an
UppASD ``jfile``::

    H = -sum_(i != j) Jij e_i . e_j

The exchange rows are an ordered list.  In particular, a pair-complete file
contains both directed rows and neither the parser nor these helpers applies a
hidden factor of two.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

from .model import MagneticCrystal


UPPASD_HAMILTONIAN_CONVENTION = "UppASD ordered-pair scalar Heisenberg"
UPPASD_HAMILTONIAN_FORMULA = "H = -sum_{i!=j} Jij e_i·e_j"


def _spin_array(model: MagneticCrystal, spin_directions: Any) -> tuple[tuple[int, ...], np.ndarray]:
    """Normalize spin directions to sorted model-site order."""

    sites = tuple(sorted(model.site_indices))
    if isinstance(spin_directions, Mapping):
        try:
            values = [spin_directions[site] for site in sites]
        except KeyError as exc:
            raise ValueError(f"spin_directions mapping is missing site {exc.args[0]}") from exc
    else:
        values = spin_directions
    array = np.asarray(values, dtype=float)
    if array.shape != (len(sites), 3) or not np.isfinite(array).all():
        raise ValueError(f"spin_directions must have shape ({len(sites)}, 3) and contain finite values")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 1e-14):
        raise ValueError("spin_directions cannot contain a zero vector")
    # The native Hamiltonian is written for unit directions.  Normalising at
    # this boundary keeps the helper correct for callers that provide
    # arbitrary nonzero representatives of the same directions.
    return sites, array / norms[:, None]


def uppasd_exchange_energy(model: MagneticCrystal, spin_directions: Any) -> float:
    """Return ``-sum_(i != j) Jij e_i dot e_j`` for the supplied directions.

    The input array follows sorted model site order.  Rows are accumulated
    exactly as supplied, including both rows of a reciprocal pair.
    """

    sites, vectors = _spin_array(model, spin_directions)
    position = {site: index for index, site in enumerate(sites)}
    energy = 0.0
    for bond in model.exchange_bonds:
        try:
            energy -= float(bond.jij) * float(np.dot(vectors[position[bond.i]], vectors[position[bond.j]]))
        except KeyError as exc:  # pragma: no cover - model validation normally catches this
            raise ValueError(f"exchange bond references unknown site {exc.args[0]}") from exc
    return float(energy)


# Short names make the convention discoverable from notebooks and tests.
exchange_energy = uppasd_exchange_energy
hamiltonian_energy = uppasd_exchange_energy


def local_exchange_field(
    model: MagneticCrystal,
    spin_directions: Any,
    *,
    moment_magnitudes: Sequence[float] | Mapping[int, float] | None = None,
) -> np.ndarray:
    """Return the exchange field coefficient implied by the ordered sum.

    Without ``moment_magnitudes`` the result is the energy-valued coefficient
    ``2 sum_j Jij e_j`` for a symmetric pair-complete input.  With moments in
    ``mu_B`` it is divided by each ``m_i`` and therefore has the usual
    energy-per-``mu_B`` field units.  For an asymmetric input the exact
    derivative is used: outgoing and incoming rows are both included.
    """

    sites, vectors = _spin_array(model, spin_directions)
    position = {site: index for index, site in enumerate(sites)}
    result = np.zeros_like(vectors, dtype=float)
    for bond in model.exchange_bonds:
        i = position[bond.i]
        j = position[bond.j]
        result[i] += float(bond.jij) * vectors[j]
        result[j] += float(bond.jij) * vectors[i]
    if moment_magnitudes is not None:
        if isinstance(moment_magnitudes, Mapping):
            try:
                moments = np.asarray([moment_magnitudes[site] for site in sites], dtype=float)
            except KeyError as exc:
                raise ValueError(f"moment_magnitudes mapping is missing site {exc.args[0]}") from exc
        else:
            moments = np.asarray(moment_magnitudes, dtype=float)
        if moments.shape != (len(sites),) or not np.isfinite(moments).all() or np.any(np.abs(moments) <= 1e-14):
            raise ValueError("moment_magnitudes must be finite, nonzero, and match the model sites")
        result = result / moments[:, None]
    return result


exchange_field = local_exchange_field


def mft_curie_energy(j_gamma: float | np.ndarray) -> float | np.ndarray:
    """Return the native-convention mean-field ``k_B T_C = 2 J(0)/3``.

    A scalar is returned as a scalar; array input is useful for independent
    scalar FM fixtures and is returned elementwise.
    """

    value = np.asarray(j_gamma, dtype=float)
    if not np.isfinite(value).all():
        raise ValueError("J(0) must be finite")
    result = (2.0 / 3.0) * value
    return float(result) if result.ndim == 0 else result


mean_field_curie_temperature = mft_curie_energy
mean_field_curie_energy = mft_curie_energy


_CONVERSION_FACTORS = {
    "uppasd": 1.0,
    "uppasd_ordered": 1.0,
    "ordered": 1.0,
    "single_counted": 0.5,
    "single-counted": 0.5,
    "pair_counted_once": 0.5,
    "half_ordered": 0.5,
    "half-ordered": 0.5,
    "negative_half_ordered": 0.5,
    "af_positive": -1.0,
    "af-positive": -1.0,
    "antiferromagnetic_positive": -1.0,
}


def convert_exchange_to_uppasd(
    jij: Any,
    *,
    source_convention: Literal[
        "uppasd", "uppasd_ordered", "ordered", "single_counted", "single-counted",
        "pair_counted_once", "half_ordered", "half-ordered", "negative_half_ordered",
        "af_positive", "af-positive", "antiferromagnetic_positive",
    ] = "uppasd",
) -> np.ndarray | float:
    """Convert common exchange conventions to literal UppASD ``Jij`` values.

    ``single_counted`` means ``H=-sum_<ij> J' e_i.e_j`` and
    ``half_ordered`` means ``H=-1/2 sum_(i!=j) J'' e_i.e_j``.  The AF-positive
    option is the same ordered counting with the opposite sign convention.
    """

    key = str(source_convention).lower().replace(" ", "_")
    if key not in _CONVERSION_FACTORS:
        choices = ", ".join(sorted(_CONVERSION_FACTORS))
        raise ValueError(f"unknown exchange convention {source_convention!r}; choose one of {choices}")
    values = np.asarray(jij, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("exchange values must be finite")
    result = values * _CONVERSION_FACTORS[key]
    return float(result) if result.ndim == 0 else result


to_uppasd_jij = convert_exchange_to_uppasd


__all__ = [
    "UPPASD_HAMILTONIAN_CONVENTION",
    "UPPASD_HAMILTONIAN_FORMULA",
    "convert_exchange_to_uppasd",
    "exchange_energy",
    "exchange_field",
    "hamiltonian_energy",
    "local_exchange_field",
    "mean_field_curie_energy",
    "mean_field_curie_temperature",
    "mft_curie_energy",
    "to_uppasd_jij",
    "uppasd_exchange_energy",
]
