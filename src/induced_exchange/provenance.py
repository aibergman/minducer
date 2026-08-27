"""Machine-readable provenance for exported induced-exchange analyses.

The provenance object is intentionally descriptive rather than a versioned
claim about the physical origin of the input ``Jij``. In particular, the
default ``K = J`` choice is recorded as an approximation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


HAMILTONIAN_CONVENTION = "H = -1/2 sum_ij J_ij e_i dot e_j"
K_APPROXIMATION = "input Jij (J-weighted induced-response approximation)"


def _array(values: Any) -> list[Any]:
    return np.asarray(values).tolist()


def build_analysis_provenance(
    *,
    units: Mapping[str, Any],
    robust_sites: Sequence[int],
    induced_sites: Sequence[int],
    q_fractional: Any,
    q_cartesian: Any,
    mode: str | None = None,
    x_source: str | None = None,
    x_values: Any = None,
    numerical_tolerances: Mapping[str, Any] | None = None,
    q_coordinates: str = "fractional",
) -> dict[str, Any]:
    """Build the canonical ``analysis_provenance`` export object."""

    response_mode = "not_applicable" if mode is None else str(mode)
    if x_source is None:
        x_source = "not_applicable" if mode is None else "unspecified"
    if mode == "j_weighted":
        kernel_source = K_APPROXIMATION
    elif mode == "historical":
        kernel_source = "historical unweighted neighbour sum (no K kernel)"
    else:
        kernel_source = "not_applicable"
    return {
        "schema_version": "1",
        "hamiltonian_convention": HAMILTONIAN_CONVENTION,
        "units": dict(units),
        "response_mode": response_mode,
        "K_source": kernel_source,
        "X_source": x_source,
        "X_values": None if x_values is None else _array(x_values),
        "classification": {
            "robust_sites": [int(site) for site in robust_sites],
            "induced_sites": [int(site) for site in induced_sites],
        },
        "q_mesh": {
            "coordinates": q_coordinates,
            "q_fractional": _array(q_fractional),
            "q_cartesian": _array(q_cartesian),
        },
        "numerical_tolerances": dict(numerical_tolerances or {}),
    }


__all__ = ["HAMILTONIAN_CONVENTION", "K_APPROXIMATION", "build_analysis_provenance"]
