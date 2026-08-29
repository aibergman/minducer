"""Machine-readable provenance for exported induced-exchange analyses.

The provenance object is intentionally descriptive rather than a claim about
the physical origin of the input ``Jij``. In particular, the default ``K = J``
choice is recorded as an approximation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


HAMILTONIAN_CONVENTION = "UppASD ordered-pair scalar Heisenberg"
HAMILTONIAN_FORMULA = "H = -sum_{i!=j} Jij e_i·e_j"
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
    g_factor: float = 2.0,
) -> dict[str, Any]:
    """Build the canonical ``analysis_provenance`` export object."""

    response_mode = "not_applicable" if mode is None else str(mode)
    if x_source is None:
        x_source = "not_applicable" if mode is None else "unspecified"
    if mode == "j_weighted":
        kernel_source = K_APPROXIMATION
        kernel_definition = "K_ab(R) = literal UppASD input Jij_ab(R), used as a J-weighted induced-response approximation"
        x_definition = "X_nu is defined by p_nu = X_nu sum_a K_nu,a e_a; inferred from p_nu^0/(sum_a K_nu,a e_a^0)"
        x_units = "1 / energy"
    elif mode == "historical":
        kernel_source = "historical unweighted neighbour sum (no K kernel)"
        kernel_definition = "unweighted selected-neighbour orientation sum"
        x_definition = "X scales the historical unweighted neighbour sum"
        x_units = "dimensionless in historical mode"
    else:
        kernel_source = "not_applicable"
        kernel_definition = "not_applicable"
        x_definition = "not_applicable"
        x_units = "not_applicable"
    return {
        "schema_version": "2",
        "convention_version": "IMX-09",
        "hamiltonian_convention": HAMILTONIAN_CONVENTION,
        "hamiltonian_formula": HAMILTONIAN_FORMULA,
        "pair_counting": "ordered",
        "jij_semantics": "literal UppASD jfile value",
        "units": dict(units),
        "g_factor": float(g_factor),
        "magnon_prefactor_convention": "2*g from ordered-pair curvature and gyromagnetic ratio",
        "response_mode": response_mode,
        "K_source": kernel_source,
        "response_kernel_definition": kernel_definition,
        "response_kernel_source": "input Jij" if mode == "j_weighted" else kernel_source,
        "induced_variable_definition": "robust r=e (dimensionless orientation); induced p=m/|m^0| (dimensionless normalized polarization)",
        "X_definition": x_definition,
        "X_units": x_units,
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


__all__ = ["HAMILTONIAN_CONVENTION", "HAMILTONIAN_FORMULA", "K_APPROXIMATION", "build_analysis_provenance"]
