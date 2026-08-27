"""Small, explicit unit conversions used by the analysis layers.

The exchange input is otherwise kept in its declared units.  These helpers
only convert an already-known energy unit; they never infer a unit from a
number.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# Energy in meV.  The Rydberg value is the CODATA-compatible conversion used
# by the common mRy convention in atomistic exchange files.
_MEV_PER_UNIT = {
    "mev": 1.0,
    "millielectronvolt": 1.0,
    "millielectronvolts": 1.0,
    "ev": 1000.0,
    "electronvolt": 1000.0,
    "electronvolts": 1000.0,
    "mry": 13.605693009,
    "millirydberg": 13.605693009,
    "millirydbergs": 13.605693009,
    "ry": 13605.693009,
    "rydberg": 13605.693009,
    "rydbergs": 13605.693009,
    "hartree": 27211.386018,
    "ha": 27211.386018,
    "j": 6.241509074e21,
}


def normalise_energy_unit(unit: str) -> str:
    """Return a canonical energy-unit spelling, preserving ``unspecified``."""

    if not isinstance(unit, str):
        raise TypeError("energy unit must be a string")
    value = unit.strip().lower().replace(" ", "")
    if value in {"", "unspecified", "unknown", "input"}:
        return "unspecified"
    if value not in _MEV_PER_UNIT:
        raise ValueError(f"unsupported energy unit {unit!r}")
    aliases = {
        "millielectronvolt": "meV",
        "millielectronvolts": "meV",
        "mev": "meV",
        "electronvolt": "eV",
        "electronvolts": "eV",
        "ev": "eV",
        "mry": "mRy",
        "millirydberg": "mRy",
        "millirydbergs": "mRy",
        "ry": "Ry",
        "rydberg": "Ry",
        "rydbergs": "Ry",
        "hartree": "Ha",
        "ha": "Ha",
        "j": "J",
    }
    return aliases[value]


def energy_conversion_factor(from_unit: str, to_unit: str) -> float:
    """Return the multiplicative factor converting one energy unit to another.

    ``unspecified`` is intentionally rejected.  A caller must either retain
    input units or explicitly confirm the input unit before converting.
    """

    source = normalise_energy_unit(from_unit)
    target = normalise_energy_unit(to_unit)
    if source == "unspecified" or target == "unspecified":
        raise ValueError("energy conversion requires both units to be specified")
    source_mev = _MEV_PER_UNIT[source.lower()]
    target_mev = _MEV_PER_UNIT[target.lower()]
    return float(source_mev / target_mev)


def convert_energy(values: Any, from_unit: str, to_unit: str) -> np.ndarray:
    """Convert scalar or array-like energies without changing their shape."""

    return np.asarray(values) * energy_conversion_factor(from_unit, to_unit)


__all__ = ["convert_energy", "energy_conversion_factor", "normalise_energy_unit"]
