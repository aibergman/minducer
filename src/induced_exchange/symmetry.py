"""Optional space-group expansion for symmetry-reduced exchange input.

The expansion uses the crystallographic symmetry returned by :mod:`spglib`.
Each input bond is treated as one representative of its orbit.  The complete
Cartesian bond displacement is rotated in fractional coordinates and the
source/target basis sites are mapped under the same operation.  Conflicting
values for symmetry-equivalent bonds are rejected; this routine never averages
or silently symmetrizes malformed exchange data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .model import ExchangeBond, MagneticCrystal


class SymmetryExpansionError(ValueError):
    """Raised when symmetry expansion cannot be performed safely."""


@dataclass(frozen=True)
class SymmetryExpansionReport:
    """Diagnostics describing one symmetry expansion."""

    input_bonds: int
    output_bonds: int
    generated_bonds: int
    symmetry_operations: int
    symprec: float
    conflicts: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return self.output_bonds != self.input_bonds

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_bonds": self.input_bonds,
            "output_bonds": self.output_bonds,
            "generated_bonds": self.generated_bonds,
            "symmetry_operations": self.symmetry_operations,
            "symprec": self.symprec,
            "changed": self.changed,
            "conflicts": list(self.conflicts),
        }


@dataclass(frozen=True)
class SymmetryExpansionResult:
    """Expanded model and the diagnostics that produced it."""

    model: MagneticCrystal
    report: SymmetryExpansionReport


def _load_spglib() -> Any:
    try:
        import spglib
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise SymmetryExpansionError(
            "symmetry expansion requires spglib; install the optional 'symmetry' dependency"
        ) from exc
    return spglib


def _site_mapping(
    fractional_positions: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    *,
    tolerance: float,
    site_types: np.ndarray | None = None,
) -> np.ndarray:
    mapping = np.empty(len(fractional_positions), dtype=int)
    for site_index, position in enumerate(fractional_positions):
        transformed = position @ rotations.T + translations
        wrapped = transformed - np.floor(transformed)
        differences = wrapped[None, :] - fractional_positions
        differences -= np.rint(differences)
        distances = np.max(np.abs(differences), axis=1)
        if site_types is not None:
            distances = np.where(site_types == site_types[site_index], distances, np.inf)
        match = int(np.argmin(distances))
        if distances[match] > tolerance:
            raise SymmetryExpansionError(
                f"spglib symmetry operation does not map basis site {site_index} onto a supplied basis site"
            )
        mapping[site_index] = match
    return mapping


def expand_exchange_symmetry(
    model: MagneticCrystal,
    *,
    symprec: float = 1e-5,
    bond_tolerance: float = 1e-8,
    value_rtol: float = 1e-8,
    value_atol: float = 1e-12,
) -> SymmetryExpansionResult:
    """Expand representative exchange bonds using the crystal space group.

    ``model.cell`` contains Cartesian direct vectors as rows and site
    positions are Cartesian, matching the rest of the package.  ``spglib``
    receives the corresponding fractional basis positions.  The input bond
    displacement is authoritative and is transformed as a complete pair
    displacement; it is not reconstructed from basis positions.

    Every symmetry-equivalent bond must have the same Jij within
    ``value_atol + value_rtol * abs(Jij)``.  A conflict raises
    :class:`SymmetryExpansionError` so an inconsistent input cannot be hidden.
    """

    if not isinstance(model, MagneticCrystal):
        raise TypeError("model must be a MagneticCrystal")
    if not np.isfinite(symprec) or symprec <= 0:
        raise ValueError("symprec must be finite and positive")
    if not np.isfinite(bond_tolerance) or bond_tolerance <= 0:
        raise ValueError("bond_tolerance must be finite and positive")
    if not np.isfinite(value_rtol) or value_rtol < 0 or not np.isfinite(value_atol) or value_atol < 0:
        raise ValueError("value tolerances must be finite and non-negative")

    spglib = _load_spglib()
    sites = sorted(model.sites, key=lambda site: site.index)
    if not sites:
        raise SymmetryExpansionError("cannot expand exchange symmetry without basis sites")
    inverse_cell = np.linalg.inv(model.cell)
    fractional_positions = np.asarray([site.position for site in sites], dtype=float) @ inverse_cell
    # MagneticSite.atom_type is deliberately sourced from posfile by the
    # canonical loader.  The momfile type token is not used here: moment
    # magnitudes must never merge or exchange crystallographic species.
    numbers_by_species: dict[Any, int] = {}
    numbers: list[int] = []
    for site in sites:
        if site.atom_type not in numbers_by_species:
            numbers_by_species[site.atom_type] = len(numbers_by_species) + 1
        numbers.append(numbers_by_species[site.atom_type])
    geometry_scale = float(np.max(np.linalg.norm(model.cell, axis=1), initial=0.0))
    if geometry_scale <= 0.0:
        raise SymmetryExpansionError("cannot expand exchange symmetry for a zero-length cell")
    # spglib's symprec is interpreted in Cartesian units. Normalize the
    # crystallographic input so the default tolerance remains meaningful for
    # cells expressed in metres (for example through inpsd.dat's ``alat``).
    symmetry_cell = np.asarray(model.cell, dtype=float) / geometry_scale
    symmetry = spglib.get_symmetry(
        (symmetry_cell, fractional_positions, np.asarray(numbers, dtype=int)),
        symprec=float(symprec),
    )
    if symmetry is None:
        raise SymmetryExpansionError("spglib could not determine crystal symmetry for the supplied cell and basis")

    rotations = np.asarray(symmetry["rotations"], dtype=int)
    translations = np.asarray(symmetry["translations"], dtype=float)
    site_index_position = {site.index: position for position, site in enumerate(sites)}
    operation_mappings = [
        _site_mapping(
            fractional_positions,
            rotation,
            translation,
            tolerance=max(float(symprec), 1e-7),
            site_types=np.asarray(numbers, dtype=int),
        )
        for rotation, translation in zip(rotations, translations)
    ]
    bond_decimals = max(0, int(np.ceil(-np.log10(float(bond_tolerance)))))
    bond_by_key: dict[tuple[int, int, tuple[float, float, float]], ExchangeBond] = {}
    conflicts: list[str] = []

    for bond in model.exchange_bonds:
        try:
            source_position = site_index_position[bond.i]
            target_position = site_index_position[bond.j]
        except KeyError as exc:
            raise SymmetryExpansionError(f"exchange bond references unknown site {exc.args[0]}") from exc
        displacement_fractional = np.asarray(bond.displacement, dtype=float) @ inverse_cell
        for mapping, rotation in zip(operation_mappings, rotations):
            # Quantize in normalized cell-length units.  The model itself may
            # be expressed in metres through ``alat``; rounding raw metre
            # values at 1e-8 would collapse every bond to zero.
            transformed_normalized = displacement_fractional @ rotation.T @ symmetry_cell
            transformed_displacement = transformed_normalized * geometry_scale
            key = (
                sites[mapping[source_position]].index,
                sites[mapping[target_position]].index,
                tuple(float(value) for value in np.round(transformed_normalized, bond_decimals)),
            )
            candidate = ExchangeBond(
                key[0],
                key[1],
                tuple(float(value) for value in transformed_displacement),
                bond.jij,
                None,
                bond.source_line,
            )
            existing = bond_by_key.get(key)
            if existing is None:
                bond_by_key[key] = candidate
            elif not np.isclose(existing.jij, candidate.jij, rtol=value_rtol, atol=value_atol):
                conflicts.append(
                    f"({key[0]}, {key[1]}, {key[2]}) has Jij {existing.jij:g} and {candidate.jij:g}"
                )

    report = SymmetryExpansionReport(
        input_bonds=len(model.exchange_bonds),
        output_bonds=len(bond_by_key),
        generated_bonds=max(0, len(bond_by_key) - len(model.exchange_bonds)),
        symmetry_operations=len(rotations),
        symprec=float(symprec),
        conflicts=tuple(dict.fromkeys(conflicts)),
    )
    if report.conflicts:
        preview = "; ".join(report.conflicts[:3])
        suffix = "" if len(report.conflicts) <= 3 else f"; ... ({len(report.conflicts)} conflicts total)"
        raise SymmetryExpansionError("symmetry-equivalent exchange values conflict: " + preview + suffix)

    expanded_bonds = sorted(
        bond_by_key.values(),
        key=lambda bond: (bond.i, bond.j, bond.distance, bond.displacement),
    )
    expanded_model = MagneticCrystal(
        cell=model.cell.copy(),
        sites=list(model.sites),
        exchange_bonds=expanded_bonds,
        units=model.units,
        source_files=dict(model.source_files),
    )
    return SymmetryExpansionResult(expanded_model, report)


__all__ = [
    "SymmetryExpansionError",
    "SymmetryExpansionReport",
    "SymmetryExpansionResult",
    "expand_exchange_symmetry",
]
