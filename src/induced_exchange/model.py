"""Validated, physics-neutral data model for atomistic exchange input.

This module deliberately stops at the input/model boundary.  It does not infer
chemical elements, exchange conventions, or induced-moment susceptibilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Hashable, Iterable

import numpy as np


@dataclass(frozen=True)
class UnitMetadata:
    """Units attached to input values; no unit conversion is done here."""

    energy: str = "unspecified"
    length: str = "unspecified"
    moment: str = "mu_B"


@dataclass(frozen=True)
class MagneticSite:
    """One basis site with a Cartesian position and reference moment."""

    index: int
    atom_type: Hashable
    position: tuple[float, float, float]
    moment: float | None
    spin_direction: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class ExchangeBond:
    """An isotropic scalar exchange entry from the Jij file.

    ``displacement`` is authoritative.  In particular, it is not rebuilt from
    the basis positions or the cell matrix.
    """

    i: int
    j: int
    displacement: tuple[float, float, float]
    jij: float
    supplied_distance: float | None = None
    source_line: int | None = None

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.displacement))


@dataclass
class ValidationIssue:
    level: str
    code: str
    message: str
    source: str | None = None
    line: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "source": self.source,
            "line": self.line,
        }


@dataclass
class ValidationReport:
    """Structured diagnostics emitted while loading and validating a model."""

    issues: list[ValidationIssue] = field(default_factory=list)
    reciprocal_missing: list[ExchangeBond] = field(default_factory=list)
    reciprocal_mismatched: list[tuple[ExchangeBond, ExchangeBond]] = field(default_factory=list)
    duplicate_bonds: list[ExchangeBond] = field(default_factory=list)
    pair_complete: bool | None = None
    real_space_hermitian: bool | None = None

    def add_warning(self, code: str, message: str, *, source: str | None = None, line: int | None = None) -> None:
        self.issues.append(ValidationIssue("warning", code, message, source, line))

    def add_error(self, code: str, message: str, *, source: str | None = None, line: int | None = None) -> None:
        self.issues.append(ValidationIssue("error", code, message, source, line))

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.level == "warning"]

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.level == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": [issue.as_dict() for issue in self.issues],
            "reciprocal_missing": len(self.reciprocal_missing),
            "reciprocal_mismatched": len(self.reciprocal_mismatched),
            "duplicate_bonds": len(self.duplicate_bonds),
            "pair_complete": self.pair_complete,
            "real_space_hermitian": self.real_space_hermitian,
        }

    to_dict = as_dict


@dataclass
class MagneticCrystal:
    """The complete validated input-level magnetic crystal."""

    cell: np.ndarray
    sites: list[MagneticSite]
    exchange_bonds: list[ExchangeBond]
    units: UnitMetadata = field(default_factory=UnitMetadata)
    source_files: dict[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cell = np.asarray(self.cell, dtype=float)
        if cell.shape != (3, 3):
            raise ValueError(f"cell must have shape (3, 3), got {cell.shape}")
        self.cell = cell

    @property
    def site_indices(self) -> set[int]:
        return {site.index for site in self.sites}

    @property
    def site_by_index(self) -> dict[int, MagneticSite]:
        return {site.index: site for site in self.sites}

    @property
    def bonds(self) -> list[ExchangeBond]:
        """Short alias for callers that use the generic term ``bonds``."""
        return self.exchange_bonds

    @property
    def cell_volume(self) -> float:
        return float(abs(np.linalg.det(self.cell)))

    @property
    def bond_distances(self) -> np.ndarray:
        return np.asarray([bond.distance for bond in self.exchange_bonds], dtype=float)


def _bond_key(bond: ExchangeBond, *, decimals: int = 10) -> tuple[int, int, tuple[float, float, float]]:
    return (bond.i, bond.j, tuple(round(value, decimals) for value in bond.displacement))


def validate_model(model: MagneticCrystal, report: ValidationReport | None = None) -> ValidationReport:
    """Validate model invariants and reciprocal real-space exchange entries."""

    report = report or ValidationReport()
    if not np.isfinite(model.cell).all():
        report.add_error("nonfinite_cell", "cell contains a non-finite value")
    if model.cell_volume <= 1e-14:
        report.add_error("zero_cell_volume", "cell has zero (or numerically zero) volume")

    indices = model.site_indices
    if len(indices) != len(model.sites):
        report.add_error("duplicate_site_index", "basis contains duplicate site indices")
    for site in model.sites:
        if not np.isfinite(site.position).all():
            report.add_error("nonfinite_position", f"site {site.index} has a non-finite position")
        if site.moment is None:
            report.add_error("missing_moment", f"site {site.index} has no moment data")
        elif not np.isfinite(site.moment):
            report.add_error("nonfinite_moment", f"site {site.index} has a non-finite moment")
        if site.spin_direction is not None:
            direction = np.asarray(site.spin_direction)
            if not np.isfinite(direction).all():
                report.add_error("nonfinite_spin", f"site {site.index} has a non-finite spin direction")
            elif np.linalg.norm(direction) <= 1e-14:
                report.add_warning("zero_spin", f"site {site.index} has a zero initial spin direction")

    seen: dict[tuple[int, int, tuple[float, float, float]], ExchangeBond] = {}
    for bond in model.exchange_bonds:
        if bond.i not in indices or bond.j not in indices:
            report.add_error("invalid_site_index", f"exchange bond ({bond.i}, {bond.j}) references an unknown site")
        if not np.isfinite(bond.displacement).all() or not np.isfinite(bond.jij):
            report.add_error("nonfinite_exchange", f"exchange bond on line {bond.source_line or '?'} is non-finite")
        if bond.i == bond.j and bond.distance <= 1e-14:
            report.add_warning("self_interaction", f"zero-displacement self interaction for site {bond.i}", line=bond.source_line)
        key = _bond_key(bond)
        if key in seen:
            report.duplicate_bonds.append(bond)
            report.add_warning("duplicate_exchange", f"duplicate exchange entry for ({bond.i}, {bond.j}, {bond.displacement})", line=bond.source_line)
        else:
            seen[key] = bond

    missing: list[ExchangeBond] = []
    mismatched: list[tuple[ExchangeBond, ExchangeBond]] = []
    for bond in model.exchange_bonds:
        reciprocal_key = (bond.j, bond.i, tuple(round(-value, 10) for value in bond.displacement))
        reciprocal = seen.get(reciprocal_key)
        if reciprocal is None:
            missing.append(bond)
        elif not np.isclose(bond.jij, reciprocal.jij, rtol=1e-8, atol=1e-12):
            mismatched.append((bond, reciprocal))
    report.reciprocal_missing = missing
    report.reciprocal_mismatched = mismatched
    report.pair_complete = not missing
    report.real_space_hermitian = not missing and not mismatched
    if missing:
        report.add_warning("missing_reciprocal", f"{len(missing)} exchange bond(s) have no likely +/-R partner")
    if mismatched:
        report.add_warning("asymmetric_reciprocal", f"{len(mismatched)} reciprocal bond pair(s) have different Jij values")
    return report
