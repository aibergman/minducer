"""UppASD-style input parsing for the IMX-01 input model.

Exchange values are read literally from the jfile.  A pair-complete file is
interpreted with the ordered-pair Hamiltonian ``H=-sum_(i!=j) Jij e_i.e_j``;
the loader never applies a factor-of-two conversion.  Jfile vectors are mapped
through :mod:`induced_exchange.vector_mapping` according to ``maptype``.
"""

from __future__ import annotations

import argparse
import math
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from .model import (
    ExchangeBond,
    MagneticCrystal,
    MagneticSite,
    UnitMetadata,
    ValidationReport,
    validate_model,
)
from .symmetry import SymmetryExpansionError, SymmetryExpansionReport, expand_exchange_symmetry
from .vector_mapping import VectorMappingError, map_exchange_file, prepare_positions


class InputFormatError(ValueError):
    """Raised when a required input row cannot be parsed."""


@dataclass
class InpsdConfig:
    input_file: Path
    cell: np.ndarray
    posfile: Path
    momfile: Path
    exchange: Path
    keywords: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    alat: float | None = None
    posfiletype: str = "C"
    maptype: int = 1
    ncell: tuple[int, int, int] | None = None
    bc: tuple[str, str, str] | None = None


@dataclass
class PositionRecord:
    site: int
    atom_type: object
    position: tuple[float, float, float]
    line: int


@dataclass
class MomentRecord:
    site: int
    moment_field: str
    moment: float
    spin_direction: tuple[float, float, float] | None
    line: int


@dataclass
class LoadedUppASD:
    model: MagneticCrystal
    report: ValidationReport
    config: InpsdConfig
    symmetry_expansion: SymmetryExpansionReport | None = None

    def __getattr__(self, name: str):
        # Keeps the result convenient for interactive/debug use while keeping
        # the report and source configuration available explicitly.
        return getattr(self.model, name)


_KNOWN_KEYWORDS = {
    "simid", "ncell", "bc", "posfile", "positions", "momfile", "moments", "exchange", "jfile",
    "cell", "alat", "posfiletype", "maptype", "hamiltonian", "elements", "do", "temperature", "tstep",
}

# These are the canonical file keywords.  Keep the older descriptive
# spellings as fallbacks so existing input decks continue to load.  The order
# is significant: when both spellings are present, the canonical keyword wins.
_FILE_KEY_ALIASES = {
    "posfile": ("posfile", "positions"),
    "momfile": ("momfile", "moments"),
    "exchange": ("exchange", "jfile"),
}


def _clean_line(line: str) -> str:
    # UppASD files commonly use '#'; accepting '!' as a comment marker is
    # useful for hand-authored files, but only when separated from a token.
    for marker in ("#", "!"):
        position = line.find(marker)
        if position >= 0 and (position == 0 or line[position - 1].isspace()):
            line = line[:position]
    return line.strip()


def _is_float(token: str) -> bool:
    try:
        value = float(token)
    except ValueError:
        return False
    return math.isfinite(value)


def _float(token: str, *, path: Path, line: int, field: str) -> float:
    try:
        value = float(token)
    except ValueError as exc:
        raise InputFormatError(f"{path}:{line}: invalid {field} value {token!r}") from exc
    if not math.isfinite(value):
        raise InputFormatError(f"{path}:{line}: non-finite {field} value {token!r}")
    return value


def _integer(token: str, *, path: Path, line: int, field: str) -> int:
    try:
        value = int(token)
    except ValueError as exc:
        raise InputFormatError(f"{path}:{line}: invalid {field} value {token!r}") from exc
    return value


def _identifier(token: str) -> object:
    try:
        return int(token)
    except ValueError:
        return token


def parse_inpsd(path: str | Path) -> InpsdConfig:
    """Parse the paths and 3x3 cell matrix from an ``inpsd.dat`` file."""

    input_file = Path(path).expanduser().resolve()
    if not input_file.is_file():
        raise FileNotFoundError(input_file)
    lines = input_file.read_text(encoding="utf-8").splitlines()
    base = input_file.parent
    values: dict[str, list[str]] = {}
    warnings: list[str] = []
    cell: np.ndarray | None = None
    alat: float | None = None
    posfiletype = "C"
    maptype = 1
    ncell: tuple[int, int, int] | None = None
    bc: tuple[str, str, str] | None = None
    index = 0
    while index < len(lines):
        line_number = index + 1
        cleaned = _clean_line(lines[index])
        index += 1
        if not cleaned:
            continue
        try:
            tokens = shlex.split(cleaned)
        except ValueError as exc:
            raise InputFormatError(f"{input_file}:{line_number}: {exc}") from exc
        if not tokens:
            continue
        key = tokens[0].lower()
        if key == "alat":
            if len(tokens) != 2:
                raise InputFormatError(f"{input_file}:{line_number}: alat must contain one numeric value in metres")
            alat = _float(tokens[1], path=input_file, line=line_number, field="alat")
            if alat <= 0:
                raise InputFormatError(f"{input_file}:{line_number}: alat must be positive")
            values[key] = tokens[1:]
            continue
        if key == "posfiletype":
            if len(tokens) != 2 or tokens[1].upper() not in {"C", "D"}:
                raise InputFormatError(f"{input_file}:{line_number}: posfiletype must be either C or D")
            posfiletype = tokens[1].upper()
            values[key] = [posfiletype]
            continue
        if key == "maptype":
            if len(tokens) != 2:
                raise InputFormatError(f"{input_file}:{line_number}: maptype must contain one value (1, 2, or 3)")
            maptype = _integer(tokens[1], path=input_file, line=line_number, field="maptype")
            if maptype not in {1, 2, 3}:
                raise InputFormatError(f"{input_file}:{line_number}: maptype must be 1, 2, or 3")
            values[key] = [str(maptype)]
            continue
        if key == "ncell":
            if len(tokens) != 4:
                raise InputFormatError(f"{input_file}:{line_number}: ncell must contain three positive integers")
            ncell_values = tuple(_integer(token, path=input_file, line=line_number, field="ncell") for token in tokens[1:])
            if any(value <= 0 for value in ncell_values):
                raise InputFormatError(f"{input_file}:{line_number}: ncell must contain three positive integers")
            ncell = ncell_values
            values[key] = tokens[1:]
            continue
        if key == "bc":
            if len(tokens) != 4:
                raise InputFormatError(f"{input_file}:{line_number}: BC must contain three values (P or F)")
            aliases = {"P": "P", "PERIODIC": "P", "F": "F", "FREE": "F", "O": "F", "OPEN": "F"}
            normalized_bc = tuple(token.upper() for token in tokens[1:])
            if any(value not in aliases for value in normalized_bc):
                raise InputFormatError(f"{input_file}:{line_number}: BC values must be P or F")
            bc = tuple(aliases[value] for value in normalized_bc)  # type: ignore[assignment]
            values[key] = list(bc)
            continue
        if key == "cell":
            rows: list[list[float]] = []
            # UppASD examples commonly separate cell components with commas.
            # Commas are punctuation here, not part of the numeric value.
            first = [token.strip(",") for token in tokens[1:]]
            if first:
                if len(first) == 9 and all(_is_float(token) for token in first):
                    cell = np.asarray([float(token) for token in first], dtype=float).reshape(3, 3)
                    continue
                if len(first) != 3 or not all(_is_float(token) for token in first):
                    raise InputFormatError(f"{input_file}:{line_number}: cell must contain three numeric values per row")
                rows.append([float(token) for token in first])
            while len(rows) < 3:
                if index >= len(lines):
                    raise InputFormatError(f"{input_file}:{line_number}: incomplete multiline cell")
                continuation_number = index + 1
                continuation = _clean_line(lines[index])
                index += 1
                if not continuation:
                    continue
                continuation_tokens = [token.strip(",") for token in shlex.split(continuation)]
                if len(continuation_tokens) != 3 or not all(_is_float(token) for token in continuation_tokens):
                    raise InputFormatError(f"{input_file}:{continuation_number}: cell continuation must contain three numeric values")
                rows.append([float(token) for token in continuation_tokens])
            cell = np.asarray(rows, dtype=float)
            continue
        values[key] = tokens[1:]
        if key not in _KNOWN_KEYWORDS:
            warnings.append(f"{input_file}:{line_number}: ignored unknown UppASD keyword {tokens[0]!r}")

    if cell is None:
        raise InputFormatError(f"{input_file}: missing cell")

    def required_path(name: str) -> Path:
        raw = next((values[key] for key in _FILE_KEY_ALIASES[name] if values.get(key)), None)
        if not raw:
            raise InputFormatError(f"{input_file}: missing {name} keyword")
        return (base / raw[0]).resolve() if not Path(raw[0]).is_absolute() else Path(raw[0]).resolve()

    return InpsdConfig(
        input_file=input_file,
        cell=cell,
        posfile=required_path("posfile"),
        momfile=required_path("momfile"),
        exchange=required_path("exchange"),
        keywords=values,
        warnings=warnings,
        alat=alat,
        posfiletype=posfiletype,
        maptype=maptype,
        ncell=ncell,
        bc=bc,
    )


def _rows(path: str | Path) -> Iterable[tuple[int, list[str]]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        cleaned = _clean_line(raw)
        if cleaned:
            try:
                yield line_number, shlex.split(cleaned)
            except ValueError as exc:
                raise InputFormatError(f"{source}:{line_number}: {exc}") from exc


def parse_posfile(
    path: str | Path,
    *,
    posfiletype: str = "C",
    cell: np.ndarray | None = None,
) -> dict[int, PositionRecord]:
    """Parse ``site atom_type x y z`` position rows.

    ``C`` interprets the coordinates as Cartesian.  ``D`` interprets them as
    direct (fractional) coordinates and converts them using the direct-cell
    vectors stored as rows of ``cell``.
    """

    source = Path(path).expanduser().resolve()
    normalized_type = str(posfiletype).upper()
    if normalized_type not in {"C", "D"}:
        raise InputFormatError(f"{source}: posfiletype must be either C or D")
    direct_cell = None
    if normalized_type == "D":
        if cell is None:
            raise InputFormatError(f"{source}: a 3x3 cell is required for posfiletype D")
        direct_cell = np.asarray(cell, dtype=float)
        if direct_cell.shape != (3, 3) or not np.isfinite(direct_cell).all():
            raise InputFormatError(f"{source}: cell must be a finite 3x3 matrix for posfiletype D")
    records: dict[int, PositionRecord] = {}
    for line, tokens in _rows(source):
        if len(tokens) != 5:
            raise InputFormatError(f"{source}:{line}: expected 5 columns: site atom_type x y z")
        site = _integer(tokens[0], path=source, line=line, field="site")
        if site in records:
            raise InputFormatError(f"{source}:{line}: duplicate site index {site}")
        position = np.asarray(
            [_float(token, path=source, line=line, field="position") for token in tokens[2:5]],
            dtype=float,
        )
        if direct_cell is not None:
            position = position @ direct_cell
        records[site] = PositionRecord(site, _identifier(tokens[1]), tuple(float(value) for value in position), line)
    return records


def parse_momfile(path: str | Path) -> dict[int, MomentRecord]:
    """Parse ``site moment_field moment [sx sy sz]`` rows.

    The second field is accepted as opaque UppASD moment-file metadata.  It is
    not a species/type identifier; species identity comes from the second
    column of ``posfile``.
    """

    source = Path(path).expanduser().resolve()
    records: dict[int, MomentRecord] = {}
    for line, tokens in _rows(source):
        if len(tokens) not in (3, 6):
            raise InputFormatError(f"{source}:{line}: expected 3 or 6 columns: site atom_type moment [sx sy sz]")
        site = _integer(tokens[0], path=source, line=line, field="site")
        if site in records:
            raise InputFormatError(f"{source}:{line}: duplicate site index {site}")
        moment = _float(tokens[2], path=source, line=line, field="moment")
        direction = None
        if len(tokens) == 6:
            direction = tuple(_float(token, path=source, line=line, field="spin direction") for token in tokens[3:6])
        records[site] = MomentRecord(site, tokens[1], moment, direction, line)
    return records


def parse_exchange(path: str | Path, *, strict: bool = True, report: ValidationReport | None = None) -> list[ExchangeBond]:
    """Parse literal UppASD ``i j rx ry rz Jij [distance]`` exchange rows.

    With ``strict=False``, malformed rows are skipped and added to ``report``.
    The optional distance is diagnostic only; the displacement norm remains
    authoritative.
    """

    source = Path(path).expanduser().resolve()
    bonds: list[ExchangeBond] = []
    for line, tokens in _rows(source):
        try:
            if len(tokens) not in (6, 7):
                raise InputFormatError(f"{source}:{line}: expected 6 or 7 columns: i j rx ry rz Jij [distance]")
            i = _integer(tokens[0], path=source, line=line, field="i")
            j = _integer(tokens[1], path=source, line=line, field="j")
            displacement = tuple(_float(token, path=source, line=line, field="displacement") for token in tokens[2:5])
            jij = _float(tokens[5], path=source, line=line, field="Jij")
            supplied_distance = _float(tokens[6], path=source, line=line, field="distance") if len(tokens) == 7 else None
            bond = ExchangeBond(i, j, displacement, jij, supplied_distance, line)
            if supplied_distance is not None and not np.isclose(supplied_distance, bond.distance, rtol=1e-6, atol=1e-8):
                if report is not None:
                    report.add_warning("distance_mismatch", f"optional distance {supplied_distance:g} disagrees with displacement norm {bond.distance:g}", source=str(source), line=line)
            bonds.append(bond)
        except InputFormatError as exc:
            if strict:
                raise
            if report is not None:
                report.add_error("malformed_exchange_row", str(exc), source=str(source), line=line)
    return bonds


def load_uppasd(
    path: str | Path,
    *,
    energy_unit: str = "mRy",
    length_unit: str = "unspecified",
    strict: bool = False,
    expand_symmetry: bool = False,
    symmetry_symprec: float = 1e-5,
) -> LoadedUppASD:
    """Load and validate an UppASD input set as one internal model.

    Set ``expand_symmetry=True`` when the exchange file contains one
    representative per crystal-symmetry orbit rather than all neighbour
    translations.  The default retains the supplied rows exactly.
    """

    config = parse_inpsd(path)
    report = ValidationReport()
    for warning in config.warnings:
        report.add_warning("unknown_keyword", warning, source=str(config.input_file))
    positions = parse_posfile(config.posfile, posfiletype=config.posfiletype, cell=config.cell)
    moments = parse_momfile(config.momfile)
    mapping_errors: list[VectorMappingError] = []
    mapped_rows = map_exchange_file(
        config.exchange,
        cell=config.cell,
        positions=prepare_positions(positions, config.cell),
        maptype=config.maptype,
        posfiletype=config.posfiletype,
        ncell=config.ncell,
        bc=config.bc,
        strict=strict,
        deduplicate=False,
        errors=mapping_errors,
    )
    for error in mapping_errors:
        report.add_error(
            error.code,
            str(error),
            source=error.source or str(config.exchange),
            line=error.line,
        )
    bonds: list[ExchangeBond] = []
    for row in mapped_rows:
        if row.supplied_distance is not None and not np.isclose(
            row.supplied_distance,
            row.distance,
            rtol=1e-6,
            atol=1e-8,
        ):
            report.add_warning(
                "distance_mismatch",
                f"optional distance {row.supplied_distance:g} disagrees with mapped vector norm {row.distance:g}",
                source=str(config.exchange),
                line=row.source_line,
            )
        bonds.append(
            ExchangeBond(
                row.site_i,
                row.site_j,
                row.rij_cart,
                row.Jij,
                row.supplied_distance,
                row.source_line,
            )
        )

    for site in sorted(set(positions) - set(moments)):
        report.add_error("missing_moment", f"site {site} appears in posfile but not momfile", source=str(config.momfile))
    for site in sorted(set(moments) - set(positions)):
        report.add_warning("moment_without_position", f"site {site} appears in momfile but not posfile", source=str(config.posfile))
    length_scale = 1.0 if config.alat is None else config.alat
    sites = [
        MagneticSite(
            index=record.site,
            atom_type=record.atom_type,
            position=tuple(float(length_scale * value) for value in record.position),
            moment=moments[record.site].moment if record.site in moments else None,
            spin_direction=moments[record.site].spin_direction if record.site in moments else None,
        )
        for record in sorted(positions.values(), key=lambda item: item.site)
    ]
    scaled_bonds = [
        ExchangeBond(
            bond.i,
            bond.j,
            tuple(float(length_scale * value) for value in bond.displacement),
            bond.jij,
            None if bond.supplied_distance is None else float(length_scale * bond.supplied_distance),
            bond.source_line,
        )
        for bond in bonds
    ]
    model = MagneticCrystal(
        cell=config.cell * length_scale,
        sites=sites,
        exchange_bonds=scaled_bonds,
        units=UnitMetadata(energy=energy_unit, length="m" if config.alat is not None else length_unit),
        source_files={"inpsd": config.input_file, "posfile": config.posfile, "momfile": config.momfile, "exchange": config.exchange},
    )
    expansion_report = None
    if expand_symmetry:
        expanded = expand_exchange_symmetry(model, symprec=symmetry_symprec)
        model = expanded.model
        expansion_report = expanded.report
        report.add_warning(
            "symmetry_expanded",
            f"expanded {expansion_report.input_bonds} representative exchange bond(s) to {expansion_report.output_bonds} using {expansion_report.symmetry_operations} spglib symmetry operation(s)",
            source=str(config.exchange),
        )
    validate_model(model, report)
    return LoadedUppASD(model, report, config, expansion_report)


# Explicit aliases make the input terminology used by different UppASD
# datasets (``exchange`` versus ``jfile``) equally discoverable.
parse_jfile = parse_exchange
parse_uppasd = load_uppasd


def _format_matrix(matrix: np.ndarray) -> str:
    return "\n".join("  " + " ".join(f"{value: .10g}" for value in row) for row in matrix)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect and validate an UppASD input set")
    parser.add_argument("inpsd", type=Path, help="path to inpsd.dat")
    parser.add_argument("--energy-unit", default="mRy", help="energy unit of Jij values (default: mRy, the UppASD convention)")
    parser.add_argument("--length-unit", default="unspecified")
    parser.add_argument("--expand-symmetry", "--symmetry", dest="expand_symmetry", action="store_true", help="expand symmetry-reduced exchange representatives with spglib")
    parser.add_argument("--symmetry-symprec", type=float, default=1e-5, help="spglib symmetry tolerance used with --expand-symmetry")
    args = parser.parse_args(argv)
    try:
        loaded = load_uppasd(
            args.inpsd,
            energy_unit=args.energy_unit,
            length_unit=args.length_unit,
            expand_symmetry=args.expand_symmetry,
            symmetry_symprec=args.symmetry_symprec,
        )
    except (FileNotFoundError, InputFormatError, SymmetryExpansionError) as exc:
        parser.error(str(exc))
    model = loaded.model
    print(f"Input: {loaded.config.input_file}")
    print("Cell:")
    print(_format_matrix(model.cell))
    if loaded.config.alat is not None:
        print(f"Alat: {loaded.config.alat:.10g} m (applied to cell, positions, and exchange displacements)")
    print(f"Cell volume: {model.cell_volume:.10g}")
    print(f"Basis sites: {len(model.sites)}")
    print("Moments: " + ", ".join(f"{site.index}={site.moment:g}" for site in model.sites if site.moment is not None))
    print(f"Exchange bonds: {len(model.exchange_bonds)}")
    if loaded.symmetry_expansion is not None:
        expansion = loaded.symmetry_expansion
        print(f"Symmetry expansion: {expansion.input_bonds} -> {expansion.output_bonds} bonds using {expansion.symmetry_operations} operations")
    if model.exchange_bonds:
        distances = model.bond_distances
        print(f"Bond distance range: {distances.min():.10g} .. {distances.max():.10g}")
    else:
        print("Bond distance range: unavailable (no exchange bonds)")
    print(f"Pair complete: {loaded.report.pair_complete}")
    print(f"Real-space Hermitian: {loaded.report.real_space_hermitian}")
    if loaded.report.issues:
        print("Validation diagnostics:")
        for issue in loaded.report.issues:
            location = f" ({issue.source}:{issue.line})" if issue.source and issue.line else ""
            print(f"- {issue.level}: {issue.message}{location}")
    return 0 if loaded.report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
