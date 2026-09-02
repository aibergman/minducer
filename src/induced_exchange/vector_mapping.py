"""UppASD ``jfile`` vector and target-cell mapping utilities.

The functions in this module keep the input vector convention explicit.  Cell
vectors are rows, so a direct/fractional vector ``v`` is converted with
``v @ cell``.  Maptype 1 treats the jfile vector as a bond vector; maptypes 2
and 3 treat it as lattice-vector coefficients and add the basis-site
separation using folded or raw positions, respectively.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt


ArrayLike = npt.ArrayLike


class VectorMappingError(ValueError):
    """Raised when an UppASD vector or target-cell row cannot be mapped."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "vector_mapping_error",
        source: str | Path | None = None,
        line: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.source = None if source is None else str(source)
        self.line = line


@dataclass(frozen=True)
class PreparedPositions:
    """Converted and folded basis positions used by the vector mapper."""

    cell: np.ndarray
    positions_raw: dict[int, np.ndarray]
    positions_folded: dict[int, np.ndarray]
    wrap_shifts: dict[int, np.ndarray]
    atom_types: dict[int, object]


@dataclass(frozen=True)
class TargetCellMatch:
    """Best basis-site match for a Cartesian target position."""

    site_id: int
    cell_offset: tuple[int, int, int]
    residual: tuple[float, float, float]

    @property
    def site(self) -> int:
        """Alias for callers that refer to the matched basis site as ``site``."""

        return self.site_id

    @property
    def inferred_target_cell_offset(self) -> tuple[int, int, int]:
        return self.cell_offset

    @property
    def match_residual(self) -> tuple[float, float, float]:
        return self.residual

    def __iter__(self):
        yield self.site_id
        yield self.cell_offset
        yield self.residual


@dataclass(frozen=True)
class JFileRow:
    """One parsed non-alloy or random-alloy jfile row."""

    site_i: int
    site_j: int
    input_Rij: tuple[float, float, float]
    Jij: float
    supplied_distance: float | None = None
    chemical_i: object | None = None
    chemical_j: object | None = None
    source_line: int | None = None


@dataclass(frozen=True)
class MappedExchangeRecord:
    """One jfile row after vector conversion and target-cell inference."""

    site_i: int
    site_j: int
    atom_type_i: object
    atom_type_j: object
    input_Rij: tuple[float, float, float]
    Jij: float
    rij_cart: tuple[float, float, float]
    distance: float
    inferred_target_cell_offset: tuple[int, int, int]
    match_residual: tuple[float, float, float]
    supplied_distance: float | None = None
    chemical_i: object | None = None
    chemical_j: object | None = None
    source_line: int | None = None


def _normalise_posfiletype(posfiletype: str) -> str:
    value = str(posfiletype).upper()
    if value not in {"C", "D"}:
        raise VectorMappingError(
            "posfiletype must be either C or D",
            code="unsupported_position_format",
        )
    return value


def _as_cell(cell: ArrayLike) -> np.ndarray:
    try:
        result = np.asarray(cell, dtype=float)
    except (TypeError, ValueError) as exc:
        raise VectorMappingError("cell must be a finite 3x3 array", code="malformed_cell") from exc
    if result.shape != (3, 3) or not np.isfinite(result).all():
        raise VectorMappingError("cell must be a finite 3x3 array", code="malformed_cell")
    if np.linalg.matrix_rank(result) < 3:
        raise VectorMappingError("cell must be non-singular", code="malformed_cell")
    return result.copy()


def _as_vector(vector: ArrayLike, *, field: str) -> np.ndarray:
    try:
        result = np.asarray(vector, dtype=float)
    except (TypeError, ValueError) as exc:
        raise VectorMappingError(f"{field} must be a finite 3-vector", code="malformed_vector") from exc
    if result.shape != (3,) or not np.isfinite(result).all():
        raise VectorMappingError(f"{field} must be a finite 3-vector", code="malformed_vector")
    return result


def _identifier(value: object) -> object:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return value


def _position_items(positions: Mapping[int, object] | Iterable[Sequence[object]]):
    if isinstance(positions, Mapping):
        for site, record in positions.items():
            atom_type = getattr(record, "atom_type", None)
            coordinates = getattr(record, "position", record)
            if isinstance(record, Mapping):
                atom_type = record.get("atom_type", atom_type)
                coordinates = record.get("position", coordinates)
            try:
                site_id = int(site)
            except (TypeError, ValueError) as exc:
                raise VectorMappingError(f"invalid site ID {site!r}", code="missing_site_id") from exc
            yield site_id, atom_type, coordinates
        return
    for row in positions:
        values = list(row)
        if len(values) != 5:
            raise VectorMappingError(
                "position rows must contain site_id atom_type x y z",
                code="malformed_position_row",
            )
        try:
            site_id = int(values[0])
        except (TypeError, ValueError) as exc:
            raise VectorMappingError(f"invalid site ID {values[0]!r}", code="missing_site_id") from exc
        yield site_id, values[1], values[2:5]


def prepare_positions(
    positions: Mapping[int, object] | Iterable[Sequence[object]],
    cell: ArrayLike,
    *,
    posfiletype: str = "C",
) -> PreparedPositions:
    """Convert and UppASD-fold basis positions into the first cell.

    ``positions`` may be a mapping of site IDs to position records/3-vectors
    or an iterable of ``site_id atom_type x y z`` rows.  ``positions_raw`` is
    retained after optional direct-coordinate conversion; ``positions_folded``
    applies ``floor(fractional + 1e-5)`` to each basis position.
    """

    direct_cell = _as_cell(cell)
    normalized_type = _normalise_posfiletype(posfiletype)
    inverse_cell = np.linalg.inv(direct_cell)
    raw: dict[int, np.ndarray] = {}
    atom_types: dict[int, object] = {}
    for site, atom_type, coordinates in _position_items(positions):
        if site in raw:
            raise VectorMappingError(f"duplicate site ID {site}", code="duplicate_site_id")
        position = _as_vector(coordinates, field="position").copy()
        if normalized_type == "D":
            position = position @ direct_cell
        raw[site] = position
        atom_types[site] = atom_type
    if not raw:
        raise VectorMappingError("positions must contain at least one site", code="missing_positions")

    folded: dict[int, np.ndarray] = {}
    shifts: dict[int, np.ndarray] = {}
    for site, position in raw.items():
        fractional = position @ inverse_cell
        shift = np.floor(fractional + 1e-5).astype(int)
        shifts[site] = shift
        folded[site] = position - shift @ direct_cell
    return PreparedPositions(direct_cell, raw, folded, shifts, atom_types)


def _prepared_positions(
    positions: PreparedPositions | Mapping[int, object] | Iterable[Sequence[object]],
    cell: ArrayLike | None,
    *,
    posfiletype: str,
) -> PreparedPositions:
    _normalise_posfiletype(posfiletype)
    if isinstance(positions, PreparedPositions):
        if cell is not None and not np.allclose(positions.cell, _as_cell(cell)):
            raise VectorMappingError("prepared positions and cell do not match", code="malformed_cell")
        return positions
    if cell is None:
        raise VectorMappingError("cell is required when positions are not prepared", code="malformed_cell")
    return prepare_positions(positions, cell, posfiletype=posfiletype)


def map_exchange_vector(
    site_i: int,
    site_j: int,
    input_Rij: ArrayLike,
    *,
    cell: ArrayLike | None = None,
    positions: PreparedPositions | Mapping[int, object] | Iterable[Sequence[object]],
    maptype: int = 1,
    posfiletype: str = "C",
) -> np.ndarray:
    """Map one jfile vector to Cartesian coordinates.

    Maptype 1 uses the vector directly for Cartesian positions, or converts
    it with ``R @ cell`` for direct positions.  Maptypes 2 and 3 always treat
    ``R`` as lattice-vector coefficients and add the folded/raw basis-site
    separation, respectively.
    """

    if maptype not in {1, 2, 3}:
        raise VectorMappingError("maptype must be 1, 2, or 3", code="invalid_maptype")
    normalized_type = _normalise_posfiletype(posfiletype)
    prepared = _prepared_positions(positions, cell, posfiletype=posfiletype)
    if site_i not in prepared.positions_raw or site_j not in prepared.positions_raw:
        missing = site_i if site_i not in prepared.positions_raw else site_j
        raise VectorMappingError(f"site ID {missing} is missing from positions", code="missing_site_id")
    vector = _as_vector(input_Rij, field="input Rij")
    if maptype == 1:
        if normalized_type == "C":
            return vector.copy()
        return vector @ prepared.cell
    translation = vector @ prepared.cell
    if maptype == 2:
        return prepared.positions_folded[site_j] - prepared.positions_folded[site_i] + translation
    return prepared.positions_raw[site_j] - prepared.positions_raw[site_i] + translation


def infer_target_site(
    target: ArrayLike,
    positions: PreparedPositions | Mapping[int, object] | Iterable[Sequence[object]],
    *,
    cell: ArrayLike | None = None,
    expected_site: int | None = None,
    posfiletype: str = "C",
    tolerance: float = 1e-5,
) -> TargetCellMatch:
    """Infer the basis site and integer cell offset for a Cartesian target.

    The residual is returned in fractional coordinates.  ``tolerance`` is a
    squared residual threshold, matching the scale used by UppASD's vector
    comparisons.
    """

    prepared = _prepared_positions(positions, cell, posfiletype=posfiletype)
    target_vector = _as_vector(target, field="target position")
    inverse_cell = np.linalg.inv(prepared.cell)
    candidates: list[tuple[float, int, np.ndarray, np.ndarray]] = []
    for site, basis_position in prepared.positions_folded.items():
        relative = (target_vector - basis_position) @ inverse_cell
        offset = np.rint(relative).astype(int)
        residual = relative - offset
        squared_residual = float(residual @ residual)
        candidates.append((squared_residual, site, offset, residual))
    squared_residual, site, offset, residual = min(candidates, key=lambda item: item[0])
    if squared_residual >= tolerance:
        raise VectorMappingError(
            f"target position does not match a basis site within tolerance; residual={residual.tolist()}",
            code="unmatched_target_position",
        )
    if expected_site is not None and site != expected_site:
        raise VectorMappingError(
            f"target position matches site {site}, not requested site {expected_site}",
            code="unmatched_target_position",
        )
    return TargetCellMatch(site, tuple(int(value) for value in offset), tuple(float(value) for value in residual))


def _normalise_ncell(ncell: Sequence[int] | None) -> tuple[int, int, int] | None:
    if ncell is None:
        return None
    try:
        values = tuple(int(value) for value in ncell)
    except (TypeError, ValueError) as exc:
        raise VectorMappingError("ncell must contain three positive integers", code="malformed_ncell") from exc
    if len(values) != 3 or any(value <= 0 for value in values):
        raise VectorMappingError("ncell must contain three positive integers", code="malformed_ncell")
    return values


def _normalise_bc(bc: Sequence[str] | None) -> tuple[str, str, str] | None:
    if bc is None:
        return None
    values = tuple(str(value).upper() for value in bc)
    aliases = {"P": "P", "PERIODIC": "P", "F": "F", "FREE": "F", "O": "F", "OPEN": "F"}
    if len(values) != 3 or any(value not in aliases for value in values):
        raise VectorMappingError("boundary conditions must be P or F in each direction", code="malformed_bc")
    return tuple(aliases[value] for value in values)  # type: ignore[return-value]


def _periodic_offset(
    offset: tuple[int, int, int],
    *,
    ncell: tuple[int, int, int] | None,
    bc: tuple[str, str, str] | None,
) -> tuple[int, int, int]:
    if ncell is None:
        return offset
    conditions = bc or ("P", "P", "P")
    result = list(offset)
    for axis, (value, size, condition) in enumerate(zip(offset, ncell, conditions)):
        if condition == "P":
            result[axis] = value % size
        elif not 0 <= value < size:
            raise VectorMappingError(
                f"target cell offset {offset} is outside the non-periodic supercell",
                code="target_out_of_bounds",
            )
    return tuple(result)


def _parse_float(token: str, *, source: Path, line: int, field: str) -> float:
    try:
        value = float(token)
    except ValueError as exc:
        raise VectorMappingError(
            f"{source}:{line}: invalid {field} value {token!r}",
            code="malformed_jfile_row",
            source=source,
            line=line,
        ) from exc
    if not np.isfinite(value):
        raise VectorMappingError(
            f"{source}:{line}: non-finite {field} value {token!r}",
            code="malformed_jfile_row",
            source=source,
            line=line,
        )
    return value


def _is_finite_float(token: str) -> bool:
    try:
        value = float(token)
    except ValueError:
        return False
    return bool(np.isfinite(value))


def _parse_int(token: str, *, source: Path, line: int, field: str) -> int:
    try:
        return int(token)
    except ValueError as exc:
        raise VectorMappingError(
            f"{source}:{line}: invalid {field} value {token!r}",
            code="malformed_jfile_row",
            source=source,
            line=line,
        ) from exc


def read_jfile(
    path: str | Path,
    *,
    strict: bool = True,
    errors: list[VectorMappingError] | None = None,
) -> list[JFileRow]:
    """Read non-alloy and random-alloy UppASD jfile rows.

    Supported layouts are ``i j r1 r2 r3 Jij [distance]`` and
    ``i j chemical_i chemical_j r1 r2 r3 Jij [distance]``.  Chemical fields
    are retained as metadata and do not replace the basis atom types.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    rows: list[JFileRow] = []
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#", 1)[0].split("!", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        try:
            if len(tokens) < 6:
                raise VectorMappingError(
                    f"{source}:{line_number}: expected at least 6 columns",
                    code="malformed_jfile_row",
                    source=source,
                    line=line_number,
                )
            # A normal jfile row is six required fields, optionally followed
            # by a distance and/or arbitrary trailing data.  Random-alloy
            # rows are recognized by their two non-numeric chemical fields;
            # this avoids mistaking trailing text on a normal row for alloy
            # metadata.
            random_alloy = len(tokens) >= 8 and (
                not _is_finite_float(tokens[2]) or not _is_finite_float(tokens[3])
            )
            if random_alloy:
                site_i = _parse_int(tokens[0], source=source, line=line_number, field="site_i")
                site_j = _parse_int(tokens[1], source=source, line=line_number, field="site_j")
                chemical_i = _identifier(tokens[2])
                chemical_j = _identifier(tokens[3])
                vector_start = 4
            else:
                site_i = _parse_int(tokens[0], source=source, line=line_number, field="site_i")
                site_j = _parse_int(tokens[1], source=source, line=line_number, field="site_j")
                chemical_i = None
                chemical_j = None
                vector_start = 2
            if len(tokens) < vector_start + 4:
                raise VectorMappingError(
                    f"{source}:{line_number}: incomplete exchange row",
                    code="malformed_jfile_row",
                    source=source,
                    line=line_number,
                )
            input_vector = tuple(
                _parse_float(tokens[index], source=source, line=line_number, field="Rij")
                for index in range(vector_start, vector_start + 3)
            )
            Jij = _parse_float(tokens[vector_start + 3], source=source, line=line_number, field="Jij")
            supplied_distance = (
                _parse_float(tokens[vector_start + 4], source=source, line=line_number, field="distance")
                if len(tokens) > vector_start + 4 and _is_finite_float(tokens[vector_start + 4])
                else None
            )
            rows.append(JFileRow(site_i, site_j, input_vector, Jij, supplied_distance, chemical_i, chemical_j, line_number))
        except VectorMappingError as exc:
            if strict:
                raise
            if errors is not None:
                errors.append(exc)
    return rows


def _replace_duplicate_vectors(
    records: list[MappedExchangeRecord],
    *,
    threshold: float = 1e-5,
) -> list[MappedExchangeRecord]:
    grouped: dict[object, list[MappedExchangeRecord]] = {}
    result: list[MappedExchangeRecord] = []
    for record in records:
        group = grouped.setdefault(record.atom_type_i, [])
        duplicate_index = next(
            (
                index
                for index, previous in enumerate(group)
                if float(np.sum((np.asarray(record.rij_cart) - np.asarray(previous.rij_cart)) ** 2)) < threshold
            ),
            None,
        )
        if duplicate_index is None:
            group.append(record)
            result.append(record)
        else:
            previous = group[duplicate_index]
            result[result.index(previous)] = record
            group[duplicate_index] = record
    return result


def map_exchange_file(
    path: str | Path,
    *,
    cell: ArrayLike | None = None,
    positions: PreparedPositions | Mapping[int, object] | Iterable[Sequence[object]],
    maptype: int = 1,
    posfiletype: str = "C",
    ncell: Sequence[int] | None = None,
    bc: Sequence[str] | None = None,
    boundary_conditions: Sequence[str] | None = None,
    strict: bool = True,
    deduplicate: bool = True,
    duplicate_threshold: float = 1e-5,
    target_tolerance: float = 1e-5,
    errors: list[VectorMappingError] | None = None,
) -> list[MappedExchangeRecord]:
    """Read and map a complete jfile into structured Cartesian records.

    Periodic target offsets are reduced modulo ``ncell``.  Free-boundary
    offsets must lie in ``[0, ncell)``.  Duplicate Cartesian vectors for the
    same central atom type are replaced by the later row by default, matching
    UppASD's last-value-wins behavior.
    """

    if boundary_conditions is not None:
        if bc is not None and tuple(bc) != tuple(boundary_conditions):
            raise VectorMappingError("bc and boundary_conditions disagree", code="malformed_bc")
        bc = boundary_conditions
    normalized_ncell = _normalise_ncell(ncell)
    normalized_bc = _normalise_bc(bc)
    prepared = _prepared_positions(positions, cell, posfiletype=posfiletype)
    rows = read_jfile(path, strict=strict, errors=errors)
    mapped: list[MappedExchangeRecord] = []
    for row in rows:
        try:
            rij_cart = map_exchange_vector(
                row.site_i,
                row.site_j,
                row.input_Rij,
                positions=prepared,
                maptype=maptype,
                posfiletype=posfiletype,
            )
            target = prepared.positions_folded[row.site_i] + rij_cart
            match = infer_target_site(
                target,
                prepared,
                expected_site=row.site_j,
                tolerance=target_tolerance,
            )
            offset = _periodic_offset(match.cell_offset, ncell=normalized_ncell, bc=normalized_bc)
            mapped.append(
                MappedExchangeRecord(
                    site_i=row.site_i,
                    site_j=row.site_j,
                    atom_type_i=prepared.atom_types[row.site_i],
                    atom_type_j=prepared.atom_types[row.site_j],
                    input_Rij=row.input_Rij,
                    Jij=row.Jij,
                    rij_cart=tuple(float(value) for value in rij_cart),
                    distance=float(np.linalg.norm(rij_cart)),
                    inferred_target_cell_offset=offset,
                    match_residual=match.residual,
                    supplied_distance=row.supplied_distance,
                    chemical_i=row.chemical_i,
                    chemical_j=row.chemical_j,
                    source_line=row.source_line,
                )
            )
        except (KeyError, VectorMappingError) as exc:
            if isinstance(exc, KeyError):
                error = VectorMappingError(
                    f"site ID {exc.args[0]} is missing from positions",
                    code="missing_site_id",
                    line=row.source_line,
                )
            else:
                error = exc
            if strict:
                raise error
            if errors is not None:
                errors.append(error)
    return _replace_duplicate_vectors(mapped, threshold=duplicate_threshold) if deduplicate else mapped


__all__ = [
    "JFileRow",
    "MappedExchangeRecord",
    "PreparedPositions",
    "TargetCellMatch",
    "VectorMappingError",
    "infer_target_site",
    "map_exchange_file",
    "map_exchange_vector",
    "prepare_positions",
    "read_jfile",
]
