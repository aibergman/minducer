"""Reciprocal-space exchange analysis.

The real-space cell convention used by :mod:`induced_exchange` is a matrix
whose *rows* are the Cartesian direct-lattice vectors.  Reciprocal vectors are
also rows and are defined with an explicit ``2*pi`` convention::

    A @ B.T = 2*pi*I

For a reduced reciprocal coordinate ``h`` and a Cartesian coordinate ``q``::

    q_cart = h @ B

The Fourier transform uses the displacement on each :class:`ExchangeBond`
verbatim; basis positions are never used to reconstruct it.

The fixed UppASD Hamiltonian convention is
``H = -sum_(i != j) Jij e_i dot e_j``.  Consequently a normalized Fourier
mode has energy proportional to ``-v^dagger J(q) v``: the ordering tendency is
associated with the *largest* eigenvalue of ``J(q)``.  The Fourier transform
itself contains no factor of two; the factor belongs to the harmonic
curvature derived from the Hamiltonian.  This is an ordering diagnostic, not a
claim that the supplied Jij are exact first-principles susceptibilities.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .model import MagneticCrystal, _cell_is_degenerate


ArrayLike = Sequence[float] | np.ndarray


def _as_cell(cell: ArrayLike) -> np.ndarray:
    matrix = np.asarray(cell, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"cell must have shape (3, 3), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("cell contains non-finite values")
    if _cell_is_degenerate(matrix):
        raise ValueError("cell has zero (or numerically zero) volume")
    return matrix


@dataclass(frozen=True)
class ReciprocalLattice:
    """Direct and reciprocal lattice bases, both represented by rows."""

    cell: np.ndarray

    def __post_init__(self) -> None:
        direct = _as_cell(self.cell)
        object.__setattr__(self, "cell", direct)
        object.__setattr__(self, "reciprocal_vectors", 2.0 * np.pi * np.linalg.inv(direct).T)

    reciprocal_vectors: np.ndarray = field(init=False, repr=False)

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.cell)))

    @property
    def reciprocal_volume(self) -> float:
        return float(abs(np.linalg.det(self.reciprocal_vectors)))

    def fractional_to_cartesian(self, q_fractional: ArrayLike) -> np.ndarray:
        """Convert reduced reciprocal coordinates to Cartesian coordinates."""

        q = np.asarray(q_fractional, dtype=float)
        if q.shape[-1:] != (3,):
            raise ValueError(f"q must end in length 3, got shape {q.shape}")
        return np.matmul(q, self.reciprocal_vectors)

    def cartesian_to_fractional(self, q_cartesian: ArrayLike) -> np.ndarray:
        """Convert Cartesian reciprocal coordinates to reduced coordinates."""

        q = np.asarray(q_cartesian, dtype=float)
        if q.shape[-1:] != (3,):
            raise ValueError(f"q must end in length 3, got shape {q.shape}")
        return np.matmul(q, np.linalg.inv(self.reciprocal_vectors))

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell": self.cell.tolist(),
            "reciprocal_vectors_2pi": self.reciprocal_vectors.tolist(),
            "volume": self.volume,
            "reciprocal_volume": self.reciprocal_volume,
        }


def reciprocal_lattice(cell: ArrayLike) -> ReciprocalLattice:
    """Construct a reciprocal lattice with the explicit ``2*pi`` convention."""

    return ReciprocalLattice(np.asarray(cell, dtype=float))


def _lattice_for(model_or_cell: MagneticCrystal | ArrayLike) -> ReciprocalLattice:
    return reciprocal_lattice(model_or_cell.cell if isinstance(model_or_cell, MagneticCrystal) else model_or_cell)


def _q_points(q_points: ArrayLike, lattice: ReciprocalLattice, coordinates: str) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q_points, dtype=float)
    if q.shape == (3,):
        q = q.reshape(1, 3)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError(f"q_points must have shape (3,) or (n, 3), got {q.shape}")
    if not np.isfinite(q).all():
        raise ValueError("q_points contains non-finite values")
    normalized = coordinates.lower().replace("_", "-")
    if normalized in {"fractional", "reduced", "reciprocal-fractional"}:
        q_fractional = q
        q_cartesian = lattice.fractional_to_cartesian(q)
    elif normalized in {"cartesian", "cart", "reciprocal-cartesian"}:
        q_cartesian = q
        q_fractional = lattice.cartesian_to_fractional(q)
    else:
        raise ValueError("coordinates must be 'fractional' or 'cartesian'")
    return q_fractional, q_cartesian


def regular_q_mesh(
    cell_or_lattice: MagneticCrystal | ArrayLike | ReciprocalLattice,
    divisions: Sequence[int],
    *,
    coordinates: str = "fractional",
    centered: bool = False,
    endpoint: bool = False,
) -> np.ndarray:
    """Return a regular reciprocal mesh.

    ``divisions`` gives the number of samples along each reciprocal basis
    vector.  The default mesh spans ``[0, 1)`` in each reduced coordinate;
    ``centered=True`` spans the equivalent interval around zero.  Cartesian
    output is selected with ``coordinates='cartesian'``.
    """

    if isinstance(cell_or_lattice, ReciprocalLattice):
        lattice = cell_or_lattice
    else:
        lattice = _lattice_for(cell_or_lattice)
    sizes = tuple(int(value) for value in divisions)
    if len(sizes) != 3 or any(value <= 0 for value in sizes):
        raise ValueError("divisions must contain three positive integers")
    if endpoint and any(value < 2 for value in sizes):
        raise ValueError("endpoint=True requires at least two samples per direction")
    axes = []
    for size in sizes:
        axis = np.linspace(0.0, 1.0, size, endpoint=endpoint) if endpoint else np.arange(size, dtype=float) / size
        if centered:
            axis = (axis + 0.5) % 1.0 - 0.5 if not endpoint else axis - 0.5
        axes.append(axis)
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    normalized = coordinates.lower().replace("_", "-")
    if normalized in {"fractional", "reduced", "reciprocal-fractional"}:
        return mesh
    if normalized in {"cartesian", "cart", "reciprocal-cartesian"}:
        return lattice.fractional_to_cartesian(mesh)
    raise ValueError("coordinates must be 'fractional' or 'cartesian'")


# Short aliases make the mesh API easy to discover from notebooks.
q_mesh = regular_q_mesh
reciprocal_mesh = regular_q_mesh


@dataclass(frozen=True)
class HermiticityReport:
    """Numerical Hermiticity diagnostics for one or more J(q) matrices."""

    is_hermitian: bool
    max_abs_error: float
    tolerance: float
    errors: np.ndarray
    violating_indices: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_hermitian": self.is_hermitian,
            "max_abs_error": self.max_abs_error,
            "tolerance": self.tolerance,
            "errors": self.errors.tolist(),
            "violating_indices": list(self.violating_indices),
        }


def check_hermiticity(
    matrices: np.ndarray,
    *,
    atol: float = 1e-10,
    rtol: float = 1e-8,
) -> HermiticityReport:
    """Check ``J(q) == J(q).conj().T`` without symmetrizing the input."""

    values = np.asarray(matrices, dtype=complex)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("matrices must have shape (nq, nsite, nsite) or (nsite, nsite)")
    errors = np.max(np.abs(values - values.conj().transpose(0, 2, 1)), axis=(1, 2), initial=0.0)
    scale = max(1.0, float(np.max(np.abs(values), initial=0.0)))
    tolerance = float(atol + rtol * scale)
    violating = tuple(int(index) for index in np.flatnonzero(errors > tolerance))
    return HermiticityReport(
        is_hermitian=not violating,
        max_abs_error=float(np.max(errors, initial=0.0)),
        tolerance=tolerance,
        errors=errors,
        violating_indices=violating,
    )


@dataclass(frozen=True)
class FourierExchangeResult:
    q_fractional: np.ndarray
    q_cartesian: np.ndarray
    matrices: np.ndarray
    site_indices: tuple[int, ...]
    hermiticity: HermiticityReport

    @property
    def J(self) -> np.ndarray:
        """Alias using the notation in the physics documentation."""

        return self.matrices

    def as_dict(self) -> dict[str, Any]:
        return {
            "q_fractional": self.q_fractional.tolist(),
            "q_cartesian": self.q_cartesian.tolist(),
            "J_real": self.matrices.real.tolist(),
            "J_imag": self.matrices.imag.tolist(),
            "site_indices": list(self.site_indices),
            "hermiticity": self.hermiticity.as_dict(),
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.as_dict(), **kwargs)


def fourier_exchange(
    model: MagneticCrystal,
    q_points: ArrayLike,
    *,
    coordinates: str = "cartesian",
    atol: float = 1e-10,
    rtol: float = 1e-8,
) -> np.ndarray:
    """Evaluate the multi-sublattice ``J_ab(q)`` Fourier transform.

    The returned array has shape ``(nq, nsite, nsite)``.  Duplicate bonds are
    retained and accumulated.  A non-pair-complete input therefore remains
    non-Hermitian and is not silently repaired.
    """

    result = exchange_fourier(model, q_points, coordinates=coordinates, atol=atol, rtol=rtol)
    return result.matrices


def exchange_fourier(
    model: MagneticCrystal,
    q_points: ArrayLike,
    *,
    coordinates: str = "cartesian",
    atol: float = 1e-10,
    rtol: float = 1e-8,
) -> FourierExchangeResult:
    """Return ``J(q)`` together with coordinates and Hermiticity diagnostics."""

    lattice = reciprocal_lattice(model.cell)
    q_fractional, q_cartesian = _q_points(q_points, lattice, coordinates)
    site_indices = tuple(sorted(model.site_indices))
    index_map = {site: position for position, site in enumerate(site_indices)}
    matrices = np.zeros((len(q_cartesian), len(site_indices), len(site_indices)), dtype=complex)

    if model.exchange_bonds:
        displacements = np.asarray([bond.displacement for bond in model.exchange_bonds], dtype=float)
        phases = np.exp(1j * (q_cartesian @ displacements.T))
        for bond_index, bond in enumerate(model.exchange_bonds):
            try:
                i = index_map[bond.i]
                j = index_map[bond.j]
            except KeyError as exc:
                raise ValueError(f"exchange bond ({bond.i}, {bond.j}) references an unknown site") from exc
            matrices[:, i, j] += bond.jij * phases[:, bond_index]

    return FourierExchangeResult(
        q_fractional=q_fractional,
        q_cartesian=q_cartesian,
        matrices=matrices,
        site_indices=site_indices,
        hermiticity=check_hermiticity(matrices, atol=atol, rtol=rtol),
    )


@dataclass(frozen=True)
class ExchangeEigenSystem:
    q_fractional: np.ndarray
    q_cartesian: np.ndarray
    matrices: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    site_indices: tuple[int, ...]
    hermiticity: HermiticityReport

    @property
    def J(self) -> np.ndarray:
        return self.matrices

    def as_dict(self) -> dict[str, Any]:
        return {
            "q_fractional": self.q_fractional.tolist(),
            "q_cartesian": self.q_cartesian.tolist(),
            "J_real": self.matrices.real.tolist(),
            "J_imag": self.matrices.imag.tolist(),
            "eigenvalues_real": self.eigenvalues.real.tolist(),
            "eigenvalues_imag": self.eigenvalues.imag.tolist(),
            "eigenvectors_real": self.eigenvectors.real.tolist(),
            "eigenvectors_imag": self.eigenvectors.imag.tolist(),
            "site_indices": list(self.site_indices),
            "hermiticity": self.hermiticity.as_dict(),
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.as_dict(), **kwargs)


def exchange_eigensystem(
    model: MagneticCrystal,
    q_points: ArrayLike,
    *,
    coordinates: str = "cartesian",
    atol: float = 1e-10,
    rtol: float = 1e-8,
) -> ExchangeEigenSystem:
    """Return eigenvalues/eigenvectors of ``J(q)``.

    Hermitian matrices use ``eigh``.  For malformed/incomplete input the
    general eigensolver is used, preserving complex eigenvalues and the
    Hermiticity warning instead of hiding the input problem.
    """

    transformed = exchange_fourier(model, q_points, coordinates=coordinates, atol=atol, rtol=rtol)
    n_q, n_sites, _ = transformed.matrices.shape
    values = np.empty((n_q, n_sites), dtype=complex)
    vectors = np.empty((n_q, n_sites, n_sites), dtype=complex)
    for index, matrix in enumerate(transformed.matrices):
        if transformed.hermiticity.errors[index] <= transformed.hermiticity.tolerance:
            current_values, current_vectors = np.linalg.eigh(matrix)
        else:
            current_values, current_vectors = np.linalg.eig(matrix)
        order = np.argsort(current_values.real)[::-1]
        values[index] = current_values[order]
        vectors[index] = current_vectors[:, order]
    return ExchangeEigenSystem(
        q_fractional=transformed.q_fractional,
        q_cartesian=transformed.q_cartesian,
        matrices=transformed.matrices,
        eigenvalues=values,
        eigenvectors=vectors,
        site_indices=transformed.site_indices,
        hermiticity=transformed.hermiticity,
    )


@dataclass(frozen=True)
class OrderingAnalysis:
    q_order_fractional: np.ndarray | None
    q_order_cartesian: np.ndarray | None
    eigenvalue: complex | None
    sublattice_eigenvector: np.ndarray | None
    gamma_index: int | None
    gamma_eigenvalue: complex | None
    gamma_locally_stable: bool | None
    local_neighbor_indices: tuple[int, ...]
    hermiticity: HermiticityReport

    @property
    def q_order(self) -> np.ndarray | None:
        """Alias for the ordering vector in Cartesian reciprocal units."""

        return self.q_order_cartesian

    def as_dict(self) -> dict[str, Any]:
        return {
            "q_order_fractional": None if self.q_order_fractional is None else self.q_order_fractional.tolist(),
            "q_order_cartesian": None if self.q_order_cartesian is None else self.q_order_cartesian.tolist(),
            "eigenvalue_real": None if self.eigenvalue is None else float(np.real(self.eigenvalue)),
            "eigenvalue_imag": None if self.eigenvalue is None else float(np.imag(self.eigenvalue)),
            "sublattice_eigenvector_real": None if self.sublattice_eigenvector is None else self.sublattice_eigenvector.real.tolist(),
            "sublattice_eigenvector_imag": None if self.sublattice_eigenvector is None else self.sublattice_eigenvector.imag.tolist(),
            "gamma_index": self.gamma_index,
            "gamma_eigenvalue_real": None if self.gamma_eigenvalue is None else float(np.real(self.gamma_eigenvalue)),
            "gamma_locally_stable": self.gamma_locally_stable,
            "local_neighbor_indices": list(self.local_neighbor_indices),
            "hermiticity": self.hermiticity.as_dict(),
        }


def ordering_analysis(
    model: MagneticCrystal,
    q_points: ArrayLike,
    *,
    coordinates: str = "cartesian",
    gamma_tolerance: float = 1e-10,
    atol: float = 1e-10,
    rtol: float = 1e-8,
) -> OrderingAnalysis:
    """Find the largest-eigenvalue ordering tendency on a supplied q set.

    ``gamma_locally_stable`` compares Gamma with the nearest non-Gamma q
    samples.  It is ``None`` if Gamma or a nonzero neighbor is absent, which
    avoids presenting a global mesh result as a local stability proof.
    """

    eigensystem = exchange_eigensystem(model, q_points, coordinates=coordinates, atol=atol, rtol=rtol)
    if len(eigensystem.eigenvalues) == 0:
        return OrderingAnalysis(None, None, None, None, None, None, None, (), eigensystem.hermiticity)
    leading = eigensystem.eigenvalues[:, 0]
    order_index = int(np.argmax(leading.real))
    # Identify Gamma in reduced coordinates modulo reciprocal lattice
    # vectors. Using |q_cartesian| would incorrectly reject q=(1,0,0),
    # although it is the same point as q=(0,0,0).
    gamma_distances = np.linalg.norm(eigensystem.q_fractional - np.rint(eigensystem.q_fractional), axis=1)
    gamma_candidates = np.flatnonzero(gamma_distances <= gamma_tolerance)
    gamma_index: int | None = int(gamma_candidates[0]) if len(gamma_candidates) else None
    gamma_value: complex | None = leading[gamma_index] if gamma_index is not None else None
    non_gamma = np.flatnonzero(gamma_distances > gamma_tolerance)
    neighbors: tuple[int, ...] = ()
    stable: bool | None = None
    if gamma_index is not None and len(non_gamma):
        distances = gamma_distances[non_gamma]
        nearest_distance = float(np.min(distances))
        nearest = non_gamma[distances <= nearest_distance * (1.0 + 1e-8)]
        neighbors = tuple(int(index) for index in nearest)
        stable = bool(np.real(gamma_value) + eigensystem.hermiticity.tolerance >= np.max(leading[nearest].real))
    return OrderingAnalysis(
        q_order_fractional=eigensystem.q_fractional[order_index],
        q_order_cartesian=eigensystem.q_cartesian[order_index],
        eigenvalue=leading[order_index],
        sublattice_eigenvector=eigensystem.eigenvectors[order_index, :, 0],
        gamma_index=gamma_index,
        gamma_eigenvalue=gamma_value,
        gamma_locally_stable=stable,
        local_neighbor_indices=neighbors,
        hermiticity=eigensystem.hermiticity,
    )


def exchange_extrema(eigensystem: ExchangeEigenSystem) -> dict[str, Any]:
    """Summarize global maximum/minimum exchange eigenvalues on a q set."""

    if eigensystem.eigenvalues.size == 0:
        return {
            "maximum": None,
            "minimum": None,
            "maximum_index": None,
            "minimum_index": None,
        }
    leading = eigensystem.eigenvalues.real
    maximum_index = np.unravel_index(int(np.argmax(leading)), leading.shape)
    minimum_index = np.unravel_index(int(np.argmin(leading)), leading.shape)
    return {
        "maximum": {
            "value": eigensystem.eigenvalues[maximum_index],
            "q_fractional": eigensystem.q_fractional[maximum_index[0]],
            "q_cartesian": eigensystem.q_cartesian[maximum_index[0]],
            "branch": int(maximum_index[1]),
        },
        "minimum": {
            "value": eigensystem.eigenvalues[minimum_index],
            "q_fractional": eigensystem.q_fractional[minimum_index[0]],
            "q_cartesian": eigensystem.q_cartesian[minimum_index[0]],
            "branch": int(minimum_index[1]),
        },
        "maximum_index": tuple(int(value) for value in maximum_index),
        "minimum_index": tuple(int(value) for value in minimum_index),
    }


@dataclass(frozen=True)
class QPath:
    q_fractional: np.ndarray
    q_cartesian: np.ndarray
    distance: np.ndarray
    tick_positions: np.ndarray
    tick_labels: tuple[str, ...]
    source: str


def _path_from_vertices(
    lattice: ReciprocalLattice,
    vertices: Sequence[tuple[str, ArrayLike]],
    *,
    n_per_segment: int,
    source: str,
) -> QPath:
    if len(vertices) < 2:
        raise ValueError("a reciprocal path needs at least two vertices")
    if n_per_segment < 1:
        raise ValueError("n_per_segment must be positive")
    labels = tuple(str(label) for label, _ in vertices)
    reduced_vertices = np.asarray([coordinates for _, coordinates in vertices], dtype=float)
    if reduced_vertices.shape != (len(vertices), 3) or not np.isfinite(reduced_vertices).all():
        raise ValueError("path vertices must contain finite 3-vectors")
    points = [reduced_vertices[0]]
    ticks = [0]
    for segment in range(len(vertices) - 1):
        segment_points = np.linspace(reduced_vertices[segment], reduced_vertices[segment + 1], n_per_segment + 1)[1:]
        points.extend(segment_points)
        ticks.append(len(points) - 1)
    q_fractional = np.asarray(points, dtype=float)
    q_cartesian = lattice.fractional_to_cartesian(q_fractional)
    distance = np.zeros(len(points), dtype=float)
    if len(points) > 1:
        distance[1:] = np.cumsum(np.linalg.norm(np.diff(q_cartesian, axis=0), axis=1))
    return QPath(q_fractional, q_cartesian, distance, np.asarray(ticks, dtype=int), labels, source)


def _fallback_vertices() -> list[tuple[str, np.ndarray]]:
    return [
        ("Gamma", np.array([0.0, 0.0, 0.0])),
        ("B1/2", np.array([0.5, 0.0, 0.0])),
        ("B2/2", np.array([0.0, 0.5, 0.0])),
        ("B3/2", np.array([0.0, 0.0, 0.5])),
        ("Gamma", np.array([0.0, 0.0, 0.0])),
    ]


def high_symmetry_path(
    model_or_cell: MagneticCrystal | ArrayLike,
    *,
    points: Mapping[str, ArrayLike] | Sequence[tuple[str, ArrayLike]] | None = None,
    n_per_segment: int = 50,
    use_seekpath: bool = True,
) -> QPath:
    """Build a reciprocal path, using seekpath when available.

    Explicit ``points`` are reduced reciprocal coordinates and always take
    precedence.  If seekpath is unavailable or cannot classify the structure,
    a transparent generic Gamma--boundary--Gamma path is returned with
    ``source='fallback'``.
    """

    lattice = _lattice_for(model_or_cell)
    if points is not None:
        vertices = list(points.items()) if isinstance(points, Mapping) else list(points)
        return _path_from_vertices(lattice, vertices, n_per_segment=n_per_segment, source="explicit")
    if use_seekpath and isinstance(model_or_cell, MagneticCrystal):
        try:
            import seekpath  # type: ignore

            positions = np.asarray([site.position for site in model_or_cell.sites], dtype=float)
            scaled_positions = positions @ np.linalg.inv(model_or_cell.cell)
            # spglib's default ``symprec`` is an absolute coordinate
            # tolerance.  UppASD inputs with ``alat`` are stored in metres,
            # so passing a 1e-10-metre cell directly makes that tolerance
            # enormous and can make an otherwise ordinary structure fail
            # symmetry detection.  Symmetry is scale invariant; normalize
            # only the temporary cell passed to seekpath and keep the actual
            # lattice for the returned Cartesian q coordinates.
            cell_scale = float(np.max(np.linalg.norm(model_or_cell.cell, axis=1), initial=0.0))
            seekpath_cell = model_or_cell.cell / cell_scale if cell_scale > 0.0 else model_or_cell.cell
            atom_type_numbers: dict[Any, int] = {}
            numbers: list[int] = []
            for site in model_or_cell.sites:
                if site.atom_type not in atom_type_numbers:
                    atom_type_numbers[site.atom_type] = len(atom_type_numbers) + 1
                numbers.append(atom_type_numbers[site.atom_type])
            structure = (
                seekpath_cell,
                scaled_positions,
                numbers,
            )
            path_data = seekpath.get_path(structure)
            point_coordinates = path_data["point_coords"]
            primitive_reciprocal = np.asarray(path_data["reciprocal_primitive_lattice"], dtype=float) / cell_scale
            vertices = []
            for start, end in path_data["path"]:
                if not vertices or vertices[-1][0] != start:
                    start_cartesian = np.asarray(point_coordinates[start], dtype=float) @ primitive_reciprocal
                    vertices.append((start, lattice.cartesian_to_fractional(start_cartesian)))
                end_cartesian = np.asarray(point_coordinates[end], dtype=float) @ primitive_reciprocal
                vertices.append((end, lattice.cartesian_to_fractional(end_cartesian)))
            return _path_from_vertices(lattice, vertices, n_per_segment=n_per_segment, source="seekpath")
        except Exception:
            # Classification is optional.  The fallback remains explicit in
            # the result so callers can report that it is not canonical.
            pass
    return _path_from_vertices(lattice, _fallback_vertices(), n_per_segment=n_per_segment, source="fallback")


@dataclass(frozen=True)
class ExchangePathData:
    path: QPath
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    hermiticity: HermiticityReport

    def as_dict(self) -> dict[str, Any]:
        return {
            "q_fractional": self.path.q_fractional.tolist(),
            "q_cartesian": self.path.q_cartesian.tolist(),
            "distance": self.path.distance.tolist(),
            "tick_positions": self.path.tick_positions.tolist(),
            "tick_labels": list(self.path.tick_labels),
            "source": self.path.source,
            "eigenvalues_real": self.eigenvalues.real.tolist(),
            "eigenvalues_imag": self.eigenvalues.imag.tolist(),
            "eigenvectors_real": self.eigenvectors.real.tolist(),
            "eigenvectors_imag": self.eigenvectors.imag.tolist(),
            "hermiticity": self.hermiticity.as_dict(),
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.as_dict(), **kwargs)


def path_exchange_data(
    model: MagneticCrystal,
    path: QPath | None = None,
    *,
    n_per_segment: int = 50,
    use_seekpath: bool = True,
) -> ExchangePathData:
    """Evaluate exchange eigenvalues along a high-symmetry path."""

    selected_path = path or high_symmetry_path(model, n_per_segment=n_per_segment, use_seekpath=use_seekpath)
    eigensystem = exchange_eigensystem(model, selected_path.q_cartesian, coordinates="cartesian")
    return ExchangePathData(selected_path, eigensystem.eigenvalues, eigensystem.eigenvectors, eigensystem.hermiticity)


def plot_exchange_path(data: ExchangePathData, ax: Any = None) -> Any:
    """Plot path eigenvalues, importing matplotlib only when requested."""

    if ax is None:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError as exc:
            raise ImportError("plot_exchange_path requires matplotlib") from exc
        _, ax = plt.subplots()
    for branch in data.eigenvalues.T:
        ax.plot(data.path.distance, branch.real)
    ax.set_xticks(data.path.distance[data.path.tick_positions], data.path.tick_labels)
    ax.set_xlabel("q-path")
    ax.set_ylabel("exchange eigenvalue")
    return ax


def exchange_heatmap_data(
    model: MagneticCrystal,
    divisions: Sequence[int] = (101, 101),
    *,
    plane: tuple[int, int] = (0, 1),
    fixed: Sequence[float] = (0.0, 0.0, 0.0),
) -> dict[str, np.ndarray]:
    """Return largest/minimum eigenvalue arrays for a reduced-coordinate 2D cut."""

    if len(divisions) != 2 or any(int(value) <= 0 for value in divisions):
        raise ValueError("divisions must contain two positive integers")
    if len(plane) != 2 or plane[0] == plane[1] or any(index not in range(3) for index in plane):
        raise ValueError("plane must contain two distinct reciprocal axes")
    fixed_q = np.asarray(fixed, dtype=float)
    if fixed_q.shape != (3,):
        raise ValueError("fixed must have length 3")
    axes = [np.arange(int(size), dtype=float) / int(size) for size in divisions]
    grid = np.zeros((len(axes[0]), len(axes[1]), 3), dtype=float)
    grid[...] = fixed_q
    grid[..., plane[0]] = axes[0][:, None]
    grid[..., plane[1]] = axes[1][None, :]
    eigensystem = exchange_eigensystem(model, grid.reshape(-1, 3), coordinates="fractional")
    leading = eigensystem.eigenvalues[:, 0].real.reshape(grid.shape[:2])
    minimum = eigensystem.eigenvalues[:, -1].real.reshape(grid.shape[:2])
    return {
        "q_fractional": grid,
        "max_eigenvalue": leading,
        "min_eigenvalue": minimum,
        "hermiticity_errors": eigensystem.hermiticity.errors.reshape(grid.shape[:2]),
    }


# Descriptive aliases for callers who prefer verb-based names.
compute_reciprocal_lattice = reciprocal_lattice
compute_jq = fourier_exchange
fourier_transform = fourier_exchange
generate_q_mesh = regular_q_mesh
compute_ordering = ordering_analysis


__all__ = [
    "ExchangeEigenSystem",
    "ExchangePathData",
    "FourierExchangeResult",
    "HermiticityReport",
    "OrderingAnalysis",
    "QPath",
    "ReciprocalLattice",
    "check_hermiticity",
    "compute_jq",
    "compute_ordering",
    "compute_reciprocal_lattice",
    "exchange_eigensystem",
    "exchange_extrema",
    "exchange_fourier",
    "exchange_heatmap_data",
    "fourier_exchange",
    "fourier_transform",
    "generate_q_mesh",
    "high_symmetry_path",
    "path_exchange_data",
    "plot_exchange_path",
    "q_mesh",
    "ordering_analysis",
    "reciprocal_lattice",
    "reciprocal_mesh",
    "regular_q_mesh",
]
