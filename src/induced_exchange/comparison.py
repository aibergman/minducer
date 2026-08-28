"""Conservative comparison diagnostics for two exchange datasets.

IMX-06 deliberately keeps comparison orchestration separate from the
reciprocal, response, downfolding, and magnon implementations.  The two
datasets may differ in their ``Jij`` values, but their crystal geometry must
be compatible before quantities indexed by basis site or reciprocal vector
are compared.

The default induced response is the explicitly labelled
``J-weighted induced-response approximation``.  None of the diagnostics in
this module establish that LKAG, frozen-magnon, or any other source is wrong;
they only report where the supplied models and responses disagree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .downfolding import DownfoldingResult, InducedExchangeDownfolding
from .induced import InducedMomentResponse
from .magnons import FMSpinWaveResult, SpinStiffnessResult, fm_magnon_spectrum, fit_spin_stiffness
from .model import MagneticCrystal
from .provenance import build_analysis_provenance
from .reciprocal import FourierExchangeResult, exchange_eigensystem, exchange_fourier, reciprocal_lattice


def _model_of(source: Any) -> MagneticCrystal:
    if isinstance(source, MagneticCrystal):
        return source
    model = getattr(source, "model", None)
    if isinstance(model, MagneticCrystal):
        return model
    raise TypeError("dataset must be a MagneticCrystal or an object exposing one as .model")


def _as_sites(values: Sequence[int] | None) -> tuple[int, ...] | None:
    if values is None:
        return None
    result = tuple(int(value) for value in values)
    if len(set(result)) != len(result):
        raise ValueError("site selections must not contain duplicates")
    return result


@dataclass(frozen=True)
class ExchangeDataset:
    """One labelled dataset and its optional explicit induced classification."""

    model: MagneticCrystal
    label: str = "Dataset"
    response: InducedMomentResponse | None = None
    robust_sites: tuple[int, ...] | None = None
    induced_sites: tuple[int, ...] | None = None
    x: float | Sequence[float] | Mapping[Any, float] | None = None
    mode: str = "j_weighted"
    kernel_model: MagneticCrystal | None = None
    neighbourhood: str | Sequence[int] | Mapping[int, Sequence[int]] = "first_shell"
    cutoff: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, MagneticCrystal):
            raise TypeError("model must be a MagneticCrystal")
        robust = _as_sites(self.robust_sites)
        induced = _as_sites(self.induced_sites)
        if robust is not None:
            object.__setattr__(self, "robust_sites", robust)
        if induced is not None:
            object.__setattr__(self, "induced_sites", induced)
        if self.response is not None:
            if self.response.model is not self.model:
                raise ValueError("response.model must be the same model as the dataset")
            if robust is not None and robust != self.response.robust_sites:
                raise ValueError("robust_sites disagrees with the supplied response")
            if induced is not None and induced != self.response.induced_sites:
                raise ValueError("induced_sites disagrees with the supplied response")

    @classmethod
    def from_model(cls, model: MagneticCrystal, *, label: str = "Dataset", **kwargs: Any) -> "ExchangeDataset":
        return cls(model=model, label=label, **kwargs)

    def induced_response(self) -> InducedMomentResponse | None:
        """Construct the configured response, or return ``None`` if unconfigured."""

        if self.response is not None:
            return self.response
        if self.robust_sites is None or self.induced_sites is None:
            return None
        return InducedMomentResponse(
            self.model,
            self.robust_sites,
            self.induced_sites,
            mode=self.mode,
            x=self.x,
            kernel_model=self.kernel_model,
            neighbourhood=self.neighbourhood,
            cutoff=self.cutoff,
        )


@dataclass(frozen=True)
class DatasetCompatibility:
    """Structural checks performed before comparison."""

    compatible: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    checks: Mapping[str, bool] = field(default_factory=dict)

    @property
    def issues(self) -> tuple[str, ...]:
        return self.errors + self.warnings

    def as_dict(self) -> dict[str, Any]:
        return {
            "compatible": self.compatible,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "checks": dict(self.checks),
        }


def validate_dataset_compatibility(
    dataset_a: MagneticCrystal | ExchangeDataset | Any,
    dataset_b: MagneticCrystal | ExchangeDataset | Any,
    *,
    atol: float = 1e-8,
    rtol: float = 1e-8,
) -> DatasetCompatibility:
    """Validate basis/cell compatibility without requiring equal exchange.

    Different moments and exchange values are expected in an A/B comparison.
    Moment differences are warnings because they affect magnon normalization,
    while cell, basis indices, and Cartesian basis positions are hard
    compatibility requirements.
    """

    a = _model_of(dataset_a.model if isinstance(dataset_a, ExchangeDataset) else dataset_a)
    b = _model_of(dataset_b.model if isinstance(dataset_b, ExchangeDataset) else dataset_b)
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}

    checks["cell"] = bool(np.allclose(a.cell, b.cell, atol=atol, rtol=rtol))
    if not checks["cell"]:
        errors.append("crystal cells are not compatible")
    indices_a, indices_b = tuple(sorted(a.site_indices)), tuple(sorted(b.site_indices))
    checks["site_indices"] = indices_a == indices_b
    if not checks["site_indices"]:
        errors.append(f"basis site indices differ: {indices_a!r} versus {indices_b!r}")
    common = sorted(set(indices_a) & set(indices_b))
    positions = [np.allclose(a.site_by_index[i].position, b.site_by_index[i].position, atol=atol, rtol=rtol) for i in common]
    checks["basis_positions"] = len(common) == len(indices_a) == len(indices_b) and all(positions)
    if not checks["basis_positions"]:
        errors.append("Cartesian basis positions are not compatible")
    checks["atom_types"] = all(a.site_by_index[i].atom_type == b.site_by_index[i].atom_type for i in common)
    if not checks["atom_types"]:
        warnings.append("atom-type labels differ for one or more common basis sites")
    moments_match = all(
        a.site_by_index[i].moment is not None
        and b.site_by_index[i].moment is not None
        and np.isclose(a.site_by_index[i].moment, b.site_by_index[i].moment, atol=atol, rtol=rtol)
        for i in common
    ) and len(common) == len(indices_a) == len(indices_b)
    checks["moments"] = moments_match
    if not moments_match:
        warnings.append("reference moment magnitudes differ or are incomplete; magnon comparisons use each dataset's own moments")
    energy_a, energy_b = a.units.energy.lower(), b.units.energy.lower()
    checks["energy_units"] = energy_a == energy_b
    if not checks["energy_units"]:
        if energy_a in {"unspecified", "unknown"} or energy_b in {"unspecified", "unknown"}:
            warnings.append("one or both energy units are unspecified; numerical Jij/magnon comparisons retain input units")
        else:
            errors.append(f"energy units differ: {a.units.energy!r} versus {b.units.energy!r}")
    return DatasetCompatibility(not errors, tuple(errors), tuple(warnings), checks)


@dataclass(frozen=True)
class OrderingSummary:
    """Largest-eigenvalue ordering diagnostic for a matrix family."""

    q_fractional: np.ndarray | None
    q_cartesian: np.ndarray | None
    eigenvalue: complex | None
    index: int | None
    kind: str
    eigenvalues: np.ndarray

    @property
    def q_order(self) -> np.ndarray | None:
        return self.q_fractional

    def as_dict(self) -> dict[str, Any]:
        return {
            "q_fractional": None if self.q_fractional is None else self.q_fractional.tolist(),
            "q_cartesian": None if self.q_cartesian is None else self.q_cartesian.tolist(),
            "eigenvalue_real": None if self.eigenvalue is None else float(np.real(self.eigenvalue)),
            "eigenvalue_imag": None if self.eigenvalue is None else float(np.imag(self.eigenvalue)),
            "index": self.index,
            "kind": self.kind,
            "eigenvalues_real": self.eigenvalues.real.tolist(),
            "eigenvalues_imag": self.eigenvalues.imag.tolist(),
        }


def _matrix_eigenvalues(matrices: np.ndarray) -> np.ndarray:
    values = np.full((len(matrices), matrices.shape[1]), np.nan + 0j, dtype=complex)
    for q_index, matrix in enumerate(matrices):
        if not np.isfinite(matrix).all():
            continue
        hermitian = np.max(np.abs(matrix - matrix.conj().T), initial=0.0) <= 1e-10 + 1e-8 * max(1.0, float(np.max(np.abs(matrix), initial=0.0)))
        current = np.linalg.eigvalsh(matrix) if hermitian else np.linalg.eigvals(matrix)
        values[q_index] = current[np.argsort(current.real)[::-1]]
    return values


def _ordering(q_fractional: np.ndarray, q_cartesian: np.ndarray, values: np.ndarray) -> OrderingSummary:
    if values.size == 0 or values.shape[1] == 0 or not np.isfinite(values[:, 0].real).any():
        return OrderingSummary(None, None, None, None, "unavailable", values)
    index = int(np.nanargmax(values[:, 0].real))
    q = q_fractional[index]
    kind = "FM at Gamma" if np.linalg.norm(q - np.rint(q)) <= 1e-8 else "AF-like/non-Gamma"
    return OrderingSummary(q, q_cartesian[index], values[index, 0], index, kind, values)


@dataclass(frozen=True)
class RealSpaceExchangeData:
    """Jij rows retained with authoritative displacement and radial shell."""

    rows: tuple[Mapping[str, Any], ...]

    @property
    def distances(self) -> np.ndarray:
        return np.asarray([row["distance"] for row in self.rows], dtype=float)

    @property
    def jij(self) -> np.ndarray:
        return np.asarray([row["jij"] for row in self.rows], dtype=float)

    def as_dict(self) -> dict[str, Any]:
        return {"rows": [dict(row) for row in self.rows]}


def _real_space_data(model: MagneticCrystal) -> RealSpaceExchangeData:
    bonds = sorted(model.exchange_bonds, key=lambda bond: (bond.distance, bond.i, bond.j, bond.displacement))
    radii = sorted({round(bond.distance, 10) for bond in bonds})
    rows = []
    for bond in bonds:
        shell = radii.index(round(bond.distance, 10)) + 1
        rows.append({
            "i": bond.i,
            "j": bond.j,
            "rx": float(bond.displacement[0]),
            "ry": float(bond.displacement[1]),
            "rz": float(bond.displacement[2]),
            "distance": float(bond.distance),
            "shell": shell,
            "jij": float(bond.jij),
        })
    return RealSpaceExchangeData(tuple(rows))


@dataclass(frozen=True)
class ExternalInducedResponse:
    """Optional DFT induced-moment comparison data."""

    moments: np.ndarray
    q: np.ndarray | None = None
    path_coordinate: np.ndarray | None = None
    source: str | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "ExternalInducedResponse":
        source = Path(path)
        rows: list[list[float]] = []
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            clean = line.split("#", 1)[0].strip()
            if not clean:
                continue
            try:
                row = [float(value) for value in clean.split()]
            except ValueError as exc:
                raise ValueError(f"{source}:{line_number}: external response contains a non-numeric row") from exc
            rows.append(row)
        if not rows:
            raise ValueError(f"{source}: external response file is empty")
        widths = {len(row) for row in rows}
        if widths == {2}:
            array = np.asarray(rows, dtype=float)
            return cls(array[:, 1], path_coordinate=array[:, 0], source=str(source))
        if widths == {4}:
            array = np.asarray(rows, dtype=float)
            return cls(array[:, 3], q=array[:, :3], source=str(source))
        raise ValueError(f"{source}: expected either 'path_coordinate m_ind' or 'qx qy qz m_ind'")

    def __post_init__(self) -> None:
        moments = np.asarray(self.moments, dtype=float)
        if moments.ndim != 1 or not np.isfinite(moments).all():
            raise ValueError("external moments must be a finite one-dimensional array")
        object.__setattr__(self, "moments", moments)
        if self.q is not None:
            q = np.asarray(self.q, dtype=float)
            if q.shape != (len(moments), 3) or not np.isfinite(q).all():
                raise ValueError("external q must have shape (n, 3)")
            object.__setattr__(self, "q", q)
        if self.path_coordinate is not None:
            coordinate = np.asarray(self.path_coordinate, dtype=float)
            if coordinate.shape != (len(moments),) or not np.isfinite(coordinate).all():
                raise ValueError("external path_coordinate must have shape (n,)")
            object.__setattr__(self, "path_coordinate", coordinate)
        if self.q is None and self.path_coordinate is None:
            raise ValueError("external response needs q coordinates or path coordinates")


@dataclass(frozen=True)
class ResponseMismatchMetrics:
    count: int
    rmse: float | None
    mae: float | None
    max_abs: float | None
    relative_rmse: float | None
    strongly_disagrees: bool

    def as_dict(self) -> dict[str, Any]:
        return {"count": self.count, "rmse": self.rmse, "mae": self.mae, "max_abs": self.max_abs, "relative_rmse": self.relative_rmse, "strongly_disagrees": self.strongly_disagrees}


@dataclass(frozen=True)
class InducedResponseComparison:
    q_fractional: np.ndarray
    q_cartesian: np.ndarray
    model_a: np.ndarray
    model_b: np.ndarray
    model_a_normalized: np.ndarray
    model_b_normalized: np.ndarray
    external: ExternalInducedResponse | None = None
    external_predicted_a: np.ndarray | None = None
    external_predicted_b: np.ndarray | None = None
    metrics_a: ResponseMismatchMetrics | None = None
    metrics_b: ResponseMismatchMetrics | None = None
    warnings: tuple[str, ...] = ()

    @property
    def response_label(self) -> str:
        return "J-weighted induced-response approximation"

    def as_dict(self) -> dict[str, Any]:
        return {
            "q_fractional": self.q_fractional.tolist(),
            "q_cartesian": self.q_cartesian.tolist(),
            "model_a": _json_array(self.model_a),
            "model_b": _json_array(self.model_b),
            "model_a_normalized": _json_array(self.model_a_normalized),
            "model_b_normalized": _json_array(self.model_b_normalized),
            "external": None if self.external is None else {
                "moments": self.external.moments.tolist(),
                "q": None if self.external.q is None else self.external.q.tolist(),
                "path_coordinate": None if self.external.path_coordinate is None else self.external.path_coordinate.tolist(),
                "source": self.external.source,
            },
            "external_predicted_a": None if self.external_predicted_a is None else self.external_predicted_a.tolist(),
            "external_predicted_b": None if self.external_predicted_b is None else self.external_predicted_b.tolist(),
            "metrics_a": None if self.metrics_a is None else self.metrics_a.as_dict(),
            "metrics_b": None if self.metrics_b is None else self.metrics_b.as_dict(),
            "warnings": list(self.warnings),
        }


def _json_array(values: Any) -> Any:
    """Serialize real arrays compactly and retain complex parts explicitly."""

    array = np.asarray(values)
    if np.iscomplexobj(array) and np.max(np.abs(array.imag), initial=0.0) > 1e-14:
        return {"real": array.real.tolist(), "imag": array.imag.tolist()}
    return array.real.tolist() if np.iscomplexobj(array) else array.tolist()


def _normalise_response_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1:
        return np.asarray(array[:, None], dtype=complex)
    if array.ndim == 2:
        return np.asarray(array, dtype=complex)
    raise ValueError("induced response must have shape (nq,) or (nq, ninduced)")


def _scalar_response(values: np.ndarray) -> np.ndarray:
    values = _normalise_response_array(values)
    if values.shape[1] == 1:
        return np.real_if_close(values[:, 0], tol=1000).real.astype(float)
    return np.linalg.norm(values, axis=1).real.astype(float)


def _path_distance(q_cartesian: np.ndarray) -> np.ndarray:
    """Return the cumulative Cartesian distance along an ordered q path."""

    q = np.asarray(q_cartesian, dtype=float)
    if len(q) <= 1:
        return np.zeros(len(q), dtype=float)
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))))


def _normalise_to_first(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=complex).copy()
    if len(result) == 0:
        return result
    reference = result[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        result = result / reference
    return result


def _response_metrics(predicted: np.ndarray, external: np.ndarray, *, threshold: float) -> ResponseMismatchMetrics:
    predicted = np.asarray(predicted, dtype=float)
    external = np.asarray(external, dtype=float)
    finite = np.isfinite(predicted) & np.isfinite(external)
    if not np.any(finite):
        return ResponseMismatchMetrics(0, None, None, None, None, False)
    error = predicted[finite] - external[finite]
    scale = max(float(np.max(np.abs(external[finite]), initial=0.0)), 1e-12)
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    maximum = float(np.max(np.abs(error), initial=0.0))
    relative = rmse / scale
    return ResponseMismatchMetrics(int(np.count_nonzero(finite)), rmse, mae, maximum, float(relative), bool(relative > threshold))


def _external_prediction(
    external: ExternalInducedResponse,
    q_cartesian: np.ndarray,
    model_values: np.ndarray,
) -> np.ndarray:
    scalar = _scalar_response(model_values)
    if external.q is not None:
        indices = [int(np.argmin(np.linalg.norm(q_cartesian - q, axis=1))) for q in external.q]
        return scalar[indices]
    if external.path_coordinate is not None:
        if len(q_cartesian) == 1:
            return np.full(len(external.path_coordinate), scalar[0])
        coordinate = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(q_cartesian, axis=0), axis=1))))
        return np.interp(external.path_coordinate, coordinate, scalar)
    raise ValueError("external response has no coordinates")


def compare_induced_response(
    response_a: InducedMomentResponse,
    response_b: InducedMomentResponse,
    q_points: Sequence[Sequence[float]] | np.ndarray,
    robust_configuration: Sequence[Any] | np.ndarray | None = None,
    *,
    coordinates: str = "fractional",
    external: ExternalInducedResponse | str | Path | None = None,
    mismatch_threshold: float = 0.1,
) -> InducedResponseComparison:
    """Predict ``m_ind(q)`` for both models and optionally compare DFT data."""

    if response_a.robust_sites != response_b.robust_sites or response_a.induced_sites != response_b.induced_sites:
        raise ValueError("response classifications must match for an induced-response comparison")
    if robust_configuration is None:
        q_array = np.asarray(q_points, dtype=float)
        count = 1 if q_array.shape == (3,) else len(q_array)
        robust_configuration = np.ones((count, len(response_a.robust_sites)), dtype=float)
    result_a = response_a.response_q(q_points, robust_configuration, coordinates=coordinates)
    result_b = response_b.response_q(q_points, robust_configuration, coordinates=coordinates)
    values_a = _normalise_response_array(result_a.induced_moments)
    values_b = _normalise_response_array(result_b.induced_moments)
    if external is not None and not isinstance(external, ExternalInducedResponse):
        external = ExternalInducedResponse.from_file(external)
    warnings = list(result_a.warnings) + list(result_b.warnings)
    prediction_a = prediction_b = metrics_a = metrics_b = None
    if external is not None:
        prediction_a = _external_prediction(external, result_a.q_cartesian, values_a)
        prediction_b = _external_prediction(external, result_b.q_cartesian, values_b)
        metrics_a = _response_metrics(prediction_a, external.moments, threshold=mismatch_threshold)
        metrics_b = _response_metrics(prediction_b, external.moments, threshold=mismatch_threshold)
        if metrics_a.strongly_disagrees or metrics_b.strongly_disagrees:
            warnings.append("The input Jij do not reproduce the supplied induced-moment response under the K=J approximation.")
    return InducedResponseComparison(
        result_a.q_fractional,
        result_a.q_cartesian,
        values_a,
        values_b,
        _normalise_to_first(values_a),
        _normalise_to_first(values_b),
        external,
        prediction_a,
        prediction_b,
        metrics_a,
        metrics_b,
        tuple(dict.fromkeys(warnings)),
    )


def predict_induced_response(
    response: InducedMomentResponse,
    q_points: Sequence[Sequence[float]] | np.ndarray,
    robust_configuration: Sequence[Any] | np.ndarray | None = None,
    *,
    coordinates: str = "fractional",
) -> Any:
    """Evaluate one selected response, defaulting to a coherent unit spiral.

    The returned :class:`InducedResponseResult` retains individual induced
    sublattices, condition numbers, and ``m_ind_q_over_m_ind_0``.
    """

    q_array = np.asarray(q_points, dtype=float)
    count = 1 if q_array.shape == (3,) else len(q_array)
    if robust_configuration is None:
        robust_configuration = np.ones((count, len(response.robust_sites)), dtype=float)
    return response.response_q(q_points, robust_configuration, coordinates=coordinates)


@dataclass(frozen=True)
class DatasetAnalysis:
    dataset: ExchangeDataset
    raw_fourier: FourierExchangeResult
    raw_eigenvalues: np.ndarray
    raw_ordering: OrderingSummary
    robust_matrices: np.ndarray | None
    robust_eigenvalues: np.ndarray | None
    robust_ordering: OrderingSummary | None
    dressed: DownfoldingResult | None
    dressed_eigenvalues: np.ndarray | None
    dressed_ordering: OrderingSummary | None
    raw_magnons: FMSpinWaveResult | None = None
    robust_magnons: FMSpinWaveResult | None = None
    dressed_magnons: FMSpinWaveResult | None = None
    raw_stiffness: SpinStiffnessResult | None = None
    robust_stiffness: SpinStiffnessResult | None = None
    dressed_stiffness: SpinStiffnessResult | None = None
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.dataset.label,
            "raw_eigenvalues_real": self.raw_eigenvalues.real.tolist(),
            "raw_eigenvalues_imag": self.raw_eigenvalues.imag.tolist(),
            "raw_ordering": self.raw_ordering.as_dict(),
            "robust_eigenvalues_real": None if self.robust_eigenvalues is None else self.robust_eigenvalues.real.tolist(),
            "robust_eigenvalues_imag": None if self.robust_eigenvalues is None else self.robust_eigenvalues.imag.tolist(),
            "robust_ordering": None if self.robust_ordering is None else self.robust_ordering.as_dict(),
            "dressed_eigenvalues_real": None if self.dressed_eigenvalues is None else self.dressed_eigenvalues.real.tolist(),
            "dressed_eigenvalues_imag": None if self.dressed_eigenvalues is None else self.dressed_eigenvalues.imag.tolist(),
            "dressed_ordering": None if self.dressed_ordering is None else self.dressed_ordering.as_dict(),
            "raw_magnons": None if self.raw_magnons is None else self.raw_magnons.as_dict(),
            "robust_magnons": None if self.robust_magnons is None else self.robust_magnons.as_dict(),
            "dressed_magnons": None if self.dressed_magnons is None else self.dressed_magnons.as_dict(),
            "raw_stiffness": None if self.raw_stiffness is None else self.raw_stiffness.as_dict(),
            "robust_stiffness": None if self.robust_stiffness is None else self.robust_stiffness.as_dict(),
            "dressed_stiffness": None if self.dressed_stiffness is None else self.dressed_stiffness.as_dict(),
            "warnings": list(self.warnings),
        }


def _subset_matrices(result: FourierExchangeResult, sites: tuple[int, ...] | None) -> np.ndarray | None:
    if sites is None:
        return None
    positions = {site: index for index, site in enumerate(result.site_indices)}
    try:
        indices = [positions[site] for site in sites]
    except KeyError as exc:
        raise ValueError(f"robust site {exc.args[0]} is absent from the exchange model") from exc
    return result.matrices[:, indices][:, :, indices]


def _analyse_dataset(
    dataset: ExchangeDataset,
    q_points: Sequence[Sequence[float]] | np.ndarray,
    *,
    coordinates: str,
    include_magnons: bool,
    stiffness_q_max: float | None,
    output_energy_unit: str | None,
) -> DatasetAnalysis:
    raw = exchange_fourier(dataset.model, q_points, coordinates=coordinates)
    raw_values = _matrix_eigenvalues(raw.matrices)
    robust = _subset_matrices(raw, dataset.robust_sites)
    robust_values = None if robust is None else _matrix_eigenvalues(robust)
    robust_order = None if robust is None else _ordering(raw.q_fractional, raw.q_cartesian, robust_values)
    warnings: list[str] = []
    dressed: DownfoldingResult | None = None
    dressed_values = None
    dressed_order = None
    response = dataset.induced_response()
    if response is not None:
        try:
            dressed = InducedExchangeDownfolding(response).evaluate(q_points, coordinates=coordinates)
            dressed_values = _matrix_eigenvalues(dressed.dressed)
            dressed_order = _ordering(dressed.q_fractional, dressed.q_cartesian, dressed_values)
            warnings.extend(dressed.warnings)
        except (ValueError, np.linalg.LinAlgError) as exc:
            warnings.append(f"dressed comparison unavailable: {exc}")
    magnons: dict[str, FMSpinWaveResult | None] = {"raw": None, "robust": None, "dressed": None}
    stiffness: dict[str, SpinStiffnessResult | None] = {"raw": None, "robust": None, "dressed": None}
    if include_magnons:
        try:
            magnons["raw"] = fm_magnon_spectrum(dataset.model, q_points, model="raw", coordinates=coordinates, output_energy_unit=output_energy_unit)
        except (ValueError, np.linalg.LinAlgError) as exc:
            warnings.append(f"raw FM magnons unavailable: {exc}")
        if dataset.robust_sites is not None:
            try:
                magnons["robust"] = fm_magnon_spectrum(dataset.model, q_points, model="robust_only", robust_sites=dataset.robust_sites, coordinates=coordinates, output_energy_unit=output_energy_unit)
            except (ValueError, np.linalg.LinAlgError) as exc:
                warnings.append(f"robust-only FM magnons unavailable: {exc}")
        if dressed is not None:
            try:
                # A result-only spectrum has no unit metadata of its own;
                # pass the source declaration explicitly for conversions.
                magnons["dressed"] = fm_magnon_spectrum(dressed, model="mryasov", output_energy_unit=output_energy_unit, input_energy_unit=dataset.model.units.energy)
            except (ValueError, np.linalg.LinAlgError) as exc:
                warnings.append(f"dressed FM magnons unavailable: {exc}")
        if stiffness_q_max is not None:
            for name, spectrum in magnons.items():
                if spectrum is not None:
                    stiffness[name] = fit_spin_stiffness(spectrum, q_max=stiffness_q_max)
    return DatasetAnalysis(
        dataset,
        raw,
        raw_values,
        _ordering(raw.q_fractional, raw.q_cartesian, raw_values),
        robust,
        robust_values,
        robust_order,
        dressed,
        dressed_values,
        dressed_order,
        magnons["raw"],
        magnons["robust"],
        magnons["dressed"],
        stiffness["raw"],
        stiffness["robust"],
        stiffness["dressed"],
        tuple(dict.fromkeys(warnings)),
    )


@dataclass(frozen=True)
class DatasetComparison:
    """All IMX-06 comparison data for datasets A and B."""

    dataset_a: DatasetAnalysis
    dataset_b: DatasetAnalysis
    compatibility: DatasetCompatibility
    induced_response: InducedResponseComparison | None
    diagnostics: tuple[str, ...]
    real_space_a: RealSpaceExchangeData
    real_space_b: RealSpaceExchangeData

    @property
    def q_fractional(self) -> np.ndarray:
        return self.dataset_a.raw_fourier.q_fractional

    @property
    def q_cartesian(self) -> np.ndarray:
        return self.dataset_a.raw_fourier.q_cartesian

    @property
    def raw_a(self) -> np.ndarray:
        return self.dataset_a.raw_eigenvalues

    @property
    def raw_b(self) -> np.ndarray:
        return self.dataset_b.raw_eigenvalues

    @property
    def robust_a(self) -> np.ndarray | None:
        return self.dataset_a.robust_eigenvalues

    @property
    def robust_b(self) -> np.ndarray | None:
        return self.dataset_b.robust_eigenvalues

    @property
    def dressed_a(self) -> np.ndarray | None:
        return self.dataset_a.dressed_eigenvalues

    @property
    def dressed_b(self) -> np.ndarray | None:
        return self.dataset_b.dressed_eigenvalues

    def as_dict(self) -> dict[str, Any]:
        def provenance(analysis: DatasetAnalysis) -> dict[str, Any]:
            response = analysis.dataset.induced_response()
            robust_sites = analysis.dataset.robust_sites or (() if response is None else response.robust_sites)
            induced_sites = analysis.dataset.induced_sites or (() if response is None else response.induced_sites)
            x_values = None
            if response is not None:
                try:
                    x_values = response._resolve_x(response.infer_x() if response._x_input is None else None)
                except (ValueError, np.linalg.LinAlgError):
                    x_values = None
            return build_analysis_provenance(
                units={"energy": analysis.dataset.model.units.energy, "length": analysis.dataset.model.units.length, "moment": analysis.dataset.model.units.moment},
                robust_sites=robust_sites,
                induced_sites=induced_sites,
                q_fractional=self.q_fractional,
                q_cartesian=self.q_cartesian,
                mode=None if response is None else response.mode,
                x_source=("not_applicable" if response is None else ("inferred from reference collinear state" if response._x_input is None else "user supplied override")),
                x_values=x_values,
                numerical_tolerances={"fourier_atol": 1e-10, "fourier_rtol": 1e-8, "ordering_gamma_tolerance": 1e-10},
            )
        return {
            "compatibility": self.compatibility.as_dict(),
            "q_fractional": self.q_fractional.tolist(),
            "q_cartesian": self.q_cartesian.tolist(),
            "analysis_provenance": {"dataset_a": provenance(self.dataset_a), "dataset_b": provenance(self.dataset_b)},
            "dataset_a": self.dataset_a.as_dict(),
            "dataset_b": self.dataset_b.as_dict(),
            "induced_response": None if self.induced_response is None else self.induced_response.as_dict(),
            "diagnostics": list(self.diagnostics),
            "real_space_a": self.real_space_a.as_dict(),
            "real_space_b": self.real_space_b.as_dict(),
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.as_dict(), **kwargs)

    def export_tables(self) -> dict[str, list[dict[str, Any]]]:
        """Return every comparison table as JSON/CSV-friendly row dictionaries."""

        tables: dict[str, list[dict[str, Any]]] = {}
        q = self.q_fractional
        for name, pair in (
            ("raw", (self.dataset_a.raw_eigenvalues, self.dataset_b.raw_eigenvalues)),
            ("robust", (self.dataset_a.robust_eigenvalues, self.dataset_b.robust_eigenvalues)),
            ("dressed", (self.dataset_a.dressed_eigenvalues, self.dataset_b.dressed_eigenvalues)),
        ):
            rows: list[dict[str, Any]] = []
            for dataset_index, values in enumerate(pair):
                if values is None:
                    continue
                label = (self.dataset_a.dataset.label, self.dataset_b.dataset.label)[dataset_index]
                for q_index in range(len(values)):
                    for branch in range(values.shape[1]):
                        rows.append({"dataset": label, "q_index": q_index, "qx": q[q_index, 0], "qy": q[q_index, 1], "qz": q[q_index, 2], "branch": branch, "eigenvalue_real": float(values[q_index, branch].real), "eigenvalue_imag": float(values[q_index, branch].imag)})
            tables[f"{name}_jq_eigenvalues"] = rows
        rows = []
        for analysis in (self.dataset_a, self.dataset_b):
            for model_name, ordering in (("raw", analysis.raw_ordering), ("robust", analysis.robust_ordering), ("dressed", analysis.dressed_ordering)):
                if ordering is not None and ordering.index is not None:
                    rows.append({"dataset": analysis.dataset.label, "model": model_name, "q_index": ordering.index, "qx": ordering.q_fractional[0], "qy": ordering.q_fractional[1], "qz": ordering.q_fractional[2], "eigenvalue_real": float(np.real(ordering.eigenvalue)), "kind": ordering.kind})
        tables["ordering"] = rows
        rows = []
        for analysis in (self.dataset_a, self.dataset_b):
            for model_name, spectrum, stiffness in (("raw", analysis.raw_magnons, analysis.raw_stiffness), ("robust", analysis.robust_magnons, analysis.robust_stiffness), ("dressed", analysis.dressed_magnons, analysis.dressed_stiffness)):
                if spectrum is None:
                    continue
                for q_index in range(len(spectrum.q_cartesian)):
                    for branch in range(spectrum.energies.shape[1]):
                        rows.append({"dataset": analysis.dataset.label, "model": model_name, "q_index": q_index, "branch": branch, "qx": spectrum.q_cartesian[q_index, 0], "qy": spectrum.q_cartesian[q_index, 1], "qz": spectrum.q_cartesian[q_index, 2], "energy_real": float(spectrum.energies[q_index, branch].real), "energy_imag": float(spectrum.energies[q_index, branch].imag), "energy_unit": spectrum.energy_unit})
                if stiffness is not None:
                    rows.append({"dataset": analysis.dataset.label, "model": f"{model_name}_stiffness", "q_index": "", "branch": stiffness.branch, "qx": "", "qy": "", "qz": "", "energy_real": stiffness.coefficient, "energy_imag": "", "energy_unit": stiffness.energy_unit})
        tables["magnons_and_stiffness"] = rows
        for suffix, real_space in (("a", self.real_space_a), ("b", self.real_space_b)):
            tables[f"real_space_{suffix}"] = [dict(row) for row in real_space.rows]
        if self.induced_response is not None:
            response = self.induced_response
            rows = []
            for q_index in range(len(response.q_fractional)):
                for dataset_name, values, normalized in ((self.dataset_a.dataset.label, response.model_a, response.model_a_normalized), (self.dataset_b.dataset.label, response.model_b, response.model_b_normalized)):
                    for induced_index in range(values.shape[1]):
                        rows.append({"dataset": dataset_name, "q_index": q_index, "qx": response.q_fractional[q_index, 0], "qy": response.q_fractional[q_index, 1], "qz": response.q_fractional[q_index, 2], "induced_index": induced_index, "m_ind_real": float(values[q_index, induced_index].real), "m_ind_imag": float(values[q_index, induced_index].imag), "m_ind_over_m0_real": float(normalized[q_index, induced_index].real), "m_ind_over_m0_imag": float(normalized[q_index, induced_index].imag)})
            tables["induced_response"] = rows
        return tables

    def export(self, directory: str | Path, *, prefix: str = "imx06") -> dict[str, Path]:
        """Write all comparison tables as CSV plus one JSON summary."""

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for name, rows in self.export_tables().items():
            path = target / f"{prefix}_{name}.csv"
            fieldnames = sorted({key for row in rows for key in row})
            with path.open("w", newline="", encoding="utf-8") as handle:
                if fieldnames:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
            written[name] = path
        summary = target / f"{prefix}_summary.json"
        summary.write_text(self.to_json(indent=2), encoding="utf-8")
        written["summary"] = summary
        return written


def compare_exchange_datasets(
    dataset_a: MagneticCrystal | ExchangeDataset | Any,
    dataset_b: MagneticCrystal | ExchangeDataset | Any,
    q_points: Sequence[Sequence[float]] | np.ndarray,
    *,
    robust_sites: Sequence[int] | None = None,
    induced_sites: Sequence[int] | None = None,
    response_a: InducedMomentResponse | None = None,
    response_b: InducedMomentResponse | None = None,
    x: float | Sequence[float] | Mapping[Any, float] | None = None,
    x_a: float | Sequence[float] | Mapping[Any, float] | None = None,
    x_b: float | Sequence[float] | Mapping[Any, float] | None = None,
    mode: str = "j_weighted",
    coordinates: str = "fractional",
    include_magnons: bool = True,
    stiffness_q_max: float | None = 0.1,
    output_energy_unit: str | None = None,
    robust_configuration: Sequence[Any] | np.ndarray | None = None,
    external_response: ExternalInducedResponse | str | Path | None = None,
    mismatch_threshold: float = 0.1,
) -> DatasetComparison:
    """Compare raw, robust-only, dressed, response, and magnon diagnostics."""

    if not isinstance(dataset_a, ExchangeDataset):
        dataset_a = ExchangeDataset(_model_of(dataset_a), label="Dataset A", response=response_a, robust_sites=_as_sites(robust_sites), induced_sites=_as_sites(induced_sites), x=x_a if x_a is not None else x, mode=mode)
    elif response_a is not None:
        raise ValueError("response_a cannot be combined with an ExchangeDataset response")
    if not isinstance(dataset_b, ExchangeDataset):
        dataset_b = ExchangeDataset(_model_of(dataset_b), label="Dataset B", response=response_b, robust_sites=_as_sites(robust_sites), induced_sites=_as_sites(induced_sites), x=x_b if x_b is not None else x, mode=mode)
    elif response_b is not None:
        raise ValueError("response_b cannot be combined with an ExchangeDataset response")
    compatibility = validate_dataset_compatibility(dataset_a, dataset_b)
    if not compatibility.compatible:
        raise ValueError("datasets are structurally incompatible: " + "; ".join(compatibility.errors))
    analysis_a = _analyse_dataset(dataset_a, q_points, coordinates=coordinates, include_magnons=include_magnons, stiffness_q_max=stiffness_q_max, output_energy_unit=output_energy_unit)
    analysis_b = _analyse_dataset(dataset_b, q_points, coordinates=coordinates, include_magnons=include_magnons, stiffness_q_max=stiffness_q_max, output_energy_unit=output_energy_unit)
    response_comparison = None
    diagnostics: list[str] = []
    for analysis in (analysis_a, analysis_b):
        diagnostics.extend(f"{analysis.dataset.label}: {warning}" for warning in analysis.warnings)
        for model_name, ordering in (("raw", analysis.raw_ordering), ("robust-only", analysis.robust_ordering), ("dressed", analysis.dressed_ordering)):
            if ordering is not None:
                diagnostics.append(f"{analysis.dataset.label} {model_name} ordering: {ordering.kind} at q = {None if ordering.q_fractional is None else ordering.q_fractional.tolist()}")
    if analysis_a.dressed_ordering is not None and analysis_a.raw_ordering.index is not None and analysis_a.dressed_ordering.index is not None:
        changed = not np.allclose(analysis_a.raw_ordering.q_fractional - analysis_a.dressed_ordering.q_fractional, np.rint(analysis_a.raw_ordering.q_fractional - analysis_a.dressed_ordering.q_fractional), atol=1e-8, rtol=0.0)
        diagnostics.append(f"Dressing changes ordering for Dataset A: {'yes' if changed else 'no'}.")
    if analysis_b.dressed_ordering is not None and analysis_b.raw_ordering.index is not None and analysis_b.dressed_ordering.index is not None:
        changed = not np.allclose(analysis_b.raw_ordering.q_fractional - analysis_b.dressed_ordering.q_fractional, np.rint(analysis_b.raw_ordering.q_fractional - analysis_b.dressed_ordering.q_fractional), atol=1e-8, rtol=0.0)
        diagnostics.append(f"Dressing changes ordering for Dataset B: {'yes' if changed else 'no'}.")
    if dataset_a.induced_response() is not None and dataset_b.induced_response() is not None:
        response_comparison = compare_induced_response(dataset_a.induced_response(), dataset_b.induced_response(), q_points, robust_configuration, coordinates=coordinates, external=external_response, mismatch_threshold=mismatch_threshold)
        diagnostics.extend(response_comparison.warnings)
    elif external_response is not None:
        diagnostics.append("external induced-response comparison unavailable because both datasets need an explicit induced classification")
    return DatasetComparison(analysis_a, analysis_b, compatibility, response_comparison, tuple(dict.fromkeys(diagnostics)), _real_space_data(dataset_a.model), _real_space_data(dataset_b.model))


def _observable_values(analysis: DatasetAnalysis, observable: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    key = observable.lower().replace("-", "_").replace(" ", "_")
    if key in {"raw", "raw_jq", "raw_eigenvalues"}:
        return analysis.raw_eigenvalues, None
    if key in {"robust", "robust_only", "robust_jq", "robust_eigenvalues"}:
        return analysis.robust_eigenvalues, None
    if key in {"dressed", "effective", "jeff", "dressed_jq"}:
        return analysis.dressed_eigenvalues, None
    if key in {"magnon", "magnons", "raw_magnons"}:
        return None if analysis.raw_magnons is None else analysis.raw_magnons.energies, None
    raise ValueError("observable must be raw, robust, dressed, or magnons")


def plot_comparison(result: DatasetComparison, *, observable: str = "raw", ax: Any | None = None) -> Any:
    """Plot one comparison observable for A and B, with lazy matplotlib import.

    ``observable='magnons'`` overlays raw, robust-only, and dressed spectra
    for each dataset when those calculations are available.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plot_comparison requires matplotlib") from exc
    if ax is None:
        _, ax = plt.subplots()
    if observable.lower() in {"magnon", "magnons", "spin_waves", "spinwaves"}:
        for analysis in (result.dataset_a, result.dataset_b):
            for model_name, spectrum in (("raw", analysis.raw_magnons), ("robust-only", analysis.robust_magnons), ("dressed", analysis.dressed_magnons)):
                if spectrum is None:
                    continue
                x = _path_distance(spectrum.q_cartesian)
                for branch in range(spectrum.energies.shape[1]):
                    ax.plot(x, spectrum.energies[:, branch].real, label=f"{analysis.dataset.label} {model_name} branch {branch + 1}")
        ax.set_xlabel("q-path")
        ax.set_ylabel("magnon energy")
        ax.set_title("magnon comparison")
        ax.legend()
        return ax
    values_a, _ = _observable_values(result.dataset_a, observable)
    values_b, _ = _observable_values(result.dataset_b, observable)
    if values_a is None or values_b is None:
        raise ValueError(f"{observable} data are unavailable")
    x = _path_distance(result.q_cartesian)
    for values, label in ((values_a, result.dataset_a.dataset.label), (values_b, result.dataset_b.dataset.label)):
        for branch in range(values.shape[1]):
            ax.plot(x, values[:, branch].real, label=f"{label} branch {branch + 1}")
    ax.set_xlabel("q-path")
    ax.set_ylabel("magnon energy" if "magnon" in observable.lower() else "J(q) eigenvalue")
    ax.set_title(f"{observable} comparison")
    ax.legend()
    return ax


def plot_stiffness_comparison(result: DatasetComparison, *, ax: Any | None = None) -> Any:
    """Plot the near-Gamma stiffness fit points and fitted curves."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plot_stiffness_comparison requires matplotlib") from exc
    if ax is None:
        _, ax = plt.subplots()
    for analysis in (result.dataset_a, result.dataset_b):
        for model_name, stiffness in (("raw", analysis.raw_stiffness), ("robust-only", analysis.robust_stiffness), ("dressed", analysis.dressed_stiffness)):
            if stiffness is None or stiffness.coefficient is None:
                continue
            radius_squared = np.sum(stiffness.q_cartesian**2, axis=1)
            ax.scatter(radius_squared, stiffness.energies.real, label=f"{analysis.dataset.label} {model_name}")
            x = np.linspace(0.0, max(float(np.max(radius_squared, initial=0.0)), stiffness.q_max**2), 50)
            ax.plot(x, stiffness.coefficient * x, alpha=0.6)
    ax.set_xlabel("|q|²")
    ax.set_ylabel("magnon energy")
    ax.set_title("spin stiffness comparison")
    ax.legend()
    return ax


def plot_real_space_comparison(result: DatasetComparison, *, ax: Any | None = None) -> Any:
    """Plot raw Jij against authoritative bond distance for A and B."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plot_real_space_comparison requires matplotlib") from exc
    if ax is None:
        _, ax = plt.subplots()
    for data, label in ((result.real_space_a, result.dataset_a.dataset.label), (result.real_space_b, result.dataset_b.dataset.label)):
        ax.scatter(data.distances, data.jij, label=label)
    ax.set_xlabel("bond distance")
    ax.set_ylabel("Jij")
    ax.set_title("real-space exchange comparison")
    ax.legend()
    return ax


def plot_induced_response_comparison(result: DatasetComparison, *, ax: Any | None = None) -> Any:
    """Plot normalized model induced response and optional external data."""

    if result.induced_response is None:
        raise ValueError("induced-response comparison is unavailable")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plot_induced_response_comparison requires matplotlib") from exc
    if ax is None:
        _, ax = plt.subplots()
    response = result.induced_response
    x = _path_distance(response.q_cartesian)
    ax.plot(x, _scalar_response(response.model_a_normalized), label=f"{result.dataset_a.dataset.label} model")
    ax.plot(x, _scalar_response(response.model_b_normalized), label=f"{result.dataset_b.dataset.label} model")
    if response.external is not None:
        external = response.external.moments
        reference = external[0] if len(external) else 1.0
        ax.plot(np.linspace(float(x[0]), float(x[-1]), len(external)), external / reference, "o", label="DFT response")
    ax.set_xlabel("q-path")
    ax.set_ylabel("m_ind(q) / m_ind(Γ)")
    ax.legend()
    return ax


def export_comparison_tables(result: DatasetComparison, directory: str | Path, *, prefix: str = "imx06") -> dict[str, Path]:
    return result.export(directory, prefix=prefix)


compare_datasets = compare_exchange_datasets
dataset_compatibility = validate_dataset_compatibility
load_external_induced_response = ExternalInducedResponse.from_file
plot_exchange_comparison = plot_comparison
export_comparison_csv = export_comparison_tables


__all__ = [
    "DatasetAnalysis",
    "DatasetComparison",
    "DatasetCompatibility",
    "ExchangeDataset",
    "ExternalInducedResponse",
    "InducedResponseComparison",
    "OrderingSummary",
    "RealSpaceExchangeData",
    "ResponseMismatchMetrics",
    "compare_datasets",
    "compare_exchange_datasets",
    "compare_induced_response",
    "dataset_compatibility",
    "export_comparison_csv",
    "export_comparison_tables",
    "load_external_induced_response",
    "plot_comparison",
    "plot_exchange_comparison",
    "plot_induced_response_comparison",
    "plot_real_space_comparison",
    "plot_stiffness_comparison",
    "predict_induced_response",
    "validate_dataset_compatibility",
]
