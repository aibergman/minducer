"""Collinear-ferromagnetic adiabatic magnon spectra.

Only a stable collinear FM reference is treated as a physical spin-wave
problem here.  The fixed convention is

    H = -1/2 sum_ij J_ij e_i dot e_j.

For transverse complex amplitudes ``u_a`` around ``e_a = z`` the quadratic
energy is ``1/2 u^dagger A(q) u`` with

    A(q) = diag(J(0) 1) - J(q).

The classical magnetic-moment equation is

    hbar d e_a/dt = -g e_a x dH/d(mu_a e_a),

where ``mu_a`` is the numerical moment in mu_B.  Therefore the energy-valued
FM dynamical matrix is

    D(q) = g^2 M^(-1/2) A(q) M^(-1/2).

UppASD's AMS implementation applies the product of the site Landé factors to
each dynamical-matrix element.  For the global ``g_factor`` supported here,
that is ``g_factor**2``; its default is UppASD's conventional value 2.0.  The
mu_B factors cancel because both the supplied moments and the magnetic moment
in the equation are expressed in mu_B.  The eigenvalues of ``D`` are
hbar*omega in the exchange-energy unit.

Induced sites never enter this dynamical basis.  A Polesya/slave calculation
uses the dressed robust matrix after analytical elimination, so its physical
low-energy spectrum is directly comparable to the Mryasov/downfolded one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from typing import Any

import numpy as np

from .downfolding import DownfoldingResult, InducedExchangeDownfolding
from .induced import InducedMomentResponse
from .model import MagneticCrystal
from .reciprocal import FourierExchangeResult, HermiticityReport, check_hermiticity, exchange_fourier
from .units import energy_conversion_factor, normalise_energy_unit


_MODEL_NAMES = {
    "raw": "raw all-rigid",
    "all_rigid": "raw all-rigid",
    "raw_all_rigid": "raw all-rigid",
    "robust_only": "robust-only raw",
    "robust_raw": "robust-only raw",
    "mryasov": "Mryasov-like downfolded robust",
    "downfolded": "Mryasov-like downfolded robust",
    "polesya": "Polesya-like slave, induced variables eliminated",
    "slave": "Polesya-like slave, induced variables eliminated",
}


def _normalise_model_name(name: str) -> str:
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in _MODEL_NAMES:
        raise ValueError("model must be 'raw', 'robust_only', 'mryasov', or 'polesya'")
    if key in {"all_rigid", "raw_all_rigid"}:
        return "raw"
    if key in {"robust_only", "robust_raw"}:
        return "robust_only"
    if key == "downfolded":
        return "mryasov"
    if key == "slave":
        return "polesya"
    return key


def _as_q_points(values: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    q = np.asarray(values, dtype=float)
    if q.shape == (3,):
        q = q.reshape(1, 3)
    if q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all():
        raise ValueError("q_points must have shape (3,) or (n, 3) and contain finite values")
    return q


def _as_moments(values: Sequence[float] | np.ndarray, n: int) -> np.ndarray:
    moments = np.asarray(values, dtype=float)
    if moments.shape != (n,) or not np.isfinite(moments).all():
        raise ValueError(f"moment_magnitudes must have shape ({n},) and contain finite values")
    if np.any(moments <= 0):
        raise ValueError("all dynamical moment magnitudes must be positive")
    return moments


def fm_harmonic_matrix(
    exchange_matrix: np.ndarray,
    exchange_at_gamma: np.ndarray,
) -> np.ndarray:
    """Construct ``A(q) = diag(J(0) 1) - J(q)`` for one q point."""

    matrix = np.asarray(exchange_matrix, dtype=complex)
    gamma = np.asarray(exchange_at_gamma, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("exchange_matrix must be square")
    if gamma.shape != matrix.shape:
        raise ValueError("exchange_at_gamma must have the same square shape as exchange_matrix")
    if not np.isfinite(matrix).all() or not np.isfinite(gamma).all():
        raise ValueError("exchange matrices contain non-finite values")
    row_sums = gamma @ np.ones(matrix.shape[0], dtype=complex)
    return np.diag(row_sums) - matrix


def fm_dynamical_matrix(
    exchange_matrix: np.ndarray,
    exchange_at_gamma: np.ndarray,
    moment_magnitudes: Sequence[float] | np.ndarray,
    *,
    g_factor: float = 2.0,
    energy_scale: float = 1.0,
) -> np.ndarray:
    """Return the energy-valued, moment-normalized FM dynamical matrix.

    The result is ``g_factor**2 * energy_scale * M^-1/2 A M^-1/2``.  This
    matches UppASD AMS, which applies the product of the site Landé factors to
    each matrix element.  The symmetric normalization is similar to the
    direct LLG matrix ``g M^-1 A`` but is Hermitian whenever the exchange input
    is Hermitian.
    """

    matrix = np.asarray(exchange_matrix, dtype=complex)
    moments = _as_moments(moment_magnitudes, matrix.shape[0])
    if not np.isfinite(g_factor) or g_factor <= 0:
        raise ValueError("g_factor must be finite and positive")
    if not np.isfinite(energy_scale) or energy_scale <= 0:
        raise ValueError("energy_scale must be finite and positive")
    harmonic = fm_harmonic_matrix(matrix, exchange_at_gamma)
    inv_sqrt = np.diag(1.0 / np.sqrt(moments))
    return float(g_factor * g_factor * energy_scale) * (inv_sqrt @ harmonic @ inv_sqrt)


@dataclass(frozen=True)
class SpinStiffnessResult:
    """Fit of the acoustic energy to ``E = D |q|^2`` near Gamma."""

    coefficient: float | None
    q_max: float
    branch: int
    point_count: int
    residual_rms: float | None
    r_squared: float | None
    q_cartesian: np.ndarray
    energies: np.ndarray
    warnings: tuple[str, ...] = ()
    energy_unit: str = "unspecified"

    @property
    def D(self) -> float | None:
        return self.coefficient

    def as_dict(self) -> dict[str, Any]:
        return {
            "coefficient": self.coefficient,
            "q_max": self.q_max,
            "branch": self.branch,
            "point_count": self.point_count,
            "residual_rms": self.residual_rms,
            "r_squared": self.r_squared,
            "q_cartesian": self.q_cartesian.tolist(),
            "energies": self.energies.tolist(),
            "energy_unit": self.energy_unit,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class FMSpinWaveResult:
    """Numerical FM spectrum plus explicit stability and unit diagnostics."""

    q_fractional: np.ndarray
    q_cartesian: np.ndarray
    exchange_matrices: np.ndarray
    harmonic_matrices: np.ndarray
    dynamical_matrices: np.ndarray
    energies: np.ndarray
    site_indices: tuple[int, ...]
    moment_magnitudes: np.ndarray
    model: str
    model_label: str
    g_factor: float
    energy_unit: str
    candidate_order_index: int | None
    candidate_order_fractional: np.ndarray | None
    gamma_index: int | None
    fm_compatible: bool
    stable: bool
    goldstone_error: float | None
    goldstone_ok: bool | None
    unstable_indices: tuple[tuple[int, int], ...]
    exchange_hermiticity: HermiticityReport
    dynamical_hermiticity: HermiticityReport
    warnings: tuple[str, ...] = ()
    stiffness: SpinStiffnessResult | None = None

    @property
    def eigenvalues(self) -> np.ndarray:
        return self.energies

    @property
    def is_stable(self) -> bool:
        return self.stable

    @property
    def is_fm_compatible(self) -> bool:
        return self.fm_compatible

    @property
    def q_order_fractional(self) -> np.ndarray | None:
        return self.candidate_order_fractional

    @property
    def magnon_energies(self) -> np.ndarray:
        return self.energies

    @property
    def acoustic_branch(self) -> np.ndarray:
        return self.energies[:, 0]

    @property
    def optical_branches(self) -> np.ndarray:
        return self.energies[:, 1:]

    @property
    def signed_harmonic_eigenvalues(self) -> np.ndarray:
        """Signed branches, including negative FM-instability modes."""

        return self.energies

    @property
    def negative_modes(self) -> tuple[tuple[int, int], ...]:
        return self.unstable_indices

    def as_dict(self) -> dict[str, Any]:
        return {
            "q_fractional": self.q_fractional.tolist(),
            "q_cartesian": self.q_cartesian.tolist(),
            "exchange_real": self.exchange_matrices.real.tolist(),
            "exchange_imag": self.exchange_matrices.imag.tolist(),
            "harmonic_real": self.harmonic_matrices.real.tolist(),
            "harmonic_imag": self.harmonic_matrices.imag.tolist(),
            "dynamical_real": self.dynamical_matrices.real.tolist(),
            "dynamical_imag": self.dynamical_matrices.imag.tolist(),
            "energies_real": self.energies.real.tolist(),
            "energies_imag": self.energies.imag.tolist(),
            "site_indices": list(self.site_indices),
            "moment_magnitudes": self.moment_magnitudes.tolist(),
            "model": self.model,
            "model_label": self.model_label,
            "g_factor": self.g_factor,
            "energy_unit": self.energy_unit,
            "candidate_order_index": self.candidate_order_index,
            "candidate_order_fractional": None if self.candidate_order_fractional is None else self.candidate_order_fractional.tolist(),
            "gamma_index": self.gamma_index,
            "fm_compatible": self.fm_compatible,
            "stable": self.stable,
            "goldstone_error": self.goldstone_error,
            "goldstone_ok": self.goldstone_ok,
            "unstable_indices": [list(item) for item in self.unstable_indices],
            "exchange_hermiticity": self.exchange_hermiticity.as_dict(),
            "dynamical_hermiticity": self.dynamical_hermiticity.as_dict(),
            "warnings": list(self.warnings),
            "stiffness": None if self.stiffness is None else self.stiffness.as_dict(),
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.as_dict(), **kwargs)


def magnon_path_data(
    result: FMSpinWaveResult,
    *,
    tick_indices: Sequence[int] | None = None,
    tick_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return plotting-ready cumulative path distances and branch energies."""

    if len(result.q_cartesian) == 0:
        distances = np.empty(0, dtype=float)
    else:
        distances = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(result.q_cartesian, axis=0), axis=1))))
    if tick_indices is None:
        indices = np.asarray([], dtype=int)
    else:
        indices = np.asarray(tuple(int(value) for value in tick_indices), dtype=int)
        if np.any(indices < 0) or np.any(indices >= len(result.q_cartesian)):
            raise ValueError("tick_indices must refer to q points in the result")
    if tick_labels is not None and len(tick_labels) != len(indices):
        raise ValueError("tick_labels must have the same length as tick_indices")
    return {
        "distance": distances,
        "q_fractional": result.q_fractional.copy(),
        "q_cartesian": result.q_cartesian.copy(),
        "energies": result.energies.copy(),
        "energy_unit": result.energy_unit,
        "branch_count": result.energies.shape[1],
        "tick_indices": indices,
        "tick_distances": distances[indices],
        "tick_labels": tuple(tick_labels or ()),
    }


def plot_magnon_path(
    result: FMSpinWaveResult,
    *,
    ax: Any | None = None,
    tick_indices: Sequence[int] | None = None,
    tick_labels: Sequence[str] | None = None,
    **plot_kwargs: Any,
) -> Any:
    """Plot all FM branches, importing matplotlib only when plotting is used."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError("plot_magnon_path requires matplotlib") from exc
    data = magnon_path_data(result, tick_indices=tick_indices, tick_labels=tick_labels)
    if ax is None:
        _, ax = plt.subplots()
    for branch in range(result.energies.shape[1]):
        ax.plot(data["distance"], result.energies[:, branch].real, **plot_kwargs)
    if len(data["tick_indices"]):
        if len(data["tick_labels"]):
            ax.set_xticks(data["tick_distances"], data["tick_labels"])
        else:
            ax.set_xticks(data["tick_distances"])
    ax.set_xlabel("q-path")
    ax.set_ylabel(f"magnon energy ({result.energy_unit})")
    return ax


def fit_spin_stiffness(
    result: FMSpinWaveResult,
    *,
    q_max: float = 0.1,
    branch: int = 0,
    min_points: int = 3,
) -> SpinStiffnessResult:
    """Fit ``E_branch(q)`` to ``D |q_cart|^2`` through the origin.

    ``q_max`` is in Cartesian reciprocal-length units and is stored in the
    returned object so a UI can expose the fitting interval directly.
    """

    if q_max <= 0 or not np.isfinite(q_max):
        raise ValueError("q_max must be finite and positive")
    if branch < 0 or branch >= result.energies.shape[1]:
        raise ValueError("branch is outside the available spectrum")
    radius = np.linalg.norm(result.q_cartesian, axis=1)
    selected = (radius > 1e-12) & (radius <= q_max + 1e-14)
    q_values = result.q_cartesian[selected]
    energies = result.energies[selected, branch]
    warnings: list[str] = []
    real_energies = np.asarray(energies.real, dtype=float)
    x = radius[selected] ** 2
    finite = np.isfinite(x) & np.isfinite(real_energies)
    q_values = q_values[finite]
    energies = energies[finite]
    x = x[finite]
    real_energies = real_energies[finite]
    if len(x) < min_points:
        warnings.append(f"only {len(x)} q point(s) lie in the visible stiffness interval; need at least {min_points}")
        return SpinStiffnessResult(None, float(q_max), branch, len(x), None, None, q_values, energies, tuple(warnings), result.energy_unit)
    denominator = float(np.dot(x, x))
    if denominator <= 0:
        warnings.append("stiffness fit has zero q^2 span")
        return SpinStiffnessResult(None, float(q_max), branch, len(x), None, None, q_values, energies, tuple(warnings), result.energy_unit)
    coefficient = float(np.dot(x, real_energies) / denominator)
    residual = real_energies - coefficient * x
    rms = float(np.sqrt(np.mean(residual**2)))
    total = float(np.sum((real_energies - np.mean(real_energies)) ** 2))
    r_squared = None if total <= 1e-30 else float(1.0 - np.sum(residual**2) / total)
    if np.any(real_energies < -1e-10):
        warnings.append("negative energies occur in the stiffness interval; the FM reference is unstable there")
    return SpinStiffnessResult(coefficient, float(q_max), branch, len(x), rms, r_squared, q_values, energies, tuple(warnings), result.energy_unit)


def _subset_result(transformed: FourierExchangeResult, sites: tuple[int, ...]) -> tuple[np.ndarray, HermiticityReport]:
    position = {site: index for index, site in enumerate(transformed.site_indices)}
    try:
        indices = [position[site] for site in sites]
    except KeyError as exc:
        raise ValueError(f"site {exc.args[0]} is not present in the exchange model") from exc
    matrices = transformed.matrices[:, indices][:, :, indices]
    return matrices, check_hermiticity(matrices, atol=transformed.hermiticity.tolerance, rtol=0.0)


def _gamma_index(q_fractional: np.ndarray) -> int | None:
    candidates = np.flatnonzero(np.linalg.norm(q_fractional - np.rint(q_fractional), axis=1) <= 1e-10)
    return None if len(candidates) == 0 else int(candidates[0])


def _spectrum_from_matrices(
    q_fractional: np.ndarray,
    q_cartesian: np.ndarray,
    exchange_matrices: np.ndarray,
    exchange_at_gamma: np.ndarray,
    sites: tuple[int, ...],
    moments: np.ndarray,
    *,
    model: str,
    model_label: str,
    g_factor: float,
    energy_scale: float,
    energy_unit: str,
    exchange_hermiticity: HermiticityReport,
    warnings: Sequence[str] = (),
    stiffness_q_max: float | None = None,
    stiffness_min_points: int = 3,
) -> FMSpinWaveResult:
    if exchange_matrices.ndim != 3 or exchange_matrices.shape[0] == 0:
        raise ValueError("at least one q point is required for a magnon spectrum")
    if exchange_matrices.shape[1:] != (len(sites), len(sites)):
        raise ValueError("exchange matrices do not match the selected dynamical sites")
    harmonic = np.stack([fm_harmonic_matrix(matrix, exchange_at_gamma) for matrix in exchange_matrices])
    dynamical = np.stack([
        fm_dynamical_matrix(matrix, exchange_at_gamma, moments, g_factor=g_factor, energy_scale=energy_scale)
        for matrix in exchange_matrices
    ])
    dyn_report = check_hermiticity(dynamical)
    energies = np.empty((len(dynamical), len(sites)), dtype=complex)
    for index, matrix in enumerate(dynamical):
        if dyn_report.errors[index] <= dyn_report.tolerance:
            values = np.linalg.eigvalsh(matrix).astype(complex)
        else:
            values = np.linalg.eigvals(matrix)
        energies[index] = values[np.argsort(values.real)]

    exchange_values = np.empty((len(exchange_matrices), len(sites)), dtype=complex)
    for index, matrix in enumerate(exchange_matrices):
        if exchange_hermiticity.errors[index] <= exchange_hermiticity.tolerance:
            values = np.linalg.eigvalsh(matrix).astype(complex)
        else:
            values = np.linalg.eigvals(matrix)
        exchange_values[index] = values[np.argsort(values.real)[::-1]]
    finite_order = np.isfinite(exchange_values[:, 0].real)
    candidate = None if not np.any(finite_order) else int(np.nanargmax(exchange_values[:, 0].real))
    candidate_q = None if candidate is None else q_fractional[candidate]
    gamma = _gamma_index(q_fractional)
    fm_compatible = candidate is not None and gamma is not None and candidate == gamma
    report_warnings = list(warnings)
    if gamma is None:
        report_warnings.append("q-point set does not contain Gamma; stable FM compatibility and Goldstone behavior cannot be certified")
    elif candidate is not None and candidate != gamma:
        report_warnings.append(
            "candidate ordering vector is not Gamma; signed harmonic eigenvalues are shown, but this is not labelled a stable FM magnon spectrum"
        )
    if not exchange_hermiticity.is_hermitian:
        report_warnings.append("exchange input is non-Hermitian; signed complex harmonic diagnostics are not a certified physical spectrum")

    scale = max(1.0, float(np.max(np.abs(dynamical), initial=0.0)))
    negative = np.argwhere(energies.real < -(1e-10 + 1e-8 * scale))
    unstable = tuple((int(q), int(branch)) for q, branch in negative)
    if unstable:
        report_warnings.append(f"{len(unstable)} negative dynamical eigenvalue(s) indicate an unstable FM reference")
    goldstone_error: float | None = None
    goldstone_ok: bool | None = None
    if gamma is not None:
        goldstone_error = float(np.min(np.abs(energies[gamma])))
        goldstone_ok = bool(goldstone_error <= 1e-8 * scale + 1e-10)
        if not goldstone_ok:
            report_warnings.append(f"Gamma has no tight Goldstone zero mode (minimum absolute energy {goldstone_error:.6g})")
    stable = bool(fm_compatible and not unstable and exchange_hermiticity.is_hermitian and (goldstone_ok is not False))
    result = FMSpinWaveResult(
        q_fractional=q_fractional,
        q_cartesian=q_cartesian,
        exchange_matrices=exchange_matrices,
        harmonic_matrices=harmonic,
        dynamical_matrices=dynamical,
        energies=energies,
        site_indices=sites,
        moment_magnitudes=moments,
        model=model,
        model_label=model_label,
        g_factor=float(g_factor),
        energy_unit=energy_unit,
        candidate_order_index=candidate,
        candidate_order_fractional=candidate_q,
        gamma_index=gamma,
        fm_compatible=fm_compatible,
        stable=stable,
        goldstone_error=goldstone_error,
        goldstone_ok=goldstone_ok,
        unstable_indices=unstable,
        exchange_hermiticity=exchange_hermiticity,
        dynamical_hermiticity=dyn_report,
        warnings=tuple(dict.fromkeys(report_warnings)),
    )
    if stiffness_q_max is not None:
        stiffness = fit_spin_stiffness(result, q_max=stiffness_q_max, min_points=stiffness_min_points)
        result = FMSpinWaveResult(**{**result.__dict__, "stiffness": stiffness})
    return result


def _model_moments(model: MagneticCrystal, sites: tuple[int, ...]) -> np.ndarray:
    values = []
    for site in sites:
        moment = model.site_by_index[site].moment
        if moment is None:
            raise ValueError(f"site {site} has no reference moment")
        values.append(moment)
    return _as_moments(values, len(sites))


def _evaluate_downfolding(
    source: InducedExchangeDownfolding | InducedMomentResponse | DownfoldingResult,
    q_points: Sequence[Sequence[float]] | np.ndarray | None,
    coordinates: str,
) -> tuple[DownfoldingResult, MagneticCrystal | None]:
    if isinstance(source, DownfoldingResult):
        if q_points is not None:
            raise ValueError("q_points cannot be supplied with an already evaluated DownfoldingResult")
        return source, None
    if isinstance(source, InducedMomentResponse):
        source = InducedExchangeDownfolding(source)
    if not isinstance(source, InducedExchangeDownfolding):
        raise TypeError("downfolding source must be a response, downfolding object, or DownfoldingResult")
    if q_points is None:
        raise ValueError("q_points are required when evaluating a downfolding object")
    return source.evaluate(q_points, coordinates=coordinates), source.response.model


def fm_magnon_spectrum(
    source: MagneticCrystal | InducedExchangeDownfolding | InducedMomentResponse | DownfoldingResult,
    q_points: Sequence[Sequence[float]] | np.ndarray | None = None,
    *,
    model: str = "raw",
    robust_sites: Sequence[int] | None = None,
    induced_sites: Sequence[int] | None = None,
    moment_magnitudes: Sequence[float] | np.ndarray | None = None,
    coordinates: str = "fractional",
    x: float | Sequence[float] | dict[Any, float] | None = None,
    kernel_model: MagneticCrystal | None = None,
    neighbourhood: str | Sequence[int] | dict[int, Sequence[int]] = "first_shell",
    cutoff: float | None = None,
    g_factor: float = 2.0,
    input_energy_unit: str | None = None,
    output_energy_unit: str | None = None,
    stiffness_q_max: float | None = None,
    stiffness_min_points: int = 3,
) -> FMSpinWaveResult:
    """Calculate a moment-normalized collinear-FM spectrum.

    ``model='raw'`` retains every site as a rigid dynamical moment.
    ``model='robust_only'`` retains only explicitly supplied robust sites.
    ``model='mryasov'`` and ``model='polesya'`` both use the robust dressed
    interaction after induced variables have been eliminated; the latter name
    records the slave-response representation, not extra slave branches.
    """

    selected_model = _normalise_model_name(model)
    source_model: MagneticCrystal | None = source if isinstance(source, MagneticCrystal) else None
    downfolded: DownfoldingResult | None = None
    if selected_model in {"mryasov", "polesya"}:
        if isinstance(source, MagneticCrystal):
            if robust_sites is None or induced_sites is None:
                raise ValueError("robust_sites and induced_sites are required for a dressed spectrum from a MagneticCrystal")
            source = InducedExchangeDownfolding(
                InducedMomentResponse(
                    source,
                    robust_sites,
                    induced_sites,
                    x=x,
                    kernel_model=kernel_model,
                    neighbourhood=neighbourhood,
                    cutoff=cutoff,
                )
            )
        downfolded, source_model = _evaluate_downfolding(source, q_points, coordinates)
        q_fractional = downfolded.q_fractional
        q_cartesian = downfolded.q_cartesian
        matrices = downfolded.dressed
        sites = tuple(downfolded.robust_sites)
        exchange_report = downfolded.dressed_hermiticity
        if source_model is not None:
            moments = _model_moments(source_model, sites)
        elif moment_magnitudes is None:
            raise ValueError("moment_magnitudes are required when source is a DownfoldingResult")
        else:
            moments = _as_moments(moment_magnitudes, len(sites))
        # A result-only path must contain Gamma because no independent source
        # object is available from which to evaluate J_eff(0).
        gamma = _gamma_index(q_fractional)
        if gamma is None:
            raise ValueError("a DownfoldingResult must contain Gamma to construct the FM reference field")
        gamma_matrix = matrices[gamma]
        warnings = list(downfolded.warnings)
        label = _MODEL_NAMES[selected_model]
    else:
        if not isinstance(source, MagneticCrystal):
            if isinstance(source, (InducedExchangeDownfolding, InducedMomentResponse, DownfoldingResult)):
                if isinstance(source, DownfoldingResult):
                    raise ValueError("raw/robust-only spectra from DownfoldingResult require the original MagneticCrystal")
                source_model = source.response.model if isinstance(source, InducedExchangeDownfolding) else source.model
            else:
                raise TypeError("source must be a MagneticCrystal or downfolding source")
        if source_model is None:
            raise TypeError("source must be a MagneticCrystal")
        if q_points is None:
            raise ValueError("q_points are required for raw and robust-only spectra")
        q = _as_q_points(q_points)
        transformed = exchange_fourier(source_model, q, coordinates=coordinates)
        if selected_model == "raw":
            sites = tuple(sorted(source_model.site_indices))
            matrices = transformed.matrices
            exchange_report = transformed.hermiticity
        else:
            if robust_sites is None:
                raise ValueError("robust_sites are required for a robust-only raw spectrum")
            sites = tuple(int(site) for site in robust_sites)
            matrices, exchange_report = _subset_result(transformed, sites)
        q_fractional = transformed.q_fractional
        q_cartesian = transformed.q_cartesian
        gamma_q = np.zeros((1, 3), dtype=float)
        gamma_transformed = exchange_fourier(source_model, gamma_q, coordinates="fractional")
        if selected_model == "raw":
            gamma_matrix = gamma_transformed.matrices[0]
        else:
            gamma_matrix, _ = _subset_result(gamma_transformed, sites)
            gamma_matrix = gamma_matrix[0]
        moments = _model_moments(source_model, sites) if moment_magnitudes is None else _as_moments(moment_magnitudes, len(sites))
        warnings = []
        label = _MODEL_NAMES[selected_model]

    declared_input = input_energy_unit or (source_model.units.energy if source_model is not None else "unspecified")
    declared_input = normalise_energy_unit(declared_input)
    if output_energy_unit is None:
        output_unit = declared_input
        energy_scale = 1.0
    else:
        output_unit = normalise_energy_unit(output_energy_unit)
        if declared_input == "unspecified":
            raise ValueError("input energy unit is unspecified; pass input_energy_unit before requesting conversion")
        energy_scale = energy_conversion_factor(declared_input, output_unit)
    if output_unit == "unspecified":
        output_unit = "unspecified"
    return _spectrum_from_matrices(
        q_fractional,
        q_cartesian,
        np.asarray(matrices, dtype=complex),
        np.asarray(gamma_matrix, dtype=complex),
        sites,
        moments,
        model=selected_model,
        model_label=label,
        g_factor=g_factor,
        energy_scale=energy_scale,
        energy_unit=output_unit,
        exchange_hermiticity=exchange_report,
        warnings=warnings,
        stiffness_q_max=stiffness_q_max,
        stiffness_min_points=stiffness_min_points,
    )


def raw_fm_magnon_spectrum(model: MagneticCrystal, q_points: Sequence[Sequence[float]] | np.ndarray, **kwargs: Any) -> FMSpinWaveResult:
    """Convenience wrapper for the all-rigid raw model."""

    return fm_magnon_spectrum(model, q_points, model="raw", **kwargs)


def robust_only_fm_magnon_spectrum(
    model: MagneticCrystal,
    q_points: Sequence[Sequence[float]] | np.ndarray,
    robust_sites: Sequence[int],
    **kwargs: Any,
) -> FMSpinWaveResult:
    """Convenience wrapper for the robust-only raw model."""

    return fm_magnon_spectrum(model, q_points, model="robust_only", robust_sites=robust_sites, **kwargs)


def mryasov_fm_magnon_spectrum(source: InducedExchangeDownfolding | InducedMomentResponse | DownfoldingResult, q_points: Sequence[Sequence[float]] | np.ndarray | None = None, **kwargs: Any) -> FMSpinWaveResult:
    """Convenience wrapper for the Mryasov-like downfolded spectrum."""

    return fm_magnon_spectrum(source, q_points, model="mryasov", **kwargs)


def polesya_fm_magnon_spectrum(source: InducedExchangeDownfolding | InducedMomentResponse | DownfoldingResult, q_points: Sequence[Sequence[float]] | np.ndarray | None = None, **kwargs: Any) -> FMSpinWaveResult:
    """Convenience wrapper for the physical Polesya-like slave spectrum."""

    return fm_magnon_spectrum(source, q_points, model="polesya", **kwargs)


# Names likely to be convenient in notebooks and older scripts.
magnon_spectrum = fm_magnon_spectrum
compute_magnon_spectrum = fm_magnon_spectrum
spin_wave_spectrum = fm_magnon_spectrum
construct_fm_dynamical_matrix = fm_dynamical_matrix
spin_stiffness = fit_spin_stiffness


__all__ = [
    "FMSpinWaveResult",
    "SpinStiffnessResult",
    "compute_magnon_spectrum",
    "construct_fm_dynamical_matrix",
    "fit_spin_stiffness",
    "fm_dynamical_matrix",
    "fm_harmonic_matrix",
    "fm_magnon_spectrum",
    "magnon_spectrum",
    "mryasov_fm_magnon_spectrum",
    "magnon_path_data",
    "plot_magnon_path",
    "polesya_fm_magnon_spectrum",
    "raw_fm_magnon_spectrum",
    "robust_only_fm_magnon_spectrum",
    "spin_stiffness",
    "spin_wave_spectrum",
]
