r"""Variational induced-moment downfolding.

The downfolding in this module is derived from one quadratic energy functional,
rather than from the response formula by pattern matching.  With the fixed
exchange convention ``H = -1/2 M^\dagger J_MM M`` we use

.. math::

   E(M,m) = -\tfrac12 M^\dagger J_{MM} M
       + \tfrac12 m^\dagger (X^{-1}-K_{mm})m
       - \operatorname{Re}(m^\dagger K_{mM}M).

Stationarity gives

.. math::

   m^* = (X^{-1}-K_{mm})^{-1}K_{mM}M
       = (I-XK_{mm})^{-1}XK_{mM}M.

Substitution gives ``E_eff = -1/2 M^dagger J_eff M`` with

.. math::

   J_eff = J_MM + K_{Mm}(X^{-1}-K_{mm})^{-1}K_{mM}.

The second form of the stationary operator is used numerically because it is
well-defined in the continuous ``X -> 0`` limit.  ``K = J_input`` remains the
explicitly labelled *J-weighted induced-response approximation*; it is not an
exact susceptibility identity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from typing import Any

import numpy as np

from .induced import InducedMomentResponse
from .model import MagneticCrystal
from .reciprocal import HermiticityReport, check_hermiticity, exchange_fourier


def _leading_eigenvalues(matrices: np.ndarray) -> np.ndarray:
    """Return eigenvalues ordered by descending real part."""

    values = np.empty((len(matrices), matrices.shape[1]), dtype=complex)
    for index, matrix in enumerate(matrices):
        if not np.isfinite(matrix).all():
            values[index] = np.nan + 0j
            continue
        error = np.max(np.abs(matrix - matrix.conj().T), initial=0.0)
        scale = max(1.0, float(np.max(np.abs(matrix), initial=0.0)))
        if error <= 1e-10 + 1e-8 * scale:
            current = np.linalg.eigvalsh(matrix)
        else:
            current = np.linalg.eigvals(matrix)
        values[index] = current[np.argsort(current.real)[::-1]]
    return values


def _normalise_energy_configuration(values: Any, n_q: int, n_robust: int) -> np.ndarray:
    array = np.asarray(values, dtype=complex)
    if array.ndim == 1:
        if n_q != 1 or array.shape != (n_robust,):
            raise ValueError(f"robust configuration must have shape ({n_q}, {n_robust})")
        array = array[None, :]
    elif array.ndim == 2 and array.shape == (n_q, n_robust):
        pass
    elif array.ndim == 3 and array.shape[:2] == (n_q, n_robust) and array.shape[2] == 3:
        pass
    else:
        raise ValueError(
            f"robust configuration must have shape ({n_q}, {n_robust}) or ({n_q}, {n_robust}, 3)"
        )
    if not np.isfinite(array).all():
        raise ValueError("robust configuration contains non-finite values")
    return array if array.ndim == 3 else array[..., None]


@dataclass(frozen=True)
class DownfoldingResult:
    """Raw, dressed, and induced correction matrices on a q-point set."""

    q_fractional: np.ndarray
    q_cartesian: np.ndarray
    raw_robust: np.ndarray
    dressed: np.ndarray
    delta_induced: np.ndarray
    response_operator: np.ndarray
    k_mm: np.ndarray
    k_mr: np.ndarray
    k_rm: np.ndarray
    condition_numbers: np.ndarray
    singular: np.ndarray
    robust_sites: tuple[int, ...]
    induced_sites: tuple[int, ...]
    response_label: str
    kernel_hermiticity: HermiticityReport
    raw_hermiticity: HermiticityReport
    dressed_hermiticity: HermiticityReport
    warnings: tuple[str, ...] = ()
    source_displacements: tuple[tuple[float, float, float], ...] = ()

    @property
    def J_raw(self) -> np.ndarray:
        return self.raw_robust

    @property
    def raw(self) -> np.ndarray:
        return self.raw_robust

    @property
    def J_eff(self) -> np.ndarray:
        return self.dressed

    @property
    def effective(self) -> np.ndarray:
        return self.dressed

    @property
    def delta_J(self) -> np.ndarray:
        return self.delta_induced

    @property
    def Xi(self) -> np.ndarray:
        """The finite ``(I-X K_mm)^-1 X`` stationary response operator."""

        return self.response_operator

    def as_dict(self) -> dict[str, Any]:
        return {
            "q_fractional": self.q_fractional.tolist(),
            "q_cartesian": self.q_cartesian.tolist(),
            "raw_robust_real": self.raw_robust.real.tolist(),
            "raw_robust_imag": self.raw_robust.imag.tolist(),
            "dressed_real": self.dressed.real.tolist(),
            "dressed_imag": self.dressed.imag.tolist(),
            "delta_induced_real": self.delta_induced.real.tolist(),
            "delta_induced_imag": self.delta_induced.imag.tolist(),
            "condition_numbers": self.condition_numbers.tolist(),
            "singular": self.singular.tolist(),
            "robust_sites": list(self.robust_sites),
            "induced_sites": list(self.induced_sites),
            "response_label": self.response_label,
            "kernel_hermiticity": self.kernel_hermiticity.as_dict(),
            "raw_hermiticity": self.raw_hermiticity.as_dict(),
            "dressed_hermiticity": self.dressed_hermiticity.as_dict(),
            "warnings": list(self.warnings),
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.as_dict(), **kwargs)


@dataclass(frozen=True)
class EnergyEquivalence:
    """Comparison of the explicit stationary and downfolded energies."""

    stationary_induced: np.ndarray
    explicit_energy: np.ndarray
    downfolded_energy: np.ndarray
    absolute_error: np.ndarray
    equivalent: bool
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "stationary_induced_real": self.stationary_induced.real.tolist(),
            "stationary_induced_imag": self.stationary_induced.imag.tolist(),
            "explicit_energy": self.explicit_energy.tolist(),
            "downfolded_energy": self.downfolded_energy.tolist(),
            "absolute_error": self.absolute_error.tolist(),
            "equivalent": self.equivalent,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class OrderingComparison:
    """Ordering-vector comparison between raw robust and dressed exchange."""

    q_fractional: np.ndarray
    raw_eigenvalues: np.ndarray
    dressed_eigenvalues: np.ndarray
    raw_order_index: int | None
    dressed_order_index: int | None
    raw_order_fractional: np.ndarray | None
    dressed_order_fractional: np.ndarray | None
    changed: bool | None
    diagnostic: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "q_fractional": self.q_fractional.tolist(),
            "raw_eigenvalues_real": self.raw_eigenvalues.real.tolist(),
            "dressed_eigenvalues_real": self.dressed_eigenvalues.real.tolist(),
            "raw_order_index": self.raw_order_index,
            "dressed_order_index": self.dressed_order_index,
            "raw_order_fractional": None if self.raw_order_fractional is None else self.raw_order_fractional.tolist(),
            "dressed_order_fractional": None if self.dressed_order_fractional is None else self.dressed_order_fractional.tolist(),
            "changed": self.changed,
            "diagnostic": self.diagnostic,
        }


@dataclass(frozen=True)
class DressedExchangeRealSpace:
    """Inverse-transform result with explicit q-sampling limitations."""

    displacements: np.ndarray
    values: np.ndarray
    site_indices: tuple[int, ...]
    q_count: int
    warnings: tuple[str, ...] = ()
    raw_values: np.ndarray | None = None
    delta_values: np.ndarray | None = None

    @property
    def jij(self) -> np.ndarray:
        return self.values

    @property
    def shell_diagnostics(self) -> list[dict[str, float | int]]:
        """Summarize raw, induced, and dressed strengths by radial shell."""

        radii = np.linalg.norm(self.displacements, axis=1)
        diagnostics: list[dict[str, float | int]] = []
        for radius in sorted({round(float(value), 10) for value in radii}):
            selected = np.isclose(radii, radius, atol=1e-10, rtol=1e-8)
            entry: dict[str, float | int] = {"radius": float(radius), "displacement_count": int(np.count_nonzero(selected))}
            for label, data in (("dressed", self.values), ("raw", self.raw_values), ("delta_induced", self.delta_values)):
                if data is not None:
                    entry[f"{label}_max_abs"] = float(np.max(np.abs(data[selected]), initial=0.0))
            diagnostics.append(entry)
        return diagnostics

    def as_dict(self) -> dict[str, Any]:
        return {
            "displacements": self.displacements.tolist(),
            "site_indices": list(self.site_indices),
            "J_real": self.values.real.tolist(),
            "J_imag": self.values.imag.tolist(),
            "raw_J_real": None if self.raw_values is None else self.raw_values.real.tolist(),
            "raw_J_imag": None if self.raw_values is None else self.raw_values.imag.tolist(),
            "delta_J_real": None if self.delta_values is None else self.delta_values.real.tolist(),
            "delta_J_imag": None if self.delta_values is None else self.delta_values.imag.tolist(),
            "q_count": self.q_count,
            "shell_diagnostics": self.shell_diagnostics,
            "warnings": list(self.warnings),
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.as_dict(), **kwargs)


class InducedExchangeDownfolding:
    """Perform variational induced-moment downfolding from an IMX-03 response.

    The constructor accepts an :class:`InducedMomentResponse`, or a model plus
    explicit ``robust_sites`` and ``induced_sites``.  In either form the same
    classification, kernel, X values, and approximation label are used by the
    explicit slave response and by the downfolded energy.
    """

    def __init__(
        self,
        response: InducedMomentResponse | MagneticCrystal,
        robust_sites: Any | None = None,
        induced_sites: Any | None = None,
        **response_options: Any,
    ) -> None:
        if isinstance(response, InducedMomentResponse):
            if robust_sites is not None or induced_sites is not None or response_options:
                raise TypeError("response options cannot be combined with an InducedMomentResponse")
            self.response = response
        elif isinstance(response, MagneticCrystal):
            if robust_sites is None or induced_sites is None:
                raise TypeError("robust_sites and induced_sites are required when constructing from a model")
            self.response = InducedMomentResponse(response, robust_sites, induced_sites, **response_options)
        else:
            raise TypeError("response must be an InducedMomentResponse or MagneticCrystal")

    @property
    def robust_sites(self) -> tuple[int, ...]:
        return self.response.robust_sites

    @property
    def induced_sites(self) -> tuple[int, ...]:
        return self.response.induced_sites

    def evaluate(
        self,
        q_points: Sequence[Sequence[float]] | np.ndarray,
        *,
        coordinates: str = "fractional",
    ) -> DownfoldingResult:
        """Evaluate ``J_MM``, ``J_eff``, and ``delta J_induced``."""

        q_fractional, q_cartesian, k_mm, k_mr, _ = self.response._q_blocks(q_points, coordinates)
        kernel_full = exchange_fourier(self.response.kernel_model, q_cartesian, coordinates="cartesian")
        # The historical mode has a deliberately local K_mM and therefore
        # does not return the full-kernel Hermiticity report from _q_blocks.
        # The source J(q) report is still the right malformed-input
        # diagnostic to attach to the variational result.
        kernel_hermiticity = kernel_full.hermiticity
        raw_full = exchange_fourier(self.response.model, q_cartesian, coordinates="cartesian")
        kernel_position = {site: index for index, site in enumerate(kernel_full.site_indices)}
        raw_position = {site: index for index, site in enumerate(raw_full.site_indices)}
        robust_kernel = [kernel_position[site] for site in self.robust_sites]
        induced_kernel = [kernel_position[site] for site in self.induced_sites]
        robust_raw = [raw_position[site] for site in self.robust_sites]
        raw = raw_full.matrices[:, robust_raw][:, :, robust_raw]
        if self.response.mode == "j_weighted":
            k_rm = kernel_full.matrices[:, robust_kernel][:, :, induced_kernel]
        else:
            k_rm = np.swapaxes(k_mr.conj(), 1, 2)

        x_values = self.response._resolve_x(self.response.infer_x() if self.response._x_input is None else None)
        x_matrix = np.diag(x_values.astype(complex))
        n_q, n_ind = len(q_fractional), len(self.induced_sites)
        operator = np.full((n_q, n_ind, n_ind), np.nan + 0j, dtype=complex)
        condition = np.full(n_q, np.inf, dtype=float)
        singular = np.zeros(n_q, dtype=bool)
        solve_failed = np.zeros(n_q, dtype=bool)
        warnings = list(self.response.infer_x().warnings if self.response._x_input is None else ())
        identity = np.eye(n_ind, dtype=complex)
        for q_index in range(n_q):
            matrix = identity - x_matrix @ k_mm[q_index]
            if n_ind == 0:
                operator[q_index] = matrix
                condition[q_index] = 1.0
                continue
            try:
                condition[q_index] = float(np.linalg.cond(matrix))
                smallest_singular = float(np.min(np.linalg.svd(matrix, compute_uv=False)))
            except np.linalg.LinAlgError:
                condition[q_index] = np.inf
                smallest_singular = 0.0
            singular[q_index] = (
                not np.isfinite(condition[q_index])
                or condition[q_index] >= self.response.condition_limit
                or smallest_singular <= self.response.singular_tolerance
            )
            if singular[q_index]:
                warnings.append(
                    f"q/index {q_index}: I - X K_mm is near singular (condition number "
                    f"{condition[q_index]:.6g}, minimum singular value {smallest_singular:.6g}); "
                    "dressed exchange is not reliable"
                )
            try:
                # This is Xi = (I - X K_mm)^-1 X, the continuous form of
                # (X^-1 - K_mm)^-1 and therefore also handles X == 0.
                operator[q_index] = np.linalg.solve(matrix, x_matrix)
            except np.linalg.LinAlgError:
                singular[q_index] = True
                solve_failed[q_index] = True
                warnings.append(f"q/index {q_index}: I - X K_mm is singular; downfolding was not regularized")

        delta = np.einsum("qij,qjk,qkl->qil", k_rm, operator, k_mr)
        dressed = raw + delta
        block_error = np.max(np.abs(k_rm - np.swapaxes(k_mr.conj(), 1, 2)), axis=(1, 2), initial=0.0)
        if np.any(block_error > kernel_hermiticity.tolerance):
            warnings.append("K_Mm differs from K_mM^dagger for one or more q points; variational energy equivalence is not certified")
        if not kernel_hermiticity.is_hermitian:
            warnings.append("kernel J(q) is not Hermitian; malformed/incomplete input was retained and not silently symmetrized")
        # A condition-limit violation is retained as a flagged, finite result
        # when the linear solve succeeds.  Only an actual failed solve is
        # replaced by NaN; no singular system is silently regularized.
        if np.any(solve_failed):
            delta[solve_failed] = np.nan + 0j
            dressed[solve_failed] = np.nan + 0j
        source_displacements = tuple(
            sorted(
                {
                    tuple(float(value) for value in bond.displacement)
                    for bond in self.response.model.exchange_bonds
                    if bond.i in self.robust_sites and bond.j in self.robust_sites
                }
            )
        )
        return DownfoldingResult(
            q_fractional=q_fractional,
            q_cartesian=q_cartesian,
            raw_robust=raw,
            dressed=dressed,
            delta_induced=delta,
            response_operator=operator,
            k_mm=k_mm,
            k_mr=k_mr,
            k_rm=k_rm,
            condition_numbers=condition,
            singular=singular,
            robust_sites=self.robust_sites,
            induced_sites=self.induced_sites,
            response_label=self.response.response_label,
            kernel_hermiticity=kernel_hermiticity,
            raw_hermiticity=check_hermiticity(raw),
            dressed_hermiticity=check_hermiticity(dressed),
            warnings=tuple(dict.fromkeys(warnings)),
            source_displacements=source_displacements,
        )

    downfold = evaluate

    def energy_equivalence(
        self,
        result: DownfoldingResult,
        robust_configuration: Any,
        *,
        induced_configuration: Any | None = None,
        rtol: float = 1e-9,
        atol: float = 1e-10,
    ) -> EnergyEquivalence:
        """Compare explicit stationary energy with the Schur-complement energy.

        If ``induced_configuration`` is omitted, the explicit energy is
        evaluated at the stationary ``m*``.  Supplying it is useful for
        checking the full quadratic functional away from stationarity.
        """

        values = _normalise_energy_configuration(robust_configuration, len(result.q_fractional), len(self.robust_sites))
        n_q, _, n_components = values.shape
        x_values = self.response._resolve_x(self.response.infer_x() if self.response._x_input is None else None)
        if induced_configuration is None:
            induced = np.einsum("qij,qjk,qkl->qil", result.response_operator, result.k_mr, values)
        else:
            induced = _normalise_energy_configuration(induced_configuration, n_q, len(self.induced_sites))
            if induced.shape[2] != n_components:
                raise ValueError("robust and induced configurations must have the same number of components")
        explicit = np.zeros(n_q, dtype=float)
        downfolded = np.zeros(n_q, dtype=float)
        warnings: list[str] = []
        for q_index in range(n_q):
            a_matrix: np.ndarray | None
            if len(x_values) == 0:
                a_matrix = np.zeros((0, 0), dtype=complex)
            elif np.any(np.abs(x_values) <= 1e-15):
                if np.any(np.abs(induced[q_index]) > atol):
                    raise ValueError("an explicit nonzero induced configuration is undefined for X=0")
                a_matrix = None
            else:
                a_matrix = np.diag(1.0 / x_values) - result.k_mm[q_index]
            for component in range(n_components):
                m = induced[q_index, :, component]
                robust = values[q_index, :, component]
                field = result.k_mr[q_index] @ robust
                raw_term = -0.5 * np.vdot(robust, result.raw_robust[q_index] @ robust).real
                induced_term = 0.0 if a_matrix is None else 0.5 * np.vdot(m, a_matrix @ m).real - np.vdot(m, field).real
                explicit[q_index] += float(raw_term + induced_term)
                downfolded[q_index] += float(-0.5 * np.vdot(robust, result.dressed[q_index] @ robust).real)
        error = np.abs(explicit - downfolded)
        scale = np.maximum(1.0, np.maximum(np.abs(explicit), np.abs(downfolded)))
        equivalent = bool(np.all(error <= atol + rtol * scale))
        if not result.kernel_hermiticity.is_hermitian:
            warnings.append("input kernel is non-Hermitian; equality is a numerical quadratic-form check, not a certified variational energy")
        if not equivalent:
            warnings.append("explicit and downfolded energies differ beyond tolerance")
        return EnergyEquivalence(induced, explicit, downfolded, error, equivalent, tuple(warnings))

    compare_energies = energy_equivalence

    def ordering_comparison(self, result: DownfoldingResult) -> OrderingComparison:
        """Report whether dressing changes the leading q-space ordering vector."""

        raw_values = _leading_eigenvalues(result.raw_robust)
        dressed_values = _leading_eigenvalues(result.dressed)
        raw_index = None if not np.isfinite(raw_values[:, 0].real).any() else int(np.nanargmax(raw_values[:, 0].real))
        dressed_index = None if not np.isfinite(dressed_values[:, 0].real).any() else int(np.nanargmax(dressed_values[:, 0].real))
        raw_q = None if raw_index is None else result.q_fractional[raw_index]
        dressed_q = None if dressed_index is None else result.q_fractional[dressed_index]
        if raw_index is None or dressed_index is None:
            changed = None
            diagnostic = "Ordering comparison unavailable because raw or dressed exchange contains no finite leading eigenvalue."
        else:
            changed = not np.allclose(raw_q - dressed_q, np.rint(raw_q - dressed_q), atol=1e-8, rtol=0.0)
            raw_kind = "Gamma" if np.linalg.norm(raw_q - np.rint(raw_q)) <= 1e-8 else "non-Gamma/AF-like"
            dressed_kind = "Gamma" if np.linalg.norm(dressed_q - np.rint(dressed_q)) <= 1e-8 else "non-Gamma/AF-like"
            if changed:
                diagnostic = f"Yes — raw predicts {raw_kind} at {raw_q.tolist()}, while dressed predicts {dressed_kind} at {dressed_q.tolist()}."
            else:
                diagnostic = f"No — raw and dressed predict the same ordering vector ({raw_kind}) at {raw_q.tolist()}."
        return OrderingComparison(result.q_fractional, raw_values, dressed_values, raw_index, dressed_index, raw_q, dressed_q, changed, diagnostic)

    ordering_diagnostic = ordering_comparison

    def inverse_fourier(
        self,
        result: DownfoldingResult,
        displacements: Sequence[Sequence[float]] | np.ndarray | None = None,
    ) -> DressedExchangeRealSpace:
        return inverse_fourier_dressed_jij(result, displacements=displacements)


def inverse_fourier_dressed_jij(
    result: DownfoldingResult,
    *,
    displacements: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> DressedExchangeRealSpace:
    """Reconstruct dressed real-space matrices on a supplied q mesh.

    This is a finite Fourier reconstruction.  It is unique only on the
    displacement grid resolved by the q mesh; warnings explicitly call out
    non-regular or undersampled meshes and aliasing limitations.
    """

    if len(result.q_cartesian) == 0:
        raise ValueError("at least one q point is required for inverse Fourier transformation")
    if displacements is None:
        displacements_array = np.asarray(result.source_displacements, dtype=float)
    else:
        displacements_array = np.asarray(displacements, dtype=float)
    if displacements_array.size == 0:
        raise ValueError("no displacements supplied and no robust-robust source displacements are available")
    if displacements_array.ndim == 1:
        displacements_array = displacements_array.reshape(1, 3)
    if displacements_array.ndim != 2 or displacements_array.shape[1] != 3 or not np.isfinite(displacements_array).all():
        raise ValueError("displacements must have shape (n, 3) and contain finite values")
    phases = np.exp(-1j * (result.q_cartesian @ displacements_array.T))
    values = np.einsum("qij,qd->dij", result.dressed, phases) / len(result.q_cartesian)
    raw_values = np.einsum("qij,qd->dij", result.raw_robust, phases) / len(result.q_cartesian)
    delta_values = np.einsum("qij,qd->dij", result.delta_induced, phases) / len(result.q_cartesian)
    warnings = [
        "inverse-transformed dressed Jij are finite-q reconstructions, not unique real-space interactions outside the sampled resolution",
    ]
    q_fractional = result.q_fractional
    if len(q_fractional) < 2:
        warnings.append("one q point cannot resolve a real-space range")
    else:
        unique_counts = [len(np.unique(np.round(q_fractional[:, axis], 10))) for axis in range(3)]
        if np.prod(unique_counts) != len(q_fractional):
            warnings.append("q points do not form a complete Cartesian reciprocal mesh; aliasing/truncation cannot be controlled")
        elif any(count < 2 for count in unique_counts):
            warnings.append("at least one reciprocal direction is sampled only once; real-space range is unresolved along that direction")
    return DressedExchangeRealSpace(
        displacements_array,
        values,
        result.robust_sites,
        len(result.q_cartesian),
        tuple(warnings),
        raw_values,
        delta_values,
    )


inverse_dressed_exchange = inverse_fourier_dressed_jij


def downfold_induced_exchange(
    response: InducedMomentResponse,
    q_points: Sequence[Sequence[float]] | np.ndarray,
    *,
    coordinates: str = "fractional",
) -> DownfoldingResult:
    """Convenience wrapper for :class:`InducedExchangeDownfolding`."""

    return InducedExchangeDownfolding(response).evaluate(q_points, coordinates=coordinates)


downfold_exchange = downfold_induced_exchange
MryasovDownfolding = InducedExchangeDownfolding
InducedMomentDownfolding = InducedExchangeDownfolding
VariationalDownfolding = InducedExchangeDownfolding
compute_downfolded_exchange = downfold_induced_exchange
mryasov_downfold = downfold_induced_exchange


__all__ = [
    "DressedExchangeRealSpace",
    "DownfoldingResult",
    "EnergyEquivalence",
    "InducedExchangeDownfolding",
    "InducedMomentDownfolding",
    "MryasovDownfolding",
    "OrderingComparison",
    "VariationalDownfolding",
    "compute_downfolded_exchange",
    "downfold_induced_exchange",
    "downfold_exchange",
    "inverse_dressed_exchange",
    "inverse_fourier_dressed_jij",
    "mryasov_downfold",
]
