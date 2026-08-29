"""Explicit Polesya-like induced-moment response models.

This module implements the response layer only.  It does not downfold the
energy or create dynamical degrees of freedom for induced sites.  The
``j_weighted`` mode uses the deliberately approximate identification
``K = J_input``; conventional LKAG exchange is *not* thereby promoted to a
formal induction kernel.

The real-space and reciprocal-space implementations use the same row/column
convention as :mod:`induced_exchange.reciprocal`: row ``i``, column ``j`` is
the field on ``i`` due to ``j`` and the input ``ExchangeBond.displacement``
is authoritative.  The native Hamiltonian convention is
``H = -sum_(i != j) Jij e_i dot e_j``.

The J-weighted response uses normalized induced polarization
``p_nu = m_nu / |m_nu^0|`` and robust orientation amplitudes ``e_a``.  Thus
``K=J_input`` has energy units and ``X`` has inverse-energy units.  The
identification is a model approximation, not an exact susceptibility.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Hashable, Literal

import numpy as np

from .model import MagneticCrystal
from .reciprocal import HermiticityReport, exchange_fourier, reciprocal_lattice


ResponseMode = Literal["historical", "unweighted", "j_weighted", "j-weighted"]
_WEIGHTED_MODES = {
    "j_weighted",
    "j-weighted",
    "jweighted",
    "weighted",
    "j_weighted_induced-response_approximation",
    "j-weighted_induced-response_approximation",
}
_HISTORICAL_MODES = {"historical", "unweighted", "local"}


def _mode_name(mode: str) -> str:
    normalized = mode.lower().replace(" ", "_")
    if normalized in _WEIGHTED_MODES:
        return "j_weighted"
    if normalized in _HISTORICAL_MODES:
        return "historical"
    raise ValueError("mode must be 'historical'/'unweighted' or 'j_weighted'")


def _unique_ints(values: Sequence[int] | Any, *, name: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate site indices")
    return result


def _groups(value: Sequence[int] | Mapping[Hashable, Sequence[int]], *, name: str) -> tuple[tuple[int, ...], dict[Hashable, tuple[int, ...]]]:
    if isinstance(value, Mapping):
        groups = {key: _unique_ints(sites, name=f"{name}[{key!r}]") for key, sites in value.items()}
        flattened = tuple(site for sites in groups.values() for site in sites)
        if len(set(flattened)) != len(flattened):
            raise ValueError(f"{name} groups contain a site more than once")
        return flattened, groups
    flattened = _unique_ints(value, name=name)
    return flattened, {name: flattened}


@dataclass(frozen=True)
class SublatticeClassification:
    """User-supplied robust and induced site/sublattice classification.

    Classification is intentionally explicit.  No decision is made from a
    moment magnitude, atom type, or exchange strength.
    """

    robust_sites: tuple[int, ...]
    induced_sites: tuple[int, ...]
    robust_sublattices: Mapping[Hashable, tuple[int, ...]] = field(default_factory=dict)
    induced_sublattices: Mapping[Hashable, tuple[int, ...]] = field(default_factory=dict)

    @classmethod
    def from_inputs(
        cls,
        robust_sites: Sequence[int] | Mapping[Hashable, Sequence[int]],
        induced_sites: Sequence[int] | Mapping[Hashable, Sequence[int]],
    ) -> "SublatticeClassification":
        robust, robust_groups = _groups(robust_sites, name="robust_sites")
        induced, induced_groups = _groups(induced_sites, name="induced_sites")
        return cls(robust, induced, robust_groups, induced_groups)

    def __post_init__(self) -> None:
        robust = _unique_ints(self.robust_sites, name="robust_sites")
        induced = _unique_ints(self.induced_sites, name="induced_sites")
        if set(robust) & set(induced):
            raise ValueError("a site cannot be both robust and induced")
        object.__setattr__(self, "robust_sites", robust)
        object.__setattr__(self, "induced_sites", induced)
        object.__setattr__(self, "robust_sublattices", dict(self.robust_sublattices) or {"robust": robust})
        object.__setattr__(self, "induced_sublattices", dict(self.induced_sublattices) or {"induced": induced})

    @property
    def site_role(self) -> dict[int, str]:
        return {**{site: "robust" for site in self.robust_sites}, **{site: "induced" for site in self.induced_sites}}


@dataclass(frozen=True)
class XInference:
    """One inferred susceptibility-like coefficient and its source field."""

    site_index: int
    moment_reference: float | None
    source_field: float
    source_contributions: tuple[float, ...]
    x: float | None
    warnings: tuple[str, ...] = ()
    reference_polarization: float | None = None

    @property
    def denominator(self) -> float:
        """Alias for the source field in ``p0 = X * source_field``."""

        return self.source_field

    def as_dict(self) -> dict[str, Any]:
        return {
            "site_index": self.site_index,
            "moment_reference": self.moment_reference,
            "reference_polarization": self.reference_polarization,
            "source_field": self.source_field,
            "source_contributions": list(self.source_contributions),
            "x": self.x,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class XInferenceResult:
    """Inference results for all induced sites."""

    per_site: Mapping[int, XInference]
    reference_robust_moments: np.ndarray
    warnings: tuple[str, ...] = ()

    @property
    def x(self) -> dict[int, float | None]:
        return {site: entry.x for site, entry in self.per_site.items()}

    @property
    def reference_robust_orientations(self) -> np.ndarray:
        """Dimensionless reference orientation amplitudes used to infer X."""

        return self.reference_robust_moments

    @property
    def reference_robust_configuration(self) -> np.ndarray:
        """Clearer alias for the dimensionless reference orientation vector."""

        return self.reference_robust_moments

    @property
    def source_fields(self) -> dict[int, float]:
        return {site: entry.source_field for site, entry in self.per_site.items()}

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference_robust_moments": self.reference_robust_moments.tolist(),
            "reference_robust_orientations": self.reference_robust_orientations.tolist(),
            "per_site": {str(site): entry.as_dict() for site, entry in self.per_site.items()},
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class InducedResponseResult:
    """Normalized polarization response and numerical diagnostics.

    ``induced_moments`` is retained as the historical public field name, but
    its values are the dimensionless ``p_nu`` amplitudes in the selected
    response model.  Use :attr:`physical_induced_moments` when reference
    moment magnitudes are available.
    """

    induced_moments: np.ndarray
    source_fields: np.ndarray
    condition_numbers: np.ndarray
    singular: np.ndarray
    induced_sites: tuple[int, ...]
    mode: str
    warnings: tuple[str, ...] = ()
    q_fractional: np.ndarray | None = None
    q_cartesian: np.ndarray | None = None
    hermiticity: HermiticityReport | None = None
    reference_induced_moments: np.ndarray | None = None

    @property
    def response(self) -> np.ndarray:
        return self.induced_moments

    @property
    def m_induced(self) -> np.ndarray:
        return self.induced_moments

    @property
    def induced_polarizations(self) -> np.ndarray:
        """Dimensionless ``p_nu = m_nu / |m_nu^0|`` response amplitudes."""

        return self.induced_moments

    @property
    def physical_induced_moments(self) -> np.ndarray | None:
        """Convert the normalized response to moment amplitudes when possible."""

        if self.reference_induced_moments is None:
            return None
        scale = np.abs(np.asarray(self.reference_induced_moments, dtype=float))
        if self.q_fractional is None:
            # Real-space vector responses are (n_induced, 3), whereas scalar
            # responses are (n_induced,).  Do not add a q axis to the former.
            if self.induced_moments.ndim == 2:
                scale = scale[:, None]
        elif self.induced_moments.ndim == 3:
            scale = scale[None, :, None]
        elif self.induced_moments.ndim >= 2:
            scale = scale[None, :]
        return self.induced_moments * scale

    @property
    def m_induced_over_reference(self) -> np.ndarray | None:
        if self.reference_induced_moments is None:
            return None
        # In the selected normalized-polarization parameterization this ratio
        # is exactly the stored p_nu response.  Keep the historical property
        # name for callers of earlier releases.
        return self.induced_moments

    @property
    def m_ind_q_over_m_ind_0(self) -> np.ndarray | None:
        """Response normalized to the Gamma response when one is present."""

        if self.q_fractional is None:
            return None
        # Reciprocal coordinates differing by an integer lattice vector are
        # equivalent Gamma points. This is important for meshes that use an
        # endpoint at 1 rather than the half-open [0, 1) convention.
        gamma = np.flatnonzero(np.linalg.norm(self.q_fractional - np.rint(self.q_fractional), axis=1) <= 1e-10)
        if len(gamma) == 0:
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.induced_moments / self.induced_moments[int(gamma[0])]

    @property
    def individual_sublattice_response(self) -> dict[int, np.ndarray]:
        """Return one response array per induced site (sublattice)."""

        return {site: self.induced_moments[..., index] for index, site in enumerate(self.induced_sites)}

    def as_dict(self) -> dict[str, Any]:
        return {
            "induced_sites": list(self.induced_sites),
            "mode": self.mode,
            "induced_moments": self.induced_moments.tolist(),
            "induced_polarizations": self.induced_polarizations.tolist(),
            "physical_induced_moments": None if self.physical_induced_moments is None else self.physical_induced_moments.tolist(),
            "source_fields": self.source_fields.tolist(),
            "condition_numbers": self.condition_numbers.tolist(),
            "singular": self.singular.tolist(),
            "warnings": list(self.warnings),
            "q_fractional": None if self.q_fractional is None else self.q_fractional.tolist(),
            "q_cartesian": None if self.q_cartesian is None else self.q_cartesian.tolist(),
            "m_induced_over_reference": None if self.m_induced_over_reference is None else self.m_induced_over_reference.tolist(),
            "m_ind_q_over_m_ind_0": None if self.m_ind_q_over_m_ind_0 is None else self.m_ind_q_over_m_ind_0.tolist(),
            "hermiticity": None if self.hermiticity is None else self.hermiticity.as_dict(),
        }

    def to_json(self, **kwargs: Any) -> str:
        import json

        return json.dumps(self.as_dict(), **kwargs)


def _normalise_configuration(values: Mapping[int, Any] | Sequence[Any] | np.ndarray, sites: tuple[int, ...], *, name: str) -> tuple[np.ndarray, bool]:
    """Return ``(site, component)`` values and whether the input was vector-valued."""

    if isinstance(values, Mapping):
        try:
            raw = [values[site] for site in sites]
        except KeyError as exc:
            raise ValueError(f"{name} mapping is missing site {exc.args[0]}") from exc
    else:
        raw = values
    array = np.asarray(raw, dtype=complex)
    if array.ndim == 0:
        if len(sites) != 1:
            raise ValueError(f"{name} scalar is only valid for one site")
        array = array.reshape(1)
    vector = False
    if array.ndim == 1:
        if array.shape[0] != len(sites):
            raise ValueError(f"{name} must contain {len(sites)} site values")
        array = array[:, None]
    elif array.ndim == 2 and array.shape == (len(sites), 3):
        vector = True
    else:
        raise ValueError(f"{name} must have shape ({len(sites)},) or ({len(sites)}, 3)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array, vector


class InducedMomentResponse:
    """Evaluate historical or J-weighted instantaneous induced moments.

    Parameters
    ----------
    model:
        Input crystal with scalar exchange bonds.
    robust_sites, induced_sites:
        Explicit site indices, or mappings from user-defined sublattice labels
        to site indices.  A :class:`SublatticeClassification` can be supplied
        instead through ``classification``.
    mode:
        ``"historical"``/``"unweighted"`` uses the selected neighbourhood;
        ``"j_weighted"`` uses the labelled ``J-weighted induced-response
        approximation`` with ``K = J_input`` by default.
    x:
        Optional scalar, site mapping, sublattice mapping, or ordered array of
        susceptibility-like coefficients.  For the J-weighted model ``X`` is
        in inverse energy units and is inferred from ``p_nu^0 = 1`` (or its
        signed reference projection) and the reference robust orientations
        when possible.
    """

    def __init__(
        self,
        model: MagneticCrystal,
        robust_sites: Sequence[int] | Mapping[Hashable, Sequence[int]] | SublatticeClassification | None = None,
        induced_sites: Sequence[int] | Mapping[Hashable, Sequence[int]] | None = None,
        *,
        classification: SublatticeClassification | None = None,
        mode: ResponseMode | str = "j_weighted",
        x: float | Sequence[float] | Mapping[Any, float] | None = None,
        kernel_model: MagneticCrystal | None = None,
        neighbourhood: str | Sequence[int] | Mapping[int, Sequence[int]] = "first_shell",
        cutoff: float | None = None,
        condition_limit: float = 1e10,
        singular_tolerance: float = 1e-12,
    ) -> None:
        if classification is not None and (robust_sites is not None or induced_sites is not None):
            raise ValueError("pass either classification or robust_sites/induced_sites, not both")
        if classification is None:
            if isinstance(robust_sites, SublatticeClassification):
                if induced_sites is not None:
                    raise ValueError("induced_sites cannot be passed with a SublatticeClassification")
                classification = robust_sites
            else:
                if robust_sites is None or induced_sites is None:
                    raise ValueError("robust_sites and induced_sites are required")
                classification = SublatticeClassification.from_inputs(robust_sites, induced_sites)
        if not set(classification.robust_sites + classification.induced_sites).issubset(model.site_indices):
            unknown = sorted(set(classification.robust_sites + classification.induced_sites) - model.site_indices)
            raise ValueError(f"classification references unknown site(s): {unknown}")
        if condition_limit <= 1 or singular_tolerance <= 0:
            raise ValueError("condition_limit must be > 1 and singular_tolerance must be positive")
        self.model = model
        self.classification = classification
        self.mode = _mode_name(mode)
        self.kernel_model = kernel_model or model
        if not set(classification.robust_sites + classification.induced_sites).issubset(self.kernel_model.site_indices):
            raise ValueError("kernel_model does not contain all classified sites")
        self.condition_limit = float(condition_limit)
        self.singular_tolerance = float(singular_tolerance)
        self.neighbourhood = neighbourhood
        self.cutoff = cutoff
        self._x_input = x
        self._local_matrix = self._build_local_matrix(neighbourhood, cutoff)

    @property
    def robust_sites(self) -> tuple[int, ...]:
        return self.classification.robust_sites

    @property
    def induced_sites(self) -> tuple[int, ...]:
        return self.classification.induced_sites

    @property
    def response_label(self) -> str:
        return "J-weighted induced-response approximation" if self.mode == "j_weighted" else "historical local/unweighted induced response"

    @property
    def local_neighbour_matrix(self) -> np.ndarray:
        return self._local_matrix.copy()

    def _build_local_matrix(
        self,
        neighbourhood: str | Sequence[int] | Mapping[int, Sequence[int]],
        cutoff: float | None,
    ) -> np.ndarray:
        n_ind, n_rob = len(self.induced_sites), len(self.robust_sites)
        result = np.zeros((n_ind, n_rob), dtype=float)
        robust_position = {site: index for index, site in enumerate(self.robust_sites)}
        bonds = self.kernel_model.exchange_bonds

        if cutoff is not None:
            if cutoff < 0:
                raise ValueError("cutoff must be non-negative")
            selector = "cutoff"
        elif isinstance(neighbourhood, str):
            selector = neighbourhood.lower().replace(" ", "_")
        elif isinstance(neighbourhood, Mapping):
            selector = "explicit_mapping"
        else:
            selector = "explicit"

        if selector in {"explicit_mapping", "explicit"}:
            if isinstance(neighbourhood, Mapping):
                selected = {site: tuple(int(value) for value in neighbourhood.get(site, ())) for site in self.induced_sites}
            else:
                selected_sites = tuple(int(value) for value in neighbourhood)
                selected = {site: selected_sites for site in self.induced_sites}
            for induced, targets in selected.items():
                if induced not in self.induced_sites:
                    raise ValueError(f"explicit neighbourhood references non-induced site {induced}")
                for target in targets:
                    if target not in robust_position:
                        raise ValueError(f"explicit neighbourhood for site {induced} references non-robust site {target}")
                    result[self.induced_sites.index(induced), robust_position[target]] += 1.0
            return result

        candidates: dict[int, list[tuple[int, float]]] = {site: [] for site in self.induced_sites}
        for bond in bonds:
            if bond.i in candidates and bond.j in robust_position:
                candidates[bond.i].append((bond.j, bond.distance))
            elif bond.j in candidates and bond.i in robust_position:
                # Geometric shell selection is direction-independent.  The
                # response itself remains direction-aware and uses bond rows.
                candidates[bond.j].append((bond.i, bond.distance))
        for induced_index, induced in enumerate(self.induced_sites):
            entries = candidates[induced]
            if selector in {"all", "all_shells", "all_neighbours", "all_neighbors"}:
                selected = entries
            elif selector == "cutoff":
                selected = [entry for entry in entries if entry[1] <= float(cutoff) + 1e-12]
            elif selector in {"first_shell", "first", "nearest"}:
                positive = [distance for _, distance in entries if distance > 1e-14]
                minimum = min(positive, default=min((distance for _, distance in entries), default=np.inf))
                selected = [entry for entry in entries if np.isclose(entry[1], minimum, rtol=1e-8, atol=1e-10)]
            else:
                raise ValueError("neighbourhood must be 'first_shell', 'all', 'cutoff', a site list, or a mapping")
            for target, _ in selected:
                result[induced_index, robust_position[target]] += 1.0
        return result

    def _reference_robust_orientations(self) -> tuple[np.ndarray, list[str]]:
        """Return dimensionless robust amplitudes projected on a common axis."""

        if not self.robust_sites:
            raise ValueError("at least one robust site is required")
        site_by_index = self.model.site_by_index
        directions = []
        for site_index in self.robust_sites:
            direction = site_by_index[site_index].spin_direction
            if direction is not None and np.linalg.norm(direction) > 1e-14:
                directions.append(np.asarray(direction, dtype=float) / np.linalg.norm(direction))
        axis = directions[0] if directions else np.array([0.0, 0.0, 1.0])
        warnings: list[str] = []
        if directions and any(abs(float(np.dot(axis, direction))) < 1.0 - 1e-8 for direction in directions[1:]):
            warnings.append("reference spin directions are non-collinear; X inference uses projection on the first robust-site axis")
        orientations = []
        for site_index in self.robust_sites:
            site = site_by_index[site_index]
            if site.moment is None:
                raise ValueError(f"robust site {site_index} has no reference moment")
            direction = site.spin_direction
            projection = 1.0 if direction is None else float(np.dot(np.asarray(direction, dtype=float) / max(np.linalg.norm(direction), 1e-300), axis))
            orientations.append(projection)
        return np.asarray(orientations, dtype=float), warnings

    # Retain the old private spelling as a compatibility shim.  The returned
    # values are now explicitly dimensionless orientations, not mu_B moments.
    def _reference_robust_moments(self) -> tuple[np.ndarray, list[str]]:
        return self._reference_robust_orientations()

    def _reference_induced_moments(self) -> np.ndarray:
        axis = np.array([0.0, 0.0, 1.0])
        for site_index in self.robust_sites:
            direction = self.model.site_by_index[site_index].spin_direction
            if direction is not None and np.linalg.norm(direction) > 1e-14:
                axis = np.asarray(direction, dtype=float)
                axis /= np.linalg.norm(axis)
                break
        values = []
        for site_index in self.induced_sites:
            site = self.model.site_by_index[site_index]
            moment = site.moment
            if moment is None:
                values.append(np.nan)
                continue
            direction = site.spin_direction
            projection = 1.0 if direction is None else float(np.dot(np.asarray(direction, dtype=float) / max(np.linalg.norm(direction), 1e-300), axis))
            values.append(float(moment) * projection)
        return np.asarray(values, dtype=float)

    def _reference_induced_polarizations(self) -> np.ndarray:
        """Return signed reference p values used by normalized X inference."""

        axis = np.array([0.0, 0.0, 1.0])
        for site_index in self.robust_sites:
            direction = self.model.site_by_index[site_index].spin_direction
            if direction is not None and np.linalg.norm(direction) > 1e-14:
                axis = np.asarray(direction, dtype=float)
                axis /= np.linalg.norm(axis)
                break
        values = []
        for site_index in self.induced_sites:
            site = self.model.site_by_index[site_index]
            if site.moment is None:
                values.append(np.nan)
                continue
            direction = site.spin_direction
            projection = 1.0 if direction is None else float(np.dot(np.asarray(direction, dtype=float) / max(np.linalg.norm(direction), 1e-300), axis))
            sign = -1.0 if float(site.moment) < 0.0 else 1.0
            values.append(sign * projection)
        return np.asarray(values, dtype=float)

    def _real_space_blocks(self) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        induced_position = {site: index for index, site in enumerate(self.induced_sites)}
        robust_position = {site: index for index, site in enumerate(self.robust_sites)}
        k_mm = np.zeros((len(self.induced_sites), len(self.induced_sites)), dtype=float)
        k_mr = np.zeros((len(self.induced_sites), len(self.robust_sites)), dtype=float)
        contributions = [[] for _ in self.induced_sites]
        for bond in self.kernel_model.exchange_bonds:
            if bond.i in induced_position and bond.j in induced_position:
                k_mm[induced_position[bond.i], induced_position[bond.j]] += bond.jij
            if bond.i in induced_position and bond.j in robust_position:
                k_mr[induced_position[bond.i], robust_position[bond.j]] += bond.jij
                contributions[induced_position[bond.i]].append(float(bond.jij))
        return k_mm, k_mr, contributions

    def _q_blocks(self, q_points: Sequence[Sequence[float]] | np.ndarray, coordinates: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, HermiticityReport | None]:
        lattice = reciprocal_lattice(self.kernel_model.cell)
        q = np.asarray(q_points, dtype=float)
        if q.shape == (3,):
            q = q.reshape(1, 3)
        if q.ndim != 2 or q.shape[1] != 3:
            raise ValueError("q_points must have shape (3,) or (n, 3)")
        normalized = coordinates.lower().replace("_", "-")
        if normalized in {"fractional", "reduced", "reciprocal-fractional"}:
            q_fractional = q
            q_cartesian = lattice.fractional_to_cartesian(q)
        elif normalized in {"cartesian", "cart", "reciprocal-cartesian"}:
            q_cartesian = q
            q_fractional = lattice.cartesian_to_fractional(q)
        else:
            raise ValueError("coordinates must be 'fractional' or 'cartesian'")
        if not np.isfinite(q).all():
            raise ValueError("q_points contains non-finite values")
        n_ind, n_rob = len(self.induced_sites), len(self.robust_sites)
        if self.mode == "historical":
            k_mm = np.zeros((len(q), n_ind, n_ind), dtype=complex)
            k_mr = np.broadcast_to(self._local_matrix, (len(q), n_ind, n_rob)).astype(complex).copy()
            return q_fractional, q_cartesian, k_mm, k_mr, None
        # ``q_cartesian`` is authoritative for the Fourier phase.  Passing
        # the original fractional coordinates here would interpret reduced
        # reciprocal coordinates as Cartesian and make the two cross blocks
        # inconsistent with downfolding at nonzero q.
        transformed = exchange_fourier(self.kernel_model, q_cartesian, coordinates="cartesian")
        position = {site: index for index, site in enumerate(transformed.site_indices)}
        k_mm = transformed.matrices[:, [position[site] for site in self.induced_sites]][:, :, [position[site] for site in self.induced_sites]]
        k_mr = transformed.matrices[:, [position[site] for site in self.induced_sites]][:, :, [position[site] for site in self.robust_sites]]
        return q_fractional, q_cartesian, k_mm, k_mr, transformed.hermiticity

    def _resolve_x(self, inference: XInferenceResult | None) -> np.ndarray:
        if self._x_input is None:
            if inference is None:
                inference = self.infer_x()
            values = [inference.per_site[site].x for site in self.induced_sites]
            if any(value is None for value in values):
                missing = [site for site, value in zip(self.induced_sites, values) if value is None]
                raise ValueError(f"cannot infer X for induced site(s) {missing}; supply an x override")
            result = np.asarray(values, dtype=float)
        elif np.isscalar(self._x_input):
            result = np.full(len(self.induced_sites), float(self._x_input), dtype=float)
        elif isinstance(self._x_input, Mapping):
            values = []
            group_for_site = {site: group for group, sites in self.classification.induced_sublattices.items() for site in sites}
            for site in self.induced_sites:
                key = site if site in self._x_input else group_for_site.get(site)
                if key not in self._x_input:
                    raise ValueError(f"x override is missing induced site/sublattice {site}")
                values.append(float(self._x_input[key]))
            result = np.asarray(values, dtype=float)
        else:
            result = np.asarray(self._x_input, dtype=float)
            if result.shape != (len(self.induced_sites),):
                raise ValueError(f"x override must be scalar, a mapping, or have shape ({len(self.induced_sites)},)")
        if not np.isfinite(result).all():
            raise ValueError("x contains non-finite values")
        return result

    def infer_x(self, *, cancellation_ratio: float = 1e-8, denominator_tolerance: float = 1e-12) -> XInferenceResult:
        """Infer ``X`` from the reference collinear state.

        ``source_field`` is the actual model-convention field multiplying X.
        A near-zero source is returned as ``x=None`` and is never divided by
        or silently regularized.
        """

        robust_orientations, warnings = self._reference_robust_orientations()
        if self.mode == "historical":
            source_matrix = self._local_matrix
            contributions = [
                (self._local_matrix[index] * robust_orientations).tolist()
                for index in range(len(self.induced_sites))
            ]
        else:
            _, source_matrix, _ = self._real_space_blocks()
            contributions = []
            for index in range(len(self.induced_sites)):
                contributions.append([value * robust_orientations[j] for j, value in enumerate(source_matrix[index]) if value != 0])
        source = source_matrix @ robust_orientations
        entries: dict[int, XInference] = {}
        site_by_index = self.model.site_by_index
        reference_polarizations = self._reference_induced_polarizations()
        for index, site_index in enumerate(self.induced_sites):
            induced_site = site_by_index[site_index]
            m0 = None if induced_site.moment is None else float(induced_site.moment)
            p0 = None if not np.isfinite(reference_polarizations[index]) else float(reference_polarizations[index])
            local_warnings: list[str] = []
            terms = np.asarray(contributions[index], dtype=float)
            if len(terms) > 1 and abs(float(np.sum(terms))) <= cancellation_ratio * max(float(np.sum(np.abs(terms))), 1e-300):
                local_warnings.append("source field is strongly cancelled by competing robust contributions")
            if abs(source[index]) <= denominator_tolerance * max(1.0, float(np.sum(np.abs(terms)))):
                local_warnings.append("source field is zero or near zero; X cannot be inferred")
                x_value = None
            elif p0 is None:
                local_warnings.append("reference induced polarization is missing; X cannot be inferred")
                x_value = None
            else:
                x_value = float(p0 / source[index])
                if x_value < 0:
                    local_warnings.append("inferred X is negative; check signs, reference orientation, and kernel convention")
                if not np.isfinite(x_value):
                    local_warnings.append("inferred X is non-finite")
                    x_value = None
            warnings.extend(f"site {site_index}: {message}" for message in local_warnings)
            entries[site_index] = XInference(
                site_index,
                m0,
                float(source[index]),
                tuple(float(value) for value in terms),
                x_value,
                tuple(local_warnings),
                reference_polarization=p0,
            )
        return XInferenceResult(entries, robust_orientations, tuple(warnings))

    def _apply(
        self,
        k_mm: np.ndarray,
        k_mr: np.ndarray,
        robust_values: np.ndarray,
        x_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        n_q = k_mm.shape[0]
        n_ind = len(self.induced_sites)
        diagonal_x = np.diag(x_values)
        output = np.full((n_q, n_ind, robust_values.shape[-1]), np.nan + 0j, dtype=complex)
        fields = np.einsum("qij,qjk->qik", k_mr, robust_values)
        condition = np.empty(n_q, dtype=float)
        singular = np.zeros(n_q, dtype=bool)
        warnings: list[str] = []
        if n_ind == 0:
            # There is no linear system to condition. Treat the empty system
            # as benign so an explicitly robust-only response is not reported
            # as singular merely because NumPy cannot condition a 0x0 array.
            return output, fields, np.ones(n_q, dtype=float), singular, warnings
        for q_index in range(n_q):
            matrix = np.eye(n_ind, dtype=complex) - diagonal_x @ k_mm[q_index]
            try:
                condition[q_index] = float(np.linalg.cond(matrix))
                smallest_singular = float(np.min(np.linalg.svd(matrix, compute_uv=False)))
            except np.linalg.LinAlgError:
                condition[q_index] = np.inf
                smallest_singular = 0.0
            singular[q_index] = (
                not np.isfinite(condition[q_index])
                or condition[q_index] >= self.condition_limit
                or smallest_singular <= self.singular_tolerance
            )
            if singular[q_index]:
                warnings.append(
                    f"q/index {q_index}: I - X K_mm is near singular (condition number "
                    f"{condition[q_index]:.6g}, minimum singular value {smallest_singular:.6g}); "
                    "possible soft/Stoner-like response"
                )
            try:
                output[q_index] = np.linalg.solve(matrix, diagonal_x @ fields[q_index])
            except np.linalg.LinAlgError:
                singular[q_index] = True
                warnings.append(f"q/index {q_index}: I - X K_mm is singular; response was not regularized")
        return output, fields, condition, singular, warnings

    def response_q(
        self,
        q_points: Sequence[Sequence[float]] | np.ndarray,
        robust_configuration: Mapping[int, Any] | Sequence[Any] | np.ndarray,
        *,
        coordinates: str = "fractional",
    ) -> InducedResponseResult:
        """Evaluate the instantaneous induced response for supplied q-space e(q).

        ``robust_configuration`` contains dimensionless robust orientation
        amplitudes and has shape ``(nq, nrobust)`` for scalar amplitudes, or
        ``(nq, nrobust, 3)`` for vector amplitudes.  A single-q
        ``(nrobust,)``/``(nrobust, 3)`` input is accepted as a convenience.
        """

        q_fractional, q_cartesian, k_mm, k_mr, hermiticity = self._q_blocks(q_points, coordinates)
        n_q = len(q_fractional)
        raw = robust_configuration
        if isinstance(raw, Mapping):
            raise ValueError("q-space robust_configuration must be an array with a q axis")
        values = np.asarray(raw, dtype=complex)
        vector = values.ndim == 3 or (n_q == 1 and values.ndim == 2 and values.shape == (len(self.robust_sites), 3))
        if values.ndim == 1:
            values = values.reshape(1, -1)
        elif values.ndim == 2 and values.shape == (len(self.robust_sites), 3):
            values = values.reshape(1, len(self.robust_sites), 3)
        if vector:
            if values.ndim != 3 or values.shape[1:] != (len(self.robust_sites), 3):
                raise ValueError("vector q-space robust_configuration must have shape (nq, nrobust, 3)")
        elif values.ndim != 2 or values.shape[1] != len(self.robust_sites):
            raise ValueError(f"q-space robust_configuration must have shape ({n_q}, {len(self.robust_sites)})")
        if values.shape[0] != n_q:
            raise ValueError(f"q_points contains {n_q} points but robust_configuration contains {values.shape[0]}")
        if not np.isfinite(values).all():
            raise ValueError("robust_configuration contains non-finite values")
        x_values = self._resolve_x(self.infer_x() if self._x_input is None else None)
        if vector:
            response, fields, condition, singular, warnings = self._apply(k_mm, k_mr, values, x_values)
        else:
            response, fields, condition, singular, warnings = self._apply(k_mm, k_mr, values[..., None], x_values)
            response = response[..., 0]
            fields = fields[..., 0]
        return InducedResponseResult(response, fields, condition, singular, self.induced_sites, self.mode, tuple(warnings), q_fractional, q_cartesian, hermiticity, self._reference_induced_moments())

    def response_real_space(self, robust_configuration: Mapping[int, Any] | Sequence[Any] | np.ndarray) -> InducedResponseResult:
        """Evaluate normalized induced polarization for a robust configuration.

        The input is a dimensionless robust orientation configuration.  This
        computes an algebraic response at one instant.  It does not propagate
        induced moments and does not create induced LLG/LSWT modes.
        """

        values, vector = _normalise_configuration(robust_configuration, self.robust_sites, name="robust_configuration")
        k_mm, k_mr, _ = self._real_space_blocks() if self.mode == "j_weighted" else (np.zeros((len(self.induced_sites), len(self.induced_sites))), self._local_matrix, [])
        x_values = self._resolve_x(self.infer_x() if self._x_input is None else None)
        response, fields, condition, singular, warnings = self._apply(k_mm[None, ...], k_mr[None, ...], values[None, ...], x_values)
        result = response[0] if vector else response[0, :, 0]
        source = fields[0] if vector else fields[0, :, 0]
        return InducedResponseResult(result, source, condition, singular, self.induced_sites, self.mode, tuple(warnings), reference_induced_moments=self._reference_induced_moments())


def instantaneous_induced_moments(
    model: MagneticCrystal,
    robust_configuration: Mapping[int, Any] | Sequence[Any] | np.ndarray,
    robust_sites: Sequence[int] | Mapping[Hashable, Sequence[int]],
    induced_sites: Sequence[int] | Mapping[Hashable, Sequence[int]],
    *,
    mode: ResponseMode | str = "j_weighted",
    x: float | Sequence[float] | Mapping[Any, float] | None = None,
    kernel_model: MagneticCrystal | None = None,
    neighbourhood: str | Sequence[int] | Mapping[int, Sequence[int]] = "first_shell",
    cutoff: float | None = None,
) -> np.ndarray:
    """Return only the instantaneous dimensionless induced polarization ``p``."""

    response = InducedMomentResponse(
        model,
        robust_sites,
        induced_sites,
        mode=mode,
        x=x,
        kernel_model=kernel_model,
        neighbourhood=neighbourhood,
        cutoff=cutoff,
    )
    return response.response_real_space(robust_configuration).induced_moments


# A concise spelling is useful in notebooks and retains the physical name.
InducedResponse = InducedMomentResponse

# Backward-compatible alias for callers of the original API spelling.
instantaneous_slave_moments = instantaneous_induced_moments


__all__ = [
    "InducedMomentResponse",
    "InducedResponse",
    "InducedResponseResult",
    "ResponseMode",
    "SublatticeClassification",
    "XInference",
    "XInferenceResult",
    "instantaneous_induced_moments",
    "instantaneous_slave_moments",
]
