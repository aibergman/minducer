"""Gradio-facing orchestration for the Induced-Moment Exchange Explorer.

This module deliberately contains presentation and workflow glue only.  The
scientific calculations are delegated to the public ``induced_exchange``
analysis APIs so that the Space and the library remain independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .downfolding import DressedExchangeRealSpace, DownfoldingResult, InducedExchangeDownfolding, inverse_fourier_dressed_jij
from .induced import InducedMomentResponse, InducedResponseResult, SublatticeClassification, XInferenceResult
from .io_uppasd import InputFormatError, LoadedUppASD, load_uppasd, parse_inpsd
from .magnons import FMSpinWaveResult, SpinStiffnessResult, fit_spin_stiffness, fm_magnon_spectrum
from .model import ExchangeBond, MagneticCrystal, MagneticSite, UnitMetadata, ValidationReport
from .provenance import build_analysis_provenance
from .reciprocal import (
    ExchangeEigenSystem,
    ExchangePathData,
    FourierExchangeResult,
    check_hermiticity,
    exchange_eigensystem,
    exchange_fourier,
    high_symmetry_path,
    ordering_analysis,
    path_exchange_data,
    reciprocal_lattice,
    regular_q_mesh,
)


class UploadMappingError(ValueError):
    """Raised when an uploaded UppASD set is missing a referenced file."""

    def __init__(self, message: str, mapping: "UploadMapping") -> None:
        super().__init__(message)
        self.mapping = mapping


MAX_UPLOAD_FILE_COUNT = 32
MAX_UPLOAD_TOTAL_BYTES = 100 * 1024 * 1024
_TEMP_DIRECTORY_PREFIXES = ("imx_upload_", "imx_results_")


@dataclass(frozen=True)
class UploadMapping:
    """Basename mapping between an ``inpsd.dat`` and browser uploads."""

    inpsd: str | None
    references: Mapping[str, str | None]
    choices: tuple[str, ...]
    unresolved: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.unresolved and all(value is not None for value in self.references.values())

    def rows(self) -> list[list[str]]:
        return [[key, value or "UNRESOLVED", "matched" if value else "missing"] for key, value in self.references.items()]


@dataclass
class AnalysisSession:
    """Everything needed to render the seven Space tabs."""

    loaded: LoadedUppASD
    robust_sites: tuple[int, ...]
    induced_sites: tuple[int, ...]
    q_fractional: np.ndarray
    q_cartesian: np.ndarray
    raw_fourier: FourierExchangeResult
    raw_eigensystem: ExchangeEigenSystem
    raw_ordering: Any
    robust_matrices: np.ndarray | None = None
    robust_eigenvalues: np.ndarray | None = None
    robust_ordering: dict[str, Any] | None = None
    response: InducedMomentResponse | None = None
    inference: XInferenceResult | None = None
    response_scan: InducedResponseResult | None = None
    downfolding: DownfoldingResult | None = None
    dressed_eigenvalues: np.ndarray | None = None
    dressed_ordering: dict[str, Any] | None = None
    path: ExchangePathData | None = None
    path_dressed: np.ndarray | None = None
    dressed_real_space: DressedExchangeRealSpace | None = None
    magnons: dict[str, FMSpinWaveResult] = field(default_factory=dict)
    stiffness: dict[str, SpinStiffnessResult] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    export_dir: Path | None = None


def _file_path(value: Any) -> Path | None:
    """Accept Gradio 4/5 file values and ordinary paths."""

    if value is None or value == "":
        return None
    if isinstance(value, (str, Path)):
        return Path(value)
    if isinstance(value, Mapping):
        for key in ("path", "name", "filepath"):
            if value.get(key):
                return Path(str(value[key]))
    for key in ("path", "name", "filepath"):
        candidate = getattr(value, key, None)
        if candidate:
            return Path(str(candidate))
    return None


def _file_values(values: Any) -> list[Path]:
    if values is None or values == "":
        return []
    if not isinstance(values, (list, tuple)):
        values = [values]
    return [path for value in values if (path := _file_path(value)) is not None]


def _validate_uploads(paths: Sequence[Path]) -> None:
    """Reject malformed or oversized browser uploads before parsing them."""

    if len(paths) > MAX_UPLOAD_FILE_COUNT:
        raise ValueError(f"Too many uploaded files ({len(paths)}); the limit is {MAX_UPLOAD_FILE_COUNT}.")
    total_bytes = 0
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        total_bytes += path.stat().st_size
    if total_bytes > MAX_UPLOAD_TOTAL_BYTES:
        limit_mib = MAX_UPLOAD_TOTAL_BYTES / (1024 * 1024)
        raise ValueError(f"Uploaded files are too large ({total_bytes / (1024 * 1024):.1f} MiB); the limit is {limit_mib:.0f} MiB.")


def _cleanup_temp_directory(path: Path | None) -> None:
    """Remove only directories this module created below the OS temp root."""

    if path is None:
        return
    try:
        candidate = path.expanduser().resolve()
        temp_root = Path(tempfile.gettempdir()).expanduser().resolve()
    except OSError:
        return
    if candidate.parent != temp_root or not candidate.name.startswith(_TEMP_DIRECTORY_PREFIXES):
        return
    shutil.rmtree(candidate, ignore_errors=True)


def cleanup_analysis_artifacts(value: AnalysisSession | Path | None) -> None:
    """Release temporary upload/export directories owned by a workflow value."""

    if isinstance(value, AnalysisSession):
        _cleanup_temp_directory(value.export_dir)
        value.export_dir = None
    elif isinstance(value, Path):
        _cleanup_temp_directory(value)


def _referenced_basename(value: Path) -> str:
    return PurePosixPath(value.as_posix()).name


def inspect_upload_mapping(inpsd_file: Any, supporting_files: Any = None, manual: Mapping[str, str] | None = None) -> UploadMapping:
    """Resolve UppASD references against uploaded basenames.

    Relative directory components from the original ``inpsd.dat`` are not
    trusted; browsers commonly flatten uploads.  Matching is therefore by
    basename and the returned choices are suitable for manual reassignment.
    """

    inpsd = _file_path(inpsd_file)
    uploads = _file_values(supporting_files)
    _validate_uploads(([inpsd] if inpsd is not None else []) + uploads)
    if inpsd is not None:
        uploads = [inpsd, *uploads]
    if inpsd is None:
        references = {key: manual.get(key) if manual else None for key in ("posfile", "momfile", "exchange")}
        unresolved = (
            "inpsd.dat: required because reciprocal-space analysis needs explicit cell vectors",
            *(f"{key}: choose an uploaded file" for key, value in references.items() if not value),
        )
        return UploadMapping(None, references, tuple(path.name for path in uploads), unresolved)
    config = parse_inpsd(inpsd)
    by_basename: dict[str, Path] = {}
    for path in uploads:
        by_basename.setdefault(path.name.casefold(), path)
    manual = manual or {}
    references = {}
    unresolved = []
    for key, reference in (("posfile", config.posfile), ("momfile", config.momfile), ("exchange", config.exchange)):
        selected = None
        manual_name = manual.get(key)
        if manual_name:
            selected = next((path for path in uploads if path.name == manual_name), None)
        if selected is None:
            selected = by_basename.get(reference.name.casefold())
        references[key] = None if selected is None else selected.name
        if selected is None:
            unresolved.append(f"{key}: {reference.name}")
    return UploadMapping(inpsd.name, references, tuple(path.name for path in uploads), tuple(unresolved))


def _stage_uploads(inpsd_file: Any, supporting_files: Any, mapping: UploadMapping) -> Path:
    inpsd = _file_path(inpsd_file)
    uploads = _file_values(supporting_files)
    if inpsd is None or not mapping.ok:
        raise UploadMappingError("Upload mapping is incomplete: " + ", ".join(mapping.unresolved or ("inpsd.dat is missing",)), mapping)
    uploads = [inpsd, *uploads]
    _validate_uploads(uploads)
    by_name = {path.name: path for path in uploads}
    root = Path(tempfile.mkdtemp(prefix="imx_upload_"))
    try:
        source_text = inpsd.read_text(encoding="utf-8")
        staged_inpsd = root / inpsd.name
        staged_inpsd.write_text(source_text, encoding="utf-8")
        config = parse_inpsd(inpsd)
        aliases = {
            "posfile": ("posfile", "positions"),
            "momfile": ("momfile", "moments"),
            "exchange": ("exchange", "jfile"),
        }
        root_resolved = root.resolve()
        for key, reference in (("posfile", config.posfile), ("momfile", config.momfile), ("exchange", config.exchange)):
            selected = by_name.get(mapping.references[key] or "")
            if selected is None:
                raise UploadMappingError(f"No uploaded file is assigned to {key}", mapping)
            # ``config`` stores resolved paths, while the staged inpsd must use
            # the original relative token to locate the copied file.  Reject
            # traversal instead of allowing a browser upload to write outside
            # the private staging directory.
            raw_reference = next(
                (config.keywords[alias][0] for alias in aliases[key] if config.keywords.get(alias)),
                reference.name,
            )
            raw_path = Path(raw_reference)
            if raw_path.is_absolute():
                raise UploadMappingError(f"{key} uses an absolute path; use a relative UppASD filename for uploaded inputs", mapping)
            target = (root / raw_path).resolve()
            if root_resolved not in target.parents:
                raise UploadMappingError(f"{key} path escapes the upload staging directory: {raw_reference!r}", mapping)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(selected, target)
        return staged_inpsd
    except Exception:
        _cleanup_temp_directory(root)
        raise


def load_uploaded_set(
    inpsd_file: Any,
    supporting_files: Any = None,
    *,
    manual_mapping: Mapping[str, str] | None = None,
    energy_unit: str = "mRy",
    length_unit: str = "unspecified",
    expand_symmetry: bool = False,
    symmetry_symprec: float = 1e-5,
) -> tuple[LoadedUppASD, UploadMapping]:
    """Stage browser uploads and load them through the canonical parser.

    When ``expand_symmetry`` is true, the parsed exchange representatives are
    expanded with spglib before the model is returned.
    """

    mapping = inspect_upload_mapping(inpsd_file, supporting_files, manual_mapping)
    if _file_path(inpsd_file) is None:
        raise UploadMappingError(
            "An inpsd.dat upload is required: individual posfile/momfile/jfile uploads do not contain the cell vectors needed for reciprocal-space analysis, and no identity cell is assumed.",
            mapping,
        )
    staged_inpsd = _stage_uploads(inpsd_file, supporting_files, mapping)
    try:
        loaded = load_uppasd(
            staged_inpsd,
            energy_unit=energy_unit,
            length_unit=length_unit,
            expand_symmetry=expand_symmetry,
            symmetry_symprec=symmetry_symprec,
        )
    finally:
        # The loader owns parsed arrays, not the staged files.  Keeping these
        # browser-upload copies alive across requests would leak temp storage.
        _cleanup_temp_directory(staged_inpsd.parent)
    return loaded, mapping


def model_summary(loaded: LoadedUppASD) -> str:
    model = loaded.model
    cell = "\n".join("  " + "  ".join(f"{value:.6g}" for value in row) for row in model.cell)
    issues = "\n".join(f"- **{issue.level}** `{issue.code}` — {issue.message}" for issue in loaded.report.issues)
    diagnostics = issues or "No parser or model warnings."
    length = f"`{model.units.length}`"
    if loaded.config.alat is not None:
        length += f" · `alat = {loaded.config.alat:.6g} m`"
    return (
        f"### Parsed input\n\n"
        f"**Cell** (rows are Cartesian direct vectors)\n```text\n{cell}\n```\n\n"
        f"**Basis:** {len(model.sites)} sites · **Jij rows:** {len(model.exchange_bonds)} · "
        f"**Energy:** `{model.units.energy}` · **Length:** {length} · **Moments:** `mu_B`\n\n"
        f"**Reciprocal pair completeness:** `{loaded.report.pair_complete}` · "
        f"**Real-space Hermitian:** `{loaded.report.real_space_hermitian}`\n\n"
        + (
            f"**Symmetry expansion:** `{loaded.symmetry_expansion.input_bonds} → {loaded.symmetry_expansion.output_bonds}` exchange rows using `{loaded.symmetry_expansion.symmetry_operations}` spglib operations\n\n"
            if loaded.symmetry_expansion is not None
            else ""
        )
        + f"#### Diagnostics\n\n{diagnostics}"
    )


DEFAULT_INDUCED_MOMENT_THRESHOLD = 0.5


def default_induced_sites(model: MagneticCrystal, *, threshold: float = DEFAULT_INDUCED_MOMENT_THRESHOLD) -> tuple[int, ...]:
    """Return the UI's conservative default induced-site selection.

    This is a workflow default, not a scientific inference made by the core
    response API.  Users can change it explicitly with the role toggle.
    """

    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("induced-site threshold must be finite and non-negative")
    return tuple(sorted(site.index for site in model.sites if site.moment is not None and site.moment < threshold))


def classification_from_induced_sites(selected: Any, model: MagneticCrystal) -> SublatticeClassification:
    """Build a complete robust/induced classification from selected site IDs."""

    if selected is None:
        selected_values: Sequence[Any] = default_induced_sites(model)
    elif isinstance(selected, (str, bytes)) or np.isscalar(selected):
        selected_values = [selected]
    else:
        selected_values = selected
    try:
        induced = tuple(sorted({int(value) for value in selected_values}))
    except (TypeError, ValueError) as exc:
        raise ValueError("induced-site selection must contain integer site indices") from exc
    known = model.site_indices
    unknown = set(induced) - known
    if unknown:
        raise ValueError(f"induced-site selection contains unknown site(s): {sorted(unknown)}")
    robust = tuple(sorted(known - set(induced)))
    return SublatticeClassification.from_inputs(robust, induced)


def classification_rows(model: MagneticCrystal, roles: Mapping[int, str] | None = None) -> list[list[Any]]:
    default_roles = {site: "induced" for site in default_induced_sites(model)}
    roles = {**default_roles, **(roles or {})}
    return [[site.index, str(site.atom_type), site.moment, roles.get(site.index, "robust")] for site in model.sites]


def _classification_from_rows(rows: Any, model: MagneticCrystal) -> SublatticeClassification:
    if rows is None:
        rows = []
    robust: list[int] = []
    induced: list[int] = []
    for row in rows:
        if isinstance(row, Mapping):
            site = row.get("site")
            role = row.get("role", "robust")
        else:
            if len(row) < 4:
                continue
            site, role = row[0], row[3]
        try:
            site_index = int(site)
        except (TypeError, ValueError):
            continue
        if str(role).strip().lower() in {"induced", "slave", "induced/slave"}:
            induced.append(site_index)
        else:
            robust.append(site_index)
    known = model.site_indices
    if not robust and not induced:
        robust = sorted(known)
    if set(robust + induced) != known:
        missing = sorted(known - set(robust + induced))
        robust.extend(missing)
    return SublatticeClassification.from_inputs(sorted(set(robust)), sorted(set(induced)))


def _parse_x(value: Any) -> float | list[float] | dict[int, float] | None:
    if value is None or not str(value).strip():
        return None
    text = str(value).strip()
    tokens = [token.strip() for token in text.replace(";", ",").split(",") if token.strip()]
    if any(":" in token or "=" in token for token in tokens):
        result: dict[int, float] = {}
        for token in tokens:
            key, raw = token.replace("=", ":", 1).split(":", 1)
            result[int(key.strip())] = float(raw)
        return result
    values = [float(token) for token in tokens]
    return values[0] if len(values) == 1 else values


def _leading_values(matrices: np.ndarray) -> np.ndarray:
    values = np.full((len(matrices), matrices.shape[1]), np.nan + 0j)
    for index, matrix in enumerate(matrices):
        if not np.isfinite(matrix).all():
            continue
        report = check_hermiticity(matrix)
        current = np.linalg.eigvalsh(matrix) if report.is_hermitian else np.linalg.eigvals(matrix)
        values[index] = current[np.argsort(current.real)[::-1]]
    return values


def _ordering_summary(q_fractional: np.ndarray, q_cartesian: np.ndarray, values: np.ndarray | None) -> dict[str, Any] | None:
    if values is None or not len(values) or not np.isfinite(values[:, 0].real).any():
        return None
    index = int(np.nanargmax(values[:, 0].real))
    q = q_fractional[index]
    kind = "Gamma / FM-like" if np.linalg.norm(q - np.rint(q)) <= 1e-8 else "non-Gamma / AF-like"
    return {"index": index, "q_fractional": q.tolist(), "q_cartesian": q_cartesian[index].tolist(), "eigenvalue": float(values[index, 0].real), "kind": kind, "values": values}


def _filtered_kernel(model: MagneticCrystal, induced: tuple[int, ...]) -> MagneticCrystal:
    bonds = [bond for bond in model.exchange_bonds if not (bond.i in induced and bond.j in induced)]
    return MagneticCrystal(model.cell.copy(), list(model.sites), bonds, model.units, dict(model.source_files))


def analyse_model(
    loaded: LoadedUppASD,
    classification: SublatticeClassification,
    *,
    mesh_size: int = 8,
    mode: str = "j_weighted",
    x: Any = None,
    include_induced_induced: bool = True,
) -> AnalysisSession:
    """Run the selected public analysis layers for one UI session."""

    size = max(2, min(int(mesh_size), 16))
    model = loaded.model
    # The visible reciprocal-space analysis is deliberately one-dimensional:
    # all q-dependent panels use the same canonical seekpath path.  A regular
    # mesh is still constructed below, but only for the auxiliary inverse
    # Fourier transform of dressed real-space exchange.
    selected_path = high_symmetry_path(model, n_per_segment=size, use_seekpath=True)
    path_data = path_exchange_data(model, selected_path)
    q_fractional = selected_path.q_fractional
    raw_fourier = exchange_fourier(model, q_fractional, coordinates="fractional")
    raw_eigensystem = exchange_eigensystem(model, q_fractional, coordinates="fractional")
    raw_order = ordering_analysis(model, q_fractional, coordinates="fractional")
    session = AnalysisSession(loaded, classification.robust_sites, classification.induced_sites, q_fractional, raw_fourier.q_cartesian, raw_fourier, raw_eigensystem, raw_order)
    session.path = path_data
    if selected_path.source != "seekpath":
        session.warnings.append("seekpath was unavailable for this structure; the reciprocal plots use the transparent fallback path.")
    if not raw_fourier.hermiticity.is_hermitian:
        session.warnings.append("Input J(q) is not Hermitian on the selected q path; malformed or incomplete +/-R input was retained.")

    if classification.robust_sites:
        positions = {site: index for index, site in enumerate(raw_fourier.site_indices)}
        robust_indices = [positions[site] for site in classification.robust_sites]
        session.robust_matrices = raw_fourier.matrices[:, robust_indices][:, :, robust_indices]
        session.robust_eigenvalues = _leading_values(session.robust_matrices)
        session.robust_ordering = _ordering_summary(q_fractional, raw_fourier.q_cartesian, session.robust_eigenvalues)

    if classification.induced_sites:
        kernel_model = model if include_induced_induced else _filtered_kernel(model, classification.induced_sites)
        response = InducedMomentResponse(model, classification.robust_sites, classification.induced_sites, mode=mode, x=_parse_x(x), kernel_model=kernel_model)
        session.response = response
        try:
            session.inference = response.infer_x()
        except (ValueError, np.linalg.LinAlgError) as exc:
            session.warnings.append(f"X inference unavailable: {exc}")
        try:
            coherent_robust = np.ones((len(q_fractional), len(classification.robust_sites)), dtype=complex)
            session.response_scan = response.response_q(q_fractional, coherent_robust, coordinates="fractional")
        except (ValueError, np.linalg.LinAlgError) as exc:
            session.warnings.append(f"q-space induced response unavailable: {exc}")
        try:
            session.downfolding = InducedExchangeDownfolding(response).evaluate(q_fractional, coordinates="fractional")
            session.dressed_eigenvalues = _leading_values(session.downfolding.dressed)
            session.dressed_ordering = _ordering_summary(q_fractional, raw_fourier.q_cartesian, session.dressed_eigenvalues)
            session.path_dressed = session.downfolding.dressed
            session.warnings.extend(session.downfolding.warnings)
        except (ValueError, np.linalg.LinAlgError) as exc:
            session.warnings.append(f"Dressed exchange unavailable: {exc}")
        if session.downfolding is not None:
            # A high-symmetry line does not provide a unique inverse Fourier
            # transform.  Use the user-selected resolution for a separate,
            # controlled auxiliary mesh and label the result accordingly.
            try:
                realspace_q = regular_q_mesh(model, (size, size, size), coordinates="fractional")
                realspace_downfolding = InducedExchangeDownfolding(response).evaluate(realspace_q, coordinates="fractional")
                session.dressed_real_space = inverse_fourier_dressed_jij(realspace_downfolding)
                session.warnings.extend(realspace_downfolding.warnings)
            except (ValueError, np.linalg.LinAlgError) as exc:
                session.warnings.append(f"Controlled real-space dressed exchange unavailable: {exc}")

    _calculate_magnons(session)
    session.warnings[:] = list(dict.fromkeys(session.warnings))
    _write_exports(session)
    return session


def default_stiffness_q_max(model: MagneticCrystal) -> float:
    """Choose a near-Gamma Cartesian q interval from the actual cell."""

    reciprocal_vectors = reciprocal_lattice(model.cell).reciprocal_vectors
    shortest = float(np.min(np.linalg.norm(reciprocal_vectors, axis=1)))
    return 0.2 * shortest


def _calculate_magnons(session: AnalysisSession) -> None:
    model = session.loaded.model
    q = session.q_fractional
    try:
        session.magnons["raw"] = fm_magnon_spectrum(model, q, model="raw", coordinates="fractional")
    except (ValueError, np.linalg.LinAlgError) as exc:
        session.warnings.append(f"Raw rigid-site magnons unavailable: {exc}")
    if session.robust_sites:
        try:
            session.magnons["robust-only"] = fm_magnon_spectrum(model, q, model="robust_only", robust_sites=session.robust_sites, coordinates="fractional")
        except (ValueError, np.linalg.LinAlgError) as exc:
            session.warnings.append(f"Robust-only magnons unavailable: {exc}")
    if session.downfolding is not None:
        robust_moments = [model.site_by_index[site].moment for site in session.robust_sites]
        try:
            if any(moment is None for moment in robust_moments):
                raise ValueError("a robust site has no reference moment")
            for name, model_name in (("dressed", "mryasov"), ("polesya", "polesya")):
                session.magnons[name] = fm_magnon_spectrum(
                    session.downfolding,
                    model=model_name,
                    moment_magnitudes=np.asarray(robust_moments, dtype=float),
                    input_energy_unit=model.units.energy,
                )
        except (ValueError, np.linalg.LinAlgError) as exc:
            session.warnings.append(f"Dressed Mryasov/Polesya magnons unavailable: {exc}")
    for name, spectrum in session.magnons.items():
        try:
            session.stiffness[name] = fit_spin_stiffness(spectrum, q_max=default_stiffness_q_max(model))
        except (ValueError, np.linalg.LinAlgError) as exc:
            session.warnings.append(f"{name} stiffness unavailable: {exc}")


def _model_dict(model: MagneticCrystal) -> dict[str, Any]:
    return {
        "cell": model.cell.tolist(),
        "units": {"energy": model.units.energy, "length": model.units.length, "moment": model.units.moment},
        "sites": [{"index": site.index, "atom_type": str(site.atom_type), "position": list(site.position), "moment": site.moment, "spin_direction": None if site.spin_direction is None else list(site.spin_direction)} for site in model.sites],
        "exchange_bonds": [{"i": b.i, "j": b.j, "displacement": list(b.displacement), "jij": b.jij, "distance": b.distance, "supplied_distance": b.supplied_distance, "source_line": b.source_line} for b in model.exchange_bonds],
    }


def _write_matrix_csv(path: Path, q: np.ndarray, matrices: np.ndarray, prefix: str = "J", path_distance: np.ndarray | None = None) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["q_index", "qx", "qy", "qz", "path_distance", "row", "column", "real", "imag"])
        for qi, point in enumerate(q):
            distance = None if path_distance is None or qi >= len(path_distance) else float(path_distance[qi])
            for row in range(matrices.shape[1]):
                for column in range(matrices.shape[2]):
                    value = matrices[qi, row, column]
                    writer.writerow([qi, *point, distance, row, column, float(value.real), float(value.imag)])


def _write_exports(session: AnalysisSession) -> None:
    target = Path(tempfile.mkdtemp(prefix="imx_results_"))
    session.export_dir = target
    response = session.response
    x_values = None
    x_source = "not_applicable"
    response_mode = None
    if response is not None:
        response_mode = response.mode
        x_source = "inferred from reference collinear state" if response._x_input is None else "user supplied override"
        try:
            x_values = response._resolve_x(session.inference if response._x_input is None else None)
        except (ValueError, np.linalg.LinAlgError):
            session.warnings.append("X values were not resolved for provenance because inference/override validation failed")
    provenance = build_analysis_provenance(
        units={"energy": session.loaded.model.units.energy, "length": session.loaded.model.units.length, "moment": session.loaded.model.units.moment},
        robust_sites=session.robust_sites,
        induced_sites=session.induced_sites,
        q_fractional=session.q_fractional,
        q_cartesian=session.q_cartesian,
        mode=response_mode,
        x_source=x_source,
        x_values=x_values,
        numerical_tolerances={
            "fourier_atol": 1e-10,
            "fourier_rtol": 1e-8,
            "ordering_gamma_tolerance": 1e-10,
            "response_condition_limit": None if response is None else response.condition_limit,
            "response_singular_tolerance": None if response is None else response.singular_tolerance,
            "goldstone_relative_tolerance": 1e-8,
        },
    )
    summary = {"model": _model_dict(session.loaded.model), "validation_report": session.loaded.report.as_dict(), "symmetry_expansion": None if session.loaded.symmetry_expansion is None else session.loaded.symmetry_expansion.as_dict(), "classification": {"robust": list(session.robust_sites), "induced": list(session.induced_sites)}, "q_fractional": session.q_fractional.tolist(), "q_sampling": None if session.path is None else {"kind": "high_symmetry_path", "source": session.path.path.source, "tick_labels": list(session.path.path.tick_labels), "tick_positions": session.path.path.tick_positions.tolist()}, "warnings": list(dict.fromkeys(session.warnings)), "analysis_provenance": provenance}
    (target / "canonical_model.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (target / "analysis_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (target / "validation_report.json").write_text(json.dumps(session.loaded.report.as_dict(), indent=2), encoding="utf-8")
    path_distance = None if session.path is None else session.path.path.distance
    _write_matrix_csv(target / "raw_jq.csv", session.q_fractional, session.raw_fourier.matrices, path_distance=path_distance)
    if session.downfolding is not None:
        _write_matrix_csv(target / "dressed_jq.csv", session.q_fractional, session.downfolding.dressed, path_distance=path_distance)
        _write_matrix_csv(target / "delta_jq.csv", session.q_fractional, session.downfolding.delta_induced, path_distance=path_distance)
    if session.response_scan is not None:
        with (target / "response.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["q_index", "qx", "qy", "qz", "path_distance", "induced_site", "m_real", "m_imag", "m_over_m0_real", "m_over_m0_imag", "condition_number", "singular"])
            normalized = session.response_scan.m_ind_q_over_m_ind_0
            for qi, point in enumerate(session.q_fractional):
                for column, site in enumerate(session.response_scan.induced_sites):
                    value = session.response_scan.induced_moments[qi, column]
                    ratio = None if normalized is None else normalized[qi, column]
                    distance = None if session.path is None else float(session.path.path.distance[qi])
                    writer.writerow([qi, *point, distance, site, float(value.real), float(value.imag), None if ratio is None else float(ratio.real), None if ratio is None else float(ratio.imag), float(session.response_scan.condition_numbers[qi]), bool(session.response_scan.singular[qi])])
    if session.dressed_real_space is not None and len(session.dressed_real_space.displacements):
        real_space = session.dressed_real_space
        with (target / "dressed_jij.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["dx", "dy", "dz", "row", "column", "dressed_real", "dressed_imag", "raw_real", "raw_imag", "delta_real", "delta_imag", "moment_normalized_real", "moment_normalized_imag", "q_sampling"])
            rescaled = _single_site_rescaled(session, real_space.values)
            for displacement_index, (displacement, matrix, raw, delta) in enumerate(zip(real_space.displacements, real_space.values, real_space.raw_values, real_space.delta_values)):
                for row in range(matrix.shape[0]):
                    for column in range(matrix.shape[1]):
                        normalized = None if rescaled is None else rescaled[displacement_index, row, column]
                        writer.writerow([*displacement, row, column, float(matrix[row, column].real), float(matrix[row, column].imag), float(raw[row, column].real), float(raw[row, column].imag), float(delta[row, column].real), float(delta[row, column].imag), None if normalized is None else float(normalized.real), None if normalized is None else float(normalized.imag), "auxiliary_complete_mesh"])
    with (target / "magnon_bands.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "q_index", "qx", "qy", "qz", "path_distance", "branch", "energy_real", "energy_imag", "energy_unit"])
        for name, spectrum in session.magnons.items():
            for qi, point in enumerate(spectrum.q_cartesian):
                path_distance = None if session.path is None or qi >= len(session.path.path.distance) else float(session.path.path.distance[qi])
                for branch, value in enumerate(spectrum.energies[qi]):
                    writer.writerow([name, qi, *point, path_distance, branch, float(value.real), float(value.imag), spectrum.energy_unit])


def shell_rows(model: MagneticCrystal) -> list[list[Any]]:
    if not model.exchange_bonds:
        return []
    radii = model.bond_distances
    scale = max(float(np.max(radii, initial=0.0)), 1.0e-300)
    normalized = radii / scale
    groups: list[tuple[float, np.ndarray]] = []
    for key in sorted({round(float(value), 8) for value in normalized}):
        selected = np.isclose(normalized, key, atol=1e-8, rtol=0.0)
        groups.append((float(np.mean(radii[selected])), selected))
    values = np.asarray([bond.jij for bond in model.exchange_bonds], dtype=float)
    return [[index, radius, int(np.count_nonzero(selected)), float(np.mean(values[selected])), float(np.min(values[selected])), float(np.max(values[selected]))]
            for index, (radius, selected) in enumerate(groups, start=1)]


def dressed_shell_rows(session: AnalysisSession) -> list[list[Any]]:
    """Summarize the auxiliary-mesh real-space robust exchange.

    The visible q-space analysis uses a one-dimensional symmetry path, which
    cannot be inverse transformed uniquely.  ``analyse_model`` therefore
    stores a reconstruction from a separate complete mesh.  For the common
    one-robust-site case these are literal scalar ``J_MM(r)``, ``Delta J(r)``
    and ``J_eff(r)`` values; for multiple robust sites the entries are the
    mean over the robust-space matrix and the table also reports the largest
    imaginary component.
    """

    real_space = session.dressed_real_space
    if real_space is None or not len(real_space.displacements):
        return []
    rescaled = _single_site_rescaled(session, real_space.values)
    radii = np.linalg.norm(real_space.displacements, axis=1)
    scale = max(float(np.max(radii, initial=0.0)), 1.0e-300)
    normalized_radii = radii / scale
    rows: list[list[Any]] = []
    for index, normalized_radius in enumerate(sorted({round(float(value), 8) for value in normalized_radii}), start=1):
        selected = np.isclose(normalized_radii, normalized_radius, atol=1e-8, rtol=0.0)
        radius = float(np.mean(radii[selected]))
        raw = real_space.raw_values[selected] if real_space.raw_values is not None else np.zeros_like(real_space.values[selected])
        delta = real_space.delta_values[selected] if real_space.delta_values is not None else np.zeros_like(real_space.values[selected])
        dressed = real_space.values[selected]
        rows.append([
            index,
            radius,
            int(np.count_nonzero(selected)),
            float(np.mean(raw.real)),
            float(np.mean(delta.real)),
            float(np.mean(dressed.real)),
            float(np.max(np.abs(dressed.imag), initial=0.0)),
            "—" if rescaled is None else float(np.mean(rescaled[selected].real)),
        ])
    return rows


def _single_site_rescaled(session: AnalysisSession, values: np.ndarray) -> np.ndarray | None:
    """Return the one-site magnon-normalized exchange when it is defined.

    For one robust site the q-dependent part of
    ``M^(-1/2) [J(0)-J(q)] M^(-1/2)`` uses ``J(r) / m_R``.  The displayed
    ``J_eff(r)`` remains the spin-Hamiltonian exchange; this companion value
    makes the rescaling used by the one-site magnon calculation explicit.
    """

    if len(session.robust_sites) != 1:
        return None
    moment = session.loaded.model.site_by_index[session.robust_sites[0]].moment
    if moment is None or not np.isfinite(moment) or moment <= 0.0:
        return None
    return np.asarray(values, dtype=complex) / float(moment)


def _path_axis(session: AnalysisSession, count: int) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], str]:
    """Return a common x-axis and symmetry ticks for all q-dependent plots."""

    if session.path is not None and len(session.path.path.distance) == count:
        path = session.path.path
        return path.distance, path.tick_positions, path.tick_labels, path.source
    return np.arange(count, dtype=float), np.asarray([], dtype=int), (), "unavailable"


def _figure_exchange(session: AnalysisSession) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    x, tick_positions, tick_labels, source = _path_axis(session, len(session.q_fractional))
    for branch in range(session.raw_eigensystem.eigenvalues.shape[1]):
        ax.plot(x, session.raw_eigensystem.eigenvalues[:, branch].real, color="#183b56", alpha=0.65, linewidth=1.1, label="raw" if branch == 0 else None)
    if session.robust_eigenvalues is not None:
        for branch in range(session.robust_eigenvalues.shape[1]):
            ax.plot(x, session.robust_eigenvalues[:, branch].real, color="#18a999", alpha=0.8, linewidth=1.2, linestyle="--", label="robust-only" if branch == 0 else None)
    if session.dressed_eigenvalues is not None:
        for branch in range(session.dressed_eigenvalues.shape[1]):
            ax.plot(x, session.dressed_eigenvalues[:, branch].real, color="#e07a5f", alpha=0.9, linewidth=1.5, label="dressed" if branch == 0 else None)
    if len(tick_positions):
        ax.set_xticks(x[tick_positions], tick_labels)
    ax.set(xlabel="q-path", ylabel=f"exchange eigenvalue ({session.loaded.model.units.energy})", title="J(q) eigenvalue scan along the high-symmetry path")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _figure_realspace(session: AnalysisSession) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    model = session.loaded.model
    distances = model.bond_distances
    values = np.asarray([bond.jij for bond in model.exchange_bonds], dtype=float)
    if len(distances):
        ax.scatter(distances, values, s=22, color="#18a999", alpha=0.78)
    else:
        ax.text(0.5, 0.5, "No exchange rows", ha="center", va="center")
    ax.axhline(0.0, color="#8b98a6", linewidth=0.8, label="_nolegend_")
    ax.set(xlabel="authoritative bond distance", ylabel=f"Jij ({model.units.energy})", title="Raw exchange versus distance")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _figure_path(session: AnalysisSession) -> Any:
    """Backward-compatible alias for the unified exchange path plot."""

    return _figure_exchange(session)


def _figure_response(session: AnalysisSession) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    if session.response_scan is None:
        ax.text(0.5, 0.5, "Classify at least one induced site and run analysis", ha="center", va="center")
        plt.close(fig)
        return fig
    ratios = session.response_scan.m_ind_q_over_m_ind_0
    x, tick_positions, tick_labels, source = _path_axis(session, len(ratios) if ratios is not None else len(session.q_fractional))
    for column, site in enumerate(session.response_scan.induced_sites):
        if ratios is not None:
            ax.plot(x, ratios[:, column].real, label=f"site {site} · Re")
            if np.max(np.abs(ratios[:, column].imag), initial=0.0) > 1e-10:
                ax.plot(x, ratios[:, column].imag, linestyle=":", label=f"site {site} · Im")
    ax.axhline(1.0, color="#8b98a6", linewidth=0.8, label="_nolegend_")
    if len(tick_positions):
        ax.set_xticks(x[tick_positions], tick_labels)
    ax.set(xlabel="q-path", ylabel="normalized induced Fourier amplitude", title="Coherent induced response: m_ind(q) / m_ind(Γ)")
    ax.grid(alpha=0.18)
    if session.response_scan.induced_sites:
        ax.legend(frameon=False)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _figure_dressed(session: AnalysisSession) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    if session.downfolding is None:
        ax.text(0.5, 0.5, "Dressed exchange unavailable", ha="center", va="center")
        plt.close(fig)
        return fig
    x, tick_positions, tick_labels, source = _path_axis(session, len(session.q_fractional))
    for values, label, color in ((session.robust_eigenvalues, "robust-only", "#18a999"), (session.dressed_eigenvalues, "dressed", "#e07a5f")):
        if values is not None:
            ax.plot(x, values[:, 0].real, label=label, color=color, linewidth=1.7)
    delta_values = _leading_values(session.downfolding.delta_induced)
    if len(delta_values):
        ax.plot(x, delta_values[:, 0].real, label="ΔJ induced", color="#8c5e8a", linewidth=1.2, linestyle=":")
    if len(tick_positions):
        ax.set_xticks(x[tick_positions], tick_labels)
    ax.set(xlabel="q-path", ylabel=f"leading eigenvalue ({session.loaded.model.units.energy})", title="Robust-space J_eff(q) along the high-symmetry path")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _figure_dressed_realspace(session: AnalysisSession) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    real_space = session.dressed_real_space
    if real_space is None or not len(real_space.displacements):
        ax.text(0.5, 0.5, "No controlled real-space dressed exchange is available", ha="center", va="center")
        plt.close(fig)
        return fig
    distances, selections = _real_space_shells(session, real_space.displacements)
    raw = real_space.raw_values
    cross = real_space.cross_values
    dressed = real_space.values
    n_sites = dressed.shape[1]
    if n_sites == 1:
        _plot_shell_series(ax, distances, selections, raw[:, 0, 0] if raw is not None else None, "J_MM(r)", "#18a999")
        if cross is not None and cross.ndim == 3 and cross.shape[1:] == (1, 1):
            _plot_shell_series(ax, distances, selections, cross[:, 0, 0], "J_Mm(r) = K_Mm(r)", "#183b56")
        _plot_shell_series(ax, distances, selections, dressed[:, 0, 0], "J_Mryasov(r)", "#e07a5f")
        # The Polesya/slave and Mryasov/downfolded curves use the same static
        # response operator in this package, so retain both labels while
        # making their numerical equivalence explicit.
        _plot_shell_series(ax, distances, selections, dressed[:, 0, 0], "J_Polesya(r) = J_Mryasov(r)", "#e07a5f", linestyle=":", marker=None)
        title = "Single-site exchange channels in real space (shell means)"
    else:
        for row in range(n_sites):
            for column in range(n_sites):
                label = f"J_Mryasov {real_space.site_indices[row]}→{real_space.site_indices[column]}(r)"
                _plot_shell_series(ax, distances, selections, dressed[:, row, column], label, "#e07a5f")
        if raw is not None:
            for row in range(n_sites):
                for column in range(n_sites):
                    label = f"J_MM {real_space.site_indices[row]}→{real_space.site_indices[column]}(r)"
                    _plot_shell_series(ax, distances, selections, raw[:, row, column], label, "#18a999", linestyle="--")
        if cross is not None:
            for row in range(cross.shape[1]):
                for column in range(cross.shape[2]):
                    robust_site = real_space.site_indices[row] if row < len(real_space.site_indices) else row
                    induced_site = real_space.induced_site_indices[column] if column < len(real_space.induced_site_indices) else column
                    label = f"J_Mm {robust_site}→{induced_site}(r)"
                    _plot_shell_series(ax, distances, selections, cross[:, row, column], label, "#183b56", linestyle="-.")
        title = "Robust-space exchange channels in real space (shell means)"
    ax.axhline(0.0, color="#8b98a6", linewidth=0.8, label="_nolegend_")
    ax.set(xlabel="Interatomic distance (alat)", ylabel=f"exchange ({session.loaded.model.units.energy})", title=title)
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _real_space_shells(session: AnalysisSession, displacements: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Group real-space rows by the same radial-shell convention as tables."""

    # Input values are stored in physical units when ``alat`` is supplied.
    # Plotting divides by that scale so the x-axis is in the implicit lattice
    # parameter convention used by UppASD (alat = 1), without changing any
    # Fourier phases or scientific result arrays.
    alat = getattr(session.loaded.config, "alat", None)
    length_scale = 1.0 if alat is None else float(alat)
    if not np.isfinite(length_scale) or length_scale <= 0.0:
        length_scale = 1.0
    radii = np.linalg.norm(displacements, axis=1) / length_scale
    scale = max(float(np.max(radii, initial=0.0)), 1.0e-300)
    normalized = radii / scale
    selections = [
        np.isclose(normalized, key, atol=1e-8, rtol=0.0)
        for key in sorted({round(float(value), 8) for value in normalized})
    ]
    shell_radii = np.asarray([float(np.mean(radii[selected])) for selected in selections], dtype=float)
    return shell_radii, selections


def _plot_shell_series(
    ax: Any,
    distances: np.ndarray,
    selections: list[np.ndarray],
    values: np.ndarray | None,
    label: str,
    color: str,
    *,
    linestyle: str = "-",
    marker: str | None = "o",
) -> None:
    """Plot shell means, avoiding arbitrary lines through individual bonds."""

    if values is None:
        return
    array = np.asarray(values)
    means_list = []
    for selected in selections:
        values_selected = np.asarray(array[selected].real, dtype=float)
        finite = np.isfinite(values_selected)
        means_list.append(float(np.mean(values_selected[finite])) if np.any(finite) else np.nan)
    means = np.asarray(means_list, dtype=float)
    kwargs: dict[str, Any] = {"color": color, "linewidth": 1.5, "linestyle": linestyle, "label": label}
    if marker is not None:
        kwargs.update(marker=marker, markersize=4.5)
    ax.plot(distances, means, **kwargs)


def _figure_dressed_realspace_delta(session: AnalysisSession) -> Any:
    """Plot the induced real-space correction separately from exchange channels."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    real_space = session.dressed_real_space
    if real_space is None or not len(real_space.displacements) or real_space.delta_values is None:
        ax.text(0.5, 0.5, "No controlled real-space induced correction is available", ha="center", va="center")
        plt.close(fig)
        return fig
    distances, selections = _real_space_shells(session, real_space.displacements)
    delta = real_space.delta_values
    if delta.shape[1:] == (1, 1):
        _plot_shell_series(ax, distances, selections, delta[:, 0, 0], "ΔJ_induced(r)", "#8c5e8a")
    else:
        for row in range(delta.shape[1]):
            for column in range(delta.shape[2]):
                label = f"ΔJ_induced {real_space.site_indices[row]}→{real_space.site_indices[column]}(r)"
                _plot_shell_series(ax, distances, selections, delta[:, row, column], label, "#8c5e8a")
    ax.axhline(0.0, color="#8b98a6", linewidth=0.8, label="_nolegend_")
    ax.set(
        xlabel="Interatomic distance (alat)",
        ylabel=f"exchange ({session.loaded.model.units.energy})",
        title="Induced correction in real space (shell means)",
    )
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _relative_induced_values(real_space: DressedExchangeRealSpace) -> np.ndarray | None:
    """Return ΔJ_induced/J_MM while masking zero direct-exchange entries."""

    if real_space.delta_values is None or real_space.raw_values is None:
        return None
    raw = np.asarray(real_space.raw_values, dtype=complex)
    delta = np.asarray(real_space.delta_values, dtype=complex)
    tolerance = max(float(np.max(np.abs(raw), initial=0.0)) * 1.0e-12, 1.0e-15)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(delta, raw, out=np.full(delta.shape, np.nan + 0.0j, dtype=complex), where=np.abs(raw) > tolerance)


def _figure_dressed_realspace_relative(session: AnalysisSession) -> Any:
    """Plot the induced correction relative to the direct J_MM exchange."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    real_space = session.dressed_real_space
    relative = None if real_space is None else _relative_induced_values(real_space)
    if real_space is None or not len(real_space.displacements) or relative is None:
        ax.text(0.5, 0.5, "No controlled relative induced correction is available", ha="center", va="center")
        plt.close(fig)
        return fig
    distances, selections = _real_space_shells(session, real_space.displacements)
    if relative.shape[1:] == (1, 1):
        _plot_shell_series(ax, distances, selections, relative[:, 0, 0], "ΔJ_induced(r) / J_MM(r)", "#8c5e8a")
    else:
        for row in range(relative.shape[1]):
            for column in range(relative.shape[2]):
                label = f"ΔJ_induced/J_MM {real_space.site_indices[row]}→{real_space.site_indices[column]}(r)"
                _plot_shell_series(ax, distances, selections, relative[:, row, column], label, "#8c5e8a")
    ax.axhline(0.0, color="#8b98a6", linewidth=0.8, label="_nolegend_")
    ax.set(
        xlabel="Interatomic distance (alat)",
        ylabel="ΔJ_induced / J_MM",
        title="Relative induced correction in real space (shell means)",
    )
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _figure_magnons(session: AnalysisSession) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    colors = {"raw": "#183b56", "robust-only": "#18a999", "dressed": "#e07a5f", "polesya": "#8c5e8a"}
    labels = {"raw": "Raw all-rigid", "robust-only": "Robust-only raw", "dressed": "Mryasov-like downfolded", "polesya": "Polesya-like slave"}
    x, tick_positions, tick_labels, source = _path_axis(session, len(next(iter(session.magnons.values())).energies) if session.magnons else 0)
    linestyles = {"raw": "-", "robust-only": "--", "dressed": "-", "polesya": ":"}
    for name, spectrum in session.magnons.items():
        for branch in range(spectrum.energies.shape[1]):
            linestyle = "--" if not spectrum.stable else linestyles.get(name, "-")
            ax.plot(x, spectrum.energies[:, branch].real, color=colors.get(name, "#4f5d75"), alpha=0.82, linestyle=linestyle, label=labels.get(name, name) if branch == 0 else None)
    if len(tick_positions):
        ax.set_xticks(x[tick_positions], tick_labels)
    ax.set(xlabel="q-path", ylabel="signed magnon energy", title="FM-compatible magnon bands along the high-symmetry path")
    ax.grid(alpha=0.18)
    if session.magnons:
        ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    plt.close(fig)
    return fig


def ordering_markdown(label: str, ordering: Any) -> str:
    if ordering is None:
        return f"**{label}:** unavailable"
    if hasattr(ordering, "q_order_fractional"):
        q = ordering.q_order_fractional
        value = ordering.eigenvalue
        kind = "Gamma / FM-like" if q is not None and np.linalg.norm(q - np.rint(q)) <= 1e-8 else "non-Gamma / AF-like"
        return f"**{label}:** `{kind}` · q = `{np.asarray(q).round(5).tolist()}` · leading λ = `{float(np.real(value)):.6g}`"
    return f"**{label}:** `{ordering['kind']}` · q = `{np.asarray(ordering['q_fractional']).round(5).tolist()}` · leading λ = `{ordering['eigenvalue']:.6g}`"


def response_markdown(session: AnalysisSession) -> str:
    if session.response is None:
        return "Classify one or more basis sites as **induced** to enable the response layer."
    path_source = "unknown" if session.path is None else session.path.path.source
    lines = [f"**Mode:** `{session.response.response_label}`", f"**q sampling:** seekpath high-symmetry path (`{path_source}`)", "", "The plot applies a coherent unit-amplitude robust spin spiral, `M_a(q) = 1`, and shows each induced Fourier amplitude divided by its value at Γ: `m_ind(q) / m_ind(Γ)`. It is a complex response amplitude, not a local induced-moment magnitude or a fraction of the reference moment; solid curves are the real part and dotted curves are the imaginary part.", "", "> **K = input Jij is a response-model approximation, not an exact identity."]
    if session.response_scan is not None:
        finite = session.response_scan.condition_numbers[np.isfinite(session.response_scan.condition_numbers)]
        maximum = float(np.max(finite)) if len(finite) else float("inf")
        lines.append(f"\nConditioning: max cond(I − X K_mm) = `{maximum:.4g}`; singular flags = `{int(np.count_nonzero(session.response_scan.singular))}`.")
    if session.inference is not None:
        lines.append("\nInferred X is susceptibility-like (inverse energy), not the induced moment itself.")
    return "\n".join(lines)


def inference_rows(inference: XInferenceResult | None) -> list[list[Any]]:
    if inference is None:
        return []
    return [[site, entry.moment_reference, entry.source_field, entry.x, "; ".join(entry.warnings) or "—"] for site, entry in inference.per_site.items()]


def magnon_markdown(session: AnalysisSession) -> str:
    if not session.magnons:
        return "No FM spectrum could be constructed for the current input."
    path_source = "unknown" if session.path is None else session.path.path.source
    lines = [f"The spectrum is evaluated only along the high-symmetry q path (`{path_source}`); the horizontal axis is Cartesian reciprocal-space path distance."]
    labels = {"raw": "Raw all-rigid", "robust-only": "Robust-only raw", "dressed": "Mryasov-like downfolded", "polesya": "Polesya-like slave"}
    for name, spectrum in session.magnons.items():
        status = "stable/FM-compatible" if spectrum.stable and spectrum.fm_compatible else "UNSTABLE — signed harmonic data only"
        lines.append(f"**{labels.get(name, name)}:** `{status}` · branches = `{spectrum.energies.shape[1]}` · negative modes = `{len(spectrum.unstable_indices)}`")
        lines.extend(f"- {warning}" for warning in spectrum.warnings[:3])
        stiffness = session.stiffness.get(name)
        if stiffness is not None:
            coefficient = "unavailable" if stiffness.coefficient is None else f"{stiffness.coefficient:.6g}"
            lines.append(f"- Spin stiffness: `D = {coefficient}` ({stiffness.energy_unit}, fit |q| ≤ {stiffness.q_max:g}; {stiffness.point_count} point(s))")
            lines.extend(f"- Stiffness note: {warning}" for warning in stiffness.warnings[:2])
    mryasov = session.magnons.get("dressed")
    polesya = session.magnons.get("polesya")
    if mryasov is not None and polesya is not None and mryasov.energies.shape == polesya.energies.shape:
        difference = float(np.max(np.abs(mryasov.energies - polesya.energies), initial=0.0))
        lines.append(f"\n**Mryasov / Polesya equivalence:** max |ΔE| = `{difference:.6g}`; both use the same adiabatically eliminated robust subspace.")
    lines.append("\nInduced sites are not added as independent magnon branches; both dressed spectra use the adiabatically eliminated robust subspace.")
    return "\n".join(lines)


def export_files(session: AnalysisSession | None) -> list[str]:
    if session is None or session.export_dir is None:
        return []
    return [str(path) for path in sorted(session.export_dir.iterdir()) if path.is_file()]


__all__ = [
    "AnalysisSession",
    "DEFAULT_INDUCED_MOMENT_THRESHOLD",
    "UploadMapping",
    "UploadMappingError",
    "analyse_model",
    "classification_from_induced_sites",
    "classification_rows",
    "cleanup_analysis_artifacts",
    "default_induced_sites",
    "default_stiffness_q_max",
    "dressed_shell_rows",
    "export_files",
    "inference_rows",
    "inspect_upload_mapping",
    "load_uploaded_set",
    "magnon_markdown",
    "model_summary",
    "ordering_markdown",
    "response_markdown",
    "shell_rows",
    "_figure_dressed",
    "_figure_dressed_realspace",
    "_figure_dressed_realspace_delta",
    "_figure_dressed_realspace_relative",
    "_figure_exchange",
    "_figure_realspace",
    "_figure_magnons",
    "_figure_path",
    "_figure_response",
]
