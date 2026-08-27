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

from .downfolding import DownfoldingResult, InducedExchangeDownfolding
from .induced import InducedMomentResponse, InducedResponseResult, SublatticeClassification, XInferenceResult
from .io_uppasd import InputFormatError, LoadedUppASD, load_uppasd, parse_inpsd
from .magnons import FMSpinWaveResult, SpinStiffnessResult, fit_spin_stiffness, fm_magnon_spectrum
from .model import ExchangeBond, MagneticCrystal, MagneticSite, UnitMetadata, ValidationReport
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
    regular_q_mesh,
)


class UploadMappingError(ValueError):
    """Raised when an uploaded UppASD set is missing a referenced file."""

    def __init__(self, message: str, mapping: "UploadMapping") -> None:
        super().__init__(message)
        self.mapping = mapping


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
    if inpsd is not None:
        uploads = [inpsd, *uploads]
    if inpsd is None:
        references = {key: manual.get(key) if manual else None for key in ("posfile", "momfile", "exchange")}
        unresolved = tuple(f"{key}: choose an uploaded file" for key, value in references.items() if not value)
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
    by_name = {path.name: path for path in uploads}
    root = Path(tempfile.mkdtemp(prefix="imx_upload_"))
    source_text = inpsd.read_text(encoding="utf-8")
    staged_inpsd = root / inpsd.name
    staged_inpsd.write_text(source_text, encoding="utf-8")
    config = parse_inpsd(inpsd)
    for key, reference in (("posfile", config.posfile), ("momfile", config.momfile), ("exchange", config.exchange)):
        selected = by_name.get(mapping.references[key] or "")
        if selected is None:
            raise UploadMappingError(f"No uploaded file is assigned to {key}", mapping)
        # ``config`` stores resolved paths, while the staged inpsd must use
        # the original relative token to locate the copied file.
        raw_reference = (config.keywords.get(key) or [reference.name])[0]
        target = root / Path(raw_reference) if not Path(raw_reference).is_absolute() else root / reference.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(selected, target)
    return staged_inpsd


def load_uploaded_set(
    inpsd_file: Any,
    supporting_files: Any = None,
    *,
    manual_mapping: Mapping[str, str] | None = None,
    energy_unit: str = "unspecified",
    length_unit: str = "unspecified",
) -> tuple[LoadedUppASD, UploadMapping]:
    """Stage browser uploads and load them through the canonical parser."""

    mapping = inspect_upload_mapping(inpsd_file, supporting_files, manual_mapping)
    if _file_path(inpsd_file) is None:
        if not mapping.ok:
            raise UploadMappingError("Individual uploads need one assigned posfile, momfile, and exchange file", mapping)
        uploads = {path.name: path for path in _file_values(supporting_files)}
        root = Path(tempfile.mkdtemp(prefix="imx_individual_"))
        staged_inpsd = root / "generated_inpsd.dat"
        staged_inpsd.write_text(
            "simid generated_upload\ncell 1 0 0\n     0 1 0\n     0 0 1\n"
            f"posfile {mapping.references['posfile']}\nmomfile {mapping.references['momfile']}\nexchange {mapping.references['exchange']}\n",
            encoding="utf-8",
        )
        for name in mapping.references.values():
            source = uploads.get(str(name))
            if source is not None:
                shutil.copyfile(source, root / str(name))
    else:
        staged_inpsd = _stage_uploads(inpsd_file, supporting_files, mapping)
    return load_uppasd(staged_inpsd, energy_unit=energy_unit, length_unit=length_unit), mapping


def model_summary(loaded: LoadedUppASD) -> str:
    model = loaded.model
    cell = "\n".join("  " + "  ".join(f"{value:.6g}" for value in row) for row in model.cell)
    issues = "\n".join(f"- **{issue.level}** `{issue.code}` — {issue.message}" for issue in loaded.report.issues)
    diagnostics = issues or "No parser or model warnings."
    return (
        f"### Parsed input\n\n"
        f"**Cell** (rows are Cartesian direct vectors)\n```text\n{cell}\n```\n\n"
        f"**Basis:** {len(model.sites)} sites · **Jij rows:** {len(model.exchange_bonds)} · "
        f"**Energy:** `{model.units.energy}` · **Moments:** `mu_B`\n\n"
        f"**Reciprocal pair completeness:** `{loaded.report.pair_complete}` · "
        f"**Real-space Hermitian:** `{loaded.report.real_space_hermitian}`\n\n"
        f"#### Diagnostics\n\n{diagnostics}"
    )


def classification_rows(model: MagneticCrystal, roles: Mapping[int, str] | None = None) -> list[list[Any]]:
    roles = roles or {}
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
    q_fractional = regular_q_mesh(model, (size, size, size), coordinates="fractional")
    raw_fourier = exchange_fourier(model, q_fractional, coordinates="fractional")
    raw_eigensystem = exchange_eigensystem(model, q_fractional, coordinates="fractional")
    raw_order = ordering_analysis(model, q_fractional, coordinates="fractional")
    session = AnalysisSession(loaded, classification.robust_sites, classification.induced_sites, q_fractional, raw_fourier.q_cartesian, raw_fourier, raw_eigensystem, raw_order)
    if not raw_fourier.hermiticity.is_hermitian:
        session.warnings.append("Input J(q) is not Hermitian on the selected mesh; malformed or incomplete +/-R input was retained.")

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
            session.warnings.extend(session.downfolding.warnings)
        except (ValueError, np.linalg.LinAlgError) as exc:
            session.warnings.append(f"Dressed exchange unavailable: {exc}")

    try:
        session.path = path_exchange_data(model, high_symmetry_path(model, n_per_segment=20))
        if session.downfolding is not None:
            session.path_dressed = InducedExchangeDownfolding(session.response).evaluate(session.path.path.q_fractional, coordinates="fractional").dressed
    except (ValueError, np.linalg.LinAlgError) as exc:
        session.warnings.append(f"High-symmetry path unavailable: {exc}")

    _calculate_magnons(session)
    _write_exports(session)
    return session


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
        try:
            robust_moments = [model.site_by_index[site].moment for site in session.robust_sites]
            if any(moment is None for moment in robust_moments):
                raise ValueError("a robust site has no reference moment")
            session.magnons["dressed"] = fm_magnon_spectrum(
                session.downfolding,
                model="mryasov",
                moment_magnitudes=np.asarray(robust_moments, dtype=float),
                input_energy_unit=model.units.energy,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            session.warnings.append(f"Dressed magnons unavailable: {exc}")
    for name, spectrum in session.magnons.items():
        try:
            session.stiffness[name] = fit_spin_stiffness(spectrum, q_max=0.1)
        except (ValueError, np.linalg.LinAlgError) as exc:
            session.warnings.append(f"{name} stiffness unavailable: {exc}")


def _model_dict(model: MagneticCrystal) -> dict[str, Any]:
    return {
        "cell": model.cell.tolist(),
        "units": {"energy": model.units.energy, "length": model.units.length, "moment": model.units.moment},
        "sites": [{"index": site.index, "atom_type": str(site.atom_type), "position": list(site.position), "moment": site.moment, "spin_direction": None if site.spin_direction is None else list(site.spin_direction)} for site in model.sites],
        "exchange_bonds": [{"i": b.i, "j": b.j, "displacement": list(b.displacement), "jij": b.jij, "distance": b.distance, "supplied_distance": b.supplied_distance, "source_line": b.source_line} for b in model.exchange_bonds],
    }


def _write_matrix_csv(path: Path, q: np.ndarray, matrices: np.ndarray, prefix: str = "J") -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["q_index", "qx", "qy", "qz", "row", "column", "real", "imag"])
        for qi, point in enumerate(q):
            for row in range(matrices.shape[1]):
                for column in range(matrices.shape[2]):
                    value = matrices[qi, row, column]
                    writer.writerow([qi, *point, row, column, float(value.real), float(value.imag)])


def _write_exports(session: AnalysisSession) -> None:
    target = Path(tempfile.mkdtemp(prefix="imx_results_"))
    session.export_dir = target
    summary = {"model": _model_dict(session.loaded.model), "validation_report": session.loaded.report.as_dict(), "classification": {"robust": list(session.robust_sites), "induced": list(session.induced_sites)}, "q_fractional": session.q_fractional.tolist(), "warnings": list(dict.fromkeys(session.warnings))}
    (target / "canonical_model.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (target / "validation_report.json").write_text(json.dumps(session.loaded.report.as_dict(), indent=2), encoding="utf-8")
    _write_matrix_csv(target / "raw_jq.csv", session.q_fractional, session.raw_fourier.matrices)
    if session.downfolding is not None:
        _write_matrix_csv(target / "dressed_jq.csv", session.q_fractional, session.downfolding.dressed)
        _write_matrix_csv(target / "delta_jq.csv", session.q_fractional, session.downfolding.delta_induced)
    if session.response_scan is not None:
        with (target / "response.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["q_index", "qx", "qy", "qz", "induced_site", "m_real", "m_imag", "m_over_m0_real", "m_over_m0_imag", "condition_number", "singular"])
            normalized = session.response_scan.m_ind_q_over_m_ind_0
            for qi, point in enumerate(session.q_fractional):
                for column, site in enumerate(session.response_scan.induced_sites):
                    value = session.response_scan.induced_moments[qi, column]
                    ratio = None if normalized is None else normalized[qi, column]
                    writer.writerow([qi, *point, site, float(value.real), float(value.imag), None if ratio is None else float(ratio.real), None if ratio is None else float(ratio.imag), float(session.response_scan.condition_numbers[qi]), bool(session.response_scan.singular[qi])])
    if session.downfolding is not None and session.downfolding.source_displacements:
        from .downfolding import inverse_fourier_dressed_jij

        real_space = inverse_fourier_dressed_jij(session.downfolding)
        with (target / "dressed_jij.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["dx", "dy", "dz", "row", "column", "dressed_real", "dressed_imag", "raw_real", "delta_real"])
            for displacement, matrix, raw, delta in zip(real_space.displacements, real_space.values, real_space.raw_values, real_space.delta_values):
                for row in range(matrix.shape[0]):
                    for column in range(matrix.shape[1]):
                        writer.writerow([*displacement, row, column, float(matrix[row, column].real), float(matrix[row, column].imag), float(raw[row, column].real), float(delta[row, column].real)])
    with (target / "magnon_bands.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "q_index", "qx", "qy", "qz", "branch", "energy_real", "energy_imag", "energy_unit"])
        for name, spectrum in session.magnons.items():
            for qi, point in enumerate(spectrum.q_cartesian):
                for branch, value in enumerate(spectrum.energies[qi]):
                    writer.writerow([name, qi, *point, branch, float(value.real), float(value.imag), spectrum.energy_unit])


def shell_rows(model: MagneticCrystal) -> list[list[Any]]:
    if not model.exchange_bonds:
        return []
    groups: dict[float, list[ExchangeBond]] = {}
    for bond in model.exchange_bonds:
        groups.setdefault(round(bond.distance, 8), []).append(bond)
    return [[index, radius, len(bonds), float(np.mean([b.jij for b in bonds])), float(np.min([b.jij for b in bonds])), float(np.max([b.jij for b in bonds]))] for index, (radius, bonds) in enumerate(sorted(groups.items()), start=1)]


def dressed_shell_rows(session: AnalysisSession) -> list[list[Any]]:
    """Summarize finite-q dressed Jij without constructing a new model."""

    if session.downfolding is None or not session.downfolding.source_displacements:
        return []
    from .downfolding import inverse_fourier_dressed_jij

    real_space = inverse_fourier_dressed_jij(session.downfolding)
    grouped: dict[float, list[float]] = {}
    for displacement, matrix in zip(real_space.displacements, real_space.values):
        grouped.setdefault(round(float(np.linalg.norm(displacement)), 8), []).append(float(np.real(matrix[0, 0])))
    return [[index, radius, len(values), float(np.mean(values)), float(np.min(values)), float(np.max(values))]
            for index, (radius, values) in enumerate(sorted(grouped.items()), start=1)]


def _figure_exchange(session: AnalysisSession) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    x = np.arange(len(session.q_fractional))
    for branch in range(session.raw_eigensystem.eigenvalues.shape[1]):
        ax.plot(x, session.raw_eigensystem.eigenvalues[:, branch].real, color="#183b56", alpha=0.65, linewidth=1.1, label="raw" if branch == 0 else None)
    if session.robust_eigenvalues is not None:
        for branch in range(session.robust_eigenvalues.shape[1]):
            ax.plot(x, session.robust_eigenvalues[:, branch].real, color="#18a999", alpha=0.8, linewidth=1.2, linestyle="--", label="robust-only" if branch == 0 else None)
    if session.dressed_eigenvalues is not None:
        for branch in range(session.dressed_eigenvalues.shape[1]):
            ax.plot(x, session.dressed_eigenvalues[:, branch].real, color="#e07a5f", alpha=0.9, linewidth=1.5, label="dressed" if branch == 0 else None)
    ax.set(xlabel="q-mesh point", ylabel=f"exchange eigenvalue ({session.loaded.model.units.energy})", title="Leading exchange bands")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
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
    ax.axhline(0.0, color="#8b98a6", linewidth=0.8)
    ax.set(xlabel="authoritative bond distance", ylabel=f"Jij ({model.units.energy})", title="Raw exchange versus distance")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    return fig


def _figure_path(session: AnalysisSession) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    if session.path is None:
        ax.text(0.5, 0.5, "High-symmetry path unavailable", ha="center", va="center")
        return fig
    for branch in session.path.eigenvalues.T:
        ax.plot(session.path.path.distance, branch.real, color="#183b56", alpha=0.75)
    ax.set_xticks(session.path.path.distance[session.path.path.tick_positions], session.path.path.tick_labels)
    ax.set(xlabel=f"q path distance · source: {session.path.path.source}", ylabel=f"exchange eigenvalue ({session.loaded.model.units.energy})", title="High-symmetry exchange path")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    return fig


def _figure_response(session: AnalysisSession) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    if session.response_scan is None:
        ax.text(0.5, 0.5, "Classify at least one induced site and run analysis", ha="center", va="center")
        return fig
    ratios = session.response_scan.m_ind_q_over_m_ind_0
    for column, site in enumerate(session.response_scan.induced_sites):
        if ratios is not None:
            ax.plot(np.arange(len(ratios)), ratios[:, column].real, label=f"site {site}")
    ax.axhline(1.0, color="#8b98a6", linewidth=0.8)
    ax.set(xlabel="q-mesh point", ylabel="m_ind(q) / m_ind(0)", title="Coherent induced response")
    ax.grid(alpha=0.18)
    if session.response_scan.induced_sites:
        ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _figure_dressed(session: AnalysisSession) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    if session.downfolding is None:
        ax.text(0.5, 0.5, "Dressed exchange unavailable", ha="center", va="center")
        return fig
    x = np.arange(len(session.q_fractional))
    for values, label, color in ((session.robust_eigenvalues, "robust-only", "#18a999"), (session.dressed_eigenvalues, "dressed", "#e07a5f")):
        if values is not None:
            ax.plot(x, values[:, 0].real, label=label, color=color, linewidth=1.7)
    delta_values = _leading_values(session.downfolding.delta_induced)
    if len(delta_values):
        ax.plot(x, delta_values[:, 0].real, label="ΔJ induced", color="#8c5e8a", linewidth=1.2, linestyle=":")
    ax.set(xlabel="q-mesh point", ylabel=f"leading eigenvalue ({session.loaded.model.units.energy})", title="Robust-only J(q) + induced correction = dressed J_eff(q)")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _figure_magnons(session: AnalysisSession) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    colors = {"raw": "#183b56", "robust-only": "#18a999", "dressed": "#e07a5f"}
    for name, spectrum in session.magnons.items():
        for branch in range(spectrum.energies.shape[1]):
            ax.plot(np.arange(len(spectrum.energies)), spectrum.energies[:, branch].real, color=colors.get(name, "#4f5d75"), alpha=0.82, linestyle="--" if not spectrum.stable else "-", label=name if branch == 0 else None)
    ax.set(xlabel="q-mesh point", ylabel="signed magnon energy", title="FM-compatible magnon bands (dashed = unstable reference)")
    ax.grid(alpha=0.18)
    if session.magnons:
        ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
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
    lines = [f"**Mode:** `{session.response.response_label}`", "", "> **K = input Jij is a response-model approximation, not an exact identity.**"]
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
    lines = []
    for name, spectrum in session.magnons.items():
        status = "stable/FM-compatible" if spectrum.stable and spectrum.fm_compatible else "UNSTABLE — signed harmonic data only"
        lines.append(f"**{name}:** `{status}` · branches = `{spectrum.energies.shape[1]}` · negative modes = `{len(spectrum.unstable_indices)}`")
        lines.extend(f"- {warning}" for warning in spectrum.warnings[:3])
    lines.append("\nInduced sites are not added as independent magnon branches; the dressed spectrum uses the adiabatically eliminated robust subspace.")
    return "\n".join(lines)


def export_files(session: AnalysisSession | None) -> list[str]:
    if session is None or session.export_dir is None:
        return []
    return [str(path) for path in sorted(session.export_dir.iterdir()) if path.is_file()]


__all__ = [
    "AnalysisSession",
    "UploadMapping",
    "UploadMappingError",
    "analyse_model",
    "classification_rows",
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
    "_figure_exchange",
    "_figure_realspace",
    "_figure_magnons",
    "_figure_path",
    "_figure_response",
]
