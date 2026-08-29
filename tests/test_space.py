from __future__ import annotations

import pytest
import numpy as np

from induced_exchange import SublatticeClassification
from induced_exchange.io_uppasd import load_uppasd
from induced_exchange.space import (
    UploadMappingError,
    _figure_dressed,
    _figure_dressed_realspace,
    _figure_dressed_realspace_delta,
    _figure_dressed_realspace_relative,
    _figure_exchange,
    _figure_magnons,
    _figure_path,
    _figure_realspace,
    _figure_response,
    analyse_model,
    classification_from_induced_sites,
    classification_rows,
    default_stiffness_q_max,
    default_induced_sites,
    dressed_shell_rows,
    load_uploaded_set,
    magnon_markdown,
    response_markdown,
)


def test_individual_uploads_require_inpsd_cell_instead_of_assuming_identity(tmp_path):
    posfile = tmp_path / "posfile"
    momfile = tmp_path / "momfile"
    jfile = tmp_path / "jfile"
    posfile.write_text("1 1 0 0 0\n", encoding="utf-8")
    momfile.write_text("1 1 1 0 0 1\n", encoding="utf-8")
    jfile.write_text("1 1 0 0 0 1\n", encoding="utf-8")

    with pytest.raises(UploadMappingError, match="inpsd.dat.*cell vectors"):
        load_uploaded_set(
            None,
            [posfile, momfile, jfile],
            manual_mapping={"posfile": "posfile", "momfile": "momfile", "exchange": "jfile"},
        )


def test_magnon_summary_displays_stiffness_and_polesya_equivalence():
    loaded = load_uppasd("examples/induced_toy/inpsd.dat", energy_unit="meV")
    session = analyse_model(loaded, SublatticeClassification.from_inputs([1], [2]), mesh_size=2, x=0.5)
    summary = magnon_markdown(session)

    assert {"dressed", "polesya"}.issubset(session.magnons)
    assert "Spin stiffness" in summary
    assert "Mryasov / Polesya equivalence" in summary


def test_role_toggle_defaults_small_moments_to_induced_and_completes_classification():
    loaded = load_uppasd("examples/uppasd_style/inpsd.dat")

    assert default_induced_sites(loaded.model) == (1,)
    assert [row[3] for row in classification_rows(loaded.model)] == ["induced", "robust"]
    classification = classification_from_induced_sites([1], loaded.model)
    assert classification.robust_sites == (2,)
    assert classification.induced_sites == (1,)
    all_robust = classification_from_induced_sites([], loaded.model)
    assert all_robust.robust_sites == (1, 2)
    assert all_robust.induced_sites == ()


def test_stiffness_interval_scales_with_cell_units():
    loaded = load_uppasd("examples/uppasd_style/inpsd.dat")

    assert default_stiffness_q_max(loaded.model) > 1e9


def test_seekpath_and_realspace_shells_handle_alat_scaled_cells():
    loaded = load_uppasd("examples/uppasd_style/inpsd.dat", expand_symmetry=True)
    session = analyse_model(loaded, SublatticeClassification.from_inputs([1], [2]), mesh_size=2, x=0.5)

    assert session.path is not None
    assert session.path.path.source == "seekpath"
    rows = dressed_shell_rows(session)
    assert rows
    assert all(np.isfinite(row[1]) and np.isfinite(row[5]) for row in rows)
    assert all(np.isfinite(row[7]) for row in rows)


def test_space_plot_helpers_close_figures_after_building_them():
    matplotlib = pytest.importorskip("matplotlib")

    matplotlib.use("Agg")
    plt = pytest.importorskip("matplotlib.pyplot")

    loaded = load_uppasd("examples/induced_toy/inpsd.dat", energy_unit="meV")
    session = analyse_model(loaded, SublatticeClassification.from_inputs([1], [2]), mesh_size=2, x=0.5)
    plt.close("all")
    figures = [
        _figure_exchange(session),
        _figure_realspace(session),
        _figure_path(session),
        _figure_response(session),
        _figure_dressed(session),
        _figure_dressed_realspace(session),
        _figure_dressed_realspace_delta(session),
        _figure_dressed_realspace_relative(session),
        _figure_magnons(session),
    ]

    assert plt.get_fignums() == []
    assert all(hasattr(figure, "savefig") for figure in figures)


def test_space_q_dependent_analysis_uses_one_seekpath_coordinate():
    loaded = load_uppasd("examples/induced_toy/inpsd.dat", energy_unit="meV")
    session = analyse_model(loaded, SublatticeClassification.from_inputs([1], [2]), mesh_size=3, x=0.5)

    assert session.path is not None
    assert session.path.path.source == "seekpath"
    assert np.allclose(session.q_fractional, session.path.path.q_fractional)
    assert np.allclose(session.response_scan.q_fractional, session.q_fractional)
    assert np.allclose(session.magnons["raw"].q_fractional, session.q_fractional)
    assert np.allclose(session.magnons["dressed"].q_fractional, session.q_fractional)
    assert session.dressed_real_space is not None
    assert session.dressed_real_space.q_count == 27


def test_space_plot_labels_explain_path_and_response_normalization():
    pytest.importorskip("matplotlib.pyplot")
    loaded = load_uppasd("examples/induced_toy/inpsd.dat", energy_unit="meV")
    session = analyse_model(loaded, SublatticeClassification.from_inputs([1], [2]), mesh_size=2, x=0.5)

    exchange_figure = _figure_exchange(session)
    response_figure = _figure_response(session)
    assert "high-symmetry path" in exchange_figure.axes[0].get_title()
    assert exchange_figure.axes[0].get_xlabel() == "q-path"
    assert "p_ind(Γ)" in response_figure.axes[0].get_title()
    assert "normalized induced polarization" in response_figure.axes[0].get_ylabel()
    assert "coherent unit-amplitude" in response_markdown(session)


def test_dressed_realspace_plots_separate_induced_correction_and_label_channels():
    pytest.importorskip("matplotlib.pyplot")
    loaded = load_uppasd("examples/induced_toy/inpsd.dat", energy_unit="meV")
    session = analyse_model(loaded, SublatticeClassification.from_inputs([1], [2]), mesh_size=2, x=0.5)

    channels = _figure_dressed_realspace(session)
    induced = _figure_dressed_realspace_delta(session)
    relative = _figure_dressed_realspace_relative(session)
    channel_labels = {line.get_label() for line in channels.axes[0].lines if not line.get_label().startswith("_")}
    induced_labels = {line.get_label() for line in induced.axes[0].lines if not line.get_label().startswith("_")}
    relative_labels = {line.get_label() for line in relative.axes[0].lines if not line.get_label().startswith("_")}

    assert {"J_MM(r)", "J_Mm(r) = K_Mm(r)", "J_Mryasov(r)", "J_Polesya(r) = J_Mryasov(r)"}.issubset(channel_labels)
    assert "ΔJ_induced(r)" not in channel_labels
    assert induced_labels == {"ΔJ_induced(r)"}
    assert relative_labels == {"ΔJ_induced(r) / J_MM(r)"}
    assert relative.axes[0].get_ylabel() == "ΔJ_induced / J_MM"


def test_realspace_plot_uses_implicit_alat_one_coordinates():
    pytest.importorskip("matplotlib.pyplot")
    loaded = load_uppasd("examples/uppasd_style/inpsd.dat", expand_symmetry=True)
    session = analyse_model(loaded, SublatticeClassification.from_inputs([1], [2]), mesh_size=2, x=0.5)
    figure = _figure_dressed_realspace(session)

    distances = np.linalg.norm(session.dressed_real_space.displacements, axis=1) / loaded.config.alat
    scale = max(float(np.max(distances, initial=0.0)), 1.0e-300)
    normalized = distances / scale
    expected = np.asarray([
        float(np.mean(distances[np.isclose(normalized, key, atol=1e-8, rtol=0.0)]))
        for key in sorted({round(float(value), 8) for value in normalized})
    ])

    assert figure.axes[0].get_xlabel() == "Interatomic distance (alat)"
    assert np.allclose(figure.axes[0].lines[0].get_xdata(), expected)
    assert np.max(figure.axes[0].lines[0].get_xdata()) < 10.0


def test_upload_staging_preserves_nested_uppasd_alias_paths(tmp_path):
    inpsd = tmp_path / "inpsd.dat"
    posfile = tmp_path / "posfile"
    momfile = tmp_path / "momfile"
    jfile = tmp_path / "jfile"
    inpsd.write_text(
        "simid demo\n"
        "cell 1 0 0\n"
        "     0 1 0\n"
        "     0 0 1\n"
        "positions data/posfile\n"
        "moments data/momfile\n"
        "exchange data/jfile\n",
        encoding="utf-8",
    )
    posfile.write_text("1 1 0 0 0\n", encoding="utf-8")
    momfile.write_text("1 1 1 0 0 1\n", encoding="utf-8")
    jfile.write_text("1 1 0 0 0 1\n", encoding="utf-8")

    loaded, mapping = load_uploaded_set(inpsd, [posfile, momfile, jfile])

    assert mapping.ok
    assert loaded.model.sites[0].moment == 1.0
    assert loaded.model.exchange_bonds[0].jij == 1.0


def test_upload_staging_rejects_parent_directory_traversal(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    inpsd = input_dir / "inpsd.dat"
    posfile = tmp_path / "posfile"
    momfile = input_dir / "momfile"
    jfile = input_dir / "jfile"
    inpsd.write_text(
        "simid demo\n"
        "cell 1 0 0\n"
        "     0 1 0\n"
        "     0 0 1\n"
        "posfile ../posfile\n"
        "momfile momfile\n"
        "exchange jfile\n",
        encoding="utf-8",
    )
    posfile.write_text("1 1 0 0 0\n", encoding="utf-8")
    momfile.write_text("1 1 1 0 0 1\n", encoding="utf-8")
    jfile.write_text("1 1 0 0 0 1\n", encoding="utf-8")

    with pytest.raises(UploadMappingError, match="escapes the upload staging directory"):
        load_uploaded_set(inpsd, [posfile, momfile, jfile])
