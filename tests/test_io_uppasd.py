from pathlib import Path

import numpy as np
import pytest

from induced_exchange import ExchangeBond, MagneticCrystal, MagneticSite, SymmetryExpansionError, expand_exchange_symmetry
from induced_exchange.io_uppasd import InputFormatError, load_uppasd, parse_exchange, parse_inpsd


def write_input(tmp_path: Path, *, exchange: str = "1 1 0 0 0 1 0\n", pos: str = "1 1 0 0 0\n", mom: str = "1 1 2.0 0 0 1\n") -> Path:
    (tmp_path / "posfile").write_text(pos)
    (tmp_path / "momfile").write_text(mom)
    (tmp_path / "jfile").write_text(exchange)
    inpsd = tmp_path / "inpsd.dat"
    inpsd.write_text(
        "simid demo\n"
        "cell 1 0 0\n"
        "  0.5 1 0\n"
        "  0 0 2\n"
        "posfile posfile\n"
        "momfile momfile\n"
        "exchange jfile\n"
    )
    return inpsd


def test_loads_multiline_nonorthogonal_input_and_relative_paths(tmp_path: Path):
    loaded = load_uppasd(write_input(tmp_path))
    assert np.allclose(loaded.cell, [[1, 0, 0], [0.5, 1, 0], [0, 0, 2]])
    assert loaded.config.posfiletype == "C"
    assert loaded.units.energy == "mRy"
    assert loaded.model.sites[0].moment == 2.0
    assert loaded.model.sites[0].spin_direction == (0.0, 0.0, 1.0)
    assert loaded.report.ok
    assert loaded.report.pair_complete


def test_distance_mismatch_is_a_structured_warning(tmp_path: Path):
    loaded = load_uppasd(write_input(tmp_path, exchange="1 1 1 0 0 3 99\n"))
    assert any(issue.code == "distance_mismatch" for issue in loaded.report.warnings)
    assert loaded.model.exchange_bonds[0].distance == 1.0


def test_duplicate_and_missing_reciprocal_are_reported(tmp_path: Path):
    loaded = load_uppasd(write_input(tmp_path, exchange="1 1 1 0 0 3\n1 1 1 0 0 3\n"))
    assert loaded.report.duplicate_bonds
    assert loaded.report.reciprocal_missing


def test_malformed_exchange_row_raises_in_strict_parser(tmp_path: Path):
    path = tmp_path / "bad.jij"
    path.write_text("1 1 0 0 nope 1\n")
    with pytest.raises(InputFormatError):
        parse_exchange(path)


def test_missing_moment_and_comments_are_diagnosed(tmp_path: Path):
    inpsd = write_input(tmp_path, pos="# comment\n1 1 0 0 0\n2 2 1 0 0\n", mom="# comment\n1 1 2\n")
    loaded = load_uppasd(inpsd)
    assert not loaded.report.ok
    assert any(issue.code == "missing_moment" for issue in loaded.report.errors)


def test_missing_file_is_explicit(tmp_path: Path):
    inpsd = write_input(tmp_path)
    (tmp_path / "jfile").unlink()
    with pytest.raises(FileNotFoundError):
        load_uppasd(inpsd)


def test_single_line_cell_is_supported(tmp_path: Path):
    inpsd = write_input(tmp_path)
    text = inpsd.read_text().replace("cell 1 0 0\n  0.5 1 0\n  0 0 2", "cell 1 0 0 0.5 1 0 0 0 2")
    inpsd.write_text(text)
    config = parse_inpsd(inpsd)
    assert config.cell.shape == (3, 3)


def test_posfiletype_d_converts_direct_positions_using_the_cell(tmp_path: Path):
    inpsd = write_input(tmp_path, pos="1 1 0.5 0.5 0.5\n")
    inpsd.write_text("posfiletype D\n" + inpsd.read_text(), encoding="utf-8")

    config = parse_inpsd(inpsd)
    loaded = load_uppasd(inpsd)

    assert config.posfiletype == "D"
    assert np.allclose(loaded.model.sites[0].position, [0.75, 0.5, 1.0])
    assert not any(issue.code == "unknown_keyword" for issue in loaded.report.issues)


def test_maptype_and_supercell_keywords_are_parsed_and_applied(tmp_path: Path):
    inpsd = write_input(tmp_path, exchange="1 1 1 0 0 4\n")
    inpsd.write_text(
        "maptype 2\n"
        "ncell 2 3 4\n"
        "BC P F P\n"
        + inpsd.read_text(),
        encoding="utf-8",
    )

    config = parse_inpsd(inpsd)
    loaded = load_uppasd(inpsd)

    assert config.maptype == 2
    assert config.ncell == (2, 3, 4)
    assert config.bc == ("P", "F", "P")
    assert np.allclose(loaded.model.exchange_bonds[0].displacement, [1, 0, 0])


def test_legacy_file_keyword_aliases_remain_fallbacks(tmp_path: Path):
    (tmp_path / "positions").write_text("1 1 0 0 0\n")
    (tmp_path / "moments").write_text("1 1 2.0 0 0 1\n")
    (tmp_path / "legacy_jfile").write_text("1 1 0 0 0 1 0\n")
    inpsd = tmp_path / "inpsd.dat"
    inpsd.write_text(
        "cell 1 0 0\n"
        "  0 1 0\n"
        "  0 0 1\n"
        "positions positions\n"
        "moments moments\n"
        "jfile legacy_jfile\n"
    )

    config = parse_inpsd(inpsd)

    assert config.posfile == (tmp_path / "positions").resolve()
    assert config.momfile == (tmp_path / "moments").resolve()
    assert config.exchange == (tmp_path / "legacy_jfile").resolve()


def test_canonical_file_keywords_take_precedence_over_fallback_aliases(tmp_path: Path):
    inpsd = write_input(tmp_path)
    (tmp_path / "positions").write_text("1 1 99 99 99\n")
    (tmp_path / "moments").write_text("1 1 99\n")
    (tmp_path / "legacy_jfile").write_text("1 1 0 0 0 99\n")
    inpsd.write_text(
        inpsd.read_text()
        + "positions positions\n"
        + "moments moments\n"
        + "jfile legacy_jfile\n"
    )

    config = parse_inpsd(inpsd)

    assert config.posfile == (tmp_path / "posfile").resolve()
    assert config.momfile == (tmp_path / "momfile").resolve()
    assert config.exchange == (tmp_path / "jfile").resolve()


def test_alat_scales_cell_positions_and_exchange_to_metres(tmp_path: Path):
    inpsd = write_input(tmp_path, exchange="1 1 1 0 0 3 1\n", pos="1 1 0.5 0 0\n")
    inpsd.write_text("alat 2.87e-10\n" + inpsd.read_text(), encoding="utf-8")

    loaded = load_uppasd(inpsd)

    assert loaded.config.alat == pytest.approx(2.87e-10)
    assert loaded.units.length == "m"
    assert np.allclose(loaded.cell, [[2.87e-10, 0, 0], [1.435e-10, 2.87e-10, 0], [0, 0, 5.74e-10]])
    assert np.allclose(loaded.model.sites[0].position, [1.435e-10, 0, 0])
    assert np.allclose(loaded.model.exchange_bonds[0].displacement, [2.87e-10, 0, 0])
    assert loaded.model.exchange_bonds[0].supplied_distance == pytest.approx(2.87e-10)
    assert not any(issue.code == "unknown_keyword" for issue in loaded.report.issues)


def test_alat_scaled_cells_remain_usable_for_symmetry_expansion(tmp_path: Path):
    pytest.importorskip("spglib")
    inpsd = write_input(
        tmp_path,
        exchange="1 1 1 0 0 1 1\n1 1 -1 0 0 1 1\n",
        pos="1 1 0 0 0\n",
    )
    inpsd.write_text("alat 2.87e-10\n" + inpsd.read_text(), encoding="utf-8")

    loaded = load_uppasd(inpsd, expand_symmetry=True)

    assert loaded.symmetry_expansion is not None
    assert loaded.report.pair_complete
    assert loaded.report.ok


def test_uppasd_style_keywords_and_cell_are_supported():
    loaded = load_uppasd("examples/uppasd_style/inpsd.dat")

    alat = 2.87e-10
    assert loaded.config.alat == pytest.approx(alat)
    assert loaded.units.length == "m"
    assert np.allclose(loaded.cell, np.diag([alat, alat, 1.36882830 * alat]))
    assert len(loaded.model.sites) == 2
    assert len(loaded.model.exchange_bonds) == 148
    assert not loaded.report.duplicate_bonds
    assert not any(issue.code == "unknown_keyword" for issue in loaded.report.issues)


def test_symmetry_expansion_completes_the_uppasd_style_neighbour_set():
    pytest.importorskip("spglib")
    loaded = load_uppasd("examples/uppasd_style/inpsd.dat", expand_symmetry=True)

    # Species are the posfile's second-column values (1 = Fe, 2 = Pt here).
    # The momfile's second field is metadata and is always 1 in this example.
    assert loaded.model.site_by_index[1].atom_type == 1
    assert loaded.model.site_by_index[2].atom_type == 2
    assert not any(issue.code == "atom_type_mismatch" for issue in loaded.report.issues)
    assert loaded.symmetry_expansion is not None
    assert loaded.symmetry_expansion.input_bonds == 148
    assert loaded.symmetry_expansion.output_bonds == len(loaded.model.exchange_bonds)
    assert loaded.symmetry_expansion.output_bonds > loaded.symmetry_expansion.input_bonds
    assert loaded.symmetry_expansion.symmetry_operations > 1
    assert loaded.report.pair_complete
    # The supplied 1->2 and 2->1 representatives are numerically different.
    # Expansion must retain that fact as a diagnostic rather than averaging it.
    assert not loaded.report.real_space_hermitian
    assert any(issue.code == "asymmetric_reciprocal" for issue in loaded.report.issues)


def test_symmetry_expansion_rejects_conflicting_equivalent_values():
    pytest.importorskip("spglib")
    model = MagneticCrystal(
        cell=np.eye(3),
        sites=[MagneticSite(1, 1, (0.0, 0.0, 0.0), 1.0)],
        exchange_bonds=[
            ExchangeBond(1, 1, (1.0, 0.0, 0.0), 1.0),
            ExchangeBond(1, 1, (-1.0, 0.0, 0.0), 2.0),
        ],
    )

    with pytest.raises(SymmetryExpansionError, match="values conflict"):
        expand_exchange_symmetry(model)
