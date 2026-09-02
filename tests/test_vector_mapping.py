from pathlib import Path

import numpy as np
import pytest

from induced_exchange import (
    VectorMappingError,
    infer_target_site,
    map_exchange_file,
    map_exchange_vector,
    prepare_positions,
    read_jfile,
)


def test_maptypes_2_and_3_agree_for_a_basis_inside_the_first_cell():
    positions = [(1, "Fe", 0.0, 0.0, 0.0)]
    prepared = prepare_positions(positions, np.eye(3))

    map2 = map_exchange_vector(1, 1, [1, 0, 0], positions=prepared, maptype=2)
    map3 = map_exchange_vector(1, 1, [1, 0, 0], positions=prepared, maptype=3)

    assert np.allclose(map2, [1, 0, 0])
    assert np.allclose(map2, map3)


def test_maptypes_2_and_3_differ_for_an_out_of_cell_basis_position(tmp_path: Path):
    positions = [(1, "Fe", -0.25, 0.0, 0.0), (2, "Pt", 0.25, 0.0, 0.0)]
    jfile = tmp_path / "jfile"
    jfile.write_text("1 2 0 0 0 4\n", encoding="utf-8")
    prepared = prepare_positions(positions, np.eye(3))

    map2 = map_exchange_file(jfile, positions=prepared, maptype=2)
    map3 = map_exchange_file(jfile, positions=prepared, maptype=3)

    assert np.allclose(map2[0].rij_cart, [-0.5, 0, 0])
    assert np.allclose(map3[0].rij_cart, [0.5, 0, 0])
    assert map2[0].inferred_target_cell_offset == (0, 0, 0)
    assert map3[0].inferred_target_cell_offset == (1, 0, 0)


def test_direct_coordinate_positions_are_converted_before_folding():
    cell = np.diag([2.0, 3.0, 4.0])
    prepared = prepare_positions([(1, 1, 0.5, 0.5, 0.5)], cell, posfiletype="D")

    assert np.allclose(prepared.positions_raw[1], [1.0, 1.5, 2.0])
    assert np.allclose(prepared.positions_folded[1], [1.0, 1.5, 2.0])


def test_cartesian_coordinate_positions_are_preserved():
    position = [0.25, 0.5, 0.75]
    prepared = prepare_positions([(1, 1, *position)], np.diag([2.0, 3.0, 4.0]), posfiletype="C")

    assert np.allclose(prepared.positions_raw[1], position)
    assert np.allclose(prepared.positions_folded[1], position)


def test_maptype_1_uses_cartesian_bond_vectors_without_basis_difference():
    prepared = prepare_positions([(1, 1, 0.0, 0.0, 0.0), (2, 2, 0.25, 0.0, 0.0)], np.eye(3))

    result = map_exchange_vector(1, 2, [0.75, 0, 0], positions=prepared, maptype=1, posfiletype="C")

    assert np.allclose(result, [0.75, 0, 0])


def test_duplicate_cartesian_vectors_use_the_last_jij(tmp_path: Path):
    jfile = tmp_path / "jfile"
    jfile.write_text("1 1 1 0 0 1\n1 1 1 0 0 2\n", encoding="utf-8")

    records = map_exchange_file(
        jfile,
        positions=[(1, "Fe", 0.0, 0.0, 0.0)],
        cell=np.eye(3),
        maptype=1,
    )

    assert len(records) == 1
    assert records[0].Jij == 2.0


def test_jfile_reader_ignores_trailing_columns_and_text(tmp_path: Path):
    jfile = tmp_path / "jfile"
    jfile.write_text("1 1 1 0 0 2 trailing columns and text\n", encoding="utf-8")

    parsed = read_jfile(jfile)
    mapped = map_exchange_file(
        jfile,
        positions=[(1, "Fe", 0.0, 0.0, 0.0)],
        cell=np.eye(3),
        maptype=1,
    )

    assert len(parsed) == 1
    assert parsed[0].Jij == 2.0
    assert mapped[0].Jij == 2.0
    assert mapped[0].supplied_distance is None


def test_periodic_offsets_are_reduced_and_free_offsets_are_rejected(tmp_path: Path):
    jfile = tmp_path / "jfile"
    jfile.write_text("1 1 2 0 0 1\n", encoding="utf-8")
    positions = [(1, 1, 0.0, 0.0, 0.0)]

    periodic = map_exchange_file(jfile, positions=positions, cell=np.eye(3), maptype=2, ncell=(2, 2, 2), bc=("P", "P", "P"))
    assert periodic[0].inferred_target_cell_offset == (0, 0, 0)

    with pytest.raises(VectorMappingError, match="outside the non-periodic supercell"):
        map_exchange_file(jfile, positions=positions, cell=np.eye(3), maptype=2, ncell=(2, 2, 2), bc=("F", "F", "F"))


def test_random_alloy_rows_retain_metadata_but_use_basis_atom_types(tmp_path: Path):
    jfile = tmp_path / "jfile"
    jfile.write_text("1 1 Fe Pt 1 0 0 3\n", encoding="utf-8")

    parsed = read_jfile(jfile)
    mapped = map_exchange_file(
        jfile,
        positions=[(1, "basis-Fe", 0.0, 0.0, 0.0)],
        cell=np.eye(3),
        maptype=1,
    )

    assert parsed[0].chemical_i == "Fe"
    assert parsed[0].chemical_j == "Pt"
    assert mapped[0].atom_type_i == "basis-Fe"
    assert mapped[0].atom_type_j == "basis-Fe"


def test_vector_mapping_reports_invalid_maptype_missing_site_and_unmatched_target():
    prepared = prepare_positions([(1, 1, 0.0, 0.0, 0.0)], np.eye(3))

    with pytest.raises(VectorMappingError, match="maptype must be 1, 2, or 3"):
        map_exchange_vector(1, 1, [0, 0, 0], positions=prepared, maptype=4)
    with pytest.raises(VectorMappingError, match="missing from positions"):
        map_exchange_vector(1, 2, [0, 0, 0], positions=prepared, maptype=1)
    with pytest.raises(VectorMappingError, match="does not match a basis site"):
        infer_target_site([0.25, 0.0, 0.0], prepared)
