from __future__ import annotations

import numpy as np

from induced_exchange.model import ExchangeBond, MagneticCrystal, MagneticSite
from induced_exchange.reciprocal import (
    exchange_eigensystem,
    exchange_fourier,
    exchange_extrema,
    exchange_heatmap_data,
    high_symmetry_path,
    ordering_analysis,
    reciprocal_lattice,
    regular_q_mesh,
)


def crystal(cell, bonds, n_sites=1):
    sites = [MagneticSite(index=i + 1, atom_type=i + 1, position=(0.0, 0.0, 0.0), moment=1.0) for i in range(n_sites)]
    return MagneticCrystal(cell=np.asarray(cell, dtype=float), sites=sites, exchange_bonds=bonds)


def bond(i, j, displacement, jij):
    return ExchangeBond(i, j, tuple(displacement), jij)


def test_reciprocal_basis_has_explicit_two_pi_convention_for_nonorthogonal_cell():
    cell = np.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 2.0]])
    lattice = reciprocal_lattice(cell)
    assert np.allclose(cell @ lattice.reciprocal_vectors.T, 2 * np.pi * np.eye(3))
    reduced = np.array([[0.25, 0.5, 0.0]])
    assert np.allclose(lattice.cartesian_to_fractional(lattice.fractional_to_cartesian(reduced)), reduced)


def test_analytic_one_dimensional_nn_ferromagnet_orders_at_gamma():
    model = crystal(np.eye(3), [bond(1, 1, (1, 0, 0), 1), bond(1, 1, (-1, 0, 0), 1)])
    q = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    result = exchange_eigensystem(model, q, coordinates="fractional")
    assert np.allclose(result.eigenvalues[:, 0].real, [2.0, -2.0])
    diagnosis = ordering_analysis(model, q, coordinates="fractional")
    assert np.allclose(diagnosis.q_order_fractional, [0, 0, 0])
    assert diagnosis.gamma_locally_stable


def test_reciprocal_gamma_is_identified_modulo_integer_reduced_coordinates():
    model = crystal(np.eye(3), [bond(1, 1, (1, 0, 0), 1), bond(1, 1, (-1, 0, 0), 1)])
    diagnosis = ordering_analysis(model, [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]], coordinates="fractional")
    assert diagnosis.gamma_index == 0
    assert diagnosis.gamma_locally_stable


def test_analytic_one_dimensional_nn_antiferromagnet_orders_at_zone_boundary():
    model = crystal(np.eye(3), [bond(1, 1, (1, 0, 0), -1), bond(1, 1, (-1, 0, 0), -1)])
    q = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    diagnosis = ordering_analysis(model, q, coordinates="fractional")
    assert np.allclose(diagnosis.q_order_fractional, [0.5, 0, 0])
    assert diagnosis.gamma_locally_stable is False


def test_simple_cubic_nn_ferromagnet_and_regular_mesh():
    bonds = []
    for axis in np.eye(3):
        bonds.extend([bond(1, 1, axis, 1), bond(1, 1, -axis, 1)])
    model = crystal(np.eye(3), bonds)
    q = regular_q_mesh(model, (2, 2, 2), coordinates="fractional")
    diagnosis = ordering_analysis(model, q, coordinates="fractional")
    assert np.allclose(diagnosis.q_order_fractional, [0, 0, 0])
    assert np.isclose(diagnosis.eigenvalue.real, 6.0)
    extrema = exchange_extrema(exchange_eigensystem(model, q, coordinates="fractional"))
    assert np.isclose(extrema["maximum"]["value"].real, 6.0)
    assert np.isclose(extrema["minimum"]["value"].real, -6.0)


def test_two_sublattice_fourier_matrix_and_eigenvector():
    model = crystal(
        np.eye(3),
        [bond(1, 2, (0, 0, 0), 2), bond(2, 1, (0, 0, 0), 2)],
        n_sites=2,
    )
    result = exchange_eigensystem(model, [[0, 0, 0]], coordinates="fractional")
    assert np.allclose(result.matrices[0], [[0, 2], [2, 0]])
    assert np.isclose(result.eigenvalues[0, 0], 2)
    assert np.allclose(np.abs(result.eigenvectors[0, :, 0]), [1 / np.sqrt(2)] * 2)


def test_incomplete_dataset_is_not_silently_symmetrized():
    model = crystal(np.eye(3), [bond(1, 1, (1, 0, 0), 1)])
    result = exchange_fourier(model, [[0.25, 0, 0]], coordinates="fractional")
    assert not result.hermiticity.is_hermitian
    assert result.hermiticity.violating_indices == (0,)
    assert np.allclose(result.matrices[0, 0, 0], np.exp(0.5j * np.pi))


def test_pair_complete_dataset_is_hermitian_at_arbitrary_q():
    model = crystal(np.eye(3), [bond(1, 1, (1, 0, 0), 1), bond(1, 1, (-1, 0, 0), 1)])
    result = exchange_fourier(model, [[0.17, 0.23, 0.0]], coordinates="fractional")
    assert result.hermiticity.is_hermitian


def test_explicit_path_and_heatmap_data_are_numeric():
    model = crystal(np.eye(3), [bond(1, 1, (1, 0, 0), 1), bond(1, 1, (-1, 0, 0), 1)])
    path = high_symmetry_path(model, points={"Gamma": [0, 0, 0], "X": [0.5, 0, 0]}, n_per_segment=4, use_seekpath=False)
    assert path.q_fractional.shape == (5, 3)
    assert path.tick_labels == ("Gamma", "X")
    heatmap = exchange_heatmap_data(model, (3, 4))
    assert heatmap["max_eigenvalue"].shape == (3, 4)


def test_seekpath_maps_named_atom_types_to_species_numbers():
    model = MagneticCrystal(
        cell=np.eye(3),
        sites=[
            MagneticSite(index=1, atom_type="Fe", position=(0.0, 0.0, 0.0), moment=1.0),
            MagneticSite(index=2, atom_type="Pt", position=(0.5, 0.0, 0.0), moment=1.0),
        ],
        exchange_bonds=[],
    )

    path = high_symmetry_path(model, n_per_segment=2)

    assert path.source == "seekpath"
    assert path.tick_labels[0] == "GAMMA"
    assert "X" in path.tick_labels


def test_empty_q_set_has_a_structured_empty_result():
    model = crystal(np.eye(3), [])
    result = exchange_fourier(model, np.empty((0, 3)), coordinates="fractional")
    assert result.matrices.shape == (0, 1, 1)
    assert result.hermiticity.is_hermitian
