from __future__ import annotations

import numpy as np

from induced_exchange import (
    ExchangeBond,
    InducedMomentResponse,
    MagneticCrystal,
    MagneticSite,
    SublatticeClassification,
    instantaneous_slave_moments,
    reciprocal_lattice,
)
from induced_exchange.downfolding import InducedExchangeDownfolding, inverse_fourier_dressed_jij
from induced_exchange.io_uppasd import load_uppasd


def site(index: int, moment: float, position=(0.0, 0.0, 0.0)) -> MagneticSite:
    return MagneticSite(index, index, position, moment, (0.0, 0.0, 1.0))


def crystal(sites, bonds) -> MagneticCrystal:
    return MagneticCrystal(np.eye(3), list(sites), list(bonds))


def test_one_robust_one_induced_infers_x_and_evaluates_slave_moment():
    model = crystal([site(1, 3.0), site(2, 1.5)], [ExchangeBond(2, 1, (0, 0, 0), 2.0)])
    response = InducedMomentResponse(model, [1], [2], mode="j_weighted")

    inference = response.infer_x()
    assert np.isclose(inference.x[2], 0.5)
    assert np.isclose(inference.source_fields[2], 2.0)
    # The response API uses dimensionless robust orientation amplitudes and
    # returns normalized induced polarization p=m/|m0|.
    assert np.allclose(response.response_real_space([4.0]).induced_moments, [4.0])


def test_historical_equal_neighbours_parallel_and_antiparallel_cancel():
    model = crystal([site(1, 2.0), site(2, 2.0), site(3, 2.0)], [])
    response = InducedMomentResponse(
        model,
        {"robust": [1, 2]},
        {"induced": [3]},
        mode="historical",
        x={"induced": 1.0},
        neighbourhood=[1, 2],
    )

    assert np.allclose(response.response_real_space([1.0, 1.0]).induced_moments, [2.0])
    assert np.allclose(response.response_real_space([1.0, -1.0]).induced_moments, [0.0])
    assert response.classification.induced_sublattices == {"induced": (3,)}


def test_unequal_j_weights_are_used_in_reference_inference_and_response():
    model = crystal(
        [site(1, 1.0), site(2, 1.0), site(3, 3.0)],
        [ExchangeBond(3, 1, (0, 0, 0), 2.0), ExchangeBond(3, 2, (0, 0, 0), 1.0)],
    )
    response = InducedMomentResponse(model, [1, 2], [3])

    assert np.isclose(response.infer_x().x[3], 1.0 / 3.0)
    assert np.allclose(response.response_real_space([1.0, 2.0]).induced_moments, [4.0 / 3.0])


def test_finite_induced_induced_coupling_matches_direct_matrix_solution():
    model = crystal(
        [site(1, 1.0), site(2, 0.0), site(3, 0.0)],
        [
            ExchangeBond(2, 1, (0, 0, 0), 2.0),
            ExchangeBond(3, 1, (0, 0, 0), 1.0),
            ExchangeBond(2, 3, (0, 0, 0), 0.5),
            ExchangeBond(3, 2, (0, 0, 0), 0.25),
        ],
    )
    response = InducedMomentResponse(model, [1], [2, 3], x={2: 0.5, 3: 0.25})
    expected = np.linalg.solve(
        np.eye(2) - np.diag([0.5, 0.25]) @ np.array([[0.0, 0.5], [0.25, 0.0]]),
        np.diag([0.5, 0.25]) @ np.array([2.0, 1.0]),
    )
    result = response.response_real_space([1.0])
    assert np.allclose(result.induced_moments, expected)
    assert np.all(result.condition_numbers < 10.0)


def test_q_space_solution_matches_real_space_at_gamma_and_exposes_diagnostics():
    model = crystal(
        [site(1, 1.0), site(2, 1.0), site(3, 1.0)],
        [ExchangeBond(3, 1, (0, 0, 0), 2.0), ExchangeBond(3, 2, (0, 0, 0), 1.0)],
    )
    response = InducedMomentResponse(model, [1, 2], [3], x={3: 0.5})
    real = response.response_real_space([2.0, 1.0])
    reciprocal = response.response_q([[0.0, 0.0, 0.0]], [[2.0, 1.0]])
    assert np.allclose(reciprocal.induced_moments[0], real.induced_moments)
    assert np.allclose(reciprocal.m_ind_q_over_m_ind_0[0], [1.0])
    assert np.isclose(reciprocal.condition_numbers[0], 1.0)


def test_fractional_q_response_matches_explicitly_converted_cartesian_q():
    model = crystal(
        [site(1, 1.0), site(2, 1.0)],
        [
            ExchangeBond(2, 1, (1.0, 0.0, 0.0), 2.0),
            ExchangeBond(2, 1, (-1.0, 0.0, 0.0), 2.0),
            ExchangeBond(1, 2, (1.0, 0.0, 0.0), 2.0),
            ExchangeBond(1, 2, (-1.0, 0.0, 0.0), 2.0),
        ],
    )
    response = InducedMomentResponse(model, [1], [2], x=0.25)
    q_fractional = np.asarray([[0.25, 0.0, 0.0], [0.125, 0.25, 0.0]])
    q_cartesian = reciprocal_lattice(model.cell).fractional_to_cartesian(q_fractional)
    fractional = response.response_q(q_fractional, [[1.0], [0.5]], coordinates="fractional")
    cartesian = response.response_q(q_cartesian, [[1.0], [0.5]], coordinates="cartesian")

    assert np.allclose(fractional.q_cartesian, q_cartesian)
    assert np.allclose(fractional.induced_moments, cartesian.induced_moments)
    assert np.allclose(fractional.source_fields, cartesian.source_fields)


def test_pair_complete_j_weighted_downfolding_has_adjoint_cross_blocks_and_hermitian_dressing():
    model = crystal(
        [site(1, 2.0), site(2, 0.8)],
        [
            ExchangeBond(1, 1, (1.0, 0.0, 0.0), 8.0),
            ExchangeBond(1, 1, (-1.0, 0.0, 0.0), 8.0),
            ExchangeBond(2, 2, (1.0, 0.0, 0.0), 0.8),
            ExchangeBond(2, 2, (-1.0, 0.0, 0.0), 0.8),
            ExchangeBond(2, 1, (0.5, 0.0, 0.0), 1.2),
            ExchangeBond(2, 1, (-0.5, 0.0, 0.0), 1.2),
            ExchangeBond(1, 2, (0.5, 0.0, 0.0), 1.2),
            ExchangeBond(1, 2, (-0.5, 0.0, 0.0), 1.2),
        ],
    )
    q = [[0.25, 0.0, 0.0]]
    result = InducedExchangeDownfolding(
        InducedMomentResponse(model, [1], [2], mode="j_weighted", x=0.5)
    ).evaluate(q)

    assert np.allclose(result.k_rm, np.swapaxes(result.k_mr.conj(), 1, 2))
    assert result.dressed_hermiticity.is_hermitian
    assert not any("K_Mm differs" in warning for warning in result.warnings)


def test_induced_toy_nonzero_q_has_no_false_block_adjoint_warning():
    loaded = load_uppasd("examples/induced_toy/inpsd.dat", energy_unit="meV")
    response = InducedMomentResponse(loaded.model, [1], [2], mode="j_weighted", x=0.5)
    result = InducedExchangeDownfolding(response).evaluate([[0.25, 0.0, 0.0]])

    assert result.kernel_hermiticity.is_hermitian
    assert result.dressed_hermiticity.is_hermitian
    assert not any("K_Mm differs from K_mM" in warning for warning in result.warnings)


def test_convenience_wrapper_is_algebraic_and_does_not_propagate():
    model = crystal([site(1, 1.0), site(2, 2.0)], [ExchangeBond(2, 1, (0, 0, 0), 3.0)])
    values = instantaneous_slave_moments(model, [2.0], [1], [2], x={2: 0.5})
    assert np.allclose(values, [3.0])


def test_classification_is_explicit_and_never_based_on_moment_size():
    classification = SublatticeClassification.from_inputs({"Fe": [1]}, {"Pt": [2]})
    assert classification.robust_sites == (1,)
    assert classification.induced_sites == (2,)


def test_near_singular_induced_response_is_flagged_without_regularization():
    model = crystal(
        [site(1, 1.0), site(2, 1.0)],
        [ExchangeBond(2, 1, (0, 0, 0), 1.0), ExchangeBond(2, 2, (0, 0, 0), 2.0)],
    )
    response = InducedMomentResponse(model, [1], [2], x={2: 0.5})
    result = response.response_real_space([1.0])
    assert result.singular[0]
    assert any("not regularized" in warning for warning in result.warnings)
    assert np.isnan(result.induced_moments).all()


def test_variational_one_induced_site_matches_analytic_dressed_exchange_and_energy():
    model = crystal(
        [site(1, 1.0), site(2, 1.0)],
        [
            ExchangeBond(1, 1, (0, 0, 0), 1.5),
            ExchangeBond(2, 1, (0, 0, 0), 2.0),
            ExchangeBond(1, 2, (0, 0, 0), 2.0),
        ],
    )
    response = InducedMomentResponse(model, [1], [2], x={2: 0.5})
    downfolding = InducedExchangeDownfolding(response)
    result = downfolding.evaluate([[0.0, 0.0, 0.0]])

    # J_eff = J_MM + K_Mm X K_mM = 1.5 + 2 * 0.5 * 2.
    assert np.allclose(result.raw_robust[0], [[1.5]])
    assert np.allclose(result.delta_induced[0], [[2.0]])
    assert np.allclose(result.dressed[0], [[3.5]])
    check = downfolding.energy_equivalence(result, [1.7])
    assert check.equivalent
    assert np.allclose(check.stationary_induced, [[[1.7]]])


def test_downfolding_includes_induced_induced_propagation():
    model = crystal(
        [site(1, 1.0), site(2, 1.0), site(3, 1.0)],
        [
            ExchangeBond(2, 1, (0, 0, 0), 2.0),
            ExchangeBond(1, 2, (0, 0, 0), 2.0),
            ExchangeBond(3, 1, (0, 0, 0), 1.0),
            ExchangeBond(1, 3, (0, 0, 0), 1.0),
            ExchangeBond(2, 3, (0, 0, 0), 0.5),
            ExchangeBond(3, 2, (0, 0, 0), 0.5),
        ],
    )
    response = InducedMomentResponse(model, [1], [2, 3], x={2: 0.5, 3: 0.25})
    result = InducedExchangeDownfolding(response).evaluate([[0.0, 0.0, 0.0]])
    expected_operator = np.linalg.solve(
        np.eye(2) - np.diag([0.5, 0.25]) @ np.array([[0.0, 0.5], [0.5, 0.0]]),
        np.diag([0.5, 0.25]),
    )
    expected_delta = np.array([[2.0, 1.0]]) @ expected_operator @ np.array([[2.0], [1.0]])
    assert np.allclose(result.response_operator[0], expected_operator)
    assert np.allclose(result.delta_induced[0], expected_delta)


def test_downfolding_marks_singular_induced_block_without_regularization():
    model = crystal(
        [site(1, 1.0), site(2, 1.0)],
        [
            ExchangeBond(2, 2, (0, 0, 0), 2.0),
            ExchangeBond(2, 1, (0, 0, 0), 1.0),
            ExchangeBond(1, 2, (0, 0, 0), 1.0),
        ],
    )
    result = InducedExchangeDownfolding(
        InducedMomentResponse(model, [1], [2], x=0.5)
    ).evaluate([[0.0, 0.0, 0.0]])
    assert result.singular[0]
    assert np.isnan(result.dressed[0]).all()
    assert any("not regularized" in warning for warning in result.warnings)


def test_downfolding_has_no_dressing_without_induced_sites_or_when_x_is_zero():
    model = crystal(
        [site(1, 1.0), site(2, 1.0)],
        [ExchangeBond(1, 1, (0, 0, 0), 2.0), ExchangeBond(2, 1, (0, 0, 0), 4.0)],
    )
    no_induced = InducedExchangeDownfolding(
        InducedMomentResponse(model, [1], [], x=0.0)
    ).evaluate([[0.0, 0.0, 0.0]])
    zero_x = InducedExchangeDownfolding(
        InducedMomentResponse(model, [1], [2], x=0.0)
    ).evaluate([[0.0, 0.0, 0.0]])
    assert np.allclose(no_induced.dressed, no_induced.raw_robust)
    assert np.allclose(zero_x.dressed, zero_x.raw_robust)
    assert np.allclose(zero_x.delta_induced, 0.0)


def test_ordering_diagnostic_and_inverse_transform_are_explicit_about_sampling():
    model = crystal(
        [site(1, 1.0), site(2, 1.0)],
        [
            ExchangeBond(1, 1, (1, 0, 0), 1.0),
            ExchangeBond(1, 1, (-1, 0, 0), 1.0),
            ExchangeBond(2, 1, (0, 0, 0), 1.0),
            ExchangeBond(1, 2, (0, 0, 0), 1.0),
        ],
    )
    response = InducedMomentResponse(model, [1], [2], x=0.0)
    downfolding = InducedExchangeDownfolding(response)
    result = downfolding.evaluate([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    comparison = downfolding.ordering_comparison(result)
    assert comparison.changed is False
    assert comparison.diagnostic.startswith("No —")
    real_space = inverse_fourier_dressed_jij(result, displacements=[[1.0, 0.0, 0.0]])
    assert real_space.values.shape == (1, 1, 1)
    assert real_space.raw_values is not None
    assert real_space.delta_values is not None
    assert real_space.shell_diagnostics[0]["dressed_max_abs"] == 2.0
    assert any("finite-q reconstructions" in warning for warning in real_space.warnings)
