from __future__ import annotations

import numpy as np

from induced_exchange import (
    ExchangeBond,
    InducedMomentResponse,
    MagneticCrystal,
    MagneticSite,
    SublatticeClassification,
    instantaneous_slave_moments,
)


def site(index: int, moment: float, position=(0.0, 0.0, 0.0)) -> MagneticSite:
    return MagneticSite(index, index, position, moment, (0.0, 0.0, 1.0))


def crystal(sites, bonds) -> MagneticCrystal:
    return MagneticCrystal(np.eye(3), list(sites), list(bonds))


def test_one_robust_one_induced_infers_x_and_evaluates_slave_moment():
    model = crystal([site(1, 3.0), site(2, 1.5)], [ExchangeBond(2, 1, (0, 0, 0), 2.0)])
    response = InducedMomentResponse(model, [1], [2], mode="j_weighted")

    inference = response.infer_x()
    assert np.isclose(inference.x[2], 0.25)
    assert np.isclose(inference.source_fields[2], 6.0)
    assert np.allclose(response.response_real_space([4.0]).induced_moments, [2.0])


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

    assert np.isclose(response.infer_x().x[3], 1.0)
    assert np.allclose(response.response_real_space([1.0, 2.0]).induced_moments, [4.0])


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
