"""Adversarial analytic limits for the IMX-08 scientific closeout."""

from __future__ import annotations

import json

import numpy as np

from induced_exchange import (
    ExchangeBond,
    InducedExchangeDownfolding,
    InducedMomentResponse,
    MagneticCrystal,
    MagneticSite,
    UnitMetadata,
    SublatticeClassification,
    exchange_fourier,
    exchange_eigensystem,
    reciprocal_lattice,
)
from induced_exchange.space import analyse_model
from induced_exchange.io_uppasd import load_uppasd
from induced_exchange.model import validate_model


def site(index: int, moment: float = 1.0) -> MagneticSite:
    return MagneticSite(index, index, (0.0, 0.0, 0.0), moment, (0.0, 0.0, 1.0))


def bond(i: int, j: int, displacement, jij: float) -> ExchangeBond:
    return ExchangeBond(i, j, tuple(displacement), jij)


def crystal(sites, bonds, cell=None) -> MagneticCrystal:
    return MagneticCrystal(np.eye(3) if cell is None else np.asarray(cell, dtype=float), list(sites), list(bonds), UnitMetadata(energy="meV"))


def robust_induced_model(jr: float, jc: float, induced_self: float = 0.0) -> MagneticCrystal:
    # The four cross rows make K_mM(q) = 2 jc cos(2 pi q_x), with its
    # Hermitian reciprocal block supplied explicitly.
    bonds = [
        bond(1, 1, (1, 0, 0), jr), bond(1, 1, (-1, 0, 0), jr),
        bond(2, 1, (1, 0, 0), jc), bond(2, 1, (-1, 0, 0), jc),
        bond(1, 2, (1, 0, 0), jc), bond(1, 2, (-1, 0, 0), jc),
    ]
    if induced_self:
        bonds.append(bond(2, 2, (0, 0, 0), induced_self))
    return crystal(
        [site(1), site(2)],
        bonds,
    )


def test_fixture_a_fm_raw_to_fm_dressed():
    model = robust_induced_model(1.0, 1.0)
    q = [[0, 0, 0], [0.5, 0, 0]]
    raw = exchange_eigensystem(model, q, coordinates="fractional")
    result = InducedExchangeDownfolding(InducedMomentResponse(model, [1], [2], x=0.2)).evaluate(q)
    assert np.argmax(raw.eigenvalues[:, 0].real) == 0
    assert np.argmax(result.raw_robust[:, 0, 0].real) == 0
    assert np.argmax(result.dressed[:, 0, 0].real) == 0
    assert np.all(np.isfinite(result.dressed))


def test_fixture_b_af_raw_to_fm_dressed():
    model = robust_induced_model(-1.0, 0.4, induced_self=-0.01)
    q = [[0, 0, 0], [0.5, 0, 0]]
    raw = exchange_eigensystem(model, q, coordinates="fractional")
    result = InducedExchangeDownfolding(InducedMomentResponse(model, [1], [2], x=10.0)).evaluate(q)
    assert np.argmax(raw.eigenvalues[:, 0].real) == 1
    assert np.argmax(result.raw_robust[:, 0, 0].real) == 1
    assert np.argmax(result.dressed[:, 0, 0].real) == 0


def test_fixture_c_af_raw_to_af_dressed():
    model = robust_induced_model(-1.0, 0.2)
    q = [[0, 0, 0], [0.5, 0, 0]]
    raw = exchange_eigensystem(model, q, coordinates="fractional")
    result = InducedExchangeDownfolding(InducedMomentResponse(model, [1], [2], x=0.5)).evaluate(q)
    assert np.argmax(raw.eigenvalues[:, 0].real) == 1
    assert np.argmax(result.raw_robust[:, 0, 0].real) == 1
    assert np.argmax(result.dressed[:, 0, 0].real) == 1


def test_fixture_d_nearly_singular_response_is_flagged_without_regularization():
    model = crystal(
        [site(1), site(2)],
        [bond(2, 1, (0, 0, 0), 1.0), bond(1, 2, (0, 0, 0), 1.0), bond(2, 2, (0, 0, 0), 1.999999999999)],
    )
    result = InducedMomentResponse(model, [1], [2], x=0.5).response_real_space([1.0])
    assert result.singular[0]
    assert np.isfinite(result.induced_moments).all()
    assert any("possible soft" in warning for warning in result.warnings)


def test_fixture_e_negative_inferred_susceptibility_is_visible():
    model = crystal([site(1, 1.0), site(2, -1.0)], [bond(2, 1, (0, 0, 0), 1.0)])
    inference = InducedMomentResponse(model, [1], [2]).infer_x()
    assert np.isclose(inference.x[2], -1.0)
    assert any("negative" in warning for warning in inference.per_site[2].warnings)


def test_fixture_f_symmetry_point_source_field_can_vanish_exactly():
    model = crystal(
        [site(1), site(2, 1.0)],
        [
            bond(2, 1, (1, 0, 0), 1.0), bond(2, 1, (-1, 0, 0), -1.0),
            bond(1, 2, (-1, 0, 0), 1.0), bond(1, 2, (1, 0, 0), -1.0),
        ],
    )
    response = InducedMomentResponse(model, [1], [2], x=0.5)
    assert response.infer_x().x[2] is None
    values = response.response_q([[0, 0, 0], [0.25, 0, 0]], [[1.0], [1.0]])
    assert np.isclose(values.source_fields[0], 0.0)
    assert not values.singular.any()


def test_goldstone_response_normalization_accepts_equivalent_gamma():
    model = crystal([site(1), site(2, 1.0)], [bond(2, 1, (0, 0, 0), 1.0)])
    values = InducedMomentResponse(model, [1], [2], x=0.5).response_q([[1, 0, 0], [0.25, 0, 0]], [[1.0], [1.0]])
    assert np.allclose(values.m_ind_q_over_m_ind_0[0], [1.0])


def test_fixture_g_induced_induced_block_close_to_instability_is_flagged():
    model = crystal(
        [site(1), site(2), site(3)],
        [
            bond(2, 1, (0, 0, 0), 1.0), bond(1, 2, (0, 0, 0), 1.0),
            bond(3, 1, (0, 0, 0), 1.0), bond(1, 3, (0, 0, 0), 1.0),
            bond(2, 3, (0, 0, 0), 1.999999999999), bond(3, 2, (0, 0, 0), 1.999999999999),
        ],
    )
    result = InducedMomentResponse(model, [1], [2, 3], x={2: 0.5, 3: 0.5}).response_q([[0, 0, 0]], [[1.0]])
    assert result.singular[0]
    assert any("possible soft" in warning for warning in result.warnings)


def test_fixture_h_multiple_induced_sublattices_are_retained():
    model = crystal(
        [site(1), site(2), site(3)],
        [
            bond(2, 1, (0, 0, 0), 1.0), bond(1, 2, (0, 0, 0), 1.0),
            bond(3, 1, (0, 0, 0), 2.0), bond(1, 3, (0, 0, 0), 2.0),
        ],
    )
    response = InducedMomentResponse(model, [1], {"A": [2], "B": [3]}, x={"A": 0.2, "B": 0.3})
    values = response.response_q([[0, 0, 0]], [[1.0]])
    dressed = InducedExchangeDownfolding(response).evaluate([[0, 0, 0]])
    assert values.induced_moments.shape == (1, 2)
    assert dressed.response_operator.shape == (1, 2, 2)
    assert response.classification.induced_sublattices == {"A": (2,), "B": (3,)}


def test_fixture_i_nonorthogonal_cartesian_phase_uses_authoritative_displacement():
    cell = [[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 2.0]]
    model = crystal([site(1)], [bond(1, 1, (0.5, 1.0, 0.0), 1.0), bond(1, 1, (-0.5, -1.0, 0.0), 1.0)], cell)
    lattice = reciprocal_lattice(cell)
    q = lattice.fractional_to_cartesian([[0.0, 0.5, 0.0]])
    result = exchange_fourier(model, q, coordinates="cartesian")
    assert np.allclose(result.matrices[0, 0, 0], -2.0)
    assert np.allclose(np.asarray(cell) @ lattice.reciprocal_vectors.T, 2 * np.pi * np.eye(3))


def test_fixture_j_asymmetric_input_is_reported_and_not_repaired():
    model = crystal([site(1)], [bond(1, 1, (1, 0, 0), 1.0)])
    report = validate_model(model)
    result = exchange_fourier(model, [[0.25, 0, 0]], coordinates="fractional")
    assert not report.real_space_hermitian
    assert any(issue.code == "missing_reciprocal" for issue in report.warnings)
    assert not result.hermiticity.is_hermitian
    assert np.allclose(result.matrices[0, 0, 0], 1j)


def test_analysis_export_contains_machine_readable_provenance():
    loaded = load_uppasd("examples/induced_toy/inpsd.dat", energy_unit="meV")
    session = analyse_model(loaded, SublatticeClassification.from_inputs([1], [2]), mesh_size=2, x=0.5)
    exported = json.loads((session.export_dir / "canonical_model.json").read_text(encoding="utf-8"))
    provenance = exported["analysis_provenance"]
    assert provenance["hamiltonian_convention"].startswith("H = -1/2")
    assert provenance["response_mode"] == "j_weighted"
    assert provenance["K_source"] == "input Jij (J-weighted induced-response approximation)"
    assert provenance["classification"] == {"robust_sites": [1], "induced_sites": [2]}
    assert (session.export_dir / "analysis_provenance.json").exists()
