from __future__ import annotations

import numpy as np

from induced_exchange import (
    ExchangeBond,
    InducedExchangeDownfolding,
    InducedMomentResponse,
    MagneticCrystal,
    MagneticSite,
    UnitMetadata,
    fm_magnon_spectrum,
    fit_spin_stiffness,
    magnon_path_data,
)


def crystal(sites, bonds, *, energy="unspecified"):
    return MagneticCrystal(np.eye(3), list(sites), list(bonds), UnitMetadata(energy=energy))


def site(index, moment=1.0):
    return MagneticSite(index, index, (0.0, 0.0, 0.0), moment, (0.0, 0.0, 1.0))


def bond(i, j, displacement, jij):
    return ExchangeBond(i, j, tuple(displacement), jij)


def test_one_sublattice_nn_fm_has_analytic_moment_normalized_dispersion():
    model = crystal([site(1)], [bond(1, 1, (1, 0, 0), 1.0), bond(1, 1, (-1, 0, 0), 1.0)])
    q = [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]]
    result = fm_magnon_spectrum(model, q)

    # hbar*omega = g/mu * [J(0)-J(q)] = 2 * [2 - 2 cos(2 pi h)].
    assert np.allclose(result.energies[:, 0].real, [0.0, 4.0, 8.0])
    assert result.goldstone_ok
    assert result.stable


def test_one_sublattice_multi_shell_fm_adds_each_shell_with_the_same_convention():
    model = crystal(
        [site(1)],
        [bond(1, 1, (1, 0, 0), 1.0), bond(1, 1, (-1, 0, 0), 1.0), bond(1, 1, (2, 0, 0), 0.5), bond(1, 1, (-2, 0, 0), 0.5)],
    )
    result = fm_magnon_spectrum(model, [[0.25, 0.0, 0.0]])
    expected = 2.0 * ((2.0 + 1.0) - (2.0 * np.cos(np.pi / 2.0) + 1.0 * np.cos(np.pi)))
    assert np.isclose(result.energies[0, 0].real, expected)


def test_robust_only_raw_drops_induced_site_from_the_dynamical_basis():
    model = crystal(
        [site(1, 2.0), site(2, 0.5)],
        [bond(1, 1, (1, 0, 0), 1.0), bond(1, 1, (-1, 0, 0), 1.0), bond(1, 2, (0, 0, 0), 2.0), bond(2, 1, (0, 0, 0), 2.0)],
    )
    result = fm_magnon_spectrum(model, [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]], model="robust_only", robust_sites=[1])
    assert result.site_indices == (1,)
    assert result.moment_magnitudes.tolist() == [2.0]
    assert result.energies.shape == (2, 1)


def test_two_rigid_sublattices_have_acoustic_and_optical_branches():
    model = crystal(
        [site(1), site(2)],
        [bond(1, 2, (0, 0, 0), 1.0), bond(2, 1, (0, 0, 0), 1.0)],
    )
    result = fm_magnon_spectrum(model, [[0.0, 0.0, 0.0]])
    assert result.energies.shape == (1, 2)
    assert np.allclose(result.energies[0].real, [0.0, 4.0])
    assert result.goldstone_ok


def test_moment_scaling_and_unit_conversion_are_explicit():
    model = crystal(
        [site(1, moment=2.0)],
        [bond(1, 1, (1, 0, 0), 1.0), bond(1, 1, (-1, 0, 0), 1.0)],
        energy="mRy",
    )
    mry = fm_magnon_spectrum(model, [[0.25, 0.0, 0.0]])
    mev = fm_magnon_spectrum(model, [[0.25, 0.0, 0.0]], output_energy_unit="meV")
    assert np.isclose(mry.energies[0, 0].real, 2.0)
    assert np.isclose(mev.energies[0, 0].real, 2.0 * 13.605693009)
    assert mev.energy_unit == "meV"


def test_non_gamma_ordering_is_flagged_as_not_a_stable_fm_spectrum():
    model = crystal([site(1)], [bond(1, 1, (1, 0, 0), -1.0), bond(1, 1, (-1, 0, 0), -1.0)])
    result = fm_magnon_spectrum(model, [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    assert not result.fm_compatible
    assert not result.stable
    assert result.candidate_order_fractional is not None
    assert result.candidate_order_fractional[0] == 0.5
    assert result.energies[1, 0].real < 0.0
    assert any("not Gamma" in warning for warning in result.warnings)


def test_stiffness_fit_uses_visible_user_interval_and_has_expected_long_wave_limit():
    model = crystal([site(1)], [bond(1, 1, (1, 0, 0), 1.0), bond(1, 1, (-1, 0, 0), 1.0)])
    q = [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0], [0.03, 0.0, 0.0]]
    result = fm_magnon_spectrum(model, q)
    stiffness = fit_spin_stiffness(result, q_max=0.25)
    assert stiffness.point_count == 3
    assert np.isclose(stiffness.D, 2.0, rtol=5e-3)
    assert stiffness.q_max == 0.25


def test_polesya_and_mryasov_physical_spectra_are_identical_without_slave_branches():
    model = crystal(
        [site(1, 1.0), site(2, 1.0)],
        [
            bond(1, 1, (1, 0, 0), 1.0),
            bond(1, 1, (-1, 0, 0), 1.0),
            bond(2, 1, (0, 0, 0), 2.0),
            bond(1, 2, (0, 0, 0), 2.0),
        ],
    )
    response = InducedMomentResponse(model, [1], [2], x=0.5)
    downfolding = InducedExchangeDownfolding(response)
    q = [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]]
    mryasov = fm_magnon_spectrum(downfolding, q, model="mryasov")
    polesya = fm_magnon_spectrum(downfolding, q, model="polesya")
    assert mryasov.energies.shape == (3, 1)
    assert np.allclose(mryasov.energies, polesya.energies)
    assert len(polesya.site_indices) == 1
    assert "induced variables eliminated" in polesya.model_label


def test_direct_dressed_construction_and_path_data_api():
    model = crystal(
        [site(1, 1.0), site(2, 1.0)],
        [bond(1, 1, (1, 0, 0), 1.0), bond(1, 1, (-1, 0, 0), 1.0), bond(2, 1, (0, 0, 0), 2.0), bond(1, 2, (0, 0, 0), 2.0)],
    )
    result = fm_magnon_spectrum(
        model,
        [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]],
        model="polesya",
        robust_sites=[1],
        induced_sites=[2],
        x=0.5,
    )
    data = magnon_path_data(result, tick_indices=[0, 1], tick_labels=["Gamma", "X"])
    assert data["energies"].shape == (2, 1)
    assert np.allclose(data["tick_distances"], [0.0, np.pi / 2.0])
    assert data["tick_labels"] == ("Gamma", "X")
