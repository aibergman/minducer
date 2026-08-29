from __future__ import annotations

import numpy as np

from induced_exchange import (
    ExchangeBond,
    InducedExchangeDownfolding,
    InducedMomentResponse,
    MagneticCrystal,
    MagneticSite,
    convert_exchange_to_uppasd,
    exchange_energy,
    exchange_fourier,
    fm_magnon_spectrum,
    local_exchange_field,
    mft_curie_energy,
    write_uppasd_jfile,
)
from induced_exchange.downfolding import inverse_fourier_dressed_jij
from induced_exchange.io_uppasd import parse_exchange


def site(index: int, moment: float = 1.0) -> MagneticSite:
    return MagneticSite(index, index, (0.0, 0.0, 0.0), moment, (0.0, 0.0, 1.0))


def bond(i: int, j: int, displacement, jij: float) -> ExchangeBond:
    return ExchangeBond(i, j, tuple(displacement), jij)


def chain(jij: float = 1.0, moment: float = 1.0) -> MagneticCrystal:
    return MagneticCrystal(
        np.eye(3),
        [site(1, moment)],
        [bond(1, 1, (1, 0, 0), jij), bond(1, 1, (-1, 0, 0), jij)],
    )


def test_literal_uppasd_chain_fourier_and_magnon_formula():
    model = chain()
    q = np.asarray([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    fourier = exchange_fourier(model, q, coordinates="fractional")
    assert np.allclose(fourier.matrices[:, 0, 0].real, [2.0, 0.0])

    result = fm_magnon_spectrum(model, q, g_factor=1.5)
    # 2*g/m [J(0)-J(q)] = 4*g*J/m [1-cos(qa)].
    assert np.allclose(result.energies[:, 0].real, [0.0, 6.0])


def test_non_default_g_scales_linearly_not_quadratically():
    model = chain()
    q = [[0.25, 0.0, 0.0]]
    g2 = fm_magnon_spectrum(model, q, g_factor=2.0).energies[0, 0].real
    g15 = fm_magnon_spectrum(model, q, g_factor=1.5).energies[0, 0].real
    assert np.isclose(g15 / g2, 1.5 / 2.0)


def test_native_energy_and_local_field_use_ordered_pair_factor_two():
    model = MagneticCrystal(
        np.eye(3),
        [site(1), site(2)],
        [bond(1, 2, (0, 0, 0), 3.0), bond(2, 1, (0, 0, 0), 3.0)],
    )
    spins = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
    assert np.isclose(exchange_energy(model, spins), -6.0)
    assert np.allclose(local_exchange_field(model, spins), [[0.0, 0.0, 6.0], [0.0, 0.0, 6.0]])
    assert np.isclose(model.exchange_energy(spins), -6.0)


def test_ordered_pair_local_field_matches_energy_second_difference():
    model = MagneticCrystal(
        np.eye(3),
        [site(1), site(2)],
        [bond(1, 2, (0, 0, 0), 3.0), bond(2, 1, (0, 0, 0), 3.0)],
    )
    step = 1.0e-5
    base = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    plus = np.asarray([[0.0, np.sin(step), np.cos(step)], [0.0, 0.0, 1.0]])
    minus = np.asarray([[0.0, -np.sin(step), np.cos(step)], [0.0, 0.0, 1.0]])
    curvature = (exchange_energy(model, plus) + exchange_energy(model, minus) - 2.0 * exchange_energy(model, base)) / step**2
    field_curvature = float(np.dot(local_exchange_field(model, base)[0], base[0]))
    assert np.isclose(curvature, field_curvature, rtol=1e-6)


def test_mft_normalization_and_boundary_conversions():
    assert np.isclose(mft_curie_energy(6.0), 4.0)
    assert np.isclose(convert_exchange_to_uppasd(4.0, source_convention="single_counted"), 2.0)
    assert np.isclose(convert_exchange_to_uppasd(4.0, source_convention="half_ordered"), 2.0)
    assert np.isclose(convert_exchange_to_uppasd(4.0, source_convention="af_positive"), -4.0)


def test_normalized_induced_response_infers_x_without_induced_moment_magnitude():
    model = MagneticCrystal(
        np.eye(3),
        [site(1, 3.0), site(2, 1.5)],
        [bond(2, 1, (0, 0, 0), 2.0), bond(1, 2, (0, 0, 0), 2.0)],
    )
    response = InducedMomentResponse(model, [1], [2])
    inference = response.infer_x()
    assert np.isclose(inference.x[2], 0.5)
    result = response.response_real_space([1.0])
    assert np.allclose(result.induced_polarizations, [1.0])
    assert np.allclose(result.physical_induced_moments, [1.5])
    vector_result = response.response_real_space([[0.0, 0.0, 1.0]])
    assert vector_result.physical_induced_moments.shape == (1, 3)
    assert np.allclose(vector_result.physical_induced_moments, [[0.0, 0.0, 1.5]])


def test_ordered_pair_downfolded_energy_and_native_jfile_round_trip(tmp_path):
    model = MagneticCrystal(
        np.eye(3),
        [site(1), site(2)],
        [
            bond(1, 1, (0, 0, 0), 1.5),
            bond(2, 1, (0, 0, 0), 2.0),
            bond(1, 2, (0, 0, 0), 2.0),
        ],
    )
    response = InducedMomentResponse(model, [1], [2], x=0.5)
    downfolding = InducedExchangeDownfolding(response)
    q = np.asarray([[0.0, 0.0, 0.0]])
    result = downfolding.evaluate(q)
    assert np.allclose(result.dressed, [[[3.5]]])
    check = downfolding.energy_equivalence(result, [1.7])
    assert check.equivalent
    assert np.allclose(check.explicit_energy, [-3.5 * 1.7**2])

    real_space = inverse_fourier_dressed_jij(result)
    output = write_uppasd_jfile(tmp_path / "dressed_jfile", real_space)
    rows = parse_exchange(output)
    assert len(rows) == 1
    assert np.isclose(rows[0].jij, 3.5)
    exported_model = MagneticCrystal(np.eye(3), [site(1)], rows)
    assert np.isclose(exported_model.exchange_energy([[0.0, 0.0, 1.0]]), -3.5)
