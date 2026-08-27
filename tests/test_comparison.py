from __future__ import annotations

import numpy as np

from induced_exchange import (
    ExchangeBond,
    ExchangeDataset,
    ExternalInducedResponse,
    MagneticCrystal,
    MagneticSite,
    UnitMetadata,
    compare_exchange_datasets,
    compare_induced_response,
    validate_dataset_compatibility,
)


def site(index: int, moment: float = 1.0, position=(0.0, 0.0, 0.0)) -> MagneticSite:
    return MagneticSite(index, index, position, moment, (0.0, 0.0, 1.0))


def bond(i: int, j: int, displacement, jij: float) -> ExchangeBond:
    return ExchangeBond(i, j, tuple(displacement), jij)


def crystal(bonds, *, moments=(1.0,), cell=None, energy="unspecified") -> MagneticCrystal:
    if cell is None:
        cell = np.eye(3)
    sites = [site(index + 1, moment) for index, moment in enumerate(moments)]
    return MagneticCrystal(np.asarray(cell, dtype=float), sites, list(bonds), UnitMetadata(energy=energy))


def test_identical_datasets_have_equal_raw_robust_dressed_and_export_tables(tmp_path):
    model = crystal(
        [bond(1, 1, (1, 0, 0), 1.0), bond(1, 1, (-1, 0, 0), 1.0), bond(2, 1, (0, 0, 0), 2.0), bond(1, 2, (0, 0, 0), 2.0)],
        moments=(1.0, 1.0),
    )
    result = compare_exchange_datasets(
        ExchangeDataset(model, label="LKAG", robust_sites=(1,), induced_sites=(2,), x=0.5),
        ExchangeDataset(model, label="frozen magnon", robust_sites=(1,), induced_sites=(2,), x=0.5),
        [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]],
        include_magnons=False,
    )
    assert result.compatibility.compatible
    assert np.allclose(result.raw_a, result.raw_b)
    assert np.allclose(result.robust_a, result.robust_b)
    assert np.allclose(result.dressed_a, result.dressed_b)
    written = result.export(tmp_path)
    assert (tmp_path / "imx06_raw_jq_eigenvalues.csv").exists()
    assert written["summary"].exists()


def test_structural_compatibility_rejects_different_basis_but_allows_exchange_difference():
    a = crystal([], moments=(1.0,))
    b = crystal([], moments=(1.0,), cell=[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    report = validate_dataset_compatibility(a, b)
    assert not report.compatible
    assert any("cells" in error for error in report.errors)


def test_sign_reversed_dataset_changes_raw_ordering_diagnostic():
    a = crystal([bond(1, 1, (1, 0, 0), 1.0), bond(1, 1, (-1, 0, 0), 1.0)])
    b = crystal([bond(1, 1, (1, 0, 0), -1.0), bond(1, 1, (-1, 0, 0), -1.0)])
    result = compare_exchange_datasets(a, b, [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], include_magnons=False, stiffness_q_max=None)
    assert result.dataset_a.raw_ordering.kind == "FM at Gamma"
    assert result.dataset_b.raw_ordering.kind == "AF-like/non-Gamma"
    assert any("Dataset B raw ordering" in item for item in result.diagnostics)


def test_response_comparison_matching_and_mismatch_are_quantified():
    model = crystal([bond(2, 1, (0, 0, 0), 2.0), bond(1, 2, (0, 0, 0), 2.0)], moments=(1.0, 1.0))
    from induced_exchange import InducedMomentResponse

    response_a = InducedMomentResponse(model, [1], [2], x=0.5)
    response_b = InducedMomentResponse(model, [1], [2], x=0.5)
    q = [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]
    matching = compare_induced_response(response_a, response_b, q, [[1.0], [1.0]], external=ExternalInducedResponse([1.0, 1.0], q=np.asarray(q)))
    assert matching.metrics_a is not None and matching.metrics_a.rmse == 0.0
    mismatching = compare_induced_response(response_a, response_b, q, [[1.0], [1.0]], external=ExternalInducedResponse([1.0, 2.0], q=np.asarray(q)))
    assert mismatching.metrics_a is not None and mismatching.metrics_a.strongly_disagrees
    assert any("do not reproduce" in warning for warning in mismatching.warnings)


def test_external_response_file_accepts_both_supported_layouts(tmp_path):
    q_file = tmp_path / "q_response.dat"
    q_file.write_text("0 0 0 1\n0.5 0 0 0.2\n", encoding="utf-8")
    path_file = tmp_path / "path_response.dat"
    path_file.write_text("0 1\n1 0.2\n", encoding="utf-8")
    assert ExternalInducedResponse.from_file(q_file).q.shape == (2, 3)
    assert ExternalInducedResponse.from_file(path_file).path_coordinate.tolist() == [0.0, 1.0]
