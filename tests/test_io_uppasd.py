from pathlib import Path

import numpy as np
import pytest

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
