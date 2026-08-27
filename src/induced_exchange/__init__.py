"""Core data structures and input readers for Induced-Moment Exchange Explorer."""

from .model import (
    ExchangeBond,
    MagneticCrystal,
    MagneticSite,
    UnitMetadata,
    ValidationReport,
)
from .reciprocal import (
    ExchangeEigenSystem,
    ExchangePathData,
    FourierExchangeResult,
    HermiticityReport,
    OrderingAnalysis,
    QPath,
    ReciprocalLattice,
    check_hermiticity,
    compute_jq,
    compute_ordering,
    compute_reciprocal_lattice,
    exchange_eigensystem,
    exchange_fourier,
    exchange_extrema,
    exchange_heatmap_data,
    fourier_exchange,
    fourier_transform,
    generate_q_mesh,
    high_symmetry_path,
    ordering_analysis,
    path_exchange_data,
    plot_exchange_path,
    q_mesh,
    reciprocal_lattice,
    reciprocal_mesh,
    regular_q_mesh,
)

_IO_EXPORTS = {
    "InputFormatError",
    "InpsdConfig",
    "LoadedUppASD",
    "load_uppasd",
    "parse_exchange",
    "parse_inpsd",
    "parse_jfile",
    "parse_momfile",
    "parse_posfile",
    "parse_uppasd",
}


def __getattr__(name: str):
    # Lazy imports keep ``python -m induced_exchange.io_uppasd`` free of the
    # runpy warning caused by importing the module before executing it.
    if name in _IO_EXPORTS:
        from . import io_uppasd

        return getattr(io_uppasd, name)
    raise AttributeError(name)

__all__ = [
    "ExchangeBond",
    "InputFormatError",
    "InpsdConfig",
    "LoadedUppASD",
    "MagneticCrystal",
    "MagneticSite",
    "UnitMetadata",
    "ValidationReport",
    "ExchangeEigenSystem",
    "ExchangePathData",
    "FourierExchangeResult",
    "HermiticityReport",
    "OrderingAnalysis",
    "QPath",
    "ReciprocalLattice",
    "check_hermiticity",
    "compute_jq",
    "compute_ordering",
    "compute_reciprocal_lattice",
    "exchange_eigensystem",
    "exchange_fourier",
    "exchange_extrema",
    "exchange_heatmap_data",
    "fourier_exchange",
    "fourier_transform",
    "generate_q_mesh",
    "high_symmetry_path",
    "ordering_analysis",
    "path_exchange_data",
    "plot_exchange_path",
    "q_mesh",
    "reciprocal_lattice",
    "reciprocal_mesh",
    "regular_q_mesh",
    "load_uppasd",
    "parse_exchange",
    "parse_inpsd",
    "parse_jfile",
    "parse_momfile",
    "parse_posfile",
    "parse_uppasd",
]
