"""Core data structures and input readers for Induced-Moment Exchange Explorer."""

from .model import (
    ExchangeBond,
    MagneticCrystal,
    MagneticSite,
    UnitMetadata,
    ValidationReport,
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
    "load_uppasd",
    "parse_exchange",
    "parse_inpsd",
    "parse_jfile",
    "parse_momfile",
    "parse_posfile",
    "parse_uppasd",
]
