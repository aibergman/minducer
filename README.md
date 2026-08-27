# Induced-Moment Exchange Explorer

This repository is being built as a conservative analysis tool for atomistic
exchange models. IMX-01 provides the input-level data model and UppASD-style
parsers. Induced-moment response models, Fourier analysis, and magnons are
separate later tasks.

## Input format

The primary input is an `inpsd.dat` file. Paths are resolved relative to that
file, and the three cell rows may be written on the same line or as a
multiline block:

```text
simid     FePtFM25
ncell     12 12 12
BC        P P P
cell      1.0 0.0 0.0
          0.5 0.5 0.0
          0.0 0.0 0.9525
posfile   ./posfile
exchange  ./jfile
momfile   ./momfile
```

`posfile` contains Cartesian basis positions:

```text
# site atom_type x y z
1 1 0.0 0.0 0.0
```

`momfile` contains a reference moment magnitude in `mu_B`; an optional initial
spin direction may follow it:

```text
# site atom_type moment [sx sy sz]
1 1 2.9913824 0.0 0.0 1.0
```

The exchange file contains scalar isotropic entries:

```text
# i j rx ry rz Jij [distance]
1 1 0.5 0.5 0.0 12.5 0.70710678
```

The displacement `(rx, ry, rz)` is authoritative and is not reconstructed from
the basis positions. The optional distance is checked against its Euclidean
norm and only used for diagnostics. No chemical element is inferred from an
atom-type number, and non-orthogonal cells and multiple basis sites are
supported.

## Inspect an input set

Install the package in editable mode once from the repository root:

```bash
python -m pip install -e .
```

Then inspect an input set with:

```bash
python -m induced_exchange.io_uppasd path/to/inpsd.dat
```

The command prints the cell, basis/moment/bond counts, bond-distance range, and
structured validation diagnostics including duplicates, self interactions,
missing moments, and likely reciprocal `+/-R` partners.
