# Induced-Moment Exchange Explorer

This repository is being built as a conservative analysis tool for atomistic
exchange models. IMX-01 provides the input-level data model and UppASD-style
parsers, and IMX-02 provides reciprocal-space exchange analysis. Induced-moment
response models and magnons are separate later tasks.

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

## Reciprocal-space exchange

IMX-02 adds reciprocal-space analysis in `induced_exchange.reciprocal`.
The rows of the input `cell` matrix are the direct Cartesian lattice vectors.
The reciprocal rows `B` obey `A @ B.T = 2*pi*I`; reduced reciprocal
coordinates `h` and Cartesian coordinates are related by `q_cart = h @ B`.
For example:

```python
from induced_exchange import (
    exchange_eigensystem,
    high_symmetry_path,
    load_uppasd,
    ordering_analysis,
    regular_q_mesh,
)

loaded = load_uppasd("path/to/inpsd.dat")
q = regular_q_mesh(loaded.model, (16, 16, 16), coordinates="fractional")
eigensystem = exchange_eigensystem(loaded.model, q, coordinates="fractional")
diagnosis = ordering_analysis(loaded.model, q, coordinates="fractional")
path = high_symmetry_path(loaded.model)
```

The fixed convention is `H = -1/2 sum_ij Jij e_i dot e_j`, so the candidate
ordering vector is where the largest eigenvalue of `J(q)` occurs. Fourier
transforms retain incomplete or asymmetric bond input and report Hermiticity
violations; they are never silently symmetrized. `path_exchange_data` and the
`as_dict()`/`to_json()` methods provide plotting and downloadable numeric data.

## Induced/slave moment response

IMX-03 adds an explicit, user-controlled induced-moment layer. Sites are never
classified from their moment size: pass robust and induced site indices (or
named sublattice mappings) explicitly.

```python
from induced_exchange import InducedMomentResponse

response = InducedMomentResponse(
    loaded.model,
    robust_sites={"Fe": [1]},
    induced_sites={"Pt": [2]},
    mode="j_weighted",
    # Optional; otherwise inferred from the reference collinear state.
    x={"Pt": 0.12},
)
inference = response.infer_x()
real_space = response.response_real_space({1: [0.0, 0.0, 1.0]})
q_space = response.response_q([[0.0, 0.0, 0.0]], [[1.0]])
```

The `historical` (also called `unweighted`) mode evaluates
`m_nu = X_nu sum_j M_j` over a first geometric shell by default. Use
`neighbourhood=[...]`, a mapping per induced site, or `cutoff=...` to select
another neighbourhood. The `j_weighted` mode evaluates

```text
m(q) = [I - X K_mm(q)]^-1 X K_mM(q) M(q)
```

and labels the default `K = J_input` choice throughout the API as the
**J-weighted induced-response approximation**. Conventional LKAG `Jij` are not
formally identical to a true induction kernel. `X` is susceptibility-like and
has the inverse energy dimension implied by the selected input units; it is
not the induced moment itself. `infer_x()` reports each source field and warns
about cancellation, suspicious signs, and near-zero denominators.

The response is algebraic and instantaneous. Induced longitudinal amplitudes
are slave variables, not independent LLG/LSWT degrees of freedom. Both q-space
and real-space results expose condition numbers and singularity flags for
`I - X K_mm`; near-singular systems are reported and never silently
regularized.

## Variational induced-exchange downfolding

IMX-04 uses the same response object to eliminate the induced variables from a
single documented quadratic energy functional:

```text
E(M,m) = -1/2 M† J_MM M
       + 1/2 m† (X⁻¹ - K_mm) m
       - Re[m† K_mM M]
```

Stationarity gives
`m* = (I - X K_mm)⁻¹ X K_mM M`. Substitution gives the Schur-complement
interaction
`J_eff = J_MM + K_Mm (X⁻¹ - K_mm)⁻¹ K_mM`.
The implementation evaluates the algebraically equivalent stable form
`(I - X K_mm)⁻¹ X`, including the continuous `X -> 0` limit. This is a
variational downfolding of the **J-weighted induced-response approximation**
when `K = J_input`; it is not an exact first-principles susceptibility.

```python
from induced_exchange import InducedExchangeDownfolding

downfolding = InducedExchangeDownfolding(response)
result = downfolding.evaluate(q, coordinates="fractional")
result.raw_robust       # direct robust-only J_MM(q)
result.dressed          # J_eff(q)
result.delta_induced    # induced correction

energy_check = downfolding.energy_equivalence(result, [[1.0]])
ordering = downfolding.ordering_comparison(result)
real_space = downfolding.inverse_fourier(result)
```

`energy_equivalence()` checks the explicit stationary slave energy against the
downfolded quadratic energy. `ordering_comparison()` reports whether induced
dressing changes the leading ordering vector, including the case where both
raw and dressed calculations remain AF-like. Inverse-transformed dressed
`Jij` are finite-q reconstructions; the result carries warnings for
undersampled or non-regular q meshes and does not claim uniqueness beyond the
sampled resolution.
