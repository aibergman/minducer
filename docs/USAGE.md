# Usage guide

## Install

The interactive application includes all optional runtime dependencies:

```bash
python -m pip install -r requirements.txt
python app.py
```

For the library and command-line inspector only:

```bash
python -m pip install -e .
python -m pip install -e '.[symmetry,paths]'
```

The second command adds `spglib` for symmetry expansion and `seekpath` for
standard high-symmetry paths. Run tests with `PYTHONPATH=src pytest -q`.

## Prepare an UppASD input set

The loader uses the literal UppASD ordered-pair convention
`H = -sum_(i != j) Jij e_i·e_j`. Do not halve or double values from a
pair-complete jfile. Conversion helpers for other Hamiltonian conventions are
available as `convert_exchange_to_uppasd` at the Python API boundary.

The loader starts from `inpsd.dat`, which must name the position, moment, and
exchange files and provide the cell for reciprocal-space work:

```text
simid     FePtFM25
ncell     12 12 12
BC        P P P
cell      1.0 0.0 0.0
          0.5 0.5 0.0
          0.0 0.0 0.9525
alat      2.87e-10
posfile   ./posfile
exchange  ./jfile
momfile   ./momfile
```

Paths are resolved relative to `inpsd.dat`. Canonical UppASD keywords are
`posfile`, `momfile`, and `exchange`; `positions`, `moments`, and `jfile` are
accepted as fallback aliases.

`posfile` stores a basis-site number, atom type, and position. By default the
three position values are Cartesian (`posfiletype C`). With `posfiletype D`,
they are direct/fractional coordinates and are converted using the cell:

```text
posfiletype D
```

```text
# site atom_type x y z
1 1 0.0 0.0 0.0
```

The `jfile` vector convention is selected with `maptype` (default `1`):

- `maptype 1`: the vector is already a bond vector; it is Cartesian for
  `posfiletype C` and direct/fractional for `posfiletype D`.
- `maptype 2`: the vector contains lattice-translation coefficients and the
  folded basis-position difference is added.
- `maptype 3`: the vector contains lattice-translation coefficients and the
  raw, pre-folded basis-position difference is added.

For maptypes 2 and 3, `ncell N1 N2 N3` and `BC P/F P/F P/F` optionally enable
periodic offset reduction or free-boundary range checks. Cell vectors are rows
and all mapped exchange vectors are stored as Cartesian vectors internally.

`momfile` stores the reference moment in `mu_B`, optionally followed by a
spin direction:

```text
# site moment_field moment [sx sy sz]
1 1 2.9913824 0.0 0.0 1.0
```

The atom type in `posfile` identifies species for symmetry handling. The
second `momfile` field is UppASD metadata, not a species identifier.

The exchange file stores scalar isotropic exchange:

```text
# i j rx ry rz Jij [distance]
1 1 0.5 0.5 0.0 12.5 0.70710678
```

The displacement is used exactly as supplied. The optional distance is checked
only as a diagnostic.

## Use the web application

1. Run `python app.py` and open the local address printed by Gradio.
2. In **Input**, choose a bundled example or upload `inpsd.dat` plus its
   referenced files. Browser uploads are matched by basename; inspect and
   correct the mapping if needed.
3. Confirm the input energy unit. Choose **symmetry-reduced** only when the
   exchange file contains orbit representatives rather than all neighbours.
4. Review the proposed robust/induced sites and change them to match your
   physical model. The moment-size suggestion is not a physical conclusion.
5. Run the analysis. The tabs show raw exchange, induced response, dressed
   exchange, and FM magnon diagnostics; all successful-analysis data can be
   downloaded from the application.

Warnings are part of the result. In particular, address unresolved files,
asymmetric reciprocal bonds, ill-conditioned induced response, and a
path-restricted ordering candidate before drawing physical conclusions.

## Inspect input on the command line

```bash
induced-exchange-uppasd examples/fept_style/inpsd.dat
```

Specify units or expand a symmetry-reduced exchange file explicitly:

```bash
induced-exchange-uppasd examples/uppasd_style/inpsd.dat \
  --energy-unit meV --expand-symmetry
```

## Python workflow

```python
from induced_exchange import (
    InducedExchangeDownfolding,
    InducedMomentResponse,
    exchange_eigensystem,
    fm_magnon_spectrum,
    high_symmetry_path,
    load_uppasd,
)

loaded = load_uppasd("examples/fept_style/inpsd.dat", energy_unit="mRy")
model = loaded.model

path = high_symmetry_path(model, n_per_segment=16)
raw = exchange_eigensystem(model, path.q_fractional, coordinates="fractional")

response = InducedMomentResponse(
    model,
    robust_sites=[1],
    induced_sites=[2],
    mode="j_weighted",
    x={2: 0.12},  # or leave unset and inspect response.infer_x()
)
downfolded = InducedExchangeDownfolding(response).evaluate(
    path.q_fractional, coordinates="fractional"
)
magnons = fm_magnon_spectrum(
    downfolded,
    model="mryasov",
    moment_magnitudes=[model.site_by_index[1].moment],
    input_energy_unit=model.units.energy,
)
```

Use real site indices from the input, not zero-based array offsets. Read the
warnings and conditioning fields on returned objects. For a complete ordering
search, replace the path with `regular_q_mesh(model, (16, 16, 16),
coordinates="fractional")`.

## Compare datasets

To compare two exchange models sharing the same structure:

```python
from induced_exchange import ExchangeDataset, compare_exchange_datasets

a = ExchangeDataset(model_a, label="dataset A", robust_sites=[1], induced_sites=[2], x=0.12)
b = ExchangeDataset(model_b, label="dataset B", robust_sites=[1], induced_sites=[2], x=0.12)
result = compare_exchange_datasets(a, b, path.q_fractional, include_magnons=True)
result.export("results", prefix="a_vs_b")
```

Compatibility checks deliberately distinguish incompatible geometry/site
layouts from expected differences in exchange values. The comparison is a
diagnostic; it does not assign causal blame to either input dataset.
