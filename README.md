---
title: Induced-Moment Exchange Explorer
emoji: 🧲
colorFrom: indigo
colorTo: teal
sdk: gradio
sdk_version: 5.50.0
python_version: "3.10"
app_file: app.py
fullWidth: true
short_description: UppASD exchange, induced response, downfolding, and FM magnons
---

# Induced-Moment Exchange Explorer

This repository is a conservative analysis tool for atomistic exchange models.
The validated Python package and the Hugging Face/Gradio Space expose UppASD
input parsing, reciprocal-space exchange, induced-moment response models,
variational downfolding, FM-compatible magnons, and dataset comparison.

## Run the Space

From the repository root:

```bash
python -m pip install -r requirements.txt
python app.py
```

The **Input** tab accepts an `inpsd.dat` plus its referenced files. References
are matched by basename, with explicit manual reassignment and unresolved-file
diagnostics. Individual `posfile`, `momfile`, and `jfile` uploads do not contain
the cell vectors needed for reciprocal-space analysis, so the UI requires
`inpsd.dat` and never silently substitutes an identity cell. CPU-friendly
examples are bundled under `examples/`.

The UI is orchestration only: all parsing and scientific calculations remain in
the `induced_exchange` package, and the existing unit tests can be run without
Gradio with `PYTHONPATH=src pytest -q`.

For Hugging Face Spaces, create a Gradio Space from this repository and push
the repository contents; the metadata above selects `app.py`, Gradio 5.50, and
Python 3.10. The Space provides its own URL, so local development does not
need `share=True`. See [the deployment audit](docs/dev/HF_DEPLOYMENT_AUDIT.md)
for the clean-build checklist and remaining release decisions.

## Scientific provenance

The induced-moment framing follows O. N. Mryasov, U. Nowak, K. Y. Guslienko,
and R. W. Chantrell, “Temperature-dependent magnetic properties of FePt:
Effective spin Hamiltonian model,” *Europhysics Letters* 69, 805 (2005),
[arXiv:physics/0411020](https://arxiv.org/abs/physics/0411020), and S. Polesya
*et al.*, “Finite-temperature magnetism of Fe$_x$Pd$_{1-x}$ and
Co$_x$Pt$_{1-x}$ alloys,” *Physical Review B* 82, 214409 (2010),
[arXiv:1008.3784](https://arxiv.org/abs/1008.3784).

These papers motivate treating moments on nominally non-magnetic sites as
induced/slave variables. This project does not claim that conventional input
`Jij` are exact first-principles susceptibilities: the UI labels `K = input Jij`
as a **J-weighted induced-response approximation**, and reports conditioning
and model-mismatch warnings.

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
alat      2.87e-10
posfile   ./posfile
exchange  ./jfile
momfile   ./momfile
```

When present, `alat` is the lattice-length unit in metres. Cell vectors,
Cartesian positions, and exchange displacements are converted from `alat`
units to metres by the loader, and the Input tab uses that value automatically
without requesting a separate length scale. If `alat` is absent, lengths are
retained in their supplied units and the length unit remains unspecified.

UppASD exchange files conventionally report `Jij` in mRy, so the UppASD loader,
CLI, and Space default to `mRy`. Pass `energy_unit="meV"` (or another explicit
unit) to `load_uppasd`, use `--energy-unit` on the CLI, or change the Input-tab
selector when the file uses a different unit.

`posfile` contains Cartesian basis positions:

```text
# site atom_type x y z
1 1 0.0 0.0 0.0
```

`momfile` contains a reference moment magnitude in `mu_B`; an optional initial
spin direction may follow it:

```text
# site moment_field moment [sx sy sz]
1 1 2.9913824 0.0 0.0 1.0
```

For species-aware symmetry handling, the second column of `posfile` is
authoritative (for example, the type labels `1` and `2` may represent Fe and
Pt in a particular setup). The second field in `momfile` is UppASD
moment-file metadata, not a species identifier, and does not determine the
spglib species mapping.

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

### Symmetry-reduced UppASD `jfile`

Some UppASD workflows write one exchange row per crystal-symmetry orbit rather
than every symmetry-related neighbour. The parser accepts the canonical
`posfile`/`momfile`/`exchange` keywords. The older `positions`/`moments`/`jfile`
spellings remain available as fallback aliases, and the canonical spelling
takes precedence if both are present. To expand a reduced `jfile`, install the
optional symmetry dependency (`python -m pip install -e '.[symmetry]'`) and
enable the opt-in spglib expansion:

```python
loaded = load_uppasd(
    "examples/uppasd_style/inpsd.dat",
    energy_unit="meV",
    expand_symmetry=True,
)
print(loaded.symmetry_expansion.as_dict())
```

The equivalent command-line switch is:

```bash
python -m induced_exchange.io_uppasd examples/uppasd_style/inpsd.dat \
    --energy-unit meV --expand-symmetry
```

The Input tab exposes the same operation as **Expand symmetry-reduced Jij with
spglib**, and the bundled `UppASD symmetry-reduced Jij` example can be selected
directly. Expansion rotates the complete Cartesian bond displacement and maps
the basis sites using the detected space group. It is not a value-fitting or
Hermitianization step: conflicting symmetry-equivalent values raise an error,
and unequal supplied reciprocal values remain visible as validation warnings.
Turn the option off when the `jfile` already contains the full neighbour set.

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

The Space uses `high_symmetry_path(..., use_seekpath=True)` as the common q
coordinate for its exchange-eigenvalue, induced-response, dressed-exchange,
and magnon plots. The displayed ordering candidate is therefore restricted to
that path, not a global reciprocal-space search. The Input-tab resolution
control sets points per path segment. The library's `regular_q_mesh` remains
available for users who need a 3-D scan.

## Induced/slave moment response

IMX-03 adds an explicit, user-controlled induced-moment layer. The library API
still requires robust and induced site indices (or named sublattice mappings)
explicitly. The Space Input tab provides an induced-site toggle for this
classification; it initially selects sites with reference moment `m < 0.5
mu_B` as induced and treats the rest as robust. Change the toggle for the
physical classification you want to analyze. The moment threshold is only a
workflow default, not a scientific classification rule.

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

## Collinear-FM magnons and stiffness

IMX-05 adds the FM-only adiabatic spin-wave layer in
`induced_exchange.magnons`. For the fixed Hamiltonian convention
`H = -1/2 sum_ij Jij e_i dot e_j`, the transverse harmonic matrix is

```text
A(q) = diag(J(0) 1) - J(q)
```

and the energy-valued, moment-normalized dynamical matrix is

```text
D(q) = g² M^(-1/2) A(q) M^(-1/2).
```

Here `M` contains the supplied moment magnitudes in `mu_B`. To match UppASD
AMS, `g_factor` is the site Landé factor and enters as the product of the two
site factors; the global default `g_factor=2.0` therefore gives `g_factor²=4`.
The resulting eigenvalues are `hbar*omega` in the declared exchange-energy
unit. The API retains that unit by default and only converts when both input
and output units are known. `meV` and `mRy` conversion is available through
`energy_conversion_factor`.

```python
from induced_exchange import fm_magnon_spectrum, fit_spin_stiffness, magnon_path_data

raw = fm_magnon_spectrum(loaded.model, q_points, model="raw")
robust = fm_magnon_spectrum(
    loaded.model, q_points, model="robust_only", robust_sites=[1]
)
stiffness = fit_spin_stiffness(raw, q_max=0.1)
# For a high-symmetry path, use magnon_path_data(raw, tick_indices=..., ...)
```

The raw model gives every site an ordinary dynamical degree of freedom. The
robust-only model drops induced sites. `model="mryasov"` and
`model="polesya"` both use the robust dressed interaction after analytical
elimination of induced variables; neither creates induced magnon branches.
Results expose acoustic/optical branches, the candidate ordering vector,
Goldstone and negative-mode diagnostics, and `stable=False` when the sampled
exchange does not support a collinear FM reference. A non-Gamma ordering
tendency is reported as signed harmonic data only; it is not labelled a
stable FM spectrum. The stiffness fit is through `E = D |q|^2` and stores the
user-visible Cartesian reciprocal `q_max` interval. The Space Magnons tab
shows the fitted `D` for every available spectrum and displays Mryasov and
Polesya/slave spectra together with their numerical maximum difference.

The UppASD-style example also includes an independent 177-point AMS reference
(`examples/uppasd_style/magnons.reference.txt`). It matches the raw two-rigid-
site calculation after the reduced `jfile` is expanded, with the reference
columns interpreted as meV and UppASD's explicit dynamical prefactor selected:

```python
import numpy as np
from induced_exchange import fm_magnon_spectrum, load_uppasd

q = np.loadtxt("examples/uppasd_style/qfile.kpath", skiprows=1, usecols=(0, 1, 2))
loaded = load_uppasd("examples/uppasd_style/inpsd.dat", expand_symmetry=True)
reference_check = fm_magnon_spectrum(loaded.model, q, model="raw", output_energy_unit="meV")
```

The package now uses the same default normalization as UppASD, so no external
scale factor is needed. In this run `qpoints.out` contains Cartesian reciprocal
vectors in UppASD's `1/alat` convention; the API's `q_cartesian` values are the
physical reciprocal vectors computed from the cell.

## Comparing two exchange datasets

IMX-06 adds conservative A/B diagnostics for cases such as LKAG versus
frozen-magnon `Jij`. The cell, site indices, and Cartesian basis positions are
checked before comparison; different exchange values are expected. Moment or
unit differences are retained as visible warnings, and no source is declared
causally wrong.

```python
from induced_exchange import ExchangeDataset, compare_exchange_datasets

a = ExchangeDataset(lkag_model, label="LKAG", robust_sites=[1], induced_sites=[2], x=0.5)
b = ExchangeDataset(frozen_magnon_model, label="frozen magnon", robust_sites=[1], induced_sites=[2], x=0.5)
comparison = compare_exchange_datasets(a, b, q_points, include_magnons=True, stiffness_q_max=0.1)
print(*comparison.diagnostics, sep="\n")
comparison.export("results", prefix="lkag_vs_frozen")
```

The result contains raw all-rigid, robust-only raw, and dressed `J_eff(q)`
eigenvalues and ordering vectors; optional raw/robust/dressed FM magnons and
stiffness; real-space `Jij` rows with authoritative distances and shell
numbers; and plotting helpers such as `plot_comparison(...,
observable="dressed")` and `plot_real_space_comparison(...)`. Export writes
CSV tables plus a JSON summary. In the Space Compare datasets tab, upload an
optional external response file in either supported format to display model
versus external response curves and relative-RMSE diagnostics.

In the Space Dressed exchange tab, `J_eff(q)` means the Fourier-space matrix in
the robust basis; the plotted branches are its leading eigenvalues. The tab
also shows shell-averaged real-space exchange channels reconstructed from a
separate complete auxiliary mesh: `J_MM(r)`, the cross block `J_Mm(r) = K_Mm(r)`,
and separately labelled but numerically equivalent Mryasov and Polesya effective
curves. The induced correction `Delta J_induced(r)` has its own plot so it does
not obscure the channel comparison, and a fourth panel shows the relative
correction `Delta J_induced(r) / J_MM(r)`. Real-space plot distances are divided
by the input `alat` when it is supplied, so the axes use the implicit UppASD
convention `alat = 1` and are labelled `Interatomic distance (alat)`.
This avoids pretending that a one-dimensional high-symmetry path uniquely
determines real-space exchange. For one robust site the result is the scalar
single-site effective exchange; for several robust sites the table retains the
matrix character. Magnon moment normalization is applied in the dynamical
matrix and is not hidden inside the displayed spin-Hamiltonian `J_eff(r)`.
For one positive-moment robust site the Space additionally displays
`J_eff(r)/m_R`, the q-dependent exchange scale entering that normalized FM
dynamical matrix.

For a coherent robust-site spiral, supply either an `ExternalInducedResponse`
or a file containing `qx qy qz m_ind`, or a two-column
`path_coordinate m_ind` table:

```python
comparison = compare_exchange_datasets(
    a, b, q_points,
    robust_configuration=spiral_amplitudes,
    external_response="dft_induced_response.dat",
)
print(comparison.induced_response.metrics_a)
```

When supplied response data strongly disagree with the `K = J` model, the
diagnostic says: “The input Jij do not reproduce the supplied induced-moment
response under the K=J approximation.” This is a model-mismatch statement,
not a claim that LKAG is wrong.

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
