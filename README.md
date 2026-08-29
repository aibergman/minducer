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
short_description: Explore UppASD exchange, induced response, downfolding, and FM magnons.
---

# Induced-Moment Exchange Explorer

Induced-Moment Exchange Explorer is a Python library and Gradio application for
analysing atomistic exchange models supplied in UppASD-style input files. It
keeps the supplied exchange data explicit, evaluates reciprocal-space exchange,
models selected sites as instantaneous induced moments, downfolds those sites,
and calculates collinear-FM magnon diagnostics.

The project is intended for transparent model analysis. It does not infer an
electronic susceptibility from a conventional `Jij` file or silently repair
incomplete exchange input.

## UppASD Hamiltonian convention

The native convention is the literal UppASD scalar-Heisenberg `jfile`
convention:

```text
H = - sum_(i != j) Jij e_i · e_j
```

The sum is over ordered pairs, so a pair-complete file contains both `(i,j)`
and `(j,i)`. Positive `Jij` is ferromagnetic. The parser stores the numerical
`jfile` values unchanged, `J(q)` is the Fourier transform of those literal
values, and exported dressed `jfile` values use the same convention without a
factor-of-two conversion.

The factor ledger is:

| quantity | factor | origin |
|---|---:|---|
| `J(q)` | `1` | literal `jfile` Fourier transform |
| local exchange field | `2` | derivative of the ordered-pair Hamiltonian |
| magnon energy | `2*g` | ordered-pair curvature times one gyromagnetic/Landé factor |
| global pair energy | ordered-pair sum | native UppASD convention |
| thermal white-noise factor | `2` | fluctuation-dissipation normalization; unrelated to pair counting |

For a different source convention, convert at the boundary with
`convert_exchange_to_uppasd`: a single-counted pair Hamiltonian
`-sum_<ij> J' e_i·e_j` and a `-1/2` ordered double sum both use
`J_UppASD = J'/2` (or `J''/2`). An AF-positive ordered convention requires a
sign change. A spin-`S` Hamiltonian written with unit directions first absorbs
the spin magnitudes into its pair coefficient and then applies the same
single-counted conversion.

## Start here

Run the application locally:

```bash
python -m pip install -r requirements.txt
python app.py
```

For library-only work, install the package and run the input inspector:

```bash
python -m pip install -e .
induced-exchange-uppasd examples/fept_style/inpsd.dat
```

The application includes small CPU-friendly examples under `examples/`. Upload
an `inpsd.dat` with its referenced `posfile`, `momfile`, and exchange file; the
explicit cell in `inpsd.dat` is required for reciprocal-space calculations.

## Documentation

- [Theory and scientific scope](docs/THEORY.md) — Hamiltonian, Fourier
  convention, induced response, downfolding, magnons, and interpretation
  limits.
- [Usage guide](docs/USAGE.md) — installation, input format, application
  workflow, CLI, and Python examples.
- [Development and deployment guide](docs/dev/DEPLOYMENT.md) — local workflow,
  release checks, and Hugging Face Spaces deployment.

## Validation

Run the test suite from the repository root:

```bash
PYTHONPATH=src pytest -q
```

The bundled tests cover input parsing, reciprocal-space conventions,
symmetry expansion, induced-response conditioning, variational downfolding,
and FM magnon behaviour.

## References

- O. N. Mryasov *et al.*, “Temperature-dependent magnetic properties of
  FePt: Effective spin Hamiltonian model,” *Europhysics Letters* **69**, 805
  (2005), [arXiv:physics/0411020](https://arxiv.org/abs/physics/0411020).
- S. Polesya *et al.*, “Finite-temperature magnetism of Fe$_x$Pd$_{1-x}$ and
  Co$_x$Pt$_{1-x}$ alloys,” *Physical Review B* **82**, 214409 (2010),
  [arXiv:1008.3784](https://arxiv.org/abs/1008.3784).
