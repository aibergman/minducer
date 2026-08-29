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
