# IMX-08 validation report

## Scope

This closeout audited the Hamiltonian and reciprocal-coordinate conventions,
pair counting, moment-normalized FM magnons, largest-eigenvalue ordering,
`X` inference and signs, induced-induced propagation, Schur-complement
downfolding, inverse Fourier reconstruction, Goldstone behavior, and
Polesya/Mryasov equivalence. The detailed scientific boundaries and adversarial
fixture expectations are in [`docs/PHYSICS_LIMITS.md`](../PHYSICS_LIMITS.md).

## Validation evidence

The IMX-08 adversarial tests are intentionally analytic and cover fixtures A–J
from the task prompt. They assert both numerical outcomes and the warnings
that should remain visible for singular, negative-sign, cancelled, or
incomplete inputs. Exported Space analyses now include an
`analysis_provenance` object and a standalone `analysis_provenance.json`.

The closeout commands are:

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python -c "import app; app.build_demo()"
```

The final run completed with **56 tests passed** and the Space smoke test
constructed a `Blocks` application successfully without launching a server.

## Interpretation

All passing checks establish internal consistency of the supplied effective
spin model and the documented approximation. They do not establish that
`K = Jij(input)` is the material's electronic susceptibility. Experimental
disagreement must therefore be investigated as possible kernel mismatch,
nonlinear/nonlocal response, or differing magnetic curvature before drawing
conclusions about induced moments.
