Task IMX-08 — Scientific validation and closeout.

Do not add major functionality in this task.

Goal:
Audit the complete implementation against analytic limits and identify situations in which the output is mathematically correct but physically misleading.

Perform an independent review of:

1. Hamiltonian convention and all factors of 2.
2. Cartesian/direct reciprocal-coordinate handling.
3. Jij pair counting.
4. moment normalization in magnons.
5. J(q) ordering criterion.
6. X inference.
7. signs in induced response.
8. Schur-complement/downfolding signs and factors.
9. induced-induced propagation.
10. inverse Fourier transform.
11. Goldstone behavior.
12. Polesya/Mryasov equivalence.

Add adversarial fixtures:

A. FM raw -> FM dressed.
B. AF raw -> FM dressed.
C. AF raw -> AF dressed.
D. Nearly singular induced response.
E. Negative inferred susceptibility due to inconsistent input.
F. J_mM whose q-dependent vector sum vanishes at a symmetry point.
G. Induced-induced coupling close to instability.
H. Multiple induced sublattices.
I. Nonorthogonal tetragonal/primitive representation.
J. Asymmetric/incomplete Jij input.

For each fixture document:
- expected physics;
- expected numerical output;
- expected warning, if any.

Add a machine-readable `analysis_provenance` object to exported results containing:
- Hamiltonian convention;
- units;
- response mode;
- K source;
- X source;
- robust/induced classification;
- q mesh;
- numerical tolerances.

Write `docs/PHYSICS_LIMITS.md` explaining prominently:

The program can exactly manipulate the supplied effective spin model, but it cannot determine the true electronic susceptibility from Jij alone.

In particular, failure of a J-weighted Mryasov/Polesya reconstruction to reproduce experiment does not by itself prove that induced moments are irrelevant. It may instead show that:
    K_ij != Jij_input,
that the response is nonlinear/nonlocal in a way not captured by the model, or that the underlying Jij represent a different magnetic curvature.

Run the full test suite and produce a concise validation report.

Do not weaken failing tests merely to obtain green CI.

Checklist:
[x] All conventions independently audited.
[x] Analytic fixtures complete.
[x] Adversarial fixtures complete.
[x] Provenance metadata implemented.
[x] Physics limitations documented.
[x] Full test suite passes.
[x] Space smoke test passes.

Commit message:
IMX-08 validate induced exchange physics and conventions
