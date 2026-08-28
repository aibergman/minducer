Task IMX-04 — Implement variational induced-moment downfolding.

Read HEAD and IMX-03 implementation first.

Goal:
From the same quadratic static-response model used by the slave-moment representation, analytically eliminate induced moments and obtain a dressed robust-spin interaction.

Do not derive the formula independently from the Polesya implementation by copying expected expressions.
Start from one documented quadratic energy functional and derive both the stationary induced moment and the effective robust-spin energy from it.

Use block notation:
    M = robust moments
    m = induced moments.

Implement the model in a convention-consistent form equivalent to:

    E(M,m)
      =
      E_MM(M)
      + 1/2 m^T A m
      - M^T K_Mm m
      - appropriate induced-induced terms

with:

    X = A^{-1}

or the generalized equivalent chosen by the implementation.

Derive:

    m*(q)

and then:

    E_eff(M) = E(M,m*(M)).

Obtain:

    J_eff_MM(q)

including induced-induced propagation.

The expected structural form is:

    J_eff_MM(q)
      =
      J_MM(q)
      +
      K_Mm(q)
      Xi_m(q)
      K_mM(q)

where:

    Xi_m(q)
      =
      [X^{-1} - K_mm(q)]^{-1

modulo the exact signs/factors dictated by the selected Hamiltonian.

Do not hard-code this expression until the energy derivation is verified.

Implement:

1. raw robust-only J_MM(q);
2. dressed J_eff_MM(q);
3. delta J_induced(q);
4. optional inverse Fourier transform to dressed real-space Jij;
5. real-space shell/range diagnostics for raw vs dressed exchange.

Inverse transform:
- use a controlled q mesh;
- report truncation/aliasing limitations;
- do not pretend the reconstructed dressed Jij are unique if the q sampling is insufficient.

Core validation:
For the same X and K, the stationary energy of the explicit slave model and the downfolded model must agree to numerical precision for random robust-spin perturbations within the linear model.

Also verify:
- no induced sites => J_eff = J_MM;
- X -> 0 => J_eff -> J_MM;
- simple one-induced-site analytic solution;
- induced-induced propagation;
- translational/Hermiticity properties.

Add a diagnostic:
    "Does induced-moment dressing alter the predicted ordering vector?"

This is especially important for cases such as FePt.

If raw and dressed both predict AF, report that clearly rather than trying to force FM stabilization.

Checklist:
[ ] Common quadratic energy functional documented.
[ ] Stationary slave solution derived.
[ ] Schur-complement/downfolding implemented.
[ ] Raw/dressed/delta J(q) available.
[ ] Explicit-vs-downfolded energy equivalence tested.
[ ] Ordering comparison implemented.
[ ] Optional dressed real-space Jij export implemented.
[ ] Numerical conditioning diagnostics implemented.
[ ] Documentation updated.

Commit message:
IMX-04 add Mryasov-style induced exchange downfolding
