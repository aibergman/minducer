# Physics limits and validation boundaries

## Central limitation

The program can exactly manipulate the supplied effective spin model: it can
Fourier transform the supplied Cartesian `Jij`, solve the selected linear
slave-response model, and perform the corresponding Schur-complement
downfolding. It cannot determine the true electronic susceptibility from
`Jij` alone.

The default induced-response calculation is therefore labelled throughout as
the **J-weighted induced-response approximation**, with `K_ij = Jij(input)`.
It is not an exact susceptibility identity. In particular, failure of a
J-weighted Mryasov/Polesya reconstruction to reproduce experiment does not by
itself prove that induced moments are irrelevant. It may instead show that

* `K_ij != Jij_input`;
* the response is nonlinear or nonlocal in a way not represented by this
  model; or
* the supplied `Jij` represent a different magnetic curvature.

These alternatives cannot be distinguished from `Jij` alone. Results should
be interpreted as diagnostics of the explicitly selected model and its
conditioning, not as a first-principles electronic response calculation.

## Conventions audited for IMX-08

The implementation uses one convention everywhere:

```text
H = -1/2 sum_ij Jij e_i dot e_j
```

Each input row is one term in that sum. If both reciprocal rows are supplied,
the factor `1/2` prevents double counting; rows are never silently duplicated
or symmetrized. The Fourier transform uses the supplied Cartesian
displacement verbatim,

```text
J_ab(q) = sum_R J_ab(R) exp(+i q dot r_ab(R)),
```

and the inverse reconstruction uses the matching negative phase on the
sampled mesh. The direct-cell matrix has Cartesian vectors as rows and the
reciprocal matrix satisfies `A @ B.T = 2*pi*I`.

For this Hamiltonian, the ordering diagnostic is the **largest** eigenvalue
of `J(q)`. FM magnons use the moment-normalized matrix
`g M^(-1/2) [diag(J(0) 1) - J(q)] M^(-1/2)`. Induced sites are eliminated and
are not added as independent magnon branches. Goldstone behavior is checked
at Gamma when Gamma is present in the supplied mesh.

`X` is inferred only from the reference collinear state when its source field
is nonzero. A negative inferred `X`, cancellation, missing reciprocal input,
non-Hermitian `J(q)`, and ill-conditioned `I - X K_mm(q)` are reported rather
than hidden.

## IMX-08 adversarial fixtures

The executable cases are in `tests/test_imx08_validation.py`. The table below
records what each case is intended to catch and what a scientifically honest
result looks like.

| Fixture | Expected physics / numerical output | Expected warning |
|---|---|---|
| A. FM raw -> FM dressed | Robust raw and dressed leading eigenvalue are both at Gamma; induced correction is finite. | None beyond the explicit approximation label. |
| B. AF raw -> FM dressed | Robust-only raw maximum is at the zone boundary, while a strong finite-q induced correction moves the dressed maximum to Gamma. | The result remains model-dependent; no claim about the true susceptibility. |
| C. AF raw -> AF dressed | A weak induced correction does not move the zone-boundary maximum. | No numerical warning when the response is well conditioned. |
| D. Nearly singular response | `I - X K_mm` is solvable but has a very large condition number; the response is correspondingly amplified. | Near-singular/possible soft-response warning; no regularization. |
| E. Negative inferred susceptibility | `X = m0 / source_field` is negative for inconsistent reference sign data. | Negative-`X` sign/convention warning. |
| F. Vanishing symmetry-point field | `K_mM(q)` cancels at the selected symmetry point, so the induced response is exactly zero there. | Inference warns if the same cancellation makes reference `X` unavailable; the zero q response itself is physical. |
| G. Induced-induced near instability | A multi-site `K_mm` block approaches an eigenvalue of `X K_mm` equal to one. | Ill-conditioning/soft-response warning. |
| H. Multiple induced sublattices | The response and downfolding retain both induced components and return a `(n_q, 2)` response / two-by-two induced operator. | None when finite and Hermitian. |
| I. Nonorthogonal primitive cell | Cartesian phase evaluation agrees with the reciprocal-basis conversion for a nonorthogonal cell. | None; position reconstruction is not used. |
| J. Asymmetric/incomplete `Jij` | Fourier output remains non-Hermitian and is not repaired; ordering uses the general eigensolver only as a diagnostic. | Missing reciprocal/asymmetric input and uncertified physical spectrum. |

The fixture suite also checks the analytic Schur-complement energy identity,
Goldstone zeros for FM-compatible cases, moment normalization, inverse-phase
reconstruction diagnostics, induced-site elimination in magnons, and
Polesya/Mryasov spectral equivalence.
