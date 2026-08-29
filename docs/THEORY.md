# Theory and scientific scope

## What the explorer computes

The explorer manipulates the effective spin model supplied by the user. It can
Fourier transform the specified exchange bonds, solve the selected linear
induced-moment model, eliminate induced variables variationally, and compute
FM-compatible spin-wave diagnostics. It cannot recover the true electronic
susceptibility from an exchange table alone.

## UppASD Hamiltonian convention

The fixed native Hamiltonian convention is

```text
H = - sum_(i != j) Jij e_i · e_j,
```

where `e_i` is a dimensionless unit spin direction and the sum is ordered.
Each exchange-file row is one directed term, and a pair-complete file contains
both reciprocal rows. `Jij` is stored and used as the literal UppASD jfile
value; missing or unequal reciprocal partners are retained and reported, never
symmetrised or averaged. Dressed values exported by minducer are already in
this convention and can be passed to UppASD without rescaling.

The factor ledger is:

| quantity | factor | origin |
|---|---:|---|
| `J(q)` | `1` | literal jfile Fourier transform |
| local exchange field | `2` | derivative of the ordered-pair sum |
| magnon energy | `2*g` | ordered-pair curvature times one gyromagnetic factor |
| global pair energy | ordered-pair sum | native UppASD representation |
| thermal white-noise factor | `2` | fluctuation-dissipation normalization, unrelated to exchange pair counting |

For `H=-sum_<ij> J' e_i·e_j`, use `J_UppASD=J'/2`. For
`H=-1/2 sum_(i!=j) J''_ij e_i·e_j`, use `J_UppASD=J''/2`. An AF-positive
ordered convention uses the corresponding sign reversal. A spin-`S`
Hamiltonian written using unit directions first absorbs the spin magnitudes
into the pair coefficient, then uses the same single-counted conversion.

## Exchange in reciprocal space

The direct-cell matrix `A` has Cartesian lattice vectors as rows. Reciprocal
vectors `B` are defined by `A @ B.T = 2πI`; reduced coordinates `h` map to
Cartesian wavevectors through `q = h @ B`. The exchange transform is

```text
J_ab(q) = sum_R J_ab(R) exp(+i q · r_ab(R)).
```

The supplied Cartesian displacement `r_ab(R)` is authoritative. The ordering
diagnostic is the location of the largest eigenvalue of `J(q)` under the
Hamiltonian convention above. If the matrix is non-Hermitian because input is
incomplete or asymmetric, the result is marked as a diagnostic rather than a
certified physical spectrum.

The application displays a seekpath high-symmetry line when available. An
ordering candidate on that line is not a global three-dimensional ordering
search. Use `regular_q_mesh` with the library API for a full mesh calculation.

## Induced moments

Sites are explicitly classified as **robust** or **induced**. The UI
initially selects moments below 0.5 `mu_B` as induced, but this is only an
editable convenience; the chosen classification, not the threshold, defines
the calculation.

The default `j_weighted` response uses the input exchange as a response kernel,
`K = J_input`, and solves

```text
p(q) = [I - X K_II(q)]^-1 X K_IR(q) e(q).
```

The selected parameterization uses robust orientation amplitudes `r=e` and
normalized induced polarizations `p=m/|m⁰|`; both are dimensionless. Thus
`K` has energy units, `X` has inverse-energy units, and `m=|m⁰|p` is recovered
only when a reference induced moment is available. `infer_x()` uses

```text
X_nu = p_nu^0 / sum_a K_nu,a e_a^0,
```

so the ordinary aligned reference has `p_nu^0=1`; the induced reference
moment magnitude does not enter the J-weighted normalization. The historical
unweighted mode instead has a dimensionless neighbour-count kernel. The code
reports near-singular matrices, cancellation, and questionable signs rather
than regularising them silently.

The alternate `historical`/`unweighted` mode uses a geometric-neighbour sum.
Neither mode makes an induced moment an independent LLG or spin-wave degree of
freedom.

### Essential limitation

The identity `K = J_input` is a **J-weighted induced-response approximation**,
not a first-principles identity. A disagreement with measured or externally
calculated induced moments may mean that the actual induction kernel differs
from input `Jij`, that response is nonlinear/nonlocal, or that the exchange
table represents another magnetic curvature. It does not establish that one
particular exchange method is wrong.

## Variational downfolding

For the J-weighted model, the induced variables are eliminated from the
quadratic functional

```text
E(r,p) = -r† J_RR r
       + p† (X^-1 - K_II) p
       - 2 Re[p† K_IR r].
```

`J_RR` and `K_II` already contain ordered directed rows. The cross term is
the sum of the two blocks `-p†K_IR r - r†K_RI p`; it is not counted again.
The local restoring term defines `X` and is not an exchange pair term.

At stationarity this produces the response above and the robust-space
interaction

```text
p*(q) = (I - X K_II(q))^-1 X K_IR(q) r(q)
E_eff(r) = -r† J_eff r
J_eff(q) = J_RR(q) + K_RI(q) (X^-1 - K_II(q))^-1 K_IR(q).
```

Here `Xi=(X^-1-K_II)^-1` has inverse-energy units and maps the energy-valued
source `K_IR r` to the dimensionless induced polarization `p`.

The implementation evaluates the equivalent stable form involving
`[I - X K_II]^-1 X`, including the continuous `X → 0` limit. The Schur
complement is already the UppASD-format dressed exchange: no additional
factor of two or one-half is applied. The Mryasov and
Polesya labels exposed by the application describe algebraically equivalent
dressed interactions in this formulation. Any real-space dressed exchange is
an inverse transform on a finite regular mesh and is explicitly resolution
dependent.

## Collinear-FM magnons

Magnons are evaluated only as a diagnostic about a collinear ferromagnetic
reference. The harmonic and moment-normalised matrices are

```text
A(q) = diag(J(0) 1) - J(q)
D(q) = 2*g M^-1/2 A(q) M^-1/2.
```

`M` contains the reference moment magnitudes in `mu_B`; `g=2` is the default.
The factor 2 is the ordered-pair harmonic curvature and `g` is the one Landé
factor. Eigenvalues are energy-valued `hbar omega` in the selected output
energy unit. For one sublattice, `J(q)=sum_j J_0j exp(i q.r_0j)` and
`hbar omega(q)=2*g/m [J(0)-J(q)]`; a nearest-neighbour chain with both `+/-a`
rows therefore has `4*g*J/m [1-cos(qa)]`.
Raw calculations retain all sites; robust-only calculations drop induced sites;
dressed calculations first eliminate them. Induced sites never appear as extra
independent magnon branches in an induced-moment calculation.

Goldstone behaviour at Gamma, negative modes, and a non-Gamma ordering
tendency are reported. A non-FM-compatible result is not presented as a
stable FM spectrum. The optional stiffness fit uses `E = D |q|²` over an
explicit near-Gamma interval.

## Input and unit conventions

UppASD exchange files commonly use mRy; therefore the loader and UI label
input as `mRy` by default. This assigns metadata and does not convert the
numbers. Pass an explicit input energy unit if the file uses another one.

When `alat` is present in `inpsd.dat`, it is interpreted in metres and scales
cell vectors, positions, and exchange displacements consistently. Without it,
lengths are preserved in their supplied units and remain unspecified.

User-entered `X` in the J-weighted mode is defined by `p = X K e`, with
`p=m/|m⁰|` dimensionless and `X` in inverse energy. Positive `X` means an
induced polarization along the model source field; a negative value means an
opposite response or indicates a sign/reference-convention issue. `K=J_input`
remains a **J-weighted induced-response approximation**, not an exact
susceptibility identity.

For symmetry-reduced exchange files, opt in to `spglib` expansion. The process
copies each representative over its geometric space-group orbit; it neither
fits nor Hermitianises values. Do not enable it for a complete neighbour list.
