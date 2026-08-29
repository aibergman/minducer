# Theory and scientific scope

## What the explorer computes

The explorer manipulates the effective spin model supplied by the user. It can
Fourier transform the specified exchange bonds, solve the selected linear
induced-moment model, eliminate induced variables variationally, and compute
FM-compatible spin-wave diagnostics. It cannot recover the true electronic
susceptibility from an exchange table alone.

The fixed Hamiltonian convention is

```text
H = -1/2 sum_ij Jij e_i · e_j,
```

where `e_i` is a unit spin direction. Each exchange-file row is one directed
term in this sum. Supplying both reciprocal rows is allowed; the factor of
one-half prevents double counting. Missing or unequal reciprocal partners are
retained and reported, never symmetrised or averaged.

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

Sites are explicitly classified as **robust** or **induced/slave**. The UI
initially selects moments below 0.5 `mu_B` as induced, but this is only an
editable convenience; the chosen classification, not the threshold, defines
the calculation.

The default `j_weighted` response uses the input exchange as a response kernel,
`K = J_input`, and solves

```text
m(q) = [I - X K_mm(q)]^-1 X K_mM(q) M(q).
```

`M` and `m` are robust and induced amplitudes, and `X` is susceptibility-like
with inverse-energy units. `infer_x()` estimates `X` from the supplied
collinear reference state when its source field is nonzero. The code reports
near-singular matrices, cancellation, and questionable signs rather than
regularising them silently.

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
E(M,m) = -1/2 M† J_MM M
       + 1/2 m† (X^-1 - K_mm) m
       - Re[m† K_mM M].
```

At stationarity this produces the response above and the robust-space
interaction

```text
J_eff(q) = J_MM(q) + K_Mm(q) (X^-1 - K_mm(q))^-1 K_mM(q).
```

The implementation evaluates the equivalent stable form involving
`[I - X K_mm]^-1 X`, including the continuous `X → 0` limit. The Mryasov and
Polesya labels exposed by the application describe algebraically equivalent
dressed interactions in this formulation. Any real-space dressed exchange is
an inverse transform on a finite regular mesh and is explicitly resolution
dependent.

## Collinear-FM magnons

Magnons are evaluated only as a diagnostic about a collinear ferromagnetic
reference. The harmonic and moment-normalised matrices are

```text
A(q) = diag(J(0) 1) - J(q)
D(q) = g² M^-1/2 A(q) M^-1/2.
```

`M` contains the reference moment magnitudes in `mu_B`; `g=2` is the default.
Eigenvalues are energy-valued `hbar omega` in the selected output energy unit.
Raw calculations retain all sites; robust-only calculations drop induced sites;
dressed calculations first eliminate them. Induced sites never appear as extra
independent magnon branches in a slave-moment calculation.

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

For symmetry-reduced exchange files, opt in to `spglib` expansion. The process
copies each representative over its geometric space-group orbit; it neither
fits nor Hermitianises values. Do not enable it for a complete neighbour list.
