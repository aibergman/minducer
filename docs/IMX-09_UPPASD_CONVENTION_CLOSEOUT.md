# IMX-09 UppASD convention close-out

Status: implemented and validated in the repository test suite.

## 1. Convention audit before the change

| Location | Convention found before IMX-09 | Desired UppASD convention | Change and rationale |
|---|---|---|---|
| `io_uppasd.py` | Parsed/stored exchange rows literally, but the convention was implicit. | Literal `Jij` from the jfile. | Clarified parser documentation; no numerical rescaling. |
| `reciprocal.py` | Fourier transform already used literal rows; module documentation said `-1/2` ordered sum. | `J_ab(q)=sum_R Jij_ab(R)e^(iq·r)` with literal rows. | Documentation only; ordering and Hermiticity algorithms were retained. |
| `magnons.py` | `g_factor**2 M^-1/2 A M^-1/2`. | `2*g_factor M^-1/2 A M^-1/2`. | Replaced the accidental default-`g=2` equivalence with the physical normalization. |
| `induced.py` | `M` and `m` were used as moment-like amplitudes while `K=J` and `X` were labelled as energy/inverse-energy. | Dimensionless robust orientations `e` and induced polarizations `p=m/|m0|`; `[K]=E`, `[X]=1/E`. | Reference inference now uses `p0/(K e0)`; response results expose normalized `p` and a physical-moment conversion. |
| `downfolding.py` | One `-1/2` quadratic functional and a matching `J_eff` formula. | One ordered-pair functional with explicit RR, RI/IR, II, and restoring terms. | Re-derived and implemented the full energy check with no extra factor in `J_eff`. |
| `provenance.py` | Schema 1 recorded `H=-1/2 sum` and did not describe all response units. | IMX-09 convention, formula, pair counting, `g`, `K`, `X`, and variable definitions. | Added schema/convention version 2 and machine-readable fields. |
| `space.py` / app | CSV diagnostics existed, but no native UppASD dressed jfile was emitted; help text used the old Hamiltonian. | Exported dressed values must be directly consumable by UppASD. | Added `dressed_jfile` export and updated visible help/labels. |
| `model.py` / `units.py` | No centralized ordered-pair energy/field or exchange-convention conversion helper. | Native energy/field helpers and boundary conversion helpers. | Added `conventions.py` and model convenience methods. |
| `README.md`, `docs/THEORY.md`, `docs/PHYSICS_LIMITS.md` | Old half-weighted convention and `g²` explanation. | Prominent UppASD convention, factor ledger, induced units, and conversions. | Updated documentation. |
| `tests/` | Regression targets encoded the old `X` inference description and lacked non-`g=2`/export checks. | Analytic IMX-09 targets. | Updated affected assertions and added `test_imx09_convention.py`. |

No input `Jij`, `J(q)`, ordering eigenvalue, or reciprocal row was multiplied
by two.

## 2. Authoritative native convention

Minducer now uses the literal UppASD scalar-Heisenberg jfile convention

```text
H_UppASD = - sum_(i != j) Jij e_i · e_j,
```

where the sum is ordered, `e_i` is a dimensionless unit direction, and a
pair-complete file contains both directed rows. Positive `Jij` is
ferromagnetic. The parser, stored model, Fourier transform, dressed result,
and native exported jfile all use the same numerical `Jij`.

For a one-sublattice ferromagnet,

```text
J(q) = sum_j J_0j exp(i q·r_0j)
A(q) = J(0) - J(q).
```

The largest eigenvalue criterion for a positive-FM exchange remains correct;
the normalization change does not alter the ordering vector. Hermiticity is
still checked and malformed/incomplete input is not repaired.

## 3. Local-field factor 2

Varying the ordered-pair energy gives, for a symmetric exchange table,

```text
∂H/∂e_i = -sum_j Jij e_j - sum_j Jji e_j
          = -2 sum_j Jij e_j.
```

Thus the exchange field is

```text
B_i = 2/(m_i mu_B) sum_j Jij e_j.
```

The factor 2 is the derivative of the ordered pair sum. It is not a second
Landé factor. `local_exchange_field()` also evaluates the exact outgoing plus
incoming derivative for an asymmetric diagnostic input.

## 4. Magnon normalization and stiffness

For transverse amplitudes `u`, the one-sublattice harmonic energy is

```text
ΔE^(2) = u† A(q) u = 1/2 u† [2 A(q)] u.
```

Therefore the physical curvature is `C(q)=2A(q)`. The moment-normalized,
energy-valued multi-sublattice dynamical matrix implemented by
`fm_dynamical_matrix()` is

```text
D(q) = 2*g M^(-1/2) A(q) M^(-1/2),
A_ab(q) = delta_ab sum_c J_ac(0) - J_ab(q).
```

For one sublattice this gives

```text
hbar omega(q) = 2*g/m [J(0)-J(q)].
```

For nearest-neighbour rows `J(+a)=J(-a)=J`,

```text
hbar omega(q) = 4*g*J/m [1-cos(qa)].
```

The stiffness routine fits the resulting energy directly to `D q²`; it adds
no manual convention correction. Site-dependent `g` is not supported, so no
non-Hermitian site-dependent extension was guessed.

## 5. Induced variables and units

Two parameterizations were considered.

### Physical moments

Writing `M_i=m_i e_i` and using physical moment-valued induced variables
`m_nu` gives exchange terms of dimension `E/mu_B²` and a susceptibility of
dimension `mu_B²/E` (or their numerical equivalents if `mu_B` is absorbed).
The literal UppASD `Jij`, which has energy units, cannot then be identified
with `K` without an unstated moment-unit conversion. This is not a transparent
interpretation of the current `K=J_input` approximation.

### Normalized induced polarization (selected)

The implementation instead uses

```text
r_a = e_a                         dimensionless robust orientation amplitude
p_nu = m_nu / |m_nu^0|            dimensionless induced polarization
K_ab = Jij_ab(input)               energy
X_nu                                1/energy.
```

The physical induced moment is recovered as `m_nu=|m_nu^0|p_nu` when the
reference magnitude is present. The public response field retains its
historical `induced_moments` name for compatibility, but its model values are
the normalized `p` amplitudes; `physical_induced_moments` gives the converted
values.

For the J-weighted approximation,

```text
p = [I - X K_II(q)]^-1 X K_IR(q) r.
```

`K=J_input` remains explicitly a **J-weighted induced-response
approximation**, never an exact susceptibility identity. In historical
unweighted mode the selected-neighbour count is dimensionless and its `X` is
accordingly dimensionless.

## 6. Reference-state X normalization

For an aligned reference, `p_nu^0=1` and the source field is

```text
s_nu^0 = sum_a K_nu,a e_a^0.
```

The inferred coefficient is therefore

```text
X_nu = p_nu^0 / s_nu^0,
```

with units `1/E`. The induced reference magnitude is used only to convert the
normalized response back to a physical moment; it does not enter this
J-weighted normalization. A signed reference projection can produce a
negative `p_nu^0` and hence a negative inferred `X`, which is reported as a
sign/reference diagnostic. Zero or strongly cancelled source fields remain
unresolved and are never regularized.

## 7. One common quadratic functional

At each q, let `r` be robust orientations and `p` induced polarizations. The
complete functional is

```text
E(r,p) = - r† J_RR r
         + p† (X^-1 - K_II) p
         - 2 Re[p† K_IR r].
```

The terms mean:

- `-r†J_RRr`: robust-robust ordered rows, including both directions;
- `-p†K_IIp`: induced-induced ordered rows, including both directions;
- `-p†K_IRr - r†K_RIp`: the two robust-induced ordered blocks, written once
  as the displayed cross term;
- `+p†X^-1p`: local induced restoring energy, which is not a pair term.

Stationarity gives

```text
(X^-1-K_II) p* = K_IR r
p* = Xi K_IR r
Xi = (X^-1-K_II)^-1
   = [I-X K_II]^-1 X.
```

Thus `[Xi]=1/energy`: it maps the energy-valued source `K_IR r` to the
dimensionless induced polarization `p`.

The second form is used in code so the `X -> 0` limit is well-defined.

## 8. Schur complement and dressed UppASD Jij

Substitution of `p*` gives

```text
E_eff(r) = -r† [J_RR + K_RI Xi K_IR] r.
```

Since the effective UppASD representation is also `E_eff=-r†J_eff r`, the
exported interaction is directly

```text
J_eff^UppASD = J_RR + K_RI Xi K_IR.
```

The correction has no extra factor of 2 or 1/2. The apparent conventional
`-1/2 r†(...)r` Schur-complement formula belongs to a half-weighted quadratic
functional; here every exchange block has been derived with the ordered-pair
counting and the restoring/cross terms have been scaled consistently.

`DownfoldingResult.energy_equivalence()` evaluates the explicit functional at
`p*` and compares it to `-r†J_eff r`, including vector transverse components
and arbitrary complex q amplitudes. This is the decisive normalization test.

## 9. Polesya/Mryasov equivalence

The Polesya-like induced response and Mryasov/downfolded result use the same `Xi` and
the same stationary `p*`. Induced sites are not added as independent LLG or
LSWT degrees of freedom. Consequently the robust stationary energy and the
adiabatic physical magnon spectrum are identical. The application retains the
two labels to distinguish the representations, not to introduce separate
induced branches.

## 10. Export and provenance

`write_uppasd_jfile()` emits rows in the six-column UppASD layout
`i j rx ry rz Jij` (with an optional distance column). It rejects non-finite
values and significant imaginary residuals rather than writing a scalar file
that changes the model. Analysis exports now include `dressed_jfile` whenever
the finite real-space reconstruction is resolvable; `dressed_jij.csv` remains a
diagnostic table with the reconstruction metadata.

The exported provenance records at least:

```text
hamiltonian_convention:       UppASD ordered-pair scalar Heisenberg
hamiltonian_formula:          H = -sum_{i!=j} Jij e_i·e_j
pair_counting:                ordered
jij_semantics:                literal UppASD jfile value
g_factor:                     numerical Landé factor
magnon_prefactor_convention:  2*g from ordered-pair curvature and gyromagnetic ratio
induced_variable_definition:  r=e, p=m/|m0|
response_kernel_definition:   K_ab(R)=literal input Jij_ab(R) in J-weighted mode
response_kernel_source:       input Jij (or historical neighbour sum)
X_definition:                 p0/(sum_a K_nu,a e_a^0)
X_units:                      1 / energy in J-weighted mode
```

## 11. Validation results

The independent IMX-09 analytic tests cover the following.

1. **One-sublattice chain:** with both `+/-a` rows, `J(q)=2J cos(qa)` and
   `hbar omega=4gJ/m[1-cos(qa)]`; for `J=m=1`, `g=1.5`, the quarter-zone
   energy is `6`.
2. **Landé scaling:** changing `g=2` to `g=1.5` changes the energy by `0.75`,
   not by `0.75²`.
3. **MFT:** `J(0)=6` gives `k_B T_C^MFT=4=2J(0)/3`.
4. **Explicit ordered energy and field:** two parallel sites with reciprocal
   rows `J=3` have energy `-6` and exchange field coefficient `6` on each
   site.
5. **One robust plus one induced site:** `J_RR=1.5`, `K_RI=K_IR=2`,
   `X=0.5` gives `Delta J=2`, `J_eff=3.5`, and both explicit and exported
   energies equal `-3.5 r²`.
6. **Induced-induced propagation:** the two-induced-site fixture agrees with
   the direct solve for `Xi` and the downfolded correction.
7. **Polesya/Mryasov:** the two labelled robust spectra are equal and contain
   no induced branches.
8. **Stiffness:** stiffness is obtained from the corrected energy dispersion
   and has no second manual factor.
9. **Round trip:** the emitted `dressed_jfile` parsed as literal UppASD rows
   reproduces the minducer downfolded energy without rescaling.

The full repository run is:

```text
83 passed, 4 skipped
```

The existing UppASD-style reference spectrum remains within its prior
comparison tolerance. With default `g=2`, the old numerical prefactor and the
new physical prefactor are exactly equal (`g²=4=2g`), so the old/new default
benchmark is unchanged. The explanation is now correctly separated into the
ordered-pair factor 2 and the single gyromagnetic factor `g`.

## 12. Backward compatibility and limitations

No persistent model-format migration machinery was required: input models are
still plain parsed UppASD rows and contain no hidden conversion. Provenance
exports now use schema version 2 and identify IMX-09. Pre-IMX-09 provenance
exports with schema version 1 and the old `H=-1/2` description must not be
silently reinterpreted; they should be treated as historical exports under the
old documented convention.

Remaining limits are intentional: `K=J_input` is a model approximation, the
response is static and algebraic, singular induced blocks are flagged rather
than regularized, native jfile export is a finite real-space reconstruction of
the sampled q mesh, and the magnon implementation remains restricted to the
collinear-FM case with a uniform `g`. Noncollinear, AF/ferrimagnetic LSWT,
DMI, exchange tensors, SOC, anisotropy, and frequency-dependent susceptibility
are outside this close-out.
