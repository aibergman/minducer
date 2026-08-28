Task IMX-05 — Implement collinear-FM magnon spectra for raw and dressed models.

Read HEAD through IMX-04.

Goal:
Calculate physically normalized adiabatic magnon spectra for models whose reference state is a stable collinear ferromagnet.

Do not yet implement general noncollinear/AF/ferrimagnetic Bogoliubov LSWT.

Implement a multi-sublattice FM spin-wave dynamical matrix for:

1. raw all-rigid model;
2. robust-only raw model;
3. Mryasov/downfolded robust model;
4. Polesya/slave model after analytical elimination of induced variables.

The physical low-energy Polesya spectrum and the Mryasov spectrum must agree when built from the same static linear response model.

Do not assign ordinary LLG/LSWT dynamical degrees of freedom to induced slave moments.

Use the moment magnitudes correctly.

For Hamiltonian convention:
    H = -1/2 sum_ij Jij e_i dot e_j

derive the multi-sublattice dynamical matrix explicitly and document the conversion to energy units.

Do not guess factors of:
- 2;
- mu_B;
- g;
- hbar.

Centralize unit/convention handling.

Provide:
- magnon energies vs q;
- acoustic branch;
- optical branches where robust multi-sublattice structure requires them;
- instability indication when eigenvalues are negative/unphysical;
- small-q spin stiffness fit:
      E(q) ~= D q^2
  using a user-visible fitting interval.

If the candidate ordering vector is not Gamma:
- refuse to label the result a stable FM magnon spectrum;
- optionally plot signed harmonic eigenvalues as a stability diagnostic;
- explain that general AF/noncollinear LSWT is outside v1.

Tests:
- one-sublattice NN FM with analytic dispersion;
- multi-shell FM;
- two-rigid-sublattice FM;
- one robust + one induced scalar model showing Polesya/Mryasov equality;
- Goldstone mode at q=0 to tight numerical tolerance;
- scaling with moment magnitude;
- unit conversion tests.

Checklist:
[ ] FM dynamical matrix derived/documented.
[ ] One-sublattice analytic test passes.
[ ] Multi-sublattice test passes.
[ ] Goldstone condition passes.
[ ] Moment normalization verified.
[ ] Polesya/Mryasov spectra agree.
[ ] Unstable FM is clearly flagged.
[ ] Spin stiffness implemented.
[ ] Plot/data API implemented.
[ ] Documentation updated.

Commit message:
IMX-05 add FM magnon and stiffness analysis
