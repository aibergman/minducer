Task IMX-02 — Implement reciprocal-space exchange analysis.

Read HEAD and use the data structures from IMX-01.

Goal:
Given real-space scalar Jij bonds, construct a trustworthy multi-sublattice J(q), reciprocal geometry, and ordering-vector analysis.

Implement:

    J_ab(q) =
        sum_{bonds a->b} Jij exp(i q dot r_ij)

using the Cartesian pair displacement supplied in the Jij file.

Requirements:

1. Reciprocal lattice
   Construct reciprocal vectors with explicit 2*pi convention.
   Keep a clear distinction between:
   - reciprocal fractional coordinates;
   - Cartesian reciprocal coordinates.

2. Fourier transform
   Support arbitrary number of basis sites.
   Vectorize over q where sensible.
   Provide both:
       J(q)
   and
       J(q) eigenvalues/eigenvectors.

3. Hermiticity
   Check:
       J(q) ~= J(q)^dagger.
   Report violations.
   Do not silently hide malformed real-space input.

4. q meshes
   - regular reciprocal mesh;
   - user-specified q points;
   - optional seekpath high-symmetry path;
   - explicit fallback path if seekpath cannot classify the cell.

5. Ordering diagnosis
   For the Hamiltonian convention fixed in HEAD, determine correctly whether magnetic ordering is associated with the largest or smallest relevant eigenvalue of J(q).
   Derive this in comments/docs and test it analytically.

   Report:
   - candidate q_order;
   - eigenvalue;
   - sublattice eigenvector;
   - whether q=Gamma appears locally stable.

6. Plots/data
   - eigenvalues of J(q) on a path;
   - maximum/minimum exchange eigenvalue over q mesh;
   - optional heat map for 2D cuts;
   - downloadable numeric arrays.

Tests:
- analytic 1D NN FM chain;
- analytic 1D AF chain;
- simple cubic NN FM;
- two-sublattice model;
- nonorthogonal cell;
- pair-complete and intentionally incomplete datasets.

Do not implement magnons yet.

Checklist:
[ ] Reciprocal basis correct.
[ ] Cartesian/fractional q conventions tested.
[ ] J(q) implemented.
[ ] Hermiticity diagnostics implemented.
[ ] Analytic FM test passes.
[ ] Analytic AF test passes.
[ ] Multi-sublattice test passes.
[ ] q-order search implemented.
[ ] Path plotting data API implemented.
[ ] Documentation updated.

Commit message:
IMX-02 add reciprocal exchange and ordering analysis
