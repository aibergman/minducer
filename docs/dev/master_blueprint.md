You are implementing a scientific Python tool and Hugging Face Space for analyzing induced magnetic moments and their effect on atomistic exchange models.

Working name:
    Induced-Moment Exchange Explorer
A shorter final name may be chosen later.

The tool is conceptually complementary to FourJ:
    FourJ: J(k) -> J_ij
    this project: J_ij -> J(q) -> induced-moment dressing -> magnetic order/magnons

The implementation must be scientifically conservative. Do not silently claim that quantities reconstructed from ordinary input Jij are exact first-principles susceptibilities.

======================================================================
PHYSICS SCOPE — VERSION 1
======================================================================

Input:
- crystal cell vectors;
- basis positions / site indices;
- magnetic moment magnitudes;
- isotropic scalar exchange interactions Jij with displacement vectors r_ij;
- optional species/sublattice labels if available;
- user classification of sublattices as:
    * robust/local-moment;
    * induced/slave moment.

Primary input format is UppASD-style files:
    inpsd.dat
    posfile
    momfile
    jfile

Example inpsd.dat syntax:

    simid     FePtFM25
    ncell     12 12 12
    BC        P P P
    cell      1.000000000000000  0.000000000000000  0.000000000000000
              0.500000000000000  0.500000000000000  0.000000000000000
              0.000000000000000  0.000000000000000  0.9525

    posfile   ./posfile
    exchange  ../fourj.jij.dat
    momfile   ./momfile

Example posfile:
    1 1 0.0 0.0 0.0

Interpret columns as:
    site_index  atom_type  x_cart  y_cart  z_cart

Example momfile:
    1 1 2.9913824 0.0 0.0 1.0

Interpret minimally as:
    site_index atom_type moment [optional sx sy sz]

Example jfile:
    i j rx ry rz Jij distance

where:
- i,j are basis/site indices;
- (rx,ry,rz) is the Cartesian displacement from i to the periodic image of j;
- Jij is scalar isotropic exchange;
- distance is optional and must never be trusted in preference to sqrt(rx^2+ry^2+rz^2).

Do not assume that coordinates in posfile are direct/fractional coordinates.
For this input convention they are Cartesian.

Units:
- preserve input energy units unless the program can infer them safely;
- allow user selection/confirmation of meV vs mRy;
- internally use one well-documented energy unit;
- moments are in mu_B unless overridden;
- Cartesian positions use the same length unit as the supplied cell and Jij displacement vectors.

Hamiltonian convention must be explicit and centralized, e.g.

    H = -1/2 sum_ij J_ij e_i dot e_j

or

    H = -sum_<ij> J_ij e_i dot e_j.

Never mix conventions.
All Fourier transforms, spin-wave matrices, and effective interactions must use the same convention.

======================================================================
MODEL LEVELS
======================================================================

Implement four conceptually distinct models.

1. RAW MODEL

All supplied magnetic sites are treated as ordinary rigid moments and all supplied Jij are retained.

This is diagnostic only if some sites are known to be induced moments.

2. ROBUST-ONLY RAW MODEL

Discard induced-site rows/columns and retain only directly supplied robust-robust Jij.

This shows what the direct robust-spin exchange alone predicts.

3. POLESYA-LIKE SLAVE-MOMENT MODEL

The induced moments are not independent LLG/LSWT degrees of freedom.

For induced subspace m and robust subspace M, define initially:

    m(q) = X [ K_mM(q) M(q) + K_mm(q) m(q) ]

therefore

    m(q) =
        [I - X K_mm(q)]^{-1}
        X K_mM(q) M(q)

For the initial implementation, allow the approximation

    K_ij = J_ij(input)

but label this throughout the UI and documentation as

    "J-weighted induced-response approximation"

and not as an exact susceptibility identity.

X may be inferred from the collinear reference state when possible:

    m_nu^0 =
        X_nu sum_j K_nu,j M_j^0

so that

    X_nu =
        m_nu^0 /
        sum_j K_nu,j M_j^0

with careful sign and zero-denominator checks.

Also support an "unweighted historical Polesya-like" mode:

    m_nu =
        X_nu sum_{j in selected neighbour shell} M_j.

Do not confuse the induced moment m_nu with the susceptibility coefficient X_nu.

4. MRYASOV-LIKE DOWNFOLDED MODEL

Analytically eliminate the induced subspace.

For a quadratic response model write the effective robust-spin interaction schematically as

    J_eff_MM(q)
      =
      J_MM(q)
      +
      K_Mm(q)
      Xi_m(q)
      K_mM(q)

where

    Xi_m(q)
      =
      [X^{-1} - K_mm(q)]^{-1}

or an algebraically equivalent expression consistent with the implemented energy convention.

Derive this carefully from a single energy functional.
Do not insert factors of 1/2 by analogy.
Unit-test the derivation.

The adiabatic low-energy magnetic spectrum obtained from the slave Polesya representation and the downfolded Mryasov representation must agree numerically when both use the same linear response model.

======================================================================
FOURIER TRANSFORM
======================================================================

Construct

    J_ab(q)
      =
      sum_R J_ab(R)
            exp[i q dot r_ab(R)]

using the actual Cartesian pair displacement r_ij supplied in the Jij file.

Do not reconstruct r_ij from site positions unless required for validation.

The implementation must correctly support:
- non-orthogonal cells;
- multiple atoms per primitive cell;
- interactions across arbitrary lattice translations;
- positive and negative displacement entries;
- incomplete +/-R input when symmetry reconstruction is explicitly requested;
- duplicate entries with diagnostics.

Hermiticity must be checked:

    J(q) = J(q)^\dagger

within numerical tolerance.

Do not silently symmetrize malformed input without reporting it.

======================================================================
MAGNETIC ORDER AND MAGNONS
======================================================================

Provide:

1. J(q) eigenvalue scan over a reciprocal mesh.
2. Candidate ordering vector.
3. High-symmetry path plotting.
4. Collinear-FM linear spin-wave spectrum for each appropriate effective model.
5. Small-q spin stiffness estimate where meaningful.

Use a robust multi-sublattice ferromagnetic dynamical matrix with proper moment normalization.

Do not generate independent magnon branches for slave induced moments.

For Polesya-like calculations, eliminate the induced variables before constructing the physical adiabatic magnon dynamical matrix.

Version 1 may reject or clearly flag:
- noncollinear ground states;
- AF/ferrimagnetic LSWT requiring local-frame/Bogoliubov treatment;
- DMI;
- exchange tensors;
- single-ion anisotropy;
- SOC;
- frequency-dependent susceptibility.

Detecting an AF ordering tendency from J(q) is allowed even if the tool cannot yet calculate its full AF magnon spectrum.

======================================================================
OUTPUTS
======================================================================

Provide raw and dressed:

- Jij tables;
- shell/radial exchange plots;
- J_ab(q);
- eigenvalues of J(q);
- ordering-vector diagnostics;
- effective/downfolded J_eff(q);
- inverse Fourier transformed dressed Jij where numerically sensible;
- induced response m_ind(q)/m_ind(0);
- magnon dispersion for FM-compatible cases;
- spin stiffness;
- downloadable CSV/JSON data.

If a second Jij dataset is uploaded, support comparison:
    dataset A vs dataset B

This is intended especially for:
    LKAG Jij vs frozen-magnon Jij.

======================================================================
SCIENTIFIC WARNINGS
======================================================================

The UI and documentation must explicitly distinguish:

EXACT GIVEN THE INPUT MODEL:
- Fourier transforms;
- matrix algebra;
- Schur complements;
- eigenvalue analysis;
- LSWT of the constructed effective Hamiltonian.

MODEL ASSUMPTIONS:
- identifying input Jij with the kernel K that induces the soft moment;
- inferring X from the collinear reference moment;
- treating susceptibility as static and linear;
- treating induced moments as instantaneous slave variables.

NOT AVAILABLE FROM CELL + MOMENTS + Jij ALONE:
- true longitudinal susceptibility;
- true transverse susceptibility;
- K_ij from first-principles response;
- frequency-dependent susceptibility;
- nonlinear Stoner response.

Do not hide these limitations.

======================================================================
CODE QUALITY
======================================================================

Keep physics independent of Gradio.

Suggested package layout:

    induced_exchange/
        __init__.py
        model.py
        io_uppasd.py
        reciprocal.py
        response.py
        downfold.py
        magnons.py
        paths.py
        validation.py
        plotting.py

    app.py
    examples/
    tests/
    README.md
    requirements.txt

Use:
- Python >=3.11;
- numpy;
- scipy;
- pandas where useful;
- matplotlib or Plotly for plots;
- spglib/seekpath optionally for reciprocal paths;
- gradio for the Space.

Avoid large dependencies unless justified.

Every physics routine must be callable independently of the UI.

======================================================================
VALIDATION PHILOSOPHY
======================================================================

Tests must include analytically solvable toy systems:

- one-sublattice FM chain/simple lattice;
- two-sublattice rigid FM;
- one robust + one induced site with known scalar response;
- induced-site downfolding with and without m-m coupling;
- Polesya/Mryasov equivalence;
- q=0 Goldstone condition for FM magnons;
- Fourier round-trip J(R) -> J(q) -> J(R);
- non-orthogonal cell;
- supplied FePt-like UppASD input.

Do not validate one implementation path solely against another implementation path that shares the same formulas.
Where possible use independent analytic references.

At completion of each task:
1. run tests;
2. update documentation;
3. tick all task checkboxes;
4. commit only the task-related changes;
5. use the requested one-line commit message.
