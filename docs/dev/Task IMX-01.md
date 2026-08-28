Task IMX-01 — Implement the core data model and robust UppASD input parsing.

Read HEAD first and obey all physics and convention requirements there.

Goal:
Convert inpsd.dat + posfile + momfile + exchange/Jij file into one validated internal magnetic-crystal model. Do not implement induced-moment physics or magnons yet.

Implement:

1. Crystal model
   - 3x3 real-space cell matrix.
   - basis sites.
   - site/type identifiers.
   - Cartesian basis positions.
   - reference magnetic moment magnitudes.
   - optional initial spin directions.
   - isotropic exchange bonds with:
       i, j, Cartesian displacement r_ij, Jij.
   - metadata for energy and length units.
   - default UppASD energy metadata to mRy, with explicit override support.

2. inpsd.dat parser
   - parse multiline `cell`;
   - parse optional `alat` length scale in metres;
   - parse `posfile`, `momfile`, `exchange`;
   - resolve paths relative to the directory containing inpsd.dat;
   - ignore unrelated UppASD keywords safely;
   - retain useful warnings instead of failing on harmless unknown keywords.

3. posfile parser
   Expected format:
       site atom_type x y z
   Positions are Cartesian for this project.

4. momfile parser
   Expected minimum:
       site moment_field moment
   Optional:
       sx sy sz

5. exchange parser
   Expected:
       i j rx ry rz Jij [distance]
   Treat supplied displacement as authoritative.
   Recompute distance and warn if optional supplied distance disagrees beyond tolerance.

6. Validation
   - valid site indices;
   - finite values;
   - nonzero cell volume;
   - duplicate exchange entries;
   - self interaction at r=0;
   - use the posfile species/type column for site identity; do not interpret
     the momfile metadata field as a species mismatch;
   - missing moment data;
   - identify likely +/-R partners;
   - report whether real-space interactions appear Hermitian/pair complete.

7. Diagnostics API
   Provide a structured validation report rather than only logging strings.

Important:
Do not infer chemical elements from type numbers.
Do not assume one site per atom type.
Do not assume orthogonal or conventional cells.

Tests:
- supplied FePt-style single-site example;
- nonorthogonal cell;
- multiline cell parser;
- relative paths;
- malformed Jij row;
- duplicate interaction;
- mismatched optional distance;
- missing file;
- comments/blank lines.

Deliver a small CLI/debug script capable of:

    python -m induced_exchange.io_uppasd path/to/inpsd.dat

printing:
- cell;
- number of basis sites;
- moments;
- number of exchange bonds;
- min/max bond distance;
- validation warnings.

Checklist:
[ ] Internal model implemented.
[ ] inpsd parser implemented.
[x] optional alat scaling implemented.
[ ] posfile parser implemented.
[ ] momfile parser implemented.
[ ] Jij parser implemented.
[ ] Relative paths handled correctly.
[ ] Validation report implemented.
[ ] Tests pass.
[ ] README input-format section added.

Commit message:
IMX-01 add UppASD input model and parsers
