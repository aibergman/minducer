Task IMX-07 — Build the Hugging Face / Gradio interface.

Read the complete HEAD and existing package first.

Goal:
Expose the already-tested Python physics package through a clean scientific UI.
Do not move physics logic into app.py.

The Space should be usable by someone familiar with UppASD without requiring manual data conversion.

Design a compact tabbed interface.

TAB 1 — INPUT

Support upload of `inpsd.dat` plus referenced files. Standalone file uploads
may be inspected for mapping, but loading requires `inpsd.dat` because the
cell vectors are not present in `posfile`, `momfile`, or `jfile`; no identity
cell is substituted.

Because browser upload paths do not preserve arbitrary relative filesystem layout reliably, implement a robust upload mapping:
- parse filenames referenced by inpsd.dat;
- match them to uploaded files by basename;
- clearly report unresolved files;
- permit manual reassignment.

Show parsed:
- cell;
- basis;
- moments;
- number of Jij;
- energy unit;
- warnings.

Allow user to classify each basis site/type as:
    robust
    induced.

The Input tab uses an explicit induced-site toggle, initialized with
`m < 0.5 mu_B` sites selected as a visible workflow default. The selected
site IDs drive the analysis; the threshold is not used by the library API as
an automatic scientific classification.

TAB 2 — RAW EXCHANGE

Show:
- Jij vs distance;
- shell table;
- J(q) bands/eigenvalues;
- predicted ordering vector;
- high-symmetry path.

TAB 3 — INDUCED RESPONSE

Controls:
- historical unweighted Polesya-like;
- J-weighted response;
- user X values;
- include/exclude induced-induced propagation.

Show:
- inferred X;
- source-field normalization;
- conditioning warnings;
- m_ind(q)/m_ind(0).

Prominently show:
    "K = input Jij is a response-model approximation, not an exact identity."

TAB 4 — DRESSED EXCHANGE

Show:
- robust-only raw J(q);
- induced contribution Delta J(q);
- dressed J_eff(q);
- raw vs dressed ordering;
- optional real-space dressed Jij.

TAB 5 — MAGNONS

Show, where physically valid:
- raw rigid-site spectrum;
- robust-only spectrum;
- dressed Mryasov spectrum;
- adiabatic Polesya/slave spectrum;
- spin stiffness.

If FM is unstable:
show stability information instead of pretending negative modes are ordinary magnons.

TAB 6 — COMPARE DATASETS

Optional Dataset B:
- raw/dressed A vs B;
- ordering;
- magnons;
- spin stiffness;
- optional DFT induced-moment-response comparison.

TAB 7 — METHODS / LIMITATIONS

Provide a concise explanation of:
- raw rigid-spin model;
- Polesya slave moments;
- Mryasov downfolding;
- J-weighted response assumption;
- what cannot be inferred from Jij alone.

Cite foundational papers in README/methods text:
- Mryasov et al., FePt induced moments/two-ion anisotropy;
- Polesya et al., induced moments in FePd/CoPt.

Do not overload the interface with equations.
Use tooltips/accordions for details.

Downloads:
- parsed canonical model JSON;
- raw J(q);
- dressed J(q);
- dressed Jij;
- magnon bands;
- response data;
- validation report.

Examples:
Bundle at least:
1. simple analytic FM model;
2. simple induced-moment toy model;
3. supplied FePt-style UppASD example.

Performance:
- target Hugging Face CPU Space;
- q meshes should have sensible defaults;
- expensive dense meshes should require explicit user request;
- cache Fourier data where possible;
- no GPU dependency.

Deployment:
- app.py
- requirements.txt
- README.md with Space metadata if needed
- startup must work with:
      python app.py

Tests must remain runnable independently of Gradio.

Checklist:
[x] Multi-file UppASD upload works.
[x] Relative path/basename mapping works.
[x] Raw exchange tab works.
[x] Response tab works.
[x] Dressed exchange tab works.
[x] Magnon tab works, including stiffness and explicit Mryasov/Polesya output.
[x] Dual dataset tab works, including optional external response comparison.
[x] Downloads work.
[x] Scientific warnings visible.
[x] Example datasets included.
[x] Optional spglib expansion is exposed for symmetry-reduced UppASD `jfile` input.
[x] Exchange, response, dressed-q, and magnon UI plots share one seekpath high-symmetry coordinate.
[x] Response normalization and the matrix/scalar meaning of `J_eff(q)` are explicit in the UI.
[x] Dressed robust-space `J_eff(r)` is reconstructed only from a separately labelled complete auxiliary mesh.
[x] Dressed real-space channels are shell-averaged into a focused `J_MM`/`J_Mm`/Mryasov/Polesya plot, with absolute and relative `Delta J_induced` plots separated and distances shown in implicit `alat = 1` units.
[x] CPU-only Space launches cleanly.
[x] Existing unit tests still pass.
[x] README completed.
[x] Hugging Face Space metadata and Python 3.10 runtime dependencies are prepared.
[x] Uploaded files are size/count limited, traversal-safe, and temporary workflow artifacts are cleaned up.

Commit message:
IMX-07 add Gradio induced exchange explorer
