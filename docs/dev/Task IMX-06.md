Task IMX-06 — Add research diagnostics for induced-moment failures and comparison of two Jij datasets.

Read HEAD through IMX-05.

Goal:
Make the tool useful for diagnosing exactly the situation where ordinary LKAG and frozen-magnon-derived interactions predict different magnetic order.

Implement optional Dataset B.

Dataset A and B each contain:
- same or compatible crystal/basis;
- moment information;
- Jij.

Typical intended use:
    A = LKAG
    B = frozen magnon

Validate structural compatibility before comparing them.

Add comparison plots for:

1. raw J(q) eigenvalues;
2. robust-only J_MM(q);
3. dressed J_eff(q);
4. predicted ordering q;
5. FM magnon spectrum where stable;
6. spin stiffness;
7. real-space Jij vs distance/shell.

Add a diagnostic summary such as:

    Dataset A raw ordering: AF-like at q = ...
    Dataset A dressed ordering: AF-like at q = ...
    Dataset B raw ordering: FM at Gamma
    Dressing changes/does not change ordering.

Do not make causal claims automatically.

Implement induced-moment-response prediction:

For a chosen coherent spin spiral on robust sites, calculate:

    m_ind(q) / m_ind(0)

from the selected response model.

Allow an optional external comparison file:

    qx qy qz m_ind

or path-coordinate + m_ind.

This can represent self-consistent DFT induced moments from frozen-magnon calculations.

Plot:
    model response
    DFT response

and provide quantitative mismatch metrics.

If supplied DFT m_ind(q) strongly disagrees with the J-weighted model, report:

    "The input Jij do not reproduce the supplied induced-moment response under the K=J approximation."

Do NOT report:
    "LKAG is wrong"

because that requires broader evidence.

Provide export of all comparison tables.

Tests:
- identical A/B datasets;
- intentionally sign-reversed dataset;
- raw AF but dressed FM analytic toy case;
- raw and dressed AF case;
- synthetic m_ind(q) matching exactly;
- synthetic mismatching response.

Checklist:
[x] Dual-dataset compatibility checks implemented.
[x] Raw/dressed comparison plots implemented.
[x] Ordering comparison implemented.
[x] Magnon comparison implemented.
[x] m_ind(q) prediction implemented.
[x] Optional DFT response comparison implemented.
[x] Scientifically cautious diagnostics implemented.
[x] Export tables implemented.
[x] Tests pass.
[x] Documentation updated.

Commit message:
IMX-06 add exchange-dressing comparison diagnostics
