Task IMX-03 — Implement induced/slave moment response.

Read HEAD carefully.

Goal:
Implement a scientifically explicit Polesya-like induced-moment layer without yet performing Mryasov downfolding.

Add user classification:
    robust sites/sublattices
    induced sites/sublattices

Do not auto-classify based only on moment size.

Implement two response modes.

A. Historical local/unweighted mode

    m_nu = X_nu sum_{j in selected neighbourhood} M_j

Neighbourhood may be:
- first geometric shell;
- user-selected cutoff;
- explicit list.

B. J-weighted nonlocal approximation

    m(q) =
        X [K_mM(q) M(q) + K_mm(q) m(q)]

with default approximation:

    K = J_input

and therefore:

    m(q) =
        [I - X K_mm(q)]^{-1}
        X K_mM(q) M(q).

Label this everywhere as:
    J-weighted induced-response approximation.

Never state or imply that conventional LKAG Jij are formally identical to the true induction kernel K.

Implement X inference from the reference collinear state where possible:

    m_nu^0 =
        X_nu sum_j K_nu,j M_j^0

Use the actual model conventions and dimensions carefully.

Provide:
- inferred X per induced site/sublattice;
- denominator/source field used to infer X;
- warnings for cancellation, near-zero denominator, negative or suspicious X;
- optional user override of X;
- m_ind(q)/m_ind(0);
- individual induced-sublattice response;
- condition number / proximity to singularity of:
      I - X K_mm(q).

A near singularity should be flagged as a possible soft/Stoner-like response and not silently regularized.

Real-space helper:
Given an arbitrary robust spin configuration, evaluate instantaneous slave induced moments using the corresponding K-weighted field.

This routine must not propagate induced moments dynamically.

Tests:
1. one robust + one induced site;
2. one induced site coupled equally to two robust spins:
   parallel -> full reference moment;
   antiparallel -> cancellation;
3. unequal K weights;
4. finite induced-induced coupling;
5. matrix solution against direct analytic result;
6. q-space vs real-space consistency.

Document carefully:
- X is susceptibility-like;
- K is the induction kernel;
- K=J is an optional approximation;
- induced longitudinal amplitude variation is slaved, not an independent LSF degree of freedom.

Checklist:
[ ] Sublattice classification implemented.
[ ] Unweighted Polesya mode implemented.
[ ] J-weighted mode implemented.
[ ] X inference implemented.
[ ] User X override implemented.
[ ] Induced-induced response implemented.
[ ] Singularity diagnostics implemented.
[ ] m_ind(q) output implemented.
[ ] Analytic tests pass.
[ ] Documentation updated.

Commit message:
IMX-03 add Polesya-like induced moment response
