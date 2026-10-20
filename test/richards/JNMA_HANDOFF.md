# Handoff: Richards FCT in proteus (JNMA implicit FCT and WRR semi-implicit FCT)

Written 20 Oct 2026 at the end of a Claude Code session with C. Kees. The session worked in a local
folder `~/Richards_JNMA`, which holds the JNMA manuscript (not in this repo) with this repo cloned
inside it. Everything needed to continue is in this repo on branch **`jnma-fct-1.9.x`** of
`https://github.com/cekees/proteus` (head `78b151c2` or later). This file is self-contained.

## Priorities for the next session
1. ~~Fix the accuracy loss of the WRR semi-implicit FCT on Q1 meshes~~ **Done** (§4).
2. ~~Implement the fully implicit JNMA FCT in proteus~~ **Done** (§5).
3. Smaller open items (§6).

## 1. Background
- **WRR paper** (Barua, Kees & Kuzmin, *Bound-preserving FCT methods for Richards' equation*, WRR, under
  review; arXiv:2604.14107). Its *semi-implicit* FCT is implemented here as `STABILIZATION_TYPE=2 FCT=True`.
  Input decks are in `test/richards/WRR_input_files/`.
- **JNMA manuscript** (Dielmann, Kuzmin & Kees, *An implicit flux-corrected transport method for Richards'
  equation*, in preparation; results computed with MFEM). Its *implicit* FCT is specified in §5 below.
  The goal is to reproduce the JNMA results with proteus, and then to implement the JNMA method here.

## 2. Environment and build (macOS arm64, as used)
```bash
mamba env create -n proteus-jnma-190 -f environment-openmpi-dev.yml
```
```bash
conda activate proteus-jnma-190 && pip install --no-build-isolation --no-cache-dir -v -e .
```
After any C++ change in `proteus/richards/Richards.h` or `proteus/pskRelations.h`, rerun the `pip install`
line; Python-only changes need no rebuild. Regression check (about 20 s, must stay 7 passed):
```bash
cd test/richards && python -m pytest -q test_richards.py
```

## 3. What this branch adds (on top of `cekees/1.9.x`)
| commit | content |
|---|---|
| 563c8a4b | `Richards.ThetaScheme` θ option (`timeIntegrationTheta` in the numerics file). Galerkin: full θ-method, Jacobian checked against finite differences. Edge-based schemes: low order stays BE, θ only on the FCT high-order flux (`Theta_h`). `PSK_type='HornungMessing'` (4) with inverses. Fix: NaN residual in saturated regions without gravity (metric τ = 1/√0). Newton line search now halves on a non-finite residual. New example `test/richards/hornung_messing/`. |
| 4915ad6e | `Coefficients(FCT_SATURATED_LOW_ORDER=...)`: Zalesak Q_i^± = 0 at DOFs whose low-order predictor is saturated (not for Gardner). `forceStrongConditions` is now honoured (it was silently forced False). With strong BCs, `u_dof_old` at Dirichlet DOFs is set to g(tⁿ) (was g(tⁿ⁺¹), which corrupted the (1−θ) terms). |
| 5b9f4f23 | `FCT_SATURATED_LOW_ORDER=True` is the default. |
| 78b151c2 | `hornung_messing`: `quad=True` (Q1, 2×2 Gauss, the JNMA element), Q1 L² error integration, `MESH=q1` in the driver. |

Example (Hornung–Messing, unit square, no gravity, T = 0.05, exact solution; see its README):
```bash
cd test/richards/hornung_messing && MESH=q1 ./run_hm_convergence.sh
```
This takes about 3 min serial and writes `convergence_q1_theta1.0.txt` / `convergence_q1_theta0.5.txt`. Without
`MESH=q1` it runs P1 triangles and writes `convergence_theta*.txt`.
Context options: `STABILIZATION_TYPE` (0 Galerkin, 2 edge-based low order), `FCT`, `nn`, `nDT`,
`theta_time`, `quad`, `strong_bc`, `fct_sat_lo`.

## 4. Priority 1 (RESOLVED): WRR FCT was only first order on Q1
**Fix:** `calculateResidual_entropy_viscosity` now sets `mDotLow` at strong-Dirichlet DOFs (bc_mask = 0) to
(m(gⁿ⁺¹) − m(gⁿ))/Δt instead of the masked row's flux divergence (candidate 1 below). Q1 FCT at θ = 1, h = 1/2 … 1/64:
1.79e-4, 3.11e-4, 2.02e-4, 6.83e-5, 1.11e-5, **1.94e-6** (orders 2.6, 2.5), below the JNMA implicit FCT (3.44e-6).
P1 is slightly better at every level (1.575e-6 → 1.546e-6 at h = 1/64). θ = ½ is unchanged (still first order).
`test_richards.py`: 7 passed. A fully unlimited correction (no bounds, no saturated rule) is unusable on
both meshes (6e-2 Q1, 4e-2 P1 at h = 1/32), so the limiter and the saturated-DOF rule are essential. The
notes below are the original diagnosis.

L² errors at h = 1/64, θ = 1, Δt = 0.2h² (the JNMA convergence setup):

| | low order | Galerkin | WRR FCT |
|---|---|---|---|
| proteus Q1 | 1.154e-4 | 2.340e-7 | **1.613e-4 (order ≈ 1.1)** |
| proteus P1 | 6.989e-5 | 2.292e-7 | 1.575e-6 (order ≈ 2.6) |
| JNMA MFEM (Q1, implicit FCT) | 1.15e-4 | 1.01e-6 | 3.44e-6 |

What is known:
- The Q1 low order reproduces the JNMA MFEM low-order column to every printed digit at all six levels
  (1.78e-4 … 1.15e-4). Problem, element, Δt and low-order operator are therefore identical.
- At h = 1/32 the error does not come from the limiter: unlimited (bounds widened to ±1e10) gives the same
  3.5e-4. It is not on the boundary (Dirichlet DOFs exact to 1e-14 with `strong_bc`), and it is not
  in the saturated zone (max 4e-4 there).
- It is concentrated **one cell from the boundary, on the unsaturated side of the interface ξ = x−y−t ∈
  [0, 0.05)**. The worst node, at (0.094, 0.031), has p_ex = −0.0062 and p_h = −0.0136. There s′(p) ≈ 0.025,
  so a mass error is amplified about 40× by the θ → p recovery.
- On P1 the same scheme is second order, so the difference is Q1-specific. Q1 couples diagonal neighbours
  in both M_C and the stiffness. Candidates to test:
  1. The mass-antidiffusion term M_C,ij(ṁᴸ_i − ṁᴸ_j) in `FCTStep` (`Richards.h`), with ṁᴸ = low-order flux
     divergence / m_i. At DOFs next to strong-Dirichlet nodes ṁᴸ of the Dirichlet neighbour is not a time
     derivative (its row is masked). Try ṁ_j = (m_j(u_j) − m_j(u_jⁿ))/Δt there, or exclude edges to masked
     DOFs.
  2. Near-saturation conditioning: extend the saturated-node rule to DOFs with small s′(pᴸ), or limit in p
     instead of θ there.
  3. Whether the correction should use θᴴ from a Galerkin predictor (as the WRR *text* says) instead of
     the low-order state (what the code does; see §6).
- Diagnostic scripts used: monkeypatch `Richards.LevelModel.FCTStep` to record `u[0].dof` before and after,
  then bin |p_h − p_ex| by distance to the boundary and by ξ. They are easy to recreate from
  `hm_l2_error.py` (`p_exact`, `integrate`).
- Done when: WRR FCT on Q1 converges at about second order at θ = 1; `test_richards.py` still passes;
  P1 results are unchanged or better.

## 5. Priority 2 (DONE): the fully implicit JNMA FCT
**Implemented** as `Coefficients(FCT=True, STABILIZATION_TYPE=2, FCT_TYPE='implicit', FCT_PRELIMITER_POWER=3.0)`.
Hornung–Messing context: `fct_type='implicit'`; sweep scheme `IFCT`.
- `LevelModel.fct_phase` ∈ {predictor, flux, final} selects the kernel through `kernelStabilizationType()`:
  Galerkin (0) for the predictor, the edge-based low order (2) to evaluate the fluxes at pᴴ, and Implicit_FCT (3)
  for the final solve. Type 3 is the low order plus `fluxCorrection` = −(m_i/Δt)(s^{n,FCT}_i − sⁿ_i).
- `Newton.solve` calls `FCTStep` → `ImplicitFCTStep` after the predictor converges. That evaluates the fluxes,
  runs the C++ `ImplicitFCTStep` (steps 2–5; three passes with ghost scatters in parallel), and sets
  `fct_resolve`. Newton then re-solves from pᴴ. `Coefficients.preStep` rearms the predictor each step.
- Q1 errors match the JNMA tables to about 1% at θ = 1 (all levels) and within 5% at θ = ½ for h ≤ 1/16 (see the
  hornung_messing README and `convergence_q1_theta*.txt`).
- Finding: without the prelimiter, Q1 θ = 1 at h = 1/64 gives 1.01e-6 instead of 3.43e-6. The FCT error of the
  JNMA tables is therefore set mostly by the prelimiter, not by the Galerkin target (§7).
- Not yet covered: parallel runs, β ≠ 0 (the bounds assume a constant density per DOF), and cases with gravity
  or ponding. Newton is used, not the L-scheme.

### Original specification
Notation: p pressure head, u = p + z hydraulic head, s(p) water content, m_ij consistent mass, m_i lumped
mass, a_ij Galerkin stiffness (∫ η(p_h)∇φ_j·∇φ_i), q_ij = (∇φ_j,∇φ_i), and the low-order upwind operator
ã_ij = η_ij q_ij, with η_ij = η(p_i) if q_ij(u_i−u_j) < 0, η(p_j) if > 0, and the average if equal. The diagonal
is set so rows sum to zero. This low order **is** proteus `STABILIZATION_TYPE=2 FCT=False` (verified above). The
tie rule and M-matrix clipping `fmax(0,-T_ij)` differ only formally.

Per time step:
1. **High-order predictor pᴴ**: solve the Galerkin θ-scheme
   M(s(p)−s(pⁿ))/Δt + θA(p)u(p) + (1−θ)A(pⁿ)u(pⁿ) = 0.
   JNMA applies M to the nodal interpolant of s; proteus Galerkin evaluates s at quadrature points.
2. **Raw fluxes** at pᴴ (θ-method high order, backward-Euler low order):
   f_ij = m_ij(ṡᴴ_i − ṡᴴ_j) − ãᴴ_ij(uᴴ_i−uᴴ_j) + θ aᴴ_ij(uᴴ_i−uᴴ_j) + (1−θ) aⁿ_ij(uⁿ_i−uⁿ_j),
   with ṡᴴ_i = (s(pᴴ_i) − s(pⁿ_i))/Δt.
3. **Nodal-variation prelimiter**: α_i^nv = 1 − ψ(|Σ_j (uᴴ_i−uᴴ_j)| / Σ_j |uᴴ_i−uᴴ_j|) with ψ(r) = r³
   (α_i^nv = 1 when the denominator is 0), over j ∈ N_i* = {j ≠ i : m_ij > 0}. Then
   f̄_ij = min(α_i^nv, α_j^nv) f_ij.
4. **Zalesak with GLOBAL bounds** [s_min, s_max] = [θ_R, θ_S] (Hornung–Messing: [0, π²/2]), relative to sⁿ:
   P_i^± = Σ max/min(0, f̄_ij), Q_i^± = (m_i/Δt)(s_max/min − sⁿ_i), R_i^± = min(1, Q_i^±/P_i^±) (1 if P = 0),
   α_ij = min(R_i^+, R_j^−) if f̄_ij > 0, min(R_i^−, R_j^+) if f̄_ij < 0.
5. **Corrected datum**: s_i^{n,FCT} = sⁿ_i + (Δt/m_i) Σ_j α_ij f̄_ij.
6. **Final implicit low-order solve** for pⁿ⁺¹: M_L(s(p) − s^{n,FCT})/Δt + Ã(p)u(p) = 0. Fluxes and factors stay
   frozen at pᴴ; there is no θ → p inversion.

JNMA solves all nonlinear systems with an L-scheme, J = (L/Δt)M_L + Ã(pᵏ) with L ≥ sup|s′|. Newton is fine
for a first implementation.

Mapping to proteus:
- Step 6 is the existing low-order residual with `mn` (= sⁿ) replaced by s^{n,FCT}. The kernel already adds
  `fluxCorrection` to the residual under `STABILIZATION_TYPE=Implicit_FCT` (=3); see the end of
  `calculateResidual_entropy_viscosity`. That hook, or simply overwriting `mn`, gives the final solve.
- Step 1 needs a Galerkin solve in the same time step: either a second model object, or a two-pass
  `NonlinearSolvers` driver that toggles the residual between `calculateResidual` and
  `calculateResidual_entropy_viscosity`.
- Steps 2–5 are a new limiter (global bounds relative to sⁿ, prelimiter, fluxes at pᴴ) next to `FCTStep`.
  The existing `FCTStep` uses local bounds relative to mᴸ, has no prelimiter, and uses fluxes at pᴸ.
- Validation: the JNMA Tables 1–2 (§7 below) and `MESH=q1 ./run_hm_convergence.sh`. The low-order column
  already matches.

## 6. Other open items
- **WRR text vs. code.** The WRR text writes the raw flux with θᴴ, ψᴴ (a high-order solution). The code
  (`calculateResidual_entropy_viscosity` → `dt_times_fH_minus_fL`, and `FCTStep`) builds it entirely at the
  low-order state ψᴸ, uses ṁᴸ in the mass term, and uses local bounds from mᴸ. One of them should be
  corrected before the WRR revision.
- **WRR FCT with θ = ½** is not a usable scheme: first order when limited (4.9e-3 at h = 1/64 on Q1) and
  divergent when unlimited.
- **Galerkin + Newton robustness** on sharp 1-D infiltration fronts (Celia, Szymkiewicz): it works only in a
  Δt window. If Δt is too large it collapses; if too small it stalls on the first step from the initial
  jump. Galerkin never started on Szymkiewicz. A Rannacher start-up or an L-scheme/Picard solver would help.
- **`WRR_input_files/bio2d`** passes `forceStrongConditions=True`, which the code now honours. Its published
  results were computed with weak BCs, so a rerun may differ. Ponded 2-D runs may also differ under the
  new default `FCT_SATURATED_LOW_ORDER=True`.
- **`WRR_input_files/test_1/re_vgm_sand_10m_1d_p.py`** lacks `galerkin = False`, so `run_test_1.sh` fails.
  The digitized Celia data (`data_celia*.csv`) that `plot_test_1.py` reads is not committed.
- The VG–Mualem law hard-codes pore connectivity ℓ = ½.

## 7. Highlighted finding: MFEM Galerkin accuracy
With identical inputs (Q1, same h and Δt), **proteus Galerkin errors are 2.4–6× smaller than the JNMA MFEM
Galerkin errors on the finer meshes (h ≤ 1/16), for both θ = 1 and θ = ½**. On the coarsest meshes MFEM is more
accurate (h = 1/2 at θ = 1: 4.25e-4 vs 1.00e-3; h ≤ 1/4 at θ = ½). Full tables: `hornung_messing/convergence_q1_theta*.txt`.

| h | MFEM θ=1 | proteus θ=1 | MFEM θ=½ | proteus θ=½ |
|---|---|---|---|---|
| 1/16 | 1.84e-5 | 4.15e-6 | 1.33e-5 | 5.52e-6 |
| 1/32 | 5.24e-6 | 8.64e-7 | 4.15e-6 | 1.07e-6 |
| 1/64 | 1.01e-6 | 2.34e-7 | 7.54e-7 | 2.32e-7 |

Both converge at about second order. A known difference is the storage term (MFEM: consistent mass on the
nodal interpolant of s; proteus: s(p_h) at the quadrature points). Testing it is cheap: add an option to
proteus Galerkin that uses Σ_j m_ij s(p_j) and rerun the Q1 sweep. If it explains the factor, the JNMA
draft should say so, because the JNMA FCT errors (3.44e-6 at h = 1/64) are bounded by this Galerkin target.

Also for the JNMA authors: their Celia figure (labelled 24 h, 1 m column) is not reproducible from the stated
inputs. The same low order fits best at about 1.5 h in a 0.2 m column, and the front shape still differs. The
proteus 24 h result agrees with an independent Picard solver (front 0.565 m below the top). Their
Szymkiewicz figures are consistent.

## 8. Reference values from the JNMA draft (MFEM, Hornung–Messing)
θ = 1 (Δt = 0.2h²), h = 1/2 … 1/64:
- LO: 1.78e-4, 1.10e-3, 7.87e-4, 4.34e-4, 2.27e-4, 1.15e-4
- Galerkin: 4.25e-4, 2.18e-4, 7.98e-5, 1.84e-5, 5.24e-6, 1.01e-6
- FCT: 9.78e-4, 6.74e-4, 1.99e-4, 5.23e-5, 1.42e-5, 3.44e-6

θ = ½ (Δt = 0.1h):
- Galerkin: 5.05e-4, 1.22e-4, 5.72e-5, 1.33e-5, 4.15e-6, 7.54e-7
- FCT: 9.86e-4, 5.66e-4, 1.71e-4, 4.63e-5, 1.31e-5, 3.28e-6
