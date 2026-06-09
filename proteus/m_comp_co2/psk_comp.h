#ifndef MPHASE_CO2_PSK_MODELS_H
#define MPHASE_CO2_PSK_MODELS_H
#include <algorithm>
#include <cmath>

namespace proteus
{
namespace m_comp_co2
{
namespace psk
{

inline void vgm_wetting(const double psiC,
                        const double alpha,
                        const double n_vg,
                        const double thetaR,
                        const double thetaSR,
                        double &thetaW,
                        double &DthetaW_DpsiC,
                        double &KWr,
                        double &DKWr_DpsiC)
{
  const double m_vg   = 1.0 - 1.0 / n_vg;
  const double thetaS = thetaR + thetaSR;
  if (psiC > 0.0) {
    double pcBar     = alpha * psiC;
    double pcBarStar = pcBar;
    if (pcBar < 1.0e-8) pcBarStar = 1.0e-8;
    const double pcBar_nM2       = pow(pcBarStar, n_vg - 2);
    const double pcBar_nM1       = pcBar_nM2 * pcBar;
    const double pcBar_n         = pcBar_nM1 * pcBar;
    const double onePlus_pcBar_n = 1.0 + pcBar_n;

    const double sBar = pow(onePlus_pcBar_n, -m_vg);
    /* using -mn = 1-n */
    const double DsBar_DpsiC =
        alpha * (1.0 - n_vg) * (sBar / onePlus_pcBar_n) * pcBar_nM1;

    const double vBar  = 1.0 - pcBar_nM1 * sBar;
    const double vBar2 = vBar * vBar;
    const double DvBar_DpsiC =
        -alpha * (n_vg - 1.0) * pcBar_nM2 * sBar - pcBar_nM1 * DsBar_DpsiC;

    thetaW        = thetaSR * sBar + thetaR; // thetaS;//
    DthetaW_DpsiC = thetaSR * DsBar_DpsiC;   // 0.0;//

    const double sqrt_sBar = sqrt(sBar);
    double       sqrt_sBarStar = sqrt_sBar;
    if (sqrt_sBar < 1.0e-8) sqrt_sBarStar = 1.0e-8;
    KWr        = sqrt_sBar * vBar2;
    DKWr_DpsiC = ((0.5 / sqrt_sBarStar) * DsBar_DpsiC * vBar2
                  + 2.0 * sqrt_sBar * vBar * DvBar_DpsiC);
  } else {
    thetaW        = thetaS;
    DthetaW_DpsiC = 0.0;
    KWr           = 1.0;
    DKWr_DpsiC    = 0.0;
  }
}

// =============================================================================
// Two-phase extensions (NOT yet wired into m_comp_co2.h's residual; module
// behavior remains single-phase Richards. These are the closures Phase B will
// consume for the non-wetting equation and the capillary-pressure coupling).
//
// vgm_kr_nonwetting : Mualem-VG non-wetting relative permeability k_rn(psiC).
//                     Standard form k_rn = sqrt(1 - S_e) * Y^2, where
//                     Y = pcBar^(n-1) * S_e (= 1 - vBar of the wetting branch).
//                     Returns 0 in the saturated branch (psiC <= 0, S_n = 0).
//
// vgm_pc_from_Se    : Inverse capillary curve, given effective saturation
//                     S_e = (theta_w - theta_R) / theta_SR (= sBar).
//                     pcBar = (S_e^(-1/m) - 1)^(1/n);  pc = pcBar / alpha.
//                     Returns (pc, dpc/dSe) with dpc/dSe < 0. Both clamped to
//                     0 at S_e >= 1 (no capillary suction in the fully-wet
//                     branch); near S_e = 0 the value diverges - clamped via
//                     a small floor so derivatives remain finite.
// =============================================================================

inline void vgm_kr_nonwetting(const double psiC,
                              const double alpha,
                              const double n_vg,
                              double &KNr,
                              double &DKNr_DpsiC)
{
  const double m_vg = 1.0 - 1.0 / n_vg;
  if (psiC > 0.0) {
    double pcBar     = alpha * psiC;
    double pcBarStar = pcBar;
    if (pcBar < 1.0e-8) pcBarStar = 1.0e-8;
    const double pcBar_nM2       = pow(pcBarStar, n_vg - 2);
    const double pcBar_nM1       = pcBar_nM2 * pcBar;
    const double pcBar_n         = pcBar_nM1 * pcBar;
    const double onePlus_pcBar_n = 1.0 + pcBar_n;

    const double sBar = pow(onePlus_pcBar_n, -m_vg);
    const double DsBar_DpsiC =
        alpha * (1.0 - n_vg) * (sBar / onePlus_pcBar_n) * pcBar_nM1;

    // Y = 1 - (1 - sBar^(1/m))^m = pcBar^(n-1) * sBar  ( = 1 - vBar )
    const double Y = pcBar_nM1 * sBar;
    const double dY_DpsiC = (n_vg - 1.0) * alpha * pcBar_nM2 * sBar
                          + pcBar_nM1 * DsBar_DpsiC;

    // sqrt(1 - sBar) factor; clamp for finite derivative as sBar -> 1.
    const double oneMinusSbar     = 1.0 - sBar;
    const double oneMinusSbarStar = (oneMinusSbar < 1.0e-12) ? 1.0e-12 : oneMinusSbar;
    const double sqrt_oms         = sqrt(oneMinusSbarStar);

    KNr        = sqrt_oms * Y * Y;
    DKNr_DpsiC = (-0.5 / sqrt_oms) * DsBar_DpsiC * Y * Y
               + 2.0 * sqrt_oms * Y * dY_DpsiC;
  } else {
    // Saturated branch (S_w = 1, S_n = 0): no non-wetting mobility.
    KNr        = 0.0;
    DKNr_DpsiC = 0.0;
  }
}

inline void vgm_pc_from_Se(const double Se,
                           const double alpha,
                           const double n_vg,
                           double &pc,
                           double &Dpc_DSe,
                           double &D2pc_DSe2)
{
  const double m_vg = 1.0 - 1.0 / n_vg;
  // Floor Se from below to keep pcBar_n bounded for Se near 0 (gas-saturated).
  // Raised from 1e-12 to 1e-3 to bound pc at the saturation cap; otherwise
  // pc reaches ~1e12/alpha at the floor and floating-point cancellation in
  // the antisymmetric edge flux F_ij + F_ji = 0 leaks mass on the order of
  // |pc|*eps_machine per edge. With the 1e-3 floor, |pc_max| ~ 1e3/alpha and
  // the antisymmetric roundoff drops by ~9 orders of magnitude.  BC closure
  // applies an equivalent cap at line 506 (Se_min_pc).
  const double SeStar  = (Se < 1.0e-3) ? 1.0e-3 : Se;
  const double SemInvM = pow(SeStar, -1.0 / m_vg);
  const double pcBar_n_raw = SemInvM - 1.0;
  // ------------------------------------------------------------------------
  // VGM regularization: SOFT FLOOR on pcBar_n.
  //
  //   pcBar_n = pcBar_n_raw + eps_pcBar
  //
  // This is C-infinity smooth across Se -> 1 (no kinks), bounds Dpc_DSe by
  // pcBar_n_soft^(1/n - 1) * (constant), and recovers the physical
  // pcBar_n_raw whenever raw >> eps_pcBar (i.e., away from full saturation).
  //
  // The earlier hard-cutoff regularization (pcBar_n = max(raw, eps)) made
  // Dpc_DSe C0-discontinuous at the threshold; Newton would stall when its
  // iterate crossed the threshold mid-step. The soft floor eliminates that.
  //
  // Cost: pc(Se = 1) = (eps_pcBar)^(1/n)/alpha ~ 0.001 instead of exactly 0.
  // Physically negligible (any DOF actually AT Se = 1 has zero gas flux
  // already because k_rn = 0 there).
  //
  // The legacy `if (Se >= 1.0)` and `pcBar_n_raw <= 0.0` early-returns are
  // dropped: with the soft floor, the formula is well-defined everywhere
  // including Se = 1 (pcBar_n_raw can dip slightly negative due to roundoff
  // when Se >= 1, but pcBar_n_soft remains >= eps_pcBar/2 > 0 as long as
  // |pcBar_n_raw| < eps_pcBar/2 which holds for Se in [1 - small, 1 + small]).
  // ------------------------------------------------------------------------
  const double eps_pcBar = 1.0e-3;
  // For Se >= 1 the raw value can be slightly negative due to roundoff;
  // clamp from below to keep the soft-floored pcBar_n strictly positive.
  const double pcBar_n_raw_clamped = (pcBar_n_raw > -0.5 * eps_pcBar)
                                       ? pcBar_n_raw
                                       : -0.5 * eps_pcBar;
  const double pcBar_n = pcBar_n_raw_clamped + eps_pcBar;
  const double pcBar = pow(pcBar_n, 1.0 / n_vg);
  pc = pcBar / alpha;

  // First derivative -- C-infinity smooth in Se now.
  //   d(pcBar_n)/d(Se)   = d(pcBar_n_raw)/d(Se)  (since eps_pcBar is constant)
  //                      = -(1/m) * Se^(-1/m - 1)
  //                      = -(1/m) * SemInvM / SeStar
  //   d(pcBar)/d(Se)     = (1/n) * pcBar_n^(1/n - 1) * d(pcBar_n)/d(Se)
  // For Se > 1 (numerical excursion) we hold dpcBar_n/dSe at its Se = 1 value
  // (which is small) -- no discontinuity introduced because pcBar_n_raw was
  // clamped above.
  // First derivative -- C-infinity smooth in Se now.
  //   d(pcBar_n)/d(Se)   = d(pcBar_n_raw)/d(Se)  (since eps_pcBar is constant)
  //                      = -(1/m) * Se^(-1/m - 1)
  //                      = -(1/m) * SemInvM / SeStar
  //   d(pcBar)/d(Se)     = (1/n) * pcBar_n^(1/n - 1) * d(pcBar_n)/d(Se)
  // For Se > 1 (numerical excursion) we hold dpcBar_n/dSe at its Se = 1 value
  // (which is small) -- no discontinuity introduced because pcBar_n_raw was
  // clamped above.
  const double dpcBar_n_dSe = -(1.0 / m_vg) * (SemInvM / SeStar);
  const double dpcBar_dSe   = (1.0 / n_vg) * (pcBar / pcBar_n) * dpcBar_n_dSe;
  // pc is built from SeStar = max(Se, 1e-3); below the floor pc is FLAT, so the
  // chain rule gives d(pc)/d(Se) = d(pc)/d(SeStar) * d(SeStar)/d(Se) with
  // d(SeStar)/d(Se) = 0 for Se < 1e-3.  Without this factor pc(Se) is flat but
  // Dpc_DSe is nonzero -> residual/Jacobian inconsistency at gas-saturated nodes
  // (Se_a -> 0), which stalls Newton near the saturation ceiling.
  const double dSeStar_dSe = (Se < 1.0e-3) ? 0.0 : 1.0;
  Dpc_DSe = (dpcBar_dSe / alpha) * dSeStar_dSe;

  // Second derivative (gas-eq (1,1) capillary sensitivity), also C-infinity.
  //   d2(pcBar_n)/d(Se)2 = (1/m)*(1/m + 1) * Se^(-1/m - 2)
  //                      = (1/m)*(1/m + 1) * SemInvM / Se^2
  //   d2(pcBar)/d(Se)2   = (1/n) * [(1/n - 1) * pcBar_n^(1/n - 2) * (d pcBar_n / dSe)^2
  //                                  + pcBar_n^(1/n - 1) * d2(pcBar_n)/d(Se)2]
  const double inv_m = 1.0 / m_vg;
  const double inv_n = 1.0 / n_vg;
  const double d2pcBar_n_dSe2 = inv_m * (inv_m + 1.0) * (SemInvM / (SeStar * SeStar));
  const double pcBar_n_pow_a  = pow(pcBar_n, inv_n - 2.0);                  // pcBar_n^(1/n - 2)
  const double pcBar_n_pow_b  = pow(pcBar_n, inv_n - 1.0);                  // pcBar_n^(1/n - 1)
  const double d2pcBar_dSe2 = inv_n * ((inv_n - 1.0) * pcBar_n_pow_a * dpcBar_n_dSe * dpcBar_n_dSe
                                       + pcBar_n_pow_b * d2pcBar_n_dSe2);
  D2pc_DSe2 = (d2pcBar_dSe2 / alpha) * dSeStar_dSe;   // 0 below the Se floor (flat pc)
}

// =============================================================================
// Inversion: given accumulated mass m (= rho * thetaW * (compressibility factor))
// recover u = -psiC. Two variants kept identical to the algebra previously
// inlined inside m_comp_co2.h.
//
// vgm_invert_analytic : closed-form VGM inverse, ignores compressibility (beta).
//                       Updates u only when thetaW = m/rho is strictly within
//                       (~thetaR, thetaS); leaves u unchanged otherwise.
//
// vgm_invert_newton   : Newton solve for rho * exp(beta*u) * thetaW(u) = m.
//                       Uses the analytic inverse for the initial guess,
//                       reverts to the input value on saturated /
//                       sub-residual / non-convergent cases.
// =============================================================================

inline void vgm_invert_analytic(const double m,
                                const double rho,
                                const double alpha,
                                const double n_vg,
                                const double thetaR,
                                const double thetaSR,
                                double &u)
{
  double psiC, pcBar, pcBar_n, sBar, thetaW, thetaS, m_vg;
  m_vg   = 1.0 - 1.0 / n_vg;
  thetaS = thetaR + thetaSR;
  thetaW = m / rho;
  if (thetaW > 1.01 * thetaR && thetaW < thetaS) {
    sBar    = (thetaW - thetaR) / thetaSR;
    pcBar_n = pow(sBar, -1.0 / m_vg) - 1.0;
    pcBar   = pow(pcBar_n, 1.0 / n_vg);
    psiC    = pcBar / alpha;
    u       = -psiC;
  }
}

inline void vgm_invert_newton(const double m,
                              const double rho,
                              const double beta,
                              const double alpha,
                              const double n_vg,
                              const double thetaR,
                              const double thetaSR,
                              double &u)
{
  const double u_prev = u;

  const double thetaS = thetaR + thetaSR;
  const double m_vg   = 1.0 - 1.0 / n_vg;

  const double psiC0 = -u;
  if (psiC0 <= 0.0) { return; } // saturated, no inversion
  const double rhom0 = rho * std::exp(beta * u); // first guess
  const double thetaW_imp = m / rhom0;
  if (thetaW_imp < 1.01 * thetaR) { return; } // no inversion below residual saturation
  const double thetaEps = 1e-12;
  double m_target = m;
  if (thetaW_imp > 0.99 * thetaS) {
    const double thetaWc = std::min(thetaW_imp, thetaS - thetaEps);
    m_target = rhom0 * thetaWc;
  }

  // Use the analytic van Genuchten inverse as the initial guess for Newton.
  {
    const double thetaW_guess = std::min(m_target / rhom0, thetaS - thetaEps);
    if (thetaW_guess > thetaR + thetaEps && thetaW_guess < thetaS - thetaEps) {
      const double sBar = (thetaW_guess - thetaR) / thetaSR;
      if (sBar > 0.0 && sBar < 1.0) {
        const double pcBar_n = std::pow(sBar, -1.0 / m_vg) - 1.0;
        if (pcBar_n > 0.0) {
          const double pcBar = std::pow(pcBar_n, 1.0 / n_vg);
          const double u_guess = -pcBar / alpha;
          if (std::isfinite(u_guess) && u_guess < 0.0) {
            u = u_guess;
          }
        }
      }
    }
  }

  /*----------------------------------------------------
    Newton solve (UNSATURATED ONLY)
  ----------------------------------------------------*/
  const int    maxIts = 50;
  const double tol    = 1e-7 * std::max(1.0, std::fabs(m));
  const double duMax  = 5e-2;

  auto theta_and_dtheta_du = [&](double u,
                                 double &thetaW,
                                 double &dtheta_du) -> bool
  {
    const double psiC = -u;

    if (psiC <= 0.0) { return false; } // no inversion in saturation
    // van Genuchten relations
    const double pcBar     = alpha * psiC;
    const double pcBarStar = (pcBar < 1e-12) ? 1e-12 : pcBar;
    const double pcBar_nM2       = std::pow(pcBarStar, n_vg - 2.0);
    const double pcBar_nM1       = pcBar_nM2 * pcBar;
    const double pcBar_n         = pcBar_nM1 * pcBar;
    const double onePlus_pcBar_n = 1.0 + pcBar_n;
    const double sBar = std::pow(onePlus_pcBar_n, -m_vg);

    const double DsBar_DpsiC =
        alpha * (1.0 - n_vg) * (sBar / onePlus_pcBar_n) * pcBar_nM1;

    thetaW    = thetaR + thetaSR * sBar;
    dtheta_du = -thetaSR * DsBar_DpsiC;
    if (thetaW <= thetaR + thetaEps) return false;
    if (thetaW >= thetaS - thetaEps) return false;
    return true;
  };

  for (int it = 0; it < maxIts; ++it)
  {
    if (-u <= 0.0) { u = u_prev; return; }

    double thetaW, dtheta_du;
    if (!theta_and_dtheta_du(u, thetaW, dtheta_du)) {
      u = u_prev; return;
    }

    const double rhom = rho * std::exp(beta * u);
    const double g  = rhom * thetaW - m_target;
    if (std::fabs(g) < tol) return;

    const double gp = rhom * (beta * thetaW + dtheta_du);
    // guard against near-zero derivative
    const double gpTol = 1e-14 * std::max(1.0, std::fabs(rhom * thetaW));
    if (std::fabs(gp) < gpTol) { u = u_prev; return; } // revert if g' bad

    double du = -g / gp;
    if (du >  duMax) du =  duMax;
    if (du < -duMax) du = -duMax;

    u += du;
    // after update, if we stepped into saturation, revert
    if (-u <= 0.0) { u = u_prev; return; }
  }
  u = u_prev;
}

// =============================================================================
// Brooks-Corey (Burdine relative-permeability) closures.
// Parallel API to the vgm_* functions: same parameter slots, same output args.
// In the BC parameterisation the second numeric parameter is the pore-size
// index lambda (taking the slot of vgm's n_vg). alpha = 1/p_d is the inverse
// entry-pressure head.
//
//   S_e = (alpha * p_c)^(-lambda)              for alpha*p_c >= 1
//   S_e = 1                                    for alpha*p_c <  1   (saturated)
//   theta_w = theta_R + theta_SR * S_e
//   k_rw    = S_e^((2+3*lambda)/lambda)        (Burdine)
//   k_rn    = (1-S_e)^2 * (1 - S_e^((2+lambda)/lambda))
//   p_c(S_e) = S_e^(-1/lambda) / alpha         (analytic inverse)
//
// BC has a discontinuity in (k_rw, k_rn, theta_w) derivatives at p_c = p_d
// (alpha*p_c = 1). Newton typically needs a smoothed regularisation here in
// production; that's deliberately not added at this layer - callers can wrap
// these closures with their own smoothing if needed.
// =============================================================================

inline void bc_wetting(const double psiC,
                       const double alpha,
                       const double lam,
                       const double thetaR,
                       const double thetaSR,
                       double &thetaW,
                       double &DthetaW_DpsiC,
                       double &KWr,
                       double &DKWr_DpsiC)
{
  const double thetaS = thetaR + thetaSR;
  if (psiC <= 0.0) {
    // saturated (no suction)
    thetaW        = thetaS;
    DthetaW_DpsiC = 0.0;
    KWr           = 1.0;
    DKWr_DpsiC    = 0.0;
    return;
  }
  const double pcBar = alpha * psiC;
  if (pcBar <= 1.0) {
    // suction below entry pressure: still fully wetting-saturated
    thetaW        = thetaS;
    DthetaW_DpsiC = 0.0;
    KWr           = 1.0;
    DKWr_DpsiC    = 0.0;
    return;
  }
  // unsaturated branch (pcBar > 1)
  const double Se        = pow(pcBar, -lam);
  const double dSe_DpsiC = -lam * alpha * pow(pcBar, -lam - 1.0);
  thetaW        = thetaR + thetaSR * Se;
  DthetaW_DpsiC = thetaSR * dSe_DpsiC;
  // Burdine k_rw
  const double exp_w = (2.0 + 3.0 * lam) / lam;
  KWr        = pow(Se, exp_w);
  DKWr_DpsiC = exp_w * pow(Se, exp_w - 1.0) * dSe_DpsiC;
}

inline void bc_kr_nonwetting(const double psiC,
                             const double alpha,
                             const double lam,
                             double &KNr,
                             double &DKNr_DpsiC)
{
  if (psiC <= 0.0) {
    // saturated (S_w = 1, S_n = 0): no non-wetting mobility
    KNr        = 0.0;
    DKNr_DpsiC = 0.0;
    return;
  }
  const double pcBar = alpha * psiC;
  if (pcBar <= 1.0) {
    // below entry pressure: gas cannot displace wetting phase
    KNr        = 0.0;
    DKNr_DpsiC = 0.0;
    return;
  }
  const double Se        = pow(pcBar, -lam);
  const double dSe_DpsiC = -lam * alpha * pow(pcBar, -lam - 1.0);
  // Burdine k_rn = (1 - S_e)^2 * (1 - S_e^((2+lambda)/lambda))
  const double exp_n  = (2.0 + lam) / lam;
  const double Y      = 1.0 - pow(Se, exp_n);
  const double oneMinusSe = 1.0 - Se;
  KNr = oneMinusSe * oneMinusSe * Y;
  // d/dS_e [ (1-S_e)^2 * Y ] = -2(1-S_e) Y + (1-S_e)^2 dY/dS_e
  // dY/dS_e = -exp_n * S_e^(exp_n - 1)
  const double dY_dSe   = -exp_n * pow(Se, exp_n - 1.0);
  const double dKNr_dSe = -2.0 * oneMinusSe * Y + oneMinusSe * oneMinusSe * dY_dSe;
  DKNr_DpsiC = dKNr_dSe * dSe_DpsiC;
}

inline void bc_pc_from_Se(const double Se,
                          const double alpha,
                          const double lam,
                          double &pc,
                          double &Dpc_DSe,
                          double &D2pc_DSe2)
{
  // ----------------------------------------------------------------------
  // Brooks-Corey p_c(S_e) with smoothed transition out of the physical range
  // (so Newton can move through Se = 1 without seeing a value-jump).
  //
  //   Se in [0, 1):       pc = (1/alpha) * Se^(-1/lambda)  (standard BC)
  //                       At Se -> 1: pc -> 1/alpha (the "entry pressure"
  //                       p_d). At Se -> 0: pc -> +infinity (asymptotic).
  //
  //   Se in [1, 1+delta]: pc smoothly ramps from 1/alpha down to 0 using a
  //                       cubic Hermite smoothstep phi(t) = 1 - 3t^2 + 2t^3
  //                       in t = (Se - 1)/delta. phi(0) = 1, phi(1) = 0,
  //                       phi'(0) = phi'(1) = 0.
  //
  //   Se > 1 + delta:     pc = 0.
  //
  // Value continuity at Se = 1: pc(1-) = pc(1+) = 1/alpha. ENTRY PRESSURE
  // PRESERVED. The previous "if (Se >= 1.0) { pc = 0 }" cutoff produced a
  // finite jump of size 1/alpha at Se = 1 and stalled Newton whenever an
  // iterate crossed this threshold.
  //
  // C1 at Se = 1: the Se >= 1 ramp's left slope is matched to the BC branch
  // slope -(1/lambda)/alpha (see that branch), so dpc/dSe is now continuous
  // across gas appearance.  (Previously the ramp left with slope 0, leaving a
  // slope kink that mispredicted Newton steps overshooting across Se = 1.)
  // ----------------------------------------------------------------------
  const double delta_smooth = 5.0e-2;

  if (Se >= 1.0 + delta_smooth) {
    pc        = 0.0;
    Dpc_DSe   = 0.0;
    D2pc_DSe2 = 0.0;
    return;
  }

  if (Se >= 1.0) {
    // Non-physical overshoot region (S_g < 0, only reached when a Newton iterate
    // crosses gas appearance from above).  Cubic-Hermite ramp from pc(1) = 1/alpha
    // (the Brooks-Corey entry pressure p_d) down to pc(1+delta) = 0.
    //
    // C1 ACROSS GAS APPEARANCE: the LEFT-end slope is matched to the BC branch
    // slope at Se = 1, dpc/dSe = -1/(alpha*lam) (see the Se in [Se_min_pc,1)
    // branch below).  The earlier ramp used phi'(0) = 0, leaving a slope kink at
    // Se = 1 (BC side -p_d/lam vs ramp side 0); a Newton iterate overshooting
    // across Se = 1 then saw an inconsistent linearization and mispredicted the
    // step.  Right end keeps value 0 and slope 0 so pc joins the pc = 0 region C1.
    //
    // Hermite basis on t = (Se - 1)/delta in [0,1] with endpoints
    //   left  (t=0): value p_d = 1/alpha, slope_t m0_t = (-1/(alpha*lam))*delta
    //   right (t=1): value 0,             slope_t 0
    // so only H00 (value) and H10 (left slope) contribute.
    const double inv_alpha = 1.0 / alpha;
    const double m0_t = (-inv_alpha / lam) * delta_smooth;   // BC slope*delta at Se=1
    const double t  = (Se - 1.0) / delta_smooth;
    const double t2 = t * t, t3 = t2 * t;
    const double H00  = 2.0 * t3 - 3.0 * t2 + 1.0;
    const double H10  = t3 - 2.0 * t2 + t;
    const double dH00 = 6.0 * t2 - 6.0 * t;
    const double dH10 = 3.0 * t2 - 4.0 * t + 1.0;
    const double d2H00 = 12.0 * t - 6.0;
    const double d2H10 = 6.0 * t - 4.0;
    pc        = H00 * inv_alpha + H10 * m0_t;
    Dpc_DSe   = (dH00 * inv_alpha + dH10 * m0_t) / delta_smooth;
    D2pc_DSe2 = (d2H00 * inv_alpha + d2H10 * m0_t) / (delta_smooth * delta_smooth);
    return;
  }

  // ----------------------------------------------------------------------
  // Gas-saturated cap.  BC's pc and dpc/dSe diverge as Se -> 0 (i.e.
  // S_n -> 1 - S_wr).  In the FluidFlower disk-injection case this kills
  // Newton once the disk fills toward the saturation ceiling -- the
  // Jacobian sees |dpc/dSe| ~ 1e2 .. 1e8 m/unit and the S_n step blows up.
  //
  // Fix: below Se_min_pc, freeze the BC tangent so pc and dpc/dSe stay
  // finite and the Jacobian stays bounded.  C^1 continuous at Se_min_pc
  // (value + slope match by construction).  d2pc/dSe2 has a finite jump
  // at the seam, which Newton tolerates (only pc and dpc/dSe enter the
  // residual and Jacobian).  Standard "Pc cap" trick from reservoir
  // simulation; doesn't change pc above Se_min_pc, so undisturbed in
  // normal operation.
  //
  // Tuning Se_min_pc: smaller = sharper cap (closer to true BC), larger
  // = more robust Newton but more physics distortion near the gas-sat
  // endpoint.  1e-2 is a typical default; reduce to 1e-3 if the cap is
  // active across an unacceptable fraction of the domain.
  // ----------------------------------------------------------------------
  const double Se_min_pc = 1.0e-2;
  if (Se < Se_min_pc) {
    const double SemInvLam_min = pow(Se_min_pc, -1.0 / lam);
    const double pc_min        = SemInvLam_min / alpha;
    const double slope_min     = -(SemInvLam_min / Se_min_pc) / (alpha * lam);
    pc        = pc_min + slope_min * (Se - Se_min_pc);
    Dpc_DSe   = slope_min;
    D2pc_DSe2 = 0.0;
    return;
  }

  // Standard BC in the physical range Se in [Se_min_pc, 1).
  const double SemInvLam = pow(Se, -1.0 / lam);
  pc = SemInvLam / alpha;
  // dpc/dS_e = -(1/(alpha*lambda)) * S_e^(-1/lambda - 1)
  //         = -(1/(alpha*lambda)) * SemInvLam / S_e
  Dpc_DSe = -(SemInvLam / Se) / (alpha * lam);
  // d2pc/dS_e2 = (1/(alpha*lambda)) * (1/lambda + 1) * S_e^(-1/lambda - 2)
  //            = (1/(alpha*lambda)) * (1/lambda + 1) * SemInvLam / S_e^2
  const double inv_lam = 1.0 / lam;
  D2pc_DSe2 = inv_lam * (inv_lam + 1.0) * (SemInvLam / (Se * Se)) / alpha;
}

inline void bc_invert_analytic(const double m,
                               const double rho,
                               const double alpha,
                               const double lam,
                               const double thetaR,
                               const double thetaSR,
                               double &u)
{
  const double thetaS = thetaR + thetaSR;
  const double thetaW = m / rho;
  if (thetaW > 1.01 * thetaR && thetaW < thetaS) {
    const double Se = (thetaW - thetaR) / thetaSR;
    if (Se > 0.0 && Se < 1.0) {
      const double pcBar = pow(Se, -1.0 / lam);
      const double psiC  = pcBar / alpha;
      u = -psiC;
    }
  }
}

inline void bc_invert_newton(const double m,
                             const double rho,
                             const double beta,
                             const double alpha,
                             const double lam,
                             const double thetaR,
                             const double thetaSR,
                             double &u)
{
  const double u_prev = u;
  const double thetaS = thetaR + thetaSR;

  const double psiC0 = -u;
  if (psiC0 <= 0.0) { return; } // saturated, no inversion
  const double rhom0 = rho * std::exp(beta * u);
  const double thetaW_imp = m / rhom0;
  if (thetaW_imp < 1.01 * thetaR) { return; } // below residual
  const double thetaEps = 1e-12;
  double m_target = m;
  if (thetaW_imp > 0.99 * thetaS) {
    const double thetaWc = std::min(thetaW_imp, thetaS - thetaEps);
    m_target = rhom0 * thetaWc;
  }

  // Analytic BC inverse as the Newton starting guess.
  {
    const double thetaW_guess = std::min(m_target / rhom0, thetaS - thetaEps);
    if (thetaW_guess > thetaR + thetaEps && thetaW_guess < thetaS - thetaEps) {
      const double Se = (thetaW_guess - thetaR) / thetaSR;
      if (Se > 0.0 && Se < 1.0) {
        const double pcBar  = pow(Se, -1.0 / lam);
        const double u_guess = -pcBar / alpha;
        if (std::isfinite(u_guess) && u_guess < 0.0) {
          u = u_guess;
        }
      }
    }
  }

  const int    maxIts = 50;
  const double tol    = 1e-7 * std::max(1.0, std::fabs(m));
  const double duMax  = 5e-2;

  auto theta_and_dtheta_du = [&](double u_in,
                                 double &thetaW,
                                 double &dtheta_du) -> bool {
    const double psiC = -u_in;
    if (psiC <= 0.0) return false;
    const double pcBar = alpha * psiC;
    if (pcBar <= 1.0) return false; // saturated side of entry pressure
    const double Se        = pow(pcBar, -lam);
    const double dSe_DpsiC = -lam * alpha * pow(pcBar, -lam - 1.0);
    thetaW    = thetaR + thetaSR * Se;
    dtheta_du = -thetaSR * dSe_DpsiC; // d(psiC)/d(u) = -1
    if (thetaW <= thetaR + thetaEps) return false;
    if (thetaW >= thetaS - thetaEps) return false;
    return true;
  };

  for (int it = 0; it < maxIts; ++it) {
    if (-u <= 0.0) { u = u_prev; return; }

    double thetaW, dtheta_du;
    if (!theta_and_dtheta_du(u, thetaW, dtheta_du)) {
      u = u_prev; return;
    }

    const double rhom = rho * std::exp(beta * u);
    const double g    = rhom * thetaW - m_target;
    if (std::fabs(g) < tol) return;

    const double gp    = rhom * (beta * thetaW + dtheta_du);
    const double gpTol = 1e-14 * std::max(1.0, std::fabs(rhom * thetaW));
    if (std::fabs(gp) < gpTol) { u = u_prev; return; }

    double du = -g / gp;
    if (du >  duMax) du =  duMax;
    if (du < -duMax) du = -duMax;
    u += du;
    if (-u <= 0.0) { u = u_prev; return; }
  }
  u = u_prev;
}

// =============================================================================
// Saturation-based variants (Phase B Step 3a). Same physics as the psiC-based
// closures above, but parameterised directly by effective saturation Se.
// Used by the formulation-(psi_w, S_w) wetting equation, where saturation is
// the primary variable rather than something inverted from psiC.
//
// Conventions:
//   Se in [0, 1] is the effective saturation = (theta_w - theta_R)/theta_SR.
//   In formulation B, u_v IS S_w; if a residual saturation theta_R/theta_S is
//   in play the caller must convert before calling these.
//   Outputs are derivatives with respect to Se (NOT psiC).
// =============================================================================

inline void vgm_wetting_from_Se(const double Se,
                                const double /*alpha*/,    // unused, kept for API parity
                                const double n_vg,
                                const double thetaR,
                                const double thetaSR,
                                double &thetaW,
                                double &DthetaW_DSe,
                                double &KWr,
                                double &DKWr_DSe)
{
  const double m_vg = 1.0 - 1.0 / n_vg;
  thetaW       = thetaR + thetaSR * Se;
  DthetaW_DSe  = thetaSR;
  if (Se >= 1.0) {
    KWr = 1.0;
    DKWr_DSe = 0.0;
    return;
  }
  if (Se <= 0.0) {
    KWr = 0.0;
    DKWr_DSe = 0.0;
    return;
  }
  // Floor for the powers near the endpoints.
  const double SeStar = (Se < 1.0e-12) ? 1.0e-12 : Se;
  const double SeInvM = pow(SeStar, 1.0 / m_vg);          // Se^(1/m)
  const double x_     = 1.0 - SeInvM;                     // 1 - Se^(1/m)
  const double xStar  = (x_ < 1.0e-12) ? 1.0e-12 : x_;
  const double y_     = pow(xStar, m_vg);                 // (1 - Se^(1/m))^m
  const double term   = 1.0 - y_;                         // 1 - (1 - Se^(1/m))^m
  const double sqrtSe = sqrt(SeStar);
  const double sqrtSe_safe = (sqrtSe < 1.0e-12) ? 1.0e-12 : sqrtSe;
  // Mualem k_rw = sqrt(Se) * [1 - (1 - Se^(1/m))^m]^2
  KWr = sqrtSe * term * term;
  // dterm/dSe = x^(m-1) * Se^(1/m - 1)  (chain rule through y = x^m, x = 1 - Se^(1/m))
  const double Dterm_DSe = pow(xStar, m_vg - 1.0) * pow(SeStar, 1.0 / m_vg - 1.0);
  DKWr_DSe = (0.5 / sqrtSe_safe) * term * term + 2.0 * sqrtSe * term * Dterm_DSe;
}

inline void vgm_kr_nonwetting_from_Se(const double Se_w,
                                      const double /*alpha*/,
                                      const double n_vg,
                                      double &KNr,
                                      double &DKNr_DSe,
                                      const double Se_trap = 1.0)
{
  // Gas-only residual trapping: see bc_kr_nonwetting_from_Se.  Remap the
  // wetting Se_w over the mobile range [0,Se_trap]; k_rn=0 for S_g<=S_gr.
  // Se_trap=1 -> original.  DKNr_DSe is d k_rn/d Se_w (1/Se_trap folded in).
  const double inv_trap = 1.0 / Se_trap;
  const double Se       = Se_w * inv_trap;
  const double m_vg = 1.0 - 1.0 / n_vg;
  if (Se >= 1.0) {
    KNr = 0.0;
    DKNr_DSe = 0.0;
    return;
  }
  if (Se <= 0.0) {
    // S_n = 1 -> k_rn = 1 in the limit; derivative degenerate.
    KNr = 1.0;
    DKNr_DSe = 0.0;
    return;
  }
  const double SeStar = (Se < 1.0e-12) ? 1.0e-12 : Se;
  const double SeInvM = pow(SeStar, 1.0 / m_vg);
  const double x_     = 1.0 - SeInvM;
  const double xStar  = (x_ < 1.0e-12) ? 1.0e-12 : x_;
  // Mualem k_rn = sqrt(1 - Se) * (1 - Se^(1/m))^(2m)
  const double y2m    = pow(xStar, 2.0 * m_vg);
  const double oneMinusSe     = 1.0 - Se;
  const double oneMinusSeStar = (oneMinusSe < 1.0e-12) ? 1.0e-12 : oneMinusSe;
  const double sqrt_oms       = sqrt(oneMinusSeStar);
  KNr = sqrt_oms * y2m;
  // d(y^(2m))/dSe = 2m * x^(2m - 1) * dx/dSe = -2 * x^(2m - 1) * Se^(1/m - 1)
  const double Dy2m_DSe = -2.0 * pow(xStar, 2.0 * m_vg - 1.0) * pow(SeStar, 1.0 / m_vg - 1.0);
  DKNr_DSe = ((-0.5 / sqrt_oms) * y2m + sqrt_oms * Dy2m_DSe) * inv_trap;
}

inline void bc_wetting_from_Se(const double Se,
                               const double /*alpha*/,
                               const double lam,
                               const double thetaR,
                               const double thetaSR,
                               double &thetaW,
                               double &DthetaW_DSe,
                               double &KWr,
                               double &DKWr_DSe)
{
  thetaW       = thetaR + thetaSR * Se;
  DthetaW_DSe  = thetaSR;
  if (Se >= 1.0) {
    // Saturated branch; BC has a one-sided derivative jump here.
    KWr = 1.0;
    DKWr_DSe = 0.0;
    return;
  }
  if (Se <= 0.0) {
    KWr = 0.0;
    DKWr_DSe = 0.0;
    return;
  }
  // Burdine k_rw = Se^((2 + 3*lambda)/lambda)
  const double exp_w = (2.0 + 3.0 * lam) / lam;
  KWr      = pow(Se, exp_w);
  DKWr_DSe = exp_w * pow(Se, exp_w - 1.0);
}

inline void bc_kr_nonwetting_from_Se(const double Se_w,
                                     const double /*alpha*/,
                                     const double lam,
                                     double &KNr,
                                     double &DKNr_DSe,
                                     const double Se_trap = 1.0)
{
  // ---------------------------------------------------------------------
  // Gas-only residual trapping (Jun 2026): remap the wetting effective
  // saturation Se_w onto [0,1] over the MOBILE gas range Se_w in [0,Se_trap],
  //   Se_trap = (1 - S_wr - S_gr)/(1 - S_wr) = 1 - S_gr/(1 - S_wr).
  // Gas is immobile (k_rn = 0) for Se_w >= Se_trap, i.e. S_g <= S_gr.
  // Se_trap = 1.0 recovers the original no-trapping closure.  Only k_rn is
  // remapped here (gas-only) -- k_rw and p_c keep the drainage Se_w.  The
  // returned DKNr_DSe is d k_rn / d Se_w (the 1/Se_trap chain factor is folded
  // in), so every caller's existing dkrn*dSe_dp chain stays correct unchanged.
  const double inv_trap = 1.0 / Se_trap;
  const double Se       = Se_w * inv_trap;
  // ---------------------------------------------------------------------
  // Option A regularisation (May 2026): cubic-Hermite endpoint bridge for
  // k_rn near the wet endpoint Se = 1.
  //
  // Burdine k_rn = (1 - Se)^2 * (1 - Se^((2+lambda)/lambda)) has a TRIPLE
  // zero at Se = 1 (value, 1st, and 2nd derivative all vanish).  Newton
  // cannot predict gas-mobile breakthrough from local linearisation: when
  // its trial step lands at Se = 1 - epsilon, the residual jumps and the
  // step direction was wrong.  This is the FluidFlower disk-injection
  // breakthrough stall (k_rn turning on at t ~ 47 s under benchmark
  // conditions).
  //
  // Fix: on a narrow window Se in [Se_a, 1] with Se_a = 1 - delta_brn,
  // replace BC by a cubic Hermite that matches BC value+slope at Se = Se_a
  // and ends at (k_rn = 0, dk_rn/dSe = -s_min_brn) at Se = 1.  Bulk physics
  // (Se in [0, Se_a]) is pure Brooks-Corey.
  //
  // Tuning:
  //   delta_brn = 5e-2: bridge spans 5% of the S_e range
  //   s_min_brn = 1e-3: |slope| at Se = 1; controls how much "advance
  //               warning" Newton gets about k_rn turning on.  Larger ->
  //               earlier breakthrough prediction, more distortion in the
  //               bridge.  At Se = 1-delta the bridge value is ~21% above
  //               pure BC k_rn (negligible compared to absolute magnitudes
  //               here ~1e-4).
  //
  // For Se > 1 (Newton overshoot) the value clamps to 0 but the derivative
  // continues at -s_min_brn so the chain rule has C^1 continuity across
  // the seam.
  // ---------------------------------------------------------------------
  const double delta_brn = 5.0e-2;
  const double s_min_brn = 1.0e-3;
  const double Se_a      = 1.0 - delta_brn;

  if (Se <= 0.0) {
    // S_n = 1 - S_wr (gas-saturated): non-wetting endpoint.
    KNr      = 1.0;
    DKNr_DSe = 0.0;
    return;
  }
  if (Se >= 1.0) {
    // Beyond the wet endpoint: value clamps to 0, derivative continues at
    // -s_min_brn for C^1 consistency with the bridge at Se = 1.
    KNr      = 0.0;
    DKNr_DSe = -s_min_brn * inv_trap;
    return;
  }

  const double exp_n = (2.0 + lam) / lam;

  if (Se < Se_a) {
    // Bulk Brooks-Corey (Burdine) k_rn = (1 - Se)^2 * (1 - Se^((2+lambda)/lambda))
    const double SeExp      = pow(Se, exp_n);
    const double oneMinusSe = 1.0 - Se;
    const double Y          = 1.0 - SeExp;
    KNr      = oneMinusSe * oneMinusSe * Y;
    DKNr_DSe = (-2.0 * oneMinusSe * Y
             - oneMinusSe * oneMinusSe * exp_n * pow(Se, exp_n - 1.0)) * inv_trap;
    return;
  }

  // Cubic-Hermite bridge on [Se_a, 1].
  // Anchors:
  //   left  (t=0): k_rn = fa = k_rn_BC(Se_a),  dk/dSe = da = dk_rn_BC/dSe(Se_a)
  //   right (t=1): k_rn = 0,                    dk/dSe = -s_min_brn
  // Parameterised by t = (Se - Se_a)/delta_brn in [0, 1].
  const double SeExp_a    = pow(Se_a, exp_n);
  const double oneMinusSe_a = 1.0 - Se_a;
  const double Y_a        = 1.0 - SeExp_a;
  const double fa = oneMinusSe_a * oneMinusSe_a * Y_a;
  const double da = -2.0 * oneMinusSe_a * Y_a
                  - oneMinusSe_a * oneMinusSe_a * exp_n * pow(Se_a, exp_n - 1.0);

  const double t  = (Se - Se_a) / delta_brn;
  const double t2 = t * t;
  const double t3 = t2 * t;

  // Hermite basis (only H00, H10, H11 contribute since fb = 0).
  const double H00 = 2.0 * t3 - 3.0 * t2 + 1.0;
  const double H10 = t3 - 2.0 * t2 + t;
  const double H11 = t3 - t2;
  const double dH00 = 6.0 * t2 - 6.0 * t;
  const double dH10 = 3.0 * t2 - 4.0 * t + 1.0;
  const double dH11 = 3.0 * t2 - 2.0 * t;

  KNr      = H00 * fa + H10 * delta_brn * da + H11 * delta_brn * (-s_min_brn);
  DKNr_DSe = (dH00 * fa + dH10 * delta_brn * da
              + dH11 * delta_brn * (-s_min_brn)) / delta_brn * inv_trap;
}

} // namespace psk
} // namespace m_comp_co2
} // namespace proteus

#endif
