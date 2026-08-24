#ifndef RICHARDS_PSK_MODELS_H
#define RICHARDS_PSK_MODELS_H
#include <algorithm>
#include <cmath>

// =============================================================================
// Pore size distribution / relative permeability (PSK) closures for Richards.
//
// Richards.h owns the PDE coefficient assembly (mass, diffusion tensor,
// buoyancy flux, their Jacobians); this header owns *only* the constitutive
// relations theta_w(psiC) and k_rw(psiC) plus their inverses.  Selecting a
// model is a single branch on PSK_TYPE_member at each call site:
//
//     PSK_TYPE 0 -> vgm_*   van Genuchten retention + Mualem k_rw  (default)
//     PSK_TYPE 1 -> bc_*    Brooks-Corey retention + Burdine k_rw
//     PSK_TYPE 2 -> bc_*    Brooks-Corey retention + Mualem  k_rw
//     PSK_TYPE 3 -> gardner_* Gardner exponential retention + k_rw = S_e
//
// Codes 1 and 2 share one retention curve and differ only in the exponent of
// k_rw = S_e^eta, so only the forward closure branches on them; the inverses
// (which see theta_w alone) treat them alike.  Code 3 is a model of its own in
// both halves; it is the closure Tracy's analytical solutions assume.
//
// Conventions shared by every routine here:
//   psiC is the suction (-u).  psiC > 0 is unsaturated, psiC <= 0 saturated.
//   vgm_ and bc_ clamp that saturated side to (thetaS, 1) with zero slope;
//   gardner_ deliberately does NOT, and the block below says why -- a zero
//   dtheta/dpsiC is a lost Jacobian diagonal wherever beta = 0.
//   Derivatives are with respect to psiC, NOT with respect to u; the caller in
//   Richards.h owns the sign flip d(psiC)/du = -1.
//   The vgm_, bc_ and gardner_ routines take the same parameter slots in the
//   same order, so the branch is a one-line swap.  In the BC parameterisation
//   the second numeric parameter is the pore-size index lambda (taking n_vg's
//   slot) and alpha = 1/p_d is the inverse entry-pressure head; under Gardner
//   alpha is the exponential decay rate [1/m] and the n_vg slot is unused.
//
// The Brooks-Corey conductivity exponent eta is NOT fixed by lambda.  Two
// closures are in common use and they do not agree: Burdine gives
// eta = (2+3*lambda)/lambda, Mualem gives eta = 2.5 + 2/lambda.  At the
// lambda = 0.592 sand of Szymkiewicz [2009], WRR 45, W10403 (Table 1, soil 5)
// that is 6.378 against 5.878 -- not a rounding difference: Burdine is a factor
// ~4 drier in k_rw by psiC = 7.5 m and ~7 by 50 m, which moves a gravity-driven
// wetting front.  Which one applies is a property of the parameter set being
// reproduced, so it is selected by PSK_TYPE and the exponent is derived from
// lambda here, in bc_eta, rather than computed by the caller.
// =============================================================================

namespace proteus
{
namespace richards
{
namespace psk
{

// -----------------------------------------------------------------------------
// van Genuchten - Mualem (pore-connectivity exponent l = 1/2)
//
//   S_e     = (1 + (alpha*psiC)^n)^(-m),   m = 1 - 1/n
//   theta_w = thetaR + thetaSR * S_e
//   k_rw    = sqrt(S_e) * (1 - (alpha*psiC)^(n-1) * S_e)^2
//
// pcBarStar floors alpha*psiC at 1e-8 so pow(pcBar, n-2) stays finite as
// psiC -> 0 for n < 2; sqrt_sBarStar floors the same way inside the k_rw
// derivative only.
// -----------------------------------------------------------------------------
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
    const double pcBar     = alpha * psiC;
    double       pcBarStar = pcBar;
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

    thetaW        = thetaSR * sBar + thetaR; //thetaS;//
    DthetaW_DpsiC = thetaSR * DsBar_DpsiC;   //0.0;//

    const double sqrt_sBar     = sqrt(sBar);
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

// Analytic van Genuchten inverse: theta_w -> psiC.  Leaves u untouched (so the
// caller keeps its previous iterate) outside the strictly unsaturated range.
//
// The dry limit is a cap on psiC, not a fraction of thetaR -- same convention as
// bc_invert_analytic below.  1.01*thetaR looks harmless but is a band in theta,
// and theta -> psiC is exponentially steep in the tail, so it silently swallows
// a huge band in head: for a sand with alpha=14.5 1/m and n=2.68 it refuses to
// invert anything drier than psiC = 3.8 m, which for a 20 m column is the whole
// unwetted region.  Every FCT-limited mass landing there was discarded (u kept
// its pre-limiter value), which breaks the conservation chain theta_limited ->
// psi and shows up as spurious infiltration.  pcBar <= 1e4 puts the cut at
// psiC = 1e4/alpha instead, i.e. far outside any physical range.
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
  const double pcBarMax = 1.0e4;
  const double SeMin    = pow(1.0 + pow(pcBarMax, n_vg), -m_vg);
  if (thetaW > thetaR + SeMin * thetaSR && thetaW < thetaS) {
    sBar    = (thetaW - thetaR) / thetaSR;
    pcBar_n = pow(sBar, -1.0 / m_vg) - 1.0;
    pcBar   = pow(pcBar_n, 1.0 / n_vg);
    psiC    = pcBar / alpha;
    u       = -psiC;
  }
}

// Newton inverse of the FULL forward mass m = rho(u) * theta_w(u), so the
// exp(beta*u) factor the analytic inverse ignores is included.  The analytic
// inverse seeds the iteration.
//
// The step cap is one capillary length, not an absolute 5 cm.  dtheta/du spans
// ~4 decades over the retention curve, so a fixed step cap cannot reach the
// root at either end: with duMax = 5e-2 the 50 iterations here covered only
// 2.5 m of head, and any correction needing more than that fell through to the
// u = u_prev revert at the bottom of the loop.
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
  if (psiC0 <= 0.0) { return; } //saturated, no inversion
  const double rhom0 = rho * std::exp(beta * u);//first guess
  const double thetaW_imp = m / rhom0;
  // Dry limit stated the same way as in vgm_invert_analytic: a cap on psiC, not
  // a fraction of thetaR.  Same reason -- theta -> psiC is exponentially steep
  // in the tail, so 1.01*thetaR is a narrow band in theta but a huge one in
  // head (psiC = 3.8 m for alpha = 14.5 1/m, n = 2.68), and every FCT-limited
  // mass landing in it was silently discarded, breaking theta_limited -> psi.
  // The analytic seed below is exact at beta = 0 and lands within a couple of
  // Newton steps otherwise, so widening the range does not lengthen the solve.
  const double pcBarMax = 1.0e4;
  const double SeMin    = std::pow(1.0 + std::pow(pcBarMax, n_vg), -m_vg);
  if (thetaW_imp < thetaR + SeMin * thetaSR) { return; } //below the dry cut
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
  const double tol    = 1e-12 * std::max(1.0, std::fabs(m));
  const double duMax  = 1.0 / alpha;

  auto theta_and_dtheta_du = [&](double u,
                                 double &thetaW,
                                 double &dtheta_du) -> bool
  {
    const double psiC = -u;

    if (psiC <= 0.0) { return false; } //no inversion in saturation
    //van Genuchten relations
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
    if (std::fabs(gp) < gpTol) { u = u_prev; return; }

    double du = -g / gp;
    if (du >  duMax) du =  duMax;
    if (du < -duMax) du = -duMax;

    u += du;
    if (-u <= 0.0) { u = u_prev; return; }
  }
  u = u_prev;
}

// -----------------------------------------------------------------------------
// Brooks-Corey
//
//   S_e     = (alpha*psiC)^(-lambda)          for alpha*psiC >= 1
//   S_e     = 1                               for alpha*psiC <  1  (saturated)
//   theta_w = thetaR + thetaSR * S_e
//   k_rw    = S_e^eta,  eta = bc_eta(lambda, kr_model)
//   psiC(S_e) = S_e^(-1/lambda) / alpha       (analytic inverse)
//
// BC has a derivative discontinuity in (theta_w, k_rw) at the entry pressure
// alpha*psiC = 1.  Newton typically needs a smoothed regularisation there in
// production; that is deliberately not added at this layer - callers can wrap
// these closures with their own smoothing if needed.
// -----------------------------------------------------------------------------

// Which k_rw closure supplies the Brooks-Corey exponent.  Maps onto PSK_TYPE:
// 1 -> burdine, 2 -> mualem.  burdine is first so it is the zero value and
// stays the default, matching what this header did before the choice existed.
enum class bc_kr { burdine = 0, mualem = 1 };

// eta(lambda) for the two closures.  Kept as a named function rather than
// inlined into bc_wetting so the two formulas sit side by side and the caller
// never has to restate either one.
inline double bc_eta(const double lam, const bc_kr kr_model)
{
  return (kr_model == bc_kr::mualem) ? (2.5 + 2.0 / lam)
                                     : ((2.0 + 3.0 * lam) / lam);
}

inline void bc_wetting(const double psiC,
                       const double alpha,
                       const double lam,
                       const double thetaR,
                       const double thetaSR,
                       double &thetaW,
                       double &DthetaW_DpsiC,
                       double &KWr,
                       double &DKWr_DpsiC,
                       const bc_kr kr_model = bc_kr::burdine)
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
  // k_rw = Se^eta, exponent set by the selected closure
  const double exp_w = bc_eta(lam, kr_model);
  KWr        = pow(Se, exp_w);
  DKWr_DpsiC = exp_w * pow(Se, exp_w - 1.0) * dSe_DpsiC;
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
  // Dry limit of the inversion, expressed the same way as in bc_invert_newton:
  // a multiple of the entry pressure, not a fraction of thetaR.  See the note
  // there for why 1.01*thetaR cannot be used with the BC tail.
  const double psiCMax_over_pd = 1.0e4;
  const double SeMin = pow(psiCMax_over_pd, -lam);
  if (thetaW > thetaR + SeMin * thetaSR && thetaW < thetaS) {
    const double Se = (thetaW - thetaR) / thetaSR;
    if (Se > 0.0 && Se < 1.0) {
      const double pcBar = pow(Se, -1.0 / lam);
      const double psiC  = pcBar / alpha;
      u = -psiC;
    }
  }
}

// Inverse of the FULL forward mass m = rho(u)*theta_w(u) against the BC
// retention curve.  Solved as a bracketed (safeguarded) Newton rather than the
// plain Newton vgm_invert_newton uses, for three reasons specific to BC:
//
//  * Dead band.  Both routines cut the dry tail at a multiple of the entry
//    pressure rather than at a fraction of thetaR (see vgm_invert_newton for
//    why a band in theta is the wrong variable to state it in); BC needs that
//    the more, since its Se ~ psiC^-lambda tail is far fatter than van
//    Genuchten's.  With thetaR = 0.05, thetaSR = 0.35, p_d = 0.5 m, lambda = 2
//    a 1.01*thetaR band would fire for every node drier than psiC ~ 13 m, where
//    under VG (n = 1.8) it does not fire until psiC ~ 1800 m.  Here the limit is
//    enforced exactly rather than through a proxy in theta: if g at that head
//    has not yet changed sign the root is drier than the band and u is left
//    alone.
//
//  * Entry-pressure wall.  theta is flat (== thetaS) for psiC <= p_d, so the
//    root can lie inside a region the retention curve cannot resolve.  That is
//    detected up front (g(u_wall) <= 0) and answered with the entry pressure
//    itself -- the driest head consistent with full saturation, and the
//    correct inverse of the mass handed in -- rather than by discarding the
//    solve and reverting to a stale iterate carrying an unrelated theta.
//
//  * Conditioning.  In the dry tail the compressibility term beta*theta
//    dominates thetaSR*dSe/du (at Se = 1e-6, beta = 1e-5 it is ~350x larger),
//    so the root is set almost entirely by rho(u).  Any seed that evaluates
//    exp(beta*u) at a stale iterate is then wrong by orders of magnitude in
//    Se, and unguarded Newton walks off.  Bracketing removes the dependence on
//    the seed: g is monotone increasing in u, so [u_floor, u_wall] brackets the
//    root by construction and every iterate stays inside it.
inline void bc_invert_newton(const double m,
                             const double rho,
                             const double beta,
                             const double alpha,
                             const double lam,
                             const double thetaR,
                             const double thetaSR,
                             double &u)
{
  if (-u <= 0.0) { return; } //ponded / saturated head: no inversion

  // g(u) = rho(u)*theta_w(u) - m, evaluated with the BC closure (theta_w
  // clamped to thetaS inside the entry pressure).  Monotone increasing in u.
  auto g_of = [&](const double uu) -> double {
    const double pcBar = alpha * (-uu);
    const double Se    = (pcBar <= 1.0) ? 1.0 : pow(pcBar, -lam);
    return rho * std::exp(beta * uu) * (thetaR + thetaSR * Se) - m;
  };

  // Upper end of the bracket: one part in 1e9 outside the entry pressure, so
  // the BC derivative (which drops to zero across it) stays defined.
  const double u_entry = -1.0 / alpha;
  const double u_wall  = -(1.0 + 1.0e-9) / alpha;
  if (g_of(u_wall) <= 0.0) { u = u_entry; return; } //root at or inside the wall

  // Lower end: the driest head the inversion is defined down to, stated as a
  // multiple of the entry pressure.  A floor on Se instead would sit at
  // psiC = p_d*Se^(-1/lambda) and so swing with lambda -- Se = 1e-6 is
  // psiC = 1e3*p_d at lambda = 2 but 1e12*p_d at lambda = 0.5, far outside any
  // head the solver will ever see and wide enough to make the bracket useless.
  const double psiCMax_over_pd = 1.0e4;
  const double u_floor = -psiCMax_over_pd / alpha;
  if (g_of(u_floor) > 0.0) { return; } //root drier than the band: leave u alone

  double lo = u_floor, hi = u_wall; // g(lo) <= 0 < g(hi)

  // Analytic BC inverse (density-free) as the opening guess; the bracket
  // catches it if the neglected exp(beta*u) puts it in the wrong place.
  {
    const double Se = (m / rho - thetaR) / thetaSR;
    if (Se > 0.0 && Se < 1.0) {
      const double u_guess = -pow(Se, -1.0 / lam) / alpha;
      if (std::isfinite(u_guess) && u_guess > lo && u_guess < hi) u = u_guess;
      else u = 0.5 * (lo + hi);
    } else u = 0.5 * (lo + hi);
  }
  if (!(u > lo && u < hi)) u = 0.5 * (lo + hi);

  // maxIts covers the bisection worst case over the widest bracket; Newton
  // reaches the root in a handful of steps whenever the guess is sane.
  const int    maxIts = 100;
  const double tol    = 1e-12 * std::max(1.0, std::fabs(m));

  for (int it = 0; it < maxIts; ++it) {
    // u is strictly inside (u_floor, u_wall), so pcBar > 1 and the BC
    // derivative below is always the unsaturated-branch one.
    const double pcBar     = alpha * (-u);
    const double Se        = pow(pcBar, -lam);
    const double dSe_DpsiC = -lam * alpha * pow(pcBar, -lam - 1.0);
    const double thetaW    = thetaR + thetaSR * Se;
    const double dtheta_du = -thetaSR * dSe_DpsiC; // d(psiC)/d(u) = -1

    const double rhom = rho * std::exp(beta * u);
    const double g    = rhom * thetaW - m;
    if (std::fabs(g) < tol) return;
    if (g > 0.0) hi = u; else lo = u;

    const double gp    = rhom * (beta * thetaW + dtheta_du);
    const double gpTol = 1e-14 * std::max(1.0, std::fabs(rhom * thetaW));
    double u_next = (std::fabs(gp) > gpTol) ? (u - g / gp) : 0.5 * (lo + hi);
    // Bisect whenever Newton leaves the bracket.
    if (!(u_next > lo && u_next < hi)) u_next = 0.5 * (lo + hi);

    if (std::fabs(u_next - u) <= 1e-15 * std::fabs(u)) return;
    u = u_next;
  }
  // Bracketed throughout, so the last iterate is the best available answer.
}

// -----------------------------------------------------------------------------
// Gardner (quasi-linear) retention + conductivity, with the Irmay-style linear
// theta_w(k_rw) pairing:
//
//   S_e     = exp(-alpha*psiC) = exp(alpha*u)
//   theta_w = thetaR + thetaSR * S_e
//   k_rw    = S_e                       i.e. K(psi) = Ks*exp(alpha*psi)
//   psiC(S_e) = -ln(S_e)/alpha          (analytic inverse, exact)
//
// alpha here is Gardner's exponent [1/m]; it takes the same parameter slot as
// van Genuchten's alpha and the Brooks-Corey inverse entry pressure, and the
// second numeric parameter (n_vg / lambda) is unused.
//
// This is the pairing Tracy's analytical solutions are built on, and both
// halves of it matter: with k_rw = S_e and theta_w affine in S_e, the Kirchhoff
// transform hbar = exp(alpha*psi) turns Richards into a linear advection-
// diffusion equation for hbar, which is what makes the closed-form steady and
// transient solutions exist.  Substituting Mualem k_rw here would keep the
// retention curve but destroy the linearisation, so the two are one model, not
// two independent choices.
//
// There is no entry pressure and no dry-end kink: theta_w and k_rw are C-inf
// for psiC > 0 and both approach saturation smoothly as psiC -> 0.  What it
// does have is an unbounded dry tail -- S_e -> 0 only as psiC -> inf -- so the
// inversions cut at alpha*psiC = 7e2, stated as a cap on head like the vgm_/bc_
// routines rather than as a band in theta.  That is where exp() underflows,
// i.e. as far out as double precision can represent the curve at all, and
// orders of magnitude beyond any head a solve will see.
//
// UNLIKE vgm_ and bc_, there is NO saturated branch: the exponential is
// continued through psiC = 0 into psiC < 0 rather than clamped to
// (theta_w, k_rw) = (thetaS, 1).  That is deliberate and it is not cosmetic.
// Clamping sets DthetaW_DpsiC = 0, hence dm = drhom*thetaW, which at beta = 0
// is exactly zero -- and the low-order diagonal in Richards.h is
//
//     globalJacobian[ii] += bc_mask[i]*(MLi*dm/dt + J_ii) + (1-bc_mask[i])
//
// where MLi*dm/dt outweighs the graph term J_ii by ~3 orders at a small dt.  A
// free DOF that reaches psi >= 0 therefore keeps its residual row but loses its
// diagonal, takes a correction ~3 orders too large, falls back below zero where
// dm > 0 again, and overshoots once more: Newton locks into an exact period-2
// orbit that no tolerance or step cap will break.  vgm_/bc_ carry the same
// clamp harmlessly because the only node pinned at psi = 0 is normally a
// Dirichlet node, which bc_mask = 0 replaces with an identity row; Gardner
// reaches psi = 0 at *free* nodes because its diffusivity is
// Ks/(alpha*thetaSR), constant over the whole curve and typically orders above
// van Genuchten's in the unsaturated range, so a sharp boundary layer drives
// interior nodes into saturation within one step.
//
// The continuation is the same exponential, so the closure stays C-inf and
// every derivative used to build the diagonal stays strictly positive.  It does
// mean k_rw > 1 and theta_w > thetaS for psi > 0, which is unphysical -- but
// Gardner is a psi <= 0 model with no saturated branch to speak of, and these
// values are only ever visited by transient iterates on the way back down.  Any
// case that genuinely ponds wants vgm_ or bc_, which cap properly.
// -----------------------------------------------------------------------------

// Dry-end cut for the Gardner inversions, in units of alpha*psiC.  exp(-7e2) is
// ~1e-304, one decade off denormal, so this is the widest cut that keeps S_e a
// normal double.
constexpr double gardner_alphaPsiCMax = 7.0e2;

// Wet-end rail on the SAME exponent, so exp(alpha*u) cannot overflow if an
// iterate runs away.  This is an arithmetic guard, not a saturation limit: at a
// typical alpha it sits hundreds of metres above ground, far outside any head a
// solve can reach without having already failed, and it is placed on the
// exponent (not on theta) so it can never fire near psi = 0 where the clamp
// above would cost the diagonal.
constexpr double gardner_alphaPsiMax = 5.0e1;

inline void gardner_wetting(const double psiC,
                            const double alpha,
                            const double n_vg,
                            const double thetaR,
                            const double thetaSR,
                            double &thetaW,
                            double &DthetaW_DpsiC,
                            double &KWr,
                            double &DKWr_DpsiC)
{
  (void)n_vg; // Gardner is a one-parameter curve
  // One branch for every psiC.  exp(-alpha*psiC) underflows to 0 past the dry
  // cut (theta_w -> thetaR, k_rw -> 0 with zero slope, all correct); the wet
  // rail below only bounds a runaway iterate.
  const double x  = -alpha * psiC;
  const double Se = std::exp(x < gardner_alphaPsiMax ? x : gardner_alphaPsiMax);
  const double dSe_DpsiC = (x < gardner_alphaPsiMax) ? (-alpha * Se) : 0.0;
  thetaW        = thetaR + thetaSR * Se;
  DthetaW_DpsiC = thetaSR * dSe_DpsiC;
  KWr           = Se;
  DKWr_DpsiC    = dSe_DpsiC;
}

// Analytic Gardner inverse: theta_w -> psiC.  Exact (the retention curve is a
// plain exponential), so this is the whole inversion whenever beta = 0.
//
// The admissible band is the forward curve's own range, dry cut to wet rail --
// NOT thetaW < thetaS.  Since gardner_wetting continues the exponential above
// psi = 0, theta_w > thetaS is a value the forward model genuinely produces and
// the inverse has to be able to return the u > 0 that generated it; cutting at
// thetaS would leave those DOFs holding a stale iterate whose theta is
// unrelated to the mass handed in, which is the conservation break the vgm_/bc_
// dry-cut comments describe, at the other end of the curve.
inline void gardner_invert_analytic(const double m,
                                    const double rho,
                                    const double alpha,
                                    const double n_vg,
                                    const double thetaR,
                                    const double thetaSR,
                                    double &u)
{
  (void)n_vg;
  const double thetaW = m / rho;
  const double SeMin  = std::exp(-gardner_alphaPsiCMax);
  const double SeMax  = std::exp(gardner_alphaPsiMax);
  if (thetaW > thetaR + SeMin * thetaSR && thetaW < thetaR + SeMax * thetaSR) {
    const double Se = (thetaW - thetaR) / thetaSR;
    const double u_new = std::log(Se) / alpha; // = -psiC; sign follows Se vs 1
    if (std::isfinite(u_new)) u = u_new;
  }
}

// Inverse of the FULL forward mass m = rho(u)*theta_w(u) against the Gardner
// curve, i.e. including the exp(beta*u) the analytic inverse drops.
//
// Bracketed Newton, like bc_invert_newton and for the same conditioning
// reason: g(u) = rho*exp(beta*u)*(thetaR + thetaSR*exp(alpha*u)) - m is a sum
// of increasing exponentials, so it is monotone in u and [u_floor, u_rail]
// brackets the root by construction.  In the dry tail the beta*theta term
// dominates thetaSR*alpha*Se, so a plain Newton seeded from a stale iterate can
// walk off exactly as it does under BC; bracketing removes the dependence on
// the seed.
//
// Both ends of the bracket are the forward curve's own limits.  There is no
// wall at u = 0: gardner_wetting has no saturated branch, so psi > 0 is inside
// the model here and a mass above rho*thetaS inverts to the positive head that
// produced it rather than being flattened onto zero.  Nothing is refused for
// being "ponded" either -- an iterate that has overshot into psi > 0 is exactly
// the state the inversion has to be able to walk back down.
inline void gardner_invert_newton(const double m,
                                  const double rho,
                                  const double beta,
                                  const double alpha,
                                  const double n_vg,
                                  const double thetaR,
                                  const double thetaSR,
                                  double &u)
{
  (void)n_vg;

  auto g_of = [&](const double uu) -> double {
    const double x  = alpha * uu;
    const double Se = std::exp(x < gardner_alphaPsiMax ? x : gardner_alphaPsiMax);
    return rho * std::exp(beta * uu) * (thetaR + thetaSR * Se) - m;
  };

  const double u_rail = gardner_alphaPsiMax / alpha;
  if (g_of(u_rail) <= 0.0) { u = u_rail; return; } //root at or above the rail

  const double u_floor = -gardner_alphaPsiCMax / alpha;
  if (g_of(u_floor) > 0.0) { return; } //root drier than the band: leave u alone

  double lo = u_floor, hi = u_rail; // g(lo) <= 0 < g(hi)

  // Density-free analytic inverse as the opening guess; the bracket catches it
  // if the neglected exp(beta*u) puts it in the wrong place.
  {
    const double Se = (m / rho - thetaR) / thetaSR;
    double u_guess = 0.5 * (lo + hi);
    if (Se > 0.0) { // Se >= 1 is admissible now: it is the psi > 0 continuation
      const double u_analytic = std::log(Se) / alpha;
      if (std::isfinite(u_analytic) && u_analytic > lo && u_analytic < hi)
        u_guess = u_analytic;
    }
    u = u_guess;
  }
  if (!(u > lo && u < hi)) u = 0.5 * (lo + hi);

  // maxIts covers the bisection worst case over the widest bracket; from the
  // analytic seed Newton lands in a couple of steps for any sane beta.
  const int    maxIts = 100;
  const double tol    = 1e-12 * std::max(1.0, std::fabs(m));

  for (int it = 0; it < maxIts; ++it) {
    const double Se        = std::exp(alpha * u);
    const double thetaW    = thetaR + thetaSR * Se;
    const double dtheta_du = thetaSR * alpha * Se;

    const double rhom = rho * std::exp(beta * u);
    const double g    = rhom * thetaW - m;
    if (std::fabs(g) < tol) return;
    if (g > 0.0) hi = u; else lo = u;

    const double gp    = rhom * (beta * thetaW + dtheta_du);
    const double gpTol = 1e-14 * std::max(1.0, std::fabs(rhom * thetaW));
    double u_next = (std::fabs(gp) > gpTol) ? (u - g / gp) : 0.5 * (lo + hi);
    // Bisect whenever Newton leaves the bracket.
    if (!(u_next > lo && u_next < hi)) u_next = 0.5 * (lo + hi);

    if (std::fabs(u_next - u) <= 1e-15 * std::fabs(u)) return;
    u = u_next;
  }
  // Bracketed throughout, so the last iterate is the best available answer.
}

} // namespace psk
} // namespace richards
} // namespace proteus

#endif
