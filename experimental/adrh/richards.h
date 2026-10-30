// Richards' equation in pressure-head form, the coefficients of proteus's richards model (proteus/richards/Richards.h,
// evaluateCoefficients) in the engine's Terms, so that the engine and the model can be compared on the same problem.
//
//     m(psi)_t + div(f(psi) - a(psi) grad psi) = 0,     psi the pressure head,
//     m = rho theta_w(psi),   a = rho K_r(psi) K_s,   f = (rho^2 / rho0) K_r(psi) K_s g
//
// with the Mualem-van Genuchten retention and relative conductivity (vgmWetting, a port of vgm_wetting in
// proteus/pskRelations.h with its regularizations: capillary pressure clipped at 1e-8 below, sqrt(S_e) clipped at 1e-8 in
// the derivative), rho = rho_transport exp(beta psi) the compressibility factor (beta = 0 in the benchmark decks), K_s a
// diagonal conductivity and g the gravity direction. Pressure head is positive when saturated: psiC = -psi > 0 is unsaturated.
#pragma once
#include <cmath>
#include "coefficients.h"

namespace adrh
{
  ADRH_FN void vgmWetting(double psiC, double alpha, double n_vg, double thetaR, double thetaSR, double &thetaW,
                                 double &DthetaW_DpsiC, double &KWr, double &DKWr_DpsiC)
  {
    const double m_vg = 1.0 - 1.0 / n_vg;
    const double thetaS = thetaR + thetaSR;
    if (psiC > 0.0)
    {
      const double pcBar = alpha * psiC;
      double pcBarStar = pcBar;
      if (pcBar < 1.0e-8) pcBarStar = 1.0e-8;
      const double pcBar_nM2 = std::pow(pcBarStar, n_vg - 2);
      const double pcBar_nM1 = pcBar_nM2 * pcBar;
      const double pcBar_n = pcBar_nM1 * pcBar;
      const double onePlus_pcBar_n = 1.0 + pcBar_n;
      const double sBar = std::pow(onePlus_pcBar_n, -m_vg);
      const double DsBar_DpsiC = alpha * (1.0 - n_vg) * (sBar / onePlus_pcBar_n) * pcBar_nM1;   // using -mn = 1-n
      const double vBar = 1.0 - pcBar_nM1 * sBar;
      const double vBar2 = vBar * vBar;
      const double DvBar_DpsiC = -alpha * (n_vg - 1.0) * pcBar_nM2 * sBar - pcBar_nM1 * DsBar_DpsiC;
      thetaW = thetaSR * sBar + thetaR;
      DthetaW_DpsiC = thetaSR * DsBar_DpsiC;
      const double sqrt_sBar = std::sqrt(sBar);
      double sqrt_sBarStar = sqrt_sBar;
      if (sqrt_sBar < 1.0e-8) sqrt_sBarStar = 1.0e-8;
      KWr = sqrt_sBar * vBar2;
      DKWr_DpsiC = (0.5 / sqrt_sBarStar) * DsBar_DpsiC * vBar2 + 2.0 * sqrt_sBar * vBar * DvBar_DpsiC;
    }
    else
    {
      thetaW = thetaS;
      DthetaW_DpsiC = 0.0;
      KWr = 1.0;
      DKWr_DpsiC = 0.0;
    }
  }

  // Brooks-Corey retention with the Mualem (psk 2) or Burdine (psk 1) conductivity, a port of bc_wetting in proteus/pskRelations.h:
  //   S_e = (alpha psiC)^(-lambda) for alpha psiC > 1, 1 inside the entry pressure; k_rw = S_e^eta, eta = 2.5 + 2/lambda (Mualem) or
  //   (2 + 3 lambda)/lambda (Burdine). alpha is the inverse entry-pressure head and lam takes the n_vg slot, as in proteus. The curve has a
  //   derivative discontinuity at the entry pressure that proteus does not smooth either.
  ADRH_FN void bcWetting(double psiC, double alpha, double lam, double thetaR, double thetaSR, int psk, double &thetaW, double &DthetaW_DpsiC,
                         double &KWr, double &DKWr_DpsiC)
  {
    const double thetaS = thetaR + thetaSR;
    const double pcBar = alpha * psiC;
    if (psiC <= 0.0 || pcBar <= 1.0)
    {
      thetaW = thetaS; DthetaW_DpsiC = 0.0; KWr = 1.0; DKWr_DpsiC = 0.0;
      return;
    }
    const double Se = std::pow(pcBar, -lam);
    const double dSe = -lam * alpha * std::pow(pcBar, -lam - 1.0);
    thetaW = thetaR + thetaSR * Se;
    DthetaW_DpsiC = thetaSR * dSe;
    const double eta = (psk == 2) ? (2.5 + 2.0 / lam) : ((2.0 + 3.0 * lam) / lam);
    KWr = std::pow(Se, eta);
    DKWr_DpsiC = eta * std::pow(Se, eta - 1.0) * dSe;
  }

  // The relative permeability of a node (the d/du derivative, u the pressure head) for the low-order operator that takes it at the upwind node.
  ADRH_FN void krNode(double u, double alpha, double n, double thetaR, double thetaSR, int psk, double &kr, double &dkr_du)
  {
    double thetaW, DthetaW, KWr, DKWr;
    if (psk == 1 || psk == 2) bcWetting(-u, alpha, n, thetaR, thetaSR, psk, thetaW, DthetaW, KWr, DKWr);
    else vgmWetting(-u, alpha, n, thetaR, thetaSR, thetaW, DthetaW, KWr, DKWr);
    kr = KWr;
    dkr_du = -DKWr;
  }

  // The storage m = rho theta_w(psi) of a node (and dm/du, u the pressure head) with the node's own retention curve: the lumped mass of the
  // low-order scheme is nodal, so a node on a soil interface has one theta(psi), the first incident element's (as proteus's nodeMaterialTypes_n).
  ADRH_FN void mNode(double u, double alpha, double n, double thetaR, double thetaSR, int psk, double rho, double beta, double &m, double &dm_du)
  {
    double thetaW, DthetaW, KWr, DKWr;
    if (psk == 1 || psk == 2) bcWetting(-u, alpha, n, thetaR, thetaSR, psk, thetaW, DthetaW, KWr, DKWr);
    else vgmWetting(-u, alpha, n, thetaR, thetaSR, thetaW, DthetaW, KWr, DKWr);
    const double rhom = rho * std::exp(beta * u), drhom = beta * rhom;
    m = rhom * thetaW;
    dm_du = -rhom * DthetaW + drhom * thetaW;
  }

  template <int nS_>
  struct RichardsVGM : DenseMasks
  {
    static constexpr int nc = 1, nSpace = nS_;
    static constexpr bool has_hamiltonian = false, has_reaction = false;
    double alpha, n_vg, thetaR, thetaSR, Ks[3], g[3], rho, rho0, beta;
    int mode;    // 0: the equation; 1: the stiffness K_s alone (relative permeability 1); 2: the stiffness with the relative permeability at the
                 // quadrature points frozen (no storage, no gravity, no derivative of K_r): the matrices of the K-upwinded low-order scheme
    int psk;     // retention and conductivity model, proteus's PSK_type: 0 van Genuchten-Mualem, 1 Brooks-Corey-Burdine, 2 Brooks-Corey-Mualem
    // Ks: the diagonal of the conductivity (nS_ entries); g: gravity (nS_ entries, as proteus's dimensionless_gravity)
    ADRH_FN RichardsVGM(double alpha_, double n_, double thetaR_, double thetaSR_, const double *Ks_, const double *g_, double rho_ = 1.0,
                        double beta_ = 0.0, double rho0_ = 1.0, int psk_ = 0, int mode_ = 0)
        : alpha(alpha_), n_vg(n_), thetaR(thetaR_), thetaSR(thetaSR_), rho(rho_), rho0(rho0_), beta(beta_), mode(mode_), psk(psk_)
    {
      for (int I = 0; I < 3; ++I) { Ks[I] = I < nS_ ? Ks_[I] : 0.0; g[I] = I < nS_ ? g_[I] : 0.0; }
    }
    ADRH_FN void evaluate(double, const double[3], const double u[1], const double[1][nS_], Terms<1, nS_> &T) const
    {
      double thetaW, DthetaW, KWr, DKWr;
      if (psk == 1 || psk == 2) bcWetting(-u[0], alpha, n_vg, thetaR, thetaSR, psk, thetaW, DthetaW, KWr, DKWr);
      else vgmWetting(-u[0], alpha, n_vg, thetaR, thetaSR, thetaW, DthetaW, KWr, DKWr);
      const double rhom = rho * std::exp(beta * u[0]), drhom = beta * rhom, rr = rhom / rho0;
      T.m[0] = rhom * thetaW;
      T.dm[0][0] = -rhom * DthetaW + drhom * thetaW;
      T.phi[0] = u[0];
      T.dphi[0][0] = 1.0;
      for (int I = 0; I < nS_; ++I)
      {
        T.a[0][0][I][I] = rhom * KWr * Ks[I];
        T.da[0][0][0][I][I] = (drhom * KWr - rhom * DKWr) * Ks[I];
        T.f[0][I] = rhom * rr * KWr * Ks[I] * g[I];
        T.df[0][0][I] = (drhom * rr * KWr + rhom * (drhom / rho0) * KWr - rhom * rr * DKWr) * Ks[I] * g[I];
      }
      if (mode != 0)
      {
        T.m[0] = 0.0; T.dm[0][0] = 0.0;
        for (int I = 0; I < nS_; ++I)
        {
          T.f[0][I] = 0.0; T.df[0][0][I] = 0.0; T.da[0][0][0][I][I] = 0.0;
          if (mode == 1) T.a[0][0][I][I] = rhom * Ks[I];
        }
      }
    }
  };

  // A column or block of several materials: up to 8 van Genuchten parameter sets and a material number per element, as proteus's
  // media types (Coefficients(nd, KsTypes, nVGtypes, alphaVGtypes, thetaRtypes, thetaSRtypes, ...), elementMaterialTypes). The
  // kernels call bindElement(eN) and get the homogeneous functor of that element's material.
  template <int nS_>
  struct RichardsMaterials : DenseMasks
  {
    static constexpr int nc = 1, nSpace = nS_, maxMaterials = 8;
    static constexpr bool has_hamiltonian = false, has_reaction = false;
    int nMaterials;
    double alpha[maxMaterials], n_vg[maxMaterials], thetaR[maxMaterials], thetaSR[maxMaterials], Ks[maxMaterials][3], g[3], rho, rho0, beta;
    int psk, mode;           // PSK_type and mode of RichardsVGM, the same for every material
    const int *material;     // (nE) the material number of each element, valid where the kernel runs; null: element number modulo the materials
    ADRH_FN RichardsMaterials(int nMaterials_, const double *alpha_, const double *n_, const double *thetaR_, const double *thetaSR_,
                              const double *Ks_, const double *g_, const int *material_, double rho_ = 1.0, double beta_ = 0.0,
                              double rho0_ = 1.0, int psk_ = 0, int mode_ = 0)
        : nMaterials(nMaterials_), rho(rho_), rho0(rho0_), beta(beta_), psk(psk_), mode(mode_), material(material_)
    {
      for (int m = 0; m < maxMaterials; ++m)
      {
        const int s = m < nMaterials_ ? m : 0;
        alpha[m] = alpha_[s]; n_vg[m] = n_[s]; thetaR[m] = thetaR_[s]; thetaSR[m] = thetaSR_[s];
        for (int I = 0; I < 3; ++I) Ks[m][I] = I < nS_ ? Ks_[s * nS_ + I] : 0.0;
      }
      for (int I = 0; I < 3; ++I) g[I] = I < nS_ ? g_[I] : 0.0;
    }
    ADRH_FN RichardsVGM<nS_> bindElement(int eN) const
    {
      const int m = material ? material[eN] : eN % nMaterials;     // no table: stripes, materials in turn by element number (benchmarks)
      return RichardsVGM<nS_>(alpha[m], n_vg[m], thetaR[m], thetaSR[m], Ks[m], g, rho, beta, rho0, psk, mode);
    }
    // used only if something calls the functor without binding an element: the first material
    ADRH_FN void evaluate(double t, const double x[3], const double u[1], const double gu[1][nS_], Terms<1, nS_> &T) const
    {
      RichardsVGM<nS_>(alpha[0], n_vg[0], thetaR[0], thetaSR[0], Ks[0], g, rho, beta, rho0, psk, mode).evaluate(t, x, u, gu, T);
    }
  };
} // namespace adrh
