#ifndef MPHASE_CO2_H
#define MPHASE_CO2_H
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <valarray>
#include "CompKernel.h"
#include "ModelFactory.h"
#include "../mprans/ArgumentsDict.h"
#include "xtensor-python/pyarray.hpp"
#include "psk_models.h"
#define nnz nSpace

namespace py = pybind11;
#define POWER_SMOOTHNESS_INDICATOR 2
#define IS_BETAij_ONE              0
#define GLOBAL_FCT                 0
namespace proteus
{
enum class STABILIZATION : int {
  Galerkin         = 0,
  EV_Stab          = 1,
  EntropyViscosity = 2,
  Implicit_FCT     = 3
};

namespace mphase_co2
{
//cek todo: revisit entry for mass transport form
// Power entropy //
inline double ENTROPY(const double &phi, const double &phiL, const double &phiR)
{
  return 1. / 2. * std::pow(fabs(phi), 2.);
}
inline double DENTROPY(const double &phi, const double &phiL, const double &phiR)
{
  return fabs(phi) * (phi >= 0 ? 1 : -1);
}
// Log entropy // for level set from 0 to 1
inline double ENTROPY_LOG(const double &phi, const double &phiL, const double &phiR)
{
  return std::log(fabs((phi - phiL) * (phiR - phi)) + 1E-14);
}
inline double DENTROPY_LOG(const double &phi, const double &phiL, const double &phiR)
{
  return (phiL + phiR - 2 * phi) * ((phi - phiL) * (phiR - phi) >= 0 ? 1 : -1) / (fabs((phi - phiL) * (phiR - phi)) + 1E-14);
}
} // namespace mphase_co2
} // namespace proteus
namespace proteus
{
namespace mphase_co2
{
class Mphase_co2_base {
  //The base class defining the interface
public:
  virtual ~Mphase_co2_base() { double anb_seepage_flux = 1e-16; }
  virtual void calculateResidual(arguments_dict &args)                   = 0;
  virtual void calculateJacobian(arguments_dict &args)                   = 0;
  virtual void invert(arguments_dict &args)                              = 0;
  virtual void FCTStep(arguments_dict &args)                             = 0;
  virtual void FCTStep_n(arguments_dict &args)                           = 0;
  virtual void kth_FCT_step(arguments_dict &args)                        = 0;
  virtual void calculateResidual_entropy_viscosity(arguments_dict &args) = 0;
  virtual void calculateMassMatrix(arguments_dict &args)                 = 0;
};

template <class CompKernelType, int nSpace, int nQuadraturePoints_element, int nDOF_mesh_trial_element, int nDOF_trial_element, int nDOF_test_element, int nQuadraturePoints_elementBoundary>
class Mphase_co2 : public Mphase_co2_base {
public:
  const int      nDOF_test_X_trial_element;
  CompKernelType ck;
  // Per-DOF density projected from q_rho in calculateResidual_entropy_viscosity.
  // Reused by invert() so the m -> u inversion uses the same variable density
  // that built the forward mass.
  std::vector<double> rho_dof_member;
  // Per-DOF phi * rho_n cached from calculateResidual_entropy_viscosity /
  // calculateMassMatrix. Reused by invert(COMPONENT=1) so the m_n -> u_n
  // inversion uses the same projected density-porosity product that built
  // the forward comp-1 mass.
  std::vector<double> rho_n_phi_dof_member;
  // PSK closure selector: 0 = VGM (van Genuchten-Mualem), 1 = BC (Brooks-Corey-Burdine).
  // Set by every top-level entry point (calculateResidual / _entropy_viscosity /
  // calculateJacobian / calculateMassMatrix) from argsDict before any
  // evaluateCoefficients call. evaluateCoefficients dispatches on this.
  int PSK_TYPE_member = 0;
  Mphase_co2() : nDOF_test_X_trial_element(nDOF_test_element * nDOF_trial_element), ck() { }
  // Wetting-equation coefficients in pressure / S_n form.
  // Primary variables:
  //   u_w = p_w        (wetting-phase pressure, Pa)
  //   u_n = S_n = 1 - S_w   (non-wetting saturation, in [0, 1 - S_wr])
  // beta is the wetting-fluid compressibility in 1/Pa. KWs is the wetting-phase
  // mobility tensor K/mu_w (units 1/(Pa*s)) -- the caller now passes K/mu_w
  // rather than the head-form hydraulic conductivity. gravity[I] is the
  // gravity vector (m/s^2). rho0 is the wetting reference density (kg/m^3),
  // used only inside the compressibility model. There is no division by rho0
  // in the gravity term: in pressure form the hydrostatic correction is
  // (grad p - rho_w g) directly.
  //
  // Closure functions vgm_/bc_*_from_Se expect wetting effective saturation
  //   S_e = (S_w - S_wr)/(1 - S_wr) = (1 - u_n - S_wr)/(1 - S_wr).
  // dS_e/du_n = -1/(1 - S_wr) (negative); the sign flip propagates through
  // every chain rule (k_rw, theta_w, p_c) when caller uses dSe_du_n.
  inline void evaluateCoefficients_from_Se(const int rowptr[nSpace], const int colind[nnz],
                                           const double rho0, const double rho_transport, const double beta,
                                           const double gravity[nSpace],
                                           const double alpha, const double n_vg,
                                           const double thetaR, const double thetaSR,
                                           const double KWs[nnz],
                                           const double &u_w, const double &u_n,
                                           double &m, double &dm_du_w, double &dm_du_n,
                                           double f[nSpace], double df_du_w[nSpace], double df_du_n[nSpace],
                                           double a[nnz], double da_du_w[nnz], double da_du_n[nnz],
                                           double as[nnz],
                                           double &kr, double &dkr_du_w, double &dkr_du_n,
                                           double &thetaW_out)
  {
    const double phi      = thetaR + thetaSR;                    // == thetaS
    const double S_wr     = thetaR / phi;                        // residual S_w
    const double one_m_Sr = 1.0 - S_wr;                          // = thetaSR/phi
    // Se in wetting form, expressed in terms of u_n = S_n.
    const double Se_raw   = (1.0 - u_n - S_wr) / one_m_Sr;
    // at the clips Se is held constant, so dSe/du_n=0.
    // Without this, the closure's DthetaW_DSe (returned regardless of
    // clipping) gets multiplied by +-1/(1-S_wr) and produces a non-zero (0,1)
    // mass Jacobian entry in the infeasible-u_n region, which tricks Newton
    // into overshooting further past saturation/residual.
    double Se, dSe_du_n;
    if (Se_raw <= 0.0)      { Se = 0.0;     dSe_du_n = 0.0; }
    else if (Se_raw >= 1.0) { Se = 1.0;     dSe_du_n = 0.0; }
    else                    { Se = Se_raw;  dSe_du_n = -1.0 / one_m_Sr; }

    double thetaW, DthetaW_DSe, KWr, DKWr_DSe;
    if (PSK_TYPE_member == 1) {
      proteus::mphase_co2::psk::bc_wetting_from_Se(
          Se, alpha, n_vg, thetaR, thetaSR,
          thetaW, DthetaW_DSe, KWr, DKWr_DSe);
    } else {
      proteus::mphase_co2::psk::vgm_wetting_from_Se(
          Se, alpha, n_vg, thetaR, thetaSR,
          thetaW, DthetaW_DSe, KWr, DKWr_DSe);
    }
    thetaW_out = thetaW;
    // Density: rho_w(p_w) = rho_transport * exp(beta * p_w); beta in 1/Pa.
    const double rhom  = rho_transport * exp(beta * u_w);
    const double drhom = beta * rhom;
    // Mass: m_w = rho_w * theta_w(S_w(u_n)).
    //       dm_w/du_n = rho_w * DthetaW_DSe * dSe/du_n  (negative branch
    //                   under monotone closures).
    m       = rhom * thetaW;
    dm_du_w = drhom * thetaW;
    dm_du_n = rhom * DthetaW_DSe * dSe_du_n;
    // Chain-rule factor for k_rw and downstream Se-derivatives.
    const double DKWr_Du_n = DKWr_DSe * dSe_du_n;
    for (int I = 0; I < nSpace; I++) {
      f[I]       = 0.0;
      df_du_w[I] = 0.0;
      df_du_n[I] = 0.0;
      for (int ii = rowptr[I]; ii < rowptr[I + 1]; ii++) {
        // Diffusion tensor a_w = rho_w * k_rw * (K/mu_w).
        a[ii]       = rhom * KWr * KWs[ii];
        da_du_w[ii] = drhom * KWr * KWs[ii];
        da_du_n[ii] = rhom * DKWr_Du_n * KWs[ii];
        // Gravity flux f_w = rho_w^2 * k_rw * (K/mu_w) * g  (pressure form;
        // no /rho0 factor).
        f[I] += rhom * rhom * KWr * KWs[ii] * gravity[colind[ii]];
        df_du_w[I] += 2.0 * drhom * rhom * KWr * KWs[ii] * gravity[colind[ii]];
        df_du_n[I] += rhom * rhom * DKWr_Du_n * KWs[ii] * gravity[colind[ii]];
        as[ii] = rhom * KWs[ii];
        kr        = KWr;
        dkr_du_w  = 0.0;            // k_rw depends on S_w (= 1-u_n) only
        dkr_du_n  = DKWr_Du_n;
      }
    }
  }

  // Inversion routines moved to psk_models.h (vgm_invert_analytic / vgm_invert_newton).

  inline void calculateCFL(const double &elementDiameter, const double df[nSpace], double &cfl)
  {
    double h, nrm_v;
    h     = elementDiameter;
    nrm_v = 0.0;
    for (int I = 0; I < nSpace; I++) nrm_v += df[I] * df[I];
    nrm_v = sqrt(nrm_v);
    cfl   = nrm_v / h;
  }

  inline void calculateSubgridError_tau(const double &elementDiameter, const double &dmt, const double dH[nSpace], double &cfl, double &tau)
  {
    double h, nrm_v, oneByAbsdt;
    h     = elementDiameter;
    nrm_v = 0.0;
    for (int I = 0; I < nSpace; I++) nrm_v += dH[I] * dH[I];
    nrm_v      = sqrt(nrm_v);
    cfl        = nrm_v / h;
    oneByAbsdt = fabs(dmt);
    tau        = 1.0 / (2.0 * nrm_v / h + oneByAbsdt + 1.0e-8);
  }

  inline void calculateSubgridError_tau(const double &Ct_sge, const double G[nSpace * nSpace], const double &A0, const double Ai[nSpace], double &tau_v, double &cfl)
  {
    double v_d_Gv = 0.0;
    for (int I = 0; I < nSpace; I++)
      for (int J = 0; J < nSpace; J++) v_d_Gv += Ai[I] * G[I * nSpace + J] * Ai[J];
    tau_v = 1.0 / sqrt(Ct_sge * A0 * A0 + v_d_Gv);
  }

  inline void calculateNumericalDiffusion(const double &shockCapturingDiffusion, const double &elementDiameter, const double &strong_residual, const double grad_u[nSpace], double &numDiff)
  {
    double h, num, den, n_grad_u;
    h        = elementDiameter;
    n_grad_u = 0.0;
    for (int I = 0; I < nSpace; I++) n_grad_u += grad_u[I] * grad_u[I];
    num     = shockCapturingDiffusion * 0.5 * h * fabs(strong_residual);
    den     = sqrt(n_grad_u) + 1.0e-8;
    numDiff = num / den;
  }

  inline void exteriorNumericalFlux(const double &bc_flux, int rowptr[nSpace], int colind[nnz], int isSeepageFace, int &isDOFBoundary, double n[nSpace], double bc_u, double K[nnz], double grad_psi[nSpace], double u, double K_rho_g[nSpace], double penalty, double &flux)
  {
    double v_I, bc_u_seepage = 0.0;
    if (isSeepageFace || isDOFBoundary) {
      flux = 0.0;
      for (int I = 0; I < nSpace; I++) {
        //gravity
        v_I = K_rho_g[I];
        //pressure head
        for (int m = rowptr[I]; m < rowptr[I + 1]; m++) { v_I -= K[m] * grad_psi[colind[m]]; }
        flux += v_I * n[I];
      }
      if (isSeepageFace) bc_u = bc_u_seepage;
      flux += penalty * (u - bc_u);
      //flux -= penalty * bc_u;
      if (isSeepageFace) {
        if (flux > 0.0) {
          isDOFBoundary = 1;
          bc_u          = bc_u_seepage;
        } else {
          isDOFBoundary = 0;
          flux          = 0.0;
        }
      }
    } else flux = bc_flux;
  }

  void exteriorNumericalFluxJacobian(const int rowptr[nSpace], const int colind[nnz], const int isDOFBoundary, const double n[nSpace], const double K[nnz], const double dK[nnz], const double grad_psi[nSpace], const double grad_v[nSpace], const double dK_rho_g[nSpace], const double v, const double penalty, double &fluxJacobian)
  {
    if (isDOFBoundary) {
      fluxJacobian = 0.0;
      for (int I = 0; I < nSpace; I++) {
        //gravity
        fluxJacobian += dK_rho_g[I] * v * n[I];
        //pressure head
        for (int m = rowptr[I]; m < rowptr[I + 1]; m++) { fluxJacobian -= (K[m] * grad_v[colind[m]] + dK[m] * v * grad_psi[colind[m]]) * n[I]; }
      }
      //Dirichlet penalty
      fluxJacobian += penalty * v;
    } else fluxJacobian = 0.0;
  }

inline void exteriorNumericalFlux2(const double &bc_flux, int rowptr[nSpace], int colind[nnz], int isSeepageFace, int &isDOFBoundary, double n[nSpace], double bc_u, double K[nnz], double grad_psi[nSpace], double u, double K_rho_g[nSpace], double penalty, double &flux, double &bflux)
  {
    double v_I, bc_u_seepage = 0.0;
    if (isSeepageFace || isDOFBoundary) {
      flux = 0.0;
      bflux = 0.0;
      for (int I = 0; I < nSpace; I++) {
        //gravity
        v_I = K_rho_g[I];
        //pressure head
        for (int m = rowptr[I]; m < rowptr[I + 1]; m++) { v_I -= K[m] * grad_psi[colind[m]]; }
        flux += v_I * n[I];
      }
      if (isSeepageFace) bc_u = bc_u_seepage;
      flux += penalty * (u - bc_u);
      bflux += penalty * (u - bc_u);
      if (isSeepageFace) {
        if (flux > 0.0) {
          isDOFBoundary = 1;
        } else {
          isDOFBoundary = 0;
          flux          = 0.0;
          bflux         = 0.0;
        }
      }
    } else {
      flux = bc_flux;
      bflux = bc_flux;
    }
  }

  void exteriorNumericalFluxJacobian2(const int rowptr[nSpace], const int colind[nnz], const int isDOFBoundary, const double n[nSpace],  const double Ks[nnz], const double K[nnz], const double dK[nnz], const double grad_psi[nSpace], const double grad_v[nSpace], const double dK_rho_g[nSpace], const double v, const double penalty, double &fluxJacobian, double &bfluxJacobian)
  {
    if (isDOFBoundary) {
      fluxJacobian = 0.0;
      bfluxJacobian = 0.0;
      for (int I = 0; I < nSpace; I++) {
        for (int m = rowptr[I]; m < rowptr[I + 1]; m++) { 
          fluxJacobian -= Ks[m] * grad_v[colind[m]] * n[I]; 
        }
      }
      //Dirichlet penalty
      bfluxJacobian = penalty * v;
    } else {
      fluxJacobian = 0.0;
      bfluxJacobian = 0.0;
    }
  }

  double seepagefluxcalculator(double anb_seepage_flux, int isSeepageFace, double dS, double flux_ext)
  {
    if (isSeepageFace) { anb_seepage_flux += flux_ext * dS; }
    return anb_seepage_flux;
  }

  void calculateResidual(arguments_dict &args)
  {
    xt::pyarray<double> &mesh_trial_ref                             = args.array<double>("mesh_trial_ref");
    xt::pyarray<double> &mesh_grad_trial_ref                        = args.array<double>("mesh_grad_trial_ref");
    xt::pyarray<double> &mesh_dof                                   = args.array<double>("mesh_dof");
    xt::pyarray<double> &mesh_velocity_dof                          = args.array<double>("mesh_velocity_dof");
    double               MOVING_DOMAIN                              = args.scalar<double>("MOVING_DOMAIN");
    xt::pyarray<int>    &mesh_l2g                                   = args.array<int>("mesh_l2g");
    xt::pyarray<double> &dV_ref                                     = args.array<double>("dV_ref");
    xt::pyarray<double> &u_trial_ref                                = args.array<double>("u_trial_ref");
    xt::pyarray<double> &u_grad_trial_ref                           = args.array<double>("u_grad_trial_ref");
    xt::pyarray<double> &u_test_ref                                 = args.array<double>("u_test_ref");
    xt::pyarray<double> &u_grad_test_ref                            = args.array<double>("u_grad_test_ref");
    xt::pyarray<double> &mesh_trial_trace_ref                       = args.array<double>("mesh_trial_trace_ref");
    xt::pyarray<double> &mesh_grad_trial_trace_ref                  = args.array<double>("mesh_grad_trial_trace_ref");
    xt::pyarray<double> &dS_ref                                     = args.array<double>("dS_ref");
    xt::pyarray<double> &u_trial_trace_ref                          = args.array<double>("u_trial_trace_ref");
    xt::pyarray<double> &u_grad_trial_trace_ref                     = args.array<double>("u_grad_trial_trace_ref");
    xt::pyarray<double> &u_test_trace_ref                           = args.array<double>("u_test_trace_ref");
    xt::pyarray<double> &u_grad_test_trace_ref                      = args.array<double>("u_grad_test_trace_ref");
    xt::pyarray<double> &normal_ref                                 = args.array<double>("normal_ref");
    xt::pyarray<double> &boundaryJac_ref                            = args.array<double>("boundaryJac_ref");
    int                  nElements_global                           = args.scalar<int>("nElements_global");
    xt::pyarray<double> &ebqe_penalty_ext                           = args.array<double>("ebqe_penalty_ext");
    xt::pyarray<int>    &elementMaterialTypes                       = args.array<int>("elementMaterialTypes");
    xt::pyarray<int>    &isSeepageFace                              = args.array<int>("isSeepageFace");
    xt::pyarray<int>    &a_rowptr                                   = args.array<int>("a_rowptr");
    xt::pyarray<int>    &a_colind                                   = args.array<int>("a_colind");
    double               rho                                        = args.scalar<double>("rho");
    double               beta                                       = args.scalar<double>("beta");

    /////////////////////////////DENSITY COUPLING  >>>> USE rho from mprans model/////////////////////////
    xt::pyarray<double> &q_rho                                     = args.array<double>("q_rho");
    xt::pyarray<double> &ebqe_rho                                  = args.array<double>("ebqe_rho");

    xt::pyarray<double> &gravity                                    = args.array<double>("gravity");
    xt::pyarray<double> &alpha                                      = args.array<double>("alpha");
    xt::pyarray<double> &n                                          = args.array<double>("n");
    xt::pyarray<double> &thetaR                                     = args.array<double>("thetaR");
    xt::pyarray<double> &thetaSR                                    = args.array<double>("thetaSR");
    xt::pyarray<double> &KWs                                        = args.array<double>("KWs");
    double               useMetrics                                 = args.scalar<double>("useMetrics");
    double               alphaBDF                                   = args.scalar<double>("alphaBDF");
    int                  lag_shockCapturing                         = args.scalar<int>("lag_shockCapturing");
    double               shockCapturingDiffusion                    = args.scalar<double>("shockCapturingDiffusion");
    double               sc_uref                                    = args.scalar<double>("sc_uref");
    double               sc_alpha                                   = args.scalar<double>("sc_alpha");
    xt::pyarray<int>    &u_l2g                                      = args.array<int>("u_l2g");
    xt::pyarray<double> &elementDiameter                            = args.array<double>("elementDiameter");
    xt::pyarray<double> &u_dof                                      = args.array<double>("u_dof");
    xt::pyarray<double> &u_dof_old                                  = args.array<double>("u_dof_old");
    xt::pyarray<double> &velocity                                   = args.array<double>("velocity");
    xt::pyarray<double> &q_m                                        = args.array<double>("q_m");
    xt::pyarray<double> &q_theta                                    = args.array<double>("q_theta");
    xt::pyarray<double> &q_u                                        = args.array<double>("q_u");
    xt::pyarray<double> &q_dV                                       = args.array<double>("q_dV");
    xt::pyarray<double> &q_m_betaBDF                                = args.array<double>("q_m_betaBDF");
    xt::pyarray<double> &cfl                                        = args.array<double>("cfl");
    xt::pyarray<double> &q_numDiff_u                                = args.array<double>("q_numDiff_u");
    xt::pyarray<double> &q_numDiff_u_last                           = args.array<double>("q_numDiff_u_last");
    int                  offset_u                                   = args.scalar<int>("offset_u");
    int                  stride_u                                   = args.scalar<int>("stride_u");
    // component-1 (S_n) mass equation args.
    // Used in the dedicated component-1 element loop appended at the end
    // of this function. Not consumed by the existing component-0 logic.
    const double         dt                                         = args.scalar<double>("dt");
    xt::pyarray<double> &u_dof_n                                    = args.array<double>("u_dof_n");
    xt::pyarray<double> &u_dof_n_old                                = args.array<double>("u_dof_n_old");
    // gas-phase density (constant for now). Will become
    // ρ_n(p_n) once Step 3 turns on real two-phase coupling.
    const double         rho_n                                      = args.scalar<double>("rho_n");
    const int            offset_n                                   = args.scalar<int>("offset_n");
    const int            stride_n                                   = args.scalar<int>("stride_n");
    xt::pyarray<double> &globalResidual                             = args.array<double>("globalResidual");
    int                  nExteriorElementBoundaries_global          = args.scalar<int>("nExteriorElementBoundaries_global");
    xt::pyarray<int>    &exteriorElementBoundariesArray             = args.array<int>("exteriorElementBoundariesArray");
    xt::pyarray<int>    &elementBoundaryElementsArray               = args.array<int>("elementBoundaryElementsArray");
    xt::pyarray<int>    &elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
    xt::pyarray<double> &ebqe_velocity_ext                          = args.array<double>("ebqe_velocity_ext");
    xt::pyarray<int>    &isDOFBoundary_u                            = args.array<int>("isDOFBoundary_u");
    xt::pyarray<double> &ebqe_bc_u_ext                              = args.array<double>("ebqe_bc_u_ext");
    // component-1 (S_n) boundary arrays.
    xt::pyarray<int>    &isDOFBoundary_n                            = args.array<int>("isDOFBoundary_n");
    xt::pyarray<double> &ebqe_bc_u_n_ext                            = args.array<double>("ebqe_bc_u_n_ext");
    xt::pyarray<int>    &isFluxBoundary_u                           = args.array<int>("isFluxBoundary_u");
    xt::pyarray<double> &ebqe_bc_flux_ext                           = args.array<double>("ebqe_bc_flux_ext");
    xt::pyarray<double> &ebqe_phi                                   = args.array<double>("ebqe_phi");
    double               epsFact                                    = args.scalar<double>("epsFact");
    xt::pyarray<double> &ebqe_u                                     = args.array<double>("ebqe_u");
    xt::pyarray<double> &ebqe_theta                                 = args.array<double>("ebqe_theta");
    xt::pyarray<double> &ebqe_flux                                  = args.array<double>("ebqe_flux");
    // VMS
    double VMS = args.scalar<double>("VMS");
    // PARAMETERS FOR EDGE BASED STABILIZATION
    double cE = args.scalar<double>("cE");
    double cK = args.scalar<double>("cK");
    // PARAMETERS FOR LOG BASED ENTROPY FUNCTION
    double uL = args.scalar<double>("uL");
    double uR = args.scalar<double>("uR");
    // PARAMETERS FOR EDGE VISCOSITY
    int               numDOFs                       = args.scalar<int>("numDOFs");
    // numDOFs is the compact component-0 free-DOF count used by the stabilized
    // DOF loops. Full-matrix slots are recovered from offset/stride-aware CSR
    // indexing against the interleaved global matrix.
    int               numDOFs_u                     = args.scalar<int>("numDOFs_u");
    int               NNZ                           = args.scalar<int>("NNZ");
    xt::pyarray<int> &csrRowIndeces_DofLoops        = args.array<int>("csrRowIndeces_DofLoops");
    xt::pyarray<int> &csrColumnOffsets_DofLoops     = args.array<int>("csrColumnOffsets_DofLoops");
    xt::pyarray<int> &csrRowIndeces_Full            = args.array<int>("csrRowIndeces_Full");
    xt::pyarray<int> &csrColumnOffsets_Full         = args.array<int>("csrColumnOffsets_Full");
    xt::pyarray<int> &csrRowIndeces_CellLoops       = args.array<int>("csrRowIndeces_CellLoops");
    xt::pyarray<int> &csrColumnOffsets_CellLoops    = args.array<int>("csrColumnOffsets_CellLoops");
    xt::pyarray<int> &csrColumnOffsets_eb_CellLoops = args.array<int>("csrColumnOffsets_eb_CellLoops");
    // C matrices
    xt::pyarray<double> &Cx         = args.array<double>("Cx");
    xt::pyarray<double> &Cy         = args.array<double>("Cy");
    xt::pyarray<double> &Cz         = args.array<double>("Cz");
    xt::pyarray<double> &CTx        = args.array<double>("CTx");
    xt::pyarray<double> &CTy        = args.array<double>("CTy");
    xt::pyarray<double> &CTz        = args.array<double>("CTz");
    xt::pyarray<double> &ML         = args.array<double>("ML");
    xt::pyarray<double> &delta_x_ij = args.array<double>("delta_x_ij");
    // PARAMETERS FOR 1st or 2nd ORDER MPP METHOD
    int LUMPED_MASS_MATRIX = args.scalar<int>("LUMPED_MASS_MATRIX");
    STABILIZATION STABILIZATION_TYPE{static_cast<STABILIZATION>(args.scalar<int>("STABILIZATION_TYPE"))};
    int ENTROPY_TYPE = args.scalar<int>("ENTROPY_TYPE");
    // PSK closure selector for evaluateCoefficients (read from argsDict).
    PSK_TYPE_member = args.scalar<int>("PSK_TYPE");
    // FOR FCT
    xt::pyarray<double> &dLow                 = args.array<double>("dLow");
    xt::pyarray<double> &fluxMatrix           = args.array<double>("fluxMatrix");
    // AUX QUANTITIES OF INTEREST
    xt::pyarray<double> &quantDOFs = args.array<double>("quantDOFs");

    assert(a_rowptr.data()[nSpace] == nnz);
    assert(a_rowptr.data()[nSpace] == nSpace);
    //cek should this be read in?
    double Ct_sge = 4.0;

    xt::pyarray<double> &anb_seepage_flux_n = args.array<double>("anb_seepage_flux_n");

    xt::pyarray<double> &velocity_couple                            = args.array<double>("velocity_couple");
    xt::pyarray<double> &ebqe_velocity_ext_couple                          = args.array<double>("ebqe_velocity_ext_couple");
    
    // xt::pyarray<double> &q_x    = args.array<double>("q_x");
    // xt::pyarray<double> &ebqe_x = args.array<double>("ebqe_x");

    //double anb_seepage_flux=0.0;
    double &anb_seepage_flux(args.scalar<double>("anb_seepage_flux"));
    xt::pyarray<double> &q_velocity = args.array<double>("q_velocity");
    anb_seepage_flux = 0.0;

    //loop over elements to compute volume integrals and load them into element and global residual
    //
    //eN is the element index
    //eN_k is the quadrature point index for a scalar
    //eN_k_nSpace is the quadrature point index for a vector
    //eN_i is the element test function index
    //eN_j is the element trial function index
    //eN_k_j is the quadrature point index for a trial function
    //eN_k_i is the quadrature point index for a trial function
    for (int eN = 0; eN < nElements_global; eN++) {
      //declare local storage for element residual and initialize
      double elementResidual_u[nDOF_test_element];
      for (int i = 0; i < nDOF_test_element; i++) { elementResidual_u[i] = 0.0; } //i
      //loop over quadrature points and compute integrands
      for (int k = 0; k < nQuadraturePoints_element; k++) {
        //compute indeces and declare local storage
        int eN_k = eN * nQuadraturePoints_element + k, eN_k_nSpace = eN_k * nSpace, eN_nDOF_trial_element = eN * nDOF_trial_element;
        double u = 0.0, grad_u[nSpace], grad_u_old[nSpace], m = 0.0, dm = 0.0, f[nSpace], df[nSpace], a[nnz], da[nnz], as[nnz], m_t = 0.0, dm_t = 0.0, pdeResidual_u = 0.0, Lstar_u[nDOF_test_element], subgridError_u = 0.0, tau = 0.0, tau0 = 0.0, tau1 = 0.0, numDiff0 = 0.0, numDiff1 = 0.0, jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace], u_grad_trial[nDOF_trial_element * nSpace], u_test_dV[nDOF_trial_element], u_grad_test_dV[nDOF_test_element * nSpace], dV, x, y, z, xt, yt, zt, G[nSpace * nSpace], G_dd_G, tr_G, norm_Rv;
        //
        //compute solution and gradients at quadrature points
        //
        ck.calculateMapping_element(eN, k, mesh_dof.data(), mesh_l2g.data(), mesh_trial_ref.data(), mesh_grad_trial_ref.data(), jac, jacDet, jacInv, x, y, z);
        ck.calculateMappingVelocity_element(eN, k, mesh_velocity_dof.data(), mesh_l2g.data(), mesh_trial_ref.data(), xt, yt, zt);
        //get the physical integration weight
        dV                = fabs(jacDet) * dV_ref.data()[k];
        q_dV.data()[eN_k] = dV;
        ck.calculateG(jacInv, G, G_dd_G, tr_G);
        //get the trial function gradients
        ck.gradTrialFromRef(&u_grad_trial_ref.data()[k * nDOF_trial_element * nSpace], jacInv, u_grad_trial);
        //get the solution
        ck.valFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], &u_trial_ref.data()[k * nDOF_trial_element], u);
        //get the solution gradients
        ck.gradFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], u_grad_trial, grad_u);

        // //populate q_x
        // const int eN_k_3d = eN_k * 3;
        // q_x.data()[eN_k_3d + 0] = x;
        // q_x.data()[eN_k_3d + 1] = y;
        // q_x.data()[eN_k_3d + 2] = z;      
        
        //precalculate test function products with integration weights
        for (int j = 0; j < nDOF_trial_element; j++) {
          u_test_dV[j] = u_test_ref.data()[k * nDOF_trial_element + j] * dV;
          for (int I = 0; I < nSpace; I++) {
            u_grad_test_dV[j * nSpace + I] = u_grad_trial[j * nSpace + I] * dV; //cek warning won't work for Petrov-Galerkin
          }
        }
        //
        //calculate pde coefficients at quadrature points
        //
        double Kr, dKr, thetaW;
        const double rho_local = q_rho.data()[eN_k];
        const double rho_velocity = std::fabs(rho_local) > 1.0e-12 ? rho_local : rho;
        // Cross-derivatives (dm_du_n, df_du_n, da_du_n, dkr_du_n) are unused in
        // the residual; the (0,1) Jacobian cross-block consumes them elsewhere.
        double dm_du_n_qp = 0.0, dkr_du_n_qp = 0.0;
        double df_du_n_qp[nSpace];
        double da_du_n_qp[nnz];
        for (int I = 0; I < nSpace; I++) df_du_n_qp[I] = 0.0;
        for (int ii = 0; ii < nnz; ii++) da_du_n_qp[ii] = 0.0;
        double u_n_qp = 0.0;
        ck.valFromDOF(u_dof_n.data(),
                      &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_n_qp);
        evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, rho_local, beta, gravity.data(),
                                     alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]],
                                     thetaR.data()[elementMaterialTypes.data()[eN]], thetaSR.data()[elementMaterialTypes.data()[eN]],
                                     &KWs.data()[elementMaterialTypes.data()[eN] * nnz], u, u_n_qp,
                                     m, dm, dm_du_n_qp, f, df, df_du_n_qp, a, da, da_du_n_qp,
                                     as, Kr, dKr, dkr_du_n_qp, thetaW);
        q_theta.data()[eN_k] = thetaW;
        

        for (int I = 0; I < nSpace; ++I) {
          q_velocity.data()[eN_k_nSpace + I] = grad_u[I];
        }
        // Darcy Velocity
        double pressure_gradient[nSpace];
        for (int J=0; J<nSpace; ++J)
          pressure_gradient[J] = grad_u[J] - rho_velocity * gravity.data()[J];
        // q_w = -(a/rho_w) * (grad p_w - rho_w g) = -(k_rw K/mu_w) * (grad p_w - rho_w g).
        for (int I=0; I<nSpace; ++I) {
          double acc = 0.0;
          for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I+1]; ++ii) {
            const int J = a_colind.data()[ii];
            acc += (a[ii] / rho_velocity) * pressure_gradient[J];
          }
          velocity.data()[eN_k_nSpace + I] = -acc;
          velocity_couple.data()[eN_k_nSpace + I] = -acc ;
          }
        //
        //calculate time derivative at quadrature points
        //
        ck.bdf(alphaBDF, q_m_betaBDF.data()[eN_k], m, dm, m_t, dm_t);
        // //
        // //calculate subgrid error (strong residual and adjoint)
        // //
        // //calculate strong residual
        // pdeResidual_u = ck.Mass_strong(m_t) + ck.Advection_strong(df, grad_u);
        // //calculate adjoint
        // for (int i = 0; i < nDOF_test_element; i++) {
        //   int i_nSpace = i * nSpace;
        //   Lstar_u[i]   = ck.Advection_adjoint(df, &u_grad_test_dV[i_nSpace]);
        // }
        // //calculate tau and tau*Res
        // calculateSubgridError_tau(elementDiameter[eN], dm_t, df, cfl[eN_k], tau0);
        // calculateSubgridError_tau(Ct_sge, G, dm_t, df, tau1, cfl[eN_k]);

        // tau = useMetrics * tau1 + (1.0 - useMetrics) * tau0;

        // subgridError_u = -tau * pdeResidual_u;
        // //
        // //calculate shock capturing diffusion
        // //
        // ck.calculateNumericalDiffusion(shockCapturingDiffusion, elementDiameter[eN], pdeResidual_u, grad_u, numDiff0);
        // ck.calculateNumericalDiffusion(shockCapturingDiffusion, sc_uref, sc_alpha, G, G_dd_G, pdeResidual_u, grad_u, numDiff1);
        // q_numDiff_u[eN_k] = useMetrics * numDiff1 + (1.0 - useMetrics) * numDiff0;
        //
        //update element residual
        //
        for (int i = 0; i < nDOF_test_element; i++) {
          int eN_k_i = eN_k * nDOF_test_element + i, eN_k_i_nSpace = eN_k_i * nSpace, i_nSpace = i * nSpace;

          elementResidual_u[i] += ck.Mass_weak(m_t, u_test_dV[i]) + ck.Advection_weak(f, &u_grad_test_dV[i_nSpace]) + ck.Diffusion_weak(a_rowptr.data(), a_colind.data(), a, grad_u, &u_grad_test_dV[i_nSpace]) + VMS * ck.SubgridError(subgridError_u, Lstar_u[i]) + VMS * ck.NumericalDiffusion(q_numDiff_u_last[eN_k], grad_u, &u_grad_test_dV[i_nSpace]);
        } //i
        //
        q_m.data()[eN_k] = m;
        q_u.data()[eN_k] = u;
      }
      //
      //load element into global residual and save element residual
      //
      for (int i = 0; i < nDOF_test_element; i++) {
        int eN_i = eN * nDOF_test_element + i;

        globalResidual.data()[offset_u + stride_u * u_l2g.data()[eN_i]] += elementResidual_u[i];
      } //i
    } //elements
    //
    //loop over exterior element boundaries to calculate surface integrals and load into element and global residuals
    //
    //ebNE is the Exterior element boundary INdex
    //ebN is the element boundary INdex
    //eN is the element index
    for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++) {
      int    ebN = exteriorElementBoundariesArray.data()[ebNE], eN = elementBoundaryElementsArray.data()[ebN * 2 + 0], ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN * 2 + 0], eN_nDOF_trial_element = eN * nDOF_trial_element;
      double elementResidual_u[nDOF_test_element];
      for (int i = 0; i < nDOF_test_element; i++) { elementResidual_u[i] = 0.0; }
      for (int kb = 0; kb < nQuadraturePoints_elementBoundary; kb++) {
        int    ebNE_kb = ebNE * nQuadraturePoints_elementBoundary + kb, ebNE_kb_nSpace = ebNE_kb * nSpace, ebN_local_kb = ebN_local * nQuadraturePoints_elementBoundary + kb, ebN_local_kb_nSpace = ebN_local_kb * nSpace;
        double u_ext = 0.0, grad_u_ext[nSpace], m_ext = 0.0, dm_ext = 0.0, f_ext[nSpace], df_ext[nSpace], a_ext[nnz], da_ext[nnz], as_ext[nnz], flux_ext = 0.0,
               //anb_seepage_flux=0.0, // for flux calculation
          bc_u_ext = 0.0, bc_grad_u_ext[nSpace], bc_m_ext = 0.0, bc_dm_ext = 0.0, bc_f_ext[nSpace], bc_df_ext[nSpace], bc_a_ext[nnz], bc_da_ext[nnz], bc_as_ext[nnz], jac_ext[nSpace * nSpace], jacDet_ext, jacInv_ext[nSpace * nSpace], boundaryJac[nSpace * (nSpace - 1)], metricTensor[(nSpace - 1) * (nSpace - 1)], metricTensorDetSqrt, dS, u_test_dS[nDOF_test_element], u_grad_trial_trace[nDOF_trial_element * nSpace], normal[3], x_ext, y_ext, z_ext, xt_ext, yt_ext, zt_ext, integralScaling, G[nSpace * nSpace], G_dd_G, tr_G;
        //
        //calculate the solution and gradients at quadrature points
        //
        //compute information about mapping from reference element to physical element
        ck.calculateMapping_elementBoundary(eN, ebN_local, kb, ebN_local_kb, mesh_dof.data(), mesh_l2g.data(), mesh_trial_trace_ref.data(), mesh_grad_trial_trace_ref.data(), boundaryJac_ref.data(), jac_ext, jacDet_ext, jacInv_ext, boundaryJac, metricTensor, metricTensorDetSqrt,
                                            normal_ref.data(), normal, x_ext, y_ext, z_ext);
        ck.calculateMappingVelocity_elementBoundary(eN, ebN_local, kb, ebN_local_kb, mesh_velocity_dof.data(), mesh_l2g.data(), mesh_trial_trace_ref.data(), xt_ext, yt_ext, zt_ext, normal, boundaryJac, metricTensor, integralScaling);
        dS = ((1.0 - MOVING_DOMAIN) * metricTensorDetSqrt + MOVING_DOMAIN * integralScaling) * dS_ref.data()[kb];
        //get the metric tensor
        //cek todo use symmetry
        ck.calculateG(jacInv_ext, G, G_dd_G, tr_G);
        //compute shape and solution information
        //shape
        ck.gradTrialFromRef(&u_grad_trial_trace_ref.data()[ebN_local_kb_nSpace * nDOF_trial_element], jacInv_ext, u_grad_trial_trace);
        //solution and gradient
        ck.valFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], &u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element], u_ext);
        ck.gradFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], u_grad_trial_trace, grad_u_ext);
        
        //populate ebqe_x
        // const int ebNE_kb_3d = ebNE_kb * 3;
        // ebqe_x.data()[ebNE_kb_3d + 0] = x_ext;
        // ebqe_x.data()[ebNE_kb_3d + 1] = y_ext;
        // ebqe_x.data()[ebNE_kb_3d + 2] = z_ext;
        
        //precalculate test function products with integration weights
        for (int j = 0; j < nDOF_trial_element; j++) { u_test_dS[j] = u_test_trace_ref.data()[ebN_local_kb * nDOF_test_element + j] * dS; }
        //
        //load the boundary values
        //
        bc_u_ext = isDOFBoundary_u.data()[ebNE_kb] * ebqe_bc_u_ext.data()[ebNE_kb] + (1 - isDOFBoundary_u.data()[ebNE_kb]) * u_ext;
        //
        //calculate the pde coefficients using the solution and the boundary values for the solution
        //
        const double rho_ext = ebqe_rho.data()[ebNE_kb];
        const double rho_velocity_ext = std::fabs(rho_ext) > 1.0e-12 ? rho_ext : rho;
        double Kr, dKr, thetaW_ext, thetaW_bc;
        // Boundary closure cross-derivatives wrt u_n; (0,1) boundary Jacobian
        // is not assembled here.
        double dm_du_n_ext = 0.0, dkr_du_n_ext = 0.0;
        double df_du_n_ext[nSpace];
        double da_du_n_ext[nnz];
        double bc_dm_du_n = 0.0, bc_dkr_du_n = 0.0;
        double bc_df_du_n[nSpace];
        double bc_da_du_n[nnz];
        for (int I = 0; I < nSpace; I++) { df_du_n_ext[I] = 0.0; bc_df_du_n[I] = 0.0; }
        for (int ii = 0; ii < nnz; ii++)  { da_du_n_ext[ii] = 0.0; bc_da_du_n[ii] = 0.0; }
        double u_n_ext_qp = 0.0;
        ck.valFromDOF(u_dof_n.data(), &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element], u_n_ext_qp);
        // Component-1 boundary Dirichlet via the standard
        // mask = isDOFBoundary_n * bc + (1 - isDOFBoundary_n) * interior.
        const double bc_u_n_ext_qp = isDOFBoundary_n.data()[ebNE_kb] * ebqe_bc_u_n_ext.data()[ebNE_kb]
                                   + (1 - isDOFBoundary_n.data()[ebNE_kb]) * u_n_ext_qp;
        evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, rho_ext, beta, gravity.data(),
                                     alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]],
                                     thetaR.data()[elementMaterialTypes.data()[eN]], thetaSR.data()[elementMaterialTypes.data()[eN]],
                                     &KWs.data()[elementMaterialTypes.data()[eN] * nnz], u_ext, u_n_ext_qp,
                                     m_ext, dm_ext, dm_du_n_ext, f_ext, df_ext, df_du_n_ext, a_ext, da_ext, da_du_n_ext,
                                     as_ext, Kr, dKr, dkr_du_n_ext, thetaW_ext);
        evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, rho_ext, beta, gravity.data(),
                                     alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]],
                                     thetaR.data()[elementMaterialTypes.data()[eN]], thetaSR.data()[elementMaterialTypes.data()[eN]],
                                     &KWs.data()[elementMaterialTypes.data()[eN] * nnz], bc_u_ext, bc_u_n_ext_qp,
                                     bc_m_ext, bc_dm_ext, bc_dm_du_n, bc_f_ext, bc_df_ext, bc_df_du_n, bc_a_ext, bc_da_ext, bc_da_du_n,
                                     bc_as_ext, Kr, dKr, bc_dkr_du_n, thetaW_bc);
        ebqe_theta.data()[ebNE_kb] = thetaW_ext;
        
        //
        //Calculate Darcy velocity on exterior face : v_ext = -(a_ext/rho) * (grad_u_ext + gravity) ---
        //
        double ext_pressure_gradient[nSpace];
        for (int J=0; J<nSpace; ++J)
          ext_pressure_gradient[J] = grad_u_ext[J] - rho_velocity_ext * gravity.data()[J];

        for (int I=0; I<nSpace; ++I) {
          double acc = 0.0;
          for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I+1]; ++ii) {
            const int J = a_colind.data()[ii];
            acc += (a_ext[ii] / rho_velocity_ext) * ext_pressure_gradient[J];
          }
          ebqe_velocity_ext.data()[ebNE_kb_nSpace + I] = -acc;
          ebqe_velocity_ext_couple.data()[ebNE_kb_nSpace + I] = -acc ;  // store vector at this boundary qp
        }


        //
        //calculate the numerical fluxes
        //
        exteriorNumericalFlux(ebqe_bc_flux_ext[ebNE_kb], a_rowptr.data(), a_colind.data(),
                              isSeepageFace.data()[ebNE], //tricky, this is a face flag not face quad
                              isDOFBoundary_u.data()[ebNE_kb], normal, bc_u_ext, a_ext, grad_u_ext, u_ext, f_ext,
                              ebqe_penalty_ext.data()[ebNE_kb], // penalty,
                              flux_ext);
        ebqe_flux.data()[ebNE_kb] = flux_ext;

        anb_seepage_flux             = seepagefluxcalculator(anb_seepage_flux, isSeepageFace.data()[ebNE], dS, flux_ext);
        anb_seepage_flux_n.data()[0] = anb_seepage_flux;
        ebqe_u.data()[ebNE_kb]       = u_ext;
        //
        //update residuals
        //
        for (int i = 0; i < nDOF_test_element; i++) {
          elementResidual_u[i] += ck.ExteriorElementBoundaryFlux(flux_ext, u_test_dS[i]);
        } //i
      } //kb

      //
      //update the element and global residual storage
      //
      for (int i = 0; i < nDOF_test_element; i++) {
        int eN_i = eN * nDOF_test_element + i;
        globalResidual.data()[offset_u + stride_u * u_l2g.data()[eN_i]] += elementResidual_u[i];
      } //i
    } //ebNE

    // ============================================================================
    // component-1 (S_n) gas-mass equation.
    //
    //   d(phi*rho_n*S_n)/dt + div F_n = 0
    //     m_n     = phi * rho_n * u_n
    //     m_n_old = phi * rho_n * u_n_old
    //
    // Gas Darcy flux F_n = -(K/mu_n) k_rn (grad p_n - rho_n g)
    //                    = -(K/mu_n) k_rn (grad p_w + dp_c/dS_n grad S_n - rho_n g)
    // Splitting into Proteus form for u_w = p_w, u_n = S_n:
    //   f_n     = rho_n^2 * k_rn * (K/mu_n) * g          (gravity advection)
    //   a_n     = rho_n   * k_rn * (K/mu_n)               (diffusion against grad u_w)
    //   a_n_pc  = rho_n   * k_rn * (K/mu_n) * dp_c/dS_n   (capillary diffusion against grad u_n)
    // At u_n = 0 (fully wet) the closure returns k_rn = 0 -> all flux contributions
    // vanish, so this loop should reproduce the single-phase result for the
    // Bioswale.
    for (int eN = 0; eN < nElements_global; eN++) {
      const int   mat_eN  = elementMaterialTypes.data()[eN];
      const double phi_eN = thetaR.data()[mat_eN] + thetaSR.data()[mat_eN];
      const double alpha_eN = alpha.data()[mat_eN];
      const double n_vg_eN  = n.data()[mat_eN];
      const double *KWs_eN  = &KWs.data()[mat_eN * nnz];
      double elementResidual_n[nDOF_test_element];
      for (int i = 0; i < nDOF_test_element; i++) elementResidual_n[i] = 0.0;
      for (int k = 0; k < nQuadraturePoints_element; k++) {
        const int eN_k = eN * nQuadraturePoints_element + k;
        const int eN_nDOF_trial_element = eN * nDOF_trial_element;
        double jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace], x_q, y_q, z_q;
        ck.calculateMapping_element(eN, k, mesh_dof.data(), mesh_l2g.data(),
                                    mesh_trial_ref.data(), mesh_grad_trial_ref.data(),
                                    jac, jacDet, jacInv, x_q, y_q, z_q);
        const double dV = std::fabs(jacDet) * dV_ref.data()[k];
        // Trial gradients in physical coords (used for grad u_w and grad N_i).
        double u_grad_trial_qp[nDOF_trial_element * nSpace];
        ck.gradTrialFromRef(&u_grad_trial_ref.data()[k * nDOF_trial_element * nSpace],
                            jacInv, u_grad_trial_qp);
        // Saturation u_n and old at QP.
        double u_n = 0.0, u_n_old = 0.0;
        ck.valFromDOF(u_dof_n.data(),
                      &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_n);
        ck.valFromDOF(u_dof_n_old.data(),
                      &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_n_old);
        // Wetting-pressure gradient at QP.
        double grad_u_w[nSpace];
        ck.gradFromDOF(u_dof.data(),
                       &u_l2g.data()[eN_nDOF_trial_element],
                       u_grad_trial_qp, grad_u_w);
        // Mass term: m_n = phi * rho_n * u_n with u_n = S_n.
        const double m_n     = phi_eN * rho_n * u_n;
        const double m_n_old = phi_eN * rho_n * u_n_old;
        const double m_n_t   = (m_n - m_n_old) / dt;
        // Saturation gradient at QP (Step 3d: needed for p_c(S_n) flux contribution).
        double grad_u_n[nSpace];
        ck.gradFromDOF(u_dof_n.data(),
                       &u_l2g.data()[eN_nDOF_trial_element],
                       u_grad_trial_qp, grad_u_n);
        // Non-wetting relative permeability via PSK closure dispatch.
        // u_n is S_n; closures take wetting effective saturation S_e.
        const double phi_loc      = thetaR.data()[mat_eN] + thetaSR.data()[mat_eN];
        const double S_wr_loc     = thetaR.data()[mat_eN] / phi_loc;
        const double one_m_Sr_loc = 1.0 - S_wr_loc;
        // Se = (S_w - S_wr)/(1-S_wr) = (1 - u_n - S_wr)/(1 - S_wr).
        const double Se_qp_raw    = (1.0 - u_n - S_wr_loc) / one_m_Sr_loc;
        // zero out dSe/du_n at the Se clip so chain-rule derivatives don't
        // fake a non-zero gradient in the infeasible-u_n range.
        double Se_qp, dSe_du_n_loc;
        if (Se_qp_raw <= 0.0)      { Se_qp = 0.0;        dSe_du_n_loc = 0.0; }
        else if (Se_qp_raw >= 1.0) { Se_qp = 1.0;        dSe_du_n_loc = 0.0; }
        else                       { Se_qp = Se_qp_raw;  dSe_du_n_loc = -1.0 / one_m_Sr_loc; }
        double KNr = 0.0, DKNr_DSe = 0.0;
        if (PSK_TYPE_member == 1) {
          proteus::mphase_co2::psk::bc_kr_nonwetting_from_Se(
              Se_qp, alpha_eN, n_vg_eN, KNr, DKNr_DSe);
        } else {
          proteus::mphase_co2::psk::vgm_kr_nonwetting_from_Se(
              Se_qp, alpha_eN, n_vg_eN, KNr, DKNr_DSe);
        }
        // Chain rule: convert d/dS_e -> d/du_n. dSe_du_n_loc carries the
        // sign flip from the S_n primary so DKNr_Du_n has the correct sign.
        DKNr_DSe *= dSe_du_n_loc;
        // Capillary derivative. Closure returns d p_c / d Se (negative under
        // monotone closures). Chain rule with dSe_du_n_loc = -1/(1-S_wr)
        // gives dpc_dSw_local = d(p_c)/d(u_n) = d(p_c)/d(S_n) (POSITIVE under
        // monotone closures, since dpc/dSw < 0). The variable is still named
        // 'dpc_dSw' for historical reasons; semantically it is dp_c/d(u_n).
        // d2pc_dSw2 stays a true second derivative (sign-invariant under the
        // chain rule since dSe/du_n squared is positive).
        double pc_qp = 0.0, dpc_dSw = 0.0, d2pc_dSw2 = 0.0;
        if (PSK_TYPE_member == 1) {
          proteus::mphase_co2::psk::bc_pc_from_Se(Se_qp, alpha_eN, n_vg_eN, pc_qp, dpc_dSw, d2pc_dSw2);
        } else {
          proteus::mphase_co2::psk::vgm_pc_from_Se(Se_qp, alpha_eN, n_vg_eN, pc_qp, dpc_dSw, d2pc_dSw2);
        }
        dpc_dSw   *= dSe_du_n_loc;
        d2pc_dSw2 *= dSe_du_n_loc * dSe_du_n_loc;
        // Build a_n (diffusion against grad u_w), a_n_p_c (diffusion against
        // grad u_n through capillary coupling), and f_n (gas gravity flux).
        double a_n[nnz];
        double a_n_p_c[nnz];
        double f_n[nSpace];
        for (int I = 0; I < nSpace; I++) f_n[I] = 0.0;
        for (int I = 0; I < nSpace; I++) {
          for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I + 1]; ii++) {
            const int J = a_colind.data()[ii];
            a_n[ii]     = rho_n * KNr * KWs_eN[ii];
            a_n_p_c[ii] = a_n[ii] * dpc_dSw;
            f_n[I] += rho_n * rho_n * KNr * KWs_eN[ii] * gravity.data()[J];
          }
        }
        // Residual integration: mass + advection (gravity) - diffusion(grad u_w) - p_c diffusion(grad u_n).
        for (int i = 0; i < nDOF_test_element; i++) {
          const double test_i = u_test_ref.data()[k * nDOF_test_element + i];
          // Mass contribution.
          elementResidual_n[i] += m_n_t * test_i * dV;
          // Advection contribution: f_n . grad N_i dV.
          for (int I = 0; I < nSpace; I++) {
            elementResidual_n[i] += f_n[I] * u_grad_trial_qp[i * nSpace + I] * dV;
          }
          // Diffusion contribution: + a_n grad u_w . grad N_i dV
          // (sign: -nabla.(a grad u) integrated against N_i gives + a grad u . grad N_i).
          for (int I = 0; I < nSpace; I++) {
            for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I + 1]; ii++) {
              const int J = a_colind.data()[ii];
              elementResidual_n[i] += a_n[ii] * grad_u_w[J]
                                    * u_grad_trial_qp[i * nSpace + I] * dV;
            }
          }
          // Step 3d: capillary diffusion: + a_n * (dp_c/dS_w) * grad u_n . grad N_i dV.
          for (int I = 0; I < nSpace; I++) {
            for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I + 1]; ii++) {
              const int J = a_colind.data()[ii];
              elementResidual_n[i] += a_n_p_c[ii] * grad_u_n[J]
                                    * u_grad_trial_qp[i * nSpace + I] * dV;
            }
          }
        }
      }
      for (int i = 0; i < nDOF_test_element; i++) {
        const int eN_i = eN * nDOF_test_element + i;
        globalResidual.data()[offset_n + stride_n * u_l2g.data()[eN_i]] += elementResidual_n[i];
      }
    }
  }

  void calculateJacobian(arguments_dict &args)
  {
    xt::pyarray<double> &mesh_trial_ref            = args.array<double>("mesh_trial_ref");
    xt::pyarray<double> &mesh_grad_trial_ref       = args.array<double>("mesh_grad_trial_ref");
    xt::pyarray<double> &mesh_dof                  = args.array<double>("mesh_dof");
    xt::pyarray<double> &mesh_velocity_dof         = args.array<double>("mesh_velocity_dof");
    double               MOVING_DOMAIN             = args.scalar<double>("MOVING_DOMAIN");
    xt::pyarray<int>    &mesh_l2g                  = args.array<int>("mesh_l2g");
    xt::pyarray<double> &dV_ref                    = args.array<double>("dV_ref");
    xt::pyarray<double> &u_trial_ref               = args.array<double>("u_trial_ref");
    xt::pyarray<double> &u_grad_trial_ref          = args.array<double>("u_grad_trial_ref");
    xt::pyarray<double> &u_test_ref                = args.array<double>("u_test_ref");
    xt::pyarray<double> &u_grad_test_ref           = args.array<double>("u_grad_test_ref");
    xt::pyarray<double> &mesh_trial_trace_ref      = args.array<double>("mesh_trial_trace_ref");
    xt::pyarray<double> &mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
    xt::pyarray<double> &dS_ref                    = args.array<double>("dS_ref");
    xt::pyarray<double> &u_trial_trace_ref         = args.array<double>("u_trial_trace_ref");
    xt::pyarray<double> &u_grad_trial_trace_ref    = args.array<double>("u_grad_trial_trace_ref");
    xt::pyarray<double> &u_test_trace_ref          = args.array<double>("u_test_trace_ref");
    xt::pyarray<double> &u_grad_test_trace_ref     = args.array<double>("u_grad_test_trace_ref");
    xt::pyarray<double> &normal_ref                = args.array<double>("normal_ref");
    xt::pyarray<double> &boundaryJac_ref           = args.array<double>("boundaryJac_ref");
    int                  nElements_global          = args.scalar<int>("nElements_global");
    xt::pyarray<double> &ebqe_penalty_ext          = args.array<double>("ebqe_penalty_ext");
    xt::pyarray<int>    &elementMaterialTypes      = args.array<int>("elementMaterialTypes");
    xt::pyarray<int>    &isSeepageFace             = args.array<int>("isSeepageFace");
    xt::pyarray<int>    &a_rowptr                  = args.array<int>("a_rowptr");
    xt::pyarray<int>    &a_colind                  = args.array<int>("a_colind");
    double               rho                       = args.scalar<double>("rho");
    double               beta                      = args.scalar<double>("beta");

    /////////////////////////////DENSITY COUPLING  >>>> USE rho from mprans model/////////////////////////
    xt::pyarray<double> &q_rho                    = args.array<double>("q_rho");
    xt::pyarray<double> &ebqe_rho                 = args.array<double>("ebqe_rho");
    //////////////////////////////////////////////////////////////////////////////////////////////////////

    xt::pyarray<double> &gravity                   = args.array<double>("gravity");
    xt::pyarray<double> &alpha                     = args.array<double>("alpha");
    xt::pyarray<double> &n                         = args.array<double>("n");
    xt::pyarray<double> &thetaR                    = args.array<double>("thetaR");
    xt::pyarray<double> &thetaSR                   = args.array<double>("thetaSR");
    xt::pyarray<double> &KWs                       = args.array<double>("KWs");
    double               useMetrics                = args.scalar<double>("useMetrics");
    double               alphaBDF                  = args.scalar<double>("alphaBDF");
    int                  lag_shockCapturing        = args.scalar<int>("lag_shockCapturing");
    double               shockCapturingDiffusion   = args.scalar<double>("shockCapturingDiffusion");
    // VMS
    double               VMS                                        = args.scalar<double>("VMS");
    xt::pyarray<int>    &u_l2g                                      = args.array<int>("u_l2g");
    xt::pyarray<double> &elementDiameter                            = args.array<double>("elementDiameter");
    xt::pyarray<double> &u_dof                                      = args.array<double>("u_dof");
    // component-1 saturation DOFs (needed by the gas-eq
    // Jacobian element loop appended at the end of this function).
    xt::pyarray<double> &u_dof_n                                    = args.array<double>("u_dof_n");
    xt::pyarray<double> &velocity                                   = args.array<double>("velocity");
    xt::pyarray<double> &q_m_betaBDF                                = args.array<double>("q_m_betaBDF");
    xt::pyarray<double> &cfl                                        = args.array<double>("cfl");
    xt::pyarray<double> &q_numDiff_u                                = args.array<double>("q_numDiff_u");
    xt::pyarray<double> &q_numDiff_u_last                           = args.array<double>("q_numDiff_u_last");
    xt::pyarray<int>    &csrRowIndeces_u_u                          = args.array<int>("csrRowIndeces_u_u");
    xt::pyarray<int>    &csrColumnOffsets_u_u                       = args.array<int>("csrColumnOffsets_u_u");
    xt::pyarray<double> &globalJacobian                             = args.array<double>("globalJacobian");
    // component-1 (S_n) Jacobian args. Used by
    // the dedicated component-1 element loop appended at the end of this
    // function. (1,1) block is the consistent mass matrix / dt; (0,1) and
    // (1,0) cross-blocks are zero in Step 1.
    const double         dt_n                                       = args.scalar<double>("dt");
    // gas-phase density (constant). Used in the (1,1) block
    // assembly: J_(1,1) = -phi*rho_n*M/dt.
    const double         rho_n                                      = args.scalar<double>("rho_n");
    xt::pyarray<int>    &csrRowIndeces_n_n                          = args.array<int>("csrRowIndeces_n_n");
    // (1,0) cross-block CSR maps for the gas-eq diffusion
    // against grad u_w. Currently allocated by the framework but unused by
    // the C++ assembly - 3c.2/3c.3 will write into them.
    xt::pyarray<int>    &csrRowIndeces_n_w                          = args.array<int>("csrRowIndeces_n_w");
    xt::pyarray<int>    &csrColumnOffsets_n_n                       = args.array<int>("csrColumnOffsets_n_n");
    xt::pyarray<int>    &csrColumnOffsets_n_w                       = args.array<int>("csrColumnOffsets_n_w");
    // (0,1) cross-block CSR maps for the wetting eq.
    //   J_{wv,ij} = (dm/du_n)*N_i*N_j + (df/du_n)*grad N_i*N_j
    //               + (da/du_n)*grad u_w*grad N_i*N_j
    // The wetting equation row index is offset_u + stride_u * dof, and the column
    // index is offset_n + stride_n * dof.
    xt::pyarray<int>    &csrRowIndeces_w_n                          = args.array<int>("csrRowIndeces_w_n");
    xt::pyarray<int>    &csrColumnOffsets_w_n                       = args.array<int>("csrColumnOffsets_w_n");
    // (0,1) cross-block boundary CSR for the wetting-eq
    // exterior-flux Jacobian.
    xt::pyarray<int>    &csrColumnOffsets_eb_w_n                    = args.array<int>("csrColumnOffsets_eb_w_n");
    // PSK closure selector for evaluateCoefficients (read from argsDict).
    PSK_TYPE_member = args.scalar<int>("PSK_TYPE");
    int                  nExteriorElementBoundaries_global          = args.scalar<int>("nExteriorElementBoundaries_global");
    xt::pyarray<int>    &exteriorElementBoundariesArray             = args.array<int>("exteriorElementBoundariesArray");
    xt::pyarray<int>    &elementBoundaryElementsArray               = args.array<int>("elementBoundaryElementsArray");
    xt::pyarray<int>    &elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
    xt::pyarray<double> &ebqe_velocity_ext                          = args.array<double>("ebqe_velocity_ext");
    xt::pyarray<int>    &isDOFBoundary_u                            = args.array<int>("isDOFBoundary_u");
    xt::pyarray<double> &ebqe_bc_u_ext                              = args.array<double>("ebqe_bc_u_ext");
    // component-1 (S_n) boundary arrays.
    xt::pyarray<int>    &isDOFBoundary_n                            = args.array<int>("isDOFBoundary_n");
    xt::pyarray<double> &ebqe_bc_u_n_ext                            = args.array<double>("ebqe_bc_u_n_ext");
    xt::pyarray<int>    &isFluxBoundary_u                           = args.array<int>("isFluxBoundary_u");
    xt::pyarray<double> &ebqe_bc_flux_ext                           = args.array<double>("ebqe_bc_flux_ext");
    xt::pyarray<int>    &csrColumnOffsets_eb_u_u                    = args.array<int>("csrColumnOffsets_eb_u_u");
    int                  LUMPED_MASS_MATRIX                         = args.scalar<int>("LUMPED_MASS_MATRIX");
    assert(a_rowptr.data()[nSpace] == nnz);
    assert(a_rowptr.data()[nSpace] == nSpace);
    double Ct_sge = 4.0;

    //
    //loop over elements to compute volume integrals and load them into the element Jacobians and global Jacobian
    //
    for (int eN = 0; eN < nElements_global; eN++) {
      double elementJacobian_u_u[nDOF_test_element][nDOF_trial_element];
      // (0,1) cross-block element storage.
      double elementJacobian_u_n[nDOF_test_element][nDOF_trial_element];
      for (int i = 0; i < nDOF_test_element; i++) {
        for (int j = 0; j < nDOF_trial_element; j++) {
          elementJacobian_u_u[i][j] = 0.0;
          elementJacobian_u_n[i][j] = 0.0;
        }
      }
      for (int k = 0; k < nQuadraturePoints_element; k++) {
        int eN_k                  = eN * nQuadraturePoints_element + k, //index to a scalar at a quadrature point
          eN_k_nSpace             = eN_k * nSpace,
            eN_nDOF_trial_element = eN * nDOF_trial_element; //index to a vector at a quadrature point

        //declare local storage
        double u = 0.0, grad_u[nSpace], m = 0.0, dm = 0.0, f[nSpace], df[nSpace], a[nnz], da[nnz], as[nnz], m_t = 0.0, dm_t = 0.0, dpdeResidual_u_u[nDOF_trial_element], Lstar_u[nDOF_test_element], dsubgridError_u_u[nDOF_trial_element], tau = 0.0, tau0 = 0.0, tau1 = 0.0, jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace], u_grad_trial[nDOF_trial_element * nSpace], dV, u_test_dV[nDOF_test_element], u_grad_test_dV[nDOF_test_element * nSpace], x, y, z, xt, yt, zt, G[nSpace * nSpace], G_dd_G, tr_G;
        //
        //calculate solution and gradients at quadrature points
        //
        //get jacobian, etc for mapping reference element
        ck.calculateMapping_element(eN, k, mesh_dof.data(), mesh_l2g.data(), mesh_trial_ref.data(), mesh_grad_trial_ref.data(), jac, jacDet, jacInv, x, y, z);
        ck.calculateMappingVelocity_element(eN, k, mesh_velocity_dof.data(), mesh_l2g.data(), mesh_trial_ref.data(), xt, yt, zt);
        //get the physical integration weight
        dV = fabs(jacDet) * dV_ref.data()[k];
        ck.calculateG(jacInv, G, G_dd_G, tr_G);
        //get the trial function gradients
        ck.gradTrialFromRef(&u_grad_trial_ref.data()[k * nDOF_trial_element * nSpace], jacInv, u_grad_trial);
        //get the solution
        ck.valFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], &u_trial_ref.data()[k * nDOF_trial_element], u);
        //get the solution gradients
        ck.gradFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], u_grad_trial, grad_u);
        //precalculate test function products with integration weights
        for (int j = 0; j < nDOF_trial_element; j++) {
          u_test_dV[j] = u_test_ref.data()[k * nDOF_trial_element + j] * dV;
          for (int I = 0; I < nSpace; I++) {
            u_grad_test_dV[j * nSpace + I] = u_grad_trial[j * nSpace + I] * dV; //cek warning won't work for Petrov-Galerkin
          }
        }
        //
        //calculate pde coefficients and derivatives at quadrature points
        //
        double Kr, dKr, thetaW;
        //const double rho_local = q_rho.data()[eN_k];

        double dm_du_n_qp = 0.0, dkr_du_n_qp = 0.0;
        double df_du_n_qp[nSpace];
        double da_du_n_qp[nnz];
        for (int I = 0; I < nSpace; I++) df_du_n_qp[I] = 0.0;
        for (int ii = 0; ii < nnz; ii++) da_du_n_qp[ii] = 0.0;
        double u_n_qp = 0.0;
        ck.valFromDOF(u_dof_n.data(),
                      &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_n_qp);
        evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, q_rho.data()[eN_k], beta, gravity.data(),
                                     alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]],
                                     thetaR.data()[elementMaterialTypes.data()[eN]], thetaSR.data()[elementMaterialTypes.data()[eN]],
                                     &KWs.data()[elementMaterialTypes.data()[eN] * nnz], u, u_n_qp,
                                     m, dm, dm_du_n_qp, f, df, df_du_n_qp, a, da, da_du_n_qp,
                                     as, Kr, dKr, dkr_du_n_qp, thetaW);
        //
        //calculate time derivatives
        //
        ck.bdf(alphaBDF, q_m_betaBDF.data()[eN_k], m, dm, m_t, dm_t);
        //
        //calculate subgrid error contribution to the Jacobian (strong residual, adjoint, jacobian of strong residual)
        //
        //calculate the adjoint times the test functions
        for (int i = 0; i < nDOF_test_element; i++) {
          int i_nSpace = i * nSpace;
          Lstar_u[i]   = ck.Advection_adjoint(df, &u_grad_test_dV[i_nSpace]);
        }
        //calculate the Jacobian of strong residual
        for (int j = 0; j < nDOF_trial_element; j++) {
          int j_nSpace        = j * nSpace;
          dpdeResidual_u_u[j] = ck.MassJacobian_strong(dm_t, u_trial_ref[k * nDOF_trial_element + j]) + ck.AdvectionJacobian_strong(df, &u_grad_trial[j_nSpace]);
        }
        //tau and tau*Res
        calculateSubgridError_tau(elementDiameter[eN], dm_t, df, cfl[eN_k], tau0);
        calculateSubgridError_tau(Ct_sge, G, dm_t, df, tau1, cfl[eN_k]);
        tau = useMetrics * tau1 + (1.0 - useMetrics) * tau0;
        for (int j = 0; j < nDOF_trial_element; j++) dsubgridError_u_u[j] = -tau * dpdeResidual_u_u[j];
        for (int i = 0; i < nDOF_test_element; i++) {
          for (int j = 0; j < nDOF_trial_element; j++) {
            int j_nSpace = j * nSpace;
            int i_nSpace = i * nSpace;
            elementJacobian_u_u[i][j] += ck.MassJacobian_weak(dm_t, u_trial_ref.data()[k * nDOF_trial_element + j], u_test_dV[i]) + ck.AdvectionJacobian_weak(df, u_trial_ref.data()[k * nDOF_trial_element + j], &u_grad_test_dV[i_nSpace]) +
                                         ck.DiffusionJacobian_weak(a_rowptr.data(), a_colind.data(), a, da, grad_u, &u_grad_test_dV[i_nSpace], 1.0, u_trial_ref.data()[k * nDOF_trial_element + j], &u_grad_trial[j_nSpace]) + VMS * ck.SubgridErrorJacobian(dsubgridError_u_u[j], Lstar_u[i]) + VMS * ck.NumericalDiffusionJacobian(q_numDiff_u_last[eN_k], &u_grad_trial[j_nSpace], &u_grad_test_dV[i_nSpace]);
          } //j
        } //i
        // (0,1) Jacobian assembly. d/du_n of the wetting-equation residual:
        //   mass:      alphaBDF * (dm/du_n) * trial_j(u_n) * test_i
        //   advection: (df/du_n) * trial_j(u_n) * grad N_i
        //   diffusion: (da/du_n) * grad u_w * trial_j(u_n) * grad N_i
        // (No "linear" diffusion term because varying u_n doesn't change grad u_w.)
        for (int i = 0; i < nDOF_test_element; i++) {
          const int i_nSpace = i * nSpace;
          double adv_sens_i = 0.0;
          double diff_sens_i = 0.0;
          for (int I = 0; I < nSpace; I++) {
            adv_sens_i += df_du_n_qp[I] * u_grad_test_dV[i_nSpace + I];
            for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I + 1]; ii++) {
              const int J = a_colind.data()[ii];
              diff_sens_i += da_du_n_qp[ii] * grad_u[J] * u_grad_test_dV[i_nSpace + I];
            }
          }
          for (int j = 0; j < nDOF_trial_element; j++) {
            const double trial_j = u_trial_ref.data()[k * nDOF_trial_element + j];
            elementJacobian_u_n[i][j] += alphaBDF * dm_du_n_qp * trial_j * u_test_dV[i];
            elementJacobian_u_n[i][j] += (adv_sens_i + diff_sens_i) * trial_j;
          }
        }
      } //k
      //
      //load into element Jacobian into global Jacobian
      //
      for (int i = 0; i < nDOF_test_element; i++) {
        int eN_i = eN * nDOF_test_element + i;
        for (int j = 0; j < nDOF_trial_element; j++) {
          int eN_i_j = eN_i * nDOF_trial_element + j;
          globalJacobian.data()[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_u_u[eN_i_j]] += elementJacobian_u_u[i][j];
          // load (0,1) cross-block coupling the wetting eq to S_n through
          // dm/du_n, df/du_n, da/du_n.
          globalJacobian.data()[csrRowIndeces_w_n.data()[eN_i] + csrColumnOffsets_w_n.data()[eN_i_j]] += elementJacobian_u_n[i][j];
        } //j
      } //i
    } //elements
    //
    //loop over exterior element boundaries to compute the surface integrals and load them into the global Jacobian
    //
    for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++) {
      int ebN = exteriorElementBoundariesArray.data()[ebNE];
      int eN = elementBoundaryElementsArray.data()[ebN * 2 + 0], ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN * 2 + 0], eN_nDOF_trial_element = eN * nDOF_trial_element;
      for (int kb = 0; kb < nQuadraturePoints_elementBoundary; kb++) {
        int ebNE_kb = ebNE * nQuadraturePoints_elementBoundary + kb, ebNE_kb_nSpace = ebNE_kb * nSpace, ebN_local_kb = ebN_local * nQuadraturePoints_elementBoundary + kb, ebN_local_kb_nSpace = ebN_local_kb * nSpace;

        double u_ext = 0.0, grad_u_ext[nSpace], m_ext = 0.0, dm_ext = 0.0, f_ext[nSpace], df_ext[nSpace], a_ext[nnz], da_ext[nnz], as_ext[nnz], dflux_u_u_ext = 0.0, bc_u_ext = 0.0,
               //bc_grad_u_ext[nSpace],
          bc_m_ext = 0.0, bc_dm_ext = 0.0, bc_f_ext[nSpace], bc_df_ext[nSpace], bc_a_ext[nnz], bc_da_ext[nnz], bc_as_ext[nnz], fluxJacobian_u_u[nDOF_trial_element], jac_ext[nSpace * nSpace], jacDet_ext, jacInv_ext[nSpace * nSpace], boundaryJac[nSpace * (nSpace - 1)], metricTensor[(nSpace - 1) * (nSpace - 1)], metricTensorDetSqrt, dS, u_test_dS[nDOF_test_element], u_grad_trial_trace[nDOF_trial_element * nSpace], normal[3], x_ext, y_ext, z_ext, xt_ext, yt_ext, zt_ext, integralScaling, G[nSpace * nSpace], G_dd_G, tr_G;
        //
        //calculate the solution and gradients at quadrature points
        //
        ck.calculateMapping_elementBoundary(eN, ebN_local, kb, ebN_local_kb, mesh_dof.data(), mesh_l2g.data(), mesh_trial_trace_ref.data(), mesh_grad_trial_trace_ref.data(), boundaryJac_ref.data(), jac_ext, jacDet_ext, jacInv_ext, boundaryJac, metricTensor, metricTensorDetSqrt,
                                            normal_ref.data(), normal, x_ext, y_ext, z_ext);
        ck.calculateMappingVelocity_elementBoundary(eN, ebN_local, kb, ebN_local_kb, mesh_velocity_dof.data(), mesh_l2g.data(), mesh_trial_trace_ref.data(), xt_ext, yt_ext, zt_ext, normal, boundaryJac, metricTensor, integralScaling);
        dS = ((1.0 - MOVING_DOMAIN) * metricTensorDetSqrt + MOVING_DOMAIN * integralScaling) * dS_ref.data()[kb];
        ck.calculateG(jacInv_ext, G, G_dd_G, tr_G);
        //compute shape and solution information
        //shape
        ck.gradTrialFromRef(&u_grad_trial_trace_ref.data()[ebN_local_kb_nSpace * nDOF_trial_element], jacInv_ext, u_grad_trial_trace);
        //solution and gradients
        ck.valFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], &u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element], u_ext);
        ck.gradFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], u_grad_trial_trace, grad_u_ext);
        //precalculate test function products with integration weights
        for (int j = 0; j < nDOF_trial_element; j++) { u_test_dS[j] = u_test_trace_ref.data()[ebN_local_kb * nDOF_test_element + j] * dS; }
        //
        //load the boundary values
        //
        bc_u_ext = isDOFBoundary_u.data()[ebNE_kb] * ebqe_bc_u_ext.data()[ebNE_kb] + (1 - isDOFBoundary_u.data()[ebNE_kb]) * u_ext;
        //
        //calculate the internal and external trace of the pde coefficients
        //
        double Kr, dKr, thetaW, thetaW_bc;
        const double rho_ext = ebqe_rho.data()[ebNE_kb];
        // Boundary closure for the wetting-equation Jacobian. (0,1) boundary
        // contribution is not assembled here.
        double dm_du_n_ext = 0.0, dkr_du_n_ext = 0.0;
        double df_du_n_ext[nSpace];
        double da_du_n_ext[nnz];
        double bc_dm_du_n = 0.0, bc_dkr_du_n = 0.0;
        double bc_df_du_n[nSpace];
        double bc_da_du_n[nnz];
        for (int I = 0; I < nSpace; I++) { df_du_n_ext[I] = 0.0; bc_df_du_n[I] = 0.0; }
        for (int ii = 0; ii < nnz; ii++)  { da_du_n_ext[ii] = 0.0; bc_da_du_n[ii] = 0.0; }
        // u_n_ext_qp & bc_u_n_ext_qp held at outer scope for the (0,1) boundary
        // Jacobian assembly below.
        double u_n_ext_qp_outer = 0.0;
        double bc_u_n_ext_qp_outer = 0.0;
        {
          double u_n_ext_qp = 0.0;
          ck.valFromDOF(u_dof_n.data(), &u_l2g.data()[eN_nDOF_trial_element],
                        &u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element], u_n_ext_qp);
          const double bc_u_n_ext_qp = isDOFBoundary_n.data()[ebNE_kb] * ebqe_bc_u_n_ext.data()[ebNE_kb]
                                     + (1 - isDOFBoundary_n.data()[ebNE_kb]) * u_n_ext_qp;
          u_n_ext_qp_outer    = u_n_ext_qp;
          bc_u_n_ext_qp_outer = bc_u_n_ext_qp;
          evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, rho_ext, beta, gravity.data(),
                                       alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]],
                                       thetaR.data()[elementMaterialTypes.data()[eN]], thetaSR.data()[elementMaterialTypes.data()[eN]],
                                       &KWs.data()[elementMaterialTypes.data()[eN] * nnz], u_ext, u_n_ext_qp,
                                       m_ext, dm_ext, dm_du_n_ext, f_ext, df_ext, df_du_n_ext, a_ext, da_ext, da_du_n_ext,
                                       as_ext, Kr, dKr, dkr_du_n_ext, thetaW);
          evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, rho_ext, beta, gravity.data(),
                                       alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]],
                                       thetaR.data()[elementMaterialTypes.data()[eN]], thetaSR.data()[elementMaterialTypes.data()[eN]],
                                       &KWs.data()[elementMaterialTypes.data()[eN] * nnz], bc_u_ext, bc_u_n_ext_qp,
                                       bc_m_ext, bc_dm_ext, bc_dm_du_n, bc_f_ext, bc_df_ext, bc_df_du_n, bc_a_ext, bc_da_ext, bc_da_du_n,
                                       bc_as_ext, Kr, dKr, bc_dkr_du_n, thetaW_bc);
        }
        //
        //calculate the flux jacobian
        //
        for (int j = 0; j < nDOF_trial_element; j++) {
          exteriorNumericalFluxJacobian(a_rowptr.data(), a_colind.data(), isDOFBoundary_u.data()[ebNE_kb], normal, a_ext, da_ext, grad_u_ext, &u_grad_trial_trace[j * nSpace], df_ext, u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element + j],
                                        ebqe_penalty_ext.data()[ebNE_kb], //penalty,
                                        fluxJacobian_u_u[j]);
        } //j
        // (0,1) boundary flux Jacobian. From exteriorNumericalFluxJacobian, the
        // parts of the flux that depend on u_n are df_ext (chain rule ->
        // df_du_n_ext * trial_j_n) and the a_ext * grad_u_ext term (chain rule
        // -> da_du_n_ext * trial_j_n * grad_u_ext). Penalty and linear
        // "a_ext * grad_v" pieces don't depend on u_n_j.
        double fluxJacobian_u_n[nDOF_trial_element];
        for (int j = 0; j < nDOF_trial_element; j++) fluxJacobian_u_n[j] = 0.0;
        if (isDOFBoundary_u.data()[ebNE_kb]) {
          double sens_n = 0.0;
          for (int I = 0; I < nSpace; I++) {
            sens_n += df_du_n_ext[I] * normal[I];
            for (int m = a_rowptr.data()[I]; m < a_rowptr.data()[I + 1]; m++) {
              sens_n -= da_du_n_ext[m] * grad_u_ext[a_colind.data()[m]] * normal[I];
            }
          }
          for (int j = 0; j < nDOF_trial_element; j++) {
            const double trial_j_n_bdry = u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element + j];
            fluxJacobian_u_n[j] = sens_n * trial_j_n_bdry;
          }
        }
        //
        //update the global Jacobian from the flux Jacobian
        //
        for (int i = 0; i < nDOF_test_element; i++) {
          int eN_i = eN * nDOF_test_element + i;
          for (int j = 0; j < nDOF_trial_element; j++) {
            int ebN_i_j = ebN * 4 * nDOF_test_X_trial_element + i * nDOF_trial_element + j;
            globalJacobian.data()[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_eb_u_u[ebN_i_j]] += fluxJacobian_u_u[j] * u_test_dS[i];
            // (0,1) cross-block boundary contribution (nonzero only on
            // Dirichlet faces for u_w).
            globalJacobian.data()[csrRowIndeces_w_n.data()[eN_i] + csrColumnOffsets_eb_w_n.data()[ebN_i_j]] += fluxJacobian_u_n[j] * u_test_dS[i];
          } //j
        } //i
      } //kb
    } //ebNE

    // ============================================================================
    // full component-1 (S_n) Jacobian, including the gas
    // Darcy flux-derivative terms.
    //
    //   m_n       = phi * rho_n * u_n             (u_n = S_n)
    //   a_n[ii]   = rho_n * k_rn(u_n) * (K/mu_n)[ii]
    //   f_n[I]    = rho_n^2 * k_rn(u_n) * (K/mu_n)[ii] * g[colind[ii]]
    //
    //   J_(1,1)[i,j] (mass)               = +phi*rho_n/dt * N_i N_j dV
    //   J_(1,1)[i,j] (advection sensitivity) = (df_n/du_n) . grad N_i * N_j dV
    //   J_(1,1)[i,j] (diffusion sensitivity) = (da_n/du_n) grad u_w . grad N_i * N_j dV
    //   J_(1,0)[i,j] (diffusion trial var.)  = a_n grad N_j . grad N_i dV
    //
    // (0,1) cross-block (wetting eq dependence on u_n) is assembled separately
    // in the element loop above.
    // ============================================================================
    for (int eN = 0; eN < nElements_global; eN++) {
      const int    mat_eN    = elementMaterialTypes.data()[eN];
      const double phi_eN    = thetaR.data()[mat_eN] + thetaSR.data()[mat_eN];
      const double alpha_eN  = alpha.data()[mat_eN];
      const double n_vg_eN   = n.data()[mat_eN];
      const double *KWs_eN   = &KWs.data()[mat_eN * nnz];
      const double dm_n_du_n = phi_eN * rho_n;       // d(phi*rho_n*S_n)/dS_n
      double elementJacobian_n_n[nDOF_test_element][nDOF_trial_element];
      double elementJacobian_n_w[nDOF_test_element][nDOF_trial_element];
      for (int i = 0; i < nDOF_test_element; i++)
        for (int j = 0; j < nDOF_trial_element; j++) {
          elementJacobian_n_n[i][j] = 0.0;
          elementJacobian_n_w[i][j] = 0.0;
        }
      for (int k = 0; k < nQuadraturePoints_element; k++) {
        const int eN_nDOF_trial_element = eN * nDOF_trial_element;
        double jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace], x_q, y_q, z_q;
        ck.calculateMapping_element(eN, k, mesh_dof.data(), mesh_l2g.data(),
                                    mesh_trial_ref.data(), mesh_grad_trial_ref.data(),
                                    jac, jacDet, jacInv, x_q, y_q, z_q);
        const double dV = std::fabs(jacDet) * dV_ref.data()[k];
        // Trial gradients in physical coords.
        double u_grad_trial_qp[nDOF_trial_element * nSpace];
        ck.gradTrialFromRef(&u_grad_trial_ref.data()[k * nDOF_trial_element * nSpace],
                            jacInv, u_grad_trial_qp);
        // Saturation u_n at QP.
        double u_n = 0.0;
        ck.valFromDOF(u_dof_n.data(),
                      &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_n);
        // Wetting-pressure gradient at QP.
        double grad_u_w[nSpace];
        ck.gradFromDOF(u_dof.data(),
                       &u_l2g.data()[eN_nDOF_trial_element],
                       u_grad_trial_qp, grad_u_w);
        // Step 3d: saturation gradient at QP for the capillary diffusion term.
        double grad_u_n[nSpace];
        ck.gradFromDOF(u_dof_n.data(),
                       &u_l2g.data()[eN_nDOF_trial_element],
                       u_grad_trial_qp, grad_u_n);
        // Non-wetting kr (and dkr/du_n for flux Jacobian terms).
        // u_n is S_n; closures take wetting effective saturation S_e.
        const double phi_loc      = thetaR.data()[mat_eN] + thetaSR.data()[mat_eN];
        const double S_wr_loc     = thetaR.data()[mat_eN] / phi_loc;
        const double one_m_Sr_loc = 1.0 - S_wr_loc;
        // Se = (1 - u_n - S_wr)/(1 - S_wr).
        const double Se_qp_raw    = (1.0 - u_n - S_wr_loc) / one_m_Sr_loc;
        double Se_qp, dSe_du_n_loc;
        if (Se_qp_raw <= 0.0)      { Se_qp = 0.0;        dSe_du_n_loc = 0.0; }
        else if (Se_qp_raw >= 1.0) { Se_qp = 1.0;        dSe_du_n_loc = 0.0; }
        else                       { Se_qp = Se_qp_raw;  dSe_du_n_loc = -1.0 / one_m_Sr_loc; }
        double KNr = 0.0, DKNr_DSe = 0.0;
        if (PSK_TYPE_member == 1) {
          proteus::mphase_co2::psk::bc_kr_nonwetting_from_Se(
              Se_qp, alpha_eN, n_vg_eN, KNr, DKNr_DSe);
        } else {
          proteus::mphase_co2::psk::vgm_kr_nonwetting_from_Se(
              Se_qp, alpha_eN, n_vg_eN, KNr, DKNr_DSe);
        }
        // Chain rule: convert d/dSe -> d/du_n.  dSe_du_n_loc carries the
        // sign flip from the S_n primary so DKNr_Du_n has the correct sign.
        DKNr_DSe *= dSe_du_n_loc;
        // Capillary derivative. Closure returns dp_c/dSe < 0 under monotone
        // closures. After the chain rule with dSe_du_n_loc < 0, dpc_dSw_local
        // is positive and represents dp_c/d(u_n) = dp_c/dS_n. The variable
        // name 'dpc_dSw' is kept for historical reasons; semantically it is
        // dp_c/d(u_n). d2pc_dSw2 stays sign-invariant (dSe/du_n squared).
        double pc_qp = 0.0, dpc_dSw = 0.0, d2pc_dSw2 = 0.0;
        if (PSK_TYPE_member == 1) {
          proteus::mphase_co2::psk::bc_pc_from_Se(Se_qp, alpha_eN, n_vg_eN, pc_qp, dpc_dSw, d2pc_dSw2);
        } else {
          proteus::mphase_co2::psk::vgm_pc_from_Se(Se_qp, alpha_eN, n_vg_eN, pc_qp, dpc_dSw, d2pc_dSw2);
        }
        dpc_dSw   *= dSe_du_n_loc;
        d2pc_dSw2 *= dSe_du_n_loc * dSe_du_n_loc;
        // Build a_n, f_n, a_n_p_c (used by trial-fn variation in J_(1,0) and
        // J_(1,1) p_c term) and da_n/du_n, df_n/du_n, da_n_p_c/du_n (used by
        // coefficient variation in J_(1,1)).
        double a_n[nnz], da_n_du_n[nnz];
        double a_n_p_c[nnz], da_n_p_c_du_n[nnz];
        double f_n[nSpace], df_n_du_n[nSpace];
        for (int I = 0; I < nSpace; I++) { f_n[I] = 0.0; df_n_du_n[I] = 0.0; }
        for (int I = 0; I < nSpace; I++) {
          for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I + 1]; ii++) {
            const int J = a_colind.data()[ii];
            a_n[ii]       = rho_n * KNr * KWs_eN[ii];
            da_n_du_n[ii] = rho_n * DKNr_DSe * KWs_eN[ii];
            // Capillary diffusion coefficient and its dependence on u_n through
            // both k_rn(u_n) and dp_c/d(u_n).
            a_n_p_c[ii]       = a_n[ii] * dpc_dSw;
            da_n_p_c_du_n[ii] = da_n_du_n[ii] * dpc_dSw + a_n[ii] * d2pc_dSw2;
            f_n[I]       += rho_n * rho_n * KNr * KWs_eN[ii] * gravity.data()[J];
            df_n_du_n[I] += rho_n * rho_n * DKNr_DSe * KWs_eN[ii] * gravity.data()[J];
          }
        }
        // Assemble per (i, j).
        for (int i = 0; i < nDOF_test_element; i++) {
          const double test_i = u_test_ref.data()[k * nDOF_test_element + i];
          // Precompute (da_n . grad u_w . grad N_i) for the (1,1) coeff sensitivity.
          double diff_coef_sens_i = 0.0;
          double cap_coef_sens_i  = 0.0;
          for (int I = 0; I < nSpace; I++) {
            const double grad_Ni_I = u_grad_trial_qp[i * nSpace + I];
            for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I + 1]; ii++) {
              const int J = a_colind.data()[ii];
              diff_coef_sens_i += da_n_du_n[ii]     * grad_u_w[J] * grad_Ni_I;
              cap_coef_sens_i  += da_n_p_c_du_n[ii] * grad_u_n[J] * grad_Ni_I;
            }
          }
          // Precompute (df_n . grad N_i) for the (1,1) gravity sensitivity.
          double adv_coef_sens_i = 0.0;
          for (int I = 0; I < nSpace; I++) {
            adv_coef_sens_i += df_n_du_n[I] * u_grad_trial_qp[i * nSpace + I];
          }
          for (int j = 0; j < nDOF_trial_element; j++) {
            const double trial_j = u_trial_ref.data()[k * nDOF_trial_element + j];
            // (1,1) mass term (Step 2 contribution).
            elementJacobian_n_n[i][j] += (dm_n_du_n * test_i * trial_j * dV) / dt_n;
            // (1,1) flux-coefficient sensitivities through k_rn(u_n) and dp_c/dS_w.
            elementJacobian_n_n[i][j] += (adv_coef_sens_i + diff_coef_sens_i + cap_coef_sens_i)
                                       * trial_j * dV;
            // (1,1) capillary diffusion trial-fn variation: a_n*dp_c/dS_w * grad N_j . grad N_i.
            double cap_trial_ij = 0.0;
            for (int I = 0; I < nSpace; I++) {
              const double grad_Ni_I = u_grad_trial_qp[i * nSpace + I];
              for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I + 1]; ii++) {
                const int J = a_colind.data()[ii];
                cap_trial_ij += a_n_p_c[ii] * u_grad_trial_qp[j * nSpace + J] * grad_Ni_I;
              }
            }
            elementJacobian_n_n[i][j] += cap_trial_ij * dV;
            // (1,0) cross-block: trial-function variation of -nabla.(a_n grad u_w).
            double diff_trial_ij = 0.0;
            for (int I = 0; I < nSpace; I++) {
              const double grad_Ni_I = u_grad_trial_qp[i * nSpace + I];
              for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I + 1]; ii++) {
                const int J = a_colind.data()[ii];
                diff_trial_ij += a_n[ii] * u_grad_trial_qp[j * nSpace + J] * grad_Ni_I;
              }
            }
            elementJacobian_n_w[i][j] += diff_trial_ij * dV;
          }
        }
      }
      for (int i = 0; i < nDOF_test_element; i++) {
        const int eN_i = eN * nDOF_test_element + i;
        for (int j = 0; j < nDOF_trial_element; j++) {
          const int eN_i_j = eN_i * nDOF_trial_element + j;
          globalJacobian.data()[csrRowIndeces_n_n.data()[eN_i] + csrColumnOffsets_n_n.data()[eN_i_j]]
              += elementJacobian_n_n[i][j];
          globalJacobian.data()[csrRowIndeces_n_w.data()[eN_i] + csrColumnOffsets_n_w.data()[eN_i_j]]
              += elementJacobian_n_w[i][j];
        }
      }
    }
  } //computeJacobian

 

  void FCTStep(arguments_dict &args)
{
  xt::pyarray<double> &bc_mask                   = args.array<double>("bc_mask");
  int                  NNZ                       = args.scalar<int>("NNZ");     // number of non-zero entries
  int                  numDOFs                   = args.scalar<int>("numDOFs"); // number of DOFs
  double               dt                        = args.scalar<double>("dt");
  xt::pyarray<double> &ML                        = args.array<double>("ML");    // lumped mass matrix (as vector)
  xt::pyarray<double> &mn                        = args.array<double>("mn");    // DOFs at time tn
  xt::pyarray<double> &mHigh                     = args.array<double>("mHigh"); // high-order mass at t^{n+1}
  xt::pyarray<double> &mLow                      = args.array<double>("mLow");  // low-order mass at t^{n+1}
  xt::pyarray<double> &mDotLow                   = args.array<double>("mDotLow");
  xt::pyarray<double> &limited_solution          = args.array<double>("limited_solution");
  xt::pyarray<int>    &csrRowIndeces_DofLoops    = args.array<int>("csrRowIndeces_DofLoops");
  xt::pyarray<int>    &csrColumnOffsets_DofLoops = args.array<int>("csrColumnOffsets_DofLoops");
  xt::pyarray<int>    &csrRowIndeces_Full        = args.array<int>("csrRowIndeces_Full");
  xt::pyarray<int>    &csrColumnOffsets_Full     = args.array<int>("csrColumnOffsets_Full");
  xt::pyarray<double> &MC                        = args.array<double>("MC");              // consistent mass matrix
  xt::pyarray<double> &dt_times_fH_minus_fL      = args.array<double>("dt_times_fH_minus_fL");
  xt::pyarray<double> &min_m_bc                  = args.array<double>("min_m_bc");
  xt::pyarray<double> &max_m_bc                  = args.array<double>("max_m_bc");
  xt::pyarray<double> &fluxCorrection            = args.array<double>("fluxCorrection");
  // flags
  int                  LUMPED_MASS_MATRIX        = args.scalar<int>("LUMPED_MASS_MATRIX");
  int                  MONOLITHIC                = args.scalar<int>("MONOLITHIC");
  const int            offset_u                  = args.scalar<int>("offset_u");
  const int            stride_u                  = args.scalar<int>("stride_u");

  // heap arrays instead of VLAs
  std::vector<double> Rpos(numDOFs, 0.0);
  std::vector<double> Rneg(numDOFs, 0.0);
  std::vector<double> FluxCorrectionMatrix(csrRowIndeces_DofLoops.at(numDOFs), 0.0);
  std::vector<double> mDot(numDOFs, 0.0);
  auto full_offset_from_compact = [&](int i_compact, int j_compact) -> int
  {
    const int full_i = offset_u + stride_u * i_compact;
    const int full_j = offset_u + stride_u * j_compact;
    for (int offset = csrRowIndeces_Full.at(full_i); offset < csrRowIndeces_Full.at(full_i + 1); ++offset)
      if (csrColumnOffsets_Full.at(offset) == full_j) return offset;
    return -1;
  };

  // for debugging bounds
  std::vector<double> localMin(numDOFs, 0.0);
  std::vector<double> localMax(numDOFs, 0.0);

  //////////////////
  // LOOP in DOFs //
  //////////////////
  int ij = 0;

 
  for (int i = 0; i < numDOFs; i++) {
 
    // local time derivative from low-order mass
    mDot.at(i) = (mLow.at(i) - mn.at(i)) / dt;

    // initialize local min/max from BC
    double mini = min_m_bc.at(i);
    double maxi = max_m_bc.at(i);

    double Pposi = 0.0, Pnegi = 0.0;

    // LOOP OVER THE SPARSITY PATTERN (j-LOOP)
    for (int offset = csrRowIndeces_DofLoops.at(i); offset < csrRowIndeces_DofLoops.at(i + 1);offset++)
    {
      int j = csrColumnOffsets_DofLoops.at(offset);
      const int full_offset = full_offset_from_compact(i, j);
      assert(full_offset >= 0);

      ////////////////////////
      // COMPUTE THE BOUNDS //
      ////////////////////////
      if (GLOBAL_FCT == 0) {
        if (MONOLITHIC == 0) {
          mini = fmin(mini, mLow.at(j));
          maxi = fmax(maxi, mLow.at(j));
        } else {
          mini = fmin(mini, mn.at(j));
          maxi = fmax(maxi, mn.at(j));
        }
      }

      mDot.at(j) = (mLow.at(j) - mn.at(j)) / dt;

      if (MONOLITHIC == 0) {
        FluxCorrectionMatrix.at(ij) = (LUMPED_MASS_MATRIX == 1 ? 0. : 1.) * dt * MC.at(full_offset) * (mDotLow.at(i) - mDotLow.at(j)) + dt_times_fH_minus_fL.at(full_offset);
      } else {
        FluxCorrectionMatrix.at(ij) = dt_times_fH_minus_fL.at(full_offset);
      }

      ///////////////////////
      // COMPUTE P VECTORS //
      ///////////////////////
      Pposi += FluxCorrectionMatrix.at(ij) * ((FluxCorrectionMatrix.at(ij) > 0) ? 1. : 0.);
      Pnegi += FluxCorrectionMatrix.at(ij) * ((FluxCorrectionMatrix.at(ij) < 0) ? 1. : 0.);

      // update ij
      ij += 1;
    } // j-loop

    ///////////////////////
    // COMPUTE Q VECTORS //
    ///////////////////////
    double gamma;
    double Qposi;
    double Qnegi;

    if (MONOLITHIC == 0) {
      Qposi = ML.at(i) * (maxi - mLow.at(i));
      Qnegi = ML.at(i) * (mini - mLow.at(i));
    } else {
      // cek todo: don't think this is right for Richards
      gamma = 10.0 * ML.at(i);
      Qposi = fmin(0.5 * ML.at(i) * (1.0 - mn.at(i)), gamma * (maxi - mn.at(i)));
      Qnegi = fmax(0.5 * ML.at(i) * (0.0 - mn.at(i)), gamma * (mini - mn.at(i)));
    }

    ///////////////////////
    // COMPUTE R VECTORS //
    ///////////////////////
    Rpos.at(i) = ((Pposi == 0.0) ? 1.0 : fmin(1.0, Qposi / Pposi));
    Rneg.at(i) = ((Pnegi == 0.0) ? 1.0 : fmin(1.0, Qnegi / Pnegi));

    // store local bounds for later bound check
    localMin.at(i) = mini;
    localMax.at(i) = maxi;
  } // i DOFs

  //////////////////////
  // COMPUTE LIMITERS //
  //////////////////////
  ij = 0;
 // std::cout << "FCT: entering second DOF loop (applying limiters)...\n";

  for (int i = 0; i < numDOFs; i++) {
    double ith_Limiter_times_FluxCorrectionMatrix = 0.0;
    double alpha_fA, alpha_dot, beta_ij = 1.0;
    // LOOP OVER THE SPARSITY PATTERN (j-LOOP)
    for (int offset = csrRowIndeces_DofLoops.at(i);  offset < csrRowIndeces_DofLoops.at(i + 1); offset++)
    {
      int j = csrColumnOffsets_DofLoops.at(offset);
      const int full_offset = full_offset_from_compact(i, j);
      assert(full_offset >= 0);
      alpha_fA = ((FluxCorrectionMatrix.at(ij) > 0.0) ? fmin(Rpos.at(i), Rneg.at(j)) : fmin(Rneg.at(i), Rpos.at(j))) * FluxCorrectionMatrix.at(ij);
      alpha_dot = fmin(1.0, beta_ij * fabs(alpha_fA) / MC.at(full_offset) / fmax(1.0e-8, fabs(mDot.at(i) - mDot.at(j))));

      if (MONOLITHIC == 0) {
        ith_Limiter_times_FluxCorrectionMatrix += alpha_fA;
      } else {
        ith_Limiter_times_FluxCorrectionMatrix += alpha_fA + (LUMPED_MASS_MATRIX == 1 ? 0. : 1.) * dt * alpha_dot * MC.at(full_offset) * (mDot.at(i) - mDot.at(j));
      }
      ij += 1;
    } // j-loop
    fluxCorrection.at(i) = -ith_Limiter_times_FluxCorrectionMatrix * bc_mask.at(i) / dt;
    limited_solution.at(i) = mLow.at(i) + 1.0 / ML.at(i) * ith_Limiter_times_FluxCorrectionMatrix * bc_mask.at(i);
}
}

  // ============================================================================
  // FCTStep_n(): Zalesak FCT limiter for the non-wetting equation (comp-1).
  //
  // Mirrors FCTStep() but operates on the comp-1 predictor state populated by
  // calculateResidual_entropy_viscosity:
  //   mLow_n[i]     -- low-order m_n at t^{n+1} (current Newton iterate)
  //   mn_n[i]       -- m_n at t^n
  //   mDotLow_n[i]  -- (mLow_n[i] - mn_n[i]) / dt
  //   dt_times_fH_minus_fL_n[ij]  -- per-edge antidiffusive flux
  //
  // Bounds: min_m_bc_n / max_m_bc_n carry boundary-aware values (Dirichlet
  // DOFs hold the imposed m_n from S_w_bc; interior DOFs start at the
  // +-1e10 sentinel set by Python and get shrunk by neighbor mLow_n values
  // in the standard Zalesak local-neighborhood loop -- mass-conservative).
  //
  // Outputs:
  //   fluxCorrection_n[i]   -- to be added to globalResidual at comp-1 rows
  //   limited_solution_n[i] -- mLow_n[i] + (limited correction) / ML_n[i],
  //                            the bound-preserving, mass-conservative m_n
  // ============================================================================
  void FCTStep_n(arguments_dict &args)
  {
    int                  numDOFs_n                   = args.scalar<int>("numDOFs_n");
    int                  NNZ_n                       = args.scalar<int>("NNZ_n");
    double               dt                          = args.scalar<double>("dt");
    xt::pyarray<double> &ML_n                        = args.array<double>("ML_n");
    xt::pyarray<double> &MC_n                        = args.array<double>("MC_n");
    xt::pyarray<double> &mn_n                        = args.array<double>("mn_n");
    xt::pyarray<double> &mLow_n                      = args.array<double>("mLow_n");
    xt::pyarray<double> &mDotLow_n                   = args.array<double>("mDotLow_n");
    xt::pyarray<double> &dt_times_fH_minus_fL_n      = args.array<double>("dt_times_fH_minus_fL_n");
    xt::pyarray<double> &min_m_bc_n                  = args.array<double>("min_m_bc_n");
    xt::pyarray<double> &max_m_bc_n                  = args.array<double>("max_m_bc_n");
    xt::pyarray<double> &fluxCorrection_n            = args.array<double>("fluxCorrection_n");
    xt::pyarray<double> &limited_solution_n          = args.array<double>("limited_solution_n");
    xt::pyarray<double> &bc_mask_n                   = args.array<double>("bc_mask_n");
    xt::pyarray<int>    &csrRowIndeces_n_DofLoops    = args.array<int>("csrRowIndeces_n_DofLoops");
    xt::pyarray<int>    &csrColumnOffsets_n_DofLoops = args.array<int>("csrColumnOffsets_n_DofLoops");
    // comp1_full_offsets[k] = offset into globalJacobian's full CSR for the
    // k-th entry of the compact comp-1 CSR; used by FCTStep_n to look up the
    // consistent-mass matrix MC at the right edge.
    xt::pyarray<int>    &comp1_full_offsets          = args.array<int>("comp1_full_offsets");
    int                  LUMPED_MASS_MATRIX          = args.scalar<int>("LUMPED_MASS_MATRIX");

    std::vector<double> Rpos(numDOFs_n, 0.0);
    std::vector<double> Rneg(numDOFs_n, 0.0);
    std::vector<double> FluxCorrectionMatrix(csrRowIndeces_n_DofLoops.at(numDOFs_n), 0.0);

    // --------- Pass 1: per-DOF bounds + Pposi / Pnegi accumulation. ---------
    int ij = 0;
    for (int i = 0; i < numDOFs_n; i++) {
      double mini = min_m_bc_n.at(i);
      double maxi = max_m_bc_n.at(i);
      double Pposi = 0.0, Pnegi = 0.0;
      for (int offset = csrRowIndeces_n_DofLoops.at(i);
           offset < csrRowIndeces_n_DofLoops.at(i + 1); offset++) {
        const int j = csrColumnOffsets_n_DofLoops.at(offset);
        if (GLOBAL_FCT == 0) {
          mini = std::fmin(mini, mLow_n.at(j));
          maxi = std::fmax(maxi, mLow_n.at(j));
        }
        const int full_off = comp1_full_offsets.at(ij);
        // FluxCorrectionMatrix[ij] = dt * MC * (mDotLow_i - mDotLow_j)
        //                          + dt_times_fH_minus_fL_n[ij]
        // (matches comp-0 FCTStep; consistency term skipped under lumped mass)
        FluxCorrectionMatrix.at(ij) =
            (LUMPED_MASS_MATRIX == 1 ? 0.0 : 1.0)
              * dt * MC_n.at(full_off) * (mDotLow_n.at(i) - mDotLow_n.at(j))
            + dt_times_fH_minus_fL_n.at(offset);
        Pposi += (FluxCorrectionMatrix.at(ij) > 0.0) ? FluxCorrectionMatrix.at(ij) : 0.0;
        Pnegi += (FluxCorrectionMatrix.at(ij) < 0.0) ? FluxCorrectionMatrix.at(ij) : 0.0;
        ij += 1;
      }
      const double Qposi = ML_n.at(i) * (maxi - mLow_n.at(i));
      const double Qnegi = ML_n.at(i) * (mini - mLow_n.at(i));
      Rpos.at(i) = (Pposi == 0.0) ? 1.0 : std::fmin(1.0, Qposi / Pposi);
      Rneg.at(i) = (Pnegi == 0.0) ? 1.0 : std::fmin(1.0, Qnegi / Pnegi);
    }

    // --------- Pass 2: per-DOF limited antidiffusive correction. ---------
    ij = 0;
    for (int i = 0; i < numDOFs_n; i++) {
      double ith_Limited_FCM = 0.0;
      for (int offset = csrRowIndeces_n_DofLoops.at(i);
           offset < csrRowIndeces_n_DofLoops.at(i + 1); offset++) {
        const int j = csrColumnOffsets_n_DofLoops.at(offset);
        // alpha_ij = min(R+_i, R-_j) if f_ij > 0, else min(R-_i, R+_j).
        // Symmetric in (i,j) -> mass-conservative.
        const double alpha_fA =
            ((FluxCorrectionMatrix.at(ij) > 0.0)
                 ? std::fmin(Rpos.at(i), Rneg.at(j))
                 : std::fmin(Rneg.at(i), Rpos.at(j)))
            * FluxCorrectionMatrix.at(ij);
        ith_Limited_FCM += alpha_fA;
        ij += 1;
      }
      fluxCorrection_n.at(i)   = -ith_Limited_FCM * bc_mask_n.at(i) / dt;
      limited_solution_n.at(i) = mLow_n.at(i)
                              + (1.0 / ML_n.at(i)) * ith_Limited_FCM * bc_mask_n.at(i);
    }
  }


  void kth_FCT_step(arguments_dict &args)
  {
    int                  NNZ                       = args.scalar<int>("NNZ");     //number on non-zero entries on sparsity pattern
    int                  numDOFs                   = args.scalar<int>("numDOFs"); //number of DOFs
    int                  num_fct_iter              = args.scalar<int>("num_fct_iter");
    double               dt                        = args.scalar<double>("dt");
    xt::pyarray<double> &lumped_mass_matrix        = args.array<double>("lumped_mass_matrix"); //lumped mass matrix (as vector)
    xt::pyarray<double> &soln                      = args.array<double>("soln");               //DOFs of solution at time tn
    xt::pyarray<double> &pn                        = args.array<double>("pn");                 //DOFs of solution at time tn
    xt::pyarray<double> &solH                      = args.array<double>("solH");               //DOFs of high order solution at tnp1
    xt::pyarray<double> &uLow                      = args.array<double>("uLow");
    xt::pyarray<double> &uDotLow                   = args.array<double>("uDotLow");
    xt::pyarray<double> &dLow                      = args.array<double>("dLow");
    xt::pyarray<double> &solLim                    = args.array<double>("limited_solution");
    xt::pyarray<double> &MC                        = args.array<double>("MC");
    xt::pyarray<double> &ML                        = args.array<double>("ML");
    xt::pyarray<double> &FluxMatrix                = args.array<double>("FluxMatrix");
    xt::pyarray<double> &limitedFlux               = args.array<double>("limited_Flux");
    xt::pyarray<int>    &csrRowIndeces_DofLoops    = args.array<int>("csrRowIndeces_DofLoops");    //csr row indeces
    xt::pyarray<int>    &csrColumnOffsets_DofLoops = args.array<int>("csrColumnOffsets_DofLoops"); //csr column offsets
    xt::pyarray<double> &MassMatrix                = args.array<double>("MassMatrix");             //mass matrix
    xt::pyarray<double> &dt_times_fH_minus_fL      = args.array<double>("dt_times_fH_minus_fL");   //low minus high order dissipative matrices
    xt::pyarray<double> &min_m_bc                  = args.array<double>("min_m_bc");               //min/max value at BCs. If DOF is not at boundary then min=1E10, max=-1E10
    xt::pyarray<double> &max_m_bc                  = args.array<double>("max_m_bc");
    int                  LUMPED_MASS_MATRIX        = args.scalar<int>("LUMPED_MASS_MATRIX");
    int                  MONOLITHIC                = args.scalar<int>("MONOLITHIC");
    double               Rpos[numDOFs], Rneg[numDOFs];
    int                  ij = 0;

    //////////////////////////////////////////////////////
    // ********** COMPUTE LOW ORDER SOLUTION ********** //
    //////////////////////////////////////////////////////
    if (num_fct_iter == 0) { // No FCT for global bounds
      for (int i = 0; i < numDOFs; i++) { solLim.data()[i] = uLow.data()[i]; }
    } else // do FCT iterations (with global bounds) on low order solution
    {
      for (int iter = 0; iter < num_fct_iter; iter++) {
        ij = 0;
        for (int i = 0; i < numDOFs; i++) {
          double maxi = 1.0, Pposi = 0;
          for (int offset = csrRowIndeces_DofLoops.data()[i]; offset < csrRowIndeces_DofLoops.data()[i + 1]; offset++) {
            int j = csrColumnOffsets_DofLoops.data()[offset];
            // compute Flux correction
            double Fluxij = FluxMatrix.data()[ij] - limitedFlux.data()[ij];
            Pposi += Fluxij * ((Fluxij > 0) ? 1. : 0.);
            // update ij
            ij += 1;
          }
          // compute Q vectors
          double mi      = ML.data()[i];
          double solLimi = solLim.data()[i];
          double Qposi   = mi * (maxi - solLimi);
          // compute R vectors
          Rpos[i] = ((Pposi == 0) ? 1. : fmin(1.0, Qposi / Pposi));
        }
        ij = 0;
        for (int i = 0; i < numDOFs; i++) {
          double ith_Limiter_times_FluxCorrectionMatrix = 0.;
          double Rposi                                  = Rpos[i];
          for (int offset = csrRowIndeces_DofLoops.data()[i]; offset < csrRowIndeces_DofLoops.data()[i + 1]; offset++) {
            int j = csrColumnOffsets_DofLoops.data()[offset];
            // Flux Correction
            double Fluxij = FluxMatrix.data()[ij] - limitedFlux.data()[ij];
            // compute limiter
            double Lij = 1.0;
            Lij        = (Fluxij > 0 ? Rposi : Rpos[j]);
            // compute limited flux
            ith_Limiter_times_FluxCorrectionMatrix += Lij * Fluxij;

            // update limited flux
            limitedFlux.data()[ij] = Lij * Fluxij;

            //update FluxMatrix
            FluxMatrix.data()[ij] = Fluxij;

            //update ij
            ij += 1;
          }
          //update limited solution
          double mi = ML.data()[i];
        }
      }
    }

    // ***************************************** //
    // ********** HIGH ORDER SOLUTION ********** //
    // ***************************************** //
    ij = 0;
    for (int i = 0; i < numDOFs; i++) {
      double mini = soln.data()[i], maxi = soln.data()[i];
      double Pposi = 0, Pnegi = 0.;
      for (int offset = csrRowIndeces_DofLoops.data()[i]; offset < csrRowIndeces_DofLoops.data()[i + 1]; offset++) {
        int j = csrColumnOffsets_DofLoops.data()[offset];
        // compute local bounds //
        mini = fmin(mini, soln.data()[j]);
        maxi = fmax(maxi, soln.data()[j]);
        // compute P vectors //
        double fij = (MC.data()[ij] * (uDotLow.data()[i] - uDotLow.data()[j]) / dt + dLow.data()[ij] * (uLow.data()[i] - uLow.data()[j]));
        Pposi += fij * (fij > 0 ? 1. : 0.);
        Pnegi += fij * (fij < 0 ? 1. : 0.);
        //update ij
        ij += 1;
      }
      // compute Q vectors //
      double mi    = ML.data()[i];
      double Qposi = mi * (maxi - solLim.data()[i]);
      double Qnegi = mi * (mini - solLim.data()[i]);
      // compute R vectors //
      Rpos[i] = ((Pposi == 0) ? 1. : fmin(1.0, Qposi / Pposi));
      Rneg[i] = ((Pnegi == 0) ? 1. : fmin(1.0, Qnegi / Pnegi));
    }

    // COMPUTE LIMITERS //
    ij = 0;
    for (int i = 0; i < numDOFs; i++) {
      double ith_limited_flux_correction = 0;
      double Rposi                       = Rpos[i];
      double Rnegi                       = Rneg[i];
      for (int offset = csrRowIndeces_DofLoops.data()[i]; offset < csrRowIndeces_DofLoops.data()[i + 1]; offset++) {
        int j = csrColumnOffsets_DofLoops.data()[offset];
        // compute flux correction
        double fij = (MC.data()[ij] * (uDotLow.data()[i] - uDotLow.data()[j]) / dt + dLow.data()[ij] * (uLow.data()[i] - uLow.data()[j]));

        // compute limiters
        double Lij = 1.0;
        Lij        = fij > 0 ? fmin(Rposi, Rneg[j]) : fmin(Rnegi, Rpos[j]);
        // compute ith_limited_flux_correction
        ith_limited_flux_correction += Lij * fij;
        ij += 1;
      }
      double mi = ML.data()[i];
      solLim[i] += 1. / mi * ith_limited_flux_correction;
    }
  }

  void calculateResidual_entropy_viscosity(arguments_dict &args)
  {
    xt::pyarray<double> &globalJacobian            = args.array<double>("globalJacobian");
    double               Theta                     = args.scalar<double>("Theta");
    double               Theta_h                   = args.scalar<double>("Theta_h");
    xt::pyarray<double> &bc_mask                   = args.array<double>("bc_mask");
    double               dt                        = args.scalar<double>("dt");
    xt::pyarray<double> &mesh_trial_ref            = args.array<double>("mesh_trial_ref");
    xt::pyarray<double> &mesh_grad_trial_ref       = args.array<double>("mesh_grad_trial_ref");
    xt::pyarray<double> &mesh_dof                  = args.array<double>("mesh_dof");
    xt::pyarray<double> &mesh_velocity_dof         = args.array<double>("mesh_velocity_dof");
    double               MOVING_DOMAIN             = args.scalar<double>("MOVING_DOMAIN");
    xt::pyarray<int>    &mesh_l2g                  = args.array<int>("mesh_l2g");
    xt::pyarray<double> &dV_ref                    = args.array<double>("dV_ref");
    xt::pyarray<double> &u_trial_ref               = args.array<double>("u_trial_ref");
    xt::pyarray<double> &u_grad_trial_ref          = args.array<double>("u_grad_trial_ref");
    xt::pyarray<double> &u_test_ref                = args.array<double>("u_test_ref");
    xt::pyarray<double> &u_grad_test_ref           = args.array<double>("u_grad_test_ref");
    xt::pyarray<double> &mesh_trial_trace_ref      = args.array<double>("mesh_trial_trace_ref");
    xt::pyarray<double> &mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
    xt::pyarray<double> &dS_ref                    = args.array<double>("dS_ref");
    xt::pyarray<double> &u_trial_trace_ref         = args.array<double>("u_trial_trace_ref");

    xt::pyarray<double> &u_grad_trial_trace_ref                     = args.array<double>("u_grad_trial_trace_ref");
    xt::pyarray<double> &u_test_trace_ref                           = args.array<double>("u_test_trace_ref");
    xt::pyarray<double> &u_grad_test_trace_ref                      = args.array<double>("u_grad_test_trace_ref");
    xt::pyarray<double> &normal_ref                                 = args.array<double>("normal_ref");
    xt::pyarray<double> &boundaryJac_ref                            = args.array<double>("boundaryJac_ref");
    int                  nElements_global                           = args.scalar<int>("nElements_global");
    xt::pyarray<double> &ebqe_penalty_ext                           = args.array<double>("ebqe_penalty_ext");
    xt::pyarray<int>    &elementMaterialTypes                       = args.array<int>("elementMaterialTypes");
    xt::pyarray<int>    &isSeepageFace                              = args.array<int>("isSeepageFace");
    xt::pyarray<int>    &a_rowptr                                   = args.array<int>("a_rowptr");
    xt::pyarray<int>    &a_colind                                   = args.array<int>("a_colind");
    double               rho                                        = args.scalar<double>("rho");
    double               beta                                       = args.scalar<double>("beta");
    //////////////////////////////Density Coupling ///////////////////////////////
    xt::pyarray<double> &q_rho                                     = args.array<double>("q_rho");
    xt::pyarray<double> &ebqe_rho                                  = args.array<double>("ebqe_rho");
    ////////////////////////////////////////////////////////////////////////////


    xt::pyarray<double> &gravity                                    = args.array<double>("gravity");
    xt::pyarray<double> &alpha                                      = args.array<double>("alpha");
    xt::pyarray<double> &n                                          = args.array<double>("n");
    xt::pyarray<double> &thetaR                                     = args.array<double>("thetaR");
    xt::pyarray<double> &thetaSR                                    = args.array<double>("thetaSR");
    xt::pyarray<double> &KWs                                        = args.array<double>("KWs");
    double               useMetrics                                 = args.scalar<double>("useMetrics");
    double               alphaBDF                                   = args.scalar<double>("alphaBDF");
    int                  lag_shockCapturing                         = args.scalar<int>("lag_shockCapturing");
    double               shockCapturingDiffusion                    = args.scalar<double>("shockCapturingDiffusion");
    double               sc_uref                                    = args.scalar<double>("sc_uref");
    double               sc_alpha                                   = args.scalar<double>("sc_alpha");
    xt::pyarray<int>    &u_l2g                                      = args.array<int>("u_l2g");
    xt::pyarray<int>    &r_l2g                                      = args.array<int>("r_l2g");
    xt::pyarray<double> &elementDiameter                            = args.array<double>("elementDiameter");
    int                  degree_polynomial                          = args.scalar<int>("degree_polynomial");
    xt::pyarray<double> &u_dof                                      = args.array<double>("u_dof");
    xt::pyarray<double> &u_dof_old                                  = args.array<double>("u_dof_old");
    xt::pyarray<double> &velocity                                   = args.array<double>("velocity");
    xt::pyarray<double> &q_m                                        = args.array<double>("q_m");
    xt::pyarray<double> &q_theta                                    = args.array<double>("q_theta");
    xt::pyarray<double> &q_u                                        = args.array<double>("q_u");
    xt::pyarray<double> &q_dV                                       = args.array<double>("q_dV");
    xt::pyarray<double> &q_m_betaBDF                                = args.array<double>("q_m_betaBDF");
    xt::pyarray<double> &cfl                                        = args.array<double>("cfl");
    xt::pyarray<double> &q_numDiff_u                                = args.array<double>("q_numDiff_u");
    xt::pyarray<double> &q_numDiff_u_last                           = args.array<double>("q_numDiff_u_last");
    int                  offset_u                                   = args.scalar<int>("offset_u");
    int                  stride_u                                   = args.scalar<int>("stride_u");
    // component-1 (S_n) mass equation args.
    // (dt is already declared at the top of this function for the EV path.)
    xt::pyarray<double> &u_dof_n                                    = args.array<double>("u_dof_n");
    xt::pyarray<double> &u_dof_n_old                                = args.array<double>("u_dof_n_old");
    // gas-phase density (constant for now). Will become
    // ρ_n(p_n) once Step 3 turns on real two-phase coupling.
    const double         rho_n                                      = args.scalar<double>("rho_n");
    const int            offset_n                                   = args.scalar<int>("offset_n");
    const int            stride_n                                   = args.scalar<int>("stride_n");
    xt::pyarray<double> &globalResidual                             = args.array<double>("globalResidual");
    int                  nExteriorElementBoundaries_global          = args.scalar<int>("nExteriorElementBoundaries_global");
    xt::pyarray<int>    &exteriorElementBoundariesArray             = args.array<int>("exteriorElementBoundariesArray");
    xt::pyarray<int>    &elementBoundaryElementsArray               = args.array<int>("elementBoundaryElementsArray");
    xt::pyarray<int>    &elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
    xt::pyarray<double> &ebqe_velocity_ext                          = args.array<double>("ebqe_velocity_ext");
    xt::pyarray<int>    &isDOFBoundary_u                            = args.array<int>("isDOFBoundary_u");
    xt::pyarray<double> &ebqe_bc_u_ext                              = args.array<double>("ebqe_bc_u_ext");
    // component-1 (S_n) boundary arrays.
    xt::pyarray<int>    &isDOFBoundary_n                            = args.array<int>("isDOFBoundary_n");
    xt::pyarray<double> &ebqe_bc_u_n_ext                            = args.array<double>("ebqe_bc_u_n_ext");
    xt::pyarray<int>    &isFluxBoundary_u                           = args.array<int>("isFluxBoundary_u");
    xt::pyarray<double> &ebqe_bc_flux_ext                           = args.array<double>("ebqe_bc_flux_ext");
    xt::pyarray<double> &ebqe_phi                                   = args.array<double>("ebqe_phi");
    double               epsFact                                    = args.scalar<double>("epsFact");
    xt::pyarray<double> &ebqe_u                                     = args.array<double>("ebqe_u");
    xt::pyarray<double> &ebqe_theta                                 = args.array<double>("ebqe_theta");
    xt::pyarray<double> &ebqe_flux                                  = args.array<double>("ebqe_flux");
    // PARAMETERS FOR EDGE BASED STABILIZATION
    double cE = args.scalar<double>("cE");
    double cK = args.scalar<double>("cK");
    // PARAMETERS FOR LOG BASED ENTROPY FUNCTION
    double uL = args.scalar<double>("uL");
    double uR = args.scalar<double>("uR");
    // PARAMETERS FOR EDGE VISCOSITY
    int               numDOFs                       = args.scalar<int>("numDOFs");
    // numDOFs is the compact component-0 free-DOF count used by the stabilized
    // DOF loops. Full matrix slots are recovered from the interleaved global
    // CSR using offset/stride-aware indexing.
    int               numDOFs_u                     = args.scalar<int>("numDOFs_u");
    int               NNZ                           = args.scalar<int>("NNZ");
    xt::pyarray<int> &csrRowIndeces_DofLoops        = args.array<int>("csrRowIndeces_DofLoops");
    xt::pyarray<int> &csrColumnOffsets_DofLoops     = args.array<int>("csrColumnOffsets_DofLoops");
    xt::pyarray<int> &csrRowIndeces_Full            = args.array<int>("csrRowIndeces_Full");
    xt::pyarray<int> &csrColumnOffsets_Full         = args.array<int>("csrColumnOffsets_Full");
    xt::pyarray<int> &csrRowIndeces_CellLoops       = args.array<int>("csrRowIndeces_CellLoops");
    xt::pyarray<int> &csrColumnOffsets_CellLoops    = args.array<int>("csrColumnOffsets_CellLoops");
    xt::pyarray<int> &csrColumnOffsets_eb_CellLoops = args.array<int>("csrColumnOffsets_eb_CellLoops");
    // C matrices
    xt::pyarray<double> &Cx  = args.array<double>("Cx");
    xt::pyarray<double> &Cy  = args.array<double>("Cy");
    xt::pyarray<double> &Cz  = args.array<double>("Cz");
    xt::pyarray<double> &CTx = args.array<double>("CTx");
    xt::pyarray<double> &CTy = args.array<double>("CTy");
    xt::pyarray<double> &CTz = args.array<double>("CTz");
    xt::pyarray<double> &ML  = args.array<double>("ML");
    xt::pyarray<double> &MC  = args.array<double>("MC");

    xt::pyarray<double> &delta_x_ij = args.array<double>("delta_x_ij");
    // PARAMETERS FOR 1st or 2nd ORDER MPP METHOD
    int LUMPED_MASS_MATRIX = args.scalar<int>("LUMPED_MASS_MATRIX");
    STABILIZATION STABILIZATION_TYPE{static_cast<STABILIZATION>(args.scalar<int>("STABILIZATION_TYPE"))};

    int ENTROPY_TYPE = args.scalar<int>("ENTROPY_TYPE");
    PSK_TYPE_member = args.scalar<int>("PSK_TYPE");
    // FOR FCT
    xt::pyarray<double> &dLow                 = args.array<double>("dLow");
    xt::pyarray<double> &fluxMatrix           = args.array<double>("fluxMatrix");
    xt::pyarray<double> &mDotLow              = args.array<double>("mDotLow");
    xt::pyarray<double> &mLow                 = args.array<double>("mLow");
    xt::pyarray<double> &dt_times_fH_minus_fL = args.array<double>("dt_times_fH_minus_fL");
    xt::pyarray<double> &min_m_bc             = args.array<double>("min_m_bc");
    xt::pyarray<double> &max_m_bc             = args.array<double>("max_m_bc");
    // AUX QUANTITIES OF INTEREST
    xt::pyarray<double> &quantDOFs = args.array<double>("quantDOFs");
    xt::pyarray<double> &mn        = args.array<double>("mn");
    xt::pyarray<double> &fluxCorrection        = args.array<double>("fluxCorrection");
    xt::pyarray<double> &limited_solution          = args.array<double>("limited_solution");
    xt::pyarray<int>    &freeDOFMaterialTypes      = args.array<int>("freeDOFMaterialTypes");
    xt::pyarray<int>    &freeDOFToNode_u           = args.array<int>("freeDOFToNode_u");

    xt::pyarray<double> &velocity_couple               = args.array<double>("velocity_couple");
    xt::pyarray<double> &ebqe_velocity_ext_couple      = args.array<double>("ebqe_velocity_ext_couple");
    // xt::pyarray<double> &q_x    = args.array<double>("q_x");
    // xt::pyarray<double> &ebqe_x = args.array<double>("ebqe_x");
    
    xt::pyarray<double> &anb_seepage_flux_n = args.array<double>("anb_seepage_flux_n");
    xt::pyarray<double> &q_velocity = args.array<double>("q_velocity");
    double &anb_seepage_flux(args.scalar<double>("anb_seepage_flux"));
    anb_seepage_flux = 0.0;
    xt::pyarray<int>    &csrRowIndeces_u_u                          = args.array<int>("csrRowIndeces_u_u");
    xt::pyarray<int>    &csrColumnOffsets_u_u                       = args.array<int>("csrColumnOffsets_u_u");
    xt::pyarray<int>    &csrColumnOffsets_eb_u_u                    = args.array<int>("csrColumnOffsets_eb_u_u");
    // CSR maps for the (1,1) Jacobian block. Used by the
    // dedicated component-1 element loop appended at the end to assemble
    // the gas-side mass-matrix Jacobian / dt.
    xt::pyarray<int>    &csrRowIndeces_n_n                          = args.array<int>("csrRowIndeces_n_n");
    // (1,0) cross-block CSR maps for the gas-eq diffusion
    // against grad u_w. Currently allocated by the framework but unused by
    // the C++ assembly - 3c.2/3c.3 will write into them.
    xt::pyarray<int>    &csrRowIndeces_n_w                          = args.array<int>("csrRowIndeces_n_w");
    xt::pyarray<int>    &csrColumnOffsets_n_n                       = args.array<int>("csrColumnOffsets_n_n");
    xt::pyarray<int>    &csrColumnOffsets_n_w                       = args.array<int>("csrColumnOffsets_n_w");
    // Comp-1 boundary CSR maps used by the exterior boundary loop appended
    // at the end of this routine.
    xt::pyarray<int>    &csrColumnOffsets_eb_n_n                    = args.array<int>("csrColumnOffsets_eb_n_n");
    xt::pyarray<int>    &csrColumnOffsets_eb_n_w                    = args.array<int>("csrColumnOffsets_eb_n_w");
    // COMPONENT-1 (m_n = phi*rho_n*u_n) EV plumbing. Mirrors the comp-0
    // EV scaffolding: a compact DOF graph (csrRowIndeces_n_DofLoops), per-edge
    // dLow_n / dEV_n storage, and per-DOF mLow_n / mDotLow_n. Sensor / sensor
    // bounds (u_n_L, u_n_R) operate on S_n; the stabilization itself acts on
    // m_n through chain rule dm_n/du_n = -phi*rho_n.
    int                  numDOFs_n                  = args.scalar<int>("numDOFs_n");
    int                  NNZ_n                      = args.scalar<int>("NNZ_n");
    xt::pyarray<int>    &csrRowIndeces_n_DofLoops    = args.array<int>("csrRowIndeces_n_DofLoops");
    xt::pyarray<int>    &csrColumnOffsets_n_DofLoops = args.array<int>("csrColumnOffsets_n_DofLoops");
    xt::pyarray<double> &dLow_n                     = args.array<double>("dLow_n");
    xt::pyarray<double> &dEV_n                      = args.array<double>("dEV_n");
    xt::pyarray<double> &fluxMatrix_n               = args.array<double>("fluxMatrix_n");
    xt::pyarray<double> &mLow_n                     = args.array<double>("mLow_n");
    xt::pyarray<double> &mDotLow_n                  = args.array<double>("mDotLow_n");
    double               u_n_L                       = args.scalar<double>("u_n_L");
    double               u_n_R                       = args.scalar<double>("u_n_R");
    xt::pyarray<double> &mn_n        = args.array<double>("mn_n");           // m_n at t^n (numDOFs_u)
    xt::pyarray<double> &quantDOFs_n = args.array<double>("quantDOFs_n");     // sensor scratch (numDOFs_u)
    // Comp-1 FCT plumbing read here so the gate at the end of this routine
    // can call FCTStep_n with all args present.
    xt::pyarray<double> &dt_times_fH_minus_fL_n = args.array<double>("dt_times_fH_minus_fL_n");
    xt::pyarray<double> &fluxCorrection_n       = args.array<double>("fluxCorrection_n");
    int                  FCT_n                  = args.scalar<int>("FCT_n");
    // double Rpos[numDOFs], Rneg[numDOFs];
     std::vector<double> Rpos(numDOFs, 0.0), Rneg(numDOFs, 0.0);
     std::vector<double> TransportMatrix(NNZ, 0.0),
                        TransportMatrixConsistent(NNZ, 0.0),
                        TransportMatrixn(NNZ, 0.0),
                        TransportMatrixConsistentn(NNZ, 0.0);
    //double FluxCorrectionMatrix[NNZ];
    // NOTE: This function follows a different (but equivalent) implementation of the smoothness based indicator than NCLS.h
    // Allocate space for the transport matrices
    // This is used for first order KUZMIN'S METHOD
    // double                TransportMatrix[NNZ], TransportMatrixConsistent[NNZ];
    // double                TransportMatrixn[NNZ], TransportMatrixConsistentn[NNZ];
    std::valarray<double> u_free_dof(numDOFs);
    std::valarray<double> u_free_dof_old(numDOFs);
    std::valarray<double> ML2(numDOFs);
    // Lumped L2 projection buffers for density
    std::vector<double> rho_dof(numDOFs, 0.0);
    std::vector<double> ML_rho(numDOFs, 0.0);
    std::fill(velocity_couple.data(), velocity_couple.data() + velocity_couple.size(), 0.0);
    std::fill(ebqe_velocity_ext_couple.data(), ebqe_velocity_ext_couple.data() + ebqe_velocity_ext_couple.size(), 0.0);
    auto full_offset_from_compact = [&](int i_compact, int j_compact) -> int
    {
      const int full_i = offset_u + stride_u * i_compact;
      const int full_j = offset_u + stride_u * j_compact;
      for (int offset = csrRowIndeces_Full.data()[full_i]; offset < csrRowIndeces_Full.data()[full_i + 1]; ++offset)
        if (csrColumnOffsets_Full.data()[offset] == full_j) return offset;
      return -1;
    };
    
     for (int eN = 0; eN < nElements_global; eN++)
      for (int j = 0; j < nDOF_trial_element; j++) {
        int eN_nDOF_trial_element                               = eN * nDOF_trial_element;
        u_free_dof[r_l2g.data()[eN_nDOF_trial_element + j]]     = u_dof.data()[u_l2g.data()[eN_nDOF_trial_element + j]];
        u_free_dof_old[r_l2g.data()[eN_nDOF_trial_element + j]] = u_dof_old.data()[u_l2g.data()[eN_nDOF_trial_element + j]];
      }
    for (int i = 0; i < NNZ; i++) {
      TransportMatrix[i]            = 0.;
      TransportMatrixConsistent[i]  = 0.;
      TransportMatrixn[i]           = 0.;
      TransportMatrixConsistentn[i] = 0.;
    }

    // Project quadrature density to nodal DOFs before constructing nodal
    // stabilization potentials so Phi uses a true nodal density field.
    for (int eN = 0; eN < nElements_global; eN++) {
      const int eN_nDOF_trial_element = eN * nDOF_trial_element;
      for (int k = 0; k < nQuadraturePoints_element; k++) {
        const int eN_k = eN * nQuadraturePoints_element + k;
        double jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace], x, y, z;
        ck.calculateMapping_element(eN, k, mesh_dof.data(), mesh_l2g.data(),
                                    mesh_trial_ref.data(), mesh_grad_trial_ref.data(),
                                    jac, jacDet, jacInv, x, y, z);
        const double dV = fabs(jacDet) * dV_ref.data()[k];
        for (int i = 0; i < nDOF_test_element; i++) {
          const int eN_i = eN * nDOF_test_element + i;
          const int free_gi = r_l2g.data()[eN_i];
          const double u_test_dV = u_test_ref.data()[k * nDOF_trial_element + i] * dV;
          rho_dof[free_gi] += q_rho.data()[eN_k] * u_test_dV;
          ML_rho[free_gi] += u_test_dV;
        }
      }
    }
    for (int i = 0; i < numDOFs; ++i) {
      if (ML_rho[i] > 0.0) rho_dof[i] /= ML_rho[i];
      else rho_dof[i] = rho;
    }
    // Cache the projected, salinity-coupled density for use in invert() so
    // the m -> u inversion is consistent with the forward residual.
    rho_dof_member = rho_dof;

    // -------- Comp-1: lumped L2 projection of phi*rho_n to DOFs --------
    // Used by invert(COMPONENT=1) for the m_n -> u_n = S_n inversion:
    //   m_n = phi * rho_n * u_n   =>   u_n = m_n / (phi*rho_n).
    // Indexed by u_l2g (full DOF numbering, no Dirichlet elimination) so it
    // covers every comp-1 node including Dirichlet ones.
    std::vector<double> rho_n_phi_dof(numDOFs_u, 0.0);
    std::vector<double> ML_n(numDOFs_u, 0.0);
    for (int eN = 0; eN < nElements_global; eN++) {
      const int    mat_eN_proj = elementMaterialTypes.data()[eN];
      const double phi_eN      = thetaR.data()[mat_eN_proj] + thetaSR.data()[mat_eN_proj];
      const int    eN_nDOF_trial_element = eN * nDOF_trial_element;
      for (int k = 0; k < nQuadraturePoints_element; k++) {
        double jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace], x_p, y_p, z_p;
        ck.calculateMapping_element(eN, k, mesh_dof.data(), mesh_l2g.data(),
                                    mesh_trial_ref.data(), mesh_grad_trial_ref.data(),
                                    jac, jacDet, jacInv, x_p, y_p, z_p);
        const double dV = std::fabs(jacDet) * dV_ref.data()[k];
        for (int i = 0; i < nDOF_test_element; i++) {
          const int    eN_i = eN * nDOF_test_element + i;
          const int    gi   = u_l2g.data()[eN_i];
          const double u_test_dV = u_test_ref.data()[k * nDOF_trial_element + i] * dV;
          rho_n_phi_dof[gi] += phi_eN * rho_n * u_test_dV;
          ML_n[gi]          += u_test_dV;
        }
      }
    }
    for (int i = 0; i < numDOFs_u; ++i) {
      if (ML_n[i] > 0.0) rho_n_phi_dof[i] /= ML_n[i];
      else rho_n_phi_dof[i] = thetaR.data()[0] + thetaSR.data()[0]; // fallback
      // Ensure positivity of the divisor used downstream.
      rho_n_phi_dof[i] = std::max(rho_n_phi_dof[i], 1.0e-16);
    }
    rho_n_phi_dof_member = rho_n_phi_dof;

    // compute entropy and init global_entropy_residual and boundary_integral
    double psi[numDOFs], eta[numDOFs], global_entropy_residual[numDOFs], boundary_integral[numDOFs];
    for (int i = 0; i < numDOFs; i++) {
      // NODAL ENTROPY //
      if (STABILIZATION_TYPE == STABILIZATION::EV_Stab) //EV stab
      {
        double solni = 1.0 * u_free_dof_old[i];
        eta[i]                      = ENTROPY_TYPE == 1 ? ENTROPY(solni, uL, uR) : ENTROPY_LOG(solni, uL, uR);
        global_entropy_residual[i]  = 0.;
      }
      boundary_integral[i] = 0.;
      ML2[i]               = 0.0;
    }

    //////////////////////////////////////////////
    // ** LOOP IN CELLS FOR CELL BASED TERMS ** //
    //////////////////////////////////////////////
    // HERE WE COMPUTE:
    //    * Time derivative term. u_t
    //    * cell based CFL (for reference)
    //    * Entropy residual
    //    * Transport matrices

    for (int eN = 0; eN < nElements_global; eN++) {
      const int eN_nDOF_trial_element = eN * nDOF_trial_element;
      const int eN_nDOF_mesh_trial_element = eN * nDOF_mesh_trial_element;
      //declare local storage for local contributions and initialize
      double elementResidual_u[nDOF_test_element], element_entropy_residual[nDOF_test_element], Phi[nDOF_trial_element], Phi_n[nDOF_trial_element];
      double elementTransport[nDOF_test_element][nDOF_trial_element], elementTransportConsistent[nDOF_test_element][nDOF_trial_element];
      double elementTransportn[nDOF_test_element][nDOF_trial_element], elementTransportConsistentn[nDOF_test_element][nDOF_trial_element];
      for (int j = 0; j < nDOF_trial_element; j++) {
        const int u_gj = u_l2g.data()[eN_nDOF_trial_element + j];
        const int free_gj = r_l2g.data()[eN_nDOF_trial_element + j];
        const int x_gj = mesh_l2g.data()[eN_nDOF_mesh_trial_element + j];
        const double rho_node_j = rho_dof[free_gj];
        Phi[j]   = u_dof.data()[u_gj];
        Phi_n[j] = u_dof_old.data()[u_gj];
        for (int I = 0; I < nSpace; I++) {
          // Pressure form: grad(p) - rho_w g (no /rho0 factor).
          Phi[j]   -= rho_node_j * mesh_dof.data()[x_gj * 3 + I] * gravity[I];
          Phi_n[j] -= rho_node_j * mesh_dof.data()[x_gj * 3 + I] * gravity[I];
        }
      }
      for (int i = 0; i < nDOF_test_element; i++) {
        elementResidual_u[i]        = 0.0;
        element_entropy_residual[i] = 0.0;
        for (int j = 0; j < nDOF_trial_element; j++) {
          elementTransport[i][j]            = 0.0;
          elementTransportConsistent[i][j]  = 0.0;
          elementTransportn[i][j]           = 0.0;
          elementTransportConsistentn[i][j] = 0.0;
        }
      }
      //loop over quadrature points and compute integrands
      for (int k = 0; k < nQuadraturePoints_element; k++) {
        //compute indeces and declare local storage
        int eN_k = eN * nQuadraturePoints_element + k, eN_k_nSpace = eN_k * nSpace;
        double
          // for entropy residual
          aux_entropy_residual = 0.,
          DENTROPY_un, DENTROPY_uni,
          //for mass matrix contributions
          u = 0.0, un = 0.0, grad_phi[nSpace], grad_phi_n[nSpace], grad_u_velocity[nSpace], velocity_loc[nSpace], u_test_dV[nDOF_trial_element], u_grad_trial[nDOF_trial_element * nSpace], u_grad_test_dV[nDOF_test_element * nSpace],
          //for general use
          jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace], dV, x, y, z, xt, yt, zt, m, dm, f[nSpace], df[nSpace], a[nnz], da[nnz], as[nnz], mn, dmn, fn[nSpace], dfn[nSpace], an[nnz], dan[nnz], asn[nnz];
        //get the physical integration weight
        ck.calculateMapping_element(eN, k, mesh_dof.data(), mesh_l2g.data(), mesh_trial_ref.data(), mesh_grad_trial_ref.data(), jac, jacDet, jacInv, x, y, z);
        ck.calculateMappingVelocity_element(eN, k, mesh_velocity_dof.data(), mesh_l2g.data(), mesh_trial_ref.data(), xt, yt, zt);
        dV = fabs(jacDet) * dV_ref.data()[k];
        //get the solution (of Newton's solver). To compute time derivative term
        ck.valFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], &u_trial_ref.data()[k * nDOF_trial_element], u);
        //get the solution at quad point at tn and tnm1 for entropy viscosity
        ck.valFromDOF(u_dof_old.data(), &u_l2g.data()[eN_nDOF_trial_element], &u_trial_ref.data()[k * nDOF_trial_element], un);
        //get the solution gradients at tn for entropy viscosity
        ck.gradTrialFromRef(&u_grad_trial_ref.data()[k * nDOF_trial_element * nSpace], jacInv, u_grad_trial);
        ck.gradFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], u_grad_trial, grad_u_velocity);

        //populate q_x 
        // const int eN_k_3d = eN_k * 3;
        // q_x.data()[eN_k_3d + 0] = x;
        // q_x.data()[eN_k_3d + 1] = y;
        // q_x.data()[eN_k_3d + 2] = z;
        //precalculate test function products with integration weights for mass matrix terms
        for (int I = 0; I < nSpace; I++) {
          grad_phi[I]  = 0.0;
          grad_phi_n[I] = 0.0;
        }
        for (int j = 0; j < nDOF_trial_element; j++) {
          u_test_dV[j] = u_test_ref.data()[k * nDOF_trial_element + j] * dV;
          for (int I = 0; I < nSpace; I++) {
            grad_phi_n[I] += Phi_n[j] * u_grad_trial[j * nSpace + I];
            grad_phi[I] += Phi[j] * u_grad_trial[j * nSpace + I];
            u_grad_test_dV[j * nSpace + I] = u_grad_trial[j * nSpace + I] * dV; //cek warning won't work for Petrov-Galerkin
          }
        }
        //
        //calculate pde coefficients at quadrature points
        //
        double Kr, dKr, Krn, dKrn, thetaW, thetaWn;
        const double rho_local = q_rho.data()[eN_k];
        const double rho_velocity = std::fabs(rho_local) > 1.0e-12 ? rho_local : rho;

        // Cross-derivative buffers are filled by _from_Se but ignored downstream;
        // the EV residual only uses (mn, fn, an) and (m, f, a).
        double dm_du_n_qp_n = 0.0, dkr_du_n_qp_n = 0.0;
        double dm_du_n_qp = 0.0, dkr_du_n_qp = 0.0;
        double df_du_n_qp_n[nSpace], df_du_n_qp[nSpace];
        double da_du_n_qp_n[nnz],    da_du_n_qp[nnz];
        for (int I = 0; I < nSpace; I++) { df_du_n_qp_n[I] = 0.0; df_du_n_qp[I] = 0.0; }
        for (int ii = 0; ii < nnz; ii++) { da_du_n_qp_n[ii] = 0.0; da_du_n_qp[ii] = 0.0; }
        double u_n_qp = 0.0, u_n_qp_old = 0.0;
        ck.valFromDOF(u_dof_n.data(),
                      &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_n_qp);
        ck.valFromDOF(u_dof_n_old.data(),
                      &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_n_qp_old);
        evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, rho_local, beta, gravity.data(),
                                     alpha.data()[elementMaterialTypes[eN]], n.data()[elementMaterialTypes[eN]],
                                     thetaR.data()[elementMaterialTypes[eN]], thetaSR.data()[elementMaterialTypes[eN]],
                                     &KWs.data()[elementMaterialTypes[eN] * nnz], un, u_n_qp_old,
                                     mn, dmn, dm_du_n_qp_n, fn, dfn, df_du_n_qp_n, an, dan, da_du_n_qp_n,
                                     asn, Krn, dKrn, dkr_du_n_qp_n, thetaWn);
        evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, rho_local, beta, gravity.data(),
                                     alpha.data()[elementMaterialTypes[eN]], n.data()[elementMaterialTypes[eN]],
                                     thetaR.data()[elementMaterialTypes[eN]], thetaSR.data()[elementMaterialTypes[eN]],
                                     &KWs.data()[elementMaterialTypes[eN] * nnz], u, u_n_qp,
                                     m, dm, dm_du_n_qp, f, df, df_du_n_qp, a, da, da_du_n_qp,
                                     as, Kr, dKr, dkr_du_n_qp, thetaW);
        q_theta.data()[eN_k] = thetaW;

        // Darcy velocity for coupling should use the direct FE gradient of the
        // pressure head. The Phi-based gradients are only for stabilization.
        for (int I = 0; I < nSpace; ++I) {
          q_velocity.data()[eN_k_nSpace + I] = grad_u_velocity[I];
        }

        double pressure_gradient[nSpace];
        for (int J = 0; J < nSpace; ++J)
          pressure_gradient[J] = grad_u_velocity[J] - rho_velocity * gravity.data()[J];

        for (int I = 0; I < nSpace; ++I) {
          double acc = 0.0;
          for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I+1]; ++ii) {
            const int J = a_colind.data()[ii];
            acc += (a[ii] / rho_velocity) * pressure_gradient[J];
          }
          velocity.data()[eN_k_nSpace + I] = -acc;
          velocity_couple.data()[eN_k_nSpace + I] = -acc;
        }
        // static int debug_ev_prints = 0;
        // if (debug_ev_prints < 6 && eN < 2 && k < 2)
        // {
        //   std::cout << "[Richards EV fixed] eN=" << eN
        //             << " k=" << k
        //             << " u=" << u
        //             << " un=" << un
        //             << " grad_phi=(" << grad_phi[0];
        //   for (int I = 1; I < nSpace; ++I)
        //     std::cout << "," << grad_phi[I];
        //   std::cout << ") grad_u_velocity=(" << grad_u_velocity[0];
        //   for (int I = 1; I < nSpace; ++I)
        //     std::cout << "," << grad_u_velocity[I];
        //   std::cout << ") velocity_couple=(" << velocity_couple.data()[eN_k_nSpace + 0];
        //   for (int I = 1; I < nSpace; ++I)
        //     std::cout << "," << velocity_couple.data()[eN_k_nSpace + I];
        //   std::cout << ")" << std::endl;
        //   for (int j = 0; j < nDOF_trial_element; ++j)
        //   {
        //     const int u_gj = u_l2g.data()[eN_nDOF_trial_element + j];
        //     const int x_gj = mesh_l2g.data()[eN_nDOF_mesh_trial_element + j];
        //     const int free_gj = r_l2g.data()[eN_nDOF_trial_element + j];
        //     std::cout << "  [EV fixed dof] j=" << j
        //               << " mapped_u=" << u_dof.data()[u_gj]
        //               << " mapped_u_old=" << u_dof_old.data()[u_gj]
        //               << " free_material=" << freeDOFMaterialTypes.data()[free_gj]
        //               << " mapped_x=(" << mesh_dof.data()[x_gj * 3 + 0]
        //               << "," << mesh_dof.data()[x_gj * 3 + 1]
        //               << "," << mesh_dof.data()[x_gj * 3 + 2]
        //               << ") Phi=" << Phi[j]
        //               << " Phi_n=" << Phi_n[j]
        //               << std::endl;
        //   }
        //   debug_ev_prints++;
        // }
       // if (nSpace != 2) {std::cout << "WARNING nSpace=" << nSpace << std::endl;}
        //
        //moving mesh
        //
        double mesh_velocity[3];
        mesh_velocity[0] = xt;
        mesh_velocity[1] = yt;
        mesh_velocity[2] = zt;
        //relative velocity at tn
        for (int I = 0; I < nSpace; I++) {
          f[I] -= MOVING_DOMAIN * m * mesh_velocity[I];
          velocity_loc[I] = df[I] * (2.0 * dm * dm / (dm * dm + fmax(1.0e-16, dm * dm)));
        }
        //////////////////////////////
        // CALCULATE CELL BASED CFL //
        //////////////////////////////
        calculateCFL(elementDiameter.data()[eN] / degree_polynomial, velocity_loc, cfl.data()[eN_k]);
        //////////////////////////////////////////////
        // CALCULATE ENTROPY RESIDUAL AT QUAD POINT //
        //////////////////////////////////////////////
        if (STABILIZATION_TYPE == STABILIZATION::EV_Stab) // EV stab
        {
          for (int I = 0; I < nSpace; I++) aux_entropy_residual += velocity_loc[I] * grad_phi_n[I];
          DENTROPY_un = ENTROPY_TYPE == 1 ? DENTROPY(un, uL, uR) : DENTROPY_LOG(un, uL, uR);
        }
        //////////////
        // ith-LOOP //
        //////////////
        for (int i = 0; i < nDOF_test_element; i++) {
          // VECTOR OF ENTROPY RESIDUAL //
          int eN_i = eN * nDOF_test_element + i;
          ML2[r_l2g.data()[eN_i]] += u_test_dV[i];
          if (STABILIZATION_TYPE == STABILIZATION::EV_Stab) // EV stab
          {
            double uni = u_dof_old.data()[u_l2g.data()[eN_i]];
            DENTROPY_uni              = ENTROPY_TYPE == 1 ? DENTROPY(uni, uL, uR) : DENTROPY_LOG(uni, uL, uR);
            element_entropy_residual[i] += (DENTROPY_un - DENTROPY_uni) * aux_entropy_residual * u_test_dV[i];
          }

          elementResidual_u[i] += m * u_test_dV[i];
          ///////////////
          // j-th LOOP // To construct transport matrices
          ///////////////
          
          for (int j = 0; j < nDOF_trial_element; j++) {
            int j_nSpace = j * nSpace;
            int i_nSpace = i * nSpace;
            elementTransport[i][j] += ck.SimpleDiffusionJacobian_weak(a_rowptr.data(), a_colind.data(), as, &u_grad_trial[j_nSpace], &u_grad_test_dV[i_nSpace]);
            elementTransportConsistent[i][j] += ck.SimpleDiffusionJacobian_weak(a_rowptr.data(), a_colind.data(), a, &u_grad_trial[j_nSpace], &u_grad_test_dV[i_nSpace]);
            elementTransportn[i][j] += ck.SimpleDiffusionJacobian_weak(a_rowptr.data(), a_colind.data(), asn, &u_grad_trial[j_nSpace], &u_grad_test_dV[i_nSpace]);
            elementTransportConsistentn[i][j] += ck.SimpleDiffusionJacobian_weak(a_rowptr.data(), a_colind.data(), an, &u_grad_trial[j_nSpace], &u_grad_test_dV[i_nSpace]);
          }
        } //i 
        //save solution for other models
        q_u.data()[eN_k] = u;
        q_m.data()[eN_k] = m;
      }
      /////////////////
      // DISTRIBUTE // load cell based element into global residual
      ////////////////
      for (int i = 0; i < nDOF_test_element; i++) {
        int eN_i = eN * nDOF_test_element + i;
        int gi   = r_l2g.data()[eN_i];
        // distribute entropy_residual
        if (STABILIZATION_TYPE == STABILIZATION::EV_Stab) // EV Stab
          global_entropy_residual[gi] += element_entropy_residual[i];
        // distribute transport matrices
        for (int j = 0; j < nDOF_trial_element; j++) {
          int eN_i_j = eN_i * nDOF_trial_element + j;
          TransportMatrix[csrRowIndeces_CellLoops.data()[eN_i] + csrColumnOffsets_CellLoops.data()[eN_i_j]] += elementTransport[i][j];
          TransportMatrixConsistent[csrRowIndeces_CellLoops.data()[eN_i] + csrColumnOffsets_CellLoops.data()[eN_i_j]] += elementTransportConsistent[i][j];
          TransportMatrixn[csrRowIndeces_CellLoops.data()[eN_i] + csrColumnOffsets_CellLoops.data()[eN_i_j]] += elementTransportn[i][j];
          TransportMatrixConsistentn[csrRowIndeces_CellLoops.data()[eN_i] + csrColumnOffsets_CellLoops.data()[eN_i_j]] += elementTransportConsistentn[i][j];
        } //j
      } //i
      
    } //elementsxw

    // double s = 0.0, sabs = 0.0;
    // double vmin = 1e300, vmax = -1e300;
    // size_t n_nonzero = 0;

    // for (size_t i = 0; i < velocity.size(); i++) {
    //   double v = velocity.data()[i];
    //   s += v;
    //   sabs += std::abs(v);
    //   vmin = std::min(vmin, v);
    //   vmax = std::max(vmax, v);
    //   if (std::abs(v) > 1e-14) n_nonzero++;
    // }

    // std::cout << "[after compute] size=" << velocity.size()
    //           << " nnz=" << n_nonzero
    //           << " min=" << vmin << " max=" << vmax
    //           << " sumabs=" << sabs << std::endl;

        //loop over exterior element boundaries to calculate surface integrals and load into element and global residuals
    //
    //ebNE is the Exterior element boundary INdex
    //ebN is the element boundary INdex
    //eN is the element index
    for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++) {
      int    ebN = exteriorElementBoundariesArray.data()[ebNE], eN = elementBoundaryElementsArray.data()[ebN * 2 + 0], ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN * 2 + 0], eN_nDOF_trial_element = eN * nDOF_trial_element;
      double elementResidual_u[nDOF_test_element];
      for (int i = 0; i < nDOF_test_element; i++) { elementResidual_u[i] = 0.0; }
      for (int kb = 0; kb < nQuadraturePoints_elementBoundary; kb++) {
        int    ebNE_kb = ebNE * nQuadraturePoints_elementBoundary + kb, ebNE_kb_nSpace = ebNE_kb * nSpace, ebN_local_kb = ebN_local * nQuadraturePoints_elementBoundary + kb, ebN_local_kb_nSpace = ebN_local_kb * nSpace;
        double u_ext = 0.0, un_ext, grad_u_ext[nSpace], m_ext = 0.0, dm_ext = 0.0, f_ext[nSpace], df_ext[nSpace], a_ext[nnz], da_ext[nnz], as_ext[nnz], 
        mn_ext = 0.0, dmn_ext = 0.0, fn_ext[nSpace], dfn_ext[nSpace], an_ext[nnz], dan_ext[nnz], asn_ext[nnz], flux_ext = 0.0, bflux_ext = 0.0,
               //anb_seepage_flux=0.0, // for flux calculation
          bc_u_ext = 0.0, bc_grad_u_ext[nSpace], bc_m_ext = 0.0, bc_dm_ext = 0.0, bc_f_ext[nSpace], bc_df_ext[nSpace], bc_a_ext[nnz], bc_da_ext[nnz], bc_as_ext[nnz], jac_ext[nSpace * nSpace], jacDet_ext, jacInv_ext[nSpace * nSpace], boundaryJac[nSpace * (nSpace - 1)], metricTensor[(nSpace - 1) * (nSpace - 1)], metricTensorDetSqrt, dS, u_test_dS[nDOF_test_element], u_grad_trial_trace[nDOF_trial_element * nSpace], normal[3], x_ext, y_ext, z_ext, xt_ext, yt_ext, zt_ext, integralScaling, G[nSpace * nSpace], G_dd_G, tr_G, fluxJacobian_u_u[nDOF_trial_element], bfluxJacobian_u_u[nDOF_trial_element], fluxJacobian_un_un[nDOF_trial_element];
        //
        //calculate the solution and gradients at quadrature points
        //
        //compute information about mapping from reference element to physical element
        ck.calculateMapping_elementBoundary(eN, ebN_local, kb, ebN_local_kb, mesh_dof.data(), mesh_l2g.data(), mesh_trial_trace_ref.data(), mesh_grad_trial_trace_ref.data(), boundaryJac_ref.data(), jac_ext, jacDet_ext, jacInv_ext, boundaryJac, metricTensor, metricTensorDetSqrt,
                                            normal_ref.data(), normal, x_ext, y_ext, z_ext);
        ck.calculateMappingVelocity_elementBoundary(eN, ebN_local, kb, ebN_local_kb, mesh_velocity_dof.data(), mesh_l2g.data(), mesh_trial_trace_ref.data(), xt_ext, yt_ext, zt_ext, normal, boundaryJac, metricTensor, integralScaling);
        dS = ((1.0 - MOVING_DOMAIN) * metricTensorDetSqrt + MOVING_DOMAIN * integralScaling) * dS_ref.data()[kb];
        //get the metric tensor
        //cek todo use symmetry
        ck.calculateG(jacInv_ext, G, G_dd_G, tr_G);
        //compute shape and solution information
        //shape
        ck.gradTrialFromRef(&u_grad_trial_trace_ref.data()[ebN_local_kb_nSpace * nDOF_trial_element], jacInv_ext, u_grad_trial_trace);
        //solution and gradient
        ck.valFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], &u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element], u_ext);
        ck.valFromDOF(u_dof_old.data(), &u_l2g.data()[eN_nDOF_trial_element], &u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element], un_ext);
        ck.gradFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], u_grad_trial_trace, grad_u_ext);
        
        //populate ebqe_x
        // const int ebNE_kb_3d = ebNE_kb * 3;
        // ebqe_x.data()[ebNE_kb_3d + 0] = x_ext;
        // ebqe_x.data()[ebNE_kb_3d + 1] = y_ext;
        // ebqe_x.data()[ebNE_kb_3d + 2] = z_ext;

        
        //precalculate test function products with integration weights
        for (int j = 0; j < nDOF_trial_element; j++) { u_test_dS[j] = u_test_trace_ref.data()[ebN_local_kb * nDOF_test_element + j] * dS; }
        //
        //load the boundary values
        //
        bc_u_ext = isDOFBoundary_u.data()[ebNE_kb] * ebqe_bc_u_ext.data()[ebNE_kb] + (1 - isDOFBoundary_u.data()[ebNE_kb]) * u_ext;
        //
        //calculate the pde coefficients using the solution and the boundary values for the solution
        //
        double bc_Kr, bc_dKr,bc_Kr_ext, bc_dKr_ext, bc_Krn, bc_dKrn, thetaW_ext, thetaWn_ext, thetaW_bc_ext;
        const double rho_ext = ebqe_rho.data()[ebNE_kb];
        const double rho_velocity_ext = std::fabs(rho_ext) > 1.0e-12 ? rho_ext : rho;

        // EV exterior boundary closure.
        double dm_du_n_ext = 0.0, dkr_du_n_ext = 0.0;
        double dmn_du_n_ext = 0.0, dkrn_du_n_ext = 0.0;
        double bc_dm_du_n = 0.0, bc_dkr_du_n = 0.0;
        double df_du_n_ext[nSpace], dfn_du_n_ext[nSpace], bc_df_du_n[nSpace];
        double da_du_n_ext[nnz], dan_du_n_ext[nnz], bc_da_du_n[nnz];
        for (int I = 0; I < nSpace; I++) {
          df_du_n_ext[I] = 0.0; dfn_du_n_ext[I] = 0.0; bc_df_du_n[I] = 0.0;
        }
        for (int ii = 0; ii < nnz; ii++) {
          da_du_n_ext[ii] = 0.0; dan_du_n_ext[ii] = 0.0; bc_da_du_n[ii] = 0.0;
        }
        double u_n_ext_qp = 0.0, u_n_ext_qp_old = 0.0;
        ck.valFromDOF(u_dof_n.data(), &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element], u_n_ext_qp);
        ck.valFromDOF(u_dof_n_old.data(), &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element], u_n_ext_qp_old);
        const double bc_u_n_ext_qp = isDOFBoundary_n.data()[ebNE_kb] * ebqe_bc_u_n_ext.data()[ebNE_kb]
                                   + (1 - isDOFBoundary_n.data()[ebNE_kb]) * u_n_ext_qp;
        evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, rho_ext, beta, gravity.data(),
                                     alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]],
                                     thetaR.data()[elementMaterialTypes.data()[eN]], thetaSR.data()[elementMaterialTypes.data()[eN]],
                                     &KWs.data()[elementMaterialTypes.data()[eN] * nnz], u_ext, u_n_ext_qp,
                                     m_ext, dm_ext, dm_du_n_ext, f_ext, df_ext, df_du_n_ext, a_ext, da_ext, da_du_n_ext,
                                     as_ext, bc_Kr, bc_dKr, dkr_du_n_ext, thetaW_ext);
        evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, rho_ext, beta, gravity.data(),
                                     alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]],
                                     thetaR.data()[elementMaterialTypes.data()[eN]], thetaSR.data()[elementMaterialTypes.data()[eN]],
                                     &KWs.data()[elementMaterialTypes.data()[eN] * nnz], un_ext, u_n_ext_qp_old,
                                     mn_ext, dmn_ext, dmn_du_n_ext, fn_ext, dfn_ext, dfn_du_n_ext, an_ext, dan_ext, dan_du_n_ext,
                                     asn_ext, bc_Krn, bc_dKrn, dkrn_du_n_ext, thetaWn_ext);
        evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, rho_ext, beta, gravity.data(),
                                     alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]],
                                     thetaR.data()[elementMaterialTypes.data()[eN]], thetaSR.data()[elementMaterialTypes.data()[eN]],
                                     &KWs.data()[elementMaterialTypes.data()[eN] * nnz], bc_u_ext, bc_u_n_ext_qp,
                                     bc_m_ext, bc_dm_ext, bc_dm_du_n, bc_f_ext, bc_df_ext, bc_df_du_n, bc_a_ext, bc_da_ext, bc_da_du_n,
                                     bc_as_ext, bc_Kr_ext, bc_dKr_ext, bc_dkr_du_n, thetaW_bc_ext);
        ebqe_theta.data()[ebNE_kb] = thetaW_ext;
        
        //
        //Calculate Darcy Velocity at external faces
        //
        //
        //Calculate Darcy Velocity at external faces
        //
        
        // double  darcy_velocity_loc_ext[nSpace];
        // for (int I = 0; I < nSpace; I++) { darcy_velocity_loc_ext[I] = 0.0; }
        
        // for (int I = 0; I < nSpace; I++) {
        //   for (int J = 0; J < nSpace; J++) { darcy_velocity_loc_ext[I] -= bc_Kr * KWs.data()[elementMaterialTypes[eN] * nSpace * nSpace + I * nSpace + J] * (grad_u_ext[J]+  gravity.data()[J]); }
        // }
        // for (int I = 0; I < nSpace; I++) { ebqe_velocity_ext_couple.data()[ebNE_kb_nSpace + I] = darcy_velocity_loc_ext[I] ; }
        
        double ext_pressure_gradient[nSpace];
        for (int J = 0; J < nSpace; ++J)
          ext_pressure_gradient[J] = grad_u_ext[J] - rho_velocity_ext * gravity.data()[J];

        for (int I = 0; I < nSpace; ++I) {
          double acc = 0.0;
          for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I+1]; ++ii) {
            const int J = a_colind.data()[ii];
            acc += (a_ext[ii] / rho_velocity_ext) * ext_pressure_gradient[J];
          }
          ebqe_velocity_ext.data()[ebNE_kb_nSpace + I] = -acc;
          ebqe_velocity_ext_couple.data()[ebNE_kb_nSpace + I] = -acc;
        }


        //
        //calculate the numerical fluxes
        //
        bool useConsistentFlux=false;
        if (useConsistentFlux) {
          exteriorNumericalFlux(ebqe_bc_flux_ext[ebNE_kb], a_rowptr.data(), a_colind.data(),
                                isSeepageFace.data()[ebNE], //tricky, this is a face flag not face quad
                                isDOFBoundary_u.data()[ebNE_kb], normal, bc_u_ext, a_ext, grad_u_ext, u_ext, f_ext,
                                ebqe_penalty_ext.data()[ebNE_kb], // penalty,
                                flux_ext);
        } else {
          exteriorNumericalFlux2(ebqe_bc_flux_ext[ebNE_kb], a_rowptr.data(), a_colind.data(),
                              isSeepageFace.data()[ebNE], //tricky, this is a face flag not face quad
                              isDOFBoundary_u.data()[ebNE_kb], normal, bc_u_ext, a_ext, grad_u_ext, u_ext, f_ext,
                              ebqe_penalty_ext.data()[ebNE_kb], // penalty,
                              flux_ext, bflux_ext);
        }

        ebqe_flux.data()[ebNE_kb] = flux_ext;

        anb_seepage_flux             = seepagefluxcalculator(anb_seepage_flux, isSeepageFace.data()[ebNE], dS, flux_ext);
        anb_seepage_flux_n.data()[0] = anb_seepage_flux;
        ebqe_u.data()[ebNE_kb]       = u_ext;
        //
        //update residuals
        //
        for (int i = 0; i < nDOF_test_element; i++) {
          if (useConsistentFlux) {
            elementResidual_u[i] += ck.ExteriorElementBoundaryFlux(flux_ext, u_test_dS[i]);
          } else {
            elementResidual_u[i] += ck.ExteriorElementBoundaryFlux(bflux_ext, u_test_dS[i]);
          }
        } //i
        for (int j = 0; j < nDOF_trial_element; j++) {
          if (useConsistentFlux) {
          exteriorNumericalFluxJacobian(a_rowptr.data(), a_colind.data(), isDOFBoundary_u.data()[ebNE_kb], normal, a_ext, da_ext, grad_u_ext, &u_grad_trial_trace[j * nSpace], df_ext, u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element + j],
                                        ebqe_penalty_ext.data()[ebNE_kb], //penalty,
                                        fluxJacobian_u_u[j]);
          } else {
            exteriorNumericalFluxJacobian2(a_rowptr.data(), a_colind.data(), isDOFBoundary_u.data()[ebNE_kb], normal, as_ext, a_ext, da_ext, grad_u_ext, &u_grad_trial_trace[j * nSpace], df_ext, u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element + j],
                                        ebqe_penalty_ext.data()[ebNE_kb], //penalty,
                                        fluxJacobian_u_u[j],bfluxJacobian_u_u[j]);
          }
          // Time-history flux Jacobian (used by the (1-Theta) part of the
          // edge loop in TransportMatrixn / TransportMatrixConsistentn).
          // Build it from the time-n state (asn_ext, dan_ext, dfn_ext) so the
          // edge loop sees a self-consistent boundary contribution under
          // partial-Theta schemes. With Theta=1 (default implicit Euler)
          // this drops out of the residual, but it must be initialised --
          // previously it was an uninitialised stack array.
          exteriorNumericalFluxJacobian(a_rowptr.data(), a_colind.data(),
              isDOFBoundary_u.data()[ebNE_kb], normal,
              asn_ext, dan_ext, grad_u_ext,
              &u_grad_trial_trace[j * nSpace], dfn_ext,
              u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element + j],
              ebqe_penalty_ext.data()[ebNE_kb],
              fluxJacobian_un_un[j]);
        } //j
        //
        //update the element and global residual storage
        //
        for (int i = 0; i < nDOF_test_element; i++) {
          int eN_i = eN * nDOF_test_element + i;
          for (int j = 0; j < nDOF_trial_element; j++) {
            int ebN_i_j = ebN * 4 * nDOF_test_X_trial_element + i * nDOF_trial_element + j;
            if (useConsistentFlux) {
              globalJacobian.data()[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_eb_u_u[ebN_i_j]] += fluxJacobian_u_u[j] * u_test_dS[i];
            } else {
              globalJacobian.data()[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_eb_u_u[ebN_i_j]] += bfluxJacobian_u_u[j] * u_test_dS[i];
              TransportMatrix[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_eb_u_u[ebN_i_j]] += fluxJacobian_u_u[j] * u_test_dS[i];
              TransportMatrixConsistent[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_eb_u_u[ebN_i_j]] += fluxJacobian_u_u[j] * u_test_dS[i];
              TransportMatrixn[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_eb_u_u[ebN_i_j]] += fluxJacobian_un_un[j] * u_test_dS[i];
              TransportMatrixConsistentn[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_eb_u_u[ebN_i_j]] += fluxJacobian_un_un[j] * u_test_dS[i];
            }
          } //j
        } //i
      } //kb
      for (int i = 0; i < nDOF_test_element; i++) {
          int eN_i = eN * nDOF_test_element + i;
          globalResidual.data()[offset_u + stride_u * u_l2g.data()[eN_i]] += elementResidual_u[i];
      }//i
    } //ebNE
    /////////////////////////////////////////////////////////////////
    // COMPUTE SMOOTHNESS INDICATOR and NORMALIZE ENTROPY RESIDUAL //
    /////////////////////////////////////////////////////////////////
    // NOTE: see NCLS.h for a different but equivalent implementation of this.
    std::vector<double> cflux(numDOFs, 0.0);
    // bound to numDOFs_u (component 0 only). For i >= numDOFs_u
    // mesh_dof.data()[i*3+I] reads past the end of mesh_dof (sized N*3) and
    // produces NaN that propagates through psi[i] / quantDOFs[i] etc. into
    // the main DOF loop's residual.
    for (int i = 0; i < numDOFs_u; i++) {
      double gi[nSpace], Cij[nSpace], xi[nSpace], etaMaxi, etaMini;
      const int node_i = freeDOFToNode_u.data()[i];
      double solni = u_free_dof_old[i];
      for (int I = 0; I < nSpace; I++) {
        solni -= rho_dof[i] * gravity.data()[I] * mesh_dof.data()[node_i * 3 + I];
      }
      if (STABILIZATION_TYPE == STABILIZATION::EV_Stab) //EV Stabilization
      {
        // For eta min and max
        etaMaxi = fabs(eta[i]);
        etaMini = fabs(eta[i]);
      }
      // initialize gi and compute xi
      for (int I = 0; I < nSpace; I++) {
        gi[I] = 0.;
        xi[I] = mesh_dof.data()[node_i * 3 + I];
      }
      // for smoothness indicator //
      double alpha_numerator_pos = 0., alpha_numerator_neg = 0., alpha_denominator_pos = 0., alpha_denominator_neg = 0.;
      for (int offset = csrRowIndeces_DofLoops.data()[i]; offset < csrRowIndeces_DofLoops.data()[i + 1]; offset++) { // First loop in j (sparsity pattern)
        int j = csrColumnOffsets_DofLoops.data()[offset];
        const int full_offset = full_offset_from_compact(i, j);
        assert(full_offset >= 0);
        const int node_j = freeDOFToNode_u.data()[j];
        if (STABILIZATION_TYPE == STABILIZATION::EV_Stab) //EV Stabilization
        {
          // COMPUTE ETA MIN AND ETA MAX //
          etaMaxi = fmax(etaMaxi, fabs(eta[j]));
          etaMini = fmin(etaMini, fabs(eta[j]));
        }
        double solnj = u_free_dof_old[j];
        for (int I = 0; I < nSpace; I++) {
          solnj -= rho_dof[j] * gravity.data()[I] * mesh_dof.data()[node_j * 3 + I];
        }
        // Update Cij matrices
        Cij[0] = Cx[full_offset];
#if nSpace == 2
        Cij[1] = Cy[full_offset];
#endif
#if nSpace == 3
        Cij[2] = Cz[full_offset];
#endif
        // COMPUTE gi VECTOR. gi=1/mi*sum_j(Cij*solj)
        for (int I = 0; I < nSpace; I++) gi[I] += Cij[I] * solnj;

        // COMPUTE numerator and denominator of smoothness indicator
        double alpha_num = solni - solnj;
        if (alpha_num >= 0.) {
          alpha_numerator_pos += alpha_num;
          alpha_denominator_pos += alpha_num;
        } else {
          alpha_numerator_neg += alpha_num;
          alpha_denominator_neg += fabs(alpha_num);
        }
      }
      // scale g vector by lumped mass matrix
      //double mass_matrix_error = abs(ML.data()[i] - ML2[i]);
      //if (mass_matrix_error > 1.0e-16) std::cout << mass_matrix_error<<" ML " << ML.data()[i] << '\t' << ML2[i] << std::endl;
      for (int I = 0; I < nSpace; I++) gi[I] /= ML.data()[i];
      if (STABILIZATION_TYPE == STABILIZATION::EV_Stab) //EV Stab
      {
        // Normalizae entropy residual
        global_entropy_residual[i] *= etaMini == etaMaxi ? 0. : 2 * cE / (etaMaxi - etaMini);
        quantDOFs.data()[i] = fabs(global_entropy_residual[i]);
      }

      // Now that I have the gi vectors, I can use them for the current i-th DOF
      double SumPos = 0., SumNeg = 0.;
      for (int offset = csrRowIndeces_DofLoops.data()[i]; offset < csrRowIndeces_DofLoops.data()[i + 1]; offset++) { // second loop in j (sparsity pattern)
        int j = csrColumnOffsets_DofLoops.data()[offset];
        const int full_offset = full_offset_from_compact(i, j);
        assert(full_offset >= 0);
        // compute gi*(xi-xj)
        double gi_times_x = 0.;
        for (int I = 0; I < nSpace; I++) {
          gi_times_x += gi[I] * delta_x_ij.data()[full_offset * 3 + I];
        }
        // compute the positive and negative part of gi*(xi-xj)
        SumPos += gi_times_x > 0 ? gi_times_x : 0;
        SumNeg += gi_times_x < 0 ? gi_times_x : 0;
      }
      double sigmaPosi  = fmin(1., (fabs(SumNeg) + 1E-15) / (SumPos + 1E-15));
      double sigmaNegi  = fmin(1., (SumPos + 1E-15) / (fabs(SumNeg) + 1E-15));
      double alpha_numi = fabs(sigmaPosi * alpha_numerator_pos + sigmaNegi * alpha_numerator_neg);
      double alpha_deni = sigmaPosi * alpha_denominator_pos + sigmaNegi * alpha_denominator_neg;
      if (IS_BETAij_ONE == 1) {
        alpha_numi = fabs(alpha_numerator_pos + alpha_numerator_neg);
        alpha_deni = alpha_denominator_pos + alpha_denominator_neg;
      }
      double alphai       = alpha_numi / (alpha_deni + 1E-15);
      quantDOFs.data()[i] = alphai;

      if (POWER_SMOOTHNESS_INDICATOR == 0) psi[i] = 1.0;
      else psi[i] = std::pow(alphai, POWER_SMOOTHNESS_INDICATOR); //NOTE: they use alpha^2 in the paper
    }
    /////////////////////////////////////////////
    // ** LOOP IN DOFs FOR EDGE BASED TERMS ** //
    /////////////////////////////////////////////
    // only iterate component-0 DOFs here. Component 1
    // (trivial gas eq) is assembled by the dedicated element loop appended
    // at the end of this function.
    for (int i = 0; i < numDOFs_u; i++) {
      int    ii = -1;
      double sum_abs_dt_times_fH_minus_fL = 0.0, MLi = ML.data()[i];
      double Kr, dKr, Krn, dKrn;
      double J_ii = 0.0;
      double ith_dissipative_term           = 0;
      double ith_low_order_dissipative_term = 0;
      double ith_flux_term                  = 0;
      double ith_consistent_flux_term       = 0;
      double dLii                           = 0.;
      double m, dm, f[nSpace], df[nSpace], a[nnz], da[nnz], as[nnz];
      double dmn, fn[nSpace], dfn[nSpace], an[nnz], dan[nnz], asn[nnz];
      // Cross-derivatives used by the (0,1) coupling wired into the low-order
      // wetting operator below: dm/du_n is added to the lumped-mass coupling
      // (row_w, col_n_i), and dkr/du_n[upwind] is added to the Kuzmin upwind
      // term at (row_w, col_n_upwind). The 'n_old' variants describe the
      // frozen time history -- they don't contribute to the Jacobian.
      double dm_du_n_fct, dkr_du_n_fct, df_du_n_fct[nSpace], da_du_n_fct[nnz];
      double dmn_du_n_fct, dkrn_du_n_fct, dfn_du_n_fct[nSpace], dan_du_n_fct[nnz];

      const double rho_i = rho_dof[i];
      const int node_i = freeDOFToNode_u.data()[i];

      double thetaW_tmp = 0.0;
      // loop over the sparsity pattern of the i-th DOF
      for (int offset = csrRowIndeces_DofLoops.data()[i]; offset < csrRowIndeces_DofLoops.data()[i + 1]; offset++) {
        int j = csrColumnOffsets_DofLoops.data()[offset];
        const int full_offset = full_offset_from_compact(i, j);
        assert(full_offset >= 0);
        if (i == j) ii = full_offset;
        const double rho_j = rho_dof[j];
        const int node_j = freeDOFToNode_u.data()[j];
        const double rho_edge = 0.5 * (rho_i + rho_j);
        double delta_phi = u_free_dof[j] - u_free_dof[i];
        double delta_phin = u_free_dof_old[j] - u_free_dof_old[i];
        // Pressure form: edge-based hydrostatic jump is rho_edge * g * delta_x
        // (no /rho0 factor). Using rho_edge rather than grad(rho g x) avoids a
        // spurious (g.x) grad(rho) term.
        for (int I = 0; I < nSpace; I++) {
          const double delta_x = mesh_dof.data()[node_j * 3 + I] - mesh_dof.data()[node_i * 3 + I];
          const double hydrostatic_jump = rho_edge * gravity.data()[I] * delta_x;
          delta_phi -= hydrostatic_jump;
          delta_phin -= hydrostatic_jump;
        }
        double dLowij, dLij, dEVij, dHij, fH, fL, fA=0.0;
        double fL_CN =0.0, fA_CN=0.0 ; 
        fH = -Theta * TransportMatrixConsistent[full_offset] * delta_phi - (1 - Theta) * TransportMatrixConsistentn[full_offset] * delta_phin;
        // fH = -Theta_h * TransportMatrixConsistent[ij] * delta_phi - (1 - Theta_h) * TransportMatrixConsistentn[ij] * delta_phin; //previous: Theta
        ith_consistent_flux_term += fH;
        fA = fH;
        fA_CN = fH;

        if (-TransportMatrix[full_offset] * delta_phi <= 0.0) {
          // Upstream is i. Kr = k_rw(u_n[node_i]); dKr = dk_rw/du_w = 0 (k_rw
          // depends on S_n only). The cross-derivative dk_rw/du_n is returned
          // in dkr_du_n_fct and wired into the (0,1) cross-block below so the
          // low-order operator is fully consistent with the (p_w, S_n) wetting
          // equation (no longer a legacy pressure-centered approximation).
          evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, rho_i, beta, gravity.data(),
                                       alpha.data()[elementMaterialTypes.data()[0]],
                                       n.data()[elementMaterialTypes.data()[0]], thetaR.data()[elementMaterialTypes.data()[0]], thetaSR.data()[elementMaterialTypes.data()[0]], &KWs.data()[elementMaterialTypes.data()[0] * nnz],
                                       u_free_dof[i], u_dof_n.data()[node_i],
                                       m, dm, dm_du_n_fct, f, df, df_du_n_fct, a, da, da_du_n_fct, as, Kr, dKr, dkr_du_n_fct, thetaW_tmp);
          fL = Theta * Kr * fmax(0.0, -TransportMatrix[full_offset]) * delta_phi;
          fL_CN = Theta_h * Kr * fmax(0.0, -TransportMatrix[full_offset]) * delta_phi;

          if (i != j) {
            // Mask off-diagonal write so Dirichlet wetting rows (bc_mask[i]=0)
            // stay as identity rows, matching the residual / diagonal masking
            // at lines below and the (0,1) cross-block masking just above.
            globalJacobian.data()[full_offset] -= bc_mask.data()[i] * Theta * Kr * fmax(0.0, -TransportMatrix[full_offset]);
            // dKr = dk_rw/du_w = 0 by construction (k_rw depends only on u_n
            // through S_e); the chain-rule term in u_n is wired through the
            // (0,1) cross-block via dkr_du_n_fct below.
            J_ii += Theta * Kr * fmax(0.0, -TransportMatrix[full_offset]);
          }
          ith_flux_term += fL;
          fA -= fL;
          // (0,1) cross-block: d/du_n[node_i] of -fL (R_w subtracts ith_flux_term).
          //   d(fL)/du_n[node_i] = Theta * (dkr/du_n) * max(0,-T[ij]) * delta_phi
          // Apply bc_mask so Dirichlet wetting DOFs keep an identity row.
          {
            const int row_w = offset_u + stride_u * i;
            const int col_n_up = offset_n + stride_n * node_i;
            for (int o = csrRowIndeces_Full.data()[row_w];
                 o < csrRowIndeces_Full.data()[row_w + 1]; o++) {
              if (csrColumnOffsets_Full.data()[o] == col_n_up) {
                globalJacobian.data()[o] -=
                    bc_mask.data()[i] * Theta * dkr_du_n_fct
                  * fmax(0.0, -TransportMatrix[full_offset]) * delta_phi;
                break;
              }
            }
          }
        } else {
          // Upstream is j.
          evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, rho_j, beta, gravity.data(),
                                       alpha.data()[elementMaterialTypes.data()[0]],
                                       n.data()[elementMaterialTypes.data()[0]], thetaR.data()[elementMaterialTypes.data()[0]], thetaSR.data()[elementMaterialTypes.data()[0]], &KWs.data()[elementMaterialTypes.data()[0] * nnz],
                                       u_free_dof[j], u_dof_n.data()[node_j],
                                       m, dm, dm_du_n_fct, f, df, df_du_n_fct, a, da, da_du_n_fct, as, Kr, dKr, dkr_du_n_fct, thetaW_tmp);
          fL = Theta * Kr * fmax(0.0, -TransportMatrix[full_offset]) * delta_phi;
          fL_CN = Theta_h * Kr * fmax(0.0, -TransportMatrix[full_offset]) * delta_phi;

          if (i != j) {
            // Mask off-diagonal write (see same rationale on the upstream-i branch above).
            // dKr = dk_rw/du_w = 0 (k_rw depends only on u_n); the chain-rule
            // term in u_n is wired through the (0,1) cross-block below.
            globalJacobian.data()[full_offset] -= bc_mask.data()[i] * Theta * Kr * fmax(0.0, -TransportMatrix[full_offset]);
            J_ii += Theta * Kr * fmax(0.0, -TransportMatrix[full_offset]);
          }
          ith_flux_term += fL;
          fA -= fL;
          // (0,1) cross-block: d/du_n[node_j] of -fL.
          {
            const int row_w = offset_u + stride_u * i;
            const int col_n_up = offset_n + stride_n * node_j;
            for (int o = csrRowIndeces_Full.data()[row_w];
                 o < csrRowIndeces_Full.data()[row_w + 1]; o++) {
              if (csrColumnOffsets_Full.data()[o] == col_n_up) {
                globalJacobian.data()[o] -=
                    bc_mask.data()[i] * Theta * dkr_du_n_fct
                  * fmax(0.0, -TransportMatrix[full_offset]) * delta_phi;
                break;
              }
            }
          }
        }
        if (-TransportMatrixn[full_offset] * delta_phin <= 0.0) {
          // Previous step, upstream is i.
          evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, rho_i, beta, gravity.data(),
                                       alpha.data()[elementMaterialTypes.data()[0]],
                                       n.data()[elementMaterialTypes.data()[0]], thetaR.data()[elementMaterialTypes.data()[0]], thetaSR.data()[elementMaterialTypes.data()[0]], &KWs.data()[elementMaterialTypes.data()[0] * nnz],
                                       u_free_dof_old[i], u_dof_n_old.data()[node_i],
                                       m, dm, dm_du_n_fct, f, df, df_du_n_fct, a, da, da_du_n_fct, as, Kr, dKr, dkr_du_n_fct, thetaW_tmp);
          fL = (1 - Theta) * Kr * fmax(0.0, -TransportMatrixn[full_offset]) * delta_phin;
          fL_CN += (1 - Theta_h) * Kr * fmax(0.0, -TransportMatrixn[full_offset]) * delta_phin;
          ith_flux_term += fL;
          fA -= fL;
          fA_CN -= fL_CN;
        } else {
          // Previous step, upstream is j.
          evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, rho_j, beta, gravity.data(),
                                       alpha.data()[elementMaterialTypes.data()[0]],
                                       n.data()[elementMaterialTypes.data()[0]], thetaR.data()[elementMaterialTypes.data()[0]], thetaSR.data()[elementMaterialTypes.data()[0]], &KWs.data()[elementMaterialTypes.data()[0] * nnz],
                                       u_free_dof_old[j], u_dof_n_old.data()[node_j],
                                       m, dm, dm_du_n_fct, f, df, df_du_n_fct, a, da, da_du_n_fct, as, Kr, dKr, dkr_du_n_fct, thetaW_tmp);
          fL = (1 - Theta) * Kr * fmax(0.0, -TransportMatrixn[full_offset]) * delta_phin;
          fL_CN += (1 - Theta_h) * Kr * fmax(0.0, -TransportMatrixn[full_offset]) * delta_phin;
          ith_flux_term += fL;
          fA -= fL;
          fA_CN -= fL_CN;
        }
        dt_times_fH_minus_fL.data()[full_offset] = dt * fA;
        //dt_times_fH_minus_fL.data()[full_offset] = dt * fA_CN;
      }
      mDotLow.data()[i] = ith_flux_term/MLi;
      cflux[i] = ith_consistent_flux_term;
      // Final per-DOF coefficient evaluations at (u_w, u_n) and (u_w_old, u_n_old):
      //   m  -> mLow.data()[i]      (current low-order mass)
      //   mn -> mn.data()[i]        (time-history mass for the dt difference)
      //   dm_du_n_fct -> consumed by the (0,1) lumped-mass coupling below.
      evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, rho_i, beta, gravity.data(),
                                   alpha.data()[elementMaterialTypes.data()[0]],
                                   n.data()[elementMaterialTypes.data()[0]], thetaR.data()[elementMaterialTypes.data()[0]], thetaSR.data()[elementMaterialTypes.data()[0]], &KWs.data()[elementMaterialTypes.data()[0] * nnz],
                                   u_free_dof[i], u_dof_n.data()[node_i],
                                   m, dm, dm_du_n_fct, f, df, df_du_n_fct, a, da, da_du_n_fct, as, Kr, dKr, dkr_du_n_fct, thetaW_tmp);
      evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, rho_i, beta, gravity.data(),
                                   alpha.data()[elementMaterialTypes.data()[0]],
                                   n.data()[elementMaterialTypes.data()[0]], thetaR.data()[elementMaterialTypes.data()[0]], thetaSR.data()[elementMaterialTypes.data()[0]], &KWs.data()[elementMaterialTypes.data()[0] * nnz],
                                   u_free_dof_old[i], u_dof_n_old.data()[node_i],
                                   mn.data()[i], dmn, dmn_du_n_fct, fn, dfn, dfn_du_n_fct, an, dan, dan_du_n_fct, asn, Krn, dKrn, dkrn_du_n_fct, thetaW_tmp);
      mLow.data()[i] = m;
      globalResidual.data()[offset_u + stride_u * i] += bc_mask.data()[i] * (MLi * (m - mn.data()[i]) / dt - ith_flux_term);
      globalJacobian.data()[ii] += bc_mask.data()[i] * (MLi * dm / dt + J_ii) + (1.0 - bc_mask.data()[i]);
      // (0,1) cross-block: lumped-mass contribution MLi * dm/du_n / dt on the
      // (row_w=i, col_n=node_i) entry. This complements the per-edge
      // dkr/du_n * max(0,-T) * delta_phi contributions wired into (row_w=i,
      // col_n=upwind_node) inside the j-loop above. Together they make the
      // low-order operator fully consistent with d R_w / d u_n.
      {
        const int row_w   = offset_u + stride_u * i;
        const int col_n   = offset_n + stride_n * node_i;
        int off_wv = -1;
        for (int o = csrRowIndeces_Full.data()[row_w];
             o < csrRowIndeces_Full.data()[row_w + 1]; o++) {
          if (csrColumnOffsets_Full.data()[o] == col_n) { off_wv = o; break; }
        }
        if (off_wv >= 0) {
          globalJacobian.data()[off_wv] += bc_mask.data()[i] * MLi * dm_du_n_fct / dt;
        }
      }
    }
    // FCT limiter execution.
    //
    // Two operating modes:
    //   * STABILIZATION_TYPE == Implicit_FCT  (Richards-style, in-Newton FCT):
    //         comp-0's fluxCorrection is injected into globalResidual so
    //         Newton sees the limited residual. This is the legacy path; it
    //         relies on small enough alpha-derivative deviations to converge.
    //   * FCT_n == 1 with STABILIZATION_TYPE == EntropyViscosity (TADR-style
    //         defect-correction):
    //         Newton solves the LOW-ORDER R_low cleanly (no Zalesak
    //         contribution to globalResidual). The limiter is computed here
    //         so limited_solution_n and fluxCorrection_n are available, but
    //         the actual scatter to self.u[1].dof happens in Python after
    //         Newton convergence (Coefficients.postStep). This matches TADR /
    //         Richards Newton.solve flow where FCT is a post-step.
    //
    // Both branches still populate the comp-0 and comp-1 FCT outputs so
    // Python can use them.
    if (FCT_n == 1 || STABILIZATION_TYPE == STABILIZATION::Implicit_FCT) {
      FCTStep(args);
      FCTStep_n(args);
    }
    if (STABILIZATION_TYPE == STABILIZATION::Implicit_FCT) {
      // Legacy in-Newton injection (kept for STAB=Implicit_FCT only).
      for (int i = 0; i < numDOFs; i++) {
        globalResidual.data()[offset_u + stride_u * i] += fluxCorrection.data()[i];
      }
      for (int i_n = 0; i_n < numDOFs_n; i_n++) {
        globalResidual.data()[offset_n + stride_n * i_n] += fluxCorrection_n.data()[i_n];
      }
    }

    // ============================================================================
    // Component-1 (S_n) -- CG-Galerkin residual with EV dissipation (dH_n).
    //
    // The conservative form of the gas equation:
    //   d(phi*rho_n*S_n)/dt + div F_n = 0
    //   F_n = -a_n   . grad u_w  -  a_n_pc . grad u_n  +  f_n
    //   a_n     = rho_n   * k_rn(S_n) * K
    //   a_n_pc  = a_n     * dp_c/dS_n
    //   f_n     = rho_n^2 * k_rn(S_n) * K * g
    //
    // Layout:
    //   1. Per-DOF projection of m_n, mn_n, eta_n (entropy sensor on S_n_old).
    //   2. CELL LOOP: consistent CG residual (advection + diffusion vs grad u_w
    //      + capillary diffusion vs grad u_n) + lumped (m_n - m_n_old)/dt at
    //      element level. Element Jacobian (1,1) + (1,0) cross-block. Element
    //      transport matrix elementTransport_n collects the linearised (1,1)
    //      operator (cap_trial + sens) for the dLow_n / dEV_n stabilization.
    //   3. SMOOTHNESS SENSOR: alpha-indicator on m_n_old -> psi_n[i].
    //   4. EDGE LOOP: dLow_n[ij] = max(-T_n[ij], -T_n[ji], 0);
    //                 dEV_n[ij]  = cE * max(psi[i], psi[j]) * dLow_n[ij].
    //   5. DOF LOOP: low-order + antidiffusive-flux decomposition.
    //      - Newton residual: R_n[i] += sum_{j!=i} dLow_n[ij]*(m_n[i]-m_n[j])
    //        (pure low-order viscosity -- most dissipative, monotone).
    //      - Antidiffusive flux stored for FCT post-step:
    //          dt * (dLow_n - dEV_n) * (m_n[j] - m_n[i]).
    //        Fully unlimited (alpha=1) this would cancel dLow back down to dEV,
    //        recovering EV-level dissipation. Zalesak's limiter scales it per
    //        edge so bounds are preserved. With FCT disabled, the effective
    //        in-Newton dissipation stays at dLow_n.
    //      - Jacobian uses dLow_n consistently with the residual; chain rule
    //        dm_n/du_n applied at DOF nodes.
    //
    // Trade-off:
    //   * Coherent plume migration (CG-Galerkin spatial accuracy).
    //   * Small undershoots at sharp fronts (CG without FCT cannot enforce a
    //     strict discrete maximum principle; dH_n damps them but does not
    //     eliminate them).
    //   * The C++ FCTStep_n machinery is available; flipping FCT=True on the
    //     Coefficients class + a Python post-step would scatter limited_solution_n
    //     into self.u[1].dof and recover bound preservation.
    // ============================================================================

    // -------- Per-DOF nodal projection (m_n, mn_n, eta_n). --------
    // rho_n_phi_dof and ML_n were already projected above for the
    // invert(COMPONENT=1) path; reuse them here.
    //   m_n = phi * rho_n * u_n (= S_n).  invert: u_n = m_n / (phi*rho_n).
    std::vector<double> m_n_DOF(numDOFs_n, 0.0);
    std::vector<double> mn_n_DOF(numDOFs_n, 0.0);
    std::vector<double> eta_n(numDOFs_n, 0.0);
    std::vector<double> global_entropy_residual_n(numDOFs_n, 0.0);
    std::vector<double> psi_n(numDOFs_n, 0.0);
    for (int i_n = 0; i_n < numDOFs_n; i_n++) {
      const double sat     = u_dof_n.data()[i_n];      // S_n at DOF i (current)
      const double sat_old = u_dof_n_old.data()[i_n];  // S_n at DOF i (t^n)
      m_n_DOF[i_n]  = rho_n_phi_dof[i_n] * sat;
      mn_n_DOF[i_n] = rho_n_phi_dof[i_n] * sat_old;
      eta_n[i_n] = ENTROPY_TYPE == 1
                 ? ENTROPY(sat_old, u_n_L, u_n_R)
                 : ENTROPY_LOG(sat_old, u_n_L, u_n_R);
      global_entropy_residual_n[i_n] = 0.0;
      mn_n.data()[i_n]        = mn_n_DOF[i_n];     // diagnostic
      quantDOFs_n.data()[i_n] = 0.0;                // reset; smoothness fills below
    }

    // Per-call comp-1 transport matrix (sized full Jacobian NNZ, mirror of
    // comp-0). Only the (1,1)-block-mapped entries are written.
    std::vector<double> TransportMatrix_n(NNZ, 0.0);

    for (int eN = 0; eN < nElements_global; eN++) {
      const int    mat_eN    = elementMaterialTypes.data()[eN];
      const double phi_eN    = thetaR.data()[mat_eN] + thetaSR.data()[mat_eN];
      const double alpha_eN  = alpha.data()[mat_eN];
      const double n_vg_eN   = n.data()[mat_eN];
      const double *KWs_eN   = &KWs.data()[mat_eN * nnz];
      const double dm_n_du_n = phi_eN * rho_n;       // d(phi*rho_n*S_n)/dS_n
      double elementResidual_n[nDOF_test_element];
      double elementMass_n[nDOF_test_element];
      double u_n_local[nDOF_trial_element];
      double u_n_old_local[nDOF_trial_element];
      double elementJacobian_n_n[nDOF_test_element][nDOF_trial_element];
      double elementJacobian_n_w[nDOF_test_element][nDOF_trial_element];
      // elementTransport_n collects ONLY the linear capillary-diffusion piece
      // (cap_trial_ij = int a_n_p_c grad N_j . grad N_i dV) -- the
      // sign-correct, symmetric operator on u_n that dLow_n stabilizes. The
      // full elementJacobian_n_n (including nonlinear sensitivities) is
      // assembled separately and is what feeds the global Jacobian.
      double elementTransport_n[nDOF_test_element][nDOF_trial_element];
      const int eN_nDOF_trial_element = eN * nDOF_trial_element;
      for (int i = 0; i < nDOF_test_element; i++) {
        elementResidual_n[i] = 0.0;
        elementMass_n[i]     = 0.0;
        for (int j = 0; j < nDOF_trial_element; j++) {
          elementJacobian_n_n[i][j] = 0.0;
          elementJacobian_n_w[i][j] = 0.0;
          elementTransport_n[i][j]  = 0.0;
        }
      }
      for (int j = 0; j < nDOF_trial_element; j++) {
        u_n_local[j]     = u_dof_n.data()[u_l2g.data()[eN_nDOF_trial_element + j]];
        u_n_old_local[j] = u_dof_n_old.data()[u_l2g.data()[eN_nDOF_trial_element + j]];
      }
      for (int k = 0; k < nQuadraturePoints_element; k++) {
        double jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace], x_q, y_q, z_q;
        ck.calculateMapping_element(eN, k, mesh_dof.data(), mesh_l2g.data(),
                                    mesh_trial_ref.data(), mesh_grad_trial_ref.data(),
                                    jac, jacDet, jacInv, x_q, y_q, z_q);
        const double dV = std::fabs(jacDet) * dV_ref.data()[k];
        double u_grad_trial_qp[nDOF_trial_element * nSpace];
        ck.gradTrialFromRef(&u_grad_trial_ref.data()[k * nDOF_trial_element * nSpace],
                            jacInv, u_grad_trial_qp);
        double u_n = 0.0, u_n_old = 0.0;
        ck.valFromDOF(u_dof_n.data(),
                      &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_n);
        ck.valFromDOF(u_dof_n_old.data(),
                      &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_n_old);
        double grad_u_w[nSpace], grad_u_n[nSpace];
        ck.gradFromDOF(u_dof.data(),
                       &u_l2g.data()[eN_nDOF_trial_element],
                       u_grad_trial_qp, grad_u_w);
        ck.gradFromDOF(u_dof_n.data(),
                       &u_l2g.data()[eN_nDOF_trial_element],
                       u_grad_trial_qp, grad_u_n);
        // u_n is S_n; closures take wetting effective saturation S_e.
        const double phi_loc      = thetaR.data()[mat_eN] + thetaSR.data()[mat_eN];
        const double S_wr_loc     = thetaR.data()[mat_eN] / phi_loc;
        const double one_m_Sr_loc = 1.0 - S_wr_loc;
        // Se = (S_w - S_wr)/(1-S_wr) = (1 - u_n - S_wr)/(1 - S_wr).
        const double Se_qp_raw    = (1.0 - u_n - S_wr_loc) / one_m_Sr_loc;
        double Se_qp, dSe_du_n_loc;
        if (Se_qp_raw <= 0.0)      { Se_qp = 0.0;        dSe_du_n_loc = 0.0; }
        else if (Se_qp_raw >= 1.0) { Se_qp = 1.0;        dSe_du_n_loc = 0.0; }
        else                       { Se_qp = Se_qp_raw;  dSe_du_n_loc = -1.0 / one_m_Sr_loc; }
        double KNr = 0.0, DKNr_DSe = 0.0;
        if (PSK_TYPE_member == 1) {
          proteus::mphase_co2::psk::bc_kr_nonwetting_from_Se(
              Se_qp, alpha_eN, n_vg_eN, KNr, DKNr_DSe);
        } else {
          proteus::mphase_co2::psk::vgm_kr_nonwetting_from_Se(
              Se_qp, alpha_eN, n_vg_eN, KNr, DKNr_DSe);
        }
        // Chain rule: convert d/dSe -> d/du_n. dSe_du_n_loc < 0 flips sign.
        DKNr_DSe *= dSe_du_n_loc;
        // Capillary derivative: closure returns dp_c/dSe < 0 (monotone), so
        // after chain rule with dSe_du_n_loc < 0 we get dp_c/d(u_n) > 0.
        // 'dpc_dSw' is kept as the variable name; semantically it is dp_c/dS_n.
        // d2pc_dSw2 multiplied by dSe_du_n_loc^2 stays positive (= d2p_c/dS_n^2).
        double pc_qp = 0.0, dpc_dSw = 0.0, d2pc_dSw2 = 0.0;
        if (PSK_TYPE_member == 1) {
          proteus::mphase_co2::psk::bc_pc_from_Se(Se_qp, alpha_eN, n_vg_eN, pc_qp, dpc_dSw, d2pc_dSw2);
        } else {
          proteus::mphase_co2::psk::vgm_pc_from_Se(Se_qp, alpha_eN, n_vg_eN, pc_qp, dpc_dSw, d2pc_dSw2);
        }
        dpc_dSw   *= dSe_du_n_loc;
        d2pc_dSw2 *= dSe_du_n_loc * dSe_du_n_loc;
        double a_n[nnz], da_n_du_n[nnz];
        double a_n_p_c[nnz], da_n_p_c_du_n[nnz];
        double f_n[nSpace], df_n_du_n[nSpace];
        for (int I = 0; I < nSpace; I++) { f_n[I] = 0.0; df_n_du_n[I] = 0.0; }
        for (int I = 0; I < nSpace; I++) {
          for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I + 1]; ii++) {
            const int J = a_colind.data()[ii];
            a_n[ii]           = rho_n * KNr * KWs_eN[ii];
            da_n_du_n[ii]     = rho_n * DKNr_DSe * KWs_eN[ii];
            a_n_p_c[ii]       = a_n[ii] * dpc_dSw;
            da_n_p_c_du_n[ii] = da_n_du_n[ii] * dpc_dSw + a_n[ii] * d2pc_dSw2;
            f_n[I]       += rho_n * rho_n * KNr * KWs_eN[ii] * gravity.data()[J];
            df_n_du_n[I] += rho_n * rho_n * DKNr_DSe * KWs_eN[ii] * gravity.data()[J];
          }
        }
        for (int i = 0; i < nDOF_test_element; i++) {
          const double test_i = u_test_ref.data()[k * nDOF_test_element + i];
          elementMass_n[i] += test_i * dV;
          // Consistent residual (NO mass term -- handled lumped below): advection
          // + diffusion(grad u_w) + capillary diffusion(grad u_n).
          for (int I = 0; I < nSpace; I++) {
            elementResidual_n[i] += f_n[I] * u_grad_trial_qp[i * nSpace + I] * dV;
          }
          for (int I = 0; I < nSpace; I++) {
            for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I + 1]; ii++) {
              const int J = a_colind.data()[ii];
              elementResidual_n[i] += a_n[ii]     * grad_u_w[J]
                                    * u_grad_trial_qp[i * nSpace + I] * dV;
              elementResidual_n[i] += a_n_p_c[ii] * grad_u_n[J]
                                    * u_grad_trial_qp[i * nSpace + I] * dV;
            }
          }
          // (1,1) consistent operator K_vv: flux-coefficient sensitivities
          // through k_rn(u_n) and dp_c/dS_w + capillary trial-fn variation.
          double diff_coef_sens_i = 0.0;
          double cap_coef_sens_i  = 0.0;
          for (int I = 0; I < nSpace; I++) {
            const double grad_Ni_I = u_grad_trial_qp[i * nSpace + I];
            for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I + 1]; ii++) {
              const int J = a_colind.data()[ii];
              diff_coef_sens_i += da_n_du_n[ii]     * grad_u_w[J] * grad_Ni_I;
              cap_coef_sens_i  += da_n_p_c_du_n[ii] * grad_u_n[J] * grad_Ni_I;
            }
          }
          double adv_coef_sens_i = 0.0;
          for (int I = 0; I < nSpace; I++) {
            adv_coef_sens_i += df_n_du_n[I] * u_grad_trial_qp[i * nSpace + I];
          }
          for (int j = 0; j < nDOF_trial_element; j++) {
            const double trial_j = u_trial_ref.data()[k * nDOF_trial_element + j];
            // K_vv -- consistent flux + capillary sensitivity (no mass).
            const double sens_ij = (adv_coef_sens_i + diff_coef_sens_i + cap_coef_sens_i)
                                 * trial_j * dV;
            elementJacobian_n_n[i][j] += sens_ij;
            double cap_trial_ij = 0.0;
            for (int I = 0; I < nSpace; I++) {
              const double grad_Ni_I = u_grad_trial_qp[i * nSpace + I];
              for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I + 1]; ii++) {
                const int J = a_colind.data()[ii];
                cap_trial_ij += a_n_p_c[ii] * u_grad_trial_qp[j * nSpace + J] * grad_Ni_I;
              }
            }
            elementJacobian_n_n[i][j] += cap_trial_ij * dV;
            // EV transport operator. Feed the FULL linearized (1,1) coupling
            // into TransportMatrix_n so dLow_n stabilizes against every
            // transport-like channel: the symmetric linear capillary
            // diffusion AND the gravity / cross-coupling / dp_c-curvature
            // sensitivities (adv_coef_sens_i = (df_n/du_n).grad N_i,
            // diff_coef_sens_i = (da_n/du_n).grad u_w.grad N_i,
            // cap_coef_sens_i = (da_n_p_c/du_n).grad u_n.grad N_i). With this
            // expansion T_n sees the same operator as the Jacobian, so
            // dLow_n = max(-T[ij], -T[ji], 0) builds a Kuzmin low-order
            // monotone update that preserves the discrete maximum principle
            // on m_n (S_n stays in [0, 1 - S_wr] up to boundary fluxes).
            elementTransport_n[i][j] += cap_trial_ij * dV + sens_ij;
            // (1,0) cross-block: diffusion trial-fn variation against grad u_w.
            double diff_trial_ij = 0.0;
            for (int I = 0; I < nSpace; I++) {
              const double grad_Ni_I = u_grad_trial_qp[i * nSpace + I];
              for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I + 1]; ii++) {
                const int J = a_colind.data()[ii];
                diff_trial_ij += a_n[ii] * u_grad_trial_qp[j * nSpace + J] * grad_Ni_I;
              }
            }
            elementJacobian_n_w[i][j] += diff_trial_ij * dV;
          }
        }
      } // end QP loop

      // -------- Lumped mass: ML_n[i] * (m_n - mn_n)/dt on the diagonal. --------
      // Applied at element level so it sums to the global lumped mass.
      for (int i = 0; i < nDOF_test_element; i++) {
        const double m_n_loc     = phi_eN * rho_n * u_n_local[i];
        const double m_n_old_loc = phi_eN * rho_n * u_n_old_local[i];
        elementResidual_n[i]      += elementMass_n[i] * (m_n_loc - m_n_old_loc) / dt;
        elementJacobian_n_n[i][i] += elementMass_n[i] * dm_n_du_n / dt;
      }

      // -------- Distribute element arrays to global storage. --------
      for (int i = 0; i < nDOF_test_element; i++) {
        const int eN_i = eN * nDOF_test_element + i;
        const int gi   = u_l2g.data()[eN_i];
        globalResidual.data()[offset_n + stride_n * gi] += elementResidual_n[i];
        for (int j = 0; j < nDOF_trial_element; j++) {
          const int eN_i_j = eN_i * nDOF_trial_element + j;
          globalJacobian.data()[csrRowIndeces_n_n.data()[eN_i] + csrColumnOffsets_n_n.data()[eN_i_j]]
              += elementJacobian_n_n[i][j];
          globalJacobian.data()[csrRowIndeces_n_w.data()[eN_i] + csrColumnOffsets_n_w.data()[eN_i_j]]
              += elementJacobian_n_w[i][j];
          // Feed the linear capillary-diffusion piece into the global
          // TransportMatrix_n for dLow_n construction below.
          TransportMatrix_n[csrRowIndeces_n_n.data()[eN_i] + csrColumnOffsets_n_n.data()[eN_i_j]]
              += elementTransport_n[i][j];
        }
      }
    }
    // ============================================================================
    // Comp-1 EV pipeline (post-element-loop):
    //   1. Smoothness sensor psi_n[i] from local DOF differences on u_n_old.
    //   2. dLow_n / dEV_n on the comp-1 DOF graph.
    //   3. dH_n = min(dLow_n, dEV_n) stabilization added to residual + Jac.
    // ============================================================================

    // -------- Smoothness sensor on S_w_old (mirrors comp-0's alpha indicator). --------
    // Uses the same Cx / Cy / Cz consistent advection matrices and ML lumped
    // mass as comp-0 -- both components share the FE space and mesh.
    for (int i_n = 0; i_n < numDOFs_n; i_n++) {
      double gi[nSpace], Cij[nSpace];
      const double solni_n = u_dof_n_old.data()[i_n];
      for (int I = 0; I < nSpace; I++) gi[I] = 0.0;
      double alpha_numerator_pos = 0., alpha_numerator_neg = 0.;
      double alpha_denominator_pos = 0., alpha_denominator_neg = 0.;
      // First DOF loop: build gi vector + alpha numerator/denominator.
      // CSR for comp-1 uses the same sparsity offsets into the FULL Jacobian
      // (Cx/Cy/Cz are full-NNZ-sized): the (i_n, j_n) full offset is found by
      // inline search through csrRowIndeces_Full at the full row offset_n +
      // stride_n * i_n.
      const int full_row_i = offset_n + stride_n * i_n;
      for (int offset = csrRowIndeces_n_DofLoops.data()[i_n];
           offset < csrRowIndeces_n_DofLoops.data()[i_n + 1]; offset++) {
        const int j_n        = csrColumnOffsets_n_DofLoops.data()[offset];
        const int full_col_j = offset_n + stride_n * j_n;
        int full_offset_ij = -1;
        for (int o = csrRowIndeces_Full.data()[full_row_i];
             o < csrRowIndeces_Full.data()[full_row_i + 1]; o++) {
          if (csrColumnOffsets_Full.data()[o] == full_col_j) { full_offset_ij = o; break; }
        }
        if (full_offset_ij < 0) continue;
        const double solnj_n = u_dof_n_old.data()[j_n];
        Cij[0] = Cx[full_offset_ij];
#if nSpace == 2
        Cij[1] = Cy[full_offset_ij];
#endif
#if nSpace == 3
        Cij[2] = Cz[full_offset_ij];
#endif
        for (int I = 0; I < nSpace; I++) gi[I] += Cij[I] * solnj_n;
        const double alpha_num = solni_n - solnj_n;
        if (alpha_num >= 0.) {
          alpha_numerator_pos   += alpha_num;
          alpha_denominator_pos += alpha_num;
        } else {
          alpha_numerator_neg   += alpha_num;
          alpha_denominator_neg += std::fabs(alpha_num);
        }
      }
      // ML.data() is sized numDOFs (comp-0 lumped mass). For comp-1 we use
      // rho_n_phi_dof's matching ML_n projection: the same lumped diagonal is
      // ML_n_glob, which equals integral of N_i over the mesh -- safe to use
      // ML.data() here because both blocks share the same FE space when
      // numDOFs == numDOFs_u. Fall back to 1.0 if out of range.
      const double ML_i = (i_n < numDOFs) ? ML.data()[i_n] : 1.0;
      for (int I = 0; I < nSpace; I++) gi[I] /= (ML_i > 0.0 ? ML_i : 1.0);
      // Second DOF loop: compute SumPos / SumNeg for sigma cancellation.
      double SumPos = 0., SumNeg = 0.;
      for (int offset = csrRowIndeces_n_DofLoops.data()[i_n];
           offset < csrRowIndeces_n_DofLoops.data()[i_n + 1]; offset++) {
        const int j_n        = csrColumnOffsets_n_DofLoops.data()[offset];
        const int full_col_j = offset_n + stride_n * j_n;
        int full_offset_ij = -1;
        for (int o = csrRowIndeces_Full.data()[full_row_i];
             o < csrRowIndeces_Full.data()[full_row_i + 1]; o++) {
          if (csrColumnOffsets_Full.data()[o] == full_col_j) { full_offset_ij = o; break; }
        }
        if (full_offset_ij < 0) continue;
        double gi_times_x = 0.0;
        for (int I = 0; I < nSpace; I++) {
          gi_times_x += gi[I] * delta_x_ij.data()[full_offset_ij * 3 + I];
        }
        SumPos += gi_times_x > 0 ? gi_times_x : 0;
        SumNeg += gi_times_x < 0 ? gi_times_x : 0;
      }
      const double sigmaPos = std::min(1.0, (std::fabs(SumNeg) + 1e-15) / (SumPos + 1e-15));
      const double sigmaNeg = std::min(1.0, (SumPos + 1e-15) / (std::fabs(SumNeg) + 1e-15));
      double alpha_num = std::fabs(sigmaPos * alpha_numerator_pos + sigmaNeg * alpha_numerator_neg);
      double alpha_den = sigmaPos * alpha_denominator_pos + sigmaNeg * alpha_denominator_neg;
      if (IS_BETAij_ONE == 1) {
        alpha_num = std::fabs(alpha_numerator_pos + alpha_numerator_neg);
        alpha_den = alpha_denominator_pos + alpha_denominator_neg;
      }
      const double alpha_i = alpha_num / (alpha_den + 1e-15);
      quantDOFs_n.data()[i_n] = alpha_i;
      psi_n[i_n] = (POWER_SMOOTHNESS_INDICATOR == 0) ? 1.0
                                                     : std::pow(alpha_i, POWER_SMOOTHNESS_INDICATOR);
    }

    // -------- Edge loop: dLow_n_ij and dEV_n_ij from TransportMatrix_n. --------
    // Kuzmin algebraic flux correction on the (1,1) Jacobian T_n = dR/du_n.
    // Off-diagonal Jac contribution from the dH*(m[i]-m[j]) dissipation is
    //   -dH_ij * rho_n_phi_dof[j_n]  (line 3382),
    // so for an M-matrix linearization we need
    //   T_ij - dH_ij * rho_n_phi_dof[j_n] <= 0  AND  T_ji - dH_ij * rho_n_phi_dof[i_n] <= 0.
    // Symmetric dH (dH_ij = dH_ji) is needed for mass conservation, so take:
    //   dH_ij = max( T_ij / rho_phi[j_n],  T_ji / rho_phi[i_n],  0 ).
    // Note the +T sign (NOT -T): when both T_ij and T_ji are positive (precisely
    // where the consistent Galerkin operator violates the maximum principle on
    // u_n, e.g. non-acute meshes or convection-dominated edges) this gives the
    // correct positive dissipation. The previous max(-T_ij, -T_ji, 0) silently
    // returned zero exactly in those regions.
    for (int i_n = 0; i_n < numDOFs_n; i_n++) {
      const int full_row_i = offset_n + stride_n * i_n;
      for (int offset = csrRowIndeces_n_DofLoops.data()[i_n];
           offset < csrRowIndeces_n_DofLoops.data()[i_n + 1]; offset++) {
        const int j_n        = csrColumnOffsets_n_DofLoops.data()[offset];
        if (i_n == j_n) { dLow_n.data()[offset] = 0.0; dEV_n.data()[offset] = 0.0; continue; }
        // Find full Jacobian offsets for T_n[i][j] and T_n[j][i].
        const int full_col_j = offset_n + stride_n * j_n;
        const int full_row_j = offset_n + stride_n * j_n;
        const int full_col_i = offset_n + stride_n * i_n;
        int full_offset_ij = -1, full_offset_ji = -1;
        for (int o = csrRowIndeces_Full.data()[full_row_i];
             o < csrRowIndeces_Full.data()[full_row_i + 1]; o++) {
          if (csrColumnOffsets_Full.data()[o] == full_col_j) { full_offset_ij = o; break; }
        }
        for (int o = csrRowIndeces_Full.data()[full_row_j];
             o < csrRowIndeces_Full.data()[full_row_j + 1]; o++) {
          if (csrColumnOffsets_Full.data()[o] == full_col_i) { full_offset_ji = o; break; }
        }
        const double T_ij = (full_offset_ij >= 0) ? TransportMatrix_n[full_offset_ij] : 0.0;
        const double T_ji = (full_offset_ji >= 0) ? TransportMatrix_n[full_offset_ji] : 0.0;
        const double inv_rho_phi_j = 1.0 / std::max(rho_n_phi_dof[j_n], 1.0e-14);
        const double inv_rho_phi_i = 1.0 / std::max(rho_n_phi_dof[i_n], 1.0e-14);
        dLow_n.data()[offset] = std::max({T_ij * inv_rho_phi_j, T_ji * inv_rho_phi_i, 0.0});
        dEV_n.data()[offset]  = cE * std::max(psi_n[i_n], psi_n[j_n]) * dLow_n.data()[offset];
      }
    }

    // -------- DOF loop: add dH_n = min(dLow_n, dEV_n) stabilization. --------
    // Acts on the conserved m_n via R[i] += sum_{j != i} dH_n[ij] * (m_n[i] - m_n[j]).
    // Jacobian chain rule: dm_n/du_n[i] = -rho_n_phi_dof[i] (DOF-dependent).
    for (int i_n = 0; i_n < numDOFs_n; i_n++) {
      double ith_flux_term_n = 0.0;
      double J_n_ii          = 0.0;
      const int full_row_i = offset_n + stride_n * i_n;
      for (int offset = csrRowIndeces_n_DofLoops.data()[i_n];
           offset < csrRowIndeces_n_DofLoops.data()[i_n + 1]; offset++) {
        const int j_n = csrColumnOffsets_n_DofLoops.data()[offset];
        if (i_n == j_n) {
          dt_times_fH_minus_fL_n.data()[offset] = 0.0;
          continue;
        }
        // Low-order monotone dissipation in the residual (Kuzmin first-order).
        // The (dLow - dEV) gap is stored as an antidiffusive flux that
        // FCTStep_n will subsequently limit with Zalesak's algorithm.
        const double dH_ij = dLow_n.data()[offset];
        ith_flux_term_n += dH_ij * (m_n_DOF[i_n] - m_n_DOF[j_n]);
        // Antidiffusive-flux storage for FCT (matches comp-0's fH - fL
        // convention: positive value pushes m_n[i] toward higher mass).
        dt_times_fH_minus_fL_n.data()[offset] =
            dt * (dLow_n.data()[offset] - dEV_n.data()[offset])
               * (m_n_DOF[j_n] - m_n_DOF[i_n]);
        // Jacobian off-diagonal (1,1): d/du_n[j] of dH*(m_n[i]-m_n[j]) =
        //                              dH * (-dm_n/du_n[j]) = -dH * rho_n_phi_dof[j_n].
        // (m_n = phi*rho_n*u_n  =>  dm_n/du_n = +phi*rho_n.)
        const int full_col_j = offset_n + stride_n * j_n;
        int full_offset_ij = -1;
        for (int o = csrRowIndeces_Full.data()[full_row_i];
             o < csrRowIndeces_Full.data()[full_row_i + 1]; o++) {
          if (csrColumnOffsets_Full.data()[o] == full_col_j) { full_offset_ij = o; break; }
        }
        if (full_offset_ij >= 0) {
          globalJacobian.data()[full_offset_ij] += -dH_ij * rho_n_phi_dof[j_n];
        }
        // Jacobian diagonal accumulate: d/du_n[i] of dH*(m_n[i]-m_n[j]) =
        //                               dH * (dm_n/du_n[i]) = +dH * rho_n_phi_dof[i_n].
        J_n_ii += dH_ij * rho_n_phi_dof[i_n];
      }
      globalResidual.data()[offset_n + stride_n * i_n] += ith_flux_term_n;
      // FCT predictor state: the current iterate is the low-order m_n at
      // t^{n+1} (residual is built with pure dLow stabilization). mDotLow_n
      // is the corresponding lumped-mass time derivative consumed by
      // FCTStep_n's consistency term.
      mLow_n.data()[i_n]    = m_n_DOF[i_n];
      mDotLow_n.data()[i_n] = (m_n_DOF[i_n] - mn_n.data()[i_n]) / dt;
      // Diagonal (1,1) full Jacobian offset.
      int full_offset_ii = -1;
      for (int o = csrRowIndeces_Full.data()[full_row_i];
           o < csrRowIndeces_Full.data()[full_row_i + 1]; o++) {
        if (csrColumnOffsets_Full.data()[o] == full_row_i) { full_offset_ii = o; break; }
      }
      if (full_offset_ii >= 0) {
        globalJacobian.data()[full_offset_ii] += J_n_ii;
      }
    }

    // ============================================================================
    // Comp-1 (S_n) exterior boundary loop.
    //
    // Mirrors the interior gas-eq flux structure at element boundaries:
    //   F_n . n = -a_n   . grad u_w . n      (Darcy diffusion against grad p_w)
    //             -a_n_pc . grad u_n . n     (capillary diffusion against grad S_n)
    //             +f_n   . n                 (gas gravity flux)
    // Dirichlet on u_n is enforced via a Nitsche-style penalty added to the
    // boundary flux. Faces with no Dirichlet on u_n contribute the interior
    // flux directly -- consistent with how the interior CG operator already
    // sees grad_u_n at trace QPs through the element-volume gradient.
    //
    // Jacobian contributions:
    //   (1,1) self    : coefficient sens (df_n/du_n, da_n/du_n, da_n_pc/du_n)
    //                   + trial-fn variation of a_n_pc * grad N_j . n
    //                   + Dirichlet penalty * trial_j.
    //   (1,0) cross   : trial-fn variation of -a_n * grad N_j . n.
    //
    // Closure dispatch uses PSK_TYPE_member already set at the top of this
    // routine.
    // ============================================================================
    for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++) {
      const int ebN = exteriorElementBoundariesArray.data()[ebNE];
      const int eN  = elementBoundaryElementsArray.data()[ebN * 2 + 0];
      const int ebN_local = elementBoundaryLocalElementBoundariesArray.data()[ebN * 2 + 0];
      const int eN_nDOF_trial_element = eN * nDOF_trial_element;
      const int    mat_eN    = elementMaterialTypes.data()[eN];
      const double phi_eN    = thetaR.data()[mat_eN] + thetaSR.data()[mat_eN];
      const double alpha_eN  = alpha.data()[mat_eN];
      const double n_vg_eN   = n.data()[mat_eN];
      const double *KWs_eN   = &KWs.data()[mat_eN * nnz];
      const double S_wr_loc      = thetaR.data()[mat_eN] / phi_eN;
      const double one_m_Sr_loc  = 1.0 - S_wr_loc;

      double elementResidual_n_eb[nDOF_test_element];
      double elementJacobian_n_n_eb[nDOF_test_element][nDOF_trial_element];
      double elementJacobian_n_w_eb[nDOF_test_element][nDOF_trial_element];
      for (int i = 0; i < nDOF_test_element; i++) {
        elementResidual_n_eb[i] = 0.0;
        for (int j = 0; j < nDOF_trial_element; j++) {
          elementJacobian_n_n_eb[i][j] = 0.0;
          elementJacobian_n_w_eb[i][j] = 0.0;
        }
      }

      for (int kb = 0; kb < nQuadraturePoints_elementBoundary; kb++) {
        const int ebNE_kb            = ebNE * nQuadraturePoints_elementBoundary + kb;
        const int ebN_local_kb       = ebN_local * nQuadraturePoints_elementBoundary + kb;
        const int ebN_local_kb_nSpace = ebN_local_kb * nSpace;

        double jac_ext[nSpace * nSpace], jacDet_ext, jacInv_ext[nSpace * nSpace];
        double boundaryJac_b[nSpace * (nSpace - 1)];
        double metricTensor_b[(nSpace - 1) * (nSpace - 1)];
        double metricTensorDetSqrt_b, dS_eb, normal_b[3];
        double xt_b, yt_b, zt_b, integralScaling_b;
        double x_eb, y_eb, z_eb;
        ck.calculateMapping_elementBoundary(eN, ebN_local, kb, ebN_local_kb,
            mesh_dof.data(), mesh_l2g.data(), mesh_trial_trace_ref.data(),
            mesh_grad_trial_trace_ref.data(), boundaryJac_ref.data(),
            jac_ext, jacDet_ext, jacInv_ext, boundaryJac_b, metricTensor_b,
            metricTensorDetSqrt_b, normal_ref.data(), normal_b,
            x_eb, y_eb, z_eb);
        ck.calculateMappingVelocity_elementBoundary(eN, ebN_local, kb, ebN_local_kb,
            mesh_velocity_dof.data(), mesh_l2g.data(), mesh_trial_trace_ref.data(),
            xt_b, yt_b, zt_b, normal_b, boundaryJac_b, metricTensor_b,
            integralScaling_b);
        dS_eb = ((1.0 - MOVING_DOMAIN) * metricTensorDetSqrt_b
                + MOVING_DOMAIN * integralScaling_b) * dS_ref.data()[kb];

        // Trace solution and gradients.
        double u_grad_trial_trace_b[nDOF_trial_element * nSpace];
        ck.gradTrialFromRef(
            &u_grad_trial_trace_ref.data()[ebN_local_kb_nSpace * nDOF_trial_element],
            jacInv_ext, u_grad_trial_trace_b);
        double u_w_ext_b = 0.0, u_n_ext_b = 0.0;
        double grad_u_w_ext_b[nSpace], grad_u_n_ext_b[nSpace];
        ck.valFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element],
                      u_w_ext_b);
        ck.valFromDOF(u_dof_n.data(), &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element],
                      u_n_ext_b);
        ck.gradFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element],
                       u_grad_trial_trace_b, grad_u_w_ext_b);
        ck.gradFromDOF(u_dof_n.data(), &u_l2g.data()[eN_nDOF_trial_element],
                       u_grad_trial_trace_b, grad_u_n_ext_b);

        // Dirichlet target for u_n (no-op when isDOFBoundary_n == 0).
        const int    isDir_n      = isDOFBoundary_n.data()[ebNE_kb];
        const double bc_u_n_ext_b = isDir_n * ebqe_bc_u_n_ext.data()[ebNE_kb]
                                  + (1 - isDir_n) * u_n_ext_b;

        // Closure at the trace, in S_n form.
        const double Se_b_raw = (1.0 - u_n_ext_b - S_wr_loc) / one_m_Sr_loc;
        double Se_b, dSe_du_n_b;
        if (Se_b_raw <= 0.0)      { Se_b = 0.0; dSe_du_n_b = 0.0; }
        else if (Se_b_raw >= 1.0) { Se_b = 1.0; dSe_du_n_b = 0.0; }
        else                      { Se_b = Se_b_raw; dSe_du_n_b = -1.0 / one_m_Sr_loc; }
        double KNr_b = 0.0, DKNr_b = 0.0;
        if (PSK_TYPE_member == 1) {
          proteus::mphase_co2::psk::bc_kr_nonwetting_from_Se(
              Se_b, alpha_eN, n_vg_eN, KNr_b, DKNr_b);
        } else {
          proteus::mphase_co2::psk::vgm_kr_nonwetting_from_Se(
              Se_b, alpha_eN, n_vg_eN, KNr_b, DKNr_b);
        }
        DKNr_b *= dSe_du_n_b;
        double pc_b = 0.0, dpc_dSn_b = 0.0, d2pc_dSn2_b = 0.0;
        if (PSK_TYPE_member == 1) {
          proteus::mphase_co2::psk::bc_pc_from_Se(
              Se_b, alpha_eN, n_vg_eN, pc_b, dpc_dSn_b, d2pc_dSn2_b);
        } else {
          proteus::mphase_co2::psk::vgm_pc_from_Se(
              Se_b, alpha_eN, n_vg_eN, pc_b, dpc_dSn_b, d2pc_dSn2_b);
        }
        dpc_dSn_b   *= dSe_du_n_b;
        d2pc_dSn2_b *= dSe_du_n_b * dSe_du_n_b;

        // Build flux coefficients and sensitivities at the trace.
        double a_n_b[nnz], da_n_du_n_b[nnz];
        double a_n_pc_b[nnz], da_n_pc_du_n_b[nnz];
        double f_n_b[nSpace], df_n_du_n_b[nSpace];
        for (int I = 0; I < nSpace; I++) {
          f_n_b[I] = 0.0; df_n_du_n_b[I] = 0.0;
        }
        for (int I = 0; I < nSpace; I++) {
          for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I + 1]; ii++) {
            const int J = a_colind.data()[ii];
            a_n_b[ii]         = rho_n * KNr_b * KWs_eN[ii];
            da_n_du_n_b[ii]   = rho_n * DKNr_b * KWs_eN[ii];
            a_n_pc_b[ii]      = a_n_b[ii] * dpc_dSn_b;
            da_n_pc_du_n_b[ii] = da_n_du_n_b[ii] * dpc_dSn_b
                               + a_n_b[ii] * d2pc_dSn2_b;
            f_n_b[I]       += rho_n * rho_n * KNr_b * KWs_eN[ii] * gravity.data()[J];
            df_n_du_n_b[I] += rho_n * rho_n * DKNr_b * KWs_eN[ii] * gravity.data()[J];
          }
        }

        // F_n . n at this QP (without the penalty term, which depends only
        // on the test function row).
        double F_n_dot_n = 0.0;
        for (int I = 0; I < nSpace; I++) {
          F_n_dot_n += f_n_b[I] * normal_b[I];
          for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I + 1]; ii++) {
            const int J = a_colind.data()[ii];
            F_n_dot_n -= a_n_b[ii]    * grad_u_w_ext_b[J] * normal_b[I];
            F_n_dot_n -= a_n_pc_b[ii] * grad_u_n_ext_b[J] * normal_b[I];
          }
        }
        const double penalty = ebqe_penalty_ext.data()[ebNE_kb];
        if (isDir_n) {
          // Nitsche penalty drives u_n at the trace toward the prescribed BC.
          F_n_dot_n += penalty * (u_n_ext_b - bc_u_n_ext_b);
        }

        // Coefficient-sensitivity precomputes (independent of trial j).
        double adv_sens_n_b = 0.0;
        double diff_sens_n_b = 0.0;
        double cap_sens_n_b  = 0.0;
        for (int I = 0; I < nSpace; I++) {
          adv_sens_n_b += df_n_du_n_b[I] * normal_b[I];
          for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I + 1]; ii++) {
            const int J = a_colind.data()[ii];
            diff_sens_n_b -= da_n_du_n_b[ii]    * grad_u_w_ext_b[J] * normal_b[I];
            cap_sens_n_b  -= da_n_pc_du_n_b[ii] * grad_u_n_ext_b[J] * normal_b[I];
          }
        }
        const double sens_total = adv_sens_n_b + diff_sens_n_b + cap_sens_n_b;

        for (int i = 0; i < nDOF_test_element; i++) {
          const double test_i_dS = u_test_trace_ref.data()[
              ebN_local_kb * nDOF_test_element + i] * dS_eb;
          // Residual contribution.
          elementResidual_n_eb[i] += F_n_dot_n * test_i_dS;
          // Jacobian (per trial j).
          for (int j = 0; j < nDOF_trial_element; j++) {
            const double trial_j_b = u_trial_trace_ref.data()[
                ebN_local_kb * nDOF_test_element + j];
            // (1,1) self: coefficient sensitivity * trial_j.
            double jac_nn = sens_total * trial_j_b;
            // (1,1) self: capillary trial-fn variation (-a_n_pc * grad N_j . n).
            double cap_trial = 0.0;
            for (int I = 0; I < nSpace; I++) {
              for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I + 1]; ii++) {
                const int J = a_colind.data()[ii];
                cap_trial -= a_n_pc_b[ii] * u_grad_trial_trace_b[j * nSpace + J]
                                          * normal_b[I];
              }
            }
            jac_nn += cap_trial;
            // (1,1) self: Nitsche penalty.
            if (isDir_n) {
              jac_nn += penalty * trial_j_b;
            }
            elementJacobian_n_n_eb[i][j] += jac_nn * test_i_dS;

            // (1,0) cross: trial-fn variation of -a_n * grad N_j . n.
            double jac_nw = 0.0;
            for (int I = 0; I < nSpace; I++) {
              for (int ii = a_rowptr.data()[I]; ii < a_rowptr.data()[I + 1]; ii++) {
                const int J = a_colind.data()[ii];
                jac_nw -= a_n_b[ii] * u_grad_trial_trace_b[j * nSpace + J]
                                    * normal_b[I];
              }
            }
            elementJacobian_n_w_eb[i][j] += jac_nw * test_i_dS;
          }
        }
      } // kb

      // Scatter element-boundary contributions to global storage.
      for (int i = 0; i < nDOF_test_element; i++) {
        const int eN_i = eN * nDOF_test_element + i;
        const int gi   = u_l2g.data()[eN_i];
        globalResidual.data()[offset_n + stride_n * gi] += elementResidual_n_eb[i];
        for (int j = 0; j < nDOF_trial_element; j++) {
          const int ebN_i_j = ebN * 4 * nDOF_test_X_trial_element
                            + i * nDOF_trial_element + j;
          globalJacobian.data()[csrRowIndeces_n_n.data()[eN_i]
              + csrColumnOffsets_eb_n_n.data()[ebN_i_j]]
              += elementJacobian_n_n_eb[i][j];
          globalJacobian.data()[csrRowIndeces_n_w.data()[eN_i]
              + csrColumnOffsets_eb_n_w.data()[ebN_i_j]]
              += elementJacobian_n_w_eb[i][j];
        }
      }
    } // ebNE
  }

  // ============================================================================
  // invert(): m -> u inversion for the FCT pipeline.
  //
  // COMPONENT:
  //   0 -> wetting eq: NOT SUPPORTED in the (p_w, S_n) formulation.
  //        Reason: the wetting mass is
  //                  m_w = rho_w(p_w) * phi * theta_w(1 - u_n)
  //        which carries information about BOTH p_w and u_n. Inverting m_w to
  //        recover p_w alone is ill-posed (and degenerate when beta = 0, where
  //        m_w doesn't depend on p_w at all). The retention-curve legacy
  //        inverse was meaningful only for single-phase Richards and is now
  //        removed; calling invert(COMPONENT=0) throws.
  //
  //   1 -> non-wetting eq: m_n -> u_n = S_n via
  //          u_n = clamp(m_n / (phi*rho_n), 0, 1 - S_wr)
  //        Uses rho_n_phi_dof_member cached by calculateResidual_entropy_viscosity
  //        / calculateMassMatrix. Output written to 'u_dof_n' (if provided) or
  //        falls back to 'u_dof'.
  //
  // The Python Coefficients class rejects STABILIZATION_TYPE='Implicit_FCT'
  // up front, so a properly-configured run never reaches COMPONENT=0 here.
  // The throw is a load-bearing safety net for misconfigured callers.
  // ============================================================================
  void invert(arguments_dict &args)
  {
    xt::pyarray<int>    &a_rowptr             = args.array<int>("a_rowptr");
    xt::pyarray<int>    &a_colind             = args.array<int>("a_colind");
    double               rho                  = args.scalar<double>("rho");        // freshwater reference (fallback)
    double               beta                 = args.scalar<double>("beta");
    xt::pyarray<double> &gravity              = args.array<double>("gravity");
    xt::pyarray<double> &alpha                = args.array<double>("alpha");
    xt::pyarray<double> &n                    = args.array<double>("n");
    xt::pyarray<double> &thetaR               = args.array<double>("thetaR");
    xt::pyarray<double> &thetaSR              = args.array<double>("thetaSR");
    xt::pyarray<double> &KWs                  = args.array<double>("KWs");
    xt::pyarray<int>    &elementMaterialTypes = args.array<int>("elementMaterialTypes");
    xt::pyarray<int>    &freeDOFMaterialTypes = args.array<int>("freeDOFMaterialTypes");
    int                  numDOFs              = args.scalar<int>("numDOFs");
    xt::pyarray<double> &mIn  = args.array<double>("limited_solution");
    xt::pyarray<double> &pOut = args.array<double>("u_dof");
    int                  USE_NEWTON_INVERT = args.scalar<int>("USE_NEWTON_INVERT");
    const int            PSK_TYPE          = args.scalar<int>("PSK_TYPE");
    const int            COMPONENT         = args.scalar<int>("COMPONENT");

    if (COMPONENT == 1) {
      // -------- Component-1 inverse: m_n -> u_n = S_n. --------
      // m_n = phi * rho_n * u_n  =>  u_n = m_n / (phi * rho_n).
      // Clamp to feasible range S_n in [0, 1 - S_wr].
      const int numDOFs_u = static_cast<int>(pOut.size());
      const bool have_rho_n_phi = (rho_n_phi_dof_member.size() ==
                                   static_cast<std::size_t>(numDOFs_u));
      const double rho_n_arg = args.scalar<double>("rho_n");
      // S_wr is material-dependent; assume material 0 if no material map exists.
      const int    mat0    = elementMaterialTypes.data()[0];
      const double phi_mat = thetaR.data()[mat0] + thetaSR.data()[mat0];
      const double S_wr    = thetaR.data()[mat0] / phi_mat;
      const double S_n_max = 1.0 - S_wr;
      // Fallback divisor must equal phi * rho_n -- not just phi -- so that
      // m_n / (phi*rho_n) recovers S_n correctly when the cached projection
      // is unavailable.
      const double rho_n_phi_fallback = phi_mat * rho_n_arg;
      for (int i = 0; i < numDOFs_u; i++) {
        const double rho_n_phi_i = have_rho_n_phi ? rho_n_phi_dof_member[i]
                                                  : rho_n_phi_fallback;
        double S_n = mIn.data()[i] / rho_n_phi_i;
        if (S_n < 0.0)     S_n = 0.0;
        if (S_n > S_n_max) S_n = S_n_max;
        pOut.data()[i] = S_n;
      }
      return;
    }

    // -------- Component-0 path: not supported. --------
    // The wetting mass m_w(p_w, u_n) cannot be inverted for p_w alone. See
    // the docstring above. Silence unused-variable warnings before the throw.
    (void)a_rowptr; (void)a_colind; (void)beta; (void)gravity;
    (void)alpha; (void)n; (void)thetaR; (void)thetaSR; (void)KWs;
    (void)freeDOFMaterialTypes; (void)numDOFs; (void)mIn; (void)pOut;
    (void)USE_NEWTON_INVERT; (void)PSK_TYPE; (void)rho;
    throw std::runtime_error(
        "mphase_co2::invert: COMPONENT=0 is not supported in the (p_w, S_n) "
        "formulation. The wetting mass m_w = rho_w(p_w)*phi*theta_w(1-u_n) "
        "does not uniquely determine p_w. Use STABILIZATION_TYPE='Galerkin' "
        "or 'EntropyViscosity'; do not invoke FCT inversion on component 0.");
  }

  void calculateMassMatrix(arguments_dict &args)
  {
    //element
    double               dt                  = args.scalar<double>("dt");
    xt::pyarray<double> &mesh_trial_ref      = args.array<double>("mesh_trial_ref");
    xt::pyarray<double> &mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
    xt::pyarray<double> &mesh_dof            = args.array<double>("mesh_dof");
    xt::pyarray<double> &mesh_velocity_dof   = args.array<double>("mesh_velocity_dof");
    double               MOVING_DOMAIN       = args.scalar<double>("MOVING_DOMAIN");
    xt::pyarray<int>    &mesh_l2g            = args.array<int>("mesh_l2g");
    xt::pyarray<double> &dV_ref              = args.array<double>("dV_ref");
    xt::pyarray<double> &u_trial_ref         = args.array<double>("u_trial_ref");
    xt::pyarray<double> &u_grad_trial_ref    = args.array<double>("u_grad_trial_ref");
    xt::pyarray<double> &u_test_ref          = args.array<double>("u_test_ref");
    xt::pyarray<double> &u_grad_test_ref     = args.array<double>("u_grad_test_ref");
    //element boundary
    xt::pyarray<double> &mesh_trial_trace_ref      = args.array<double>("mesh_trial_trace_ref");
    xt::pyarray<double> &mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
    xt::pyarray<double> &dS_ref                    = args.array<double>("dS_ref");
    xt::pyarray<double> &u_trial_trace_ref         = args.array<double>("u_trial_trace_ref");
    xt::pyarray<double> &u_grad_trial_trace_ref    = args.array<double>("u_grad_trial_trace_ref");
    xt::pyarray<double> &u_test_trace_ref          = args.array<double>("u_test_trace_ref");
    xt::pyarray<double> &u_grad_test_trace_ref     = args.array<double>("u_grad_test_trace_ref");
    xt::pyarray<double> &normal_ref                = args.array<double>("normal_ref");
    xt::pyarray<double> &boundaryJac_ref           = args.array<double>("boundaryJac_ref");
    //physics
    int nElements_global = args.scalar<int>("nElements_global");
    //new
    xt::pyarray<double> &ebqe_penalty_ext     = args.array<double>("ebqe_penalty_ext");
    xt::pyarray<int>    &elementMaterialTypes = args.array<int>("elementMaterialTypes");
    xt::pyarray<int>    &isSeepageFace        = args.array<int>("isSeepageFace");
    xt::pyarray<int>    &a_rowptr             = args.array<int>("a_rowptr");
    xt::pyarray<int>    &a_colind             = args.array<int>("a_colind");
    double               rho                  = args.scalar<double>("rho");
    double               beta                 = args.scalar<double>("beta");

    xt::pyarray<double> &q_rho                = args.array<double>("q_rho");

    xt::pyarray<double> &gravity              = args.array<double>("gravity");
    xt::pyarray<double> &alpha                = args.array<double>("alpha");
    xt::pyarray<double> &n                    = args.array<double>("n");
    xt::pyarray<double> &thetaR               = args.array<double>("thetaR");
    xt::pyarray<double> &thetaSR              = args.array<double>("thetaSR");
    xt::pyarray<double> &KWs                  = args.array<double>("KWs");
    //end new
    double               useMetrics                                 = args.scalar<double>("useMetrics");
    double               alphaBDF                                   = args.scalar<double>("alphaBDF");
    int                  lag_shockCapturing                         = args.scalar<int>("lag_shockCapturing");
    double               shockCapturingDiffusion                    = args.scalar<double>("shockCapturingDiffusion");
    xt::pyarray<int>    &u_l2g                                      = args.array<int>("u_l2g");
    xt::pyarray<int>    &r_l2g                                      = args.array<int>("r_l2g");
    xt::pyarray<double> &elementDiameter                            = args.array<double>("elementDiameter");
    int                  degree_polynomial                          = args.scalar<int>("degree_polynomial");
    xt::pyarray<double> &u_dof                                      = args.array<double>("u_dof");
    // u_dof_n always present in argsDict (getJacobian sets it from
    // self.u[1].dof in mphase_co2.py).
    xt::pyarray<double> &u_dof_n                                    = args.array<double>("u_dof_n");
    xt::pyarray<double> &velocity                                   = args.array<double>("velocity");
    xt::pyarray<double> &q_m_betaBDF                                = args.array<double>("q_m_betaBDF");
    xt::pyarray<double> &cfl                                        = args.array<double>("cfl");
    xt::pyarray<double> &q_numDiff_u_last                           = args.array<double>("q_numDiff_u_last");
    xt::pyarray<int>    &csrRowIndeces_u_u                          = args.array<int>("csrRowIndeces_u_u");
    xt::pyarray<int>    &csrColumnOffsets_u_u                       = args.array<int>("csrColumnOffsets_u_u");
    xt::pyarray<int>    &csrRowIndeces_n_n                          = args.array<int>("csrRowIndeces_n_n");
    // (1,0) cross-block CSR maps for the gas-eq diffusion
    // against grad u_w. Currently allocated by the framework but unused by
    // the C++ assembly - 3c.2/3c.3 will write into them.
    xt::pyarray<int>    &csrRowIndeces_n_w                          = args.array<int>("csrRowIndeces_n_w");
    xt::pyarray<int>    &csrColumnOffsets_n_n                       = args.array<int>("csrColumnOffsets_n_n");
    xt::pyarray<int>    &csrColumnOffsets_n_w                       = args.array<int>("csrColumnOffsets_n_w");
    xt::pyarray<double> &globalJacobian                             = args.array<double>("globalJacobian");
    xt::pyarray<double> &delta_x_ij                                 = args.array<double>("delta_x_ij");
    int                  nExteriorElementBoundaries_global          = args.scalar<int>("nExteriorElementBoundaries_global");
    xt::pyarray<int>    &exteriorElementBoundariesArray             = args.array<int>("exteriorElementBoundariesArray");
    xt::pyarray<int>    &elementBoundaryElementsArray               = args.array<int>("elementBoundaryElementsArray");
    xt::pyarray<int>    &elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
    xt::pyarray<double> &ebqe_velocity_ext                          = args.array<double>("ebqe_velocity_ext");
    xt::pyarray<int>    &isDOFBoundary_u                            = args.array<int>("isDOFBoundary_u");
    xt::pyarray<double> &ebqe_bc_u_ext                              = args.array<double>("ebqe_bc_u_ext");
    // component-1 (S_n) boundary arrays.
    xt::pyarray<int>    &isDOFBoundary_n                            = args.array<int>("isDOFBoundary_n");
    xt::pyarray<double> &ebqe_bc_u_n_ext                            = args.array<double>("ebqe_bc_u_n_ext");
    xt::pyarray<int>    &isFluxBoundary_u                           = args.array<int>("isFluxBoundary_u");
    xt::pyarray<double> &ebqe_bc_flux_ext                           = args.array<double>("ebqe_bc_flux_ext");
    xt::pyarray<int>    &csrColumnOffsets_eb_u_u                    = args.array<int>("csrColumnOffsets_eb_u_u");
    int                  LUMPED_MASS_MATRIX                         = args.scalar<int>("LUMPED_MASS_MATRIX");
    // PSK closure selector for evaluateCoefficients (read from argsDict).
    PSK_TYPE_member = args.scalar<int>("PSK_TYPE");
    double Ct_sge = 4.0;
    //
    //loop over elements to compute volume integrals and load them into the element Jacobians and global Jacobian
    //
    for (int eN = 0; eN < nElements_global; eN++) {
      double elementJacobian_u_u[nDOF_test_element][nDOF_trial_element];
      for (int i = 0; i < nDOF_test_element; i++)
        for (int j = 0; j < nDOF_trial_element; j++) { elementJacobian_u_u[i][j] = 0.0; }
      for (int k = 0; k < nQuadraturePoints_element; k++) {
        int eN_k                  = eN * nQuadraturePoints_element + k, //index to a scalar at a quadrature point
          eN_k_nSpace             = eN_k * nSpace,
            eN_nDOF_trial_element = eN * nDOF_trial_element; //index to a vector at a quadrature point
        //declare local storage
        double u = 0.0, grad_u[nSpace], m = 0.0, dm = 0.0, f[nSpace], df[nSpace], a[nnz], da[nnz], as[nnz], m_t = 0.0, dm_t = 0.0, dpdeResidual_u_u[nDOF_trial_element], Lstar_u[nDOF_test_element], dsubgridError_u_u[nDOF_trial_element], tau = 0.0, tau0 = 0.0, tau1 = 0.0, jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace], u_grad_trial[nDOF_trial_element * nSpace], dV, u_test_dV[nDOF_test_element], u_grad_test_dV[nDOF_test_element * nSpace], x, y, z, xt, yt, zt,
          G[nSpace * nSpace], G_dd_G, tr_G;

        //get jacobian, etc for mapping reference element
        ck.calculateMapping_element(eN, k, mesh_dof.data(), mesh_l2g.data(), mesh_trial_ref.data(), mesh_grad_trial_ref.data(), jac, jacDet, jacInv, x, y, z);
        ck.calculateMappingVelocity_element(eN, k, mesh_velocity_dof.data(), mesh_l2g.data(), mesh_trial_ref.data(), xt, yt, zt);
        //get the physical integration weight
        dV = fabs(jacDet) * dV_ref.data()[k];
        ck.calculateG(jacInv, G, G_dd_G, tr_G);
        //get the trial function gradients
        ck.gradTrialFromRef(&u_grad_trial_ref.data()[k * nDOF_trial_element * nSpace], jacInv, u_grad_trial);
        //get the solution
        ck.valFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], &u_trial_ref.data()[k * nDOF_trial_element], u);
        //get the solution gradients
        ck.gradFromDOF(u_dof.data(), &u_l2g.data()[eN_nDOF_trial_element], u_grad_trial, grad_u);
        //precalculate test function products with integration weights
        for (int j = 0; j < nDOF_trial_element; j++) {
          u_test_dV[j] = u_test_ref.data()[k * nDOF_trial_element + j] * dV;
          for (int I = 0; I < nSpace; I++) {
            u_grad_test_dV[j * nSpace + I] = u_grad_trial[j * nSpace + I] * dV; //cek warning won't work for Petrov-Galerkin
          }
        }
        //
        //calculate pde coefficients and derivatives at quadrature points
        //
        double Kr, dKr, thetaW;
        //const double rho_local = q_rho.data()[eN_k];
        // NOTE: the cek hack at line ~3170 (`dm = 1.0`) and the dm_t=1.0
        // override at the Jacobian assembly below make the final mass matrix
        // output independent of the coefficients here (result is just
        // M = integral N_i N_j dV regardless). Coefficient call is kept for
        // moving-mesh / VMS consistency.
        double dm_du_n_mm = 0.0, dkr_du_n_mm = 0.0;
        double df_du_n_mm[nSpace], da_du_n_mm[nnz];
        for (int I = 0; I < nSpace; I++) df_du_n_mm[I] = 0.0;
        for (int ii = 0; ii < nnz; ii++) da_du_n_mm[ii] = 0.0;
        double u_n_qp = 0.0;
        ck.valFromDOF(u_dof_n.data(),
                      &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_n_qp);
        evaluateCoefficients_from_Se(a_rowptr.data(), a_colind.data(), rho, q_rho.data()[eN_k], beta, gravity.data(),
                                     alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]],
                                     thetaR.data()[elementMaterialTypes.data()[eN]], thetaSR.data()[elementMaterialTypes.data()[eN]],
                                     &KWs.data()[elementMaterialTypes.data()[eN] * nnz], u, u_n_qp,
                                     m, dm, dm_du_n_mm, f, df, df_du_n_mm, a, da, da_du_n_mm,
                                     as, Kr, dKr, dkr_du_n_mm, thetaW);
        //
        //moving mesh
        //
        double mesh_velocity[3];
        mesh_velocity[0] = xt;
        mesh_velocity[1] = yt;
        mesh_velocity[2] = zt;
        for (int I = 0; I < nSpace; I++) {
          f[I] -= MOVING_DOMAIN * m * mesh_velocity[I];
          df[I] -= MOVING_DOMAIN * dm * mesh_velocity[I];
        }
        //
        //calculate time derivatives
        //
        //cek hack
        dm = 1.0;
        ck.bdf(alphaBDF,
               q_m_betaBDF.data()[eN_k], //since m_t isn't used, we don't have to correct mass
               m, dm, m_t, dm_t);
        //
        //calculate subgrid error contribution to the Jacobian (strong residual, adjoint, jacobian of strong residual)
        //
        //calculate the adjoint times the test functions
        for (int i = 0; i < nDOF_test_element; i++) {
          int i_nSpace = i * nSpace;
          Lstar_u[i]   = ck.Advection_adjoint(df, &u_grad_test_dV[i_nSpace]);
        }
        //calculate the Jacobian of strong residual
        for (int j = 0; j < nDOF_trial_element; j++) {
          int j_nSpace        = j * nSpace;
          dpdeResidual_u_u[j] = ck.MassJacobian_strong(dm_t, u_trial_ref.data()[k * nDOF_trial_element + j]) + ck.AdvectionJacobian_strong(df, &u_grad_trial[j_nSpace]);
        }
        //tau and tau*Res
        calculateSubgridError_tau(elementDiameter.data()[eN], dm_t, df, cfl.data()[eN_k], tau0);

        calculateSubgridError_tau(Ct_sge, G, dm_t, df, tau1, cfl.data()[eN_k]);
        tau = useMetrics * tau1 + (1.0 - useMetrics) * tau0;

        for (int j = 0; j < nDOF_trial_element; j++) dsubgridError_u_u[j] = -tau * dpdeResidual_u_u[j];
        for (int i = 0; i < nDOF_test_element; i++) {
          for (int j = 0; j < nDOF_trial_element; j++) {
            if (LUMPED_MASS_MATRIX == 1) {
              if (i == j) elementJacobian_u_u[i][j] += u_test_dV[i];
            } else {
              int j_nSpace = j * nSpace;
              int i_nSpace = i * nSpace;
              dm_t = 1.0; //we are solving for continuum density explicitly
              elementJacobian_u_u[i][j] += ck.MassJacobian_weak(dm_t, u_trial_ref.data()[k * nDOF_trial_element + j], u_test_dV[i]);
            }
          } //j
        } //i
      } //k
      //
      //load into element Jacobian into global Jacobian
      //
      for (int i = 0; i < nDOF_test_element; i++) {
        int eN_i = eN * nDOF_test_element + i;
        int I    = u_l2g.data()[eN_i];
        for (int j = 0; j < nDOF_trial_element; j++) {
          int eN_i_j = eN_i * nDOF_trial_element + j;
          int J      = u_l2g.data()[eN * nDOF_trial_element + j];
          //globalJacobian.data()[csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_u_u.data()[eN_i_j]] += elementJacobian_u_u[i][j];
          delta_x_ij.data()[3 * (csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_u_u.data()[eN_i_j]) + 0] = mesh_dof.data()[I * 3 + 0] - mesh_dof.data()[J * 3 + 0];
          delta_x_ij.data()[3 * (csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_u_u.data()[eN_i_j]) + 1] = mesh_dof.data()[I * 3 + 1] - mesh_dof.data()[J * 3 + 1];
          delta_x_ij.data()[3 * (csrRowIndeces_u_u.data()[eN_i] + csrColumnOffsets_u_u.data()[eN_i_j]) + 2] = mesh_dof.data()[I * 3 + 2] - mesh_dof.data()[J * 3 + 2];
        } //j
      } //i
    } //elements
    // (1,1) block: unit consistent mass / dt. Used by the time integrator.
    // Also refresh the cached nodal phi*rho_n so invert(COMPONENT=1) sees the
    // correct projection if mass-matrix assembly runs more recently than the
    // residual.
    const int numDOFs_u_mm = static_cast<int>(u_dof.size());
    std::vector<double> rho_n_phi_mm(numDOFs_u_mm, 0.0);
    std::vector<double> ML_n_mm(numDOFs_u_mm, 0.0);
    for (int eN = 0; eN < nElements_global; eN++) {
      double elementJacobian_n_n[nDOF_test_element][nDOF_trial_element];
      for (int i = 0; i < nDOF_test_element; i++)
        for (int j = 0; j < nDOF_trial_element; j++) { elementJacobian_n_n[i][j] = 0.0; }
      const int    mat_eN_mm = elementMaterialTypes.data()[eN];
      const double phi_eN_mm = thetaR.data()[mat_eN_mm] + thetaSR.data()[mat_eN_mm];
      const double rho_n_arg = args.scalar<double>("rho_n");
      for (int k = 0; k < nQuadraturePoints_element; k++) {
        const int eN_nDOF_trial_element = eN * nDOF_trial_element;
        double jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace], x, y, z;
        ck.calculateMapping_element(eN, k, mesh_dof.data(), mesh_l2g.data(),
                                    mesh_trial_ref.data(), mesh_grad_trial_ref.data(),
                                    jac, jacDet, jacInv, x, y, z);
        const double dV = fabs(jacDet) * dV_ref.data()[k];
        for (int i = 0; i < nDOF_test_element; i++) {
          const double test_i = u_test_ref.data()[k * nDOF_test_element + i];
          const int    gi     = u_l2g.data()[eN * nDOF_test_element + i];
          rho_n_phi_mm[gi] += phi_eN_mm * rho_n_arg * test_i * dV;
          ML_n_mm[gi]      += test_i * dV;
          for (int j = 0; j < nDOF_trial_element; j++) {
            const double trial_j = u_trial_ref.data()[k * nDOF_trial_element + j];
            elementJacobian_n_n[i][j] += (test_i * trial_j * dV) / dt;
          }
        }
      }
      for (int i = 0; i < nDOF_test_element; i++) {
        const int eN_i = eN * nDOF_test_element + i;
        for (int j = 0; j < nDOF_trial_element; j++) {
          const int eN_i_j = eN_i * nDOF_trial_element + j;
          const int full_offset = csrRowIndeces_n_n.data()[eN_i] + csrColumnOffsets_n_n.data()[eN_i_j];
          if (std::fabs(globalJacobian.data()[full_offset]) < 1.0e-14)
            globalJacobian.data()[full_offset] += elementJacobian_n_n[i][j];
        }
      }
    }
    for (int i = 0; i < numDOFs_u_mm; ++i) {
      if (ML_n_mm[i] > 0.0) rho_n_phi_mm[i] /= ML_n_mm[i];
      else rho_n_phi_mm[i] = thetaR.data()[0] + thetaSR.data()[0];
      rho_n_phi_mm[i] = std::max(rho_n_phi_mm[i], 1.0e-16);
    }
    rho_n_phi_dof_member = rho_n_phi_mm;
  } //computeMassMatrix
}; //Mphase_co2

inline Mphase_co2_base *newmphase_co2(int nSpaceIn, int nQuadraturePoints_elementIn, int nDOF_mesh_trial_elementIn, int nDOF_trial_elementIn, int nDOF_test_elementIn, int nQuadraturePoints_elementBoundaryIn, int CompKernelFlag)
{
  if (nSpaceIn == 1)
    return proteus::chooseAndAllocateDiscretization1D<Mphase_co2_base, Mphase_co2, CompKernel>(nSpaceIn, nQuadraturePoints_elementIn, nDOF_mesh_trial_elementIn, nDOF_trial_elementIn, nDOF_test_elementIn, nQuadraturePoints_elementBoundaryIn, CompKernelFlag);
  else if (nSpaceIn == 2)
    return proteus::chooseAndAllocateDiscretization2D<Mphase_co2_base, Mphase_co2, CompKernel>(nSpaceIn, nQuadraturePoints_elementIn, nDOF_mesh_trial_elementIn, nDOF_trial_elementIn, nDOF_test_elementIn, nQuadraturePoints_elementBoundaryIn, CompKernelFlag);
  else {
    assert(nSpaceIn == 3);
    return proteus::chooseAndAllocateDiscretization<Mphase_co2_base, Mphase_co2, CompKernel>(nSpaceIn, nQuadraturePoints_elementIn, nDOF_mesh_trial_elementIn, nDOF_trial_elementIn, nDOF_test_elementIn, nQuadraturePoints_elementBoundaryIn, CompKernelFlag);
  }
}
} // namespace mphase_co2
} // namespace proteus
#endif
