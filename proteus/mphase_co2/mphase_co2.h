#ifndef MPHASE_CO2_H
#define MPHASE_CO2_H
#include <cmath>
#include <iostream>
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
  Mphase_co2() : nDOF_test_X_trial_element(nDOF_test_element * nDOF_trial_element), ck() { }
  inline void evaluateCoefficients(const int rowptr[nSpace], const int colind[nnz], const double rho0, const double rho_transport, const double beta, const double gravity[nSpace], const double alpha, const double n_vg, const double thetaR, const double thetaSR, const double KWs[nnz], const double &u, double &m, double &dm, double f[nSpace], double df[nSpace], double a[nnz], double da[nnz], double as[nnz], double &kr, double &dkr, double &thetaW_out)
  {
    double thetaW, DthetaW_DpsiC, KWr, DKWr_DpsiC;
    proteus::mphase_co2::psk::vgm_wetting(
        -u, alpha, n_vg, thetaR, thetaSR,
        thetaW, DthetaW_DpsiC, KWr, DKWr_DpsiC);

    thetaW_out = thetaW;
    // Density uses transported salinity scaled by the compressibility factor.
    const double rhom  = rho_transport * exp(beta * u);
    const double drhom = beta * rhom;
    m     = rhom * thetaW;
    dm    = -rhom * DthetaW_DpsiC + drhom * thetaW;
    const double rho_ratio = rhom / rho0;
    for (int I = 0; I < nSpace; I++) {
      f[I]  = 0.0;
      df[I] = 0.0;
      for (int ii = rowptr[I]; ii < rowptr[I + 1]; ii++) {
        a[ii]  = rhom * KWr * KWs[ii];
        da[ii] = (drhom * KWr - rhom * DKWr_DpsiC) * KWs[ii];
        f[I] += rhom * rho_ratio * KWr * KWs[ii] * gravity[colind[ii]];
        df[I] += (drhom * rho_ratio * KWr +
                  rhom * (drhom / rho0) * KWr -
                  rhom * rho_ratio * DKWr_DpsiC) * KWs[ii] * gravity[colind[ii]];
        as[ii] = rhom * KWs[ii];
        kr     = KWr;
        dkr    = -DKWr_DpsiC;
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
    // Phase B Step 1: component-1 (S_w) trivial-mass equation args.
    // Used in the dedicated component-1 element loop appended at the end
    // of this function. Not consumed by the existing component-0 logic.
    const double         dt                                         = args.scalar<double>("dt");
    xt::pyarray<double> &u_dof_v                                    = args.array<double>("u_dof_v");
    xt::pyarray<double> &u_dof_v_old                                = args.array<double>("u_dof_v_old");
    const int            offset_v                                   = args.scalar<int>("offset_v");
    const int            stride_v                                   = args.scalar<int>("stride_v");
    xt::pyarray<double> &globalResidual                             = args.array<double>("globalResidual");
    int                  nExteriorElementBoundaries_global          = args.scalar<int>("nExteriorElementBoundaries_global");
    xt::pyarray<int>    &exteriorElementBoundariesArray             = args.array<int>("exteriorElementBoundariesArray");
    xt::pyarray<int>    &elementBoundaryElementsArray               = args.array<int>("elementBoundaryElementsArray");
    xt::pyarray<int>    &elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
    xt::pyarray<double> &ebqe_velocity_ext                          = args.array<double>("ebqe_velocity_ext");
    xt::pyarray<int>    &isDOFBoundary_u                            = args.array<int>("isDOFBoundary_u");
    xt::pyarray<double> &ebqe_bc_u_ext                              = args.array<double>("ebqe_bc_u_ext");
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
        evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_local, beta, gravity.data(), alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]], thetaR.data()[elementMaterialTypes.data()[eN]],
                             thetaSR.data()[elementMaterialTypes.data()[eN]], &KWs.data()[elementMaterialTypes.data()[eN] * nnz], u, m, dm, f, df, a, da, as, Kr, dKr, thetaW);
        q_theta.data()[eN_k] = thetaW;
        

        for (int I = 0; I < nSpace; ++I) {
          q_velocity.data()[eN_k_nSpace + I] = grad_u[I];
        }
        // Darcy Velocity
        double pressure_gradient[nSpace];
        const double rho_ratio = rho_velocity / rho;
        for (int J=0; J<nSpace; ++J)
          pressure_gradient[J] = grad_u[J] - rho_ratio * gravity.data()[J];
        // for each row I, acc = sum_j (a_{Ij}/rho) * gp[j]
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
        evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_ext, beta, gravity.data(), alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]], thetaR.data()[elementMaterialTypes.data()[eN]],
                             thetaSR.data()[elementMaterialTypes.data()[eN]], &KWs.data()[elementMaterialTypes.data()[eN] * nnz], u_ext, m_ext, dm_ext, f_ext, df_ext, a_ext, da_ext, as_ext, Kr, dKr, thetaW_ext);
        evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_ext, beta, gravity.data(), alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]], thetaR.data()[elementMaterialTypes.data()[eN]],
                             thetaSR.data()[elementMaterialTypes.data()[eN]], &KWs.data()[elementMaterialTypes.data()[eN] * nnz], bc_u_ext, bc_m_ext, bc_dm_ext, bc_f_ext, bc_df_ext, bc_a_ext, bc_da_ext, bc_as_ext, Kr, dKr, thetaW_bc);
        ebqe_theta.data()[ebNE_kb] = thetaW_ext;
        
        //
        //Calculate Darcy velocity on exterior face : v_ext = -(a_ext/rho) * (grad_u_ext + gravity) ---
        //
        double ext_pressure_gradient[nSpace];
        const double rho_ratio_ext = rho_velocity_ext / rho;
        for (int J=0; J<nSpace; ++J)
          ext_pressure_gradient[J] = grad_u_ext[J] - rho_ratio_ext * gravity.data()[J];

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
    // Phase B Step 1: component-1 (S_w) trivial mass equation
    //
    //   d(S_w)/dt = 0   ==>   R_v[i] = sum_eN  int_eN  (u_v - u_v_old)/dt * N_i dV
    //
    // No flux, no source, no coupling. The residual is purely the consistent
    // mass-matrix action on (u_v - u_v_old), so when Newton converges the
    // gas-component DOFs equal their previous-step values - i.e. S_w stays
    // pinned at its initial condition. This block is fully decoupled from the
    // component-0 (psi_w) loop above; both share the same FE space (u_l2g).
    // ============================================================================
    for (int eN = 0; eN < nElements_global; eN++) {
      double elementResidual_v[nDOF_test_element];
      for (int i = 0; i < nDOF_test_element; i++) elementResidual_v[i] = 0.0;
      for (int k = 0; k < nQuadraturePoints_element; k++) {
        const int eN_k = eN * nQuadraturePoints_element + k;
        const int eN_nDOF_trial_element = eN * nDOF_trial_element;
        double jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace], x_q, y_q, z_q;
        ck.calculateMapping_element(eN, k, mesh_dof.data(), mesh_l2g.data(),
                                    mesh_trial_ref.data(), mesh_grad_trial_ref.data(),
                                    jac, jacDet, jacInv, x_q, y_q, z_q);
        const double dV = std::fabs(jacDet) * dV_ref.data()[k];
        double u_v = 0.0, u_v_old = 0.0;
        ck.valFromDOF(u_dof_v.data(),
                      &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_v);
        ck.valFromDOF(u_dof_v_old.data(),
                      &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_v_old);
        const double m_v_t = (u_v - u_v_old) / dt;
        for (int i = 0; i < nDOF_test_element; i++) {
          const double test_i = u_test_ref.data()[k * nDOF_test_element + i];
          elementResidual_v[i] += m_v_t * test_i * dV;
        }
      }
      for (int i = 0; i < nDOF_test_element; i++) {
        const int eN_i = eN * nDOF_test_element + i;
        globalResidual.data()[offset_v + stride_v * u_l2g.data()[eN_i]] += elementResidual_v[i];
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
    xt::pyarray<double> &velocity                                   = args.array<double>("velocity");
    xt::pyarray<double> &q_m_betaBDF                                = args.array<double>("q_m_betaBDF");
    xt::pyarray<double> &cfl                                        = args.array<double>("cfl");
    xt::pyarray<double> &q_numDiff_u                                = args.array<double>("q_numDiff_u");
    xt::pyarray<double> &q_numDiff_u_last                           = args.array<double>("q_numDiff_u_last");
    xt::pyarray<int>    &csrRowIndeces_u_u                          = args.array<int>("csrRowIndeces_u_u");
    xt::pyarray<int>    &csrColumnOffsets_u_u                       = args.array<int>("csrColumnOffsets_u_u");
    xt::pyarray<double> &globalJacobian                             = args.array<double>("globalJacobian");
    // Phase B Step 1: component-1 (S_w) trivial-mass Jacobian args. Used by
    // the dedicated component-1 element loop appended at the end of this
    // function. (1,1) block is the consistent mass matrix / dt; (0,1) and
    // (1,0) cross-blocks are zero in Step 1.
    const double         dt_v                                       = args.scalar<double>("dt");
    xt::pyarray<int>    &csrRowIndeces_v_v                          = args.array<int>("csrRowIndeces_v_v");
    xt::pyarray<int>    &csrColumnOffsets_v_v                       = args.array<int>("csrColumnOffsets_v_v");
    int                  nExteriorElementBoundaries_global          = args.scalar<int>("nExteriorElementBoundaries_global");
    xt::pyarray<int>    &exteriorElementBoundariesArray             = args.array<int>("exteriorElementBoundariesArray");
    xt::pyarray<int>    &elementBoundaryElementsArray               = args.array<int>("elementBoundaryElementsArray");
    xt::pyarray<int>    &elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
    xt::pyarray<double> &ebqe_velocity_ext                          = args.array<double>("ebqe_velocity_ext");
    xt::pyarray<int>    &isDOFBoundary_u                            = args.array<int>("isDOFBoundary_u");
    xt::pyarray<double> &ebqe_bc_u_ext                              = args.array<double>("ebqe_bc_u_ext");
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
      for (int i = 0; i < nDOF_test_element; i++) {
        for (int j = 0; j < nDOF_trial_element; j++) { elementJacobian_u_u[i][j] = 0.0; }
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

        evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, q_rho.data()[eN_k], beta, gravity.data(), alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]], thetaR.data()[elementMaterialTypes.data()[eN]],
                             thetaSR.data()[elementMaterialTypes.data()[eN]], &KWs.data()[elementMaterialTypes.data()[eN] * nnz], u, m, dm, f, df, a, da, as, Kr, dKr, thetaW);
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
      } //k
      //
      //load into element Jacobian into global Jacobian
      //
      for (int i = 0; i < nDOF_test_element; i++) {
        int eN_i = eN * nDOF_test_element + i;
        for (int j = 0; j < nDOF_trial_element; j++) {
          int eN_i_j = eN_i * nDOF_trial_element + j;
          globalJacobian.data()[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_u_u[eN_i_j]] += elementJacobian_u_u[i][j];
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

        evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_ext, beta, gravity.data(), alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]], thetaR.data()[elementMaterialTypes.data()[eN]],
                             thetaSR.data()[elementMaterialTypes.data()[eN]], &KWs.data()[elementMaterialTypes.data()[eN] * nnz], u_ext, m_ext, dm_ext, f_ext, df_ext, a_ext, da_ext, as_ext, Kr, dKr, thetaW);
        evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_ext, beta, gravity.data(), alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]], thetaR.data()[elementMaterialTypes.data()[eN]],
                             thetaSR.data()[elementMaterialTypes.data()[eN]], &KWs.data()[elementMaterialTypes.data()[eN] * nnz], bc_u_ext, bc_m_ext, bc_dm_ext, bc_f_ext, bc_df_ext, bc_a_ext, bc_da_ext, bc_as_ext, Kr, dKr, thetaW_bc);
        //
        //calculate the flux jacobian
        //
        for (int j = 0; j < nDOF_trial_element; j++) {
          exteriorNumericalFluxJacobian(a_rowptr.data(), a_colind.data(), isDOFBoundary_u.data()[ebNE_kb], normal, a_ext, da_ext, grad_u_ext, &u_grad_trial_trace[j * nSpace], df_ext, u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element + j],
                                        ebqe_penalty_ext.data()[ebNE_kb], //penalty,
                                        fluxJacobian_u_u[j]);
        } //j
        //
        //update the global Jacobian from the flux Jacobian
        //
        for (int i = 0; i < nDOF_test_element; i++) {
          int eN_i = eN * nDOF_test_element + i;
          for (int j = 0; j < nDOF_trial_element; j++) {
            int ebN_i_j = ebN * 4 * nDOF_test_X_trial_element + i * nDOF_trial_element + j;
            globalJacobian.data()[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_eb_u_u[ebN_i_j]] += fluxJacobian_u_u[j] * u_test_dS[i];
          } //j
        } //i
      } //kb
    } //ebNE

    // ============================================================================
    // Phase B Step 1: component-1 (S_w) trivial Jacobian
    //   J_(1,1)[i,j] = (1/dt) * sum_eN int_eN N_i N_j dV   (consistent mass / dt)
    // No off-diagonal blocks contributed in Step 1.
    // ============================================================================
    for (int eN = 0; eN < nElements_global; eN++) {
      double elementJacobian_v_v[nDOF_test_element][nDOF_trial_element];
      for (int i = 0; i < nDOF_test_element; i++)
        for (int j = 0; j < nDOF_trial_element; j++) elementJacobian_v_v[i][j] = 0.0;
      for (int k = 0; k < nQuadraturePoints_element; k++) {
        double jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace], x_q, y_q, z_q;
        ck.calculateMapping_element(eN, k, mesh_dof.data(), mesh_l2g.data(),
                                    mesh_trial_ref.data(), mesh_grad_trial_ref.data(),
                                    jac, jacDet, jacInv, x_q, y_q, z_q);
        const double dV = std::fabs(jacDet) * dV_ref.data()[k];
        for (int i = 0; i < nDOF_test_element; i++) {
          const double test_i = u_test_ref.data()[k * nDOF_test_element + i];
          for (int j = 0; j < nDOF_trial_element; j++) {
            const double trial_j = u_trial_ref.data()[k * nDOF_trial_element + j];
            elementJacobian_v_v[i][j] += (test_i * trial_j * dV) / dt_v;
          }
        }
      }
      for (int i = 0; i < nDOF_test_element; i++) {
        const int eN_i = eN * nDOF_test_element + i;
        for (int j = 0; j < nDOF_trial_element; j++) {
          const int eN_i_j = eN_i * nDOF_trial_element + j;
          globalJacobian.data()[csrRowIndeces_v_v.data()[eN_i] + csrColumnOffsets_v_v.data()[eN_i_j]]
              += elementJacobian_v_v[i][j];
        }
      }
    }
  } //computeJacobian

  // void FCTStep(arguments_dict &args)
  // {
  //   xt::pyarray<double> &bc_mask                   = args.array<double>("bc_mask");
  //   int                  NNZ                       = args.scalar<int>("NNZ");     //number on non-zero entries on sparsity pattern
  //   int                  numDOFs                   = args.scalar<int>("numDOFs"); //number of DOFs
  //   double               dt                        = args.scalar<double>("dt");
  //   xt::pyarray<double> &ML                        = args.array<double>("ML"); //lumped mass matrix (as vector)
  //   xt::pyarray<double> &mn                        = args.array<double>("mn");               //DOFs of solution at time tn
  //   xt::pyarray<double> &mHigh                     = args.array<double>("mHigh");               //DOFs of high order solution at tnp1
  //   xt::pyarray<double> &mLow                      = args.array<double>("mLow");
  //   xt::pyarray<double> &mDotHigh                     = args.array<double>("mDotHigh");               //DOFs of high order solution at tnp1
  //   xt::pyarray<double> &mDotLow                      = args.array<double>("mDotLow");
  //   xt::pyarray<double> &limited_solution          = args.array<double>("limited_solution");
  //   xt::pyarray<int>    &csrRowIndeces_DofLoops    = args.array<int>("csrRowIndeces_DofLoops");    //csr row indeces
  //   xt::pyarray<int>    &csrColumnOffsets_DofLoops = args.array<int>("csrColumnOffsets_DofLoops"); //csr column offsets
  //   xt::pyarray<double> &MC                        = args.array<double>("MC");             //mass matrix
  //   xt::pyarray<double> &dt_times_fH_minus_fL      = args.array<double>("dt_times_fH_minus_fL");   //low minus high order dissipative matrices
  //   xt::pyarray<double> &min_m_bc                  = args.array<double>("min_m_bc");               //min/max value at BCs. If DOF is not at boundary then min=1E10, max=-1E10
  //   xt::pyarray<double> &max_m_bc                  = args.array<double>("max_m_bc");
  //   xt::pyarray<double> &fluxCorrection                  = args.array<double>("fluxCorrection");
  //   //flags
  //   int                  LUMPED_MASS_MATRIX        = args.scalar<int>("LUMPED_MASS_MATRIX");
  //   int                  MONOLITHIC                = args.scalar<int>("MONOLITHIC");
  //   double               Rpos[numDOFs], Rneg[numDOFs];
  //   double               FluxCorrectionMatrix[NNZ];
  //   double               mDot[numDOFs];

  //   //////////////////
  //   // LOOP in DOFs //
  //   //////////////////
  //   int ij = 0;
  //   for (int i = 0; i < numDOFs; i++) {
  //     mDot[i] = (mLow.data()[i] - mn.data()[i])/dt;
  //     //cek todo: add boundary data--these are just initialized
  //     //will need to pass p_bc at DOF and calc M
  //     double mini=min_m_bc.data()[i], maxi=max_m_bc.data()[i];
  //     //we're doing local FCT
  //     //if (GLOBAL_FCT == 1) {
  //     //  mini = 0.;
  //     //  maxi = 1.;
  //     //}

  //     double Pposi = 0, Pnegi = 0;
  //     // LOOP OVER THE SPARSITY PATTERN (j-LOOP)//
  //     for (int offset = csrRowIndeces_DofLoops.data()[i]; offset < csrRowIndeces_DofLoops.data()[i + 1]; offset++) {
  //       int j = csrColumnOffsets_DofLoops.data()[offset];
  //       ////////////////////////
  //       // COMPUTE THE BOUNDS //
  //       ////////////////////////
  //       if (GLOBAL_FCT == 0) {
  //         if (MONOLITHIC == 0) {
  //           mini = fmin(mini, mLow[j]);
  //           maxi = fmax(maxi, mLow[j]);
  //         } else {
  //           mini = fmin(mini, mn.data()[j]);
  //           maxi = fmax(maxi, mn.data()[j]);
  //         }
  //       }
  //       // i-th row of flux correction matrix
  //       //double I_plus_ML_minus_MC = (i == j ? 1. : 0.) * (1. + ML.data()[i]) - MC.data()[ij];
  //       //mDot[i] += I_plus_ML_minus_MC * (mHigh.data()[j] - mn.data()[j]) / ML.data()[i];
  //       mDot[j] = (mLow.data()[j] - mn.data()[j])/dt;
  //       if (MONOLITHIC == 0) {
  //         FluxCorrectionMatrix[ij] = (LUMPED_MASS_MATRIX == 1 ? 0. : 1.) * dt * MC.data()[ij] * (mDotLow.data()[i] - mDotLow.data()[j]) + dt_times_fH_minus_fL.data()[ij];
  //       } else {
  //         FluxCorrectionMatrix[ij] = dt_times_fH_minus_fL.data()[ij];
  //       }
  //       ///////////////////////
  //       // COMPUTE P VECTORS //
  //       ///////////////////////
  //       Pposi += FluxCorrectionMatrix[ij] * ((FluxCorrectionMatrix[ij] > 0) ? 1. : 0.);
  //       Pnegi += FluxCorrectionMatrix[ij] * ((FluxCorrectionMatrix[ij] < 0) ? 1. : 0.);

  //       //update ij
  //       ij += 1;
  //     }
  //     ///////////////////////
  //     // COMPUTE Q VECTORS //
  //     ///////////////////////
  //     double gamma;
  //     double Qposi;
  //     double Qnegi;
  //     if (MONOLITHIC == 0) {
  //       Qposi = ML.data()[i] * (maxi - mLow[i]);
  //       Qnegi = ML.data()[i] * (mini - mLow[i]);
  //     } else {
  //       //cek todo: don't think this is right for Richards
  //       gamma = 10.0 * ML.data()[i];
  //       Qposi = fmin(0.5 * ML.data()[i] * (1.0 - mn.data()[i]), gamma * (maxi - mn[i]));
  //       Qnegi = fmax(0.5 * ML.data()[i] * (0.0 - mn.data()[i]), gamma * (mini - mn[i]));
  //     }
  //     ///////////////////////
  //     // COMPUTE R VECTORS //
  //     ///////////////////////
  //     Rpos[i] = ((Pposi == 0) ? 1. : fmin(1.0, Qposi / Pposi));
  //     Rneg[i] = ((Pnegi == 0) ? 1. : fmin(1.0, Qnegi / Pnegi));
  //   } // i DOFs

  //   //////////////////////
  //   // COMPUTE LIMITERS //
  //   //////////////////////
  //   ij = 0;
  //   for (int i = 0; i < numDOFs; i++) {
  //     double ith_Limiter_times_FluxCorrectionMatrix = 0.;
  //     double alpha_fA, alpha_dot, beta_ij = 1.0;
  //     // LOOP OVER THE SPARSITY PATTERN (j-LOOP)//
  //     for (int offset = csrRowIndeces_DofLoops.data()[i]; offset < csrRowIndeces_DofLoops.data()[i + 1]; offset++) {
  //       int j = csrColumnOffsets_DofLoops.data()[offset];
  //       alpha_fA     = ((FluxCorrectionMatrix[ij] > 0) ? fmin(Rpos[i], Rneg[j]) : fmin(Rneg[i], Rpos[j])) * FluxCorrectionMatrix[ij];
  //       alpha_dot    = fmin(1.0, beta_ij * fabs(alpha_fA) / MC.data()[ij] / fmax(1.0e-8, fabs(mDot[i] - mDot[j])));
  //       if (MONOLITHIC == 0) {
  //         ith_Limiter_times_FluxCorrectionMatrix += alpha_fA;
  //       } else {
  //         ith_Limiter_times_FluxCorrectionMatrix += alpha_fA + (LUMPED_MASS_MATRIX == 1 ? 0. : 1.) * dt * alpha_dot * MC.data()[ij] * (mDot[i] - mDot[j]);
  //       }
  //       ij += 1;
  //     }

  //     fluxCorrection.data()[i] = -ith_Limiter_times_FluxCorrectionMatrix*bc_mask[i]/dt;
  //     limited_solution.data()[i] = mLow[i] + 1. / ML.data()[i] * ith_Limiter_times_FluxCorrectionMatrix * bc_mask[i];

  //     //cek todo: double check that the below is not necesary. The limted_solution should already be within the bounds
  //     //Calculate the min and max mass bounds
  //     //double mMin = rho * thetaR.data()[elementMaterialTypes.data()[0]];
  //     //double mMax = rho * (thetaR.data()[elementMaterialTypes.data()[0]] + thetaSR.data()[elementMaterialTypes.data()[0]]);

  //     // Check if the limited mass is within bounds
  //     //if (limited_mass < mMin || limited_mass > mMax) {
  //     //  limited_solution.data()[i] = solL[i]; // Fallback to lower-order solution
  //     //} else {
  //     //  limited_solution.data()[i] = limited_mass; // Assign the limited mass
  //     //}
  //   }
  // }

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
    // Phase B Step 1: component-1 (S_w) trivial-mass equation args.
    // (dt is already declared at the top of this function for the EV path.)
    xt::pyarray<double> &u_dof_v                                    = args.array<double>("u_dof_v");
    xt::pyarray<double> &u_dof_v_old                                = args.array<double>("u_dof_v_old");
    const int            offset_v                                   = args.scalar<int>("offset_v");
    const int            stride_v                                   = args.scalar<int>("stride_v");
    xt::pyarray<double> &globalResidual                             = args.array<double>("globalResidual");
    int                  nExteriorElementBoundaries_global          = args.scalar<int>("nExteriorElementBoundaries_global");
    xt::pyarray<int>    &exteriorElementBoundariesArray             = args.array<int>("exteriorElementBoundariesArray");
    xt::pyarray<int>    &elementBoundaryElementsArray               = args.array<int>("elementBoundaryElementsArray");
    xt::pyarray<int>    &elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
    xt::pyarray<double> &ebqe_velocity_ext                          = args.array<double>("ebqe_velocity_ext");
    xt::pyarray<int>    &isDOFBoundary_u                            = args.array<int>("isDOFBoundary_u");
    xt::pyarray<double> &ebqe_bc_u_ext                              = args.array<double>("ebqe_bc_u_ext");
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
    // Phase B Step 1: CSR maps for the (1,1) Jacobian block. Used by the
    // dedicated component-1 element loop appended at the end to assemble
    // the gas-side mass-matrix Jacobian / dt.
    xt::pyarray<int>    &csrRowIndeces_v_v                          = args.array<int>("csrRowIndeces_v_v");
    xt::pyarray<int>    &csrColumnOffsets_v_v                       = args.array<int>("csrColumnOffsets_v_v");
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
          // Match the main variable-density operator: grad(u) - (rho/rho0) g.
          Phi[j]   -= (rho_node_j / rho) * mesh_dof.data()[x_gj * 3 + I] * gravity[I];
          Phi_n[j] -= (rho_node_j / rho) * mesh_dof.data()[x_gj * 3 + I] * gravity[I];
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

        evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_local, beta, gravity.data(), alpha.data()[elementMaterialTypes[eN]], n.data()[elementMaterialTypes[eN]], thetaR.data()[elementMaterialTypes[eN]], thetaSR.data()[elementMaterialTypes[eN]],
                             &KWs.data()[elementMaterialTypes[eN] * nnz], un, mn, dmn, fn, dfn, an, dan, asn, Krn, dKrn, thetaWn);
        evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_local, beta, gravity.data(), alpha.data()[elementMaterialTypes[eN]], n.data()[elementMaterialTypes[eN]], thetaR.data()[elementMaterialTypes[eN]], thetaSR.data()[elementMaterialTypes[eN]],
                             &KWs.data()[elementMaterialTypes[eN] * nnz], u, m, dm, f, df, a, da, as, Kr, dKr, thetaW);
        q_theta.data()[eN_k] = thetaW;

        // Darcy velocity for coupling should use the direct FE gradient of the
        // pressure head. The Phi-based gradients are only for stabilization.
        for (int I = 0; I < nSpace; ++I) {
          q_velocity.data()[eN_k_nSpace + I] = grad_u_velocity[I];
        }

        double pressure_gradient[nSpace];
        const double rho_ratio = rho_velocity / rho;
        for (int J = 0; J < nSpace; ++J)
          pressure_gradient[J] = grad_u_velocity[J] - rho_ratio * gravity.data()[J];

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

        evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_ext, beta, gravity.data(), alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]], thetaR.data()[elementMaterialTypes.data()[eN]],
                             thetaSR.data()[elementMaterialTypes.data()[eN]], &KWs.data()[elementMaterialTypes.data()[eN] * nnz], u_ext, m_ext, dm_ext, f_ext, df_ext, a_ext, da_ext, as_ext, bc_Kr, bc_dKr, thetaW_ext);
        evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_ext, beta, gravity.data(), alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]], thetaR.data()[elementMaterialTypes.data()[eN]],
                             thetaSR.data()[elementMaterialTypes.data()[eN]], &KWs.data()[elementMaterialTypes.data()[eN] * nnz], un_ext, mn_ext, dmn_ext, fn_ext, dfn_ext, an_ext, dan_ext, asn_ext, bc_Krn, bc_dKrn, thetaWn_ext);
        evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_ext, beta, gravity.data(), alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]], thetaR.data()[elementMaterialTypes.data()[eN]],
                             thetaSR.data()[elementMaterialTypes.data()[eN]], &KWs.data()[elementMaterialTypes.data()[eN] * nnz], bc_u_ext, bc_m_ext, bc_dm_ext, bc_f_ext, bc_df_ext, bc_a_ext, bc_da_ext, bc_as_ext, bc_Kr_ext,bc_dKr_ext, thetaW_bc_ext);
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
        const double rho_ratio_ext = rho_velocity_ext / rho;
        for (int J = 0; J < nSpace; ++J)
          ext_pressure_gradient[J] = grad_u_ext[J] - rho_ratio_ext * gravity.data()[J];

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
          //probably need isDOFBoundary_un here
          //exteriorNumericalFluxJacobian(a_rowptr.data(), a_colind.data(), isDOFBoundary_u.data()[ebNE_kb], normal, asn_ext, dan_ext, grad_u_ext, &u_grad_trial_trace[j * nSpace], dfn_ext, u_trial_trace_ref.data()[ebN_local_kb * nDOF_test_element + j],
          //                              ebqe_penalty_ext.data()[ebNE_kb], //penalty,
          //                              fluxJacobian_un_un[j]);
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
    // Phase B Step 1: bound to numDOFs_u (component 0 only). For i >= numDOFs_u
    // mesh_dof.data()[i*3+I] reads past the end of mesh_dof (sized N*3) and
    // produces NaN that propagates through psi[i] / quantDOFs[i] etc. into
    // the main DOF loop's residual.
    for (int i = 0; i < numDOFs_u; i++) {
      double gi[nSpace], Cij[nSpace], xi[nSpace], etaMaxi, etaMini;
      const int node_i = freeDOFToNode_u.data()[i];
      double solni = u_free_dof_old[i];
      for (int I = 0; I < nSpace; I++) {
        solni -= (rho_dof[i] / rho) * gravity.data()[I] * mesh_dof.data()[node_i * 3 + I];
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
          solnj -= (rho_dof[j] / rho) * gravity.data()[I] * mesh_dof.data()[node_j * 3 + I];
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
    // Phase B Step 1: only iterate component-0 DOFs here. Component 1
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
        // Match evaluateCoefficients(): use an edge-based hydrostatic jump
        // with local density scaling rho/rho0, rather than grad(rho g·x),
        // which would introduce a spurious (g·x) grad(rho) term.
        for (int I = 0; I < nSpace; I++) {
          const double delta_x = mesh_dof.data()[node_j * 3 + I] - mesh_dof.data()[node_i * 3 + I];
          const double hydrostatic_jump = (rho_edge / rho) * gravity.data()[I] * delta_x;
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
          evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_i, beta, gravity.data(),
                               alpha.data()[elementMaterialTypes.data()[0]], //cek hack, only for 1 material
                               n.data()[elementMaterialTypes.data()[0]], thetaR.data()[elementMaterialTypes.data()[0]], thetaSR.data()[elementMaterialTypes.data()[0]], &KWs.data()[elementMaterialTypes.data()[0] * nnz], u_free_dof[i], m, dm, f, df, a, da, as, Kr, dKr, thetaW_tmp);
          fL = Theta * Kr * fmax(0.0, -TransportMatrix[full_offset]) * delta_phi;
          fL_CN = Theta_h * Kr * fmax(0.0, -TransportMatrix[full_offset]) * delta_phi;

          if (i != j) {
            globalJacobian.data()[full_offset] -= Theta * Kr * fmax(0.0, -TransportMatrix[full_offset]);
            J_ii -= -Theta * Kr * fmax(0.0, -TransportMatrix[full_offset]) + Theta * dKr * fmax(0.0, -TransportMatrix[full_offset]) * delta_phi;
          }
          ith_flux_term += fL;
          fA -= fL;
        } else {
          evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_j, beta, gravity.data(),
                               alpha.data()[elementMaterialTypes.data()[0]], //cek hack, only for 1 material
                               n.data()[elementMaterialTypes.data()[0]], thetaR.data()[elementMaterialTypes.data()[0]], thetaSR.data()[elementMaterialTypes.data()[0]], &KWs.data()[elementMaterialTypes.data()[0] * nnz], u_free_dof[j], m, dm, f, df, a, da, as, Kr, dKr, thetaW_tmp);
          fL = Theta * Kr * fmax(0.0, -TransportMatrix[full_offset]) * delta_phi;
          fL_CN = Theta_h * Kr * fmax(0.0, -TransportMatrix[full_offset]) * delta_phi;

          if (i != j) {
            globalJacobian.data()[full_offset] -= Theta * Kr * fmax(0.0, -TransportMatrix[full_offset]) + Theta * dKr * fmax(0.0, -TransportMatrix[full_offset]) * delta_phi;
            J_ii -= -Theta * Kr * fmax(0.0, -TransportMatrix[full_offset]);
          }
          ith_flux_term += fL;
          fA -= fL;
        }
        if (-TransportMatrixn[full_offset] * delta_phin <= 0.0) {
          evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_i, beta, gravity.data(),
                               alpha.data()[elementMaterialTypes.data()[0]], //cek hack, only for 1 material
                               n.data()[elementMaterialTypes.data()[0]], thetaR.data()[elementMaterialTypes.data()[0]], thetaSR.data()[elementMaterialTypes.data()[0]], &KWs.data()[elementMaterialTypes.data()[0] * nnz], u_free_dof_old[i], m, dm, f, df, a, da, as, Kr, dKr, thetaW_tmp);
          fL = (1 - Theta) * Kr * fmax(0.0, -TransportMatrixn[full_offset]) * delta_phin;
          fL_CN += (1 - Theta_h) * Kr * fmax(0.0, -TransportMatrixn[full_offset]) * delta_phin;
          ith_flux_term += fL;
          fA -= fL;
          fA_CN -= fL_CN;
        } else {
          evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_j, beta, gravity.data(),
                               alpha.data()[elementMaterialTypes.data()[0]], //cek hack, only for 1 material
                               n.data()[elementMaterialTypes.data()[0]], thetaR.data()[elementMaterialTypes.data()[0]], thetaSR.data()[elementMaterialTypes.data()[0]], &KWs.data()[elementMaterialTypes.data()[0] * nnz], u_free_dof_old[j], m, dm, f, df, a, da, as, Kr, dKr, thetaW_tmp);
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
      evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_i, beta, gravity.data(),
                           alpha.data()[elementMaterialTypes.data()[0]], //cek hack, only for 1 material
                           n.data()[elementMaterialTypes.data()[0]], thetaR.data()[elementMaterialTypes.data()[0]], thetaSR.data()[elementMaterialTypes.data()[0]], &KWs.data()[elementMaterialTypes.data()[0] * nnz], u_free_dof[i], m, dm, f, df, a, da, as, Kr, dKr, thetaW_tmp);
      evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, rho_i, beta, gravity.data(),
                           alpha.data()[elementMaterialTypes.data()[0]], //cek hack, only for 1 material
                           n.data()[elementMaterialTypes.data()[0]], thetaR.data()[elementMaterialTypes.data()[0]], thetaSR.data()[elementMaterialTypes.data()[0]], &KWs.data()[elementMaterialTypes.data()[0] * nnz], u_free_dof_old[i], mn.data()[i], dmn, fn, dfn, an, dan, asn, Krn, dKrn, thetaW_tmp);
      mLow.data()[i] = m;
      globalResidual.data()[offset_u + stride_u * i] += bc_mask.data()[i] * (MLi * (m - mn.data()[i]) / dt - ith_flux_term);
      globalJacobian.data()[ii] += bc_mask.data()[i] * (MLi * dm / dt + J_ii) + (1.0 - bc_mask.data()[i]);
    }
    if (STABILIZATION_TYPE == STABILIZATION::Implicit_FCT) {
      FCTStep(args);
      for (int i = 0; i < numDOFs; i++) {
        globalResidual.data()[offset_u + stride_u * i] += fluxCorrection.data()[i];
      }
    }

    // ============================================================================
    // Phase B Step 1: component-1 (S_w) trivial mass equation
    //   d(S_w)/dt = 0  ->  R_v[i] = sum_eN int_eN (u_v - u_v_old)/dt * N_i dV
    //   J_(1,1)[i,j]  = (1/dt) * sum_eN int_eN N_i N_j dV   (consistent mass / dt)
    // Same body as the equivalent block in calculateResidual; mirrored here so
    // STABILIZATION_TYPE != Galerkin runs (entropy-viscosity / FCT) also assemble
    // the component-1 residual AND the (1,1) Jacobian block. (1,0) and (0,1)
    // cross-blocks remain zero in Step 1 by construction.
    // ============================================================================
    for (int eN = 0; eN < nElements_global; eN++) {
      double elementResidual_v[nDOF_test_element];
      double elementJacobian_v_v[nDOF_test_element][nDOF_trial_element];
      for (int i = 0; i < nDOF_test_element; i++) {
        elementResidual_v[i] = 0.0;
        for (int j = 0; j < nDOF_trial_element; j++) elementJacobian_v_v[i][j] = 0.0;
      }
      for (int k = 0; k < nQuadraturePoints_element; k++) {
        const int eN_nDOF_trial_element = eN * nDOF_trial_element;
        double jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace], x_q, y_q, z_q;
        ck.calculateMapping_element(eN, k, mesh_dof.data(), mesh_l2g.data(),
                                    mesh_trial_ref.data(), mesh_grad_trial_ref.data(),
                                    jac, jacDet, jacInv, x_q, y_q, z_q);
        const double dV = std::fabs(jacDet) * dV_ref.data()[k];
        double u_v = 0.0, u_v_old = 0.0;
        ck.valFromDOF(u_dof_v.data(),
                      &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_v);
        ck.valFromDOF(u_dof_v_old.data(),
                      &u_l2g.data()[eN_nDOF_trial_element],
                      &u_trial_ref.data()[k * nDOF_trial_element], u_v_old);
        const double m_v_t = (u_v - u_v_old) / dt;
        for (int i = 0; i < nDOF_test_element; i++) {
          const double test_i = u_test_ref.data()[k * nDOF_test_element + i];
          elementResidual_v[i] += m_v_t * test_i * dV;
          for (int j = 0; j < nDOF_trial_element; j++) {
            const double trial_j = u_trial_ref.data()[k * nDOF_trial_element + j];
            elementJacobian_v_v[i][j] += (test_i * trial_j * dV) / dt;
          }
        }
      }
      for (int i = 0; i < nDOF_test_element; i++) {
        const int eN_i = eN * nDOF_test_element + i;
        globalResidual.data()[offset_v + stride_v * u_l2g.data()[eN_i]] += elementResidual_v[i];
        for (int j = 0; j < nDOF_trial_element; j++) {
          const int eN_i_j = eN_i * nDOF_trial_element + j;
          globalJacobian.data()[csrRowIndeces_v_v.data()[eN_i] + csrColumnOffsets_v_v.data()[eN_i_j]]
              += elementJacobian_v_v[i][j];
        }
      }
    }
  }

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
    const bool have_rho_dof = (rho_dof_member.size() == static_cast<std::size_t>(numDOFs));
    int USE_NEWTON_INVERT = args.scalar<int>("USE_NEWTON_INVERT");

    for (int i = 0; i < numDOFs; i++) {
      const int material_i = freeDOFMaterialTypes.data()[i];
      const double rho_i = have_rho_dof ? rho_dof_member[i] : rho;
      if (USE_NEWTON_INVERT){
        proteus::mphase_co2::psk::vgm_invert_newton(
            mIn.data()[i], rho_i, beta,
            alpha.data()[material_i], n.data()[material_i],
            thetaR.data()[material_i], thetaSR.data()[material_i],
            pOut.data()[i]);
      }
      else{
        const int mat0 = elementMaterialTypes.data()[0];
        proteus::mphase_co2::psk::vgm_invert_analytic(
            mIn.data()[i], rho_i,
            alpha.data()[mat0], n.data()[mat0],
            thetaR.data()[mat0], thetaSR.data()[mat0],
            pOut.data()[i]);
      }
      }
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
    xt::pyarray<double> &velocity                                   = args.array<double>("velocity");
    xt::pyarray<double> &q_m_betaBDF                                = args.array<double>("q_m_betaBDF");
    xt::pyarray<double> &cfl                                        = args.array<double>("cfl");
    xt::pyarray<double> &q_numDiff_u_last                           = args.array<double>("q_numDiff_u_last");
    xt::pyarray<int>    &csrRowIndeces_u_u                          = args.array<int>("csrRowIndeces_u_u");
    xt::pyarray<int>    &csrColumnOffsets_u_u                       = args.array<int>("csrColumnOffsets_u_u");
    xt::pyarray<int>    &csrRowIndeces_v_v                          = args.array<int>("csrRowIndeces_v_v");
    xt::pyarray<int>    &csrColumnOffsets_v_v                       = args.array<int>("csrColumnOffsets_v_v");
    xt::pyarray<double> &globalJacobian                             = args.array<double>("globalJacobian");
    xt::pyarray<double> &delta_x_ij                                 = args.array<double>("delta_x_ij");
    int                  nExteriorElementBoundaries_global          = args.scalar<int>("nExteriorElementBoundaries_global");
    xt::pyarray<int>    &exteriorElementBoundariesArray             = args.array<int>("exteriorElementBoundariesArray");
    xt::pyarray<int>    &elementBoundaryElementsArray               = args.array<int>("elementBoundaryElementsArray");
    xt::pyarray<int>    &elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
    xt::pyarray<double> &ebqe_velocity_ext                          = args.array<double>("ebqe_velocity_ext");
    xt::pyarray<int>    &isDOFBoundary_u                            = args.array<int>("isDOFBoundary_u");
    xt::pyarray<double> &ebqe_bc_u_ext                              = args.array<double>("ebqe_bc_u_ext");
    xt::pyarray<int>    &isFluxBoundary_u                           = args.array<int>("isFluxBoundary_u");
    xt::pyarray<double> &ebqe_bc_flux_ext                           = args.array<double>("ebqe_bc_flux_ext");
    xt::pyarray<int>    &csrColumnOffsets_eb_u_u                    = args.array<int>("csrColumnOffsets_eb_u_u");
    int                  LUMPED_MASS_MATRIX                         = args.scalar<int>("LUMPED_MASS_MATRIX");
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
        evaluateCoefficients(a_rowptr.data(), a_colind.data(), rho, q_rho.data()[eN_k], beta, gravity.data(), alpha.data()[elementMaterialTypes.data()[eN]], n.data()[elementMaterialTypes.data()[eN]], thetaR.data()[elementMaterialTypes.data()[eN]],
                             thetaSR.data()[elementMaterialTypes.data()[eN]], &KWs.data()[elementMaterialTypes.data()[eN] * nnz], u, m, dm, f, df, a, da, as, Kr, dKr, thetaW);
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
    for (int eN = 0; eN < nElements_global; eN++) {
      double elementJacobian_v_v[nDOF_test_element][nDOF_trial_element];
      for (int i = 0; i < nDOF_test_element; i++)
        for (int j = 0; j < nDOF_trial_element; j++) { elementJacobian_v_v[i][j] = 0.0; }
      for (int k = 0; k < nQuadraturePoints_element; k++) {
        const int eN_nDOF_trial_element = eN * nDOF_trial_element;
        double jac[nSpace * nSpace], jacDet, jacInv[nSpace * nSpace], x, y, z;
        ck.calculateMapping_element(eN, k, mesh_dof.data(), mesh_l2g.data(),
                                    mesh_trial_ref.data(), mesh_grad_trial_ref.data(),
                                    jac, jacDet, jacInv, x, y, z);
        const double dV = fabs(jacDet) * dV_ref.data()[k];
        for (int i = 0; i < nDOF_test_element; i++) {
          const double test_i = u_test_ref.data()[k * nDOF_test_element + i];
          for (int j = 0; j < nDOF_trial_element; j++) {
            const double trial_j = u_trial_ref.data()[k * nDOF_trial_element + j];
            elementJacobian_v_v[i][j] += (test_i * trial_j * dV) / dt;
          }
        }
      }
      for (int i = 0; i < nDOF_test_element; i++) {
        const int eN_i = eN * nDOF_test_element + i;
        for (int j = 0; j < nDOF_trial_element; j++) {
          const int eN_i_j = eN_i * nDOF_trial_element + j;
          const int full_offset = csrRowIndeces_v_v.data()[eN_i] + csrColumnOffsets_v_v.data()[eN_i_j];
          if (std::fabs(globalJacobian.data()[full_offset]) < 1.0e-14)
            globalJacobian.data()[full_offset] += elementJacobian_v_v[i][j];
        }
      }
    }
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
