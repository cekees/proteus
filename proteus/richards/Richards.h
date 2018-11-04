#ifndef Richards_H
#define Richards_H
#include <cmath>
#include <iostream>
#include <valarray>
#include "CompKernel.h"
#include "ModelFactory.h"
#define nnz nSpace
#define POWER_SMOOTHNESS_INDICATOR 2
#define IS_BETAij_ONE 0
#define GLOBAL_FCT 0
namespace proteus
{
  // Power entropy //
  inline double ENTROPY(const double& phi, const double& phiL, const double& phiR){
    return 1./2.*std::pow(fabs(phi),2.);
  }
  inline double DENTROPY(const double& phi, const double& phiL, const double& phiR){
    return fabs(phi)*(phi>=0 ? 1 : -1);
  }
  // Log entropy // for level set from 0 to 1
  inline double ENTROPY_LOG(const double& phi, const double& phiL, const double& phiR){
    return std::log(fabs((phi-phiL)*(phiR-phi))+1E-14);
  }
  inline double DENTROPY_LOG(const double& phi, const double& phiL, const double& phiR){
    return (phiL+phiR-2*phi)*((phi-phiL)*(phiR-phi)>=0 ? 1 : -1)/(fabs((phi-phiL)*(phiR-phi))+1E-14);
  }
}

namespace proteus
{
  class Richards_base
  {
    //The base class defining the interface
  public:
    virtual ~Richards_base(){}
    virtual void calculateResidual(//element
                                   double dt,
				   double* mesh_trial_ref,
				   double* mesh_grad_trial_ref,
				   double* mesh_dof,
				   double* mesh_velocity_dof,
				   double MOVING_DOMAIN,
				   int* mesh_l2g,
				   double* dV_ref,
				   double* u_trial_ref,
				   double* u_grad_trial_ref,
				   double* u_test_ref,
				   double* u_grad_test_ref,
				   //element boundary
				   double* mesh_trial_trace_ref,
				   double* mesh_grad_trial_trace_ref,
				   double* dS_ref,
				   double* u_trial_trace_ref,
				   double* u_grad_trial_trace_ref,
				   double* u_test_trace_ref,
				   double* u_grad_test_trace_ref,
				   double* normal_ref,
				   double* boundaryJac_ref,
				   //physics
				   int nElements_global,
				   double* ebqe_penalty_ext,
				   int* elementMaterialTypes,	
				   int* isSeepageFace,
				   int* a_rowptr,
				   int* a_colind,
				   double rho,
				   double beta,
				   double* gravity,
				   double* alpha,
				   double* n,
				   double* thetaR,
				   double* thetaSR,
				   double* KWs,
                                   double useMetrics,
                                   double alphaBDF,
                                   int lag_shockCapturing,
                                   double shockCapturingDiffusion,
                                   double sc_uref,
                                   double sc_alpha,
				   int* u_l2g,
                                   int* r_l2g,
				   double* elementDiameter,
                                   int degree_polynomial,
				   double* u_dof,
                                   double* u_dof_old,	
				   double* velocity,
				   double* q_m,
				   double* q_u,
				   double* q_m_betaBDF,
				   double* cfl,
                                   double* edge_based_cfl,
				   double* q_numDiff_u, 
				   double* q_numDiff_u_last, 
				   int offset_u, int stride_u, 
				   double* globalResidual,
				   int nExteriorElementBoundaries_global,
				   int* exteriorElementBoundariesArray,
				   int* elementBoundaryElementsArray,
				   int* elementBoundaryLocalElementBoundariesArray,
				   double* ebqe_velocity_ext,
				   int* isDOFBoundary_u,
				   double* ebqe_bc_u_ext,
				   int* isFluxBoundary_u,
				   double* ebqe_bc_flux_u_ext,
				   double* ebqe_phi,double epsFact,
				   double* ebqe_u,
				   double* ebqe_flux,
                                   // PARAMETERS FOR EDGE BASED STABILIZATION
                                   double cE,
                                   double cK,
                                   // PARAMETERS FOR LOG BASED ENTROPY FUNCTION
                                   double uL,
                                   double uR,
                                   // PARAMETERS FOR EDGE VISCOSITY
                                   int numDOFs,
                                   int NNZ,
                                   int* csrRowIndeces_DofLoops,
                                   int* csrColumnOffsets_DofLoops,
                                   int* csrRowIndeces_CellLoops,
                                   int* csrColumnOffsets_CellLoops,
                                   int* csrColumnOffsets_eb_CellLoops,
                                   // C matrices
                                   double* Cx,
                                   double* Cy,
                                   double* Cz,
                                   double* CTx,
                                   double* CTy,
                                   double* CTz,
                                   double* ML,
                                   double* delta_x_ij,
                                   // PARAMETERS FOR 1st or 2nd ORDER MPP METHOD
                                   int LUMPED_MASS_MATRIX,
                                   int STABILIZATION_TYPE,
                                   int ENTROPY_TYPE,
                                   // FOR FCT
                                   double* dLow,
                                   double* fluxMatrix,
                                   double* uDotLow,
                                   double* uLow,
                                   double* dt_times_dH_minus_dL,
                                   double* min_u_bc,
                                   double* max_u_bc,
                                   // AUX QUANTITIES OF INTEREST
                                   double* quantDOFs)=0;
    virtual void calculateJacobian(//element
                                   double dt,
				   double* mesh_trial_ref,
				   double* mesh_grad_trial_ref,
				   double* mesh_dof,
				   double* mesh_velocity_dof,
				   double MOVING_DOMAIN,
				   int* mesh_l2g,
				   double* dV_ref,
				   double* u_trial_ref,
				   double* u_grad_trial_ref,
				   double* u_test_ref,
				   double* u_grad_test_ref,
				   //element boundary
				   double* mesh_trial_trace_ref,
				   double* mesh_grad_trial_trace_ref,
				   double* dS_ref,
				   double* u_trial_trace_ref,
				   double* u_grad_trial_trace_ref,
				   double* u_test_trace_ref,
				   double* u_grad_test_trace_ref,
				   double* normal_ref,
				   double* boundaryJac_ref,
				   //physics
				   int nElements_global,
				   //new
				   double* ebqe_penalty_ext,
				   int* elementMaterialTypes,	
				   int* isSeepageFace,
				   int* a_rowptr,
				   int* a_colind,
				   double rho,
				   double beta,
				   double* gravity,
				   double* alpha,
				   double* n,
				   double* thetaR,
				   double* thetaSR,
				   double* KWs,
				   //end new
			           double useMetrics, 
				   double alphaBDF,
				   int lag_shockCapturing,
				   double shockCapturingDiffusion,
				   int* u_l2g,
                                   int* r_l2g,
				   double* elementDiameter,
                                   int degree_polynomial,
                                   double* u_dof, 
				   double* velocity,
				   double* q_m_betaBDF, 
				   double* cfl,
				   double* q_numDiff_u_last, 
				   int* csrRowIndeces_u_u,int* csrColumnOffsets_u_u,
				   double* globalJacobian,
                                   double* delta_x_ij,
				   int nExteriorElementBoundaries_global,
				   int* exteriorElementBoundariesArray,
				   int* elementBoundaryElementsArray,
				   int* elementBoundaryLocalElementBoundariesArray,
				   double* ebqe_velocity_ext,
				   int* isDOFBoundary_u,
				   double* ebqe_bc_u_ext,
				   int* isFluxBoundary_u,
				   double* ebqe_bc_flux_u_ext,
				   int* csrColumnOffsets_eb_u_u,
                                   int LUMPED_MASS_MATRIX)=0;
    virtual void FCTStep(int NNZ, //number on non-zero entries on sparsity pattern
                         int numDOFs, //number of DOFs
                         double* lumped_mass_matrix, //lumped mass matrix (as vector)
                         double* soln, //DOFs of solution at time tn
                         double* solH, //DOFs of high order solution at tnp1
                         double* uLow,
                         double* limited_solution,
                         int* csrRowIndeces_DofLoops, //csr row indeces
                         int* csrColumnOffsets_DofLoops, //csr column offsets
                         double* MassMatrix, //mass matrix
                         double* dt_times_dH_minus_dL, //low minus high order dissipative matrices
                         double* min_u_bc, //min/max value at BCs. If DOF is not at boundary then min=1E10, max=-1E10
                         double* max_u_bc,
                         int LUMPED_MASS_MATRIX
                         )=0;
    virtual void kth_FCT_step(double dt,
			      int num_fct_iter,
			      int NNZ, //number on non-zero entries on sparsity pattern
			      int numDOFs, //number of DOFs
			      double* MC,
			      double* ML,
			      double* soln,
			      double* solLim, //DOFs of high order solution at tnp1
			      double* uDotLow,
			      double* uLow,
			      double* dLow,
			      double* FluxMatrix,
			      double* limitedFlux,
			      int* csrRowIndeces_DofLoops, 
			      int* csrColumnOffsets_DofLoops)=0;
    virtual void calculateResidual_entropy_viscosity(//element
                                                     double dt,
                                                     double* mesh_trial_ref,
                                                     double* mesh_grad_trial_ref,
                                                     double* mesh_dof,
                                                     double* mesh_velocity_dof,
                                                     double MOVING_DOMAIN,
                                                     int* mesh_l2g,
                                                     double* dV_ref,
                                                     double* u_trial_ref,
                                                     double* u_grad_trial_ref,
                                                     double* u_test_ref,
                                                     double* u_grad_test_ref,
                                                     //element boundary
                                                     double* mesh_trial_trace_ref,
                                                     double* mesh_grad_trial_trace_ref,
                                                     double* dS_ref,
                                                     double* u_trial_trace_ref,
                                                     double* u_grad_trial_trace_ref,
                                                     double* u_test_trace_ref,
                                                     double* u_grad_test_trace_ref,
                                                     double* normal_ref,
                                                     double* boundaryJac_ref,
                                                     //physics
                                                     int nElements_global,
                                                     double* ebqe_penalty_ext,
                                                     int* elementMaterialTypes,	
                                                     int* isSeepageFace,
                                                     int* a_rowptr,
                                                     int* a_colind,
                                                     double rho,
                                                     double beta,
                                                     double* gravity,
                                                     double* alpha,
                                                     double* n,
                                                     double* thetaR,
                                                     double* thetaSR,
                                                     double* KWs,
                                                     double useMetrics,
                                                     double alphaBDF,
                                                     int lag_shockCapturing,
                                                     double shockCapturingDiffusion,
                                                     double sc_uref,
                                                     double sc_alpha,
                                                     int* u_l2g,
                                                     int* r_l2g,
                                                     double* elementDiameter,
                                                     int degree_polynomial,
                                                     double* u_dof,
                                                     double* u_dof_old,
                                                     double* velocity,
                                                     double* q_m,
                                                     double* q_u,
                                                     double* q_m_betaBDF,
                                                     double* cfl,
                                                     double* edge_based_cfl,
                                                     double* q_numDiff_u,
                                                     double* q_numDiff_u_last,
                                                     int offset_u, int stride_u,
                                                     double* globalResidual,
                                                     int nExteriorElementBoundaries_global,
                                                     int* exteriorElementBoundariesArray,
                                                     int* elementBoundaryElementsArray,
                                                     int* elementBoundaryLocalElementBoundariesArray,
                                                     double* ebqe_velocity_ext,
                                                     int* isDOFBoundary_u,
                                                     double* ebqe_bc_u_ext,
                                                     int* isFluxBoundary_u,
                                                     double* ebqe_bc_flux_u_ext,
                                                     double* ebqe_phi,double epsFact,
                                                     double* ebqe_u,
                                                     double* ebqe_flux,
                                                     // PARAMETERS FOR EDGE BASED STABILIZATION
                                                     double cE,
                                                     double cK,
                                                     // PARAMETERS FOR LOG BASED ENTROPY FUNCTION
                                                     double uL,
                                                     double uR,
                                                     // PARAMETERS FOR EDGE VISCOSITY
                                                     int numDOFs,
                                                     int NNZ,
                                                     int* csrRowIndeces_DofLoops,
                                                     int* csrColumnOffsets_DofLoops,
                                                     int* csrRowIndeces_CellLoops,
                                                     int* csrColumnOffsets_CellLoops,
                                                     int* csrColumnOffsets_eb_CellLoops,
                                                     // C matrices
                                                     double* Cx,
                                                     double* Cy,
                                                     double* Cz,
                                                     double* CTx,
                                                     double* CTy,
                                                     double* CTz,
                                                     double* ML,
                                                     double* delta_x_ij,
                                                     // PARAMETERS FOR 1st or 2nd ORDER MPP METHOD
                                                     int LUMPED_MASS_MATRIX,
                                                     int STABILIZATION_TYPE,
                                                     int ENTROPY_TYPE,
                                                     // FOR FCT
						     double* dLow,
						     double* fluxMatrix,
                                                     double* uDotLow,
						     double* uLow,
                                                     double* dt_times_dH_minus_dL,
                                                     double* min_u_bc,
                                                     double* max_u_bc,
                                                     // AUX QUANTITIES OF INTEREST
                                                     double* quantDOFs)=0;
    virtual void calculateMassMatrix(//element
                                     double dt,
                                     double* mesh_trial_ref,
                                     double* mesh_grad_trial_ref,
                                     double* mesh_dof,
                                     double* mesh_velocity_dof,
                                     double MOVING_DOMAIN,
                                     int* mesh_l2g,
                                     double* dV_ref,
                                     double* u_trial_ref,
                                     double* u_grad_trial_ref,
                                     double* u_test_ref,
                                     double* u_grad_test_ref,
                                     //element boundary
                                     double* mesh_trial_trace_ref,
                                     double* mesh_grad_trial_trace_ref,
                                     double* dS_ref,
                                     double* u_trial_trace_ref,
                                     double* u_grad_trial_trace_ref,
                                     double* u_test_trace_ref,
                                     double* u_grad_test_trace_ref,
                                     double* normal_ref,
                                     double* boundaryJac_ref,
                                     //physics
                                     int nElements_global,
                                     //new
                                     double* ebqe_penalty_ext,
                                     int* elementMaterialTypes,	
                                     int* isSeepageFace,
                                     int* a_rowptr,
                                     int* a_colind,
                                     double rho,
                                     double beta,
                                     double* gravity,
                                     double* alpha,
                                     double* n,
                                     double* thetaR,
                                     double* thetaSR,
                                     double* KWs,
                                     //end new
                                     double useMetrics, 
                                     double alphaBDF,
                                     int lag_shockCapturing,
                                     double shockCapturingDiffusion,
                                     int* u_l2g,
                                     int* r_l2g,
                                     double* elementDiameter,
                                     int degree_polynomial,
                                     double* u_dof, 
                                     double* velocity,
                                     double* q_m_betaBDF, 
                                     double* cfl,
                                     double* q_numDiff_u_last, 
                                     int* csrRowIndeces_u_u,int* csrColumnOffsets_u_u,
                                     double* globalJacobian,
                                     double* delta_x_ij,
                                     int nExteriorElementBoundaries_global,
                                     int* exteriorElementBoundariesArray,
                                     int* elementBoundaryElementsArray,
                                     int* elementBoundaryLocalElementBoundariesArray,
                                     double* ebqe_velocity_ext,
                                     int* isDOFBoundary_u,
                                     double* ebqe_bc_u_ext,
                                     int* isFluxBoundary_u,
                                     double* ebqe_bc_flux_u_ext,
                                     int* csrColumnOffsets_eb_u_u,
                                     int LUMPED_MASS_MATRIX)=0;
  };

  template<class CompKernelType,
    int nSpace,
    int nQuadraturePoints_element,
    int nDOF_mesh_trial_element,
    int nDOF_trial_element,
    int nDOF_test_element,
    int nQuadraturePoints_elementBoundary>
    class Richards : public Richards_base
    {
    public:
      const int nDOF_test_X_trial_element;
      CompKernelType ck;
    Richards():
      nDOF_test_X_trial_element(nDOF_test_element*nDOF_trial_element),
        ck()
          {}
      inline
        void evaluateCoefficients(const int rowptr[nSpace],
                                  const int colind[nnz],
                                  const double rho,
                                  const double beta,
                                  const double gravity[nSpace],
                                  const double alpha,
                                  const double n_vg,
                                  const double thetaR,
                                  const double thetaSR,
                                  const double KWs[nnz],			      
                                  const double& u,
                                  double& m,
                                  double& dm,
                                  double f[nSpace],
                                  double df[nSpace],
                                  double a[nnz],
                                  double da[nnz],
                                  double& kr,
                                  double& dkr)
      {
        const int nSpace2=nSpace*nSpace;
        register double psiC,
          pcBar,pcBar_n,pcBar_nM1,pcBar_nM2,
          onePlus_pcBar_n,
          sBar,sqrt_sBar,DsBar_DpsiC,
          thetaW,DthetaW_DpsiC,
          vBar,vBar2,DvBar_DpsiC,
          KWr,DKWr_DpsiC,
          rho2=rho*rho,
          thetaS,
          rhom,drhom,m_vg,pcBarStar,sqrt_sBarStar;
        psiC = -u;
        m_vg = 1.0 - 1.0/n_vg;
        thetaS = thetaR + thetaSR;
        if (psiC > 0.0)
          {
            pcBar = alpha*psiC;
            pcBarStar = pcBar;
            if (pcBar < 1.0e-8)
              pcBarStar=1.0e-8;
            pcBar_nM2 = pow(pcBarStar,n_vg-2);
            pcBar_nM1 = pcBar_nM2*pcBar;
            pcBar_n   = pcBar_nM1*pcBar;
            onePlus_pcBar_n = 1.0 + pcBar_n;
	  
            sBar = pow(onePlus_pcBar_n,-m_vg);
            /* using -mn = 1-n */
            DsBar_DpsiC = alpha*(1.0-n_vg)*(sBar/onePlus_pcBar_n)*pcBar_nM1;
	  
            vBar = 1.0-pcBar_nM1*sBar;
            vBar2 = vBar*vBar;
            DvBar_DpsiC = -alpha*(n_vg-1.0)*pcBar_nM2*sBar - pcBar_nM1*DsBar_DpsiC;
	  
            thetaW = thetaSR*sBar + thetaR;
            DthetaW_DpsiC = thetaSR * DsBar_DpsiC; 
	  
            sqrt_sBar = sqrt(sBar);
            sqrt_sBarStar = sqrt_sBar;
            if (sqrt_sBar < 1.0e-8)
              sqrt_sBarStar = 1.0e-8;
            KWr= sqrt_sBar*vBar2;
            DKWr_DpsiC= ((0.5/sqrt_sBarStar)*DsBar_DpsiC*vBar2
                         +
                         2.0*sqrt_sBar*vBar*DvBar_DpsiC);
          }
        else
          {
            thetaW        = thetaS;
            DthetaW_DpsiC = 0.0;
            KWr           = 1.0;
            DKWr_DpsiC    = 0.0;
          }
        //slight compressibility
        //cek hack, force incompressible
        rhom = rho;//*exp(beta*u);
        drhom = 0.0;//beta*rhom;
        m = rhom*thetaW;
        dm = -rhom*DthetaW_DpsiC+drhom*thetaW;
        for (int I=0;I<nSpace;I++)
          {
            f[I] = 0.0;
            df[I] = 0.0;
            for (int ii=rowptr[I]; ii < rowptr[I+1]; ii++)
              {
                f[I]  += rho2*KWr*KWs[ii]*gravity[colind[ii]];
                df[I] += -rho2*DKWr_DpsiC*KWs[ii]*gravity[colind[ii]];
                a[ii]  = rho*KWs[ii];
                da[ii] = 0.0;
                kr = KWr;
                dkr=DKWr_DpsiC;
              }
          }
      }
      
      inline
        void evaluateInverseCoefficients(const int rowptr[nSpace],
                                         const int colind[nnz],
                                         const double rho,
                                         const double beta,
                                         const double gravity[nSpace],
                                         const double alpha,
                                         const double n_vg,
                                         const double thetaR,
                                         const double thetaSR,
                                         const double KWs[nnz],			      
                                         double& u,
                                         const double& m,
                                         const double& dm,
                                         const double f[nSpace],
                                         const double df[nSpace],
                                         const double a[nnz],
                                         const double da[nnz])
      {
        register double psiC,
          pcBar,pcBar_n,
          sBar,
          thetaW,
          thetaS,
          m_vg;
        m_vg = 1.0 - 1.0/n_vg;
        thetaS = thetaR + thetaSR;
        thetaW = m/rho;
        sBar = (fmax(thetaR+1.0e-8, fmin(thetaS, thetaW)) - thetaR)/thetaSR;
        //sBar = (thetaW - thetaR)/thetaSR;
        pcBar_n = pow(sBar, -1.0/m_vg) - 1.0;
        pcBar = pow(pcBar_n, 1.0/n_vg);
        psiC = pcBar/alpha;
        u = - psiC;
        if (thetaW > thetaS || thetaW <= thetaR)
          {
            //cek debug
            std::cout<<"n "<<n_vg<<std::endl
                     <<"m "<<m_vg<<std::endl
                     <<"thetaR "<<thetaR<<std::endl
                     <<"theta "<<thetaW<<std::endl
                     <<"thetaS "<<thetaS<<std::endl
                     <<"sBar "<<sBar<<std::endl
                     <<"pcBar_n "<<pcBar_n<<std::endl
                     <<"pcBar "<<pcBar<<std::endl
                     <<"psiC "<<psiC<<std::endl;
          }
      }

      inline
        void calculateCFL(const double& elementDiameter,
                          const double df[nSpace],
                          double& cfl)
      {
        double h,nrm_v;
        h = elementDiameter;
        nrm_v=0.0;
        for(int I=0;I<nSpace;I++)
          nrm_v+=df[I]*df[I];
        nrm_v = sqrt(nrm_v);
        cfl = nrm_v/h;
      }
      
      inline
        void calculateSubgridError_tau(const double& elementDiameter,
                                       const double& dmt,
                                       const double dH[nSpace],
                                       double& cfl,
                                       double& tau)
      {
        double h,nrm_v,oneByAbsdt;
        h = elementDiameter;
        nrm_v=0.0;
        for(int I=0;I<nSpace;I++)
          nrm_v+=dH[I]*dH[I];
        nrm_v = sqrt(nrm_v);
        cfl = nrm_v/h;
        oneByAbsdt =  fabs(dmt);
        tau = 1.0/(2.0*nrm_v/h + oneByAbsdt + 1.0e-8);
      }

 
      inline
        void calculateSubgridError_tau(     const double&  Ct_sge,
                                            const double   G[nSpace*nSpace],
                                            const double&  A0,
                                            const double   Ai[nSpace],
                                            double& tau_v,
                                            double& cfl)	
      {
        double v_d_Gv=0.0; 
        for(int I=0;I<nSpace;I++) 
          for (int J=0;J<nSpace;J++) 
            v_d_Gv += Ai[I]*G[I*nSpace+J]*Ai[J];     
    
        tau_v = 1.0/sqrt(Ct_sge*A0*A0 + v_d_Gv);    
      } 
 
 

      inline 
        void calculateNumericalDiffusion(const double& shockCapturingDiffusion,
                                         const double& elementDiameter,
                                         const double& strong_residual,
                                         const double grad_u[nSpace],
                                         double& numDiff)
      {
        double h,
          num,
          den,
          n_grad_u;
        h = elementDiameter;
        n_grad_u = 0.0;
        for (int I=0;I<nSpace;I++)
          n_grad_u += grad_u[I]*grad_u[I];
        num = shockCapturingDiffusion*0.5*h*fabs(strong_residual);
        den = sqrt(n_grad_u) + 1.0e-8;
        numDiff = num/den;
      }

      inline
        void exteriorNumericalFlux(const double& bc_flux,
                                   int rowptr[nSpace],
                                   int colind[nnz],
                                   int isSeepageFace,
                                   int& isDOFBoundary,
                                   double n[nSpace],
                                   double bc_u,
                                   double K[nnz],
                                   double grad_psi[nSpace],
                                   double u,
                                   double K_rho_g[nSpace],
                                   double penalty,
                                   double& flux)
      {
        double v_I,bc_u_seepage=0.0;
        if (isSeepageFace || isDOFBoundary)
          {
            flux = 0.0;
            for(int I=0;I<nSpace;I++)
              {
                //gravity
                v_I = K_rho_g[I];
                //pressure head
                for(int m=rowptr[I];m<rowptr[I+1];m++)
                  {
                    v_I -= K[m]*grad_psi[colind[m]];
                  }
                flux += v_I*n[I];
              }
            if (isSeepageFace)
              bc_u = bc_u_seepage;
            flux += penalty*(u-bc_u);
            if (isSeepageFace)
              {
                if (flux > 0.0)
                  {
                    isDOFBoundary = 1;
                    bc_u = bc_u_seepage;
                  }
                else
                  {
                    isDOFBoundary = 0;
                    flux = 0.0;
                  }
              }	  
            /* //set DOF flag and flux correctly if seepage face */
            /* if (isSeepageFace) */
            /*   { */
            /*     if (flux < 0.0 || u < bc_u_seepage) */
            /* 	{ */
            /* 	  isDOFBoundary = 0; */
            /* 	  flux = 0.0; */
            /* 	} */
            /*     else */
            /* 	{ */
            /* 	  isDOFBoundary = 1; */
            /* 	  bc_u = bc_u_seepage; */
            /* 	} */
            /*   } */
            /* //Dirichlet penalty */
            /* if (isDOFBoundary) */
            /*   flux += penalty*(u-bc_u); */
          }
        else
          flux = bc_flux;
      }

      void exteriorNumericalFluxJacobian(const int rowptr[nSpace],
                                         const int colind[nnz],
                                         const int isDOFBoundary,
                                         const double n[nSpace],
                                         const double K[nnz],
                                         const double dK[nnz],
                                         const double grad_psi[nSpace],
                                         const double grad_v[nSpace],
                                         const double dK_rho_g[nSpace],
                                         const double v,
                                         const double penalty,
                                         double& fluxJacobian)
      {
        if (isDOFBoundary)
          {
            fluxJacobian = 0.0;
            for(int I=0;I<nSpace;I++)
              {
                //gravity
                fluxJacobian += dK_rho_g[I]*v*n[I];
                //pressure head
                for(int m=rowptr[I]; m<rowptr[I+1]; m++)
                  {
                    fluxJacobian -= (K[m]*grad_v[colind[m]] + dK[m]*v*grad_psi[colind[m]])*n[I];
                  }
              }
            //Dirichlet penalty
            fluxJacobian += penalty*v;
          }
        else
          fluxJacobian = 0.0;
      }

      void calculateResidual(//element
                             double dt,
                             double* mesh_trial_ref,
                             double* mesh_grad_trial_ref,
                             double* mesh_dof,
                             double* mesh_velocity_dof,
                             double MOVING_DOMAIN,
                             int* mesh_l2g,
                             double* dV_ref,
                             double* u_trial_ref,
                             double* u_grad_trial_ref,
                             double* u_test_ref,
                             double* u_grad_test_ref,
                             //element boundary
                             double* mesh_trial_trace_ref,
                             double* mesh_grad_trial_trace_ref,
                             double* dS_ref,
                             double* u_trial_trace_ref,
                             double* u_grad_trial_trace_ref,
                             double* u_test_trace_ref,
                             double* u_grad_test_trace_ref,
                             double* normal_ref,
                             double* boundaryJac_ref,
                             //physics
                             int nElements_global,
                             double* ebqe_penalty_ext,
                             int* elementMaterialTypes,	
                             int* isSeepageFace,
                             int* a_rowptr,
                             int* a_colind,
                             double rho,
                             double beta,
                             double* gravity,
                             double* alpha,
                             double* n,
                             double* thetaR,
                             double* thetaSR,
                             double* KWs,
                             double useMetrics,
                             double alphaBDF,
                             int lag_shockCapturing,
                             double shockCapturingDiffusion,
                             double sc_uref,
                             double sc_alpha,
                             int* u_l2g,
                             int* r_l2g,
                             double* elementDiameter,
                             int degree_polynomial,
                             double* u_dof,
                             double* u_dof_old,	
                             double* velocity,
                             double* q_m,
                             double* q_u,
                             double* q_m_betaBDF,
                             double* cfl,
                             double* edge_based_cfl,
                             double* q_numDiff_u, 
                             double* q_numDiff_u_last, 
                             int offset_u, int stride_u, 
                             double* globalResidual,
                             int nExteriorElementBoundaries_global,
                             int* exteriorElementBoundariesArray,
                             int* elementBoundaryElementsArray,
                             int* elementBoundaryLocalElementBoundariesArray,
                             double* ebqe_velocity_ext,
                             int* isDOFBoundary_u,
                             double* ebqe_bc_u_ext,
                             int* isFluxBoundary_u,
                             double* ebqe_bc_flux_u_ext,
                             double* ebqe_phi,double epsFact,
                             double* ebqe_u,
                             double* ebqe_flux,
                             // PARAMETERS FOR EDGE BASED STABILIZATION
                             double cE,
                             double cK,
                             // PARAMETERS FOR LOG BASED ENTROPY FUNCTION
                             double uL,
                             double uR,
                             // PARAMETERS FOR EDGE VISCOSITY
                             int numDOFs,
                             int NNZ,
                             int* csrRowIndeces_DofLoops,
                             int* csrColumnOffsets_DofLoops,
                             int* csrRowIndeces_CellLoops,
                             int* csrColumnOffsets_CellLoops,
                             int* csrColumnOffsets_eb_CellLoops,
                             // C matrices
                             double* Cx,
                             double* Cy,
                             double* Cz,
                             double* CTx,
                             double* CTy,
                             double* CTz,
                             double* ML,
                             double* delta_x_ij,
                             // PARAMETERS FOR 1st or 2nd ORDER MPP METHOD
                             int LUMPED_MASS_MATRIX,
                             int STABILIZATION_TYPE,
                             int ENTROPY_TYPE,
                             // FOR FCT
                             double* dLow,
                             double* fluxMatrix,
                             double* uDotLow,
                             double* uLow,
                             double* dt_times_dH_minus_dL,
                             double* min_u_bc,
                             double* max_u_bc,
                             // AUX QUANTITIES OF INTEREST
                             double* quantDOFs)
      {
        assert(a_rowptr[nSpace] == nnz);
        assert(a_rowptr[nSpace] == nSpace);
        //cek should this be read in?
        double Ct_sge = 4.0;

        //loop over elements to compute volume integrals and load them into element and global residual
        //
        //eN is the element index
        //eN_k is the quadrature point index for a scalar
        //eN_k_nSpace is the quadrature point index for a vector
        //eN_i is the element test function index
        //eN_j is the element trial function index
        //eN_k_j is the quadrature point index for a trial function
        //eN_k_i is the quadrature point index for a trial function
        for(int eN=0;eN<nElements_global;eN++)
          {
            //declare local storage for element residual and initialize
            register double elementResidual_u[nDOF_test_element];
            for (int i=0;i<nDOF_test_element;i++)
              {
                elementResidual_u[i]=0.0;
              }//i
            //loop over quadrature points and compute integrands
            for  (int k=0;k<nQuadraturePoints_element;k++)
              {
                //compute indeces and declare local storage
                register int eN_k = eN*nQuadraturePoints_element+k,
                  eN_k_nSpace = eN_k*nSpace,
                  eN_nDOF_trial_element = eN*nDOF_trial_element;
                register double u=0.0,grad_u[nSpace],grad_u_old[nSpace],
                  m=0.0,dm=0.0,
                  f[nSpace],df[nSpace],
                  a[nnz],da[nnz],
                  m_t=0.0,dm_t=0.0,
                  pdeResidual_u=0.0,
                  Lstar_u[nDOF_test_element],
                  subgridError_u=0.0,
                  tau=0.0,tau0=0.0,tau1=0.0,
                  numDiff0=0.0,numDiff1=0.0,
                  jac[nSpace*nSpace],
                  jacDet,
                  jacInv[nSpace*nSpace],
                  u_grad_trial[nDOF_trial_element*nSpace],
                  u_test_dV[nDOF_trial_element],
                  u_grad_test_dV[nDOF_test_element*nSpace],
                  dV,x,y,z,xt,yt,zt,
                  G[nSpace*nSpace],G_dd_G,tr_G,norm_Rv;
                //
                //compute solution and gradients at quadrature points
                //
                ck.calculateMapping_element(eN,
                                            k,
                                            mesh_dof,
                                            mesh_l2g,
                                            mesh_trial_ref,
                                            mesh_grad_trial_ref,
                                            jac,
                                            jacDet,
                                            jacInv,
                                            x,y,z);
                ck.calculateMappingVelocity_element(eN,
                                                    k,
                                                    mesh_velocity_dof,
                                                    mesh_l2g,
                                                    mesh_trial_ref,
                                                    xt,yt,zt);
                //get the physical integration weight
                dV = fabs(jacDet)*dV_ref[k];
                ck.calculateG(jacInv,G,G_dd_G,tr_G);
                //get the trial function gradients
                ck.gradTrialFromRef(&u_grad_trial_ref[k*nDOF_trial_element*nSpace],jacInv,u_grad_trial);
                //get the solution
                ck.valFromDOF(u_dof,&u_l2g[eN_nDOF_trial_element],&u_trial_ref[k*nDOF_trial_element],u);
                //get the solution gradients
                ck.gradFromDOF(u_dof,&u_l2g[eN_nDOF_trial_element],u_grad_trial,grad_u);
                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    u_test_dV[j] = u_test_ref[k*nDOF_trial_element+j]*dV;
                    for (int I=0;I<nSpace;I++)
                      {
                        u_grad_test_dV[j*nSpace+I]   = u_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin
                      }
                  }
                //
                //calculate pde coefficients at quadrature points
                //
                double Kr,dKr;
                evaluateCoefficients(a_rowptr,
                                     a_colind,
                                     rho,
                                     beta,
                                     gravity,
                                     alpha[elementMaterialTypes[eN]],
                                     n[elementMaterialTypes[eN]],
                                     thetaR[elementMaterialTypes[eN]],
                                     thetaSR[elementMaterialTypes[eN]],
                                     &KWs[elementMaterialTypes[eN]*nnz],			      
                                     u,
                                     m,
                                     dm,
                                     f,
                                     df,
                                     a,
                                     da,
                                     Kr,
                                     dKr);
                //
                //calculate time derivative at quadrature points
                //
                ck.bdf(alphaBDF,
                       q_m_betaBDF[eN_k],
                       m,
                       dm,
                       m_t,
                       dm_t);
                /* // */
                /* //calculate subgrid error (strong residual and adjoint) */
                /* // */
                /* //calculate strong residual */
                /* pdeResidual_u = ck.Mass_strong(m_t) + ck.Advection_strong(df,grad_u); */
                /* //calculate adjoint */
                /* for (int i=0;i<nDOF_test_element;i++) */
                /* 	{ */
                /* 	  // register int eN_k_i_nSpace = (eN_k*nDOF_trial_element+i)*nSpace; */
                /* 	  // Lstar_u[i]  = ck.Advection_adjoint(df,&u_grad_test_dV[eN_k_i_nSpace]); */
                /* 	  register int i_nSpace = i*nSpace; */
                /* 	  Lstar_u[i]  = ck.Advection_adjoint(df,&u_grad_test_dV[i_nSpace]); */
                /* 	} */
                /* //calculate tau and tau*Res */
                /* calculateSubgridError_tau(elementDiameter[eN],dm_t,df,cfl[eN_k],tau0); */
                /* calculateSubgridError_tau(Ct_sge, */
                /*                           G, */
                /* 				dm_t, */
                /* 				df, */
                /* 				tau1, */
                /* 			        cfl[eN_k]); */
					
                /* tau = useMetrics*tau1+(1.0-useMetrics)*tau0; */

                /* subgridError_u = -tau*pdeResidual_u; */
                /* // */
                /* //calculate shock capturing diffusion */
                /* // */

	      
                /* ck.calculateNumericalDiffusion(shockCapturingDiffusion,elementDiameter[eN],pdeResidual_u,grad_u,numDiff0);	       */
                /* //ck.calculateNumericalDiffusion(shockCapturingDiffusion,G,pdeResidual_u,grad_u_old,numDiff1); */
                /* ck.calculateNumericalDiffusion(shockCapturingDiffusion,sc_uref, sc_alpha,G,G_dd_G,pdeResidual_u,grad_u,numDiff1); */
                /* q_numDiff_u[eN_k] = useMetrics*numDiff1+(1.0-useMetrics)*numDiff0; */
                //std::cout<<tau<<"   "<<q_numDiff_u[eN_k]<<std::endl;
                // 
                //update element residual 
                // 
                for(int i=0;i<nDOF_test_element;i++) 
                  { 
                    register int eN_k_i=eN_k*nDOF_test_element+i,
                      eN_k_i_nSpace = eN_k_i*nSpace,
                      i_nSpace=i*nSpace;

                    elementResidual_u[i] += ck.Mass_weak(m_t,u_test_dV[i]) + 
                      ck.Advection_weak(f,&u_grad_test_dV[i_nSpace]) + 
                      ck.Diffusion_weak(a_rowptr,a_colind,a,grad_u,&u_grad_test_dV[i_nSpace]);
                    /* +  */
                    /*   ck.SubgridError(subgridError_u,Lstar_u[i]) +  */
                    /*   ck.NumericalDiffusion(q_numDiff_u_last[eN_k],grad_u,&u_grad_test_dV[i_nSpace]);  */	      
                  }//i
                //
                q_m[eN_k] = m;
                q_u[eN_k] = u;
              }
            //
            //load element into global residual and save element residual
            //
            for(int i=0;i<nDOF_test_element;i++) 
              { 
                register int eN_i=eN*nDOF_test_element+i;
          
                globalResidual[offset_u+stride_u*u_l2g[eN_i]] += elementResidual_u[i];
              }//i
          }//elements
        //
        //loop over exterior element boundaries to calculate surface integrals and load into element and global residuals
        //
        //ebNE is the Exterior element boundary INdex
        //ebN is the element boundary INdex
        //eN is the element index
        for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++) 
          { 
            register int ebN = exteriorElementBoundariesArray[ebNE], 
              eN  = elementBoundaryElementsArray[ebN*2+0],
              ebN_local = elementBoundaryLocalElementBoundariesArray[ebN*2+0],
              eN_nDOF_trial_element = eN*nDOF_trial_element;
            register double elementResidual_u[nDOF_test_element];
            for (int i=0;i<nDOF_test_element;i++)
              {
                elementResidual_u[i]=0.0;
              }
            for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++) 
              { 
                register int ebNE_kb = ebNE*nQuadraturePoints_elementBoundary+kb,
                  ebNE_kb_nSpace = ebNE_kb*nSpace,
                  ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb,
                  ebN_local_kb_nSpace = ebN_local_kb*nSpace;
                register double u_ext=0.0,
                  grad_u_ext[nSpace],
                  m_ext=0.0,
                  dm_ext=0.0,
                  f_ext[nSpace],
                  df_ext[nSpace],
                  a_ext[nnz],
                  da_ext[nnz],
                  flux_ext=0.0,
                  bc_u_ext=0.0,
                  bc_grad_u_ext[nSpace],
                  bc_m_ext=0.0,
                  bc_dm_ext=0.0,
                  bc_f_ext[nSpace],
                  bc_df_ext[nSpace],
                  bc_a_ext[nnz],
                  bc_da_ext[nnz],
                  jac_ext[nSpace*nSpace],
                  jacDet_ext,
                  jacInv_ext[nSpace*nSpace],
                  boundaryJac[nSpace*(nSpace-1)],
                  metricTensor[(nSpace-1)*(nSpace-1)],
                  metricTensorDetSqrt,
                  dS,
                  u_test_dS[nDOF_test_element],
                  u_grad_trial_trace[nDOF_trial_element*nSpace],
                  normal[3],x_ext,y_ext,z_ext,xt_ext,yt_ext,zt_ext,integralScaling,
                  G[nSpace*nSpace],G_dd_G,tr_G;
                // 
                //calculate the solution and gradients at quadrature points 
                // 
                //compute information about mapping from reference element to physical element
                ck.calculateMapping_elementBoundary(eN,
                                                    ebN_local,
                                                    kb,
                                                    ebN_local_kb,
                                                    mesh_dof,
                                                    mesh_l2g,
                                                    mesh_trial_trace_ref,
                                                    mesh_grad_trial_trace_ref,
                                                    boundaryJac_ref,
                                                    jac_ext,
                                                    jacDet_ext,
                                                    jacInv_ext,
                                                    boundaryJac,
                                                    metricTensor,
                                                    metricTensorDetSqrt,
                                                    normal_ref,
                                                    normal,
                                                    x_ext,y_ext,z_ext);
                ck.calculateMappingVelocity_elementBoundary(eN,
                                                            ebN_local,
                                                            kb,
                                                            ebN_local_kb,
                                                            mesh_velocity_dof,
                                                            mesh_l2g,
                                                            mesh_trial_trace_ref,
                                                            xt_ext,yt_ext,zt_ext,
                                                            normal,
                                                            boundaryJac,
                                                            metricTensor,
                                                            integralScaling);
                dS = ((1.0-MOVING_DOMAIN)*metricTensorDetSqrt + MOVING_DOMAIN*integralScaling)*dS_ref[kb];
                //get the metric tensor
                //cek todo use symmetry
                ck.calculateG(jacInv_ext,G,G_dd_G,tr_G);
                //compute shape and solution information
                //shape
                ck.gradTrialFromRef(&u_grad_trial_trace_ref[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,u_grad_trial_trace);
                //solution and gradients	
                ck.valFromDOF(u_dof,&u_l2g[eN_nDOF_trial_element],&u_trial_trace_ref[ebN_local_kb*nDOF_test_element],u_ext);
                ck.gradFromDOF(u_dof,&u_l2g[eN_nDOF_trial_element],u_grad_trial_trace,grad_u_ext);
                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    u_test_dS[j] = u_test_trace_ref[ebN_local_kb*nDOF_test_element+j]*dS;
                  }
                //
                //load the boundary values
                //
                bc_u_ext = isDOFBoundary_u[ebNE_kb]*ebqe_bc_u_ext[ebNE_kb]+(1-isDOFBoundary_u[ebNE_kb])*u_ext;
                // 
                //calculate the pde coefficients using the solution and the boundary values for the solution 
                //
                double Kr, dKr;
                evaluateCoefficients(a_rowptr,
                                     a_colind,
                                     rho,
                                     beta,
                                     gravity,
                                     alpha[elementMaterialTypes[eN]],
                                     n[elementMaterialTypes[eN]],
                                     thetaR[elementMaterialTypes[eN]],
                                     thetaSR[elementMaterialTypes[eN]],
                                     &KWs[elementMaterialTypes[eN]*nnz],			      
                                     u_ext,
                                     m_ext,
                                     dm_ext,
                                     f_ext,
                                     df_ext,
                                     a_ext,
                                     da_ext,
                                     Kr,
                                     dKr);
                evaluateCoefficients(a_rowptr,
                                     a_colind,
                                     rho,
                                     beta,
                                     gravity,
                                     alpha[elementMaterialTypes[eN]],
                                     n[elementMaterialTypes[eN]],
                                     thetaR[elementMaterialTypes[eN]],
                                     thetaSR[elementMaterialTypes[eN]],
                                     &KWs[elementMaterialTypes[eN]*nnz],			      
                                     bc_u_ext,
                                     bc_m_ext,
                                     bc_dm_ext,
                                     bc_f_ext,
                                     bc_df_ext,
                                     bc_a_ext,
                                     bc_da_ext,
                                     Kr,
                                     dKr);
                // 
                //calculate the numerical fluxes 
                // 
                exteriorNumericalFlux(ebqe_bc_flux_u_ext[ebNE_kb],
                                      a_rowptr,
                                      a_colind,
                                      isSeepageFace[ebNE],//tricky, this is a face flag not face quad
                                      isDOFBoundary_u[ebNE_kb],
                                      normal,
                                      bc_u_ext,
                                      a_ext,
                                      grad_u_ext,
                                      u_ext,
                                      f_ext,
                                      ebqe_penalty_ext[ebNE_kb],//				    penalty,
                                      flux_ext);
                ebqe_flux[ebNE_kb] = flux_ext;
                ebqe_u[ebNE_kb] = u_ext;
                //
                //update residuals
                //
                for (int i=0;i<nDOF_test_element;i++)
                  {
                    int ebNE_kb_i = ebNE_kb*nDOF_test_element+i;

                    elementResidual_u[i] += ck.ExteriorElementBoundaryFlux(flux_ext,u_test_dS[i]);
                  }//i
              }//kb
            //
            //update the element and global residual storage
            //
            for (int i=0;i<nDOF_test_element;i++)
              {
                int eN_i = eN*nDOF_test_element+i;

                globalResidual[offset_u+stride_u*u_l2g[eN_i]] += elementResidual_u[i];
              }//i
          }//ebNE
      }

      void calculateJacobian(//element
                             double dt,
                             double* mesh_trial_ref,
                             double* mesh_grad_trial_ref,
                             double* mesh_dof,
                             double* mesh_velocity_dof,
                             double MOVING_DOMAIN,
                             int* mesh_l2g,
                             double* dV_ref,
                             double* u_trial_ref,
                             double* u_grad_trial_ref,
                             double* u_test_ref,
                             double* u_grad_test_ref,
                             //element boundary
                             double* mesh_trial_trace_ref,
                             double* mesh_grad_trial_trace_ref,
                             double* dS_ref,
                             double* u_trial_trace_ref,
                             double* u_grad_trial_trace_ref,
                             double* u_test_trace_ref,
                             double* u_grad_test_trace_ref,
                             double* normal_ref,
                             double* boundaryJac_ref,
                             //physics
                             int nElements_global,
                             //new
                             double* ebqe_penalty_ext,
                             int* elementMaterialTypes,	
                             int* isSeepageFace,
                             int* a_rowptr,
                             int* a_colind,
                             double rho,
                             double beta,
                             double* gravity,
                             double* alpha,
                             double* n,
                             double* thetaR,
                             double* thetaSR,
                             double* KWs,
                             //end new
                             double useMetrics, 
                             double alphaBDF,
                             int lag_shockCapturing,
                             double shockCapturingDiffusion,
                             int* u_l2g,
                             int* r_l2g,
                             double* elementDiameter,
                             int degree_polynomial,
                             double* u_dof, 
                             double* velocity,
                             double* q_m_betaBDF, 
                             double* cfl,
                             double* q_numDiff_u_last, 
                             int* csrRowIndeces_u_u,int* csrColumnOffsets_u_u,
                             double* globalJacobian,
                             double* delta_x_ij,
                             int nExteriorElementBoundaries_global,
                             int* exteriorElementBoundariesArray,
                             int* elementBoundaryElementsArray,
                             int* elementBoundaryLocalElementBoundariesArray,
                             double* ebqe_velocity_ext,
                             int* isDOFBoundary_u,
                             double* ebqe_bc_u_ext,
                             int* isFluxBoundary_u,
                             double* ebqe_bc_flux_u_ext,
                             int* csrColumnOffsets_eb_u_u,
                             int LUMPED_MASS_MATRIX)
      {
        assert(a_rowptr[nSpace] == nnz);
        assert(a_rowptr[nSpace] == nSpace);
        double Ct_sge = 4.0;
    
        //
        //loop over elements to compute volume integrals and load them into the element Jacobians and global Jacobian
        //
        for(int eN=0;eN<nElements_global;eN++)
          {
            register double  elementJacobian_u_u[nDOF_test_element][nDOF_trial_element];
            for (int i=0;i<nDOF_test_element;i++)
              for (int j=0;j<nDOF_trial_element;j++)
                {
                  elementJacobian_u_u[i][j]=0.0;
                }
            for  (int k=0;k<nQuadraturePoints_element;k++)
              {
                int eN_k = eN*nQuadraturePoints_element+k, //index to a scalar at a quadrature point
                  eN_k_nSpace = eN_k*nSpace,
                  eN_nDOF_trial_element = eN*nDOF_trial_element; //index to a vector at a quadrature point

                //declare local storage
                register double u=0.0,
                  grad_u[nSpace],
                  m=0.0,dm=0.0,
                  f[nSpace],df[nSpace],
                  a[nnz],da[nnz],
                  m_t=0.0,dm_t=0.0,
                  dpdeResidual_u_u[nDOF_trial_element],
                  Lstar_u[nDOF_test_element],
                  dsubgridError_u_u[nDOF_trial_element],
                  tau=0.0,tau0=0.0,tau1=0.0,
                  jac[nSpace*nSpace],
                  jacDet,
                  jacInv[nSpace*nSpace],
                  u_grad_trial[nDOF_trial_element*nSpace],
                  dV,
                  u_test_dV[nDOF_test_element],
                  u_grad_test_dV[nDOF_test_element*nSpace],
                  x,y,z,xt,yt,zt,
                  G[nSpace*nSpace],G_dd_G,tr_G;
                //
                //calculate solution and gradients at quadrature points
                //
                //get jacobian, etc for mapping reference element
                ck.calculateMapping_element(eN,
                                            k,
                                            mesh_dof,
                                            mesh_l2g,
                                            mesh_trial_ref,
                                            mesh_grad_trial_ref,
                                            jac,
                                            jacDet,
                                            jacInv,
                                            x,y,z);
                ck.calculateMappingVelocity_element(eN,
                                                    k,
                                                    mesh_velocity_dof,
                                                    mesh_l2g,
                                                    mesh_trial_ref,
                                                    xt,yt,zt);
                //get the physical integration weight
                dV = fabs(jacDet)*dV_ref[k];
                ck.calculateG(jacInv,G,G_dd_G,tr_G);
                //get the trial function gradients
                ck.gradTrialFromRef(&u_grad_trial_ref[k*nDOF_trial_element*nSpace],jacInv,u_grad_trial);
                //get the solution 	
                ck.valFromDOF(u_dof,&u_l2g[eN_nDOF_trial_element],&u_trial_ref[k*nDOF_trial_element],u);
                //get the solution gradients
                ck.gradFromDOF(u_dof,&u_l2g[eN_nDOF_trial_element],u_grad_trial,grad_u);
                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    u_test_dV[j] = u_test_ref[k*nDOF_trial_element+j]*dV;
                    for (int I=0;I<nSpace;I++)
                      {
                        u_grad_test_dV[j*nSpace+I]   = u_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin
                      }
                  }
                //
                //calculate pde coefficients and derivatives at quadrature points
                //
                double Kr,dKr;
                evaluateCoefficients(a_rowptr,
                                     a_colind,
                                     rho,
                                     beta,
                                     gravity,
                                     alpha[elementMaterialTypes[eN]],
                                     n[elementMaterialTypes[eN]],
                                     thetaR[elementMaterialTypes[eN]],
                                     thetaSR[elementMaterialTypes[eN]],
                                     &KWs[elementMaterialTypes[eN]*nnz],			      
                                     u,
                                     m,
                                     dm,
                                     f,
                                     df,
                                     a,
                                     da,
                                     Kr,
                                     dKr);
                //
                //calculate time derivatives
                //
                ck.bdf(alphaBDF,
                       q_m_betaBDF[eN_k],
                       m,
                       dm,
                       m_t,
                       dm_t);
                // //
                // //calculate subgrid error contribution to the Jacobian (strong residual, adjoint, jacobian of strong residual)
                // //
                // //calculate the adjoint times the test functions
                // for (int i=0;i<nDOF_test_element;i++)
                // 	{
                // 	  // int eN_k_i_nSpace = (eN_k*nDOF_trial_element+i)*nSpace;
                // 	  // Lstar_u[i]=ck.Advection_adjoint(df,&u_grad_test_dV[eN_k_i_nSpace]);	      
                // 	  register int i_nSpace = i*nSpace;
                // 	  Lstar_u[i]=ck.Advection_adjoint(df,&u_grad_test_dV[i_nSpace]);	      
                // 	}
                // //calculate the Jacobian of strong residual
                // for (int j=0;j<nDOF_trial_element;j++)
                // 	{
                // 	  //int eN_k_j=eN_k*nDOF_trial_element+j;
                // 	  //int eN_k_j_nSpace = eN_k_j*nSpace;
                // 	  int j_nSpace = j*nSpace;
                // 	  dpdeResidual_u_u[j]= ck.MassJacobian_strong(dm_t,u_trial_ref[k*nDOF_trial_element+j]) +
                // 	    ck.AdvectionJacobian_strong(df,&u_grad_trial[j_nSpace]);
                // 	}
                // //tau and tau*Res
                // calculateSubgridError_tau(elementDiameter[eN],
                // 				dm_t,
                // 				df,
                // 				cfl[eN_k],
                // 				tau0);
  
                // calculateSubgridError_tau(Ct_sge,
                //                           G,
                // 				dm_t,
                // 				df,
                // 				tau1,
                // 			        cfl[eN_k]);
                // tau = useMetrics*tau1+(1.0-useMetrics)*tau0;

                // for(int j=0;j<nDOF_trial_element;j++)
                // 	dsubgridError_u_u[j] = -tau*dpdeResidual_u_u[j];
                for(int i=0;i<nDOF_test_element;i++)
                  {
                    //int eN_k_i=eN_k*nDOF_test_element+i;
                    //int eN_k_i_nSpace=eN_k_i*nSpace;
                    for(int j=0;j<nDOF_trial_element;j++) 
                      { 
                        //int eN_k_j=eN_k*nDOF_trial_element+j;
                        //int eN_k_j_nSpace = eN_k_j*nSpace;
                        int j_nSpace = j*nSpace;
                        int i_nSpace = i*nSpace;
                        elementJacobian_u_u[i][j] += ck.MassJacobian_weak(dm_t,u_trial_ref[k*nDOF_trial_element+j],u_test_dV[i]) + 
                          ck.AdvectionJacobian_weak(df,u_trial_ref[k*nDOF_trial_element+j],&u_grad_test_dV[i_nSpace]) +
                          ck.DiffusionJacobian_weak(a_rowptr,a_colind,a,da,
                                                    grad_u,&u_grad_test_dV[i_nSpace],1.0,
                                                    u_trial_ref[k*nDOF_trial_element+j],&u_grad_trial[j_nSpace]);
                        // +
                        // 	ck.SubgridErrorJacobian(dsubgridError_u_u[j],Lstar_u[i]) + 
                        // 	ck.NumericalDiffusionJacobian(q_numDiff_u_last[eN_k],&u_grad_trial[j_nSpace],&u_grad_test_dV[i_nSpace]); 
                      }//j
                  }//i
              }//k
            //
            //load into element Jacobian into global Jacobian
            //
            for (int i=0;i<nDOF_test_element;i++)
              {
                int eN_i = eN*nDOF_test_element+i;
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    int eN_i_j = eN_i*nDOF_trial_element+j;
                    globalJacobian[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_u_u[eN_i_j]] += elementJacobian_u_u[i][j];
                  }//j
              }//i
          }//elements
        //
        //loop over exterior element boundaries to compute the surface integrals and load them into the global Jacobian
        //
        for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++) 
          { 
            register int ebN = exteriorElementBoundariesArray[ebNE]; 
            register int eN  = elementBoundaryElementsArray[ebN*2+0],
              ebN_local = elementBoundaryLocalElementBoundariesArray[ebN*2+0],
              eN_nDOF_trial_element = eN*nDOF_trial_element;
            for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++) 
              { 
                register int ebNE_kb = ebNE*nQuadraturePoints_elementBoundary+kb,
                  ebNE_kb_nSpace = ebNE_kb*nSpace,
                  ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb,
                  ebN_local_kb_nSpace = ebN_local_kb*nSpace;

                register double u_ext=0.0,
                  grad_u_ext[nSpace],
                  m_ext=0.0,
                  dm_ext=0.0,
                  f_ext[nSpace],
                  df_ext[nSpace],
                  a_ext[nnz],
                  da_ext[nnz],
                  dflux_u_u_ext=0.0,
                  bc_u_ext=0.0,
                  //bc_grad_u_ext[nSpace],
                  bc_m_ext=0.0,
                  bc_dm_ext=0.0,
                  bc_f_ext[nSpace],
                  bc_df_ext[nSpace],
                  bc_a_ext[nnz],
                  bc_da_ext[nnz],
                  fluxJacobian_u_u[nDOF_trial_element],
                  jac_ext[nSpace*nSpace],
                  jacDet_ext,
                  jacInv_ext[nSpace*nSpace],
                  boundaryJac[nSpace*(nSpace-1)],
                  metricTensor[(nSpace-1)*(nSpace-1)],
                  metricTensorDetSqrt,
                  dS,
                  u_test_dS[nDOF_test_element],
                  u_grad_trial_trace[nDOF_trial_element*nSpace],
                  normal[3],x_ext,y_ext,z_ext,xt_ext,yt_ext,zt_ext,integralScaling,
                  G[nSpace*nSpace],G_dd_G,tr_G;
                // 
                //calculate the solution and gradients at quadrature points 
                // 
                ck.calculateMapping_elementBoundary(eN,
                                                    ebN_local,
                                                    kb,
                                                    ebN_local_kb,
                                                    mesh_dof,
                                                    mesh_l2g,
                                                    mesh_trial_trace_ref,
                                                    mesh_grad_trial_trace_ref,
                                                    boundaryJac_ref,
                                                    jac_ext,
                                                    jacDet_ext,
                                                    jacInv_ext,
                                                    boundaryJac,
                                                    metricTensor,
                                                    metricTensorDetSqrt,
                                                    normal_ref,
                                                    normal,
                                                    x_ext,y_ext,z_ext);
                ck.calculateMappingVelocity_elementBoundary(eN,
                                                            ebN_local,
                                                            kb,
                                                            ebN_local_kb,
                                                            mesh_velocity_dof,
                                                            mesh_l2g,
                                                            mesh_trial_trace_ref,
                                                            xt_ext,yt_ext,zt_ext,
                                                            normal,
                                                            boundaryJac,
                                                            metricTensor,
                                                            integralScaling);
                dS = ((1.0-MOVING_DOMAIN)*metricTensorDetSqrt + MOVING_DOMAIN*integralScaling)*dS_ref[kb];
                //dS = metricTensorDetSqrt*dS_ref[kb];
                ck.calculateG(jacInv_ext,G,G_dd_G,tr_G);
                //compute shape and solution information
                //shape
                ck.gradTrialFromRef(&u_grad_trial_trace_ref[ebN_local_kb_nSpace*nDOF_trial_element],jacInv_ext,u_grad_trial_trace);
                //solution and gradients	
                ck.valFromDOF(u_dof,&u_l2g[eN_nDOF_trial_element],&u_trial_trace_ref[ebN_local_kb*nDOF_test_element],u_ext);
                ck.gradFromDOF(u_dof,&u_l2g[eN_nDOF_trial_element],u_grad_trial_trace,grad_u_ext);
                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    u_test_dS[j] = u_test_trace_ref[ebN_local_kb*nDOF_test_element+j]*dS;
                  }
                //
                //load the boundary values
                //
                bc_u_ext = isDOFBoundary_u[ebNE_kb]*ebqe_bc_u_ext[ebNE_kb]+(1-isDOFBoundary_u[ebNE_kb])*u_ext;
                // 
                //calculate the internal and external trace of the pde coefficients 
                //
                double Kr, dKr;
                evaluateCoefficients(a_rowptr,
                                     a_colind,
                                     rho,
                                     beta,
                                     gravity,
                                     alpha[elementMaterialTypes[eN]],
                                     n[elementMaterialTypes[eN]],
                                     thetaR[elementMaterialTypes[eN]],
                                     thetaSR[elementMaterialTypes[eN]],
                                     &KWs[elementMaterialTypes[eN]*nnz],			      
                                     u_ext,
                                     m_ext,
                                     dm_ext,
                                     f_ext,
                                     df_ext,
                                     a_ext,
                                     da_ext,
                                     Kr,
                                     dKr);
                evaluateCoefficients(a_rowptr,
                                     a_colind,
                                     rho,
                                     beta,
                                     gravity,
                                     alpha[elementMaterialTypes[eN]],
                                     n[elementMaterialTypes[eN]],
                                     thetaR[elementMaterialTypes[eN]],
                                     thetaSR[elementMaterialTypes[eN]],
                                     &KWs[elementMaterialTypes[eN]*nnz],			      
                                     bc_u_ext,
                                     bc_m_ext,
                                     bc_dm_ext,
                                     bc_f_ext,
                                     bc_df_ext,
                                     bc_a_ext,
                                     bc_da_ext,
                                     Kr,
                                     dKr);
                //
                //calculate the flux jacobian
                //
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    //register int ebNE_kb_j = ebNE_kb*nDOF_trial_element+j;
                    register int ebN_local_kb_j=ebN_local_kb*nDOF_trial_element+j;
                    exteriorNumericalFluxJacobian(a_rowptr,
                                                  a_colind,
                                                  isDOFBoundary_u[ebNE_kb],
                                                  normal,
                                                  a_ext,
                                                  da_ext,
                                                  grad_u_ext,
                                                  &u_grad_trial_trace[j*nSpace],
                                                  df_ext,
                                                  u_trial_trace_ref[ebN_local_kb*nDOF_test_element+j],
                                                  ebqe_penalty_ext[ebNE_kb],//penalty,
                                                  fluxJacobian_u_u[j]);
                  }//j
                //
                //update the global Jacobian from the flux Jacobian
                //
                for (int i=0;i<nDOF_test_element;i++)
                  {
                    register int eN_i = eN*nDOF_test_element+i;
                    //register int ebNE_kb_i = ebNE_kb*nDOF_test_element+i;
                    for (int j=0;j<nDOF_trial_element;j++)
                      {
                        register int ebN_i_j = ebN*4*nDOF_test_X_trial_element + i*nDOF_trial_element + j;
		      
                        globalJacobian[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_eb_u_u[ebN_i_j]] += fluxJacobian_u_u[j]*u_test_dS[i];
                      }//j
                  }//i
              }//kb
          }//ebNE
      }//computeJacobian
      void FCTStep(int NNZ, //number on non-zero entries on sparsity pattern
                   int numDOFs, //number of DOFs
                   double* lumped_mass_matrix, //lumped mass matrix (as vector)
                   double* soln, //DOFs of solution at time tn
                   double* solH, //DOFs of high order solution at tnp1
                   double* uLow,
                   double* limited_solution,
                   int* csrRowIndeces_DofLoops, //csr row indeces
                   int* csrColumnOffsets_DofLoops, //csr column offsets
                   double* MassMatrix, //mass matrix
                   double* dt_times_dH_minus_dL, //low minus high order dissipative matrices
                   double* min_u_bc, //min/max value at BCs. If DOF is not at boundary then min=1E10, max=-1E10
                   double* max_u_bc,
                   int LUMPED_MASS_MATRIX
                   )
      {
        register double Rpos[numDOFs], Rneg[numDOFs];
        register double FluxCorrectionMatrix[NNZ];
        register double solL[numDOFs];
        //////////////////
        // LOOP in DOFs //
        //////////////////
        int ij=0;
        for (int i=0; i<numDOFs; i++)
          {
            //read some vectors
            double solHi = solH[i];
            double solni = soln[i];
            double mi = lumped_mass_matrix[i];
            // compute low order solution
            // mi*(uLi-uni) + dt*sum_j[(Tij+dLij)*unj] = 0
            solL[i] = uLow[i];

            double mini=min_u_bc[i], maxi=max_u_bc[i]; // init min/max with value at BCs (NOTE: if no boundary then min=1E10, max=-1E10)
            if (GLOBAL_FCT==1)
              {
                mini = 0.;
                maxi = 1.;
              }

            double Pposi=0, Pnegi=0;
            // LOOP OVER THE SPARSITY PATTERN (j-LOOP)//
            for (int offset=csrRowIndeces_DofLoops[i]; offset<csrRowIndeces_DofLoops[i+1]; offset++)
              {
                int j = csrColumnOffsets_DofLoops[offset];
                ////////////////////////
                // COMPUTE THE BOUNDS //
                ////////////////////////
                if (GLOBAL_FCT == 0)
                  {
                    mini = fmin(mini,soln[j]);
                    maxi = fmax(maxi,soln[j]);
                  }
                // i-th row of flux correction matrix
                double ML_minus_MC = (LUMPED_MASS_MATRIX == 1 ? 0. : (i==j ? 1. : 0.)*mi - MassMatrix[ij]);
                FluxCorrectionMatrix[ij] = ML_minus_MC * (solH[j]-soln[j] - (solHi-solni))
                  + dt_times_dH_minus_dL[ij]*(soln[j]-solni);

                ///////////////////////
                // COMPUTE P VECTORS //
                ///////////////////////
                Pposi += FluxCorrectionMatrix[ij]*((FluxCorrectionMatrix[ij] > 0) ? 1. : 0.);
                Pnegi += FluxCorrectionMatrix[ij]*((FluxCorrectionMatrix[ij] < 0) ? 1. : 0.);

                //update ij
                ij+=1;
              }
            ///////////////////////
            // COMPUTE Q VECTORS //
            ///////////////////////
            double Qposi = mi*(maxi-solL[i]);
            double Qnegi = mi*(mini-solL[i]);

            ///////////////////////
            // COMPUTE R VECTORS //
            ///////////////////////
            Rpos[i] = ((Pposi==0) ? 1. : fmin(1.0,Qposi/Pposi));
            Rneg[i] = ((Pnegi==0) ? 1. : fmin(1.0,Qnegi/Pnegi));
          } // i DOFs

        //////////////////////
        // COMPUTE LIMITERS //
        //////////////////////
        ij=0;
        for (int i=0; i<numDOFs; i++)
          {
            double ith_Limiter_times_FluxCorrectionMatrix = 0.;
            double Rposi = Rpos[i], Rnegi = Rneg[i];
            // LOOP OVER THE SPARSITY PATTERN (j-LOOP)//
            for (int offset=csrRowIndeces_DofLoops[i]; offset<csrRowIndeces_DofLoops[i+1]; offset++)
              {
                int j = csrColumnOffsets_DofLoops[offset];
                ith_Limiter_times_FluxCorrectionMatrix +=
                  ((FluxCorrectionMatrix[ij]>0) ? fmin(Rposi,Rneg[j]) : fmin(Rnegi,Rpos[j]))
                  * FluxCorrectionMatrix[ij];
                //ith_Limiter_times_FluxCorrectionMatrix += FluxCorrectionMatrix[ij];
                //update ij
                ij+=1;
              }
            limited_solution[i] = solL[i] + 1./lumped_mass_matrix[i]*ith_Limiter_times_FluxCorrectionMatrix;
          }
      }

      void kth_FCT_step(double dt,
                        int num_fct_iter,
                        int NNZ, //number on non-zero entries on sparsity pattern
                        int numDOFs, //number of DOFs
                        double* MC,
                        double* ML,
                        double* soln,
                        double* solLim, // INPUT/OUTPUT
                        double* uDotLow,
                        double* uLow,
                        double* dLow,
                        double* FluxMatrix, 
                        double* limitedFlux, // INPUT/OUTPUT
                        int* csrRowIndeces_DofLoops, 
                        int* csrColumnOffsets_DofLoops)
      {
        register double Rpos[numDOFs], Rneg[numDOFs];
        int ij=0;

        //////////////////////////////////////////////////////
        // ********** COMPUTE LOW ORDER SOLUTION ********** //
        //////////////////////////////////////////////////////
        if (num_fct_iter == 0)
          { // No FCT for global bounds 
            for (int i=0; i<numDOFs; i++)
              {
                solLim[i] = uLow[i];
              }
          }
        else // do FCT iterations (with global bounds) on low order solution 
          {
            for (int iter=0; iter<num_fct_iter; iter++)
              {
                ij=0;
                for (int i=0; i<numDOFs; i++)
                  {
                    double maxi=1.0, Pposi=0;
                    for (int offset=csrRowIndeces_DofLoops[i];
                         offset<csrRowIndeces_DofLoops[i+1]; offset++)
                      {
                        int j = csrColumnOffsets_DofLoops[offset];
                        // compute Flux correction
                        double Fluxij = FluxMatrix[ij] - limitedFlux[ij];	      
                        Pposi += Fluxij*((Fluxij > 0) ? 1. : 0.);
                        // update ij
                        ij+=1;
                      }
                    // compute Q vectors
                    double mi = ML[i];
                    double solLimi = solLim[i];
                    double Qposi = mi*(maxi-solLimi);
                    // compute R vectors
                    Rpos[i] = ((Pposi==0) ? 1. : fmin(1.0,Qposi/Pposi));
                  }
                ij=0;
                for (int i=0; i<numDOFs; i++)
                  {
                    double ith_Limiter_times_FluxCorrectionMatrix = 0.;
                    double Rposi = Rpos[i];
                    for (int offset=csrRowIndeces_DofLoops[i];
                         offset<csrRowIndeces_DofLoops[i+1]; offset++)
                      {
                        int j = csrColumnOffsets_DofLoops[offset];
                        // Flux Correction
                        double Fluxij = FluxMatrix[ij] - limitedFlux[ij];
                        // compute limiter
                        double Lij = 1.0;
                        Lij = (Fluxij>0 ? Rposi : Rpos[j]);		    
                        // compute limited flux 
                        ith_Limiter_times_FluxCorrectionMatrix += Lij*Fluxij;		    
			
                        // update limited flux
                        limitedFlux[ij] = Lij*Fluxij;
			
                        //update FluxMatrix
                        FluxMatrix[ij] = Fluxij;		   
			
                        //update ij
                        ij+=1;
                      }
                    //update limited solution
                    double mi = ML[i];
                    solLim[i] += 1.0/mi*ith_Limiter_times_FluxCorrectionMatrix;	    
                  }
              }
          }

        // ***************************************** //
        // ********** HIGH ORDER SOLUTION ********** //
        // ***************************************** //
        ij=0;
        for (int i=0; i<numDOFs; i++)
          {
            double mini=soln[i], maxi=soln[i];
            double Pposi = 0, Pnegi = 0.;
            for (int offset=csrRowIndeces_DofLoops[i];
                 offset<csrRowIndeces_DofLoops[i+1]; offset++)
              {
                int j = csrColumnOffsets_DofLoops[offset];
                // compute local bounds //
                mini = fmin(mini,soln[j]);
                maxi = fmax(maxi,soln[j]);
                // compute P vectors //
                double fij = dt*(MC[ij]*(uDotLow[i]-uDotLow[j]) + dLow[ij]*(uLow[i]-uLow[j]));
                Pposi += fij * (fij > 0 ? 1. : 0.);
                Pnegi += fij * (fij < 0 ? 1. : 0.);
                //update ij
                ij+=1;
              }
            // compute Q vectors //
            double mi = ML[i];
            double Qposi = mi*(maxi-solLim[i]);
            double Qnegi = mi*(mini-solLim[i]);
            // compute R vectors //
            Rpos[i] = ((Pposi==0) ? 1. : fmin(1.0,Qposi/Pposi));
            Rneg[i] = ((Pnegi==0) ? 1. : fmin(1.0,Qnegi/Pnegi));	    	
          }

        // COMPUTE LIMITERS //
        ij=0;
        for (int i=0; i<numDOFs; i++)
          {
            double ith_limited_flux_correction = 0;
            double Rposi = Rpos[i];
            double Rnegi = Rneg[i];
            for (int offset=csrRowIndeces_DofLoops[i]; offset<csrRowIndeces_DofLoops[i+1]; offset++)
              {
                int j = csrColumnOffsets_DofLoops[offset];
                // compute flux correction
                double fij = dt*(MC[ij]*(uDotLow[i]-uDotLow[j]) + dLow[ij]*(uLow[i]-uLow[j]));
                // compute limiters
                double Lij = 1.0;
                Lij = fij > 0 ? fmin(Rposi,Rneg[j]) : fmin(Rnegi,Rpos[j]);
                // compute ith_limited_flux_correction
                ith_limited_flux_correction += Lij*fij;
                ij+=1;
              }
            double mi = ML[i];
            solLim[i] += 1./mi*ith_limited_flux_correction;
          }
      }

      void calculateResidual_entropy_viscosity(//element
                                               double dt,
                                               double* mesh_trial_ref,
                                               double* mesh_grad_trial_ref,
                                               double* mesh_dof,
                                               double* mesh_velocity_dof,
                                               double MOVING_DOMAIN,
                                               int* mesh_l2g,
                                               double* dV_ref,
                                               double* u_trial_ref,
                                               double* u_grad_trial_ref,
                                               double* u_test_ref,
                                               double* u_grad_test_ref,
                                               //element boundary
                                               double* mesh_trial_trace_ref,
                                               double* mesh_grad_trial_trace_ref,
                                               double* dS_ref,
                                               double* u_trial_trace_ref,
                                               double* u_grad_trial_trace_ref,
                                               double* u_test_trace_ref,
                                               double* u_grad_test_trace_ref,
                                               double* normal_ref,
                                               double* boundaryJac_ref,
                                               //physics
                                               int nElements_global,
                                               double* ebqe_penalty_ext,
                                               int* elementMaterialTypes,	
                                               int* isSeepageFace,
                                               int* a_rowptr,
                                               int* a_colind,
                                               double rho,
                                               double beta,
                                               double* gravity,
                                               double* alpha,
                                               double* n,
                                               double* thetaR,
                                               double* thetaSR,
                                               double* KWs,
                                               double useMetrics,
                                               double alphaBDF,
                                               int lag_shockCapturing,
                                               double shockCapturingDiffusion,
                                               double sc_uref,
                                               double sc_alpha,
                                               int* u_l2g,
                                               int* r_l2g,
                                               double* elementDiameter,
                                               int degree_polynomial,
                                               double* u_dof,
                                               double* u_dof_old,	
                                               double* velocity,
                                               double* q_m,
                                               double* q_u,
                                               double* q_m_betaBDF,
                                               double* cfl,
                                               double* edge_based_cfl,
                                               double* q_numDiff_u, 
                                               double* q_numDiff_u_last, 
                                               int offset_u, int stride_u, 
                                               double* globalResidual,
                                               int nExteriorElementBoundaries_global,
                                               int* exteriorElementBoundariesArray,
                                               int* elementBoundaryElementsArray,
                                               int* elementBoundaryLocalElementBoundariesArray,
                                               double* ebqe_velocity_ext,
                                               int* isDOFBoundary_u,
                                               double* ebqe_bc_u_ext,
                                               int* isFluxBoundary_u,
                                               double* ebqe_bc_flux_u_ext,
                                               double* ebqe_phi,double epsFact,
                                               double* ebqe_u,
                                               double* ebqe_flux,
                                               // PARAMETERS FOR EDGE BASED STABILIZATION
                                               double cE,
                                               double cK,
                                               // PARAMETERS FOR LOG BASED ENTROPY FUNCTION
                                               double uL,
                                               double uR,
                                               // PARAMETERS FOR EDGE VISCOSITY
                                               int numDOFs,
                                               int NNZ,
                                               int* csrRowIndeces_DofLoops,
                                               int* csrColumnOffsets_DofLoops,
                                               int* csrRowIndeces_CellLoops,
                                               int* csrColumnOffsets_CellLoops,
                                               int* csrColumnOffsets_eb_CellLoops,
                                               // C matrices
                                               double* Cx,
                                               double* Cy,
                                               double* Cz,
                                               double* CTx,
                                               double* CTy,
                                               double* CTz,
                                               double* ML,
                                               double* delta_x_ij,
                                               // PARAMETERS FOR 1st or 2nd ORDER MPP METHOD
                                               int LUMPED_MASS_MATRIX,
                                               int STABILIZATION_TYPE,
                                               int ENTROPY_TYPE,
                                               // FOR FCT
                                               double* dLow,
                                               double* fluxMatrix,
                                               double* uDotLow,
                                               double* uLow,
                                               double* dt_times_dH_minus_dL,
                                               double* min_u_bc,
                                               double* max_u_bc,
                                               // AUX QUANTITIES OF INTEREST
                                               double* quantDOFs)
      {
        register double Rpos[numDOFs], Rneg[numDOFs];
        //register double FluxCorrectionMatrix[NNZ];
        // NOTE: This function follows a different (but equivalent) implementation of the smoothness based indicator than NCLS.h
        // Allocate space for the transport matrices
        // This is used for first order KUZMIN'S METHOD
        register double TransportMatrix[NNZ];
        std::valarray<double> u_free_dof_old(numDOFs);
        for(int eN=0;eN<nElements_global;eN++)
          for (int j=0;j<nDOF_trial_element;j++)
            {
              register int eN_nDOF_trial_element = eN*nDOF_trial_element;
              u_free_dof_old[r_l2g[eN_nDOF_trial_element+j]] = u_dof_old[u_l2g[eN_nDOF_trial_element+j]];
            }
        for (int i=0; i<NNZ; i++)
          {
            TransportMatrix[i] = 0.;
          }

        // compute entropy and init global_entropy_residual and boundary_integral
        register double psi[numDOFs], eta[numDOFs], global_entropy_residual[numDOFs], boundary_integral[numDOFs];
        for (int i=0; i<numDOFs; i++)
          {
            // NODAL ENTROPY //
            if (STABILIZATION_TYPE==1) //EV stab
              {
                double porosity_times_solni = 1.0*u_free_dof_old[i];
                eta[i] = ENTROPY_TYPE == 1 ? ENTROPY(porosity_times_solni,uL,uR) : ENTROPY_LOG(porosity_times_solni,uL,uR);
                global_entropy_residual[i]=0.;
              }
            boundary_integral[i]=0.;
          }

        //////////////////////////////////////////////
        // ** LOOP IN CELLS FOR CELL BASED TERMS ** //
        //////////////////////////////////////////////
        // HERE WE COMPUTE:
        //    * Time derivative term. porosity*u_t
        //    * cell based CFL (for reference)
        //    * Entropy residual
        //    * Transport matrices
        for(int eN=0;eN<nElements_global;eN++)
          {
            //declare local storage for local contributions and initialize
            register double
              elementResidual_u[nDOF_test_element],
              element_entropy_residual[nDOF_test_element],Phi[nDOF_trial_element];
            register double  elementTransport[nDOF_test_element][nDOF_trial_element];
            for (int i=0;i<nDOF_test_element;i++)
              {
                Phi[i] = u_dof_old[i];
                for (int I=0;I<nSpace;I++)
                  Phi[i] -= rho*mesh_dof[i*3+I]*gravity[I];
                elementResidual_u[i]=0.0;
                element_entropy_residual[i]=0.0;
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    elementTransport[i][j]=0.0;
                  }
              }
            //loop over quadrature points and compute integrands
            for  (int k=0;k<nQuadraturePoints_element;k++)
              {
                //compute indeces and declare local storage
                register int eN_k = eN*nQuadraturePoints_element+k,
                  eN_k_nSpace = eN_k*nSpace,
                  eN_nDOF_trial_element = eN*nDOF_trial_element;
                register double
                  // for entropy residual
                  aux_entropy_residual=0., DENTROPY_un, DENTROPY_uni,
                  //for mass matrix contributions
                  u=0.0, un=0.0, grad_un[nSpace], porosity_times_velocity[nSpace],
                  u_test_dV[nDOF_trial_element],
                  u_grad_trial[nDOF_trial_element*nSpace],
                  u_grad_test_dV[nDOF_test_element*nSpace],
                  //for general use
                  jac[nSpace*nSpace], jacDet, jacInv[nSpace*nSpace],
                  dV,x,y,z,xt,yt,zt,
                  m,dm,f[nSpace],df[nSpace],a[nnz],da[nnz];
                //get the physical integration weight
                ck.calculateMapping_element(eN,
                                            k,
                                            mesh_dof,
                                            mesh_l2g,
                                            mesh_trial_ref,
                                            mesh_grad_trial_ref,
                                            jac,
                                            jacDet,
                                            jacInv,
                                            x,y,z);
                ck.calculateMappingVelocity_element(eN,
                                                    k,
                                                    mesh_velocity_dof,
                                                    mesh_l2g,
                                                    mesh_trial_ref,
                                                    xt,yt,zt);
                dV = fabs(jacDet)*dV_ref[k];
                //get the solution (of Newton's solver). To compute time derivative term
                ck.valFromDOF(u_dof,&u_l2g[eN_nDOF_trial_element],&u_trial_ref[k*nDOF_trial_element],u);
                //get the solution at quad point at tn and tnm1 for entropy viscosity
                ck.valFromDOF(u_dof_old,&u_l2g[eN_nDOF_trial_element],&u_trial_ref[k*nDOF_trial_element],un);
                //get the solution gradients at tn for entropy viscosity
                ck.gradTrialFromRef(&u_grad_trial_ref[k*nDOF_trial_element*nSpace],jacInv,u_grad_trial);
                //precalculate test function products with integration weights for mass matrix terms
                for (int I=0;I<nSpace;I++)
                  grad_un[I] =0.0;
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    u_test_dV[j] = u_test_ref[k*nDOF_trial_element+j]*dV;
                    for (int I=0;I<nSpace;I++)
                      {
                        grad_un[I] += Phi[j]*u_grad_trial[j*nSpace+I];
                        u_grad_test_dV[j*nSpace+I] = u_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin
                      }
                  }
                
                //
                //calculate pde coefficients at quadrature points
                //
                double Kr, dKr;
                evaluateCoefficients(a_rowptr,
                                     a_colind,
                                     rho,
                                     beta,
                                     gravity,
                                     alpha[elementMaterialTypes[eN]],
                                     n[elementMaterialTypes[eN]],
                                     thetaR[elementMaterialTypes[eN]],
                                     thetaSR[elementMaterialTypes[eN]],
                                     &KWs[elementMaterialTypes[eN]*nnz],			      
                                     un,
                                     m,
                                     dm,
                                     f,
                                     df,
                                     a,
                                     da,
                                     Kr,
                                     dKr);
                //
                //moving mesh
                //
                double mesh_velocity[3];
                mesh_velocity[0] = xt;
                mesh_velocity[1] = yt;
                mesh_velocity[2] = zt;
                //relative velocity at tn
                for (int I=0;I<nSpace;I++)
                  {
                    f[I] -= MOVING_DOMAIN*m*mesh_velocity[I];
                    porosity_times_velocity[I] = df[I]*(2.0*dm*dm/(dm*dm + fmax(1.0e-16,dm*dm)));
                  }
                //////////////////////////////
                // CALCULATE CELL BASED CFL //
                //////////////////////////////
                calculateCFL(elementDiameter[eN]/degree_polynomial,porosity_times_velocity,cfl[eN_k]);

                //////////////////////////////////////////////
                // CALCULATE ENTROPY RESIDUAL AT QUAD POINT //
                //////////////////////////////////////////////
                if (STABILIZATION_TYPE==1) // EV stab
                  {
                    for (int I=0;I<nSpace;I++)
                      aux_entropy_residual += porosity_times_velocity[I]*grad_un[I];
                    DENTROPY_un = ENTROPY_TYPE==1 ? DENTROPY(un,uL,uR) : DENTROPY_LOG(un,uL,uR);
                  }
                //////////////
                // ith-LOOP //
                //////////////
                for(int i=0;i<nDOF_test_element;i++)
                  {
                    // VECTOR OF ENTROPY RESIDUAL //
                    int eN_i=eN*nDOF_test_element+i;
                    if (STABILIZATION_TYPE==1) // EV stab
                      {
                        int gi = offset_u+stride_u*u_l2g[eN_i]; //global i-th index
                        double porosity_times_uni = 1.0*u_dof_old[gi];
                        DENTROPY_uni = ENTROPY_TYPE == 1 ? DENTROPY(porosity_times_uni,uL,uR) : DENTROPY_LOG(porosity_times_uni,uL,uR);
                        element_entropy_residual[i] += (DENTROPY_un - DENTROPY_uni)*aux_entropy_residual*u_test_dV[i];
                      }
                    //cek todo, fix elemnetResidual
                    elementResidual_u[i] += (u-un)*u_test_dV[i];
                    ///////////////
                    // j-th LOOP // To construct transport matrices
                    ///////////////
                    for(int j=0;j<nDOF_trial_element;j++)
                      {
                        int j_nSpace = j*nSpace;
                        int i_nSpace = i*nSpace;
                        elementTransport[i][j] += ck.SimpleDiffusionJacobian_weak(a_rowptr,
                                                                                  a_colind,
                                                                                  a,
                                                                                  &u_grad_trial[j_nSpace],
                                                                                  &u_grad_test_dV[i_nSpace]);
                      }
                  }//i
                //save solution for other models
                q_u[eN_k] = u;
                q_m[eN_k] = m;
              }
            /////////////////
            // DISTRIBUTE // load cell based element into global residual
            ////////////////
            for(int i=0;i<nDOF_test_element;i++)
              {
                int eN_i=eN*nDOF_test_element+i;
                int gi = offset_u+stride_u*r_l2g[eN_i]; //global i-th index

                // distribute global residual for (lumped) mass matrix
                globalResidual[gi] += elementResidual_u[i];
                // distribute entropy_residual
                if (STABILIZATION_TYPE==1) // EV Stab
                  global_entropy_residual[gi] += element_entropy_residual[i];

                // distribute transport matrices
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    int eN_i_j = eN_i*nDOF_trial_element+j;
                    TransportMatrix[csrRowIndeces_CellLoops[eN_i] + csrColumnOffsets_CellLoops[eN_i_j]]
                      += elementTransport[i][j];
                  }//j
              }//i
          }//elements

        /* ////////////////////////////////////////////////////////////////////////////////////////// */
        /* // ADD OUTFLOW BOUNDARY TERM TO TRANSPORT MATRICES AND COMPUTE INFLOW BOUNDARY INTEGRAL // */
        /* ////////////////////////////////////////////////////////////////////////////////////////// */
        /* //   * Compute outflow boundary integral as a matrix; i.e., int_B[ (vel.normal)*wi*wj*dx] */
        /* for (int ebNE = 0; ebNE < nExteriorElementBoundaries_global; ebNE++) */
        /*   { */
        /*     double min_u_bc_local = 1E10, max_u_bc_local = -1E10; */
        /*     register int ebN = exteriorElementBoundariesArray[ebNE]; */
        /*     register int eN  = elementBoundaryElementsArray[ebN*2+0], */
        /*       ebN_local = elementBoundaryLocalElementBoundariesArray[ebN*2+0], */
        /*       eN_nDOF_trial_element = eN*nDOF_trial_element; */
        /*     register double elementResidual_u[nDOF_test_element]; */
        /*     for (int i=0;i<nDOF_test_element;i++) */
        /*       elementResidual_u[i]=0.0; */
        /*     // loop on quad points */
        /*     for  (int kb=0;kb<nQuadraturePoints_elementBoundary;kb++) */
        /*       { */
        /*         register int ebNE_kb = ebNE*nQuadraturePoints_elementBoundary+kb, */
        /*           ebNE_kb_nSpace = ebNE_kb*nSpace, */
        /*           ebN_local_kb = ebN_local*nQuadraturePoints_elementBoundary+kb, */
        /*           ebN_local_kb_nSpace = ebN_local_kb*nSpace; */
        /*         register double */
        /*           u_ext=0.0, bc_u_ext=0.0, */
        /*           porosity_times_velocity[nSpace], */
        /*           flux_ext=0.0, dflux_ext=0.0, */
        /*           fluxTransport[nDOF_trial_element], */
        /*           jac_ext[nSpace*nSpace], */
        /*           jacDet_ext, */
        /*           jacInv_ext[nSpace*nSpace], */
        /*           boundaryJac[nSpace*(nSpace-1)], */
        /*           metricTensor[(nSpace-1)*(nSpace-1)], */
        /*           metricTensorDetSqrt, */
        /*           dS, */
        /*           u_test_dS[nDOF_test_element], */
        /*           normal[nSpace],x_ext,y_ext,z_ext,xt_ext,yt_ext,zt_ext,integralScaling,porosity_ext; */
        /*         // calculate mappings */
        /*         ck.calculateMapping_elementBoundary(eN, */
        /*                                             ebN_local, */
        /*                                             kb, */
        /*                                             ebN_local_kb, */
        /*                                             mesh_dof, */
        /*                                             mesh_l2g, */
        /*                                             mesh_trial_trace_ref, */
        /*                                             mesh_grad_trial_trace_ref, */
        /*                                             boundaryJac_ref, */
        /*                                             jac_ext, */
        /*                                             jacDet_ext, */
        /*                                             jacInv_ext, */
        /*                                             boundaryJac, */
        /*                                             metricTensor, */
        /*                                             metricTensorDetSqrt, */
        /*                                             normal_ref, */
        /*                                             normal, */
        /*                                             x_ext,y_ext,z_ext); */
        /*         ck.calculateMappingVelocity_elementBoundary(eN, */
        /*                                                     ebN_local, */
        /*                                                     kb, */
        /*                                                     ebN_local_kb, */
        /*                                                     mesh_velocity_dof, */
        /*                                                     mesh_l2g, */
        /*                                                     mesh_trial_trace_ref, */
        /*                                                     xt_ext,yt_ext,zt_ext, */
        /*                                                     normal, */
        /*                                                     boundaryJac, */
        /*                                                     metricTensor, */
        /*                                                     integralScaling); */
        /*         dS = ((1.0-MOVING_DOMAIN)*metricTensorDetSqrt + MOVING_DOMAIN*integralScaling)*dS_ref[kb]; */
        /*         //compute shape and solution information */
        /*         ck.valFromDOF(u_dof,&u_l2g[eN_nDOF_trial_element],&u_trial_trace_ref[ebN_local_kb*nDOF_test_element],u_ext); */
        /*         //precalculate test function products with integration weights */
        /*         for (int j=0;j<nDOF_trial_element;j++) */
        /*           u_test_dS[j] = u_test_trace_ref[ebN_local_kb*nDOF_test_element+j]*dS; */

        /*         //VRANS */
        /*         porosity_ext = 1.0; */
        /*         // */
        /*         //moving mesh */
        /*         // */
        /*         double mesh_velocity[3]; */
        /*         mesh_velocity[0] = xt_ext; */
        /*         mesh_velocity[1] = yt_ext; */
        /*         mesh_velocity[2] = zt_ext; */
        /*         //std::cout<<"mesh_velocity ext"<<std::endl; */
        /*         for (int I=0;I<nSpace;I++) */
        /*           porosity_times_velocity[I] = porosity_ext*(ebqe_velocity_ext[ebNE_kb_nSpace+I] - MOVING_DOMAIN*mesh_velocity[I]); */
        /*         // */
        /*         //calculate the fluxes */
        /*         // */
        /*         double flow = 0.; */
        /*         for (int I=0; I < nSpace; I++) */
        /*           flow += normal[I]*porosity_times_velocity[I]; */
        /*         //cek todo, fix boundary conditions */
        /*         if (flow >= 0 && isFluxBoundary_u[ebNE_kb] != 1 )  //outflow. This is handled via the transport matrices. Then flux_ext=0 and dflux_ext!=0 */
        /*           { */
        /*             dflux_ext = flow; */
        /*             flux_ext = 0; */
        /*             // save external u */
        /*             ebqe_u[ebNE_kb] = u_ext; */
        /*           } */
        /*         else // inflow. This is handled via the boundary integral. Then flux_ext!=0 and dflux_ext=0 */
        /*           { */
        /*             dflux_ext = 0; */
        /*             // save external u */
        /*             ebqe_u[ebNE_kb] = isDOFBoundary_u[ebNE_kb]*ebqe_bc_u_ext[ebNE_kb]+(1-isDOFBoundary_u[ebNE_kb])*u_ext; */
        /*             if (isDOFBoundary_u[ebNE_kb] == 1) */
        /*               flux_ext = ebqe_bc_u_ext[ebNE_kb]*flow; */
        /*             else if (isFluxBoundary_u[ebNE_kb] == 1) */
        /*               flux_ext = ebqe_bc_flux_u_ext[ebNE_kb]; */
        /*             else */
        /*               { */
        /*                 std::cout<<"warning: VOF open boundary with no external trace, setting to zero for inflow"<<std::endl; */
        /*                 flux_ext = 0.0; */
        /*               } */
        /*           } */

        /*         for (int j=0;j<nDOF_trial_element;j++) */
        /*           { */
        /*             // elementResidual. This is to include the inflow boundary integral. */
        /*             // NOTE: here I assume that we use a Galerkin approach st nDOF_test_element = nDOF_trial_element */
        /*             elementResidual_u[j] += flux_ext*u_test_dS[j]; */
        /*             register int ebN_local_kb_j=ebN_local_kb*nDOF_trial_element+j; */
        /*             fluxTransport[j] = dflux_ext*u_trial_trace_ref[ebN_local_kb_j]; */
        /*           }//j */
        /*         /////////////////////////////////////////////////////// */
        /*         // DISTRIBUTE OUTFLOW BOUNDARY TO TRANSPORT MATRICES // */
        /*         /////////////////////////////////////////////////////// */
        /*         for (int i=0;i<nDOF_test_element;i++) */
        /*           { */
        /*             register int eN_i = eN*nDOF_test_element+i; */
        /*             for (int j=0;j<nDOF_trial_element;j++) */
        /*               { */
        /*                 register int ebN_i_j = ebN*4*nDOF_test_X_trial_element + i*nDOF_trial_element + j; */
        /*                 TransportMatrix[csrRowIndeces_CellLoops[eN_i] + csrColumnOffsets_eb_CellLoops[ebN_i_j]] */
        /*                   += fluxTransport[j]*u_test_dS[i]; */
        /*                 TransposeTransportMatrix[csrRowIndeces_CellLoops[eN_i] + csrColumnOffsets_eb_CellLoops[ebN_i_j]] */
        /*                   += fluxTransport[i]*u_test_dS[j]; */
        /*               }//j */
        /*           }//i */
        /*         // local min/max at boundary */
        /*         min_u_bc_local = fmin(ebqe_u[ebNE_kb], min_u_bc_local); */
        /*         max_u_bc_local = fmax(ebqe_u[ebNE_kb], max_u_bc_local); */
        /*       }//kb */
        /*     // global min/max at boundary */
        /*     for (int i=0;i<nDOF_test_element;i++) */
        /*       { */
        /*         int eN_i = eN*nDOF_test_element+i; */
        /*         int gi = offset_u+stride_u*r_l2g[eN_i]; //global i-th index */
        /*         globalResidual[gi] += dt*elementResidual_u[i]; */
        /*         boundary_integral[gi] += elementResidual_u[i]; */
        /*         min_u_bc[gi] = fmin(min_u_bc_local,min_u_bc[gi]); */
        /*         max_u_bc[gi] = fmax(max_u_bc_local,max_u_bc[gi]); */
        /*       } */
        /*   }//ebNE */
        // END OF ADDING BOUNDARY TERM TO TRANSPORT MATRICES and COMPUTING BOUNDARY INTEGRAL //

        /////////////////////////////////////////////////////////////////
        // COMPUTE SMOOTHNESS INDICATOR and NORMALIZE ENTROPY RESIDUAL //
        /////////////////////////////////////////////////////////////////
        // NOTE: see NCLS.h for a different but equivalent implementation of this.
        int ij = 0;
        for (int i=0; i<numDOFs; i++)
          {
            double gi[nSpace], Cij[nSpace], xi[nSpace], etaMaxi, etaMini;
            if (STABILIZATION_TYPE==1) //EV Stabilization
              {
                // For eta min and max
                etaMaxi = fabs(eta[i]);
                etaMini = fabs(eta[i]);
              }
            double porosity_times_solni = 1.0*u_free_dof_old[i];
            // initialize gi and compute xi
            for (int I=0; I < nSpace; I++)
              {
                gi[I] = 0.;
                xi[I] = mesh_dof[i*3+I];
              }
            // for smoothness indicator //
            double alpha_numerator_pos = 0., alpha_numerator_neg = 0., alpha_denominator_pos = 0., alpha_denominator_neg = 0.;
            for (int offset=csrRowIndeces_DofLoops[i]; offset<csrRowIndeces_DofLoops[i+1]; offset++)
              { // First loop in j (sparsity pattern)
                int j = csrColumnOffsets_DofLoops[offset];
                if (STABILIZATION_TYPE==1) //EV Stabilization
                  {
                    // COMPUTE ETA MIN AND ETA MAX //
                    etaMaxi = fmax(etaMaxi,fabs(eta[j]));
                    etaMini = fmin(etaMini,fabs(eta[j]));
                  }
                double porosity_times_solnj = 1.0*u_free_dof_old[j];
                // Update Cij matrices
                Cij[0] = Cx[ij];
                Cij[1] = Cy[ij];
#if nSpace == 3
                Cij[2] = Cz[ij];
#endif
                // COMPUTE gi VECTOR. gi=1/mi*sum_j(Cij*solj)
                for (int I=0; I < nSpace; I++)
                  gi[I] += Cij[I]*porosity_times_solnj;

                // COMPUTE numerator and denominator of smoothness indicator
                double alpha_num = porosity_times_solni - porosity_times_solnj;
                if (alpha_num >= 0.)
                  {
                    alpha_numerator_pos += alpha_num;
                    alpha_denominator_pos += alpha_num;
                  }
                else
                  {
                    alpha_numerator_neg += alpha_num;
                    alpha_denominator_neg += fabs(alpha_num);
                  }
                //update ij
                ij+=1;
              }
            // scale g vector by lumped mass matrix
            for (int I=0; I < nSpace; I++)
              gi[I] /= ML[i];
            if (STABILIZATION_TYPE==1) //EV Stab
              {
                // Normalizae entropy residual
                global_entropy_residual[i] *= etaMini == etaMaxi ? 0. : 2*cE/(etaMaxi-etaMini);
                quantDOFs[i] = fabs(global_entropy_residual[i]);
              }

            // Now that I have the gi vectors, I can use them for the current i-th DOF
            double SumPos=0., SumNeg=0.;
            for (int offset=csrRowIndeces_DofLoops[i]; offset<csrRowIndeces_DofLoops[i+1]; offset++)
              { // second loop in j (sparsity pattern)
                int j = csrColumnOffsets_DofLoops[offset];
                // compute xj
                double xj[nSpace];
                for (int I=0; I < nSpace; I++)
                  xj[I] = mesh_dof[j*3+I];
                // compute gi*(xi-xj)
                double gi_times_x=0.;
                for (int I=0; I < nSpace; I++)
                  {
                    //if (delta_x_ij[offset*3+I] > 0.0)
                    //  assert( (xi[I] - xj[I]) == delta_x_ij[offset*3+I]);
                    //gi_times_x += gi[I]*(xi[I]-xj[I]);
                    gi_times_x += gi[I]*delta_x_ij[offset*3+I];
                  }
                // compute the positive and negative part of gi*(xi-xj)
                SumPos += gi_times_x > 0 ? gi_times_x : 0;
                SumNeg += gi_times_x < 0 ? gi_times_x : 0;
              }
            double sigmaPosi = fmin(1.,(fabs(SumNeg)+1E-15)/(SumPos+1E-15));
            double sigmaNegi = fmin(1.,(SumPos+1E-15)/(fabs(SumNeg)+1E-15));
            double alpha_numi = fabs(sigmaPosi*alpha_numerator_pos + sigmaNegi*alpha_numerator_neg);
            double alpha_deni = sigmaPosi*alpha_denominator_pos + sigmaNegi*alpha_denominator_neg;
            if (IS_BETAij_ONE == 1)
              {
                alpha_numi = fabs(alpha_numerator_pos + alpha_numerator_neg);
                alpha_deni = alpha_denominator_pos + alpha_denominator_neg;
              }
            double alphai = alpha_numi/(alpha_deni+1E-15);
            quantDOFs[i] = alphai;

            if (POWER_SMOOTHNESS_INDICATOR==0)
              psi[i] = 1.0;
            else
              psi[i] = std::pow(alphai,POWER_SMOOTHNESS_INDICATOR); //NOTE: they use alpha^2 in the paper
          }
        /////////////////////////////////////////////
        // ** LOOP IN DOFs FOR EDGE BASED TERMS ** //
        /////////////////////////////////////////////
        ij=0;
        for (int i=0; i<numDOFs; i++)
          {
            // NOTE: Transport matrices already have the porosity considered. ---> Dissipation matrices as well.
            double solni = u_free_dof_old[i]; // solution at time tn for the ith DOF
            for (int I=0;I<nSpace;I++)
              solni -= rho*gravity[I]*mesh_dof[i*3+I];
            double porosityi = 1.0;
            double ith_dissipative_term = 0;
            double ith_low_order_dissipative_term = 0;
            double ith_flux_term = 0;
            double dLii = 0.;
            double m,dm,f[nSpace],df[nSpace],a[nnz],da[nnz];

            // loop over the sparsity pattern of the i-th DOF
            double Kr, dKr;
            for (int offset=csrRowIndeces_DofLoops[i]; offset<csrRowIndeces_DofLoops[i+1]; offset++)
              {
                int j = csrColumnOffsets_DofLoops[offset];
                double solnj = u_free_dof_old[j]; // solution at time tn for the jth DOF
                for (int I=0;I<nSpace;I++)
                  solnj -= rho*gravity[I]*mesh_dof[j*3+I];
                double porosityj = 1.0;
                double dLowij, dLij, dEVij, dHij;
                if (-TransportMatrix[ij]*(solnj - solni) <= 0.0)
                  {
                    evaluateCoefficients(a_rowptr,
                                         a_colind,
                                         rho,
                                         beta,
                                         gravity,
                                         alpha[elementMaterialTypes[0]],//cek hack, only for 1 material
                                         n[elementMaterialTypes[0]],
                                         thetaR[elementMaterialTypes[0]],
                                         thetaSR[elementMaterialTypes[0]],
                                         &KWs[elementMaterialTypes[0]*nnz],			      
                                         u_free_dof_old[i],
                                         m,
                                         dm,
                                         f,
                                         df,
                                         a,
                                         da,
                                         Kr,
                                         dKr);
                    ith_flux_term += Kr*fmax(0.0, -TransportMatrix[ij])*(solnj - solni);
                  }
                else
                  {
                    evaluateCoefficients(a_rowptr,
                                         a_colind,
                                         rho,
                                         beta,
                                         gravity,
                                         alpha[elementMaterialTypes[0]],//cek hack, only for 1 material
                                         n[elementMaterialTypes[0]],
                                         thetaR[elementMaterialTypes[0]],
                                         thetaSR[elementMaterialTypes[0]],
                                         &KWs[elementMaterialTypes[0]*nnz],			      
                                         u_free_dof_old[j],
                                         m,
                                         dm,
                                         f,
                                         df,
                                         a,
                                         da,
                                         Kr,
                                         dKr);
                    ith_flux_term += Kr*fmax(0.0, -TransportMatrix[ij])*(solnj - solni);
                  }
                //std::cout<<"i "<<i<<" j "<<j<<" grad ij "<<(solnj - solni)<<" flux ij "<<fmax(0.0, -TransportMatrix[ij])*(solnj - solni)<<std::endl;
                /* if (i != j) //NOTE: there is really no need to check for i!=j (see formula for ith_dissipative_term) */
                /*   { */
                /*     // artificial compression */
                /*     double solij = 0.5*(porosityi*solni+porosityj*solnj); */
                /*     double Compij = cK*fmax(solij*(1.0-solij),0.0)/(fabs(porosityi*solni-porosityj*solnj)+1E-14); */
                /*     // first-order dissipative operator */
                /*     dLowij = fmax(fabs(TransportMatrix[ij]),fabs(TransportMatrix[ij]));//symmetric */
                /*     //dLij = fmax(0.,fmax(psi[i]*TransportMatrix[ij], // Approach by S. Badia */
                /*     //              psi[j]*TransposeTransportMatrix[ij])); */
                /*     dLij = dLowij*fmax(psi[i],psi[j]); // enhance the order to 2nd order. No EV */
                /*     if (STABILIZATION_TYPE==1) //EV Stab */
                /*       { */
                /*         // high-order (entropy viscosity) dissipative operator */
                /*         dEVij = fmax(fabs(global_entropy_residual[i]), */
                /*                      fabs(global_entropy_residual[j])); */
                /*         dHij = fmin(dLowij,dEVij) * fmax(1.0-Compij,0.0); // artificial compression */
                /*       } */
                /*     else // smoothness based indicator */
                /*       { */
                /*         dHij = dLij * fmax(1.0-Compij,0.0); // artificial compression */
                /*         //dHij = dLowij; */
                /*         //dLij = dLowij; */

                /*         dEVij = fmax(fabs(global_entropy_residual[i]), */
                /*                      fabs(global_entropy_residual[j])); */
                /*         //dHij = fmin(dLowij,dEVij);//\*fmax(1.0-Compij,0.0); // artificial compression */
                /*         //dHij = 0.; */
                /*         dHij = dLowij; */
                /*         //  */
                /*       } */
                /*     dHij = dLowij; */
                /*     dLij = dLowij; */
                /*     dLow[ij]=dLowij; */
		    
                /*     //dissipative terms */
                /*     ith_dissipative_term += dHij*(solnj-solni); */
                /*     ith_low_order_dissipative_term += dLij*(solnj-solni); */
                /*     //dHij - dLij. This matrix is needed during FCT step */
                /*     dt_times_dH_minus_dL[ij] = dt*(dHij - dLij); */
                /*     dLii -= dLij; */
		    
                /*     fluxMatrix[ij] = -dt*(TransportMatrix[ij]*solnj */
                /*                           -TransposeTransportMatrix[ij]*solni */
                /*                           -dHij*(solnj-solni)); */
                /*   } */
                /* else //i==j */
                /*   { */
                /*     dt_times_dH_minus_dL[ij] = 0; */
                /*     fluxMatrix[ij] = 0;//TransportMatrix[ij]*solnj; */
                /*     dLow[ij]=0.; // not true but works since *= (ui-uj) */
                /*   } */
                //update ij
                ij+=1;
              }
            double mi = ML[i];
            // compute edge_based_cfl
            //edge_based_cfl[i] = 2.*fabs(dLii)/mi;


            uDotLow[i] = 1.0/mi*ith_flux_term;
            //+ boundary_integral[i]);
            //- ith_low_order_dissipative_term);
            evaluateCoefficients(a_rowptr,
                                 a_colind,
                                 rho,
                                 beta,
                                 gravity,
                                 alpha[elementMaterialTypes[0]],//cek hack, only for 1 material
                                 n[elementMaterialTypes[0]],
                                 thetaR[elementMaterialTypes[0]],
                                 thetaSR[elementMaterialTypes[0]],
                                 &KWs[elementMaterialTypes[0]*nnz],			      
                                 u_free_dof_old[i],
                                 m,
                                 dm,
                                 f,
                                 df,
                                 a,
                                 da,
                                 Kr,
                                 dKr);
            uLow[i] = m + dt*uDotLow[i];//cek should introduce mn,mnp1 or somethign clearer
            //cek debug
            //std::cout<<"dt*divergence "<<dt*uDotLow[i]<<std::endl;
            //std::cout<<"mass density old "<<m<<std::endl;
            m = uLow[i];
            //std::cout<<"mass density "<<m<<std::endl;
            evaluateInverseCoefficients(a_rowptr,
                                        a_colind,
                                        rho,
                                        beta,
                                        gravity,
                                        alpha[elementMaterialTypes[0]],//cek hack, only for 1 material
                                        n[elementMaterialTypes[0]],
                                        thetaR[elementMaterialTypes[0]],
                                        thetaSR[elementMaterialTypes[0]],
                                        &KWs[elementMaterialTypes[0]*nnz],			      
                                        uLow[i],
                                        m,
                                        dm,
                                        f,
                                        df,
                                        a,
                                        da);
            globalResidual[i] = uLow[i];
            //uLow[i] = u_dof_old[i] - dt/mi*(ith_flux_term
            //						  + boundary_integral[i]
            //						  - ith_low_order_dissipative_term);
            // update residual
            //if (LUMPED_MASS_MATRIX==1)
            //globalResidual[i] = u_dof_old[i] - dt/mi*(ith_flux_term
            //					      + boundary_integral[i]
            //					      - ith_dissipative_term);
            //else
            //globalResidual[i] += dt*(ith_flux_term - ith_dissipative_term);
          }
	
        //ij=0;
        //for (int i=0; i<numDOFs; i++)
        //{
        //  double mini=0.0;
        //  double maxi=1.0;
        //  double Pposi=0;
        //  double Pnegi=0;
        //  for (int offset=csrRowIndeces_DofLoops[i];
        //	 offset<csrRowIndeces_DofLoops[i+1]; offset++)
        //  {
        //    int j = csrColumnOffsets_DofLoops[offset];
        //    Pposi += FluxCorrectionMatrix[ij]*((FluxCorrectionMatrix[ij] > 0) ? 1. : 0.);
        //    Pnegi += FluxCorrectionMatrix[ij]*((FluxCorrectionMatrix[ij] < 0) ? 1. : 0.);
        //    ij+=1;
        //  }
        //  double mi = ML[i];
        //  double solni = u_dof_old[i];
        //  double Qposi = mi*(maxi-solni);
        //  double Qnegi = mi*(mini-solni);

        //std::cout << Qposi << std::endl;
        //  Rpos[i] = ((Pposi==0) ? 1. : fmin(1.0,Qposi/Pposi));
        //  Rneg[i] = ((Pnegi==0) ? 1. : fmin(1.0,Qnegi/Pnegi));

        //if (Rpos[i] < 0)
        //{
        //	std::cout << mi << "\t" << maxi << "\t" << solni << std::endl;
        //	std::cout << Qposi << "\t" << Pposi << std::endl;
        //	std::cout << Rpos[i] << std::endl;
        //abort();
        //}
        //}
	
        //ij=0;
        //for (int i=0; i<numDOFs; i++)
        //{
        //  double ith_Limiter_times_FluxCorrectionMatrix = 0.;
        //  double Rposi = Rpos[i];
        //  double Rnegi = Rneg[i];

        //if (Rposi > 1.0 || Rposi < 0.0)	      
        //std::cout << "Rposi: " << Rposi << std::endl;
        //if (Rnegi > 1.0 || Rnegi < 0.0)	      
        //std::cout << "Rnegi: " << Rnegi << std::endl;	    
	    
        // LOOP OVER THE SPARSITY PATTERN (j-LOOP)//
        //  for (int offset=csrRowIndeces_DofLoops[i];
        //	 offset<csrRowIndeces_DofLoops[i+1]; offset++)
        //    {
        //	int j = csrColumnOffsets_DofLoops[offset];
        //	double Lij = 1.0;
        //	Lij = ((FluxCorrectionMatrix[ij]>0) ? fmin(Rposi,Rneg[j]) : fmin(Rnegi,Rpos[j]));
        //	//Lij=0.0;
        //	ith_Limiter_times_FluxCorrectionMatrix += Lij*FluxCorrectionMatrix[ij];
        //	//update ij
        //	ij+=1;
        //    }
        //  double mi = ML[i];
        //  double solni = u_dof_old[i];    
        //  globalResidual[i] = solni + 1.0/mi*ith_Limiter_times_FluxCorrectionMatrix;
        //}
	
      }

      void calculateMassMatrix(//element
                               double dt,
                               double* mesh_trial_ref,
                               double* mesh_grad_trial_ref,
                               double* mesh_dof,
                               double* mesh_velocity_dof,
                               double MOVING_DOMAIN,
                               int* mesh_l2g,
                               double* dV_ref,
                               double* u_trial_ref,
                               double* u_grad_trial_ref,
                               double* u_test_ref,
                               double* u_grad_test_ref,
                               //element boundary
                               double* mesh_trial_trace_ref,
                               double* mesh_grad_trial_trace_ref,
                               double* dS_ref,
                               double* u_trial_trace_ref,
                               double* u_grad_trial_trace_ref,
                               double* u_test_trace_ref,
                               double* u_grad_test_trace_ref,
                               double* normal_ref,
                               double* boundaryJac_ref,
                               //physics
                               int nElements_global,
                               //new
                               double* ebqe_penalty_ext,
                               int* elementMaterialTypes,	
                               int* isSeepageFace,
                               int* a_rowptr,
                               int* a_colind,
                               double rho,
                               double beta,
                               double* gravity,
                               double* alpha,
                               double* n,
                               double* thetaR,
                               double* thetaSR,
                               double* KWs,
                               //end new
                               double useMetrics, 
                               double alphaBDF,
                               int lag_shockCapturing,
                               double shockCapturingDiffusion,
                               int* u_l2g,
                               int* r_l2g,
                               double* elementDiameter,
                               int degree_polynomial,
                               double* u_dof, 
                               double* velocity,
                               double* q_m_betaBDF, 
                               double* cfl,
                               double* q_numDiff_u_last, 
                               int* csrRowIndeces_u_u,int* csrColumnOffsets_u_u,
                               double* globalJacobian,
                               double* delta_x_ij,
                               int nExteriorElementBoundaries_global,
                               int* exteriorElementBoundariesArray,
                               int* elementBoundaryElementsArray,
                               int* elementBoundaryLocalElementBoundariesArray,
                               double* ebqe_velocity_ext,
                               int* isDOFBoundary_u,
                               double* ebqe_bc_u_ext,
                               int* isFluxBoundary_u,
                               double* ebqe_bc_flux_u_ext,
                               int* csrColumnOffsets_eb_u_u,
                               int LUMPED_MASS_MATRIX)
      {
        //std::cout<<"ndjaco  address "<<q_numDiff_u_last<<std::endl;
        double Ct_sge = 4.0;
        //
        //loop over elements to compute volume integrals and load them into the element Jacobians and global Jacobian
        //
        for(int eN=0;eN<nElements_global;eN++)
          {
            register double  elementJacobian_u_u[nDOF_test_element][nDOF_trial_element];
            for (int i=0;i<nDOF_test_element;i++)
              for (int j=0;j<nDOF_trial_element;j++)
                {
                  elementJacobian_u_u[i][j]=0.0;
                }
            for  (int k=0;k<nQuadraturePoints_element;k++)
              {
                int eN_k = eN*nQuadraturePoints_element+k, //index to a scalar at a quadrature point
                  eN_k_nSpace = eN_k*nSpace,
                  eN_nDOF_trial_element = eN*nDOF_trial_element; //index to a vector at a quadrature point

                //declare local storage
                register double u=0.0,
                  grad_u[nSpace],
                  m=0.0,dm=0.0,
                  f[nSpace],df[nSpace],
                  a[nnz],da[nnz],
                  m_t=0.0,dm_t=0.0,
                  dpdeResidual_u_u[nDOF_trial_element],
                  Lstar_u[nDOF_test_element],
                  dsubgridError_u_u[nDOF_trial_element],
                  tau=0.0,tau0=0.0,tau1=0.0,
                  jac[nSpace*nSpace],
                  jacDet,
                  jacInv[nSpace*nSpace],
                  u_grad_trial[nDOF_trial_element*nSpace],
                  dV,
                  u_test_dV[nDOF_test_element],
                  u_grad_test_dV[nDOF_test_element*nSpace],
                  x,y,z,xt,yt,zt,
                  //VRANS
                  porosity,
                  //
                  G[nSpace*nSpace],G_dd_G,tr_G;

                //get jacobian, etc for mapping reference element
                ck.calculateMapping_element(eN,
                                            k,
                                            mesh_dof,
                                            mesh_l2g,
                                            mesh_trial_ref,
                                            mesh_grad_trial_ref,
                                            jac,
                                            jacDet,
                                            jacInv,
                                            x,y,z);
                ck.calculateMappingVelocity_element(eN,
                                                    k,
                                                    mesh_velocity_dof,
                                                    mesh_l2g,
                                                    mesh_trial_ref,
                                                    xt,yt,zt);
                //get the physical integration weight
                dV = fabs(jacDet)*dV_ref[k];
                ck.calculateG(jacInv,G,G_dd_G,tr_G);
                //get the trial function gradients
                ck.gradTrialFromRef(&u_grad_trial_ref[k*nDOF_trial_element*nSpace],jacInv,u_grad_trial);
                //get the solution
                ck.valFromDOF(u_dof,&u_l2g[eN_nDOF_trial_element],&u_trial_ref[k*nDOF_trial_element],u);
                //get the solution gradients
                ck.gradFromDOF(u_dof,&u_l2g[eN_nDOF_trial_element],u_grad_trial,grad_u);
                //precalculate test function products with integration weights
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    u_test_dV[j] = u_test_ref[k*nDOF_trial_element+j]*dV;
                    for (int I=0;I<nSpace;I++)
                      {
                        u_grad_test_dV[j*nSpace+I]   = u_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin
                      }
                  }
                //VRANS
                porosity = 1.0;
                //
                //
                //calculate pde coefficients and derivatives at quadrature points
                //
                double Kr,dKr;
                evaluateCoefficients(a_rowptr,
                                     a_colind,
                                     rho,
                                     beta,
                                     gravity,
                                     alpha[elementMaterialTypes[eN]],
                                     n[elementMaterialTypes[eN]],
                                     thetaR[elementMaterialTypes[eN]],
                                     thetaSR[elementMaterialTypes[eN]],
                                     &KWs[elementMaterialTypes[eN]*nnz],			      
                                     u,
                                     m,
                                     dm,
                                     f,
                                     df,
                                     a,
                                     da,
                                     Kr,
                                     dKr);
                //
                //moving mesh
                //
                double mesh_velocity[3];
                mesh_velocity[0] = xt;
                mesh_velocity[1] = yt;
                mesh_velocity[2] = zt;
                //std::cout<<"qj mesh_velocity"<<std::endl;
                for(int I=0;I<nSpace;I++)
                  {
                    //std::cout<<mesh_velocity[I]<<std::endl;
                    f[I] -= MOVING_DOMAIN*m*mesh_velocity[I];
                    df[I] -= MOVING_DOMAIN*dm*mesh_velocity[I];
                  }
                //
                //calculate time derivatives
                //
                ck.bdf(alphaBDF,
                       q_m_betaBDF[eN_k],//since m_t isn't used, we don't have to correct mass
                       m,
                       dm,
                       m_t,
                       dm_t);
                //
                //calculate subgrid error contribution to the Jacobian (strong residual, adjoint, jacobian of strong residual)
                //
                //calculate the adjoint times the test functions
                for (int i=0;i<nDOF_test_element;i++)
                  {
                    // int eN_k_i_nSpace = (eN_k*nDOF_trial_element+i)*nSpace;
                    // Lstar_u[i]=ck.Advection_adjoint(df,&u_grad_test_dV[eN_k_i_nSpace]);
                    register int i_nSpace = i*nSpace;
                    Lstar_u[i]=ck.Advection_adjoint(df,&u_grad_test_dV[i_nSpace]);
                  }
                //calculate the Jacobian of strong residual
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    //int eN_k_j=eN_k*nDOF_trial_element+j;
                    //int eN_k_j_nSpace = eN_k_j*nSpace;
                    int j_nSpace = j*nSpace;
                    dpdeResidual_u_u[j]= ck.MassJacobian_strong(dm_t,u_trial_ref[k*nDOF_trial_element+j]) +
                      ck.AdvectionJacobian_strong(df,&u_grad_trial[j_nSpace]);
                  }
                //tau and tau*Res
                calculateSubgridError_tau(elementDiameter[eN],
                                          dm_t,
                                          df,
                                          cfl[eN_k],
                                          tau0);

                calculateSubgridError_tau(Ct_sge,
                                          G,
                                          dm_t,
                                          df,
                                          tau1,
                                          cfl[eN_k]);
                tau = useMetrics*tau1+(1.0-useMetrics)*tau0;

                for(int j=0;j<nDOF_trial_element;j++)
                  dsubgridError_u_u[j] = -tau*dpdeResidual_u_u[j];
                //double h=elementDiameter[eN];
                for(int i=0;i<nDOF_test_element;i++)
                  {
                    //int eN_k_i=eN_k*nDOF_test_element+i;
                    //int eN_k_i_nSpace=eN_k_i*nSpace;
                    for(int j=0;j<nDOF_trial_element;j++)
                      {
                        if (LUMPED_MASS_MATRIX==1)
                          {
                            if (i==j)
                              elementJacobian_u_u[i][j] += u_test_dV[i];
                          }
                        else
                          {
                            //int eN_k_j=eN_k*nDOF_trial_element+j;
                            //int eN_k_j_nSpace = eN_k_j*nSpace;
                            int j_nSpace = j*nSpace;
                            int i_nSpace = i*nSpace;
                            //std::cout<<"jac "<<'\t'<<q_numDiff_u_last[eN_k]<<'\t'<<dm_t<<'\t'<<df[0]<<df[1]<<'\t'<<dsubgridError_u_u[j]<<std::endl;
                            dm_t = 1.0;//cek, will solving for continuum density explicitly
                            elementJacobian_u_u[i][j] +=
                              dt*ck.MassJacobian_weak(dm_t,u_trial_ref[k*nDOF_trial_element+j],u_test_dV[i]);
                          }
                      }//j
                  }//i
              }//k
            //
            //load into element Jacobian into global Jacobian
            //
            for (int i=0;i<nDOF_test_element;i++)
              {
                int eN_i = eN*nDOF_test_element+i;
                int I = u_l2g[eN_i];
                for (int j=0;j<nDOF_trial_element;j++)
                  {
                    int eN_i_j = eN_i*nDOF_trial_element+j;
                    int J = u_l2g[eN*nDOF_trial_element+j];
                    globalJacobian[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_u_u[eN_i_j]] += elementJacobian_u_u[i][j];
                    delta_x_ij[3*(csrRowIndeces_u_u[eN_i] + csrColumnOffsets_u_u[eN_i_j])+0] = mesh_dof[I*3+0] - mesh_dof[J*3+0];
                    delta_x_ij[3*(csrRowIndeces_u_u[eN_i] + csrColumnOffsets_u_u[eN_i_j])+1] = mesh_dof[I*3+1] - mesh_dof[J*3+1];
                    delta_x_ij[3*(csrRowIndeces_u_u[eN_i] + csrColumnOffsets_u_u[eN_i_j])+2] = mesh_dof[I*3+2] - mesh_dof[J*3+2];
                  }//j
              }//i
          }//elements
      }//computeMassMatrix
    };//Richards

  inline Richards_base* newRichards(int nSpaceIn,
                                    int nQuadraturePoints_elementIn,
                                    int nDOF_mesh_trial_elementIn,
                                    int nDOF_trial_elementIn,
                                    int nDOF_test_elementIn,
                                    int nQuadraturePoints_elementBoundaryIn,
                                    int CompKernelFlag)
  {
    if (nSpaceIn == 1)
      return proteus::chooseAndAllocateDiscretization1D<Richards_base,Richards,CompKernel>(nSpaceIn,
                                                                                           nQuadraturePoints_elementIn,
                                                                                           nDOF_mesh_trial_elementIn,
                                                                                           nDOF_trial_elementIn,
                                                                                           nDOF_test_elementIn,
                                                                                           nQuadraturePoints_elementBoundaryIn,
                                                                                           CompKernelFlag);
    else if (nSpaceIn == 2)
      return proteus::chooseAndAllocateDiscretization2D<Richards_base,Richards,CompKernel>(nSpaceIn,
                                                                                         nQuadraturePoints_elementIn,
                                                                                         nDOF_mesh_trial_elementIn,
                                                                                         nDOF_trial_elementIn,
                                                                                         nDOF_test_elementIn,
                                                                                         nQuadraturePoints_elementBoundaryIn,
                                                                                         CompKernelFlag);
    else
      return proteus::chooseAndAllocateDiscretization<Richards_base,Richards,CompKernel>(nSpaceIn,
                                                                                         nQuadraturePoints_elementIn,
                                                                                         nDOF_mesh_trial_elementIn,
                                                                                         nDOF_trial_elementIn,
                                                                                         nDOF_test_elementIn,
                                                                                         nQuadraturePoints_elementBoundaryIn,
                                                                                         CompKernelFlag);
  }
}//proteus
#endif
