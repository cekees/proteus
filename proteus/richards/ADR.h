#ifndef ADR_H
#define ADR_H
#include <cmath>
#include <iostream>
#include <valarray>
#include "CompKernel.h"
#include "ModelFactory.h"
#include "../mprans/ArgumentsDict.h"
#include "xtensor-python/pyarray.hpp"
#define nnz nSpace

namespace py = pybind11;
#define POWER_SMOOTHNESS_INDICATOR 2
#define IS_BETAij_ONE 0
#define GLOBAL_FCT 0

namespace proteus
{
  class ADR_base
  {
    //The base class defining the interface
  public:
    virtual ~ADR_base(){}
    virtual void calculateResidual(arguments_dict& args)=0;
    virtual void calculateJacobian(arguments_dict& args)=0;
  };

  template<class CompKernelType,
	   int nSpace,
	   int nQuadraturePoints_element,
	   int nDOF_mesh_trial_element,
	   int nDOF_trial_element,
	   int nDOF_test_element,
	   int nQuadraturePoints_elementBoundary>

  class ADR : public ADR_base
  {
  public:
    const int nDOF_test_X_trial_element;
    CompKernelType ck;
    ADR():
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
	 		      double as[nnz],
	 		      double& kr,
	 		      double& dkr)
     {
       const int nSpace2 = nSpace * nSpace;
       double psiC;
       double pcBar;
       double pcBar_n;
       double pcBar_nM1;
       double pcBar_nM2;
       double onePlus_pcBar_n;
       double sBar;
       double sqrt_sBar;
       double DsBar_DpsiC;
       double thetaW;
       double DthetaW_DpsiC;
       double vBar;
       double vBar2;
       double DvBar_DpsiC;
       double KWr;
       double DKWr_DpsiC;
       double rho2 = rho * rho;
       double thetaS;
       double rhom;
       double drhom;
       double m_vg;
       double pcBarStar;
       double sqrt_sBarStar;
       m = u; //rhom*thetaW; //u
       dm = 1.0; //-rhom*DthetaW_DpsiC+drhom*thetaW;// 1.0;//
       for (int I=0;I<nSpace;I++)
	 {
	   f[I] = 0.0;
	   df[I] = 0.0;

	   for (int ii=rowptr[I]; ii < rowptr[I+1]; ii++)
	     {
	 	  double velocity = 5.0;
	 	  double D= 0.02;	
	       f[I]  = velocity*u; //rho2*KWr*KWs[ii]*gravity[colind[ii]];//velocity *u;//
	       df[I] = velocity; //-rho2*DKWr_DpsiC*KWs[ii]*gravity[colind[ii]];//velocity
	       a[ii]  = D ; //rho*KWs[ii]; //rho*KWr*KWs[ii];
	       da[ii] = 0.0; //-rho*DKWr_DpsiC*KWs[ii]; //0.0;//
	       //as[ii]  = 1.0; //rho*KWs[ii];//0.0;//rho*KWs[ii];
	       //kr = 1.0;//KWr;// 0.0
	       //dkr=0.0; //mod picard DKWr_DpsiC;
	     }
	 }
     }
    void calculateResidual(arguments_dict& args)
    {
      xt::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
      xt::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
      xt::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
      xt::pyarray<double>& mesh_velocity_dof = args.array<double>("mesh_velocity_dof");
      double MOVING_DOMAIN = args.scalar<double>("MOVING_DOMAIN");
      xt::pyarray<int>& mesh_l2g = args.array<int>("mesh_l2g");
      xt::pyarray<double>& dV_ref = args.array<double>("dV_ref");
      xt::pyarray<double>& u_trial_ref = args.array<double>("u_trial_ref");
      xt::pyarray<double>& u_grad_trial_ref = args.array<double>("u_grad_trial_ref");
      xt::pyarray<double>& u_test_ref = args.array<double>("u_test_ref");
      xt::pyarray<double>& u_grad_test_ref = args.array<double>("u_grad_test_ref");
      xt::pyarray<double>& mesh_trial_trace_ref = args.array<double>("mesh_trial_trace_ref");
      xt::pyarray<double>& mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
      xt::pyarray<double>& dS_ref = args.array<double>("dS_ref");
      xt::pyarray<double>& u_trial_trace_ref = args.array<double>("u_trial_trace_ref");
      xt::pyarray<double>& u_grad_trial_trace_ref = args.array<double>("u_grad_trial_trace_ref");
      xt::pyarray<double>& u_test_trace_ref = args.array<double>("u_test_trace_ref");
      xt::pyarray<double>& u_grad_test_trace_ref = args.array<double>("u_grad_test_trace_ref");
      xt::pyarray<double>& normal_ref = args.array<double>("normal_ref");
      xt::pyarray<double>& boundaryJac_ref = args.array<double>("boundaryJac_ref");
      int nElements_global = args.scalar<int>("nElements_global");
      xt::pyarray<double>& ebqe_penalty_ext = args.array<double>("ebqe_penalty_ext");
      xt::pyarray<int>& elementMaterialTypes = args.array<int>("elementMaterialTypes");
      xt::pyarray<int>& isSeepageFace = args.array<int>("isSeepageFace");
      xt::pyarray<int>& a_rowptr = args.array<int>("a_rowptr");
      xt::pyarray<int>& a_colind = args.array<int>("a_colind");
      double rho = args.scalar<double>("rho");
      double beta = args.scalar<double>("beta");
      xt::pyarray<double>& gravity = args.array<double>("gravity");
      xt::pyarray<double>& alpha = args.array<double>("alpha");
      xt::pyarray<double>& n = args.array<double>("n");
      xt::pyarray<double>& thetaR = args.array<double>("thetaR");
      xt::pyarray<double>& thetaSR = args.array<double>("thetaSR");
      xt::pyarray<double>& KWs = args.array<double>("KWs");
      double useMetrics = args.scalar<double>("useMetrics");
      double alphaBDF = args.scalar<double>("alphaBDF");
      int lag_shockCapturing = args.scalar<int>("lag_shockCapturing");
      double shockCapturingDiffusion = args.scalar<double>("shockCapturingDiffusion");
      double sc_uref = args.scalar<double>("sc_uref");
      double sc_alpha = args.scalar<double>("sc_alpha");
      xt::pyarray<int>& u_l2g = args.array<int>("u_l2g");
      xt::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
      xt::pyarray<double>& u_dof = args.array<double>("u_dof");
      xt::pyarray<double>& u_dof_old = args.array<double>("u_dof_old");
      xt::pyarray<double>& velocity = args.array<double>("velocity");
      xt::pyarray<double>& q_m = args.array<double>("q_m");
      xt::pyarray<double>& q_u = args.array<double>("q_u");
      xt::pyarray<double>& q_dV = args.array<double>("q_dV");
      xt::pyarray<double>& q_m_betaBDF = args.array<double>("q_m_betaBDF");
      xt::pyarray<double>& cfl = args.array<double>("cfl");
      xt::pyarray<double>& q_numDiff_u = args.array<double>("q_numDiff_u");
      xt::pyarray<double>& q_numDiff_u_last = args.array<double>("q_numDiff_u_last");
      int offset_u = args.scalar<int>("offset_u");
      int stride_u = args.scalar<int>("stride_u");
      xt::pyarray<double>& globalResidual = args.array<double>("globalResidual");
      int nExteriorElementBoundaries_global = args.scalar<int>("nExteriorElementBoundaries_global");
      xt::pyarray<int>& exteriorElementBoundariesArray = args.array<int>("exteriorElementBoundariesArray");
      xt::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
      xt::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
      xt::pyarray<double>& ebqe_velocity_ext = args.array<double>("ebqe_velocity_ext");
      xt::pyarray<int>& isDOFBoundary_u = args.array<int>("isDOFBoundary_u");
      xt::pyarray<double>& ebqe_bc_u_ext = args.array<double>("ebqe_bc_u_ext");
      xt::pyarray<int>& isFluxBoundary_u = args.array<int>("isFluxBoundary_u");
      xt::pyarray<double>& ebqe_bc_flux_ext = args.array<double>("ebqe_bc_flux_ext");
      xt::pyarray<double>& ebqe_phi = args.array<double>("ebqe_phi");
      double epsFact = args.scalar<double>("epsFact");
      xt::pyarray<double>& ebqe_u = args.array<double>("ebqe_u");
      xt::pyarray<double>& ebqe_flux = args.array<double>("ebqe_flux");
      // PARAMETERS FOR EDGE BASED STABILIZATION
      double cE = args.scalar<double>("cE");
      double cK = args.scalar<double>("cK");
      // PARAMETERS FOR LOG BASED ENTROPY FUNCTION
      double uL = args.scalar<double>("uL");
      double uR = args.scalar<double>("uR");
      // PARAMETERS FOR EDGE VISCOSITY
      int numDOFs = args.scalar<int>("numDOFs");
      int NNZ = args.scalar<int>("NNZ");
      xt::pyarray<int>& csrRowIndeces_DofLoops = args.array<int>("csrRowIndeces_DofLoops");
      xt::pyarray<int>& csrColumnOffsets_DofLoops = args.array<int>("csrColumnOffsets_DofLoops");
      xt::pyarray<int>& csrRowIndeces_CellLoops = args.array<int>("csrRowIndeces_CellLoops");
      xt::pyarray<int>& csrColumnOffsets_CellLoops = args.array<int>("csrColumnOffsets_CellLoops");
      xt::pyarray<int>& csrColumnOffsets_eb_CellLoops = args.array<int>("csrColumnOffsets_eb_CellLoops");
      // C matrices
      xt::pyarray<double>& Cx = args.array<double>("Cx");
      xt::pyarray<double>& Cy = args.array<double>("Cy");
      xt::pyarray<double>& Cz = args.array<double>("Cz");
      xt::pyarray<double>& CTx = args.array<double>("CTx");
      xt::pyarray<double>& CTy = args.array<double>("CTy");
      xt::pyarray<double>& CTz = args.array<double>("CTz");
      xt::pyarray<double>& ML = args.array<double>("ML");
      xt::pyarray<double>& delta_x_ij = args.array<double>("delta_x_ij");
      // PARAMETERS FOR 1st or 2nd ORDER MPP METHOD
      int LUMPED_MASS_MATRIX  = args.scalar<int>("LUMPED_MASS_MATRIX");
      int STABILIZATION_TYPE = args.scalar<int>("STABILIZATION_TYPE");
      int ENTROPY_TYPE = args.scalar<int>("ENTROPY_TYPE");
      // FOR FCT
      xt::pyarray<double>& dLow = args.array<double>("dLow");
      xt::pyarray<double>& fluxMatrix = args.array<double>("fluxMatrix");
      xt::pyarray<double>& uDotLow = args.array<double>("uDotLow");
      xt::pyarray<double>& uLow = args.array<double>("uLow");
      xt::pyarray<double>& dt_times_fH_minus_fL = args.array<double>("dt_times_fH_minus_fL");
      xt::pyarray<double>& min_s_bc = args.array<double>("min_s_bc");
      xt::pyarray<double>& max_s_bc = args.array<double>("max_s_bc");
      // AUX QUANTITIES OF INTEREST
      xt::pyarray<double>& quantDOFs = args.array<double>("quantDOFs");
      xt::pyarray<double>& sLow = args.array<double>("sLow");
      xt::pyarray<double>& sn = args.array<double>("sn");

      assert(a_rowptr.data()[nSpace] == nnz);
      assert(a_rowptr.data()[nSpace] == nSpace);
      //cek should this be read in?
      double Ct_sge = 4.0;
	//   if (LUMPED_MASS_MATRIX ==1)
    // {
    //     mass_lumping(nElements_global, nDOF_test_element, nQuadraturePoints_element, u_test_ref, dV_ref, ML, u_l2g);
    // }

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
	  double elementResidual_u[nDOF_test_element];
	  for (int i=0;i<nDOF_test_element;i++)
	    {
	      elementResidual_u[i]=0.0;
	    }//i
	  //loop over quadrature points and compute integrands
	  for  (int k=0;k<nQuadraturePoints_element;k++)
	    {
	      //compute indeces and declare local storage
	      int eN_k = eN*nQuadraturePoints_element+k,
		eN_k_nSpace = eN_k*nSpace,
		eN_nDOF_trial_element = eN*nDOF_trial_element;
	      double u=0.0,grad_u[nSpace],grad_u_old[nSpace],
		m=0.0,dm=0.0,
		f[nSpace],df[nSpace],
		a[nnz],da[nnz],as[nnz],
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
					  mesh_dof.data(),
					  mesh_l2g.data(),
					  mesh_trial_ref.data(),
					  mesh_grad_trial_ref.data(),
					  jac,
					  jacDet,
					  jacInv,
					  x,y,z);
	      ck.calculateMappingVelocity_element(eN,
						  k,
						  mesh_velocity_dof.data(),
						  mesh_l2g.data(),
						  mesh_trial_ref.data(),
						  xt,yt,zt);
	      //get the physical integration weight
	      dV = fabs(jacDet)*dV_ref.data()[k];
	      q_dV.data()[eN_k] = dV;
	      ck.calculateG(jacInv,G,G_dd_G,tr_G);
	      //get the trial function gradients
	      ck.gradTrialFromRef(&u_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],jacInv,u_grad_trial);
	      //get the solution
	      ck.valFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],&u_trial_ref.data()[k*nDOF_trial_element],u);
	      //get the solution gradients
	      ck.gradFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],u_grad_trial,grad_u);
	      //precalculate test function products with integration weights
	      for (int j=0;j<nDOF_trial_element;j++)
		{
		  u_test_dV[j] = u_test_ref.data()[k*nDOF_trial_element+j]*dV;
		  for (int I=0;I<nSpace;I++)
		    {
		      u_grad_test_dV[j*nSpace+I]   = u_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin
		    }
		}
	      //
	//       //calculate pde coefficients at quadrature points
	//       //
	      double Kr,dKr;
	      evaluateCoefficients(a_rowptr.data(),
				   a_colind.data(),
				   rho,
				   beta,
				   gravity.data(),
				   alpha.data()[elementMaterialTypes.data()[eN]],
				   n.data()[elementMaterialTypes.data()[eN]],
				   thetaR.data()[elementMaterialTypes.data()[eN]],
				   thetaSR.data()[elementMaterialTypes.data()[eN]],
				   &KWs.data()[elementMaterialTypes.data()[eN]*nnz],
				   u,
				   m,
				   dm,
				   f,
				   df,
				   a,
				   da,
				   as,
				   Kr,
				   dKr);
	      //
	//       //calculate time derivative at quadrature points
	//       //
	      ck.bdf(alphaBDF,
		     q_m_betaBDF.data()[eN_k],
		     m,
		     dm,
		     m_t,
		     dm_t);
		

	//       //update element residual
	//       //
	      for(int i=0;i<nDOF_test_element;i++)
		{
		  int eN_k_i=eN_k*nDOF_test_element+i,
		    eN_k_i_nSpace = eN_k_i*nSpace,
		    i_nSpace=i*nSpace;
			if (LUMPED_MASS_MATRIX==1)
                {
                    // Lumped mass matrix contribution
                    globalResidual.data()[offset_u + stride_u*u_l2g.data()[eN*nDOF_test_element + i]] += u_test_dV[i] * m_t;
                }
                else
                {
                    elementResidual_u[i] += ck.Mass_weak(m_t, u_test_dV[i]);
                }

		  elementResidual_u[i] += ck.Advection_weak(f,&u_grad_test_dV[i_nSpace]) +
		    					  ck.Diffusion_weak(a_rowptr.data(),a_colind.data(),a,grad_u,&u_grad_test_dV[i_nSpace]);
								  //+ ck.Mass_weak(m_t,u_test_dV[i]) 
		  /* +  */
		//   /*   ck.SubgridError(subgridError_u,Lstar_u[i]) +  */
		//   /*   ck.NumericalDiffusion(q_numDiff_u_last[eN_k],grad_u,&u_grad_test_dV[i_nSpace]);  */
		 }//i
	    //   //
	       q_m.data()[eN_k] = m;
	       q_u.data()[eN_k] = u;
	     }
	//   //
	  //load element into global residual and save element residual
	  //
	  for(int i=0;i<nDOF_test_element;i++)
	    {
	      int eN_i=eN*nDOF_test_element+i;

	      globalResidual.data()[offset_u+stride_u*u_l2g.data()[eN_i]] += elementResidual_u[i];
	    }//i
	}//elements
     }

    void calculateJacobian(arguments_dict& args)
    {
      xt::pyarray<double>& mesh_trial_ref = args.array<double>("mesh_trial_ref");
      xt::pyarray<double>& mesh_grad_trial_ref = args.array<double>("mesh_grad_trial_ref");
      xt::pyarray<double>& mesh_dof = args.array<double>("mesh_dof");
      xt::pyarray<double>& mesh_velocity_dof = args.array<double>("mesh_velocity_dof");
      double MOVING_DOMAIN = args.scalar<double>("MOVING_DOMAIN");
      xt::pyarray<int>& mesh_l2g = args.array<int>("mesh_l2g");
      xt::pyarray<double>& dV_ref = args.array<double>("dV_ref");
      xt::pyarray<double>& u_trial_ref = args.array<double>("u_trial_ref");
      xt::pyarray<double>& u_grad_trial_ref = args.array<double>("u_grad_trial_ref");
      xt::pyarray<double>& u_test_ref = args.array<double>("u_test_ref");
      xt::pyarray<double>& u_grad_test_ref = args.array<double>("u_grad_test_ref");
      xt::pyarray<double>& mesh_trial_trace_ref = args.array<double>("mesh_trial_trace_ref");
      xt::pyarray<double>& mesh_grad_trial_trace_ref = args.array<double>("mesh_grad_trial_trace_ref");
      xt::pyarray<double>& dS_ref = args.array<double>("dS_ref");
      xt::pyarray<double>& u_trial_trace_ref = args.array<double>("u_trial_trace_ref");
      xt::pyarray<double>& u_grad_trial_trace_ref = args.array<double>("u_grad_trial_trace_ref");
      xt::pyarray<double>& u_test_trace_ref = args.array<double>("u_test_trace_ref");
      xt::pyarray<double>& u_grad_test_trace_ref = args.array<double>("u_grad_test_trace_ref");
      xt::pyarray<double>& normal_ref = args.array<double>("normal_ref");
      xt::pyarray<double>& boundaryJac_ref = args.array<double>("boundaryJac_ref");
      int nElements_global = args.scalar<int>("nElements_global");
      xt::pyarray<double>& ebqe_penalty_ext = args.array<double>("ebqe_penalty_ext");
      xt::pyarray<int>& elementMaterialTypes = args.array<int>("elementMaterialTypes");
      xt::pyarray<int>& isSeepageFace = args.array<int>("isSeepageFace");
      xt::pyarray<int>& a_rowptr = args.array<int>("a_rowptr");
      xt::pyarray<int>& a_colind = args.array<int>("a_colind");
      double rho = args.scalar<double>("rho");
      double beta = args.scalar<double>("beta");
      xt::pyarray<double>& gravity = args.array<double>("gravity");
      xt::pyarray<double>& alpha = args.array<double>("alpha");
      xt::pyarray<double>& n = args.array<double>("n");
      xt::pyarray<double>& thetaR = args.array<double>("thetaR");
      xt::pyarray<double>& thetaSR = args.array<double>("thetaSR");
      xt::pyarray<double>& KWs = args.array<double>("KWs");
      double useMetrics = args.scalar<double>("useMetrics");
      double alphaBDF = args.scalar<double>("alphaBDF");
      int lag_shockCapturing = args.scalar<int>("lag_shockCapturing");
      double shockCapturingDiffusion = args.scalar<double>("shockCapturingDiffusion");
      xt::pyarray<int>& u_l2g = args.array<int>("u_l2g");
      xt::pyarray<double>& elementDiameter = args.array<double>("elementDiameter");
      xt::pyarray<double>& u_dof = args.array<double>("u_dof");
      xt::pyarray<double>& velocity = args.array<double>("velocity");
      xt::pyarray<double>& q_m_betaBDF = args.array<double>("q_m_betaBDF");
      xt::pyarray<double>& cfl = args.array<double>("cfl");
      xt::pyarray<double>& q_numDiff_u_last = args.array<double>("q_numDiff_u_last");
      xt::pyarray<int>& csrRowIndeces_u_u = args.array<int>("csrRowIndeces_u_u");
      xt::pyarray<int>& csrColumnOffsets_u_u = args.array<int>("csrColumnOffsets_u_u");
      xt::pyarray<double>& globalJacobian = args.array<double>("globalJacobian");
      int nExteriorElementBoundaries_global = args.scalar<int>("nExteriorElementBoundaries_global");
      xt::pyarray<int>& exteriorElementBoundariesArray = args.array<int>("exteriorElementBoundariesArray");
      xt::pyarray<int>& elementBoundaryElementsArray = args.array<int>("elementBoundaryElementsArray");
      xt::pyarray<int>& elementBoundaryLocalElementBoundariesArray = args.array<int>("elementBoundaryLocalElementBoundariesArray");
      xt::pyarray<double>& ebqe_velocity_ext = args.array<double>("ebqe_velocity_ext");
      xt::pyarray<int>& isDOFBoundary_u = args.array<int>("isDOFBoundary_u");
      xt::pyarray<double>& ebqe_bc_u_ext = args.array<double>("ebqe_bc_u_ext");
      xt::pyarray<int>& isFluxBoundary_u = args.array<int>("isFluxBoundary_u");
      xt::pyarray<double>& ebqe_bc_flux_ext = args.array<double>("ebqe_bc_flux_ext");
      xt::pyarray<int>& csrColumnOffsets_eb_u_u = args.array<int>("csrColumnOffsets_eb_u_u");
      int LUMPED_MASS_MATRIX = args.scalar<int>("LUMPED_MASS_MATRIX");
      assert(a_rowptr.data()[nSpace] == nnz);
      assert(a_rowptr.data()[nSpace] == nSpace);
      double Ct_sge = 4.0;

      //
      //loop over elements to compute volume integrals and load them into the element Jacobians and global Jacobian
      //
      for(int eN=0;eN<nElements_global;eN++)
	{
	  double  elementJacobian_u_u[nDOF_test_element][nDOF_trial_element];
	  for (int i=0;i<nDOF_test_element;i++)
	    {
	      for (int j=0;j<nDOF_trial_element;j++)
		{
		  elementJacobian_u_u[i][j]=0.0;
		}
	    }
	  for (int k=0;k<nQuadraturePoints_element;k++)
	    {
	      int eN_k = eN*nQuadraturePoints_element+k, //index to a scalar at a quadrature point
		eN_k_nSpace = eN_k*nSpace,
		eN_nDOF_trial_element = eN*nDOF_trial_element; //index to a vector at a quadrature point

	      //declare local storage
	      double u=0.0,
		grad_u[nSpace],
		m=0.0,dm=0.0,
		f[nSpace],df[nSpace],
		a[nnz],da[nnz],as[nnz],
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
					  mesh_dof.data(),
					  mesh_l2g.data(),
					  mesh_trial_ref.data(),
					  mesh_grad_trial_ref.data(),
					  jac,
					  jacDet,
					  jacInv,
					  x,y,z);
	      ck.calculateMappingVelocity_element(eN,
						  k,
						  mesh_velocity_dof.data(),
						  mesh_l2g.data(),
						  mesh_trial_ref.data(),
						  xt,yt,zt);
	      //get the physical integration weight
	      dV = fabs(jacDet)*dV_ref.data()[k];
	      ck.calculateG(jacInv,G,G_dd_G,tr_G);
	      //get the trial function gradients
	      ck.gradTrialFromRef(&u_grad_trial_ref.data()[k*nDOF_trial_element*nSpace],jacInv,u_grad_trial);
	      //get the solution
	      ck.valFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],&u_trial_ref.data()[k*nDOF_trial_element],u);
	      //get the solution gradients
	      ck.gradFromDOF(u_dof.data(),&u_l2g.data()[eN_nDOF_trial_element],u_grad_trial,grad_u);
	      //precalculate test function products with integration weights
	      for (int j=0;j<nDOF_trial_element;j++)
		{
		  u_test_dV[j] = u_test_ref.data()[k*nDOF_trial_element+j]*dV;
		  for (int I=0;I<nSpace;I++)
		    {
		      u_grad_test_dV[j*nSpace+I]   = u_grad_trial[j*nSpace+I]*dV;//cek warning won't work for Petrov-Galerkin
		    }
		}
	      //
	      //calculate pde coefficients and derivatives at quadrature points
	      //
	      double Kr,dKr;
	      evaluateCoefficients(a_rowptr.data(),
				   a_colind.data(),
				   rho,
				   beta,
				   gravity.data(),
				   alpha.data()[elementMaterialTypes.data()[eN]],
				   n.data()[elementMaterialTypes.data()[eN]],
				   thetaR.data()[elementMaterialTypes.data()[eN]],
				   thetaSR.data()[elementMaterialTypes.data()[eN]],
				   &KWs.data()[elementMaterialTypes.data()[eN]*nnz],
				   u,
				   m,
				   dm,
				   f,
				   df,
				   a,
				   da,
				   as,
				   Kr,
				   dKr);
	      //
	      //calculate time derivatives
	      
	      ck.bdf(alphaBDF,
		     q_m_betaBDF.data()[eN_k],
		     m,
		     dm,
		     m_t,
		     dm_t);

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

		      elementJacobian_u_u[i][j] +=  ck.MassJacobian_weak(dm_t,u_trial_ref.data()[k*nDOF_trial_element+j],u_test_dV[i]) +
			  								ck.AdvectionJacobian_weak(df,u_trial_ref.data()[k*nDOF_trial_element+j],&u_grad_test_dV[i_nSpace]) +
										    ck.DiffusionJacobian_weak(a_rowptr.data(),a_colind.data(),a,da,
			 			  					grad_u,&u_grad_test_dV[i_nSpace],1.0,
			 			  					u_trial_ref.data()[k*nDOF_trial_element+j],&u_grad_trial[j_nSpace]);
											//+
		      // +
		      //     ck.SubgridErrorJacobian(dsubgridError_u_u[j],Lstar_u[i]) +
		      //     ck.NumericalDiffusionJacobian(q_numDiff_u_last[eN_k],&u_grad_trial[j_nSpace],&u_grad_test_dV[i_nSpace]);
		    }//j
		}//i
	     }//k
	//   //
	  //load into element Jacobian into global Jacobian
	  //
	  for (int i=0;i<nDOF_test_element;i++)
	    {
	      int eN_i = eN*nDOF_test_element+i;
	      for (int j=0;j<nDOF_trial_element;j++)
		{
		  int eN_i_j = eN_i*nDOF_trial_element+j;
		  globalJacobian.data()[csrRowIndeces_u_u[eN_i] + csrColumnOffsets_u_u[eN_i_j]] += elementJacobian_u_u[i][j];
		}//j
	    }//i
	}//elements
 
    }//computeJacobian

  };//ADR
    
 

  inline ADR_base* newADR(int nSpaceIn,
				    int nQuadraturePoints_elementIn,
				    int nDOF_mesh_trial_elementIn,
				    int nDOF_trial_elementIn,
				    int nDOF_test_elementIn,
				    int nQuadraturePoints_elementBoundaryIn,
				    int CompKernelFlag)
  {
    if (nSpaceIn == 1)
      return proteus::chooseAndAllocateDiscretization1D<ADR_base,ADR,CompKernel>(nSpaceIn,
											   nQuadraturePoints_elementIn,
											   nDOF_mesh_trial_elementIn,
											   nDOF_trial_elementIn,
											   nDOF_test_elementIn,
											   nQuadraturePoints_elementBoundaryIn,
											   CompKernelFlag);
    else if (nSpaceIn == 2)
      return proteus::chooseAndAllocateDiscretization2D<ADR_base,ADR,CompKernel>(nSpaceIn,
											   nQuadraturePoints_elementIn,
											   nDOF_mesh_trial_elementIn,
											   nDOF_trial_elementIn,
											   nDOF_test_elementIn,
											   nQuadraturePoints_elementBoundaryIn,
											   CompKernelFlag);
    else
      {
	assert(nSpaceIn == 3);
	return proteus::chooseAndAllocateDiscretization<ADR_base,ADR,CompKernel>(nSpaceIn,
											   nQuadraturePoints_elementIn,
											   nDOF_mesh_trial_elementIn,
											   nDOF_trial_elementIn,
											   nDOF_test_elementIn,
											   nQuadraturePoints_elementBoundaryIn,
											   CompKernelFlag);
      }
  }
};//proteus
#endif
