# A type of -*- python -*- file
import numpy
cimport numpy

cdef extern from "mprans/AddedMass.h" namespace "proteus":
    cdef cppclass cppAddedMass_base:
        void calculateResidual(double * mesh_trial_ref,
                               double * mesh_grad_trial_ref,
                               double * mesh_dof,
                               int * mesh_l2g,
                               double * dV_ref,
                               double * u_trial_ref,
                               double * u_grad_trial_ref,
                               double * u_test_ref,
                               double * u_grad_test_ref,
                               double * mesh_trial_trace_ref,
                               double * mesh_grad_trial_trace_ref,
                               double * dS_ref,
                               double * u_trial_trace_ref,
                               double * u_grad_trial_trace_ref,
                               double * u_test_trace_ref,
                               double * u_grad_test_trace_ref,
                               double * normal_ref,
                               double * boundaryJac_ref,
                               int nElements_global,
                               int nElementBoundaries_owned,
                               int * u_l2g,
                               double * u_dof,
                               double * q_rho,
                               int offset_u,
                               int stride_u,
                               double * globalResidual,
                               int nExteriorElementBoundaries_global,
                               int * exteriorElementBoundariesArray,
                               int * elementBoundaryElementsArray,
                               int * elementBoundaryLocalElementBoundariesArray,
                               int * elementBoundaryMaterialTypesArray,
                               double* Aij)
        void calculateJacobian(double * mesh_trial_ref,
                               double * mesh_grad_trial_ref,
                               double * mesh_dof,
                               int * mesh_l2g,
                               double * dV_ref,
                               double * u_trial_ref,
                               double * u_grad_trial_ref,
                               double * u_test_ref,
                               double * u_grad_test_ref,
                               double * mesh_trial_trace_ref,
                               double * mesh_grad_trial_trace_ref,
                               double * dS_ref,
                               double * u_trial_trace_ref,
                               double * u_grad_trial_trace_ref,
                               double * u_test_trace_ref,
                               double * u_grad_test_trace_ref,
                               double * normal_ref,
                               double * boundaryJac_ref,
                               int nElements_global,
                               int * u_l2g,
                               double * u_dof,
                               double * q_rho,
                               int * csrRowIndeces_u_u,
                               int * csrColumnOffsets_u_u,
                               double * globalJacobian,
                               int nExteriorElementBoundaries_global,
                               int * exteriorElementBoundariesArray,
                               int * elementBoundaryElementsArray,
                               int * elementBoundaryLocalElementBoundariesArray,
                               int * csrColumnOffsets_eb_u_u)
    cppAddedMass_base * newAddedMass(int nSpaceIn,
                                 int nQuadraturePoints_elementIn,
                                 int nDOF_mesh_trial_elementIn,
                                 int nDOF_trial_elementIn,
                                 int nDOF_test_elementIn,
                                 int nQuadraturePoints_elementBoundaryIn,
                                 int CompKernelFlag)

cdef class AddedMass:
    cdef cppAddedMass_base * thisptr

    def __cinit__(self,
                  int nSpaceIn,
                  int nQuadraturePoints_elementIn,
                  int nDOF_mesh_trial_elementIn,
                  int nDOF_trial_elementIn,
                  int nDOF_test_elementIn,
                  int nQuadraturePoints_elementBoundaryIn,
                  int CompKernelFlag):
        self.thisptr = newAddedMass(nSpaceIn,
                                  nQuadraturePoints_elementIn,
                                  nDOF_mesh_trial_elementIn,
                                  nDOF_trial_elementIn,
                                  nDOF_test_elementIn,
                                  nQuadraturePoints_elementBoundaryIn,
                                  CompKernelFlag)

    def __dealloc__(self):
        del self.thisptr

    def calculateResidual(self,
                          numpy.ndarray mesh_trial_ref,
                          numpy.ndarray mesh_grad_trial_ref,
                          numpy.ndarray mesh_dof,
                          numpy.ndarray mesh_l2g,
                          numpy.ndarray dV_ref,
                          numpy.ndarray u_trial_ref,
                          numpy.ndarray u_grad_trial_ref,
                          numpy.ndarray u_test_ref,
                          numpy.ndarray u_grad_test_ref,
                          numpy.ndarray mesh_trial_trace_ref,
                          numpy.ndarray mesh_grad_trial_trace_ref,
                          numpy.ndarray dS_ref,
                          numpy.ndarray u_trial_trace_ref,
                          numpy.ndarray u_grad_trial_trace_ref,
                          numpy.ndarray u_test_trace_ref,
                          numpy.ndarray u_grad_test_trace_ref,
                          numpy.ndarray normal_ref,
                          numpy.ndarray boundaryJac_ref,
                          int nElements_global,
                          int nElementBoundaries_owned,
                          numpy.ndarray u_l2g,
                          numpy.ndarray u_dof,
                          numpy.ndarray q_rho,
                          int offset_u,
                          int stride_u,
                          numpy.ndarray globalResidual,
                          int nExteriorElementBoundaries_global,
                          numpy.ndarray exteriorElementBoundariesArray,
                          numpy.ndarray elementBoundaryElementsArray,
                          numpy.ndarray elementBoundaryLocalElementBoundariesArray,
                          numpy.ndarray elementBoundaryMaterialTypesArray,
                          numpy.ndarray Aij):
        self.thisptr.calculateResidual(< double*> mesh_trial_ref.data,
                                       < double * > mesh_grad_trial_ref.data,
                                       < double * > mesh_dof.data,
                                       < int * > mesh_l2g.data,
                                       < double * > dV_ref.data,
                                       < double * > u_trial_ref.data,
                                       < double * > u_grad_trial_ref.data,
                                       < double * > u_test_ref.data,
                                       < double * > u_grad_test_ref.data,
                                       < double * > mesh_trial_trace_ref.data,
                                       < double * > mesh_grad_trial_trace_ref.data,
                                       < double * > dS_ref.data,
                                       < double * > u_trial_trace_ref.data,
                                       < double * > u_grad_trial_trace_ref.data,
                                       < double * > u_test_trace_ref.data,
                                       < double * > u_grad_test_trace_ref.data,
                                       < double * > normal_ref.data,
                                       < double * > boundaryJac_ref.data,
                                       nElements_global,
                                       nElementBoundaries_owned,
                                       < int * > u_l2g.data,
                                       < double * > u_dof.data,
                                       < double * > q_rho.data,
                                       offset_u,
                                       stride_u,
                                       < double * > globalResidual.data,
                                       nExteriorElementBoundaries_global,
                                       < int * > exteriorElementBoundariesArray.data,
                                       < int * > elementBoundaryElementsArray.data,
                                       < int * > elementBoundaryLocalElementBoundariesArray.data,
                                       < int *> elementBoundaryMaterialTypesArray.data,
                                       < double* > Aij.data)

    def calculateJacobian(self,
                          numpy.ndarray mesh_trial_ref,
                          numpy.ndarray mesh_grad_trial_ref,
                          numpy.ndarray mesh_dof,
                          numpy.ndarray mesh_l2g,
                          numpy.ndarray dV_ref,
                          numpy.ndarray u_trial_ref,
                          numpy.ndarray u_grad_trial_ref,
                          numpy.ndarray u_test_ref,
                          numpy.ndarray u_grad_test_ref,
                          numpy.ndarray mesh_trial_trace_ref,
                          numpy.ndarray mesh_grad_trial_trace_ref,
                          numpy.ndarray dS_ref,
                          numpy.ndarray u_trial_trace_ref,
                          numpy.ndarray u_grad_trial_trace_ref,
                          numpy.ndarray u_test_trace_ref,
                          numpy.ndarray u_grad_test_trace_ref,
                          numpy.ndarray normal_ref,
                          numpy.ndarray boundaryJac_ref,
                          int nElements_global,
                          numpy.ndarray u_l2g,
                          numpy.ndarray u_dof,
                          numpy.ndarray q_rho,
                          numpy.ndarray csrRowIndeces_u_u,
                          numpy.ndarray csrColumnOffsets_u_u,
                          globalJacobian,
                          int nExteriorElementBoundaries_global,
                          numpy.ndarray exteriorElementBoundariesArray,
                          numpy.ndarray elementBoundaryElementsArray,
                          numpy.ndarray elementBoundaryLocalElementBoundariesArray,
                          numpy.ndarray csrColumnOffsets_eb_u_u):
        cdef numpy.ndarray rowptr, colind, globalJacobian_a
        (rowptr, colind, globalJacobian_a) = globalJacobian.getCSRrepresentation()
        self.thisptr.calculateJacobian(< double*> mesh_trial_ref.data,
                                        < double * > mesh_grad_trial_ref.data,
                                        < double * > mesh_dof.data,
                                        < int * > mesh_l2g.data,
                                        < double * > dV_ref.data,
                                        < double * > u_trial_ref.data,
                                        < double * > u_grad_trial_ref.data,
                                        < double * > u_test_ref.data,
                                        < double * > u_grad_test_ref.data,
                                        < double * > mesh_trial_trace_ref.data,
                                        < double * > mesh_grad_trial_trace_ref.data,
                                        < double * > dS_ref.data,
                                        < double * > u_trial_trace_ref.data,
                                        < double * > u_grad_trial_trace_ref.data,
                                        < double * > u_test_trace_ref.data,
                                        < double * > u_grad_test_trace_ref.data,
                                        < double * > normal_ref.data,
                                        < double * > boundaryJac_ref.data,
                                        nElements_global,
                                        < int * > u_l2g.data,
                                        < double * > u_dof.data,
                                        < double * > q_rho.data,
                                        < int * > csrRowIndeces_u_u.data,
                                        < int * > csrColumnOffsets_u_u.data,
                                        < double * > globalJacobian_a.data,
                                        nExteriorElementBoundaries_global,
                                        < int * > exteriorElementBoundariesArray.data,
                                        < int * > elementBoundaryElementsArray.data,
                                        < int * > elementBoundaryLocalElementBoundariesArray.data,
                                        < int * > csrColumnOffsets_eb_u_u.data)
