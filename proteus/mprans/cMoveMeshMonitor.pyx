cimport numpy as np
import numpy as np
from libcpp cimport bool
from proteus.Profiling import logEvent
from proteus.mprans import MeshSmoothing as ms


cdef class cCoefficients:
    cdef public:
        object pyCoefficients
        double C

    def __cinit__(self):
        self.C = 1.

    def attachPyCoefficients(self,
                             object pyCoefficients):
        self.pyCoefficients = pyCoefficients

    def preStep(self):
        pc = self.pyCoefficients
        self.cppPreStep(q_uOfX=pc.uOfXTatQuadrature,
                        q_J=pc.model.q['abs(det(J))'],
                        q_weights=pc.model.elementQuadratureWeights[('u', 0)],
                        areas_=pc.areas,
                        q_rci=pc.model.q[('r', 0)],
                        q_fci=pc.model.q[('f', 0)],
                        t=pc.t)

    cdef cppPreStep(self,
                    double[:,:] q_uOfX,
                    double[:,:] q_J,
                    double[:] q_weights,
                    double[:] areas_,
                    double[:, :] q_rci,
                    double[:, :, :] q_fci,
                    double t):
        cdef double integral_1_over_f = 0.
        cdef int N_eN = q_J.shape[0]
        cdef int nE = 0
        cdef double area = 0
        for e in xrange(len(q_J)):
            area = 0
            for k in xrange(len(q_weights)):
                area += q_J[e, k]*q_weights[k]
                integral_1_over_f += q_J[e, k]*q_weights[k]/q_uOfX[e, k]
            areas_[e] = area
            nE += 1
        self.C = integral_1_over_f/nE  # update scaling coefficient

        for eN in range(len(q_rci)):
            for k in range(len(q_rci[eN])):
                for kk in range(len(q_fci[eN, k])):
                    q_fci[eN, k, kk] = 0.0
                q_rci[eN, k] = -(1./(q_uOfX[eN, k]*self.C)-1./areas_[eN])


    def pseudoTimeStepping(self,
                           xx,
                           eps=1.):
        pc = self.pyCoefficients
        return self.cppPseudoTimeStepping(xx=xx,
                                          eps=eps,
                                          nodeMaterialTypes=pc.mesh.nodeMaterialTypes,
                                          grads=pc.grads,
                                          areas_nodes=pc.areas_nodes,
                                          areas=pc.areas,
                                          nd=int(pc.nd),
                                          u_phi=pc.u_phi,
                                          bNs=pc.boundaryNormals,
                                          fixedNodes=pc.fixedNodes)

    cdef cppPseudoTimeStepping(self,
                               double[:,:] xx,
                               double eps,
                               int[:] nodeMaterialTypes,
                               double[:,:] grads,
                               double[:] areas_nodes,
                               double[:] areas,
                               int nd,
                               double[:] u_phi=None,
                               double[:,:] bNs=None,
                               int[:] fixedNodes=None):
        logEvent("Pseudo-time stepping with dt={eps} (0<t<1)".format(eps=eps),
                 level=3)
        pc = self.pyCoefficients
        cdef double[:] t_range = np.linspace(0., 1., int(1./eps+1))[1:]
        cdef int[:] eN_phi = np.zeros(len(xx), dtype=np.int32)
        cdef int[:] nearest_nodes = np.zeros(len(xx), dtype=np.int32)
        cdef double t_last = 0
        cdef double dt = 0
        cdef int flag
        cdef bool fixed
        cdef double area = 0
        cdef int eN
        cdef double ls_phi
        cdef double f
        cdef double[:] dphi = np.zeros(nd)
        cdef double[:] v_grad = np.zeros(nd)
        cdef bool found_vars
        cdef object femSpace = pc.model.u[0].femSpace
        # initialise mesh memoryview before loop
        cdef double[:,:] nodeArray = pc.mesh.nodeArray
        cdef int[:] nodeStarOffsets = pc.mesh.nodeStarOffsets
        cdef int[:] nodeStarArray = pc.mesh.nodeStarArray
        cdef int[:] nodeElementOffsets = pc.mesh.nodeElementOffsets
        cdef int[:] nodeElementsArray = pc.mesh.nodeElementsArray
        cdef double[:,:] elementBarycentersArray = pc.mesh.elementBarycentersArray
        cdef int[:,:] elementNeighborsArray = pc.mesh.elementNeighborsArray
        cdef int[:,:] elementBoundariesArray = pc.mesh.elementBoundariesArray
        cdef double[:,:] elementBoundaryBarycentersArray = pc.mesh.elementBoundaryBarycentersArray
        cdef int[:] exteriorElementBoundariesBoolArray = np.zeros(pc.mesh.nElementBoundaries_global, dtype=np.int32)
        for b_i in pc.mesh.exteriorElementBoundariesArray:
            exteriorElementBoundariesBoolArray[b_i] = 1
        cdef double[:,:,:] elementNormalsArray
        if pc.nd == 2:
            elementNormalsArray = ms.getElementBoundaryNormalsTriangle2D(nodeArray,
                                                                         elementBoundariesArray,
                                                                         pc.mesh.elementBoundaryNodesArray,
                                                                         elementBoundaryBarycentersArray,
                                                                         elementBarycentersArray)
        elif pc.nd == 3:
            elementNormalsArray = ms.getElementBoundaryNormalsTetra3D(nodeArray,
                                                                      elementBoundariesArray,
                                                                      pc.mesh.elementBoundaryNodesArray,
                                                                      elementBoundaryBarycentersArray,
                                                                      elementBarycentersArray)
        for j, t in enumerate(t_range):
            logEvent("Pseudo-time stepping t={t}".format(t=t), level=3)
            for i in range(len(xx)):
                # reinitialise values
                fixed = False
                found_vars = True
                area = 0.
                for ndi in range(nd):
                    v_grad[ndi] = 0.
                    dphi[ndi] = 0.
                dt = t-t_last
                flag = nodeMaterialTypes[i]
                if flag != 0 and bNs is not None:
                    if bNs[flag, 0] == 0. and bNs[flag, 1] == 0. and bNs[flag, 2] == 0:
                        fixed = True
                    if fixedNodes is not None:
                        if fixedNodes[flag] == 1:
                            fixed = True
                if not fixed:  # either flag==0 or not fixed
                    if j == 0:  # nodes are at original position (no search)
                        v_grad = grads[i]
                        area = areas_nodes[i]
                        nearest_nodes[i] = i
                        eN = -1
                        eN_phi[i] = eN
                        if u_phi is not None:
                            ls_phi = u_phi[i]
                    else:  # find node that moved already (search)
                        # line below needs optimisation:
                        eN, nearest_node = pyxSearchNearestNodeElement(x=xx[i],
                                                                       nodeArray=nodeArray,
                                                                       nodeStarOffsets=nodeStarOffsets,
                                                                       nodeStarArray=nodeStarArray,
                                                                       nodeElementOffsets=nodeElementOffsets,
                                                                       nodeElementsArray=nodeElementsArray,
                                                                       elementBarycentersArray=elementBarycentersArray,
                                                                       elementNeighborsArray=elementNeighborsArray,
                                                                       elementBoundariesArray=elementBoundariesArray,
                                                                       elementBoundaryBarycentersArray=elementBoundaryBarycentersArray,
                                                                       exteriorElementBoundariesBoolArray=exteriorElementBoundariesBoolArray,
                                                                       elementNormalsArray=elementNormalsArray,
                                                                       nearest_node=nearest_nodes[i],
                                                                       eN=eN_phi[i])
                        xi = femSpace.elementMaps.getInverseValue(eN, xx[i])
                        eN_phi[i] = eN
                        nearest_nodes[i] = nearest_node
                        if eN is None:
                            found_vars = False
                            print("Element not found for:", i)
                        else:
                            # line below needs optimisation:
                            v_grad = pc.getGradientValue(eN, xi)
                            area = areas[eN]
                        if u_phi is not None and eN is not None:
                            # line below needs optimisation:
                            ls_phi = pc.getLevelSetValue(eN, xx[i])
                        else:
                            ls_phi = None
                    # line below needs optimisation:
                    f = pc.evaluateFunAtX(x=xx[i], ls_phi=ls_phi)
                    if found_vars:
                        for ndi in range(nd):
                            dphi[ndi] = v_grad[ndi]/(t*1./(f*pc.C)+(1-t)*1./area)
                    # # -----
                    # Euler
                    # # -----
                    if flag == 0:  # node in fluid domain
                        for ndi in range(nd):
                            xx[i, ndi] += dphi[ndi]*dt
                    elif fixed is False:  # slide along boundary
                        for ndi in range(nd):
                            xx[i, ndi] += dphi[ndi]*(1-np.abs(bNs[flag, ndi]))*dt
            t_last = t


    # def evaluateFunAtX(self, x, ls_phi=None):
    #     pc = self.pyCoefficients
    #     f = self.cppEvaluateFunAtX(x=x,
    #                                he_min=pc.he_min,
    #                                he_max=pc.he_max,
    #                                ls_phi=ls_phi,
    #                                fun=pc.fun,
    #                                t=pc.t)

    # cdef double cppEvaluateFunAtX(self,
    #                               double[:] x,
    #                               double he_min,
    #                               double he_max,
    #                               double ls_phi=None,
    #                               object fun=None,
    #                               double t=None):
    #     cdef double f
    #     if fun:
    #         f = fun(x, self.t)
    #     else:
    #         f = 0.
    #     if ls_phi is not None:
    #         f = min(abs(ls_phi), f)
    #     f = max(he_min, f)
    #     f = min(he_max, f)
    #     return f

    # def evaluateFunAtNodes(self):
    #     pc = self.pyCoefficients
    #     self.cppEvaluateFunAtNodes(nodeArray=pc.mesh.nodeArray,
    #                                uOfXTatNodes=pc.uOfXTatNodes,
    #                                he_min=pc.he_min,
    #                                he_max=pc.he_max,
    #                                fun=pc.func,
    #                                t=pc.t,
    #                                u_phi=pc.u_phi)

    # cdef cppEvaluateFunAtNodes(self,
    #                            double[:] nodeArray,
    #                            double[:] uOfXTatNodes,
    #                            double he_min,
    #                            double he_max,
    #                            object fun=None,
    #                            double t=None,
    #                            double[:] u_phi=None):
    #     cdef double f
    #     for i in range(len(self.mesh.nodeArray)):
    #         if fun:
    #             f = fun(nodeArray[i], t)
    #         else:
    #             f = 0.
    #         if u_phi is not None:
    #             f = min(abs(u_phi[i]), f)
    #         f = max(he_min, f)
    #         f = min(he_max, f)
    #         uOfXTatNodes[i] = f

    # def evaluateFunAtQuadraturePoints(self):
    #     pc = self.pyCoefficients
    #     self.cppEvaluateFunAtQuadraturePoints(qx=pc.model.q['x'],
    #                                           uOfXTatQuadrature=pc.uOfXTatQuadrature,
    #                                           he_min=pc.he_min,
    #                                           he_max=pc.he_max,
    #                                           q_phi=pc.q_phi,
    #                                           fun=pc.myfunc,
    #                                           t=pc.t)

    # cdef cppEvaluateFunAtQuadraturePoints(self,
    #                                       double[:,:,:] qx,
    #                                       double[:,:] uOfXTatQuadrature,
    #                                       double he_min,
    #                                       double he_max,
    #                                       double[:,:] q_phi=None,
    #                                       fun=None,
    #                                       t=None):
    #     cdef int N_k = qx.shape[1]
    #     cdef double f
    #     for e in xrange(len(qx)):
    #         for k in xrange(N_k):
    #             if fun:
    #                 f = fun(qx[e, k], self.t)
    #             if q_phi is not None:
    #                 f = min(abs(q_phi[e, k]), f)
    #             f = max(he_min, f)
    #             f = min(he_max, f)
    #             self.uOfXTatQuadrature[e, k] = f


def recoveryAtNodes(variable,
                    nodeElementsArray,
                    nodeElementOffsets):
    return cppRecoveryAtNodes(variable=variable,
                              nodeElementsArray=nodeElementsArray,
                              nodeElementOffsets=nodeElementOffsets)

cdef double[:] cppRecoveryAtNodes(double[:] variable,
                                  int[:] nodeElementsArray,
                                  int[:] nodeElementOffsets):
    """
    variable:
         Variable in element
    """
    cdef double[:] recovered_variable = np.zeros(len(nodeElementOffsets)-1)
    cdef int nb_el
    cdef double var_sum
    for node in range(len(nodeElementOffsets)-1):
        nb_el = 0
        var_sum = 0
        for eOffset in range(nodeElementOffsets[node],
                             nodeElementOffsets[node+1]):
            nb_el += 1
            var_sum += variable[nodeElementsArray[eOffset]]
        recovered_variable[node] = var_sum/nb_el
    return recovered_variable

def gradientRecoveryAtNodes(grads,
                            nodeElementsArray,
                            nodeElementOffsets,
                            nd):
    return cppGradientRecoveryAtNodes(grads=grads,
                                      nodeElementsArray=nodeElementsArray,
                                      nodeElementOffsets=nodeElementOffsets,
                                      nd=nd)

cdef double[:,:] cppGradientRecoveryAtNodes(double[:,:,:] grads,
                                            int[:] nodeElementsArray,
                                            int[:] nodeElementOffsets,
                                            int nd):
    cdef double[:, :] recovered_grads = np.zeros((len(nodeElementOffsets)-1, nd))
    cdef int nb_el
    cdef double[:] grad_sum = np.zeros(nd)
    for node in range(len(nodeElementOffsets)-1):
        nb_el = 0
        for ndi in range(nd):
            grad_sum[ndi] = 0.
        for eOffset in range(nodeElementOffsets[node],
                             nodeElementOffsets[node+1]):
            nb_el += 1
            eN = nodeElementsArray[eOffset]
            # for k in range(n_quad):
            #     grad_k = grads[eN, k]
            #     grad_eN_av += grad_k
            # grad_eN_av /= n_quad
            for ndi in range(nd):
                grad_sum[ndi] += grads[eN, 0, ndi]  # same value at all quad points
        for ndi in range(nd):
            recovered_grads[node, ndi] = grad_sum[ndi]/nb_el
    return recovered_grads


cdef tuple pyxSearchNearestNodeElementFromMeshObject(object mesh,
                                                     double[:] x,
                                                     int[:] exteriorElementBoundariesBoolArray,
                                                     double[:,:,:] elementNormalsArray,
                                                     int nearest_node,
                                                     int eN):
    n, d = ms.getLocalNearestNode(coords=x,
                                  nodeArray=mesh.nodeArray,
                                  nodeStarOffsets=mesh.nodeStarOffsets,
                                  nodeStarArray=mesh.nodeStarArray,
                                  node=nearest_node)
    n, d = ms.getLocalNearestElementAroundNode(coords=x,
                                               nodeElementOffsets=mesh.nodeElementOffsets,
                                               nodeElementsArray=mesh.nodeElementsArray,
                                               elementBarycenterArray=mesh.elementBarycentersArray,
                                               node=n)
    eN, d = ms.getLocalNearestElementNormal(coords=x,
                                            elementNormalsArray=elementNormalsArray,
                                            elementBoundariesArray=mesh.elementBoundariesArray,
                                            elementBarycentersArray=mesh.elementBoundaryBarycentersArray,
                                            elementNeighborsArray=mesh.elementNeighborsArray,
                                            elementBarycentersArray=mesh.elementBarycentersArray,
                                            exteriorElementBoundariesBoolArray=exteriorElementBoundariesBoolArray,
                                            eN=eN)
    return eN, n


cdef tuple pyxSearchNearestNodeElement(double[:] x,
                                       double[:,:] nodeArray,
                                       int[:] nodeStarOffsets,
                                       int[:] nodeStarArray,
                                       int[:] nodeElementOffsets,
                                       int[:] nodeElementsArray,
                                       double[:,:] elementBarycentersArray,
                                       int[:,:] elementNeighborsArray,
                                       int[:,:] elementBoundariesArray,
                                       double[:,:] elementBoundaryBarycentersArray,
                                       int[:] exteriorElementBoundariesBoolArray,
                                       double[:,:,:] elementNormalsArray,
                                       int nearest_node,
                                       int eN):
    n, d = ms.getLocalNearestNode(coords=x,
                                  nodeArray=nodeArray,
                                  nodeStarOffsets=nodeStarOffsets,
                                  nodeStarArray=nodeStarArray,
                                  node=nearest_node)
    eN, d = ms.getLocalNearestElementAroundNode(coords=x,
                                               nodeElementOffsets=nodeElementOffsets,
                                               nodeElementsArray=nodeElementsArray,
                                               elementBarycentersArray=elementBarycentersArray,
                                               node=n)
    eN, d = ms.getLocalNearestElementNormal(coords=x,
                                            elementNormalsArray=elementNormalsArray,
                                            elementBoundariesArray=elementBoundariesArray,
                                            elementBoundaryBarycentersArray=elementBoundaryBarycentersArray,
                                            elementNeighborsArray=elementNeighborsArray,
                                            elementBarycentersArray=elementBarycentersArray,
                                            exteriorElementBoundariesBoolArray=exteriorElementBoundariesBoolArray,
                                            eN=eN)
    return eN, n
