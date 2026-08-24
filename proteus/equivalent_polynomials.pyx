# A type of -*- python -*- file
import numpy as np
cimport numpy as np
cimport equivalent_polynomials as eqp
#Note: this Simplex class is for testing equivalent polynomials in python
#It uses a simplistic approach to dealing with the Simplex template
#It is not intended to take full advantage of the C++ implementation

cdef extern from *:
    ctypedef int nSpace1T "1"
    ctypedef int nSpace2T "2"
    ctypedef int nSpace3T "3"
    # nP_ifem is the NUMBER of IFEM basis functions per element, matching how ADR.h
    # instantiates it (nDOF_trial_element): 3 for P1 on a triangle, 6 for P2. The tests below
    # exercise only the H/ImH/D moment fit, so pass the P1 nodal count for each dimension
    # (nSpace+1) -- that is what sizes permutation/phi/nodes inside Simplex.
    ctypedef int nDOF1T "2"
    ctypedef int nDOF2T "3"
    ctypedef int nDOF3T "4"
    ctypedef int nP1T "1"
    ctypedef int nP2T "2"
    ctypedef int nP3T "3"
    ctypedef int nP4T "4"
    ctypedef int nQT "50"
    ctypedef int nEBQT "50"
#nQ=50 will provide enough space for testing most quadrature rules
#but will be slow


cdef class Simplex:
    cdef np.ndarray xiBuffer
    cdef int nSpace
    cdef int nP
    cdef int nQ
    cdef int nEBQ
    cdef int q
    cdef bool inside_out
    cdef int edge
    cdef int corner
    cdef np.ndarray _H
    cdef np.ndarray _ImH
    cdef np.ndarray _D
    #instatiate template classes for 1,2,3D and P1-P3
    cdef eqp.cSimplex[nSpace1T,nDOF1T,nP1T,nQT,nEBQT] s11
    cdef eqp.cSimplex[nSpace1T,nDOF1T,nP2T,nQT,nEBQT] s12
    cdef eqp.cSimplex[nSpace1T,nDOF1T,nP3T,nQT,nEBQT] s13
    cdef eqp.cSimplex[nSpace2T,nDOF2T,nP1T,nQT,nEBQT] s21
    cdef eqp.cSimplex[nSpace2T,nDOF2T,nP2T,nQT,nEBQT] s22
    cdef eqp.cSimplex[nSpace2T,nDOF2T,nP3T,nQT,nEBQT] s23
    cdef eqp.cSimplex[nSpace3T,nDOF3T,nP1T,nQT,nEBQT] s31
    cdef eqp.cSimplex[nSpace3T,nDOF3T,nP2T,nQT,nEBQT] s32
    cdef eqp.cSimplex[nSpace3T,nDOF3T,nP3T,nQT,nEBQT] s33
    def __cinit__(self, nSpace, nP, nQ):
        self.xiBuffer=np.zeros((50,3),'d')
        self.nSpace = nSpace
        self.nP = nP
        self.nQ=nQ
        self.nEBQ=nQ#cek hack
        self.q=0
    def calculate(self, np.ndarray phi_dof, np.ndarray phi_nodes, np.ndarray xi):
        self.xiBuffer[:xi.shape[0]]=xi
        if (self.nSpace,self.nP) == (1,1):
            icase = self.s11.calculate(<double*>(phi_dof.data), <double*>(phi_nodes.data), <double*>(self.xiBuffer.data),False)
            self._H = np.asarray(<double[:self.nQ]>self.s11.get_H())
            self._ImH = np.asarray(<double[:self.nQ]>self.s11.get_ImH())
            self._D = np.asarray(<double[:self.nQ]>self.s11.get_D())
            self.inside_out = self.s11.inside_out
            self.edge = self.s11.edge
            self.corner = self.s11.corner
        elif (self.nSpace,self.nP) == (1,2):
            icase = self.s12.calculate(<double*>(phi_dof.data), <double*>(phi_nodes.data), <double*>(self.xiBuffer.data),False)
            self._H = np.asarray(<double[:self.nQ]>self.s12.get_H())
            self._ImH = np.asarray(<double[:self.nQ]>self.s12.get_ImH())
            self._D = np.asarray(<double[:self.nQ]>self.s12.get_D())
            self.inside_out = self.s12.inside_out
            self.edge = self.s12.edge
            self.corner = self.s12.corner
        elif (self.nSpace,self.nP) == (1,3):
            icase = self.s13.calculate(<double*>(phi_dof.data), <double*>(phi_nodes.data), <double*>(self.xiBuffer.data),False)
            self._H = np.asarray(<double[:self.nQ]>self.s13.get_H())
            self._ImH = np.asarray(<double[:self.nQ]>self.s13.get_ImH())
            self._D = np.asarray(<double[:self.nQ]>self.s13.get_D())
            self.inside_out = self.s13.inside_out
            self.edge = self.s13.edge
            self.corner = self.s13.corner
        elif (self.nSpace,self.nP) == (2,1):
            icase = self.s21.calculate(<double*>(phi_dof.data), <double*>(phi_nodes.data), <double*>(self.xiBuffer.data),False)
            self._H = np.asarray(<double[:self.nQ]>self.s21.get_H())
            self._ImH = np.asarray(<double[:self.nQ]>self.s21.get_ImH())
            self._D = np.asarray(<double[:self.nQ]>self.s21.get_D())
            self.inside_out = self.s21.inside_out
            self.edge = self.s21.edge
            self.corner = self.s21.corner
        elif (self.nSpace,self.nP) == (2,2):
            icase = self.s22.calculate(<double*>(phi_dof.data), <double*>(phi_nodes.data), <double*>(self.xiBuffer.data),False)
            self._H = np.asarray(<double[:self.nQ]>self.s22.get_H())
            self._ImH = np.asarray(<double[:self.nQ]>self.s22.get_ImH())
            self._D = np.asarray(<double[:self.nQ]>self.s22.get_D())
            self.inside_out = self.s22.inside_out
            self.edge = self.s22.edge
            self.corner = self.s22.corner
        elif (self.nSpace,self.nP) == (2,3):
            icase = self.s23.calculate(<double*>(phi_dof.data), <double*>(phi_nodes.data), <double*>(self.xiBuffer.data),False)
            self._H = np.asarray(<double[:self.nQ]>self.s23.get_H())
            self._ImH = np.asarray(<double[:self.nQ]>self.s23.get_ImH())
            self._D = np.asarray(<double[:self.nQ]>self.s23.get_D())
            self.inside_out = self.s23.inside_out
            self.edge = self.s23.edge
            self.corner = self.s23.corner
        if (self.nSpace,self.nP) == (3,1):
            icase = self.s31.calculate(<double*>(phi_dof.data), <double*>(phi_nodes.data), <double*>(self.xiBuffer.data),False)
            self._H = np.asarray(<double[:self.nQ]>self.s31.get_H())
            self._ImH = np.asarray(<double[:self.nQ]>self.s31.get_ImH())
            self._D = np.asarray(<double[:self.nQ]>self.s31.get_D())
            self.inside_out = self.s31.inside_out
            self.edge = self.s31.edge
            self.corner = self.s31.corner
        elif (self.nSpace,self.nP) == (3,2):
            icase = self.s32.calculate(<double*>(phi_dof.data), <double*>(phi_nodes.data), <double*>(self.xiBuffer.data),False)
            self._H = np.asarray(<double[:self.nQ]>self.s32.get_H())
            self._ImH = np.asarray(<double[:self.nQ]>self.s32.get_ImH())
            self._D = np.asarray(<double[:self.nQ]>self.s32.get_D())
            self.inside_out = self.s32.inside_out
            self.edge = self.s32.edge
            self.corner = self.s32.corner
        elif (self.nSpace,self.nP) == (3,3):
            icase = self.s33.calculate(<double*>(phi_dof.data), <double*>(phi_nodes.data), <double*>(self.xiBuffer.data),False)
            self._H = np.asarray(<double[:self.nQ]>self.s33.get_H())
            self._ImH = np.asarray(<double[:self.nQ]>self.s33.get_ImH())
            self._D = np.asarray(<double[:self.nQ]>self.s33.get_D())
            self.inside_out = self.s33.inside_out
            self.edge = self.s33.edge
            self.corner = self.s33.corner
    def set_quad(self, int q):
        self.q=q
    # Mirrors Simplex::set_quad() in equivalent_polynomials.h, including its precedence:
    # the edge/corner degenerate cases are resolved BEFORE inside_out. Keying off inside_out
    # alone (as this wrapper used to) reports H and ImH exchanged whenever a cell is flagged
    # corner==1, which also sets inside_out for the P2 basis solve.
    @property
    def H(self):
        if self.edge == -1 or self.corner == -1:
            return 0.0
        elif self.edge == 1 or self.corner == 1:
            return 1.0
        elif self.inside_out:
            return self._ImH[self.q]
        else:
            return self._H[self.q]
    @property
    def ImH(self):
        if self.edge == -1 or self.corner == -1:
            return 1.0
        elif self.edge == 1 or self.corner == 1:
            return 0.0
        elif self.inside_out:
            return self._H[self.q]
        else:
            return self._ImH[self.q]
    @property
    def D(self):
        # NOT zeroed for edge/corner: when the interface lies along an element edge it has real
        # nonzero measure (the edge length), which the moment fit already carries in _D.
        return self._D[self.q]

def calc_edge_H(double phi0, double phi1, int nP):
    """Edge-restricted moment-fit Heaviside on a cut edge (reference t in [0,1] from the edge's
    node 0 to node 1). phi0, phi1 must have opposite signs. Returns the monomial coefficients
    of H_hat in {1, t, ..., t**nP}, fitting the indicator of {phi > 0}."""
    cdef np.ndarray C_H = np.zeros(nP+1)
    if nP == 1:   eqp.calculate_edge_H[nP1T](phi0, phi1, <double*>C_H.data)
    elif nP == 2: eqp.calculate_edge_H[nP2T](phi0, phi1, <double*>C_H.data)
    elif nP == 3: eqp.calculate_edge_H[nP3T](phi0, phi1, <double*>C_H.data)
    elif nP == 4: eqp.calculate_edge_H[nP4T](phi0, phi1, <double*>C_H.data)
    else: raise ValueError("nP must be 1, 2, 3, or 4")
    return C_H

def eval_edge_poly(np.ndarray C, double t, int nP):
    if nP == 1:   return eqp.evaluate_edge_poly[nP1T](<double*>C.data, t)
    elif nP == 2: return eqp.evaluate_edge_poly[nP2T](<double*>C.data, t)
    elif nP == 3: return eqp.evaluate_edge_poly[nP3T](<double*>C.data, t)
    elif nP == 4: return eqp.evaluate_edge_poly[nP4T](<double*>C.data, t)
    else: raise ValueError("nP must be 1, 2, 3, or 4")
