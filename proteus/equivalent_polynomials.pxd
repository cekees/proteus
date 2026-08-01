# A type of -*- python -*- file
# cython: language_level=3
from libcpp cimport bool
cdef extern from "equivalent_polynomials.h" namespace "equivalent_polynomials":
    cdef void calculate_edge_H "equivalent_polynomials::calculate_edge_H"[nP](double phi0, double phi1, double* C_H)
    cdef double evaluate_edge_poly "equivalent_polynomials::evaluate_edge_poly"[nP](double* C, double t)
    cdef cppclass cSimplex "equivalent_polynomials::Simplex"[nSpace,nP_ifem,nP,nQ,nEBQ]:
      cSimplex "Simplex"()
      int calculate(double* phi_dof, double* phi_nodes, double* xi_r, bool isBoundary);
      double* get_H()
      double* get_ImH()
      double* get_D()
      bool inside_out
