// Algebraic stabilization: a lumped mass matrix and discrete upwinding of the assembled spatial operator.
//
// The Galerkin form with a consistent mass matrix is not monotone: a sharp front (advection-dominated transport, Richards' equation in
// a dry medium) overshoots and undershoots. The classical cure that needs only the assembled matrix, and so works for any coefficient
// functor, is two things:
//   1. mass lumping: the mass term at each node is M_L,i (m(u_i) - m(u_old,i)) / dt, with M_L,i the integral of the basis function i;
//   2. discrete upwinding: add to the spatial operator the symmetric artificial diffusion
//          d_ij = max(0, J_ij, J_ji)   (i != j),     r_i += sum_j d_ij (u_i - u_j),     J_ij -= d_ij,  J_ii += sum_j d_ij,
//      with J the Jacobian of the spatial residual (the "sign check on the off-diagonals of the whole operator"). The result has
//      non-positive off-diagonals, so with the lumped mass it is an M-matrix and the scheme satisfies a discrete maximum principle.
//      It is the low-order scheme of flux-corrected transport and of monolithic limiting: the antidiffusive correction a later
//      limiter adds is the difference between this operator and the Galerkin one, edge by edge, over the same pattern and
//      the same transpose map built here.
// The artificial diffusion has zero row sums, so a constant is still a solution. D depends on the Jacobian at the current iterate and
// is held fixed in the Jacobian of the stabilized system (an approximate Newton step: it converges, not quadratically).
//
// One component (nc = 1), the P1 and Q1 spaces where the row-sum lumped mass is positive, and the CSR pattern of a scalar unknown with
// sorted columns and a structurally symmetric pattern.
#pragma once
#include <algorithm>
#include <stdexcept>
#include <vector>
#include "engine.h"

namespace adrh
{
  // ---- the pattern -----------------------------------------------------------------------------------------------

  // tpos[p] is the position of entry (j, i) for the entry p = (i, j): the transpose map. Throws if the pattern is not symmetric.
  inline std::vector<int> buildTransposeMap(int nRows, const int *rowptr, const int *colidx)
  {
    std::vector<int> t(rowptr[nRows]);
    for (int i = 0; i < nRows; ++i)
      for (int p = rowptr[i]; p < rowptr[i + 1]; ++p)
      {
        const int j = colidx[p];
        const int *lo = colidx + rowptr[j], *hi = colidx + rowptr[j + 1];
        const int *q = std::lower_bound(lo, hi, i);
        if (q == hi || *q != i) throw std::runtime_error("the sparsity pattern is not structurally symmetric");
        t[p] = (int)(q - colidx);
      }
    return t;
  }

  // ---- discrete upwinding ----------------------------------------------------------------------------------------

  // pass 1, one entry: the artificial diffusion d_p (zero on the diagonal)
  ADRH_FN double dissipationEntry(int i, int p, const int *colidx, const int *tpos, const double *Jv)
  {
    if (colidx[p] == i) return 0.0;
    const double a = Jv[p], b = Jv[tpos[p]];
    const double m = a > b ? a : b;
    return m > 0.0 ? m : 0.0;
  }

  // pass 2, one row: add the diffusion to the residual and the (frozen-D) Jacobian. Touches only row i.
  ADRH_FN void applyDissipationRow(int i, const int *rowptr, const int *colidx, const double *d, double *Jv, const double *u,
                                          double *r)
  {
    double sum = 0.0, add = 0.0;
    int diag = -1;
    for (int p = rowptr[i]; p < rowptr[i + 1]; ++p)
    {
      const int j = colidx[p];
      if (j == i) { diag = p; continue; }
      add += d[p] * (u[i] - u[j]);
      sum += d[p];
      Jv[p] -= d[p];
    }
    r[i] += add;
    if (diag >= 0) Jv[diag] += sum;
  }

  // host reference: both passes in place on the residual r and the values Jv
  inline void algebraicDissipation(int nRows, const int *rowptr, const int *colidx, const int *tpos, double *Jv, const double *u, double *r)
  {
    std::vector<double> d(rowptr[nRows]);
    for (int i = 0; i < nRows; ++i)
      for (int p = rowptr[i]; p < rowptr[i + 1]; ++p) d[p] = dissipationEntry(i, p, colidx, tpos, Jv);
    for (int i = 0; i < nRows; ++i) applyDissipationRow(i, rowptr, colidx, d.data(), Jv, u, r);
  }

  // ---- the lumped mass -------------------------------------------------------------------------------------------

  // One element: w[a] the integral of basis a over the element, and the coefficient functor's mass m and its derivative dm at each node
  // value (ue), and the old-time mass mold (zero for a steady problem). The functor's m must depend on u, x and t only, as a
  // storage term does; its gradient argument is passed as zero.
  template <class Coefficients, class Space>
  ADRH_FN ADRH_FORCE_INLINE void lumpedMassOneBound(const Coefficients &C, const double *X, const double *ue, const double *uo, int nQ,
                                                   const double *qp, const double *qw, double t, double dt, double *w, double *m,
                                                   double *dm, double *mold)
  {
    using Co = Coefficients;
    constexpr int nS = Space::nS, nD = Space::nDOF;
    static_assert(Co::nc == 1, "algebraic stabilization is for a scalar unknown");
    PointGeometry<Space> G;
    for (int a = 0; a < nD; ++a) w[a] = 0.0;
    for (int q = 0; q < nQ; ++q)
    {
      G.compute(X, qp + q * nS);
      const double dV = qw[q] * G.adet;
      for (int a = 0; a < nD; ++a) w[a] += dV * G.w[a];
    }
    Terms<1, nS> T;
    double gu0[1][nS] = {};
    for (int a = 0; a < nD; ++a)
    {
      m[a] = 0.0; dm[a] = 0.0; mold[a] = 0.0;
      if constexpr (Co::has_mass)
      {
        zeroMasked<Co>(T);
        C.evaluate(t, X + a * 3, ue + a, gu0, T);
        m[a] = T.m[0]; dm[a] = T.dm[0][0];
        if (uo && dt > 0.0)
        {
          zeroMasked<Co>(T);
          C.evaluate(t - dt, X + a * 3, uo + a, gu0, T);
          mold[a] = T.m[0];
        }
      }
    }
  }

  template <class Coefficients, class Space>
  ADRH_FN ADRH_FORCE_INLINE void lumpedMassOne(const Coefficients &C, const double *X, const double *ue, const double *uo, int nQ,
                                              const double *qp, const double *qw, double t, double dt, double *w, double *m, double *dm,
                                              double *mold, int eN = 0)
  {
    if constexpr (HasBindElement<Coefficients>::value)
    {
      const auto Cb = C.bindElement(eN);
      lumpedMassOneBound<std::decay_t<decltype(Cb)>, Space>(Cb, X, ue, uo, nQ, qp, qw, t, dt, w, m, dm, mold);
    }
    else
      lumpedMassOneBound<Coefficients, Space>(C, X, ue, uo, nQ, qp, qw, t, dt, w, m, dm, mold);
  }

  template <class Coefficients, class Space>
  void lumpedMassKernel(const Coefficients &C, int nE, int nQ, const double *xn, const double *u, const double *uold, const double *qp,
                        const double *qw, double t, double dt, double *w, double *m, double *dm, double *mold)
  {
    constexpr int nD = Space::nDOF;
    for (int e = 0; e < nE; ++e)
      lumpedMassOne<Coefficients, Space>(C, xn + e * nD * 3, u + e * nD, uold ? uold + e * nD : nullptr, nQ, qp, qw, t, dt, w + e * nD,
                                         m + e * nD, dm + e * nD, mold + e * nD, e);
  }
} // namespace adrh
