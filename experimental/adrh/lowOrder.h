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
// FCT (below): the limiter on top. The conserved variable is m (m = u for a linear storage term; the moisture content for Richards), v the variable
// the artificial diffusion acts on (u). With m_L, v_L the low-order solution, delta = m_L - m_old, M_C the consistent and M_L the lumped
// mass, the Galerkin residual at the low-order solution is a sum of antisymmetric edge fluxes,
//      f_ij = M_C,ij (delta_j - delta_i) / dt + d_ij (v_L,j - v_L,i),     f_ji = -f_ij,
// and the corrected conserved variable is m_i = m_L,i + (dt / M_L,i) sum_j alpha_ij g_ij, g_ij = -f_ij, with limiters 0 <= alpha_ij = alpha_ji <= 1
// (Zalesak) chosen so that m_i stays between the smallest and largest value of m_L and m_old over node i and its neighbours. The caller maps m back to u. A symmetric alpha
// keeps the mass sum_i M_L,i u_i exactly; alpha = 1 everywhere is one linearized step toward the Galerkin solution. Three row-parallel passes.
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

  // ---- the consistent mass matrix ---------------------------------------------------------------------------------

  // One element: Mloc (nDOF x nDOF), the integral of w_a w_b over the element. The unit mass matrix of the conserved variable: the FCT
  // edge fluxes use it whatever the storage function m(u) is.
  template <class Space>
  ADRH_FN ADRH_FORCE_INLINE void massMatrixOne(const double *X, int nQ, const double *qp, const double *qw, double *Mloc)
  {
    constexpr int nS = Space::nS, nD = Space::nDOF;
    PointGeometry<Space> G;
    for (int k = 0; k < nD * nD; ++k) Mloc[k] = 0.0;
    for (int q = 0; q < nQ; ++q)
    {
      G.compute(X, qp + q * nS);
      const double dV = qw[q] * G.adet;
      for (int a = 0; a < nD; ++a)
        for (int b = 0; b < nD; ++b) Mloc[a * nD + b] += dV * G.w[a] * G.w[b];
    }
  }

  template <class Space>
  void massMatrixKernel(int nE, int nQ, const double *xn, const double *qp, const double *qw, double *M)
  {
    constexpr int nD = Space::nDOF;
    for (int e = 0; e < nE; ++e) massMatrixOne<Space>(xn + e * nD * 3, nQ, qp, qw, M + e * nD * nD);
  }

  // ---- flux-corrected transport -----------------------------------------------------------------------------------

  // pass 1, row i: g_ij = -f_ij for the entries with j > i, written to both (i, j) and, negated, (j, i): the pair is computed once, so the
  // antisymmetry is exact; each location is written once (no race)
  ADRH_FN void fctFluxRow(int i, const int *rowptr, const int *colidx, const int *tpos, const double *Mc, const double *d, const double *vL,
                          const double *mL, const double *mold, double dt, double *G)
  {
    const double di = mL[i] - mold[i];
    for (int p = rowptr[i]; p < rowptr[i + 1]; ++p)
    {
      const int j = colidx[p];
      if (j < i) continue;
      if (j == i) { G[p] = 0.0; continue; }
      const double dj = mL[j] - mold[j];
      const double f = Mc[p] * (dj - di) / dt + d[p] * (vL[j] - vL[i]);
      G[p] = -f;
      G[tpos[p]] = f;
    }
  }

  // pass 2, row i: the Zalesak ratios R+_i, R-_i from the sums of the incoming positive and negative fluxes and the room to the local extrema
  ADRH_FN void fctBoundsRow(int i, const int *rowptr, const int *colidx, const double *G, const double *mL, const double *mold,
                            const double *ML, double dt, double *Rp, double *Rm)
  {
    double Pp = 0.0, Pm = 0.0;
    double umax = mL[i] > mold[i] ? mL[i] : mold[i], umin = mL[i] < mold[i] ? mL[i] : mold[i];
    for (int p = rowptr[i]; p < rowptr[i + 1]; ++p)
    {
      const int j = colidx[p];
      const double g = G[p];
      if (g > 0.0) Pp += g; else Pm += g;
      const double hi = mL[j] > mold[j] ? mL[j] : mold[j], lo = mL[j] < mold[j] ? mL[j] : mold[j];
      if (hi > umax) umax = hi;
      if (lo < umin) umin = lo;
    }
    const double Qp = ML[i] / dt * (umax - mL[i]), Qm = ML[i] / dt * (umin - mL[i]);
    const double rp = Pp > 0.0 ? Qp / Pp : 1.0, rm = Pm < 0.0 ? Qm / Pm : 1.0;
    Rp[i] = rp < 1.0 ? rp : 1.0;
    Rm[i] = rm < 1.0 ? rm : 1.0;
  }

  // pass 3, row i: the limited update. alpha_ij = min(R+_i, R-_j) for a flux into i, min(R-_i, R+_j) for one out of i: symmetric in (i, j).
  ADRH_FN void fctApplyRow(int i, const int *rowptr, const int *colidx, const double *G, const double *Rp, const double *Rm,
                           const double *ML, const double *mL, double dt, double *mOut)
  {
    double sum = 0.0;
    for (int p = rowptr[i]; p < rowptr[i + 1]; ++p)
    {
      const int j = colidx[p];
      const double g = G[p];
      if (g > 0.0) { const double a = Rp[i] < Rm[j] ? Rp[i] : Rm[j]; sum += a * g; }
      else if (g < 0.0) { const double a = Rm[i] < Rp[j] ? Rm[i] : Rp[j]; sum += a * g; }
    }
    mOut[i] = mL[i] + dt / ML[i] * sum;
  }

  // the dissipation values of a Jacobian (pass 1 of the discrete upwinding on its own): d on the pattern
  inline void dissipationValues(int nRows, const int *rowptr, const int *colidx, const int *tpos, const double *Jv, double *d)
  {
    for (int i = 0; i < nRows; ++i)
      for (int p = rowptr[i]; p < rowptr[i + 1]; ++p) d[p] = dissipationEntry(i, p, colidx, tpos, Jv);
  }

  // host reference of the three passes: m_out from the low-order state (v_L for the dissipation, m_L and m_old the conserved variable)
  inline void fctLimit(int nRows, const int *rowptr, const int *colidx, const int *tpos, const double *Mc, const double *d, const double *ML,
                       const double *vL, const double *mL, const double *mold, double dt, double *mOut)
  {
    std::vector<double> G(rowptr[nRows]), Rp(nRows), Rm(nRows);
    for (int i = 0; i < nRows; ++i) fctFluxRow(i, rowptr, colidx, tpos, Mc, d, vL, mL, mold, dt, G.data());
    for (int i = 0; i < nRows; ++i) fctBoundsRow(i, rowptr, colidx, G.data(), mL, mold, ML, dt, Rp.data(), Rm.data());
    for (int i = 0; i < nRows; ++i) fctApplyRow(i, rowptr, colidx, G.data(), Rp.data(), Rm.data(), ML, mL, dt, mOut);
  }

  // ---- the K-upwinded low-order operator (Richards' equation in head form) ----------------------------------------------

  // The Galerkin discretization of a strongly nonlinear conductivity (Richards in a dry medium: K_r falls by ten orders of magnitude over one element)
  // takes K from the quadrature points, where the dry side dominates, so a wetting front cannot enter the element: the discrete flux is
  // ~1e-10 of the true one. The cure used by proteus's low-order and FCT schemes is to write the flux on the total head phi = psi - g.x
  // (gravity included) and to take the conductivity at the UPWIND node,
  //      flux into i from j:   F_ij = kr_up c_ij (phi_j - phi_i),   c_ij = max(0, -S_ij),   up = i if (-S_ij (phi_j - phi_i) <= 0), else j,
  // S the stiffness K_s grad phi_i . grad phi_j (conductivity 1 relative; the matrix of the element kernel in mode 1). Row sums vanish and
  // the same node is upwind from either side of an edge, so the scheme is conservative; with the lumped mass it is an M-matrix.
  // Row i: r_i = -sum_j F_ij and the Jacobian of that (kr, dkr the nodal relative permeability and its derivative with respect to u; phi = u + const).
  ADRH_FN void upwindFluxRow(int i, const int *rowptr, const int *colidx, const double *S, const double *phi, const double *kr, const double *dkr,
                             double *r, double *J)
  {
    double rsum = 0.0, Jii = 0.0;
    int ii = -1;
    for (int p = rowptr[i]; p < rowptr[i + 1]; ++p)
    {
      const int j = colidx[p];
      if (j == i) { ii = p; continue; }
      const double c = S[p] < 0.0 ? -S[p] : 0.0, dphi = phi[j] - phi[i];
      double F, dFi, dFj;
      if (-S[p] * dphi <= 0.0) { F = kr[i] * c * dphi; dFi = dkr[i] * c * dphi - kr[i] * c; dFj = kr[i] * c; }
      else { F = kr[j] * c * dphi; dFi = -kr[j] * c; dFj = dkr[j] * c * dphi + kr[j] * c; }
      rsum -= F;
      J[p] = -dFj;
      Jii -= dFi;
    }
    r[i] = rsum;
    if (ii >= 0) J[ii] = Jii;
  }

  // the Galerkin flux (frozen stiffness C, the matrix of mode 2) minus the K-upwinded one, into i from j: antisymmetric, the antidiffusive flux of FCT
  ADRH_FN void antidiffusiveFluxRow(int i, const int *rowptr, const int *colidx, const double *C, const double *S, const double *phi,
                                    const double *kr, double *A)
  {
    for (int p = rowptr[i]; p < rowptr[i + 1]; ++p)
    {
      const int j = colidx[p];
      if (j == i) { A[p] = 0.0; continue; }
      const double c = S[p] < 0.0 ? -S[p] : 0.0, dphi = phi[j] - phi[i];
      const double krup = (-S[p] * dphi <= 0.0) ? kr[i] : kr[j];
      A[p] = (-C[p] - krup * c) * dphi;
    }
  }

  // pass 1 of FCT with a given antidiffusive flux A (into i from j): g_ij = -M_C,ij (delta_j - delta_i) / dt + A_ij, as fctFluxRow
  ADRH_FN void fctFluxRowA(int i, const int *rowptr, const int *colidx, const int *tpos, const double *Mc, const double *A, const double *mL,
                           const double *mold, double dt, double *G)
  {
    const double di = mL[i] - mold[i];
    for (int p = rowptr[i]; p < rowptr[i + 1]; ++p)
    {
      const int j = colidx[p];
      if (j < i) continue;
      if (j == i) { G[p] = 0.0; continue; }
      const double dj = mL[j] - mold[j];
      const double g = -Mc[p] * (dj - di) / dt + A[p];
      G[p] = g;
      G[tpos[p]] = -g;
    }
  }

  // FCT bounds from the HEAD, for a medium whose nodes have different retention curves: the local extrema of the moisture content over a node and its
  // neighbours are not comparable across soils (the same head is a different moisture content), so the limiter bounds the head, which is continuous
  // across a soil interface, and maps the bounds to moisture content with the node's own curve, m_i(max_j psi_j) and m_i(min_j psi_j). For one material
  // this is the bound of fctBoundsRow (m is increasing in psi). headExtremaRow: the extrema over row i of two states a and b (the low-order and old
  // heads; pass the same array twice to bound over one state alone).
  ADRH_FN void headExtremaRow(int i, const int *rowptr, const int *colidx, const double *a, const double *b, double *pmax, double *pmin)
  {
    double hi = a[i] > b[i] ? a[i] : b[i], lo = a[i] < b[i] ? a[i] : b[i];
    for (int p = rowptr[i]; p < rowptr[i + 1]; ++p)
    {
      const int j = colidx[p];
      const double h = a[j] > b[j] ? a[j] : b[j], l = a[j] < b[j] ? a[j] : b[j];
      if (h > hi) hi = h;
      if (l < lo) lo = l;
    }
    pmax[i] = hi; pmin[i] = lo;
  }

  // pass 2 of FCT with the bounds mmax_i, mmin_i given (as fctBoundsRow, which takes them over mL and mold)
  ADRH_FN void fctBoundsRowGiven(int i, const int *rowptr, const int *colidx, const double *G, const double *mL, const double *mmax, const double *mmin,
                                 const double *ML, double dt, double *Rp, double *Rm)
  {
    double Pp = 0.0, Pm = 0.0;
    for (int p = rowptr[i]; p < rowptr[i + 1]; ++p)
    {
      const double g = G[p];
      if (g > 0.0) Pp += g; else Pm += g;
    }
    const double Qp = ML[i] / dt * (mmax[i] - mL[i]), Qm = ML[i] / dt * (mmin[i] - mL[i]);
    const double rp = Pp > 0.0 ? Qp / Pp : 1.0, rm = Pm < 0.0 ? Qm / Pm : 1.0;
    Rp[i] = rp < 1.0 ? rp : 1.0;
    Rm[i] = rm < 1.0 ? rm : 1.0;
  }

  // the host reference: fctLimitFlux with the bounds given
  inline void fctLimitFluxBounded(int nRows, const int *rowptr, const int *colidx, const int *tpos, const double *Mc, const double *A, const double *ML,
                                  const double *mL, const double *moldFlux, const double *mmax, const double *mmin, double dt, double *mOut)
  {
    std::vector<double> G(rowptr[nRows]), Rp(nRows), Rm(nRows);
    for (int i = 0; i < nRows; ++i) fctFluxRowA(i, rowptr, colidx, tpos, Mc, A, mL, moldFlux, dt, G.data());
    for (int i = 0; i < nRows; ++i) fctBoundsRowGiven(i, rowptr, colidx, G.data(), mL, mmax, mmin, ML, dt, Rp.data(), Rm.data());
    for (int i = 0; i < nRows; ++i) fctApplyRow(i, rowptr, colidx, G.data(), Rp.data(), Rm.data(), ML, mL, dt, mOut);
  }

  // moldFlux: the old state the flux pass differences mL against (delta = mL - moldFlux); moldBounds: the one the local extrema are taken over
  // together with mL (pass mL itself to take them over the low-order state alone, as proteus's non-monolithic FCTStep does)
  inline void fctLimitFlux(int nRows, const int *rowptr, const int *colidx, const int *tpos, const double *Mc, const double *A, const double *ML,
                           const double *mL, const double *moldFlux, const double *moldBounds, double dt, double *mOut)
  {
    std::vector<double> G(rowptr[nRows]), Rp(nRows), Rm(nRows);
    for (int i = 0; i < nRows; ++i) fctFluxRowA(i, rowptr, colidx, tpos, Mc, A, mL, moldFlux, dt, G.data());
    for (int i = 0; i < nRows; ++i) fctBoundsRow(i, rowptr, colidx, G.data(), mL, moldBounds, ML, dt, Rp.data(), Rm.data());
    for (int i = 0; i < nRows; ++i) fctApplyRow(i, rowptr, colidx, G.data(), Rp.data(), Rm.data(), ML, mL, dt, mOut);
  }
} // namespace adrh
