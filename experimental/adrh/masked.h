// The element kernel with the coefficient functor's sparsity as compile-time constants.
//
// A functor declares which terms and which derivative blocks can be nonzero as static constexpr
// predicates (DenseMasks below says "all of them"; a functor overrides the ones it knows). The
// loops over components are unrolled at compile time with constexpr_for, and `if constexpr` drops
// every block whose mask is false, so nothing of it is stored, zeroed, multiplied or added.
// The storage stays the dense Terms of adrh.h; only blocks the masks allow are ever touched.
#pragma once
#include <type_traits>
#include "adrh.h"

namespace adrh
{
  template <int B, int E, class F>
  ADRH_FN ADRH_FORCE_INLINE void constexpr_for(F &&f)
  {
    if constexpr (B < E)
    {
      f(std::integral_constant<int, B>{});
      constexpr_for<B + 1, E>(f);
    }
  }

  // Everything is possibly nonzero. A functor inherits this and redefines what it knows is zero.
  struct DenseMasks
  {
    static constexpr bool has_mass = true, has_advection = true, has_diffusion = true,
                          has_hamiltonian = true, has_reaction = true;
    static constexpr bool m_nz(int) { return true; }
    static constexpr bool dm_nz(int, int) { return true; }
    static constexpr bool f_nz(int) { return true; }
    static constexpr bool df_nz(int, int) { return true; }
    static constexpr bool a_nz(int, int) { return true; }          // a^{ik}
    static constexpr bool da_nz(int, int, int) { return true; }    // d a^{ik} / d u_j
    static constexpr bool dphi_nz(int, int) { return true; }       // d phi^k / d u_j
    static constexpr bool H_nz(int) { return true; }
    static constexpr bool dH_nz(int, int) { return true; }
    static constexpr bool r_nz(int) { return true; }
    static constexpr bool dr_nz(int, int) { return true; }
  };

  // zero only the entries of T that the masks allow the functor to write (the rest is never read)
  template <class C>
  ADRH_FN ADRH_FORCE_INLINE void zeroMasked(Terms<C::nc, C::nSpace> &T)
  {
    constexpr int nc = C::nc, nS = C::nSpace;
    constexpr_for<0, nc>([&](auto i_) ADRH_LAMBDA_INLINE {
      constexpr int i = decltype(i_)::value;
      if constexpr (C::has_mass && C::m_nz(i)) T.m[i] = 0.0;
      if constexpr (C::has_reaction && C::r_nz(i)) T.r[i] = 0.0;
      if constexpr (C::has_hamiltonian && C::H_nz(i)) T.H[i] = 0.0;
      if constexpr (C::has_advection && C::f_nz(i))
        for (int I = 0; I < nS; ++I) T.f[i][I] = 0.0;
      constexpr_for<0, nc>([&](auto j_) ADRH_LAMBDA_INLINE {
        constexpr int j = decltype(j_)::value;
        if constexpr (C::has_mass && C::dm_nz(i, j)) T.dm[i][j] = 0.0;
        if constexpr (C::has_reaction && C::dr_nz(i, j)) T.dr[i][j] = 0.0;
        if constexpr (C::has_diffusion && C::dphi_nz(i, j)) T.dphi[i][j] = 0.0;
        if constexpr (C::has_advection && C::df_nz(i, j))
          for (int I = 0; I < nS; ++I) T.df[i][j][I] = 0.0;
        if constexpr (C::has_hamiltonian && C::dH_nz(i, j))
          for (int I = 0; I < nS; ++I) T.dH[i][j][I] = 0.0;
        if constexpr (C::has_diffusion && C::a_nz(i, j))
          for (int I = 0; I < nS; ++I)
            for (int J = 0; J < nS; ++J) T.a[i][j][I][J] = 0.0;
        constexpr_for<0, nc>([&](auto k_) ADRH_LAMBDA_INLINE {
          constexpr int k = decltype(k_)::value;
          if constexpr (C::has_diffusion && C::da_nz(i, j, k))
            for (int I = 0; I < nS; ++I)
              for (int J = 0; J < nS; ++J) T.da[i][j][k][I][J] = 0.0;
        });
      });
    });
  }

  // One element: R (nc, nS+1) and J (nc, nS+1, nc, nS+1) from the node coordinates X (nS+1, 3), the DOF values ue and
  // uo (nc, nS+1; uo may be null for a steady problem) and the reference quadrature. This is the function a
  // Kokkos kernel calls per element (kokkos_kernel.h); elementKernelMasked below is the loop over it.
  // NQ > 0 fixes the number of quadrature points at compile time (the loop over points can then be unrolled and
  // the weights folded in); NQ = 0 reads nQ at run time.
  template <class Coefficients, int nS, int NQ = 0>
  ADRH_FN ADRH_FORCE_INLINE void elementMaskedOne(const Coefficients &C, const double *X, const double *ue, const double *uo,
                                int nQ, const double *qp, const double *qw, double t, double dt,
                                double *Rout, double *Jout)
  {
    using Co = Coefficients;
    constexpr int nc = Co::nc;
    constexpr int nD = nS + 1;
    static_assert(Co::nSpace == nS, "coefficient space dimension");
    double gw[nD][nS], adet;
    affineP1<nS>(X, gw, adet);
    double Rloc[nc][nD] = {};
    double Jloc[nc][nD][nc][nD] = {};
    double gu[nc][nS], guo[nc][nS];
    for (int j = 0; j < nc; ++j)
      for (int I = 0; I < nS; ++I)
      {
        gu[j][I] = 0.0; guo[j][I] = 0.0;
        for (int b = 0; b < nD; ++b)
        {
          gu[j][I] += ue[j * nD + b] * gw[b][I];
          if (uo) guo[j][I] += uo[j * nD + b] * gw[b][I];
        }
      }
    Terms<nc, nS> T, To;
    const int nPoints = NQ > 0 ? NQ : nQ;
    for (int q = 0; q < nPoints; ++q)
    {
      double w[nD];
      w[0] = 1.0;
      for (int I = 0; I < nS; ++I) w[0] -= qp[q * nS + I];
      for (int b = 1; b < nD; ++b) w[b] = qp[q * nS + b - 1];
      double x[3] = {0.0, 0.0, 0.0};
      for (int b = 0; b < nD; ++b)
        for (int I = 0; I < 3; ++I) x[I] += w[b] * X[b * 3 + I];
      const double dV = qw[q] * adet;
      double uq[nc], uoq[nc];
      for (int j = 0; j < nc; ++j)
      {
        uq[j] = 0.0; uoq[j] = 0.0;
        for (int b = 0; b < nD; ++b)
        {
          uq[j] += ue[j * nD + b] * w[b];
          if (uo) uoq[j] += uo[j * nD + b] * w[b];
        }
      }
      zeroMasked<Co>(T);
      C.evaluate(t, x, uq, gu, T);
      double mold[nc] = {};
      if constexpr (Co::has_mass)
        if (dt > 0.0 && uo)
        {
          zeroMasked<Co>(To);
          C.evaluate(t - dt, x, uoq, guo, To);
          constexpr_for<0, nc>([&](auto i_) ADRH_LAMBDA_INLINE {
            constexpr int i = decltype(i_)::value;
            if constexpr (Co::m_nz(i)) mold[i] = To.m[i];
          });
        }
      const double idt = (dt > 0.0) ? 1.0 / dt : 0.0;
      double gphi[nc][nS];
      for (int k = 0; k < nc; ++k)
        for (int I = 0; I < nS; ++I) gphi[k][I] = 0.0;
      if constexpr (Co::has_diffusion)
        constexpr_for<0, nc>([&](auto k_) ADRH_LAMBDA_INLINE {
          constexpr int k = decltype(k_)::value;
          constexpr_for<0, nc>([&](auto j_) ADRH_LAMBDA_INLINE {
            constexpr int j = decltype(j_)::value;
            if constexpr (Co::dphi_nz(k, j))
              for (int I = 0; I < nS; ++I) gphi[k][I] += T.dphi[k][j] * gu[j][I];
          });
        });
      constexpr_for<0, nc>([&](auto i_) ADRH_LAMBDA_INLINE {
        constexpr int i = decltype(i_)::value;
        double flux[nS] = {};
        if constexpr (Co::has_advection && Co::f_nz(i))
          for (int I = 0; I < nS; ++I) flux[I] -= T.f[i][I];
        if constexpr (Co::has_diffusion)
          constexpr_for<0, nc>([&](auto k_) ADRH_LAMBDA_INLINE {
            constexpr int k = decltype(k_)::value;
            if constexpr (Co::a_nz(i, k))
              for (int I = 0; I < nS; ++I)
                for (int Jx = 0; Jx < nS; ++Jx) flux[I] += T.a[i][k][I][Jx] * gphi[k][Jx];
          });
        for (int a = 0; a < nD; ++a)
        {
          double s = 0.0;
          if constexpr (Co::has_mass && Co::m_nz(i)) s += (T.m[i] - mold[i]) * idt * w[a];
          if constexpr (Co::has_hamiltonian && Co::H_nz(i)) s += T.H[i] * w[a];
          if constexpr (Co::has_reaction && Co::r_nz(i)) s += T.r[i] * w[a];
          for (int I = 0; I < nS; ++I) s += flux[I] * gw[a][I];
          Rloc[i][a] += dV * s;
        }
        constexpr_for<0, nc>([&](auto j_) ADRH_LAMBDA_INLINE {
          constexpr int j = decltype(j_)::value;
          for (int a = 0; a < nD; ++a)
            for (int b = 0; b < nD; ++b)
            {
              double jab = 0.0;
              if constexpr (Co::has_mass && Co::dm_nz(i, j)) jab += T.dm[i][j] * idt * w[b] * w[a];
              if constexpr (Co::has_reaction && Co::dr_nz(i, j)) jab += T.dr[i][j] * w[b] * w[a];
              if constexpr (Co::has_hamiltonian && Co::dH_nz(i, j))
                for (int I = 0; I < nS; ++I) jab += (T.dH[i][j][I] * gw[b][I]) * w[a];
              if constexpr (Co::has_advection && Co::df_nz(i, j))
                for (int I = 0; I < nS; ++I) jab -= T.df[i][j][I] * w[b] * gw[a][I];
              if constexpr (Co::has_diffusion)
                constexpr_for<0, nc>([&](auto k_) ADRH_LAMBDA_INLINE {
                  constexpr int k = decltype(k_)::value;
                  if constexpr (Co::da_nz(i, k, j))
                    for (int I = 0; I < nS; ++I)
                      for (int Jx = 0; Jx < nS; ++Jx)
                        jab += T.da[i][k][j][I][Jx] * w[b] * gphi[k][Jx] * gw[a][I];
                  if constexpr (Co::a_nz(i, k) && Co::dphi_nz(k, j))
                    for (int I = 0; I < nS; ++I)
                      for (int Jx = 0; Jx < nS; ++Jx)
                        jab += T.a[i][k][I][Jx] * T.dphi[k][j] * gw[b][Jx] * gw[a][I];
                });
              Jloc[i][a][j][b] += dV * jab;
            }
        });
      });
    }
    for (int k = 0; k < nc * nD; ++k) Rout[k] = (&Rloc[0][0])[k];
    for (int k = 0; k < nc * nD * nc * nD; ++k) Jout[k] = (&Jloc[0][0][0][0])[k];
  }

  // Same arguments and results as elementKernel (adrh.h), with the loops bounded by the masks.
  template <class Coefficients, int nS>
  void elementKernelMasked(const Coefficients &C, int nE, int nQ,
                           const double *xn, const double *u, const double *uold,
                           const double *qp, const double *qw, double t, double dt,
                           double *R, double *J)
  {
    constexpr int nc = Coefficients::nc;
    constexpr int nD = nS + 1;
    for (int eN = 0; eN < nE; ++eN)
      elementMaskedOne<Coefficients, nS>(C, xn + eN * nD * 3, u + eN * nc * nD, uold ? uold + eN * nc * nD : nullptr,
                                         nQ, qp, qw, t, dt, R + eN * nc * nD, J + eN * nc * nD * nc * nD);
  }
} // namespace adrh
