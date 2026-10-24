// The element kernel for any Lagrange Space (spaces.h): the masked weak form of masked.h with the basis, its
// physical gradients and the Jacobian taken at every quadrature point from the space's isoparametric geometry.
// For P1 simplices it computes what elementKernelMasked computes.
//   xn (nE, nDOF, 3)  the element's DOF node coordinates   u, uold (nE, nc, nDOF)
//   qp (nQ, nS), qw (nQ)  reference quadrature            R (nE, nc, nDOF), J (nE, nc, nDOF, nc, nDOF)
#pragma once
#include <cstring>
#include "masked.h"
#include "spaces.h"

namespace adrh
{
  template <class Coefficients, class Space>
  void elementKernelSpace(const Coefficients &C, int nE, int nQ,
                          const double *xn, const double *u, const double *uold,
                          const double *qp, const double *qw, double t, double dt,
                          double *R, double *J)
  {
    using Co = Coefficients;
    constexpr int nc = Co::nc, nS = Space::nS, nD = Space::nDOF;
    static_assert(Co::nSpace == nS, "coefficient space dimension");
    PointGeometry<Space> G;
    Terms<nc, nS> T, To;
    for (int eN = 0; eN < nE; ++eN)
    {
      const double *X = xn + eN * nD * 3;
      double Rloc[nc][nD] = {};
      double Jloc[nc][nD][nc][nD] = {};
      const double *ue = u + eN * nc * nD;
      const double *uo = uold ? uold + eN * nc * nD : nullptr;
      for (int q = 0; q < nQ; ++q)
      {
        G.compute(X, qp + q * nS);
        const double *w = G.w;
        const auto &gw = G.gw;
        const double dV = qw[q] * G.adet;
        double uq[nc], uoq[nc], gu[nc][nS], guo[nc][nS];
        for (int j = 0; j < nc; ++j)
        {
          uq[j] = 0.0; uoq[j] = 0.0;
          for (int I = 0; I < nS; ++I) { gu[j][I] = 0.0; guo[j][I] = 0.0; }
          for (int b = 0; b < nD; ++b)
          {
            uq[j] += ue[j * nD + b] * w[b];
            for (int I = 0; I < nS; ++I) gu[j][I] += ue[j * nD + b] * gw[b][I];
            if (uo)
            {
              uoq[j] += uo[j * nD + b] * w[b];
              for (int I = 0; I < nS; ++I) guo[j][I] += uo[j * nD + b] * gw[b][I];
            }
          }
        }
        zeroMasked<Co>(T);
        C.evaluate(t, G.x, uq, gu, T);
        double mold[nc] = {};
        if constexpr (Co::has_mass)
          if (dt > 0.0 && uo)
          {
            zeroMasked<Co>(To);
            C.evaluate(t - dt, G.x, uoq, guo, To);
            constexpr_for<0, nc>([&](auto i_) {
              constexpr int i = decltype(i_)::value;
              if constexpr (Co::m_nz(i)) mold[i] = To.m[i];
            });
          }
        const double idt = (dt > 0.0) ? 1.0 / dt : 0.0;
        double gphi[nc][nS];
        for (int k = 0; k < nc; ++k)
          for (int I = 0; I < nS; ++I) gphi[k][I] = 0.0;
        if constexpr (Co::has_diffusion)
          constexpr_for<0, nc>([&](auto k_) {
            constexpr int k = decltype(k_)::value;
            constexpr_for<0, nc>([&](auto j_) {
              constexpr int j = decltype(j_)::value;
              if constexpr (Co::dphi_nz(k, j))
                for (int I = 0; I < nS; ++I) gphi[k][I] += T.dphi[k][j] * gu[j][I];
            });
          });
        constexpr_for<0, nc>([&](auto i_) {
          constexpr int i = decltype(i_)::value;
          double flux[nS] = {};
          if constexpr (Co::has_advection && Co::f_nz(i))
            for (int I = 0; I < nS; ++I) flux[I] -= T.f[i][I];
          if constexpr (Co::has_diffusion)
            constexpr_for<0, nc>([&](auto k_) {
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
          constexpr_for<0, nc>([&](auto j_) {
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
                  constexpr_for<0, nc>([&](auto k_) {
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
      std::memcpy(R + eN * nc * nD, Rloc, sizeof(Rloc));
      std::memcpy(J + eN * nc * nD * nc * nD, Jloc, sizeof(Jloc));
    }
  }
} // namespace adrh
