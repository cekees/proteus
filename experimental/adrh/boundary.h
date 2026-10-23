// Exterior-face terms of the weak form for P1 simplices (tetrahedra), as a separate kernel.
//
// The Galerkin form integrated by parts leaves, on every exterior face,
//     + integral over the face of  F . n  w  dS,   F = f(u) - a grad(phi)   (the total flux),
// which proteus adds to the element residual as (advectiveFlux + diffusiveFlux) w dS. A face carries
// one of the following (the BC functor says which, per face and quadrature point):
//   strong Dirichlet: nothing here, the unknown is removed or its row replaced by the driver;
//   flux face: the advective flux is the interior one, f(u) . n (outflow; the engine does not yet
//     look at inflow), and the diffusive flux is prescribed, g = -a grad u . n, a function of x, t and n.
// Weak Dirichlet (SIPG/IIPG penalty and adjoint terms) and upwinding against a prescribed inflow value are
// the next steps; they are further BC functor kinds, not changes of this structure.
//
// Output is per face (not scattered): Rf (nF, nc, nD) and Jf (nF, nc, nD, nc, nD); the driver adds
// them to the element rows of the face's element.
#pragma once
#include <cmath>
#include <cstring>
#include "masked.h"

namespace adrh
{
  // A flux boundary condition: g_i = prescribed diffusive flux of component i, -a grad u . n
  template <int nc>
  struct FluxValues
  {
    double g[nc];
    void zero() { std::memset(g, 0, sizeof(g)); }
  };

  template <class Coefficients, class BC>
  void exteriorFluxFaceKernel(const Coefficients &C, const BC &bc, int nF, int nQf,
                              const int *faceLocal,            // (nF) local face index k = the opposite node
                              const double *xn,                // (nF, 4, 3) coordinates of the face's element
                              const double *u,                 // (nF, nc, 4) the element's dofs
                              const double *qpf, const double *qwf,   // face quadrature on the reference triangle
                              double t, double *Rf, double *Jf)
  {
    using Co = Coefficients;
    constexpr int nc = Co::nc, nS = 3, nD = 4;
    static_assert(Co::nSpace == 3, "tetrahedra");
    for (int F = 0; F < nF; ++F)
    {
      const double *X = xn + F * nD * 3;
      const int k = faceLocal[F];
      int fn[3], m = 0;
      for (int b = 0; b < nD; ++b)
        if (b != k) fn[m++] = b;
      double e1[3], e2[3], cr[3];
      for (int I = 0; I < 3; ++I)
      {
        e1[I] = X[fn[1] * 3 + I] - X[fn[0] * 3 + I];
        e2[I] = X[fn[2] * 3 + I] - X[fn[0] * 3 + I];
      }
      cr[0] = e1[1] * e2[2] - e1[2] * e2[1];
      cr[1] = e1[2] * e2[0] - e1[0] * e2[2];
      cr[2] = e1[0] * e2[1] - e1[1] * e2[0];
      const double len = std::sqrt(cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]);
      double nrm[3] = {cr[0] / len, cr[1] / len, cr[2] / len};
      double out = 0.0;                                   // orient away from the opposite node
      for (int I = 0; I < 3; ++I) out += nrm[I] * (X[fn[0] * 3 + I] - X[k * 3 + I]);
      if (out < 0.0) for (int I = 0; I < 3; ++I) nrm[I] = -nrm[I];
      // gradients of u on the element are constant; the coefficient functor wants grad u (unused by the
      // flux terms here but part of its interface), so compute them from the element's affine map
      double gw[nD][nS], adet;
      affineP1<nS>(X, gw, adet);
      const double *ue = u + F * nc * nD;
      double gu[nc][nS];
      for (int j = 0; j < nc; ++j)
        for (int I = 0; I < nS; ++I)
        {
          gu[j][I] = 0.0;
          for (int b = 0; b < nD; ++b) gu[j][I] += ue[j * nD + b] * gw[b][I];
        }
      double Rloc[nc][nD] = {};
      double Jloc[nc][nD][nc][nD] = {};
      Terms<nc, nS> T;
      for (int q = 0; q < nQf; ++q)
      {
        double w[nD] = {0.0, 0.0, 0.0, 0.0};
        w[fn[0]] = 1.0 - qpf[q * 2] - qpf[q * 2 + 1];
        w[fn[1]] = qpf[q * 2];
        w[fn[2]] = qpf[q * 2 + 1];
        double x[3] = {0.0, 0.0, 0.0};
        for (int b = 0; b < nD; ++b)
          for (int I = 0; I < 3; ++I) x[I] += w[b] * X[b * 3 + I];
        double uq[nc];
        for (int j = 0; j < nc; ++j)
        {
          uq[j] = 0.0;
          for (int b = 0; b < nD; ++b) uq[j] += ue[j * nD + b] * w[b];
        }
        zeroMasked<Co>(T);
        C.evaluate(t, x, uq, gu, T);
        FluxValues<nc> g;
        g.zero();
        bc.diffusiveFlux(t, x, nrm, g.g);
        const double dS = qwf[q] * len;
        constexpr_for<0, nc>([&](auto i_) {
          constexpr int i = decltype(i_)::value;
          double Fn = g.g[i];
          if constexpr (Co::has_advection && Co::f_nz(i))
            for (int I = 0; I < 3; ++I) Fn += T.f[i][I] * nrm[I];
          for (int a = 0; a < nD; ++a) Rloc[i][a] += dS * Fn * w[a];
          constexpr_for<0, nc>([&](auto j_) {
            constexpr int j = decltype(j_)::value;
            if constexpr (Co::has_advection && Co::df_nz(i, j))
            {
              double dfn = 0.0;
              for (int I = 0; I < 3; ++I) dfn += T.df[i][j][I] * nrm[I];
              for (int a = 0; a < nD; ++a)
                for (int b = 0; b < nD; ++b) Jloc[i][a][j][b] += dS * dfn * w[b] * w[a];
            }
          });
        });
      }
      std::memcpy(Rf + F * nc * nD, Rloc, sizeof(Rloc));
      std::memcpy(Jf + F * nc * nD * nc * nD, Jloc, sizeof(Jloc));
    }
  }

  // g = -kappa grad(u_exact) . n for u_exact = sin(omega.x + omega0): the flux of the sine solution's diffusion
  struct SineDiffusiveFlux
  {
    double kappa, omega[3], omega0;
    SineDiffusiveFlux(double kappa_, const double *omega_, double omega0_) : kappa(kappa_), omega0(omega0_)
    {
      for (int I = 0; I < 3; ++I) omega[I] = omega_[I];
    }
    void diffusiveFlux(double, const double x[3], const double n[3], double *g) const
    {
      double arg = omega0, wn = 0.0;
      for (int I = 0; I < 3; ++I) { arg += omega[I] * x[I]; wn += omega[I] * n[I]; }
      g[0] = -kappa * wn * std::cos(arg);
    }
  };
} // namespace adrh
