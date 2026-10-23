// Exterior-face terms of the weak form for P1 simplices (tetrahedra), as a separate kernel.
//
// The Galerkin form integrated by parts leaves, on every exterior face,
//     + integral over the face of  F . n  w  dS,   F = f(u) - sum_k a^{ik} grad phi^k   (the total flux),
// which proteus adds to the element residual as (advectiveFlux + diffusiveFlux) w dS
// (femIntegrals.c updateExteriorElementBoundaryFlux). What stands in for F . n at a boundary quadrature
// point is decided per component by the BC functor, one of the kinds below. The structure follows
// proteus's exterior numerical flux classes (NumericalFlux.py, numericalFlux.c) so the two can be
// compared term by term:
//
//   Strong         nothing here; the driver removes the unknown or replaces its row
//                  (StrongDirichletFactory, or numericalFluxType=None).
//   Flux           the total flux is given. Diffusive part: g = -a grad u . n, prescribed (a
//                  diffusiveFluxBoundaryCondition). Advective part: prescribed where a value is given (an
//                  advectiveFluxBoundaryCondition; no Jacobian, as proteus zeroes dadvectiveFlux_left there);
//                  where none is, the interior flux f(u) . n on outflow (df . n >= 0) and ZERO on inflow,
//                  which is what StrongDirichlet's 'outFlow' does (calculateExteriorNumericalAdvectiveFlux_NoBC
//                  drops the inflow flux).
//   WeakDirichlet  the value g_D is imposed weakly, as Advection_DiagonalUpwind_Diffusion_{IIPG,NIPG,SIPG}_exterior
//                  do. Advective part: upwinded between f(u) . n and f(g_D) . n by the signs of df(u) . n and
//                  df(g_D) . n (calculateExteriorNumericalAdvectiveFlux: a shock speed decides the mixed case, a
//                  transonic rarefaction is averaged), with a Jacobian only where the interior flux is taken.
//                  Diffusive part: -a grad phi . n + (C / h^p) (u - g_D), h the face diameter (its longest
//                  edge); C, p = 2, 1 for IIPG, 10, 1 for NIPG, 100, 1 for SIPG. Adjoint part (sigma = 0 IIPG,
//                  -1 NIPG, +1 SIPG): - sigma (u - g_D) (a^T n) . grad w dS, in the residual and the Jacobian.
//
// nc > 1: the kind is per component i. The diffusive part of component i sums over the potentials k with
// a^{ik} nonzero; penalty and adjoint are on the jump of u_k against g_D[k], and zero where component k is
// not itself WeakDirichlet (proteus: bc_u is u where no Dirichlet value is set). The upwind decision uses
// df^{ii}, as proteus does; the Jacobian of the interior flux uses the whole row df^{ij}.
//
// Two places where the engine is not proteus, on purpose: (1) proteus's adjoint
// (femIntegrals.c updateExteriorElementBoundaryDiffusionAdjoint) contracts the tensor as
// sum_{I,J} n_I a_{JI} grad w_I, the column sums of a, which equals (a^T n) . grad w only for a diagonal a;
// the engine uses (a^T n) . grad w. (2) proteus's Jacobian of the adjoint ignores da/du; so does the engine's,
// but its diffusive-flux Jacobian carries the da term like the element kernel. The regressions have an
// isotropic, constant a, where none of this shows.
//
// Output is per face (not scattered): Rf (nF, nc, nD) and Jf (nF, nc, nD, nc, nD); the driver adds
// them to the element rows of the face's element.
#pragma once
#include <cmath>
#include <cstring>
#include "masked.h"

namespace adrh
{
  enum BCKind : int { Strong = 0, Flux = 1, WeakDirichlet = 2 };

  // What the BC functor fills in at one boundary quadrature point, per component.
  template <int nc>
  struct FaceBC
  {
    int kind[nc];
    double gD[nc];       // WeakDirichlet: the Dirichlet value
    double gdiff[nc];    // Flux: the prescribed diffusive flux -a grad u . n (0 = no diffusive flux)
    bool hasAdv[nc];     // Flux: whether the advective flux is prescribed too,
    double fadv[nc];     //       and its value f . n
    void reset()
    {
      for (int i = 0; i < nc; ++i)
      {
        kind[i] = Strong; gD[i] = 0.0; gdiff[i] = 0.0; hasAdv[i] = false; fadv[i] = 0.0;
      }
    }
  };

  // Penalty and adjoint parameters of the weak Dirichlet kinds (proteus's defaults per class).
  struct WeakDirichletParams
  {
    double penalty_constant, penalty_power, sigma;
    static WeakDirichletParams IIPG() { return {2.0, 1.0, 0.0}; }
    static WeakDirichletParams NIPG() { return {10.0, 1.0, -1.0}; }
    static WeakDirichletParams SIPG() { return {100.0, 1.0, 1.0}; }
  };

  // The boundary kernel. bc.evaluate(t, x, n, F, FaceBC<nc>&) is called per face F and quadrature point;
  // the face index lets a functor look a per-face kind up.
  template <class Coefficients, class BC>
  void exteriorFaceKernel(const Coefficients &C, const BC &bc, const WeakDirichletParams &wd,
                          int nF, int nQf,
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
      // face diameter = its longest edge (mesh.cpp computeGeometricInfo_tetrahedron), for the penalty
      double h = 0.0;
      for (int p = 0; p < 3; ++p)
        for (int r = p + 1; r < 3; ++r)
        {
          double d2 = 0.0;
          for (int I = 0; I < 3; ++I)
          {
            const double d = X[fn[p] * 3 + I] - X[fn[r] * 3 + I];
            d2 += d * d;
          }
          h = std::fmax(h, std::sqrt(d2));
        }
      const double penalty = wd.penalty_constant / std::pow(h, wd.penalty_power);
      // gradients of the basis and of u on the element (constant for P1)
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
      Terms<nc, nS> T, Tb;
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
        FaceBC<nc> B;
        B.reset();
        bc.evaluate(t, x, nrm, F, B);
        bool anyWeak = false;
        for (int i = 0; i < nc; ++i) anyWeak = anyWeak || (B.kind[i] == WeakDirichlet);
        zeroMasked<Co>(T);
        C.evaluate(t, x, uq, gu, T);
        // the boundary state: g_D where a component is WeakDirichlet, the interior value elsewhere
        double ub[nc];
        for (int j = 0; j < nc; ++j) ub[j] = (B.kind[j] == WeakDirichlet) ? B.gD[j] : uq[j];
        if (anyWeak)
        {
          zeroMasked<Co>(Tb);
          C.evaluate(t, x, ub, gu, Tb);
        }
        // grad phi^k = sum_j dphi[k][j] grad u_j
        double gphi[nc][nS];
        for (int kk = 0; kk < nc; ++kk)
          for (int I = 0; I < nS; ++I)
          {
            gphi[kk][I] = 0.0;
            for (int j = 0; j < nc; ++j) gphi[kk][I] += T.dphi[kk][j] * gu[j][I];
          }
        const double dS = qwf[q] * len;
        constexpr_for<0, nc>([&](auto i_) {
          constexpr int i = decltype(i_)::value;
          if (B.kind[i] == Strong) return;
          double Fn = 0.0;                 // the flux . n at this point
          double dFn[nc] = {};             // d Fn / d u_j, the part that multiplies w_b (no gradients)
          // advective part
          if constexpr (Co::has_advection && Co::f_nz(i))
          {
            double fl = 0.0, dl = 0.0;
            for (int I = 0; I < 3; ++I) fl += T.f[i][I] * nrm[I];
            if constexpr (Co::df_nz(i, i))
              for (int I = 0; I < 3; ++I) dl += T.df[i][i][I] * nrm[I];
            bool interior = false, prescribed = false;
            if (B.kind[i] == Flux)
            {
              if (B.hasAdv[i]) { prescribed = true; Fn += B.fadv[i]; }
              else interior = (dl >= 0.0);           // inflow without a value: dropped, as proteus does
            }
            else                                     // WeakDirichlet: upwind against f(g_D)
            {
              double fr = 0.0, dr = 0.0;
              for (int I = 0; I < 3; ++I) fr += Tb.f[i][I] * nrm[I];
              if constexpr (Co::df_nz(i, i))
                for (int I = 0; I < 3; ++I) dr += Tb.df[i][i][I] * nrm[I];
              if (dl >= 0.0 && dr >= 0.0) interior = true;
              else if (dl <= 0.0 && dr <= 0.0) interior = false;
              else if (dl >= 0.0 && dr <= 0.0)
              {
                const double uj = ub[i] - uq[i], fj = fr - fl;
                const double s = (std::fabs(uj) > std::fabs(fj) * 1.0e-16) ? fj / uj : 0.0;
                interior = (s >= 0.0);
              }
              else
              {                                      // transonic rarefaction: proteus averages the two
                Fn += 0.5 * (fl + fr);
                prescribed = true;                   // and takes half the interior speed as the Jacobian
                constexpr_for<0, nc>([&](auto j_) {
                  constexpr int j = decltype(j_)::value;
                  if constexpr (Co::df_nz(i, j))
                    for (int I = 0; I < 3; ++I) dFn[j] += 0.5 * T.df[i][j][I] * nrm[I];
                });
              }
              if (!prescribed && !interior) Fn += fr;
            }
            if (interior)
            {
              Fn += fl;
              constexpr_for<0, nc>([&](auto j_) {
                constexpr int j = decltype(j_)::value;
                if constexpr (Co::df_nz(i, j))
                  for (int I = 0; I < 3; ++I) dFn[j] += T.df[i][j][I] * nrm[I];
              });
            }
          }
          // diffusive part
          if constexpr (Co::has_diffusion)
          {
            if (B.kind[i] == Flux)
              Fn += B.gdiff[i];
            else
            {
              constexpr_for<0, nc>([&](auto k_) {
                constexpr int kk = decltype(k_)::value;
                if constexpr (Co::a_nz(i, kk))
                {
                  double an[nS];                     // (a^{ik})^T n
                  for (int J = 0; J < nS; ++J)
                  {
                    an[J] = 0.0;
                    for (int I = 0; I < nS; ++I) an[J] += nrm[I] * T.a[i][kk][I][J];
                  }
                  const double jump = uq[kk] - ub[kk];   // zero unless component kk is WeakDirichlet
                  double qn = 0.0;
                  for (int J = 0; J < nS; ++J) qn -= an[J] * gphi[kk][J];
                  Fn += qn + penalty * jump;
                  // Jacobian of -n . a grad phi^k + penalty (u_k - g) and of the adjoint, which carry grad w_b
                  constexpr_for<0, nc>([&](auto j_) {
                    constexpr int j = decltype(j_)::value;
                    for (int b = 0; b < nD; ++b)
                    {
                      double d = 0.0;
                      if constexpr (Co::dphi_nz(kk, j))
                        for (int J = 0; J < nS; ++J) d -= an[J] * T.dphi[kk][j] * gw[b][J];
                      if constexpr (Co::da_nz(i, kk, j))
                        for (int I = 0; I < nS; ++I)
                          for (int J = 0; J < nS; ++J) d -= nrm[I] * T.da[i][kk][j][I][J] * gphi[kk][J] * w[b];
                      if (j == kk) d += penalty * w[b];
                      for (int a = 0; a < nD; ++a) Jloc[i][a][j][b] += dS * d * w[a];
                    }
                  });
                  if (wd.sigma != 0.0)
                  {
                    for (int a = 0; a < nD; ++a)
                    {
                      double angw = 0.0;
                      for (int J = 0; J < nS; ++J) angw += an[J] * gw[a][J];
                      Rloc[i][a] -= dS * wd.sigma * jump * angw;
                      if (B.kind[kk] == WeakDirichlet)
                        for (int b = 0; b < nD; ++b) Jloc[i][a][kk][b] -= dS * wd.sigma * w[b] * angw;
                    }
                  }
                }
              });
            }
          }
          for (int a = 0; a < nD; ++a)
          {
            Rloc[i][a] += dS * Fn * w[a];
            constexpr_for<0, nc>([&](auto j_) {
              constexpr int j = decltype(j_)::value;
              if (dFn[j] != 0.0)
                for (int b = 0; b < nD; ++b) Jloc[i][a][j][b] += dS * dFn[j] * w[b] * w[a];
            });
          }
        });
      }
      std::memcpy(Rf + F * nc * nD, Rloc, sizeof(Rloc));
      std::memcpy(Jf + F * nc * nD * nc * nD, Jloc, sizeof(Jloc));
    }
  }

  // Boundary data of the sine solution u = sin(omega.x + omega0) for the regression. The kind of each face
  // comes from an array: 1 = Flux (diffusive flux given; advection leaves by the interior value, inflow dropped),
  // 2 = Flux with the advective flux (b.n) u given too, 3 = WeakDirichlet with u given; anything else = Strong.
  struct SineBC
  {
    double b[3], kappa, omega[3], omega0;
    const int *faceKind;
    SineBC(const double *b_, double kappa_, const double *omega_, double omega0_, const int *faceKind_)
        : kappa(kappa_), omega0(omega0_), faceKind(faceKind_)
    {
      for (int I = 0; I < 3; ++I) { b[I] = b_[I]; omega[I] = omega_[I]; }
    }
    void evaluate(double, const double x[3], const double n[3], int F, FaceBC<1> &B) const
    {
      double arg = omega0, wn = 0.0, bn = 0.0;
      for (int I = 0; I < 3; ++I) { arg += omega[I] * x[I]; wn += omega[I] * n[I]; bn += b[I] * n[I]; }
      const double ue = std::sin(arg), gradn = wn * std::cos(arg);
      switch (faceKind[F])
      {
      case 1: B.kind[0] = Flux; B.gdiff[0] = -kappa * gradn; break;
      case 2: B.kind[0] = Flux; B.gdiff[0] = -kappa * gradn; B.hasAdv[0] = true; B.fadv[0] = bn * ue; break;
      case 3: B.kind[0] = WeakDirichlet; B.gD[0] = ue; break;
      default: B.kind[0] = Strong;
      }
    }
  };
} // namespace adrh
